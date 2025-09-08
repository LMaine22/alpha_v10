# alpha_discovery/meta_labeling/core.py
"""
Core Meta-Labeling System Implementation

This module contains the main MetaLabelingSystem class that orchestrates
the entire meta-labeling pipeline from feature extraction to evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import settings
from .types import MetaLabelingResults
from .features import MetaFeatureExtractor
from .models import MetaModelTrainer
from .evaluation import MetaEvaluator
from .artifacts import MetaArtifactGenerator


class MetaLabelingSystem:
    """Main meta-labeling system that orchestrates the entire pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the meta-labeling system."""
        self.config = settings.meta_labeling
        if config:
            self.config.update(config)
        
        # Initialize components
        self.feature_extractor = MetaFeatureExtractor(self.config)
        self.model_trainer = MetaModelTrainer(self.config)
        self.evaluator = MetaEvaluator(self.config)
        self.artifact_generator = MetaArtifactGenerator(self.config)
        
        # Results storage
        self.results: List[MetaLabelingResults] = []
        
        logging.info(f"Meta-Labeling System initialized with config: {self.config}")
    
    def run_meta_labeling(self, gauntlet_survivors: List[Dict], 
                         oos_ledgers: Dict[str, pd.DataFrame],
                         master_df: pd.DataFrame,
                         signals_df: pd.DataFrame,
                         signals_metadata: List[Dict]) -> List[MetaLabelingResults]:
        """
        Run meta-labeling on gauntlet survivors.
        
        Args:
            gauntlet_survivors: List of setups that passed the gauntlet
            oos_ledgers: Dictionary of OOS trade ledgers by setup_id
            master_df: Master data frame with all features
            signals_df: Signals data frame
            signals_metadata: Signals metadata
            
        Returns:
            List of MetaLabelingResults for each setup
        """
        if not self.config.enabled:
            logging.info("Meta-labeling is disabled, skipping...")
            return []
        
        logging.info(f"Starting meta-labeling on {len(gauntlet_survivors)} gauntlet survivors")
        
        results = []
        
        # Group survivors by setup for processing
        setup_groups = self._group_survivors_by_setup(gauntlet_survivors)
        
        for setup_id, setup_info in tqdm(setup_groups.items(), desc="Meta-labeling setups"):
            try:
                result = self._process_setup(
                    setup_id, setup_info, oos_ledgers, master_df, signals_df, signals_metadata
                )
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing setup {setup_id}: {e}")
                error_result = MetaLabelingResults(
                    setup_id=setup_id,
                    ticker=setup_info.get('ticker', 'Unknown'),
                    direction=setup_info.get('direction', 'long'),
                    base_expected_value=0.0,
                    meta_expected_value=0.0,
                    base_sharpe=0.0,
                    meta_sharpe=0.0,
                    base_max_drawdown=0.0,
                    meta_max_drawdown=0.0,
                    base_trade_count=0,
                    meta_trade_count=0,
                    trade_retention_rate=0.0,
                    model_accuracy=0.0,
                    model_precision=0.0,
                    model_recall=0.0,
                    model_f1=0.0,
                    model_calibration=0.0,
                    feature_importance={},
                    meta_decisions=[],
                    status="error",
                    error_message=str(e)
                )
                results.append(error_result)
        
        self.results = results
        
        # Generate summary artifacts
        if self.config.generate_artifacts:
            self.artifact_generator.generate_summary_artifacts(results)
        
        logging.info(f"Meta-labeling complete. Processed {len(results)} setups.")
        return results
    
    def _group_survivors_by_setup(self, gauntlet_survivors: List[Dict]) -> Dict[str, Dict]:
        """Group gauntlet survivors by setup_id."""
        setup_groups = defaultdict(list)
        
        for survivor in gauntlet_survivors:
            setup_id = survivor.get('setup_id', 'unknown')
            setup_groups[setup_id].append(survivor)
        
        # Convert to setup info format
        setup_info = {}
        for setup_id, survivors in setup_groups.items():
            # Get common info from first survivor
            first_survivor = survivors[0]
            setup_info[setup_id] = {
                'ticker': first_survivor.get('ticker', 'Unknown'),
                'direction': first_survivor.get('direction', 'long'),
                'survivors': survivors
            }
        
        return setup_info
    
    def _process_setup(self, setup_id: str, setup_info: Dict, 
                      oos_ledgers: Dict[str, pd.DataFrame],
                      master_df: pd.DataFrame, signals_df: pd.DataFrame,
                      signals_metadata: List[Dict]) -> MetaLabelingResults:
        """Process a single setup for meta-labeling."""
        
        # Get OOS ledger for this setup
        if setup_id not in oos_ledgers:
            return self._create_insufficient_data_result(setup_id, setup_info, "No OOS ledger found")
        
        oos_ledger = oos_ledgers[setup_id]
        
        if oos_ledger.empty:
            return self._create_insufficient_data_result(setup_id, setup_info, "Empty OOS ledger")
        
        # Check minimum trade requirements
        if len(oos_ledger) < self.config.min_trades_for_meta:
            return self._create_insufficient_data_result(
                setup_id, setup_info, 
                f"Insufficient trades: {len(oos_ledger)} < {self.config.min_trades_for_meta}"
            )
        
        # Extract meta-features
        meta_features = self.feature_extractor.extract_features(
            setup_id, setup_info, oos_ledger, master_df, signals_df, signals_metadata
        )
        
        if meta_features.empty:
            return self._create_insufficient_data_result(setup_id, setup_info, "No meta-features extracted")
        
        # Create meta-labels (WIN/LOSS based on options P&L)
        meta_labels = self._create_meta_labels(oos_ledger)
        
        # Train meta-model
        model_result = self.model_trainer.train_model(
            setup_id, meta_features, meta_labels, oos_ledger
        )
        
        if model_result is None:
            return self._create_insufficient_data_result(setup_id, setup_info, "Model training failed")
        
        # Evaluate meta-model
        evaluation_result = self.evaluator.evaluate_model(
            setup_id, setup_info, model_result, meta_features, meta_labels, oos_ledger
        )
        
        return evaluation_result
    
    def _create_meta_labels(self, oos_ledger: pd.DataFrame) -> pd.Series:
        """Create meta-labels based on options trade P&L."""
        # WIN = 1 if P&L > 0, LOSS = 0 if P&L <= 0
        pnl_column = None
        for col in ['pnl_dollars', 'realized_pnl', 'pnl_pct']:
            if col in oos_ledger.columns:
                pnl_column = col
                break
        
        if pnl_column is None:
            raise ValueError("No P&L column found in OOS ledger")
        
        # Create binary labels: 1 for positive P&L, 0 for negative/zero
        labels = (oos_ledger[pnl_column] > 0).astype(int)
        
        return labels
    
    def _create_insufficient_data_result(self, setup_id: str, setup_info: Dict, 
                                       reason: str) -> MetaLabelingResults:
        """Create a result for setups with insufficient data."""
        return MetaLabelingResults(
            setup_id=setup_id,
            ticker=setup_info.get('ticker', 'Unknown'),
            direction=setup_info.get('direction', 'long'),
            base_expected_value=0.0,
            meta_expected_value=0.0,
            base_sharpe=0.0,
            meta_sharpe=0.0,
            base_max_drawdown=0.0,
            meta_max_drawdown=0.0,
            base_trade_count=0,
            meta_trade_count=0,
            trade_retention_rate=0.0,
            model_accuracy=0.0,
            model_precision=0.0,
            model_recall=0.0,
            model_f1=0.0,
            model_calibration=0.0,
            feature_importance={},
            meta_decisions=[],
            status="insufficient_data",
            error_message=reason
        )
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from meta-labeling results."""
        if not self.results:
            return {"error": "No results available"}
        
        # Count by status
        status_counts = defaultdict(int)
        for result in self.results:
            status_counts[result.status] += 1
        
        # Calculate average improvements
        trained_results = [r for r in self.results if r.status == "trained"]
        
        if not trained_results:
            return {
                "status_counts": dict(status_counts),
                "trained_count": 0,
                "error": "No successfully trained models"
            }
        
        avg_ev_improvement = np.mean([r.meta_expected_value - r.base_expected_value for r in trained_results])
        avg_sharpe_improvement = np.mean([r.meta_sharpe - r.base_sharpe for r in trained_results])
        avg_retention_rate = np.mean([r.trade_retention_rate for r in trained_results])
        avg_accuracy = np.mean([r.model_accuracy for r in trained_results])
        
        return {
            "status_counts": dict(status_counts),
            "trained_count": len(trained_results),
            "avg_ev_improvement": avg_ev_improvement,
            "avg_sharpe_improvement": avg_sharpe_improvement,
            "avg_retention_rate": avg_retention_rate,
            "avg_accuracy": avg_accuracy,
            "total_setups": len(self.results)
        }
