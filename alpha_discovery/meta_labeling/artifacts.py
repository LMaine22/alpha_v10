# alpha_discovery/meta_labeling/artifacts.py
"""
Meta-Labeling Artifacts Generation

This module generates detailed artifacts and reports for meta-labeling results,
including per-setup summaries, global statistics, and decision logs.
"""

from __future__ import annotations

import logging
import os
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from ..config import settings
from .types import MetaLabelingResults


class MetaArtifactGenerator:
    """Generates artifacts and reports for meta-labeling results."""
    
    def __init__(self, config: Any):
        """Initialize the artifact generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_summary_artifacts(self, results: List[MetaLabelingResults], 
                                 output_dir: str = "meta_labeling_artifacts") -> None:
        """Generate all summary artifacts."""
        if not self.config.generate_artifacts:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate different types of artifacts
        self._generate_per_setup_summary(results, output_dir)
        self._generate_global_summary(results, output_dir)
        self._generate_decision_logs(results, output_dir)
        self._generate_feature_importance_summary(results, output_dir)
        self._generate_performance_comparison(results, output_dir)
        
        self.logger.info(f"Meta-labeling artifacts generated in {output_dir}")
    
    def _generate_per_setup_summary(self, results: List[MetaLabelingResults], 
                                   output_dir: str) -> None:
        """Generate per-setup summary."""
        summary_data = []
        
        for result in results:
            summary_data.append({
                'setup_id': result.setup_id,
                'ticker': result.ticker,
                'direction': result.direction,
                'status': result.status,
                'error_message': result.error_message,
                
                # Base performance
                'base_expected_value': result.base_expected_value,
                'base_sharpe': result.base_sharpe,
                'base_max_drawdown': result.base_max_drawdown,
                'base_trade_count': result.base_trade_count,
                
                # Meta performance
                'meta_expected_value': result.meta_expected_value,
                'meta_sharpe': result.meta_sharpe,
                'meta_max_drawdown': result.meta_max_drawdown,
                'meta_trade_count': result.meta_trade_count,
                
                # Improvements
                'ev_improvement': result.meta_expected_value - result.base_expected_value,
                'sharpe_improvement': result.meta_sharpe - result.base_sharpe,
                'dd_improvement': result.base_max_drawdown - result.meta_max_drawdown,
                
                # Model performance
                'model_accuracy': result.model_accuracy,
                'model_precision': result.model_precision,
                'model_recall': result.model_recall,
                'model_f1': result.model_f1,
                'model_calibration': result.model_calibration,
                
                # Trade statistics
                'trade_retention_rate': result.trade_retention_rate,
                'trade_reduction': result.base_trade_count - result.meta_trade_count,
            })
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{output_dir}/per_setup_summary.csv", index=False)
        
        # Save as JSON
        with open(f"{output_dir}/per_setup_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        self.logger.info(f"Per-setup summary saved to {output_dir}/per_setup_summary.csv")
    
    def _generate_global_summary(self, results: List[MetaLabelingResults], 
                                output_dir: str) -> None:
        """Generate global summary statistics."""
        # Filter successful results
        successful_results = [r for r in results if r.status == "trained"]
        failed_results = [r for r in results if r.status != "trained"]
        
        # Status counts
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Calculate aggregate statistics
        if successful_results:
            avg_ev_improvement = np.mean([r.meta_expected_value - r.base_expected_value for r in successful_results])
            avg_sharpe_improvement = np.mean([r.meta_sharpe - r.base_sharpe for r in successful_results])
            avg_retention_rate = np.mean([r.trade_retention_rate for r in successful_results])
            avg_accuracy = np.mean([r.model_accuracy for r in successful_results])
            
            # Calculate total trade impact
            total_base_trades = sum([r.base_trade_count for r in successful_results])
            total_meta_trades = sum([r.meta_trade_count for r in successful_results])
            total_trade_reduction = total_base_trades - total_meta_trades
            
            # Calculate weighted improvements
            total_base_ev = sum([r.base_expected_value * r.base_trade_count for r in successful_results])
            total_meta_ev = sum([r.meta_expected_value * r.meta_trade_count for r in successful_results])
            weighted_ev_improvement = (total_meta_ev - total_base_ev) / max(total_base_trades, 1)
        else:
            avg_ev_improvement = 0.0
            avg_sharpe_improvement = 0.0
            avg_retention_rate = 0.0
            avg_accuracy = 0.0
            total_base_trades = 0
            total_meta_trades = 0
            total_trade_reduction = 0
            weighted_ev_improvement = 0.0
        
        global_summary = {
            'meta_labeling_config': {
                'enabled': self.config.enabled,
                'min_trades_for_meta': self.config.min_trades_for_meta,
                'min_trade_retention': self.config.min_trade_retention,
                'max_trade_retention': self.config.max_trade_retention,
                'threshold_optimization_metric': self.config.threshold_optimization_metric,
                'model_type': self.config.model_type,
            },
            'status_counts': status_counts,
            'successful_setups': len(successful_results),
            'failed_setups': len(failed_results),
            'total_setups': len(results),
            'success_rate': len(successful_results) / len(results) if results else 0.0,
            
            'performance_improvements': {
                'avg_ev_improvement': avg_ev_improvement,
                'avg_sharpe_improvement': avg_sharpe_improvement,
                'weighted_ev_improvement': weighted_ev_improvement,
                'avg_retention_rate': avg_retention_rate,
                'avg_model_accuracy': avg_accuracy,
            },
            
            'trade_statistics': {
                'total_base_trades': total_base_trades,
                'total_meta_trades': total_meta_trades,
                'total_trade_reduction': total_trade_reduction,
                'trade_reduction_rate': total_trade_reduction / max(total_base_trades, 1),
            },
            
            'failed_setup_reasons': {
                result.status: [r.setup_id for r in failed_results if r.status == result.status]
                for result in failed_results
            }
        }
        
        # Save as JSON
        with open(f"{output_dir}/global_summary.json", 'w') as f:
            json.dump(global_summary, f, indent=2, default=str)
        
        self.logger.info(f"Global summary saved to {output_dir}/global_summary.json")
    
    def _generate_decision_logs(self, results: List[MetaLabelingResults], 
                               output_dir: str) -> None:
        """Generate detailed decision logs."""
        all_decisions = []
        
        for result in results:
            if result.status == "trained" and result.meta_decisions:
                for decision in result.meta_decisions:
                    decision['setup_id'] = result.setup_id
                    decision['ticker'] = result.ticker
                    decision['direction'] = result.direction
                    all_decisions.append(decision)
        
        if all_decisions:
            # Save as CSV
            df = pd.DataFrame(all_decisions)
            df.to_csv(f"{output_dir}/meta_decisions.csv", index=False)
            
            # Save as JSON
            with open(f"{output_dir}/meta_decisions.json", 'w') as f:
                json.dump(all_decisions, f, indent=2, default=str)
            
            self.logger.info(f"Meta decisions saved to {output_dir}/meta_decisions.csv")
        else:
            self.logger.warning("No meta decisions to save")
    
    def _generate_feature_importance_summary(self, results: List[MetaLabelingResults], 
                                           output_dir: str) -> None:
        """Generate feature importance summary."""
        feature_importance_data = []
        
        for result in results:
            if result.status == "trained" and result.feature_importance:
                for feature, importance in result.feature_importance.items():
                    feature_importance_data.append({
                        'setup_id': result.setup_id,
                        'ticker': result.ticker,
                        'feature': feature,
                        'importance': importance
                    })
        
        if feature_importance_data:
            # Save as CSV
            df = pd.DataFrame(feature_importance_data)
            df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
            
            # Calculate aggregate feature importance
            agg_importance = df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
            agg_importance = agg_importance.sort_values('mean', ascending=False)
            agg_importance.to_csv(f"{output_dir}/aggregate_feature_importance.csv", index=False)
            
            self.logger.info(f"Feature importance saved to {output_dir}/feature_importance.csv")
        else:
            self.logger.warning("No feature importance data to save")
    
    def _generate_performance_comparison(self, results: List[MetaLabelingResults], 
                                        output_dir: str) -> None:
        """Generate performance comparison analysis."""
        successful_results = [r for r in results if r.status == "trained"]
        
        if not successful_results:
            self.logger.warning("No successful results for performance comparison")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for result in successful_results:
            comparison_data.append({
                'setup_id': result.setup_id,
                'ticker': result.ticker,
                'direction': result.direction,
                'base_ev': result.base_expected_value,
                'meta_ev': result.meta_expected_value,
                'ev_improvement': result.meta_expected_value - result.base_expected_value,
                'ev_improvement_pct': ((result.meta_expected_value - result.base_expected_value) / 
                                     abs(result.base_expected_value) * 100) if result.base_expected_value != 0 else 0,
                'base_sharpe': result.base_sharpe,
                'meta_sharpe': result.meta_sharpe,
                'sharpe_improvement': result.meta_sharpe - result.base_sharpe,
                'base_dd': result.base_max_drawdown,
                'meta_dd': result.meta_max_drawdown,
                'dd_improvement': result.base_max_drawdown - result.meta_max_drawdown,
                'retention_rate': result.trade_retention_rate,
                'model_accuracy': result.model_accuracy,
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save detailed comparison
        df.to_csv(f"{output_dir}/performance_comparison.csv", index=False)
        
        # Generate summary statistics
        summary_stats = {
            'ev_improvements': {
                'mean': df['ev_improvement'].mean(),
                'median': df['ev_improvement'].median(),
                'std': df['ev_improvement'].std(),
                'min': df['ev_improvement'].min(),
                'max': df['ev_improvement'].max(),
                'positive_count': (df['ev_improvement'] > 0).sum(),
                'negative_count': (df['ev_improvement'] < 0).sum(),
            },
            'sharpe_improvements': {
                'mean': df['sharpe_improvement'].mean(),
                'median': df['sharpe_improvement'].median(),
                'std': df['sharpe_improvement'].std(),
                'min': df['sharpe_improvement'].min(),
                'max': df['sharpe_improvement'].max(),
                'positive_count': (df['sharpe_improvement'] > 0).sum(),
                'negative_count': (df['sharpe_improvement'] < 0).sum(),
            },
            'retention_rates': {
                'mean': df['retention_rate'].mean(),
                'median': df['retention_rate'].median(),
                'std': df['retention_rate'].std(),
                'min': df['retention_rate'].min(),
                'max': df['retention_rate'].max(),
            },
            'model_accuracies': {
                'mean': df['model_accuracy'].mean(),
                'median': df['model_accuracy'].median(),
                'std': df['model_accuracy'].std(),
                'min': df['model_accuracy'].min(),
                'max': df['model_accuracy'].max(),
            }
        }
        
        # Save summary statistics
        with open(f"{output_dir}/performance_summary_stats.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        self.logger.info(f"Performance comparison saved to {output_dir}/performance_comparison.csv")
