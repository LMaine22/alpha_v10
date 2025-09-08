# alpha_discovery/meta_labeling/evaluation.py
"""
Meta-Labeling Evaluation Module

This module handles evaluation of meta-models, including shadow mode
replay, performance comparison, and decision tracking.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from ..config import settings
from ..eval.nav import sharpe, nav_daily_returns_from_ledger
from .types import MetaLabelingResults


class MetaEvaluator:
    """Evaluates meta-models and generates performance comparisons."""
    
    def __init__(self, config: Any):
        """Initialize the evaluator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, setup_id: str, setup_info: Dict, model_result: Any,
                      features: pd.DataFrame, labels: pd.Series, 
                      oos_ledger: pd.DataFrame) -> MetaLabelingResults:
        """
        Evaluate a meta-model and generate results.
        
        Args:
            setup_id: Setup identifier
            setup_info: Setup information
            model_result: Trained model result
            features: Meta-features
            labels: Binary labels
            oos_ledger: OOS trade ledger
            
        Returns:
            MetaLabelingResults with evaluation metrics
        """
        self.logger.info(f"Evaluating meta-model for setup {setup_id}")
        
        # Align data
        common_index = features.index.intersection(labels.index).intersection(oos_ledger.index)
        if len(common_index) == 0:
            raise ValueError("No common index between features, labels, and ledger")
        
        X = features.loc[common_index]
        y = labels.loc[common_index]
        ledger = oos_ledger.loc[common_index]
        
        # Get model predictions
        y_proba = model_result.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= model_result.threshold).astype(int)
        
        # Calculate base performance metrics
        base_metrics = self._calculate_base_metrics(ledger)
        
        # Calculate meta performance metrics (shadow mode)
        meta_metrics, meta_decisions = self._calculate_meta_metrics(
            ledger, y_proba, model_result.threshold
        )
        
        # Calculate model performance metrics
        model_metrics = self._calculate_model_metrics(y, y_pred, y_proba)
        
        # Calculate feature importance
        feature_importance = model_result.feature_importance
        
        return MetaLabelingResults(
            setup_id=setup_id,
            ticker=setup_info.get('ticker', 'Unknown'),
            direction=setup_info.get('direction', 'long'),
            base_expected_value=base_metrics['expected_value'],
            meta_expected_value=meta_metrics['expected_value'],
            base_sharpe=base_metrics['sharpe'],
            meta_sharpe=meta_metrics['sharpe'],
            base_max_drawdown=base_metrics['max_drawdown'],
            meta_max_drawdown=meta_metrics['max_drawdown'],
            base_trade_count=base_metrics['trade_count'],
            meta_trade_count=meta_metrics['trade_count'],
            trade_retention_rate=meta_metrics['retention_rate'],
            model_accuracy=model_metrics['accuracy'],
            model_precision=model_metrics['precision'],
            model_recall=model_metrics['recall'],
            model_f1=model_metrics['f1'],
            model_calibration=model_result.calibration_score,
            feature_importance=feature_importance,
            meta_decisions=meta_decisions,
            status="trained"
        )
    
    def _calculate_base_metrics(self, ledger: pd.DataFrame) -> Dict[str, float]:
        """Calculate base performance metrics."""
        # Get P&L column
        pnl_col = None
        for col in ['pnl_dollars', 'realized_pnl', 'pnl_pct']:
            if col in ledger.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            raise ValueError("No P&L column found in ledger")
        
        pnl_values = ledger[pnl_col]
        
        # Calculate metrics
        expected_value = pnl_values.mean()
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_values)
        max_drawdown = self._calculate_max_drawdown(pnl_values)
        trade_count = len(ledger)
        
        return {
            'expected_value': expected_value,
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count
        }
    
    def _calculate_meta_metrics(self, ledger: pd.DataFrame, y_proba: np.ndarray, 
                               threshold: float) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Calculate meta-filtered performance metrics."""
        # Apply meta-filter
        meta_decisions = (y_proba >= threshold).astype(int)
        
        # Filter ledger based on meta decisions
        filtered_ledger = ledger[meta_decisions == 1]
        
        if len(filtered_ledger) == 0:
            # No trades passed meta-filter
            return {
                'expected_value': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'trade_count': 0,
                'retention_rate': 0.0
            }, []
        
        # Calculate metrics on filtered trades
        pnl_col = None
        for col in ['pnl_dollars', 'realized_pnl', 'pnl_pct']:
            if col in filtered_ledger.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            raise ValueError("No P&L column found in filtered ledger")
        
        pnl_values = filtered_ledger[pnl_col]
        
        expected_value = pnl_values.mean()
        sharpe_ratio = self._calculate_sharpe_ratio(pnl_values)
        max_drawdown = self._calculate_max_drawdown(pnl_values)
        trade_count = len(filtered_ledger)
        retention_rate = len(filtered_ledger) / len(ledger)
        
        # Create meta decisions log
        decisions_log = []
        for i, (idx, decision) in enumerate(zip(ledger.index, meta_decisions)):
            decisions_log.append({
                'trade_index': i,
                'entry_date': ledger.loc[idx, 'entry_date'],
                'exit_date': ledger.loc[idx, 'exit_date'],
                'predicted_probability': y_proba[i],
                'meta_decision': 'TAKE' if decision == 1 else 'SKIP',
                'base_pnl': ledger.loc[idx, pnl_col],
                'meta_pnl': ledger.loc[idx, pnl_col] if decision == 1 else 0.0
            })
        
        return {
            'expected_value': expected_value,
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'retention_rate': retention_rate
        }, decisions_log
    
    def _calculate_model_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                                y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Return maximum drawdown
        return abs(drawdown.min())
    
    def generate_shadow_curves(self, results: List[MetaLabelingResults], 
                              output_dir: str) -> None:
        """Generate shadow mode equity curves for comparison."""
        if not self.config.generate_artifacts:
            return
        
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Meta-Labeling Shadow Mode Results', fontsize=16)
        
        # Plot 1: Expected Value Comparison
        setup_ids = [r.setup_id for r in results if r.status == "trained"]
        base_ev = [r.base_expected_value for r in results if r.status == "trained"]
        meta_ev = [r.meta_expected_value for r in results if r.status == "trained"]
        
        axes[0, 0].bar(range(len(setup_ids)), base_ev, alpha=0.7, label='Base', color='blue')
        axes[0, 0].bar(range(len(setup_ids)), meta_ev, alpha=0.7, label='Meta', color='red')
        axes[0, 0].set_title('Expected Value Comparison')
        axes[0, 0].set_ylabel('Expected Value')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Sharpe Ratio Comparison
        base_sharpe = [r.base_sharpe for r in results if r.status == "trained"]
        meta_sharpe = [r.meta_sharpe for r in results if r.status == "trained"]
        
        axes[0, 1].bar(range(len(setup_ids)), base_sharpe, alpha=0.7, label='Base', color='blue')
        axes[0, 1].bar(range(len(setup_ids)), meta_sharpe, alpha=0.7, label='Meta', color='red')
        axes[0, 1].set_title('Sharpe Ratio Comparison')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Trade Retention Rate
        retention_rates = [r.trade_retention_rate for r in results if r.status == "trained"]
        
        axes[1, 0].bar(range(len(setup_ids)), retention_rates, alpha=0.7, color='green')
        axes[1, 0].set_title('Trade Retention Rate')
        axes[1, 0].set_ylabel('Retention Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Model Accuracy
        accuracies = [r.model_accuracy for r in results if r.status == "trained"]
        
        axes[1, 1].bar(range(len(setup_ids)), accuracies, alpha=0.7, color='orange')
        axes[1, 1].set_title('Model Accuracy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/meta_labeling_shadow_results.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Shadow mode results saved to {output_dir}/meta_labeling_shadow_results.png")
