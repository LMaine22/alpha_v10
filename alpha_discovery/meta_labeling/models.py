# alpha_discovery/meta_labeling/models.py
"""
Meta-Model Training for Meta-Labeling

This module handles training of binary classifiers for meta-labeling,
including cross-validation, threshold optimization, and model selection.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ..config import settings


@dataclass
class ModelResult:
    """Result from meta-model training."""
    model: Any
    threshold: float
    cv_scores: Dict[str, List[float]]
    feature_importance: Dict[str, float]
    calibration_score: float
    best_params: Dict[str, Any]


class MetaModelTrainer:
    """Trains meta-models for trade filtering."""
    
    def __init__(self, config: Any):
        """Initialize the model trainer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, setup_id: str, features: pd.DataFrame, 
                   labels: pd.Series, oos_ledger: pd.DataFrame) -> Optional[ModelResult]:
        """
        Train a meta-model for a setup.
        
        Args:
            setup_id: Setup identifier
            features: Meta-features DataFrame
            labels: Binary labels (1=WIN, 0=LOSS)
            oos_ledger: OOS trade ledger
            
        Returns:
            ModelResult or None if training fails
        """
        self.logger.info(f"Training meta-model for setup {setup_id}")
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        if len(common_index) == 0:
            self.logger.error(f"No common index between features and labels for {setup_id}")
            return None
        
        X = features.loc[common_index]
        y = labels.loc[common_index]
        
        # Check for sufficient data
        if len(X) < self.config.min_trades_for_meta:
            self.logger.warning(f"Insufficient data for {setup_id}: {len(X)} < {self.config.min_trades_for_meta}")
            return None
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Time-ordered cross-validation
        cv_scores = self._time_series_cv(X, y, oos_ledger.loc[common_index])
        
        if not cv_scores:
            self.logger.error(f"Cross-validation failed for {setup_id}")
            return None
        
        # Train final model on all data
        model = self._create_model()
        model.fit(X, y)
        
        # Calibrate the model for probability estimation
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X, y)
        
        # Calculate calibration score
        calibration_score = self._calculate_calibration_score(calibrated_model, X, y)
        
        # Optimize threshold
        threshold = self._optimize_threshold(calibrated_model, X, y, oos_ledger.loc[common_index])
        
        # Get feature importance
        feature_importance = self._get_feature_importance(model, X.columns)
        
        return ModelResult(
            model=calibrated_model,
            threshold=threshold,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            calibration_score=calibration_score,
            best_params=self._get_best_params(model)
        )
    
    def _time_series_cv(self, X: pd.DataFrame, y: pd.Series, 
                       oos_ledger: pd.DataFrame) -> Dict[str, List[float]]:
        """Perform time-ordered cross-validation."""
        # Sort by entry date to ensure time order
        entry_dates = oos_ledger['entry_date']
        sorted_indices = entry_dates.argsort()
        
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]
        
        # Time series split with embargo
        embargo_days = self.config.embargo_days
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, test_idx in tscv.split(X_sorted):
            # Apply embargo period
            train_end_date = entry_dates.iloc[train_idx[-1]]
            test_start_date = entry_dates.iloc[test_idx[0]]
            
            # Skip if test set is too close to training set
            if (test_start_date - train_end_date).days < embargo_days:
                continue
            
            X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
            y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
            
            # Train model
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate scores
            cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            cv_scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        return cv_scores
    
    def _create_model(self):
        """Create a model based on configuration."""
        model_type = self.config.model_type
        
        if model_type == "logistic_regression":
            return LogisticRegression(
                random_state=42,
                max_iter=1000,
                **self.config.model_params
            )
        elif model_type == "random_forest":
            return RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                **self.config.model_params
            )
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ValueError("XGBoost is not available. Please install it or use a different model type.")
            return xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                **self.config.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _optimize_threshold(self, model, X: pd.DataFrame, y: pd.Series, 
                          oos_ledger: pd.DataFrame) -> float:
        """Optimize decision threshold based on expected value."""
        # Get predicted probabilities
        y_proba = model.predict_proba(X)[:, 1]
        
        # Get P&L values
        pnl_col = None
        for col in ['pnl_dollars', 'realized_pnl', 'pnl_pct']:
            if col in oos_ledger.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            # Default threshold if no P&L data
            return 0.5
        
        pnl_values = oos_ledger[pnl_col].values
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_metric = -np.inf
        
        for threshold in thresholds:
            # Apply threshold
            decisions = (y_proba >= threshold).astype(int)
            
            # Calculate retention rate
            retention_rate = decisions.mean()
            
            # Check retention constraints
            if retention_rate < self.config.min_trade_retention:
                continue
            if retention_rate > self.config.max_trade_retention:
                continue
            
            # Calculate metric based on configuration
            if self.config.threshold_optimization_metric == "expected_value":
                # Expected value of filtered trades
                filtered_pnl = pnl_values[decisions == 1]
                if len(filtered_pnl) > 0:
                    metric = filtered_pnl.mean()
                else:
                    metric = -np.inf
            elif self.config.threshold_optimization_metric == "sharpe":
                # Sharpe ratio of filtered trades
                filtered_pnl = pnl_values[decisions == 1]
                if len(filtered_pnl) > 1:
                    metric = filtered_pnl.mean() / filtered_pnl.std() if filtered_pnl.std() > 0 else 0
                else:
                    metric = -np.inf
            elif self.config.threshold_optimization_metric == "sortino":
                # Sortino ratio of filtered trades
                filtered_pnl = pnl_values[decisions == 1]
                negative_returns = filtered_pnl[filtered_pnl < 0]
                if len(filtered_pnl) > 1 and len(negative_returns) > 0:
                    downside_std = negative_returns.std()
                    metric = filtered_pnl.mean() / downside_std if downside_std > 0 else 0
                else:
                    metric = -np.inf
            else:
                raise ValueError(f"Unknown optimization metric: {self.config.threshold_optimization_metric}")
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold
        
        return best_threshold
    
    def _calculate_calibration_score(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate calibration score (reliability of probability estimates)."""
        try:
            y_proba = model.predict_proba(X)[:, 1]
            
            # Bin probabilities and calculate calibration
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y[in_bin].mean()
                    avg_confidence_in_bin = y_proba[in_bin].mean()
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return 1.0 - calibration_error  # Higher is better
        except Exception as e:
            self.logger.warning(f"Calibration score calculation failed: {e}")
            return 0.0
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from the model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            
            # Normalize importances
            if len(importances) > 0:
                importances = importances / importances.sum()
            
            return dict(zip(feature_names, importances))
        except Exception as e:
            self.logger.warning(f"Feature importance extraction failed: {e}")
            return {}
    
    def _get_best_params(self, model) -> Dict[str, Any]:
        """Get best parameters from the model."""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Parameter extraction failed: {e}")
            return {}
