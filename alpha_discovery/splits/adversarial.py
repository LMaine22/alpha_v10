"""Adversarial drift detection for distribution shifts."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def compute_adversarial_auc(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
    cv_folds: int = 3
) -> Tuple[float, Optional[pd.Series]]:
    """
    Adversarial validation: train classifier to distinguish train vs test.
    
    High AUC (>0.6) indicates distribution shift between train and test.
    Feature importances show which features drive the shift.
    
    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        n_estimators: Number of trees in RandomForest
        max_depth: Maximum tree depth
        random_state: Random seed
        cv_folds: Number of CV folds for scoring
        
    Returns:
        (auc_score, feature_importances) where:
        - auc_score: Cross-validated ROC-AUC (0.5=no shift, 1.0=total shift)
        - feature_importances: Series of importances (None if fit fails)
    """
    if X_train.empty or X_test.empty:
        return 0.5, None
    
    # Ensure same features
    common_cols = X_train.columns.intersection(X_test.columns)
    if len(common_cols) == 0:
        return 0.5, None
    
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Create labels: 0=train, 1=test
    y_train = np.zeros(len(X_train))
    y_test = np.ones(len(X_test))
    
    X_combined = pd.concat([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, y_test])
    
    # Handle missing values
    X_combined = X_combined.fillna(X_combined.median())
    
    try:
        # Train adversarial classifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Cross-validated AUC
        cv_scores = cross_val_score(
            clf, X_combined, y_combined,
            cv=cv_folds, scoring='roc_auc', n_jobs=-1
        )
        auc = float(np.mean(cv_scores))
        
        # Fit on full data for feature importances
        clf.fit(X_combined, y_combined)
        importances = pd.Series(
            clf.feature_importances_,
            index=common_cols
        ).sort_values(ascending=False)
        
        return auc, importances
        
    except Exception as e:
        print(f"[adversarial] Classifier failed: {e}")
        return 0.5, None


def compute_feature_importance_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Identify which features show the most distribution shift.
    
    Args:
        X_train: Training features
        X_test: Test features
        top_k: Number of top drifting features to return
        
    Returns:
        DataFrame with top-k features and their drift scores
    """
    auc, importances = compute_adversarial_auc(X_train, X_test)
    
    if importances is None:
        return pd.DataFrame()
    
    drift_df = pd.DataFrame({
        'feature': importances.index[:top_k],
        'drift_importance': importances.values[:top_k],
        'overall_auc': auc
    })
    
    return drift_df


def check_drift_gate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    auc_threshold: float = 0.65,
    **kwargs
) -> Tuple[bool, float, str]:
    """
    Gate check: reject test split if distribution shift is too large.
    
    Args:
        X_train: Training features
        X_test: Test features
        auc_threshold: Maximum allowable AUC (default 0.65)
        **kwargs: Additional args passed to compute_adversarial_auc
        
    Returns:
        Tuple of (passed, auc_score, message)
        - passed: True if drift is within threshold
        - auc_score: Measured AUC
        - message: Explanation
    """
    auc, _ = compute_adversarial_auc(X_train, X_test, **kwargs)
    
    passed = auc < auc_threshold
    
    if passed:
        message = f"✓ Drift check passed (AUC={auc:.3f} < {auc_threshold})"
    else:
        message = f"✗ Drift check FAILED (AUC={auc:.3f} >= {auc_threshold})"
    
    return passed, auc, message
