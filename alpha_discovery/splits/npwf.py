"""Nested Purged Walk-Forward (NPWF) for inner GA selection."""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np


def make_inner_folds(
    df_train_outer: pd.DataFrame,
    label_horizon_days: int,
    feature_lookback_tail: int,
    k_folds: int = 5
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Create nested inner folds for GA hyperparameter selection.
    
    Uses the same purge/embargo logic as PAWF but operates on the
    outer training window. Returns k anchored walk-forward folds.
    
    Args:
        df_train_outer: Training data from outer split (already purged)
        label_horizon_days: Forecast horizon for purge calculation
        feature_lookback_tail: Max feature lookback for embargo
        k_folds: Number of inner folds (default 5)
        
    Returns:
        List of (train_idx, test_idx) tuples for inner CV
        
    Raises:
        ValueError: If insufficient data for k folds
    """
    if not isinstance(df_train_outer.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if df_train_outer.empty:
        raise ValueError("Training data cannot be empty")
    
    # Calculate purge and embargo (same as PAWF)
    purge_days = label_horizon_days
    embargo_days = max(feature_lookback_tail, 5)
    
    train_idx = df_train_outer.index.sort_values()
    data_start = train_idx.min()
    data_end = train_idx.max()
    total_days = (data_end - data_start).days
    
    # Minimum requirements
    min_train_days = 252  # ~1 year
    min_test_days = 21    # ~1 month
    
    if total_days < (min_train_days + min_test_days + purge_days + embargo_days) * k_folds * 0.5:
        raise ValueError(
            f"Insufficient data for {k_folds} folds. "
            f"Have {total_days} days, need ~{(min_train_days + min_test_days) * k_folds * 0.5}"
        )
    
    # Create expanding window folds
    folds = []
    
    # Divide the data into k roughly equal segments for test windows
    test_segment_days = total_days // (k_folds + 1)  # Leave room for initial training
    
    for fold_num in range(k_folds):
        # Test window: positioned at increasing offsets
        test_start_offset = min_train_days + (fold_num * test_segment_days)
        test_start = data_start + pd.Timedelta(days=test_start_offset)
        test_end = test_start + pd.Timedelta(days=min_test_days)
        
        # Don't exceed data range
        if test_end > data_end:
            break
        
        # Training window: expanding from start, with purge before test
        fold_train_start = data_start
        fold_train_end = test_start - pd.Timedelta(days=purge_days + 1)
        
        # Check minimum training requirement
        if (fold_train_end - fold_train_start).days < min_train_days * 0.8:
            continue  # Skip if training window too small
        
        # Get actual date ranges from the index
        train_mask = (train_idx >= fold_train_start) & (train_idx <= fold_train_end)
        test_mask = (train_idx >= test_start) & (train_idx <= test_end)
        
        fold_train_idx = train_idx[train_mask]
        fold_test_idx = train_idx[test_mask]
        
        # Apply embargo: remove train points within embargo_days after test_end
        embargo_cutoff = test_end + pd.Timedelta(days=embargo_days)
        fold_train_idx = fold_train_idx[fold_train_idx < embargo_cutoff]
        
        # Ensure we have data in both windows
        if len(fold_train_idx) < 100 or len(fold_test_idx) < 10:
            continue
        
        folds.append((fold_train_idx, fold_test_idx))
    
    if not folds:
        raise ValueError(
            f"No valid inner folds generated from {total_days} days. "
            f"Check purge_days={purge_days}, embargo_days={embargo_days}"
        )
    
    return folds


def summarize_inner_folds(
    folds: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
) -> pd.DataFrame:
    """
    Create a summary DataFrame of inner folds.
    
    Args:
        folds: List of (train_idx, test_idx) tuples
        
    Returns:
        DataFrame with fold metadata
    """
    rows = []
    for i, (train_idx, test_idx) in enumerate(folds):
        rows.append({
            "fold": i,
            "train_start": train_idx.min().date(),
            "train_end": train_idx.max().date(),
            "test_start": test_idx.min().date(),
            "test_end": test_idx.max().date(),
            "train_obs": len(train_idx),
            "test_obs": len(test_idx),
            "train_span_days": (train_idx.max() - train_idx.min()).days,
            "test_span_days": (test_idx.max() - test_idx.min()).days
        })
    
    return pd.DataFrame(rows)
