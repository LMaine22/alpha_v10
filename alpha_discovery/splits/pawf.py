"""Purged Anchored Walk-Forward (PAWF) outer splits."""

from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np

from .ids import SplitSpec, generate_split_id


def build_pawf_splits(
    df: pd.DataFrame,
    label_horizon_days: int,
    feature_lookback_tail: int,
    min_train_months: int = 36,
    test_window_days: int = 21,
    step_months: int = 1,
    regime_version: str = "R1"
) -> List[SplitSpec]:
    """
    Build Purged Anchored Walk-Forward outer splits.
    
    Implements expanding window walk-forward with:
    - Minimum training window (default 36 months)
    - Fixed test window (default 21 trading days)
    - Monthly step increments
    - Automatic purge = label_horizon_days
    - Automatic embargo = max(feature_lookback_tail, 5)
    
    Args:
        df: Master DataFrame with DatetimeIndex
        label_horizon_days: Forecast horizon in days (for purge calculation)
        feature_lookback_tail: Maximum feature lookback window (for embargo)
        min_train_months: Minimum training window in months (default 36)
        test_window_days: Test window size in calendar days (default 21)
        step_months: Step size for walking forward (default 1 month)
        regime_version: Regime model version tag (default "R1")
        
    Returns:
        List of SplitSpec objects with proper purge/embargo
        
    Raises:
        ValueError: If insufficient data for minimum requirements
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Calculate purge and embargo
    purge_days = label_horizon_days
    embargo_days = max(feature_lookback_tail, 5)
    
    # Get data boundaries
    data_start = df.index.min()
    data_end = df.index.max()
    
    # Minimum training period
    min_train_period = pd.DateOffset(months=min_train_months)
    first_possible_test_start = data_start + min_train_period
    
    if first_possible_test_start >= data_end:
        raise ValueError(
            f"Insufficient data: need at least {min_train_months} months "
            f"but have {(data_end - data_start).days / 30:.1f} months"
        )
    
    splits = []
    current_test_start = first_possible_test_start
    
    while current_test_start < data_end:
        # Test window
        test_start = current_test_start
        test_end = test_start + pd.DateOffset(days=test_window_days - 1)
        
        # Don't exceed data range
        if test_end > data_end:
            break
        
        # Training window (expanding from start, with purge before test)
        train_start = data_start
        train_end = test_start - pd.DateOffset(days=purge_days + 1)
        
        # Ensure minimum training data
        if (train_end - train_start).days < 30 * min_train_months * 0.8:
            # Skip this split if training window too small
            current_test_start += pd.DateOffset(months=step_months)
            continue
        
        # Create split spec
        spec = SplitSpec(
            outer_id=f"outer_{len(splits):03d}",
            split_version="PAWF_v1",
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            purge_days=purge_days,
            embargo_days=embargo_days,
            label_horizon=label_horizon_days,
            feature_lookback_tail=feature_lookback_tail,
            regime_version=regime_version,
            event_class="normal"
        )
        
        splits.append(spec)
        
        # Step forward
        current_test_start += pd.DateOffset(months=step_months)
    
    if not splits:
        raise ValueError(
            f"No valid splits generated. Check min_train_months={min_train_months} "
            f"and data span={(data_end - data_start).days / 30:.1f} months"
        )
    
    return splits


def summarize_pawf_splits(splits: List[SplitSpec]) -> pd.DataFrame:
    """
    Create a summary DataFrame of PAWF splits.
    
    Args:
        splits: List of SplitSpec objects
        
    Returns:
        DataFrame with split metadata
    """
    rows = []
    for spec in splits:
        rows.append({
            "split_id": generate_split_id(spec),
            "outer_id": spec.outer_id,
            "train_start": spec.train_start.date(),
            "train_end": spec.train_end.date(),
            "test_start": spec.test_start.date(),
            "test_end": spec.test_end.date(),
            "train_days": spec.train_span_days,
            "test_days": spec.test_span_days,
            "purge_days": spec.purge_days,
            "embargo_days": spec.embargo_days,
            "horizon": spec.label_horizon,
            "regime_ver": spec.regime_version
        })
    
    return pd.DataFrame(rows)
