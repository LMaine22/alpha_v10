# alpha_discovery/eval/selection.py
"""
Support filtering and selection utilities for forecast-based evaluation.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from ..config import settings


def apply_support_filters(trigger_series: pd.Series, min_support: int = 12) -> pd.Index:
    """
    Apply support filters to identify valid trigger dates that meet minimum firing threshold.
    
    Args:
        trigger_series: Boolean series indicating when signals fire
        min_support: Minimum number of fires required for a setup to be valid
        
    Returns:
        Index of dates where the setup meets the minimum support requirement
    """
    if trigger_series.empty:
        return pd.Index([], dtype='datetime64[ns]')
    
    # Check if total number of fires meets minimum support
    total_fires = trigger_series.sum()
    if total_fires < min_support:
        return pd.Index([], dtype='datetime64[ns]')
    
    # If total fires meets support, return all trigger dates
    return trigger_series[trigger_series].index


def get_valid_trigger_dates(signals_df: pd.DataFrame, setup: List[str], min_support: int) -> pd.DatetimeIndex:
    """
    Get all dates where a setup fired, provided it meets minimum total support.
    """
    if not setup:
        return pd.DatetimeIndex([])
    
    mask = get_trigger_mask(signals_df, setup)
    
    if mask.sum() < min_support:
        return pd.DatetimeIndex([])
        
    return signals_df.index[mask]


def get_trigger_mask(signals_df: pd.DataFrame, setup: List[str]) -> pd.Series:
    """
    Computes the boolean mask for final trigger events (eligibility AND activation).
    """
    if not setup or not all(s in signals_df.columns for s in setup):
        return pd.Series(False, index=signals_df.index)
        
    return signals_df[setup].all(axis=1)


def get_eligibility_mask(
    setup: List[str],
    signals_df: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    signals_meta: List[dict]
) -> pd.Series:
    """
    A day is eligible if all underlying features for the setup's signals are non-NaN.
    This is the true conjunct logic before the final signal activation.
    """
    meta_map = {m['signal_id']: m for m in signals_meta}
    source_features = set()
    for sid in setup:
        if sid in meta_map:
            source_features.add(meta_map[sid]['feature_name'])

    if not source_features:
        return pd.Series(False, index=signals_df.index)

    # A day is eligible if none of the source features are NaN
    return feature_matrix[list(source_features)].notna().all(axis=1)


def filter_returns_by_support(
    returns_series: pd.Series,
    trigger_series: pd.Series,
    min_support: int = 12
) -> pd.Series:
    """
    Filter returns to only include periods where the setup meets minimum support.
    
    Args:
        returns_series: Series of returns to filter
        trigger_series: Boolean series indicating signal fires
        min_support: Minimum support required
        
    Returns:
        Filtered returns series
    """
    valid_dates = apply_support_filters(trigger_series, min_support)
    
    if valid_dates.empty:
        return pd.Series(dtype=float, index=returns_series.index)
    
    # Create a mask for valid dates
    valid_mask = returns_series.index.isin(valid_dates)
    
    # Return filtered series (NaN for invalid dates)
    filtered_returns = returns_series.copy()
    filtered_returns[~valid_mask] = np.nan
    
    return filtered_returns


def calculate_setup_support_stats(
    trigger_series: pd.Series,
    min_support: int = 12
) -> dict:
    """
    Calculate support statistics for a trigger series.
    
    Args:
        trigger_series: Boolean series indicating signal fires
        min_support: Minimum support required
        
    Returns:
        Dictionary with support statistics
    """
    total_fires = trigger_series.sum()
    valid_dates = apply_support_filters(trigger_series, min_support)
    valid_fires = len(valid_dates)
    
    # Calculate streak statistics
    signal_streaks = trigger_series.groupby((~trigger_series).cumsum()).cumsum()
    max_streak = signal_streaks.max() if not signal_streaks.empty else 0
    avg_streak = signal_streaks[trigger_series].mean() if trigger_series.any() else 0
    
    return {
        'total_fires': int(total_fires),
        'valid_fires': int(valid_fires),
        'support_ratio': valid_fires / total_fires if total_fires > 0 else 0.0,
        'max_streak': int(max_streak),
        'avg_streak': float(avg_streak),
        'meets_min_support': valid_fires >= min_support
    }


# Re-export selection_core functions for backward compatibility
from .selection_core import *  # noqa: F401,F403

# Keep an explicit __all__ for clarity
from .selection_core import (
    TickerBest,
    score_ticker_horizon,
    select_best_horizon_per_ticker,
    stepwise_select_portfolio,
    filter_ledger_to_selection,
    portfolio_daily_returns,
    portfolio_metrics,
    assemble_portfolio_stepwise,
    selection_summary,
)

__all__ = [
    "TickerBest",
    "score_ticker_horizon",
    "select_best_horizon_per_ticker",
    "stepwise_select_portfolio",
    "filter_ledger_to_selection",
    "portfolio_daily_returns",
    "portfolio_metrics",
    "assemble_portfolio_stepwise",
    "selection_summary",
    # New functions
    "apply_support_filters",
    "get_valid_trigger_dates",
    "filter_returns_by_support",
    "calculate_setup_support_stats",
]