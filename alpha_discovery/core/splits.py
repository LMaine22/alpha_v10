from __future__ import annotations
from typing import List, Tuple, Optional, NamedTuple
import pandas as pd
from datetime import date

from ..config import settings

# A structured way to return the different split sets
class HybridSplits(NamedTuple):
    discovery_cv: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
    oos: List[pd.DatetimeIndex]
    gauntlet: Optional[pd.DatetimeIndex]


def _years_to_months(years: float) -> int:
    """Converts (possibly fractional) years to the nearest whole number of months."""
    return max(1, int(round(years * 12)))


def make_walkforward_splits(index: pd.DatetimeIndex, n_folds: int, embargo_days: int):
    """Simple time-ordered walk-forward splits used by GA metrics only."""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) < 10 or n_folds < 1: return []
    test_len = max(5, len(idx) // (n_folds + 1))
    embargo = pd.Timedelta(days=int(embargo_days))
    purge = pd.Timedelta(days=int(settings.validation.purge_days))
    splits = []
    start = 0
    for k in range(n_folds):
        train_end_pos = min((k + 1) * test_len, len(idx) - 1)
        train_end_time = idx[train_end_pos]
        train_idx = idx[(idx < train_end_time - purge)]
        test_start_time = train_end_time + embargo
        test_end_time = test_start_time + pd.Timedelta(days=test_len) # This is an approximation
        test_idx = idx[(idx >= test_start_time) & (idx < test_end_time)]
        if train_idx.size and test_idx.size:
            splits.append((train_idx, test_idx))
    return splits


def create_hybrid_splits(data_index: pd.DatetimeIndex) -> HybridSplits:
    """
    Orchestrates the creation of the full three-stage split:
    1. Discovery CV (walk-forward splits for training/validation)
    2. True OOS (hold-out periods for final evaluation)
    3. Forward Gauntlet (a final rolling window for live simulation)

    Args:
        data_index: The complete DatetimeIndex of the dataset.

    Returns:
        A HybridSplits object containing all generated splits.
    """
    if not isinstance(data_index, pd.DatetimeIndex) or data_index.empty:
        return HybridSplits(discovery_cv=[], oos=[], gauntlet=None)

    # 1. Generate Discovery CV splits
    discovery_splits = _create_discovery_cv_splits(data_index)

    if not discovery_splits:
        # If no discovery splits can be made, we can't create the others either
        return HybridSplits(discovery_cv=[], oos=[], gauntlet=None)

    # Get the end of the last test set from discovery to anchor the OOS/Gauntlet
    last_discovery_test_end = discovery_splits[-1][1].max()

    # 2. Generate True OOS splits
    oos_splits = _create_oos_splits(data_index, start_after=last_discovery_test_end)

    # 3. Generate Forward Gauntlet split
    # Gauntlet should start after the last OOS period ends.
    last_oos_end = oos_splits[-1].max() if oos_splits else last_discovery_test_end
    gauntlet_split = _create_gauntlet_split(data_index, start_after=last_oos_end)
    
    return HybridSplits(
        discovery_cv=discovery_splits,
        oos=oos_splits,
        gauntlet=gauntlet_split
    )


def _create_discovery_cv_splits(data_index: pd.DatetimeIndex) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Creates walk-forward splits for the Discovery CV phase."""
    splits_cfg = settings.splits
    train_months = _years_to_months(splits_cfg.discovery_train_years)
    test_months = _years_to_months(splits_cfg.discovery_test_years)
    step_months = int(splits_cfg.discovery_step_months)
    embargo_days = int(settings.validation.embargo_days)

    train_period = pd.DateOffset(months=train_months)
    test_period = pd.DateOffset(months=test_months)
    step_period = pd.DateOffset(months=step_months)
    embargo_period = pd.DateOffset(days=embargo_days)
    purge_period = pd.DateOffset(days=int(settings.validation.purge_days))
    
    cv_splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    
    start_date = data_index.min()
    end_date = data_index.max()
    current_start = start_date

    while len(cv_splits) < splits_cfg.n_discovery_folds:
        train_end_raw = current_start + train_period
        
        # Effective train end is truncated by purge period
        train_end_effective = train_end_raw - purge_period
        
        # Test start is separated from the *effective* train end by the embargo
        test_start = train_end_effective + embargo_period
        test_end = test_start + test_period

        if test_end > end_date:
            break

        # Train indices use the raw (un-purged) end for windowing
        train_indices = data_index[(data_index >= current_start) & (data_index < train_end_effective)]
        test_indices = data_index[(data_index >= test_start) & (data_index < test_end)]

        if not train_indices.empty and not test_indices.empty:
            cv_splits.append((train_indices, test_indices))

        current_start += step_period
        
    return cv_splits


def _create_oos_splits(data_index: pd.DatetimeIndex, start_after: pd.Timestamp) -> List[pd.DatetimeIndex]:
    """Creates one or more hold-out OOS periods."""
    splits_cfg = settings.splits
    oos_splits: List[pd.DatetimeIndex] = []
    
    # OOS starts after an additional embargo period from the last CV test date
    embargo = pd.DateOffset(days=int(settings.validation.embargo_days))
    current_start = start_after + embargo
    
    if current_start >= data_index.max():
        return []

    fold_period = pd.DateOffset(months=splits_cfg.oos_fold_months)
    
    for _ in range(splits_cfg.n_oos_folds):
        fold_end = current_start + fold_period
        
        if fold_end > data_index.max():
            # If a full fold can't be made, take what's left
            fold_end = data_index.max()
        
        oos_indices = data_index[(data_index >= current_start) & (data_index <= fold_end)]
        
        if not oos_indices.empty:
            oos_splits.append(oos_indices)
        
        current_start = fold_end
        if current_start >= data_index.max():
            break
            
    return oos_splits


def _create_gauntlet_split(data_index: pd.DatetimeIndex, start_after: pd.Timestamp) -> Optional[pd.DatetimeIndex]:
    """Creates the final rolling 'live' gauntlet window."""
    splits_cfg = settings.splits
    
    # Gauntlet starts after an additional embargo period
    embargo = pd.DateOffset(days=int(settings.validation.embargo_days))
    start_date = start_after + embargo

    if start_date >= data_index.max():
        return None

    end_date = pd.to_datetime(splits_cfg.gauntlet_end_date) if splits_cfg.gauntlet_end_date else data_index.max()
    
    gauntlet_indices = data_index[(data_index >= start_date) & (data_index <= end_date)]

    return gauntlet_indices if not gauntlet_indices.empty else None
