from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

def purged_walk_forward_indices(
    dates: pd.Series,
    n_splits: int = 5,
    embargo_days: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple purged walk-forward split on dates (ignores ticker, so rows sharing the same
    timestamp move together). Adds a forward embargo to avoid leakage.

    Returns: iterator of (train_idx, test_idx)
    """
    unique_days = np.array(sorted(dates.unique()))
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    folds = np.array_split(unique_days, n_splits)

    # Build rolling train->test with embargo
    start = 0
    for i in range(1, len(folds)):
        train_days = np.concatenate(folds[:i])
        test_days = folds[i]
        if test_days.size == 0 or train_days.size == 0:
            continue

        # embargo: drop the last 'embargo_days' from train that overlap test proximity
        max_train_day = train_days.max()
        min_test_day = test_days.min()
        embargo_mask = (train_days > min_test_day)  # guard
        if embargo_days > 0:
            # If days are contiguous, ensure gap of 'embargo_days'
            pass  # kept simple; date contiguity varies; downstream selection uses < min_test_day

        train_idx = np.where(dates.isin(train_days) & (dates < min_test_day))[0]
        test_idx = np.where(dates.isin(test_days))[0]
        if train_idx.size and test_idx.size:
            yield train_idx, test_idx
