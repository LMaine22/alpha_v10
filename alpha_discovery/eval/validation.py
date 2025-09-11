# alpha_discovery/eval/validation.py

import pandas as pd
from typing import List, Tuple, Union

from ..config import settings


Number = Union[int, float]


def _years_to_months(years: Number) -> int:
    """
    Convert (possibly fractional) years to whole months.
    Example: 0.5 -> 6, 3 -> 36.
    Rounds to nearest integer month for stability.
    """
    months = int(round(float(years) * 12))
    return max(months, 1)


def create_walk_forward_splits(
        data_index: pd.DatetimeIndex,
        train_years: Number = 3,
        test_years: Number = 1,
        step_months: int = 9
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Creates a list of training and testing splits for walk-forward validation.

    Args:
        data_index: The complete DatetimeIndex of the dataset.
        train_years: Training period length in years (int or float).
        test_years: Testing period length in years (int or float).
        step_months: Step (months) to roll the window forward each split.

    Returns:
        A list of tuples: (train_index, test_index).
    """
    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    if data_index.empty:
        print(" Warning: data_index is empty; no splits can be created.")
        return splits

    start_date = data_index.min()
    end_date = data_index.max()

    # Convert year lengths to month-based DateOffsets
    train_months = _years_to_months(train_years)
    test_months = _years_to_months(test_years)

    train_period = pd.DateOffset(months=train_months)
    test_period = pd.DateOffset(months=test_months)
    step_period = pd.DateOffset(months=int(step_months))
    embargo_period = pd.DateOffset(days=int(settings.validation.embargo_days))

    current_start = start_date

    print("\n--- Creating Walk-Forward Splits ---")
    
    while True:
        train_end = current_start + train_period
        test_start = train_end + embargo_period
        test_end = test_start + test_period

        # Stop if we can't create a complete test window
        if test_end > end_date:
            break

        train_indices = data_index[(data_index >= current_start) & (data_index < train_end)]
        test_indices = data_index[(data_index >= test_start) & (data_index < test_end)]

        if not train_indices.empty and not test_indices.empty:
            splits.append((train_indices, test_indices))
            print(
                f"Created Split {len(splits)}: "
                f"Train ({train_indices.min().date()} to {train_indices.max().date()}), "
                f"Test ({test_indices.min().date()} to {test_indices.max().date()})"
            )

        # Move to next window
        current_start += step_period

    print(f"Generated {len(splits)} walk-forward splits.")
    print(f"Total folds created: {len(splits)}")
    return splits
