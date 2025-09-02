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
        step_months: int = 12
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Creates a list of training and testing splits for walk-forward validation.

    Behavior:
      1) Generate standard forward-rolling splits with FIXED test length.
         Only full test windows are included.
      2) Then append ONE extra split that is BACK-ALIGNED to end exactly
         at the last available date in data_index, with the same fixed test length,
         provided a valid training window (with embargo) exists and this split
         is not a duplicate of the last forward split.

    Args:
        data_index: The complete DatetimeIndex of the dataset.
        train_years: Training period length in YEARS (int or float).
        test_years:  Testing period length in YEARS (int or float).
        step_months: Step (months) to roll the window forward each split.

    Returns:
        A list of tuples: (train_index, test_index).

    Notes:
        - Fractional years are converted to whole months via rounding.
        - Windows are half-open intervals: [start, end)
        - Embargo days are enforced between train_end and test_start.
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
    # -------------------------------
    # 1) Standard forward-rolling splits (full test windows only)
    # -------------------------------
    while True:
        train_end = current_start + train_period
        test_start = train_end + embargo_period
        test_end = test_start + test_period  # exclusive bound

        # Require full test window within bounds
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

        next_start = current_start + step_period
        if next_start <= current_start:
            break
        current_start = next_start

    # -------------------------------
    # 2) Back-aligned final full window
    # -------------------------------
    # We force the final test window to end at data_end (inclusive),
    # keeping the SAME test length. Because our selection is [start, end),
    # use test_end_exclusive = end_date + 1 day to include end_date.
    try:
        test_end_exclusive = end_date + pd.DateOffset(days=1)
        test_start_back = test_end_exclusive - test_period  # same fixed length
        # Enforce embargo: training must end before the test (with embargo gap)
        train_end_back = test_start_back - embargo_period
        train_start_back = train_end_back - train_period

        # Build indices
        train_idx_back = data_index[(data_index >= train_start_back) & (data_index < train_end_back)]
        test_idx_back = data_index[(data_index >= test_start_back) & (data_index < test_end_exclusive)]

        # Valid if both sides have data, and training window is not empty
        if not train_idx_back.empty and not test_idx_back.empty:
            # Avoid adding a duplicate of the last forward split
            is_duplicate = False
            if splits:
                last_train, last_test = splits[-1]
                if (not last_test.empty and
                        last_test.min() == test_idx_back.min() and
                        last_test.max() == test_idx_back.max()):
                    is_duplicate = True

            if not is_duplicate:
                splits.append((train_idx_back, test_idx_back))
                print(
                    f"Created Back-Aligned Final Split: "
                    f"Train ({train_idx_back.min().date()} to {train_idx_back.max().date()}), "
                    f"Test ({test_idx_back.min().date()} to {test_idx_back.max().date()})"
                )
    except Exception as e:
        print(f" Warning: could not create back-aligned final split: {e}")

    print(f"Generated {len(splits)} walk-forward splits.")
    return splits
