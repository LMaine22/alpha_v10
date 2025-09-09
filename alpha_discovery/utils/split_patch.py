import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Sequence

Split = Tuple[Sequence, Sequence]  # (train_idx, test_idx)

def _is_datetime_like(arr) -> bool:
    try:
        s = pd.to_datetime(np.array(arr))
        return not pd.isna(s).all()
    except Exception:
        return False

def _to_datetime_index(arr) -> pd.DatetimeIndex:
    return pd.to_datetime(np.array(arr)).tz_localize(None).normalize()

def extend_last_test_window(
    splits: List[Split],
    full_index: pd.Index,
    as_of: Optional[pd.Timestamp] = None,
) -> List[Split]:
    """
    Extend the LAST fold's TEST indices up to 'as_of' (or to the end of 'full_index' if as_of None).

    Requirements / behavior:
    - Works when split indices are **datetime-like** (recommended).
    - If the last test end already >= as_of, this is a no-op.
    - Does NOT change any TRAIN windows. It only extends the last TEST window forward.
    - Ensures the added dates exist in 'full_index' (avoids weekends/holidays).

    Args:
        splits: list of (train_idx, test_idx)
        full_index: the master trading calendar (DatetimeIndex) used to build splits
        as_of: optional cap date. default = full_index.max()

    Returns:
        new_splits: same list length, with last test_idx extended
    """
    if not splits:
        return splits

    if not isinstance(full_index, pd.DatetimeIndex):
        full_index = pd.to_datetime(full_index).tz_localize(None).normalize()

    cap = pd.to_datetime(as_of).tz_localize(None).normalize() if as_of is not None else full_index.max().normalize()

    # Pull the last fold
    train_idx_last, test_idx_last = splits[-1]

    # We only support datetime-like indices for safe extension
    if not _is_datetime_like(test_idx_last):
        raise TypeError(
            "extend_last_test_window expects datetime-like test indices on the last fold. "
            "Got non-datetime indices; please convert your split generator to use dates."
        )

    test_dt = _to_datetime_index(test_idx_last)
    last_end = test_dt.max()

    # No change needed
    if last_end >= cap:
        return splits

    # Build extension: take any dates in full_index strictly after last_end and <= cap
    mask = (full_index > last_end) & (full_index <= cap)
    ext = full_index[mask]

    if len(ext) == 0:
        return splits

    # Stitch: original test + extension (preserve original dtype as datetime)
    extended_test = pd.DatetimeIndex(test_dt.union(ext)).sort_values()

    new_splits = list(splits)
    new_splits[-1] = (train_idx_last, extended_test)
    return new_splits
