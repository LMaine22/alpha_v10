from __future__ import annotations
import numpy as np
import pandas as pd
import hashlib, joblib, os  # caching additions

"""Utility functions for calendar-safe label alignment and trigger handling.

These are NOT fallback metrics— they provide the canonical mapping of trigger timestamps
and horizon label end points onto the observed trading calendar (the index actually used
for pricing / returns). They deliberately avoid business-day arithmetic because that can
misalign around market holidays and special sessions.
"""

__all__ = [
    "index_after_n",
    "align_triggers_to_calendar",
    "label_pairs",
    "cached_label_pairs",
]

def index_after_n(trading_index: pd.DatetimeIndex, starts: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Return trading_index timestamps n steps after each start; NaT for out-of-range.

    Parameters
    ----------
    trading_index : DatetimeIndex
        Full ordered set of trading timestamps (must be unique & sorted).
    starts : DatetimeIndex
        Event/trigger timestamps (need not be aligned yet to trading_index).
    n : int
        Forward step count (e.g., forecast horizon in trading days).
    """
    if not isinstance(starts, pd.DatetimeIndex):
        starts = pd.DatetimeIndex(starts)
    if not isinstance(trading_index, pd.DatetimeIndex):
        trading_index = pd.DatetimeIndex(trading_index)
    trading_index = trading_index.sort_values().unique()
    pos = np.searchsorted(trading_index.values, starts.values, side="left")
    label_pos = pos + int(n)
    ok = (label_pos >= 0) & (label_pos < len(trading_index))
    out = pd.DatetimeIndex([pd.NaT] * len(label_pos))
    out.values[ok] = trading_index.values[label_pos[ok]]
    return out

def align_triggers_to_calendar(trading_index: pd.DatetimeIndex, triggers: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Snap triggers to the *next or same* trading day; drop those beyond calendar end.

    Uses a left-search (lower bound) so if a trigger is exactly on a trading timestamp it
    keeps that day, otherwise advances to the next available trading day. Out-of-range
    triggers are dropped.
    """
    if not isinstance(trading_index, pd.DatetimeIndex):
        trading_index = pd.DatetimeIndex(trading_index)
    trading_index = trading_index.sort_values().unique()
    if not isinstance(triggers, pd.DatetimeIndex):
        triggers = pd.DatetimeIndex(triggers)
    triggers = triggers.sort_values().unique()
    pos = np.searchsorted(trading_index.values, triggers.values, side="left")
    ok = (pos >= 0) & (pos < len(trading_index))
    out = pd.DatetimeIndex([pd.NaT] * len(pos))
    out.values[ok] = trading_index.values[pos[ok]]
    return out.dropna()

def label_pairs(trading_index: pd.DatetimeIndex, triggers: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Return a DataFrame with aligned (t_start, t_end) pairs on the trading calendar.

    Steps:
    1. Align raw triggers onto calendar (snap forward if needed).
    2. Compute label end at +h steps (calendar step, not timedelta) using index_after_n.
    3. Drop rows with out-of-range ends or non-increasing intervals.
    """
    t0 = align_triggers_to_calendar(trading_index, triggers)
    t1 = index_after_n(trading_index, t0, horizon)
    df = pd.DataFrame({"t0": t0, "t1": t1}).dropna()
    return df[df["t1"] > df["t0"]].reset_index(drop=True)


# -------------------------------------------------------------
# Persistent caching layer for label pair generation
# -------------------------------------------------------------
_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache", "label_pairs"))
os.makedirs(_CACHE_DIR, exist_ok=True)

def _hash_dtindex(idx: pd.DatetimeIndex) -> str:
    idx = pd.DatetimeIndex(idx)
    if idx.empty:
        return "empty"  # stable token
    arr = idx.view("i8")  # ns since epoch
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]

def _hash_triggers(triggers: pd.DatetimeIndex) -> str:
    triggers = pd.DatetimeIndex(triggers)
    if triggers.empty:
        return "empty"
    arr = triggers.view("i8")
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]

def cached_label_pairs(trading_index: pd.DatetimeIndex,
                       triggers: pd.DatetimeIndex,
                       horizon: int) -> pd.DataFrame:
    """Persistent on-disk cache of label_pairs().

    Keyed by SHA1 of (trading_index, triggers, horizon). Safe for reuse across
    candidates that share identical trigger sets & calendar.
    """
    idx_hash = _hash_dtindex(trading_index)
    trg_hash = _hash_triggers(triggers)
    key = f"{idx_hash}_{trg_hash}_{int(horizon)}.pkl"
    path = os.path.join(_CACHE_DIR, key)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass  # fall through to recompute
    df = label_pairs(trading_index, triggers, horizon)
    try:
        joblib.dump(df, path, compress=3)
    except Exception:
        pass
    return df
