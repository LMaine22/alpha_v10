"""Purged Anchored Walk-Forward (PAWF) outer splits."""

from __future__ import annotations
from typing import List, Dict, Tuple
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
    embargo_cap_days: int = 21,
    regime_version: str = "R1",
) -> List[SplitSpec]:
    """Build Purged Anchored Walk-Forward outer splits.

    The routine walks the data forward in trading-day units (``test_window_days``)
    while anchoring the training window at the start of the series. Each split
    uses ``purge_days = label_horizon_days`` and an embargo equal to the feature
    lookback tail, but the final "tail" split dynamically relaxes those buffers
    just enough to keep the last trading week in-sample.

    Args:
        df: Master DataFrame with ``DatetimeIndex``.
        label_horizon_days: Forecast horizon (used for purge sizing).
        feature_lookback_tail: Maximum feature lookback (baseline embargo).
        min_train_months: Minimum anchored training span in months.
        test_window_days: Target test window in trading days.
        step_months: Calendar step for advancing test windows.
        regime_version: Regime model tag to stamp on each ``SplitSpec``.

    Returns:
        List of ``SplitSpec`` objects covering the entire data range.

    Raises:
        ValueError: If there is not enough history to satisfy the minimum
            training window.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    trading_days = df.index.unique().sort_values()
    if trading_days.empty:
        raise ValueError("No trading days available to build PAWF splits")

    data_start = trading_days[0]
    data_end = trading_days[-1]

    # Minimum training period (calendar months converted to first trading day after offset)
    min_train_period = pd.DateOffset(months=min_train_months)
    first_possible = data_start + min_train_period
    first_idx = trading_days.searchsorted(first_possible, side='left')
    if first_idx >= len(trading_days):
        raise ValueError(
            f"Insufficient data: need at least {min_train_months} months but have {(data_end - data_start).days / 30:.1f} months"
        )

    min_train_trading_days = int(np.floor(min_train_months * 21))
    min_test_trading_days = 5

    splits: List[SplitSpec] = []
    tail_used = False

    def _adjust_tail_buffers(
        purge_days: int,
        embargo_days: int,
        test_len: int,
    ) -> Tuple[int, int, Dict[str, Dict[str, object]]]:
        """Relax purge/embargo just enough to keep ≥min_test_trading_days test bars."""
        adjustments: Dict[str, Dict[str, object]] = {}
        max_buffer = max(0, test_len - min_test_trading_days)

        if embargo_days > max_buffer:
            new_embargo = int(max_buffer)
            adjustments["embargo_days"] = {"from": int(embargo_days), "to": new_embargo}
            embargo_days = new_embargo

        if purge_days > max_buffer and max_buffer >= label_horizon_days:
            new_purge = int(max_buffer)
            adjustments["purge_days"] = {
                "from": int(purge_days),
                "to": new_purge,
            }
            purge_days = new_purge

        return purge_days, embargo_days, adjustments

    def _make_split(test_start_idx: int, test_end_idx: int, *, auto_appended: bool = False) -> bool:
        nonlocal tail_used

        test_len = test_end_idx - test_start_idx + 1
        if test_len < min_test_trading_days:
            return False

        purge_days = int(max(0, label_horizon_days))
        purge_days = min(purge_days, test_start_idx)
        train_end_idx = test_start_idx - purge_days - 1
        while train_end_idx < 0 and purge_days > 0:
            purge_days -= 1
            train_end_idx = test_start_idx - purge_days - 1
        if train_end_idx < 0:
            train_end_idx = 0

        train_start_idx = 0
        cap = int(max(0, min(feature_lookback_tail, embargo_cap_days)))
        embargo_days = int(max(5, cap))
        if embargo_days >= test_len:
            embargo_days = max(0, test_len - 1)

        adjustments: Dict[str, Dict[str, int]] = {}
        is_tail = test_end_idx == len(trading_days) - 1
        if is_tail:
            tail_used = True
            orig_purge, orig_embargo = purge_days, embargo_days
            purge_days, embargo_days, adjustments = _adjust_tail_buffers(
                purge_days,
                embargo_days,
                test_len,
            )
            if adjustments.get("purge_days"):
                train_end_idx = max(0, test_start_idx - purge_days - 1)
            if adjustments and (orig_purge != purge_days or orig_embargo != embargo_days):
                print("[PAWF] Tail buffer adjustments ->", adjustments)

        train_span = train_end_idx - train_start_idx + 1
        if train_span < min_train_trading_days:
            return False
        if train_end_idx <= train_start_idx:
            return False

        notes: Dict[str, object] = {
            "test_trading_days": int(test_len),
            "tail_used": bool(is_tail),
        }
        if adjustments:
            notes["tail_buffer_adjustments"] = adjustments
        if auto_appended:
            notes["auto_appended"] = True

        spec = SplitSpec(
            outer_id=f"outer_{len(splits):03d}",
            split_version="PAWF_v1",
            train_start=trading_days[train_start_idx],
            train_end=trading_days[train_end_idx],
            test_start=trading_days[test_start_idx],
            test_end=trading_days[test_end_idx],
            purge_days=int(max(0, purge_days)),
            embargo_days=int(max(0, embargo_days)),
            label_horizon=label_horizon_days,
            feature_lookback_tail=feature_lookback_tail,
            regime_version=regime_version,
            event_class="normal",
            is_tail=is_tail,
            notes=notes,
        )

        splits.append(spec)
        return True

    current_start_idx = first_idx
    while current_start_idx < len(trading_days):
        test_start_idx = current_start_idx
        test_end_idx = min(test_start_idx + test_window_days - 1, len(trading_days) - 1)

        # For tail windows ensure we have at least the minimum test trading days
        if test_end_idx == len(trading_days) - 1 and (test_end_idx - test_start_idx + 1) < min_test_trading_days:
            test_start_idx = max(0, len(trading_days) - min_test_trading_days)
            test_end_idx = len(trading_days) - 1

        _make_split(test_start_idx, test_end_idx)

        if splits and splits[-1].test_end == data_end:
            break

        next_start_date = trading_days[test_start_idx] + pd.DateOffset(months=step_months)
        current_start_idx = trading_days.searchsorted(next_start_date, side='left')
        if current_start_idx <= test_start_idx:
            current_start_idx = test_start_idx + 1

    if not splits:
        raise ValueError(
            f"No valid splits generated. Check min_train_months={min_train_months} and available data span"
        )

    if splits[-1].test_end != data_end:
        tail_start_idx = max(0, len(trading_days) - test_window_days)
        tail_end_idx = len(trading_days) - 1
        if tail_end_idx - tail_start_idx + 1 < min_test_trading_days:
            tail_start_idx = max(0, len(trading_days) - min_test_trading_days)
        appended = _make_split(tail_start_idx, len(trading_days) - 1, auto_appended=True)
        if not appended:
            raise AssertionError(
                "Unable to append PAWF tail split that reaches the data end; insufficient history"
            )

    if splits[-1].test_end != data_end:
        raise AssertionError(
            f"PAWF final split does not reach data end ({splits[-1].test_end.date()} != {data_end.date()})"
        )

    tail_aligned = splits[-1].test_end == data_end
    summary = {
        "first_train_start": splits[0].train_start.date(),
        "last_test_end": splits[-1].test_end.date(),
        "n_outer": len(splits),
        "tail_aligned": tail_aligned,
    }
    print(f"[PAWF] Summary: {summary}")

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
            "regime_ver": spec.regime_version,
            "is_tail": getattr(spec, "is_tail", False),
            "tail_buffer_adjustments": (spec.notes or {}).get("tail_buffer_adjustments"),
            "auto_appended": bool((spec.notes or {}).get("auto_appended", False)),
        })

    return pd.DataFrame(rows)
