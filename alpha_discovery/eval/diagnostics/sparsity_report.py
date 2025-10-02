"""Diagnostics: Make sparsity + fold skipping causes auditable.

Purpose
-------
Provide a transparent per-fold report so we can answer:
  * How many test days and test triggers did each CPCV++ fold contain?
  * How many label pairs (calendar-aligned) were realized for a given horizon?
  * How much training data was discarded by interval purging + embargo?

This directly addresses the symptom of previously "mysterious" skipped folds.
If a fold is dropped upstream it will be obvious here because `n_test_triggers`
(or `final_train_count`) will be zero / extremely small.

Usage Example
-------------
from alpha_discovery.core.splits import CombinatorialPurgedCV
from alpha_discovery.eval.utils import label_pairs
from alpha_discovery.eval.diagnostics.sparsity_report import sparsity_report

cpcv = CombinatorialPurgedCV(...)
splits = cpcv.split(discovery_index, forecast_horizon, trading_index=discovery_index, trigger_times=trigger_times)
rep = sparsity_report(
    trading_index=discovery_index,
    trigger_times=trigger_times,
    splits=splits,
    horizon=forecast_horizon,
    label_pairs_fn=label_pairs,
    full_index=discovery_index,
    embargo_days=int(round(cpcv.embargo_pct * len(discovery_index) / cpcv.n_splits))
)
print(rep)

Columns
-------
fold                1-based fold id
test_start/end      boundaries of the adaptive test window
n_test_days         number of calendar trading days in test window
n_test_triggers     trigger count inside test window
n_label_pairs       realized (t0,t1) label intervals for this horizon (calendar-safe)
candidate_train_cnt raw train candidates before purge (all X < test_start)
purged_overlaps     number of candidates removed due to label overlap with test window
embargo_removed     number removed by embargo trimming after purge
final_train_count   remaining training timestamps supplied to the model metrics
purged_pct          purged_overlaps / candidate_train_cnt (NaN if candidate_train_cnt=0)

Notes
-----
* We recompute the purge counts locally to expose the magnitude of leakage prevention.
* If adaptive expansion occurred you will see `n_test_days` > base slice size.
* A fold with n_test_triggers < configured min_test_triggers was already expanded; if still low
  you will likely see it excluded downstream (fail-closed) and the report justifies why.
"""
from __future__ import annotations
from typing import Callable, Iterable, List, Tuple, Optional
import pandas as pd
import numpy as np

from ...core.splits import purge_train_for_test_window, label_end_on_index

# Type alias for clarity
tSplit = Tuple[pd.DatetimeIndex, pd.DatetimeIndex]


def sparsity_report(
    trading_index: pd.DatetimeIndex,
    trigger_times: pd.DatetimeIndex,
    splits: List[tSplit],
    horizon: int,
    label_pairs_fn: Callable[[pd.DatetimeIndex, pd.DatetimeIndex, Iterable[int]], pd.DataFrame],
    *,
    full_index: Optional[pd.DatetimeIndex] = None,
    embargo_days: Optional[int] = None,
) -> pd.DataFrame:
    """Return a per-fold sparsity / purge diagnostics DataFrame.

    Parameters
    ----------
    trading_index : DatetimeIndex
        Master ordered trading calendar used for label alignment.
    trigger_times : DatetimeIndex
        All (global) trigger timestamps for the candidate.
    splits : list[(train_idx, test_idx)]
        CPCV++ (or other) splits already produced (train & test indices).
    horizon : int
        Forecast horizon in trading steps (not calendar days) for label end mapping.
    label_pairs_fn : callable
        Function that returns a DataFrame of label intervals given (trading_index, triggers, horizons_list).
    full_index : DatetimeIndex, optional
        Complete index used to construct splits. If provided we can recompute candidate
        train set BEFORE purge to quantify purge effect. Defaults to trading_index.
    embargo_days : int, optional
        If provided, we approximate the embargo removal count after purging.
    """
    if not isinstance(trading_index, pd.DatetimeIndex):
        raise TypeError("trading_index must be a DatetimeIndex")
    if not isinstance(trigger_times, pd.DatetimeIndex):
        trigger_times = pd.DatetimeIndex(trigger_times)

    full_index = pd.DatetimeIndex(full_index if full_index is not None else trading_index)
    rows = []

    for i, (tr, te) in enumerate(splits, 1):
        if len(te) == 0:
            rows.append({
                "fold": i,
                "test_start": pd.NaT,
                "test_end": pd.NaT,
                "n_test_days": 0,
                "n_test_triggers": 0,
                "n_label_pairs": 0,
                "candidate_train_cnt": np.nan,
                "purged_overlaps": np.nan,
                "embargo_removed": np.nan,
                "final_train_count": len(tr),
                "purged_pct": np.nan,
            })
            continue

        te_start, te_end = te.min(), te.max()
        test_triggers = trigger_times[(trigger_times >= te_start) & (trigger_times <= te_end)]

        # Realized label pairs (calendar-safe) for this horizon
        try:
            pairs_df = label_pairs_fn(trading_index, test_triggers, [horizon])
            n_pairs = len(pairs_df)
        except Exception:
            n_pairs = 0

        # Reconstruct candidate train BEFORE purge: all timestamps strictly before test_start
        candidate_train = full_index[full_index < te_start]
        candidate_train_cnt = len(candidate_train)

        # Recompute purge locally to measure overlaps removed
        purged_train = purge_train_for_test_window(candidate_train, te_start, te_end, trading_index, horizon)
        purged_overlaps = candidate_train_cnt - len(purged_train)

        embargo_removed = 0
        final_train_count = len(tr)
        if embargo_days is not None and embargo_days > 0 and len(purged_train) > 0:
            # Approximate embargo application mirroring CPCV logic:
            embargo_cut = te_end + pd.Timedelta(days=int(embargo_days))
            # In original CPCV we ensure train labels end before embargo_cut - horizon
            # We approximate by filtering purged_train similarly then count delta.
            label_ends = label_end_on_index(trading_index, purged_train, horizon)
            keep_mask = (label_ends.notna()) & (label_ends < (embargo_cut - pd.Timedelta(days=horizon)))
            after_embargo = purged_train[keep_mask]
            embargo_removed = len(purged_train) - len(after_embargo)
            # final_train_count should match provided train idx length; we keep provided authoritative value.

        purged_pct = (purged_overlaps / candidate_train_cnt) if candidate_train_cnt > 0 else np.nan

        rows.append({
            "fold": i,
            "test_start": te_start,
            "test_end": te_end,
            "n_test_days": len(te),
            "n_test_triggers": len(test_triggers),
            "n_label_pairs": n_pairs,
            "candidate_train_cnt": candidate_train_cnt,
            "purged_overlaps": purged_overlaps,
            "embargo_removed": embargo_removed,
            "final_train_count": final_train_count,
            "purged_pct": purged_pct,
        })

    return pd.DataFrame(rows)

__all__ = ["sparsity_report"]
