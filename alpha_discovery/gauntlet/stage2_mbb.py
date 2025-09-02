# alpha_discovery/gauntlet/stage2_mbb.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    from ..config import Settings
except Exception:
    class Settings:  # type: ignore
        pass

# We rely on your eval utilities for NAV and Sharpe
from ..eval.nav import nav_daily_returns_from_ledger, sharpe


# ---------------------------
# Utilities for MBB on Sharpe
# ---------------------------

def estimate_block_len(
    T: int,
    method: str = "auto",
    lmin: int = 5,
    lmax: int = 50,
    series: Optional[pd.Series] = None,
) -> int:
    """
    Estimate moving-block bootstrap block length.
    Default 'auto' uses sqrt(T), clamped to [lmin, lmax].
    """
    if T <= 0:
        return max(1, lmin)
    if method == "auto":
        L = int(round(np.sqrt(T)))
    else:
        # Fallback to sqrt(T)
        L = int(round(np.sqrt(T)))
    return int(max(lmin, min(lmax, L)))


def _mbb_resample(series: np.ndarray, block_len: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving-block bootstrap: sample overlapping blocks of length L
    until we build a resample of length 'size'.
    """
    n = len(series)
    if n == 0:
        return series
    if block_len <= 0:
        block_len = 1
    # All possible starting indices for overlapping blocks
    starts = np.arange(0, max(1, n - block_len + 1))
    out = []
    total = 0
    while total < size:
        s = int(rng.integers(0, len(starts)))
        start = starts[s]
        blk = series[start : start + block_len]
        out.append(blk)
        total += len(blk)
    res = np.concatenate(out, axis=0)[:size]
    return res


def pvalue_sharpe_gt0_via_mbb(
    returns: pd.Series,
    B: int = 500,
    rng: Optional[np.random.Generator] = None,
    block_len: int = 10,
) -> float:
    """
    One-sided p-value for H0: Sharpe <= 0 vs H1: Sharpe > 0
    via moving-block bootstrap on daily returns.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    r = returns.dropna().astype(float).values
    n = len(r)
    if n < 5:
        return 1.0

    # Observed Sharpe
    r_obs = pd.Series(r)
    sr_obs = float(sharpe(r_obs))

    # Bootstrap Sharpe distribution under the empirical null
    # (We use centered returns to avoid trivial drift driving p-values)
    r_centered = r - float(np.mean(r))
    sr_boot = np.empty(B, dtype=float)
    for b in range(B):
        res = _mbb_resample(r_centered, block_len=block_len, size=n, rng=rng)
        sr_boot[b] = float(sharpe(pd.Series(res)))

    # P(Sharpe_boot >= Sharpe_obs) under null of no edge
    # For H0: Sharpe <= 0, a conservative p-value is share of boot SR >= sr_obs
    p = float(np.mean(sr_boot >= sr_obs))
    return max(0.0, min(1.0, p))


# -----------------------------------------
# Stage-2: fold-based API (kept for compat)
# -----------------------------------------

def run_stage2_mbb(
    run_dir: str,
    fold_num: int,
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Legacy fold-based Stage-2 that loads the fold ledger from disk.
    Kept for compatibility with any existing callers. New OOS flow should
    use run_stage2_mbb_on_ledger(...) instead.
    """
    cfg = dict(config or {})
    B = int(cfg.get("mbb_B", 500))
    block_method = str(cfg.get("block_len_method", "auto"))
    lmin = int(cfg.get("block_len_min", 5))
    lmax = int(cfg.get("block_len_max", 50))
    seed = int(cfg.get("seed", 42))

    if stage1_df is None or stage1_df.empty:
        return pd.DataFrame(columns=["setup_id", "rank", "T", "sr_train", "mbb_block_len", "pvalue_sharpe_gt0"])

    setup_id = stage1_df["setup_id"].iloc[0]
    rank = stage1_df.get("rank", pd.Series([None])).iloc[0]

    # Try a couple of likely ledger paths
    cand_paths = [
        os.path.join(run_dir, f"fold_{fold_num}", "pareto_ledger.csv"),
        os.path.join(run_dir, f"fold_{fold_num}", "front", "pareto_ledger.csv"),
    ]
    fold_ledger = None
    for p in cand_paths:
        if os.path.exists(p):
            try:
                fold_ledger = pd.read_csv(p)
                break
            except Exception:
                pass
    if fold_ledger is None or not isinstance(fold_ledger, pd.DataFrame) or fold_ledger.empty:
        return pd.DataFrame([{
            "setup_id": setup_id,
            "rank": rank,
            "T": 0,
            "sr_train": 0.0,
            "mbb_block_len": None,
            "pvalue_sharpe_gt0": 1.0,
        }])

    # Compute returns for this setup only if the ledger contains multiple
    try:
        base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_cap = 100_000.0

    ret = nav_daily_returns_from_ledger(fold_ledger, base_capital=base_cap)
    T = int(len(ret))
    if T < 10:
        return pd.DataFrame([{
            "setup_id": setup_id,
            "rank": rank,
            "T": T,
            "sr_train": float(sharpe(ret)),
            "mbb_block_len": None,
            "pvalue_sharpe_gt0": 1.0,
        }])

    L = estimate_block_len(T, method=block_method, lmin=lmin, lmax=lmax, series=ret)
    p = pvalue_sharpe_gt0_via_mbb(ret, B=B, rng=np.random.default_rng(int(seed)), block_len=int(L))
    sr_obs = sharpe(ret)

    return pd.DataFrame([{
        "setup_id": setup_id,
        "rank": rank,
        "T": T,
        "sr_train": float(sr_obs),
        "mbb_block_len": int(L),
        "pvalue_sharpe_gt0": float(p),
    }])


# -----------------------------------------------------
# Stage-2 (NEW): operate directly on a given OOS ledger
# -----------------------------------------------------

def run_stage2_mbb_on_ledger(
    fold_ledger: pd.DataFrame,
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Stage-2 for an arbitrary ledger (not tied to a run_dir/fold).
    Expects stage1_df to contain a single row for the setup to test.
    """
    cfg = dict(config or {})
    B = int(cfg.get("mbb_B", 500))
    block_method = str(cfg.get("block_len_method", "auto"))
    lmin = int(cfg.get("block_len_min", 5))
    lmax = int(cfg.get("block_len_max", 50))
    seed = int(cfg.get("seed", 42))

    if stage1_df is None or stage1_df.empty:
        return pd.DataFrame(columns=["setup_id", "rank", "T", "sr_train", "mbb_block_len", "pvalue_sharpe_gt0"])

    setup_id = str(stage1_df["setup_id"].iloc[0])
    rank = stage1_df.get("rank", pd.Series([None])).iloc[0]

    try:
        base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_cap = 100_000.0

    ret = nav_daily_returns_from_ledger(fold_ledger, base_capital=base_cap)
    T = int(len(ret))
    if T < 10:
        return pd.DataFrame([{
            "setup_id": setup_id,
            "rank": rank,
            "T": T,
            "sr_train": float(sharpe(ret)),
            "mbb_block_len": None,
            "pvalue_sharpe_gt0": 1.0,
        }])

    L = estimate_block_len(T, method=block_method, lmin=lmin, lmax=lmax, series=ret)
    p = pvalue_sharpe_gt0_via_mbb(ret, B=B, rng=np.random.default_rng(int(seed)), block_len=int(L))
    sr_obs = sharpe(ret)

    return pd.DataFrame([{
        "setup_id": setup_id,
        "rank": rank,
        "T": T,
        "sr_train": float(sr_obs),
        "mbb_block_len": int(L),
        "pvalue_sharpe_gt0": float(p),
    }])
