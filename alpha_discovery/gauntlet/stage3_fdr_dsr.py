# alpha_discovery/gauntlet/stage3_fdr_dsr.py
from __future__ import annotations

from typing import Dict, Any, Mapping, Optional, List

import numpy as np
import pandas as pd


# -------------------------
# Deflated Sharpe (utility)
# -------------------------

def _psr(sr: float, T: int, skew: float, kurt: float) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado) as a helper.
    Returns the probability that SR > 0 given sample SR / higher moments.
    """
    if T <= 1:
        return 0.0
    # Lo's SR variance adjustment (skew, kurtosis)
    # sigma_sr^2 ≈ (1 + 0.5*sr^2*(kurt - 1) - sr*skew) / (T - 1)
    var_sr = (1.0 + 0.5 * (sr ** 2) * (kurt - 1.0) - sr * skew) / max(1, (T - 1))
    std_sr = float(np.sqrt(max(1e-12, var_sr)))
    z = sr / std_sr
    # one-sided prob SR > 0
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def dsr(sr: float, skew: float, kurt: float, T: int, N_eff: float = 1.0) -> float:
    """
    Deflated Sharpe Ratio (approx):
      Convert SR to probability via PSR, then adjust for multiple testing
      by comparing vs. 1 - alpha*, where alpha* = 1 - (1 - alpha)^(1/N_eff).
    Returns a conservative score in [0, 1]; higher = better.
    """
    psr_val = _psr(sr=sr, T=T, skew=skew, kurt=kurt)
    # Interpret PSR as 1 - alpha (alpha = false positive rate)
    alpha = max(0.0, min(1.0, 1.0 - psr_val))
    # Deflate alpha via Sidak approx using N_eff
    alpha_star = 1.0 - (1.0 - alpha) ** (1.0 / max(1.0, float(N_eff)))
    # Return deflated "confidence"
    dsr_val = 1.0 - alpha_star
    return float(max(0.0, min(1.0, dsr_val)))


# --------------------------------------------------------
# Stage-3 (per-setup) on a provided ledger (NEW helper)
# --------------------------------------------------------

def stage3_on_ledger(
    fold_ledger: pd.DataFrame,
    base_capital: float,
    stage2_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Stage-3 (BH-FDR + DSR) for a single-setup ledger.
    With a single setup, BH reduces to p <= q and N_eff = 1.
    """
    if stage2_df is None or stage2_df.empty:
        return pd.DataFrame(columns=["setup_id", "rank", "fdr_pass", "dsr", "N_eff"])

    q = float(cfg.get("fdr_q", 0.10))
    rec = stage2_df.iloc[0]
    setup_id = str(rec.get("setup_id"))
    rank = rec.get("rank")
    pval = float(rec.get("pvalue_sharpe_gt0", 1.0))
    fdr_pass = bool(pval <= q)

    dsr_val = None
    if fdr_pass:
        from ..eval.nav import nav_daily_returns_from_ledger as daily_returns_from_ledger
        from ..eval.nav import sharpe as sharpe_ratio

        ret = daily_returns_from_ledger(fold_ledger, base_capital=base_capital)
        T = len(ret)
        if T >= 20:
            sr = float(sharpe_ratio(ret))
            skew = float(ret.skew())
            kurt = float(ret.kurtosis() + 3.0)
            dsr_val = dsr(sr=sr, skew=skew, kurt=kurt, T=T, N_eff=1.0)
        else:
            dsr_val = 0.0

    return pd.DataFrame(
        [
            {
                "setup_id": setup_id,
                "rank": rank,
                "fdr_pass": fdr_pass,
                "dsr": float(dsr_val) if dsr_val is not None else None,
                "N_eff": 1.0,
            }
        ]
    )


# ------------------------------------------------------------
# Cohort-wide Stage-3 on OOS (shared N_eff)  (NEW and RECOMMENDED)
# ------------------------------------------------------------

def _align_returns_map(returns_map: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Align per-setup daily-return series into a single DataFrame (index aligned)."""
    if not returns_map:
        return pd.DataFrame()
    df = pd.DataFrame(returns_map).sort_index()
    df = df.dropna(how="all")
    return df


def _effective_trials_from_corr(C: np.ndarray) -> float:
    """
    Eigenvalue-based effective number of tests:
      N_eff = (sum λ)^2 / sum(λ^2), λ eigenvalues of correlation matrix.
    """
    C = (C + C.T) / 2.0
    eigvals = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals ** 2))
    if s2 <= 0.0:
        return float(C.shape[0])
    return max(1.0, (s1 * s1) / s2)


def _bh_fdr(pvals: np.ndarray, q: float) -> np.ndarray:
    """
    Benjamini–Hochberg mask (True = pass) at level q on vector of p-values.
    """
    m = int(len(pvals))
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    k = np.max(np.where(ranked <= thresholds)[0]) if np.any(ranked <= thresholds) else -1
    passed = np.zeros(m, dtype=bool)
    if k >= 0:
        passed[order[: k + 1]] = True
    return passed


def stage3_cohort_oos(
    stage2_df_all: pd.DataFrame,
    returns_map: Mapping[str, pd.Series],
    base_capital: float,  # kept for symmetry / future extensions
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Cohort-wide BH-FDR + DSR on OOS p-values, with shared N_eff derived from the
    correlation structure of all OOS daily-return series.

    Inputs:
      - stage2_df_all: rows [setup_id, pvalue_sharpe_gt0, T, ...]
      - returns_map: {setup_id -> daily_returns Series (OOS)}
    """
    if stage2_df_all is None or stage2_df_all.empty or not returns_map:
        return pd.DataFrame(columns=["setup_id", "fdr_pass", "dsr", "N_eff"])

    q = float(cfg.get("fdr_q", 0.10))

    # Align returns and compute correlation
    aligned = _align_returns_map(returns_map)
    # Keep only series with sufficient length
    valid_cols: List[str] = [c for c in aligned.columns if aligned[c].notna().sum() >= 20]
    aligned = aligned[valid_cols]
    if aligned.shape[1] == 0:
        return pd.DataFrame(columns=["setup_id", "fdr_pass", "dsr", "N_eff"])

    C = aligned.corr().fillna(0.0).values
    N_eff = _effective_trials_from_corr(C)

    # BH-FDR across the cohort's p-values
    s2 = stage2_df_all.set_index("setup_id").reindex(valid_cols)
    pvals = s2["pvalue_sharpe_gt0"].astype(float).values
    mask = _bh_fdr(pvals, q=q)

    # Compute DSR for passers
    from ..eval.nav import sharpe as sharpe_ratio  # reuse your SR calc

    out_rows: List[Dict[str, Any]] = []
    for setup_id, passed in zip(valid_cols, mask):
        dsr_val = None
        if bool(passed):
            r = aligned[setup_id].dropna()
            if len(r) >= 20:
                sr = float(sharpe_ratio(r))
                skew = float(r.skew())
                kurt = float(r.kurtosis() + 3.0)
                dsr_val = dsr(sr=sr, skew=skew, kurt=kurt, T=len(r), N_eff=float(N_eff))
            else:
                dsr_val = 0.0

        out_rows.append(
            {
                "setup_id": str(setup_id),
                "fdr_pass": bool(passed),
                "dsr": float(dsr_val) if dsr_val is not None else None,
                "N_eff": float(N_eff),
            }
        )

    return pd.DataFrame(out_rows)
