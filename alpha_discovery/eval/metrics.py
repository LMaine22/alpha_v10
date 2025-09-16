# alpha_discovery/eval/metrics.py
"""
Core performance metrics and helpers (robust, MAR-aware).
This version fixes the pathological "Sortino = 0.0" issue by:
  1) Using a MAR-aware Sortino with downside defined as r < MAR (zeros don't pollute downside).
  2) Returning a large finite cap when there's no downside (rather than 0.0).
  3) Providing a small-sample fallback for the bootstrap that returns the point estimate.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from ..config import settings

TRADING_DAYS_PER_YEAR = 252.0
_EPS = 1e-12
_SORTINO_CAP_IF_NO_DOWNSIDE = 100.0  # large finite value so GA can rank cleanly


# =========================
# Sanitizers & helpers
# =========================
def _sanitize_series(s: Optional[pd.Series]) -> pd.Series:
    """Ensure float dtype and drop NaNs/Infs."""
    if s is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def winsorize(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clip a series to [lower_q, upper_q] quantiles (no-op on empty)."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    lq = float(max(min(lower_q, 0.49), 0.0))
    uq = float(max(min(upper_q, 1.0), lq + 1e-9))
    lower = series.quantile(lq)
    upper = series.quantile(uq)
    return series.clip(lower=lower, upper=upper)


def _daily_required_return_from_settings() -> float:
    """
    Annual RF from settings → daily MAR used by Sortino.
    This anchors downside to economic opportunity cost.
    """
    try:
        rf_annual = float(getattr(settings.options, "constant_r", 0.0))
    except Exception:
        rf_annual = 0.0
    return float(rf_annual) / TRADING_DAYS_PER_YEAR


# =========================
# Expectancy (per-trade pnl_pct)
# =========================
def calculate_expectancy(ledger: Optional[pd.DataFrame]) -> float:
    """
    Expectancy computed from the trade ledger's per-trade returns.
    Falls back to 0.0 if the ledger is empty or column missing.
    """
    if ledger is None or len(ledger) == 0:
        return 0.0
    col_candidates = ("pnl_pct", "pnl%", "ret_pct", "return_pct")
    cols_lower = {c.lower(): c for c in ledger.columns}
    col = None
    for c in col_candidates:
        if c in ledger.columns:
            col = c
            break
        if c.lower() in cols_lower:
            col = cols_lower[c.lower()]
            break
    if col is None:
        return 0.0
    s = pd.to_numeric(ledger[col], errors="coerce").dropna().astype(float)
    return float(s.mean()) if len(s) else 0.0


# =========================
# Sortino (point estimate)
# =========================
def calculate_sortino_ratio(series: pd.Series, required_return: Optional[float] = None) -> float:
    """
    Sortino ratio computed against MAR (default: daily risk-free from settings).
    - Downside = {r : r < MAR} (strictly <, zeros won't create fake downside)
    - If there is *no downside*, returns a large finite cap (not 0.0)
    """
    s = _sanitize_series(series)
    if len(s) < 2:
        return 0.0

    mar = _daily_required_return_from_settings() if required_return is None else float(required_return)
    excess = s - mar
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return _SORTINO_CAP_IF_NO_DOWNSIDE

    denom = float(downside.std(ddof=1))
    if not np.isfinite(denom) or denom < _EPS:
        # Degenerate downside dispersion
        return _SORTINO_CAP_IF_NO_DOWNSIDE if excess.mean() > 0.0 else 0.0

    sortino = float(excess.mean() / max(denom, _EPS))
    # Annualize
    sortino *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return sortino


# =========================
# Sortino (block bootstrap quantiles)
# =========================
def block_bootstrap_sortino(
    returns_series: pd.Series,
    block_size: int = 5,
    num_iterations: int = 500,
    required_return: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Sortino quantiles via overlapping-block bootstrap.
    Robust to small-sample pathologies (no downside / all-∞ results).
    """
    rs = _sanitize_series(returns_series)
    n = len(rs)
    mar = _daily_required_return_from_settings() if required_return is None else float(required_return)

    # Small-sample fallback: return the point estimate
    if n < max(block_size * 2, 10):
        pe = calculate_sortino_ratio(rs, required_return=mar)
        return {"sortino_median": float(pe), "sortino_lb": float(pe), "sortino_ub": float(pe)}

    b = int(block_size)
    if b <= 0 or n < b:
        pe = calculate_sortino_ratio(rs, required_return=mar)
        return {"sortino_median": float(pe), "sortino_lb": float(pe), "sortino_ub": float(pe)}

    x = rs.to_numpy(copy=False)
    n_blocks = n - b + 1
    if n_blocks <= 0:
        pe = calculate_sortino_ratio(rs, required_return=mar)
        return {"sortino_median": float(pe), "sortino_lb": float(pe), "sortino_ub": float(pe)}

    stride = x.strides[0]
    blocks = np.lib.stride_tricks.as_strided(x, shape=(n_blocks, b), strides=(stride, stride))

    k = max(n // b, 1)  # blocks per bootstrap sample
    sortino_vals = np.empty(int(num_iterations), dtype=float)

    rng = np.random.default_rng()
    for i in range(int(num_iterations)):
        idx = rng.integers(0, n_blocks, size=k, endpoint=False)
        sample = blocks[idx].ravel()[:n]
        val = calculate_sortino_ratio(pd.Series(sample), required_return=mar)
        # Map +/-inf to large finite sentinel so quantiles are defined
        if not np.isfinite(val):
            val = _SORTINO_CAP_IF_NO_DOWNSIDE if val > 0 else 0.0
        sortino_vals[i] = float(val)

    # We now have a finite array by construction
    return {
        "sortino_median": float(np.nanmedian(sortino_vals)),
        "sortino_lb": float(np.nanpercentile(sortino_vals, 5)),
        "sortino_ub": float(np.nanpercentile(sortino_vals, 95)),
    }


# =========================
# Sharpe (block bootstrap)
# =========================
def _sharpe_point(series: pd.Series) -> float:
    s = _sanitize_series(series)
    if len(s) < 2:
        return 0.0
    mu = float(s.mean()) * TRADING_DAYS_PER_YEAR
    sd = float(s.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if not np.isfinite(sd) or sd < _EPS:
        return 0.0
    return float(mu / sd)


def block_bootstrap_sharpe(
    returns_series: pd.Series,
    block_size: int = 5,
    num_iterations: int = 500,
    trading_days_per_year: float = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """
    Overlapping-block bootstrap Sharpe quantiles.
    """
    rs = _sanitize_series(returns_series)
    n = len(rs)
    if n < max(block_size * 2, 10):
        pe = _sharpe_point(rs)
        return {"sharpe_median": float(pe), "sharpe_lb": float(pe), "sharpe_ub": float(pe)}

    b = int(block_size)
    if b <= 0 or n < b:
        pe = _sharpe_point(rs)
        return {"sharpe_median": float(pe), "sharpe_lb": float(pe), "sharpe_ub": float(pe)}

    x = rs.to_numpy(copy=False)
    n_blocks = n - b + 1
    if n_blocks <= 0:
        pe = _sharpe_point(rs)
        return {"sharpe_median": float(pe), "sharpe_lb": float(pe), "sharpe_ub": float(pe)}

    stride = x.strides[0]
    blocks = np.lib.stride_tricks.as_strided(x, shape=(n_blocks, b), strides=(stride, stride))

    k = max(n // b, 1)
    vals = np.empty(int(num_iterations), dtype=float)
    rng = np.random.default_rng()
    for i in range(int(num_iterations)):
        idx = rng.integers(0, n_blocks, size=k, endpoint=False)
        sample = blocks[idx].ravel()[:n]
        vals[i] = _sharpe_point(pd.Series(sample))

    return {
        "sharpe_median": float(np.nanmedian(vals)),
        "sharpe_lb": float(np.nanpercentile(vals, 5)),
        "sharpe_ub": float(np.nanpercentile(vals, 95)),
    }


# =========================
# Omega & Max Drawdown
# =========================
def calculate_omega_ratio(series: pd.Series, threshold: float = 0.0) -> float:
    """
    Omega ratio: integral of gains above threshold divided by losses below.
    Computed via discrete sums.
    """
    s = _sanitize_series(series)
    if s.empty:
        return 0.0
    gains = np.clip(s - threshold, 0.0, None).sum()
    losses = np.clip(threshold - s, 0.0, None).sum()
    if losses < _EPS:
        return float(np.inf) if gains > 0 else 0.0
    return float(gains / max(losses, _EPS))


def calculate_max_drawdown(series: pd.Series) -> float:
    """
    Max drawdown computed on a simple NAV from the return series.
    """
    s = _sanitize_series(series)
    if s.empty:
        return 0.0
    nav = (1.0 + s).cumprod()
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    mdd = float(dd.min())
    return mdd if np.isfinite(mdd) else 0.0


# =========================
# Main Portfolio Metrics Calculator
# =========================
def calculate_portfolio_metrics(
    daily_returns: pd.Series,
    portfolio_ledger: Optional[pd.DataFrame] = None,
    do_winsorize: bool = True,
) -> Dict[str, float]:
    """
    Compute the core metric bundle used by GA and reporting.
    - Applies light winsorization for robustness (if enough samples).
    - Returns Sortino (median/lb/ub), Sharpe (median/lb/ub), Omega, MaxDD, Expectancy, Support.
    """
    s = _sanitize_series(daily_returns)
    if s.empty:
        return {
            "support": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "omega_ratio": 0.0,
            "sharpe_median": 0.0, "sharpe_lb": 0.0, "sharpe_ub": 0.0,
            "sortino_median": 0.0, "sortino_lb": 0.0, "sortino_ub": 0.0,
        }

    if do_winsorize and len(s) > 20:
        alpha = float(getattr(settings.reporting, "trimmed_alpha", 0.05))
        s = winsorize(s, lower_q=alpha, upper_q=1.0 - alpha)

    # --- MAR-aware Sortino (with bootstrap) ---
    mar = _daily_required_return_from_settings()
    sortino_stats = block_bootstrap_sortino(s, block_size=5, num_iterations=500, required_return=mar)

    # --- Other metrics ---
    expectancy = calculate_expectancy(portfolio_ledger) if portfolio_ledger is not None else 0.0
    support = float(len(s))
    sharpe_stats = block_bootstrap_sharpe(s)
    omega = calculate_omega_ratio(s)
    max_dd = calculate_max_drawdown(s)

    out = {
        "support": support,
        "expectancy": float(expectancy),
        "max_drawdown": float(max_dd),
        "omega_ratio": float(omega),
        **{k: float(v) for k, v in sharpe_stats.items()},
        **{k: float(v) for k, v in sortino_stats.items()},
    }

    # Replace non-finite with 0.0 to keep downstream vectorized math safe
    return {k: (0.0 if (v is None or not np.isfinite(v)) else float(v)) for k, v in out.items()}
