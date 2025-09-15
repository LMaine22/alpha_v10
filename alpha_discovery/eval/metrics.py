# alpha_discovery/eval/metrics.py
"""
Core performance metrics and helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from ..config import settings

TRADING_DAYS_PER_YEAR = 252.0


def _sanitize_series(s: Optional[pd.Series]) -> pd.Series:
    """Ensure float dtype and drop NaNs/Infs."""
    if s is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s.astype(float)
    return s


# =========================
# Robust helpers
# =========================

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
# Expectancy
# =========================

def calculate_expectancy(trade_ledger: pd.DataFrame) -> float:
    """
    Per-trade expectancy (dollars): E = p_win*avg_win + (1-p_win)*avg_loss
    """
    if trade_ledger is None or trade_ledger.empty or 'pnl_dollars' not in trade_ledger.columns:
        return 0.0

    pnl = pd.to_numeric(trade_ledger['pnl_dollars'], errors='coerce').dropna()
    if pnl.empty:
        return 0.0

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    n_trades = n_wins + n_losses
    if n_trades == 0:
        return 0.0

    win_rate = n_wins / n_trades
    loss_rate = n_losses / n_trades

    avg_win = wins.mean() if n_wins > 0 else 0.0
    avg_loss = losses.mean() if n_losses > 0 else 0.0  # negative or zero

    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)
    return float(expectancy) if np.isfinite(expectancy) else 0.0


# =========================
# Sortino Ratio (MAR-aware)
# =========================

def calculate_sortino_ratio(series: pd.Series, required_return: Optional[float] = None) -> float:
    """
    Sortino ratio computed against a positive MAR (default: daily risk-free).
    This removes the ∞→hardcoded-100 sentinel pathology.
    """
    s = _sanitize_series(series)
    if len(s) < 2:
        return 0.0

    mar = _daily_required_return_from_settings() if required_return is None else float(required_return)
    mean_return = s.mean()

    downside = s[s < mar]
    if downside.empty:
        # True mathematical result is +inf (no downside vs MAR).
        return float("inf")

    # Semideviation around MAR
    downside_deviation = np.sqrt(np.mean(np.square(downside - mar)))
    if downside_deviation < 1e-12:
        return float("inf") if (mean_return > mar) else 0.0

    sortino = (mean_return - mar) / downside_deviation
    # Annualize
    sortino *= np.sqrt(TRADING_DAYS_PER_YEAR)
    return float(sortino) if np.isfinite(sortino) else float("inf")


def block_bootstrap_sortino(
    returns_series: pd.Series,
    block_size: int = 5,
    num_iterations: int = 500,
    required_return: Optional[float] = None,
) -> Dict[str, float]:
    """
    Sortino bootstrap with overlapping blocks and a positive MAR.
    We compute Sortino on each bootstrap sample vs MAR, then take finite quantiles.
    """
    rs = _sanitize_series(returns_series)
    n = len(rs)
    if n < max(block_size * 2, 10):
        return {"sortino_median": 0.0, "sortino_lb": 0.0, "sortino_ub": 0.0}

    b = int(block_size)
    if b <= 0 or n < b:
        return {"sortino_median": 0.0, "sortino_lb": 0.0, "sortino_ub": 0.0}

    x = rs.to_numpy(copy=False)
    n_blocks = n - b + 1
    if n_blocks <= 0:
        return {"sortino_median": 0.0, "sortino_lb": 0.0, "sortino_ub": 0.0}

    stride = x.strides[0]
    blocks = np.lib.stride_tricks.as_strided(x, shape=(n_blocks, b), strides=(stride, stride))

    k = max(n // b, 1)
    sortino_vals = np.empty(int(num_iterations), dtype=float)
    mar = _daily_required_return_from_settings() if required_return is None else float(required_return)

    rng = np.random.default_rng()
    for i in range(int(num_iterations)):
        idx = rng.integers(0, n_blocks, size=k, endpoint=False)
        sample = blocks[idx].ravel()[:n]
        sortino_vals[i] = calculate_sortino_ratio(pd.Series(sample), required_return=mar)

    # Keep finite values only for quantiles
    finite_mask = np.isfinite(sortino_vals)
    finite = sortino_vals[finite_mask]
    if finite.size == 0:
        # Pathological case: every resample is ∞ → define a very high but finite floor
        # using the single-sample Sortino against MAR to avoid returning 0s.
        single = calculate_sortino_ratio(rs, required_return=mar)
        if not np.isfinite(single):
            return {"sortino_median": 0.0, "sortino_lb": 0.0, "sortino_ub": 0.0}
        finite = np.array([single], dtype=float)

    return {
        "sortino_median": float(np.nanmedian(finite)),
        "sortino_lb": float(np.nanpercentile(finite, 5)),
        "sortino_ub": float(np.nanpercentile(finite, 95)),
    }


# =========================
# Existing Metrics (Kept for compatibility/reporting)
# =========================
def block_bootstrap_sharpe(
    returns_series: pd.Series,
    block_size: int = 5,
    num_iterations: int = 500,
    trading_days_per_year: int = int(TRADING_DAYS_PER_YEAR),
) -> Dict[str, float]:
    """
    Sharpe bootstrap with overlapping blocks (robust to autocorrelation).
    Vectorized/NumPy implementation. Returns median and 5/95 percentiles.
    """
    rs = _sanitize_series(returns_series)
    n = len(rs)
    if n < max(block_size * 2, 10):
        return {"sharpe_median": 0.0, "sharpe_lb": 0.0, "sharpe_ub": 0.0}

    b = int(block_size)
    if b <= 0 or n < b:
        return {"sharpe_median": 0.0, "sharpe_lb": 0.0, "sharpe_ub": 0.0}

    x = rs.to_numpy(copy=False)
    n_blocks = n - b + 1
    if n_blocks <= 0:
        return {"sharpe_median": 0.0, "sharpe_lb": 0.0, "sharpe_ub": 0.0}

    stride = x.strides[0]
    blocks = np.lib.stride_tricks.as_strided(x, shape=(n_blocks, b), strides=(stride, stride))

    k = max(n // b, 1)  # blocks per bootstrap sample
    sharpe_vals = np.empty(int(num_iterations), dtype=float)
    sqrt_annual = np.sqrt(trading_days_per_year)

    rng = np.random.default_rng()
    for i in range(int(num_iterations)):
        idx = rng.integers(0, n_blocks, size=k, endpoint=False)
        sample = blocks[idx].ravel()[:n]
        std = sample.std(ddof=1)
        if std <= 1e-12 or not np.isfinite(std):
            sharpe_vals[i] = 0.0
        else:
            sharpe_vals[i] = (sample.mean() / std) * sqrt_annual

    sharpe_vals = sharpe_vals[np.isfinite(sharpe_vals)]
    if sharpe_vals.size == 0:
        return {"sharpe_median": 0.0, "sharpe_lb": 0.0, "sharpe_ub": 0.0}

    return {
        "sharpe_median": float(np.nanmedian(sharpe_vals)),
        "sharpe_lb": float(np.nanpercentile(sharpe_vals, 5)),
        "sharpe_ub": float(np.nanpercentile(sharpe_vals, 95)),
    }


def calculate_omega_ratio(series: pd.Series, required_return: float = 0.0) -> float:
    """
    Analytic Omega = (sum gains over threshold) / (abs sum losses below threshold).
    Returns +inf if there are no losses.
    """
    s = _sanitize_series(series)
    if s.empty:
        return 0.0
    x = s - float(required_return)
    gains = x[x > 0].sum()
    losses = x[x < 0].sum()
    denom = abs(losses)
    if denom < 1e-12:
        return float("inf")
    return float(gains / denom)


def calculate_max_drawdown(series: pd.Series) -> float:
    """
    Max drawdown computed from cumulative product of (1 + r_t).
    Returns negative number (e.g., -0.35 for -35%).
    """
    s = _sanitize_series(series)
    if s.empty:
        return 0.0
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = float(dd.min())
    if not np.isfinite(mdd):
        return 0.0
    return mdd


# =========================
# Main Portfolio Metrics Calculator
# =========================
def calculate_portfolio_metrics(
    daily_returns: pd.Series,
    portfolio_ledger: Optional[pd.DataFrame] = None,
    do_winsorize: bool = True,
) -> Dict[str, float]:
    """
    Canonical metrics from a daily return series and the underlying ledger.
    """
    s = _sanitize_series(daily_returns)
    if s.empty:
        return {}

    if do_winsorize:
        alpha = float(getattr(settings.reporting, "trimmed_alpha", 0.05))
        if len(s) > 20:  # Avoid winsorizing very small samples
            s = winsorize(s, lower_q=alpha, upper_q=1.0 - alpha)

    # --- MAR-aware Sortino (with bootstrap) ---
    mar = _daily_required_return_from_settings()
    sortino_stats = block_bootstrap_sortino(s, block_size=5, num_iterations=500, required_return=mar)

    # --- Other metrics ---
    expectancy = calculate_expectancy(portfolio_ledger) if portfolio_ledger is not None else 0.0
    support = float(len(daily_returns))
    sharpe_stats = block_bootstrap_sharpe(s)
    omega = calculate_omega_ratio(s)
    max_dd = calculate_max_drawdown(s)

    out = {
        "support": support,
        "expectancy": expectancy,
        "max_drawdown": float(max_dd),
        "omega_ratio": float(omega),
    }
    out.update({k: float(v) for k, v in sharpe_stats.items()})
    out.update({k: float(v) for k, v in sortino_stats.items()})

    return {k: (0.0 if v is None or not np.isfinite(v) else float(v)) for k, v in out.items()}
