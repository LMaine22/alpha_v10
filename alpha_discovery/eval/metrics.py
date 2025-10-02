# alpha_discovery/eval/metrics.py
"""
Institutional-grade metrics library for trading P&L evaluation.

Design goals
------------
- Vast coverage: returns-, drawdown-, tail-, trade-level, and robustness metrics.
- Robustness: MAR-aware downside; bootstrap lower-bounds; stationary/MBB bootstraps.
- No hard caps: allow np.inf where appropriate (e.g., no losing trades), PLUS provide
  finite lower-bound variants so optimizers can rank safely.
- Stability on small samples: deterministic fallbacks; gentle winsorization optional.
- Minimal deps: numpy, pandas. Optional settings import for MAR/risk-free.

Key groups
----------
1) Sanitizers/Utilities: safe handling of NaN/Inf, winsorization, NAV, CAGR, drawdowns.
2) Return Distribution Metrics: Sharpe, Sortino (MAR-aware), Omega, Calmar, Sterling,
   Burke, UPI/Ulcer, Pain/GTPI, Profit Factor, Payoff Ratio, Hit Rate, Skew/Kurt, Tail index.
3) Tail Risk: VaR/CVaR (historical, Gaussian, Cornish-Fisher), CDaR (Conditional Drawdown).
4) Bootstraps: MBB (overlapping) and Stationary bootstrap; generic metric bootstrapping.
5) Probabilistic/Deflated Sharpe: PSR and DSR (Bailey & López de Prado approximations).
6) Benchmark/Tracking (optional input): Beta/Alpha/IR if benchmark series provided.
7) Bundles: compute_portfolio_metrics_bundle(...) returns a rich dictionary.

Notes
-----
- "no losing trades" yields PF=inf, Omega=inf, etc. We *also* return bootstrap LBs
  (finite), which you can prioritize in GA fitness.
- Astronomical values: no hard caps; LB/quantiles are the robust controls.
- MAR: daily MAR derived from settings.options.constant_r if available, else 0.

"""

from __future__ import annotations
from typing import Dict, Optional, Callable, Tuple, Iterable, Any
import math
import numpy as np
import pandas as pd

# Optional settings import (safe if absent)
try:
    from ..config import settings
    _HAS_SETTINGS = True
except Exception:
    settings = None  # type: ignore
    _HAS_SETTINGS = False

# -------------------------
# Constants & small epsilons
# -------------------------
TRADING_DAYS_PER_YEAR = 252.0
_EPS = 1e-12


# =========================
# 1) SANITIZERS & UTILITIES
# =========================
def _to_float_series(x: Optional[pd.Series]) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return s.astype(float)


def sanitize(returns: Optional[pd.Series]) -> pd.Series:
    """Float series, drop NaN/Inf; index preserved if possible."""
    return _to_float_series(returns)


def winsorize(
    s: pd.Series, lower_q: float = 0.005, upper_q: float = 0.995
) -> pd.Series:
    """Winsorize at quantiles (gentle, opt-in). No-op on empty."""
    s = _to_float_series(s)
    if s.empty:
        return s
    lq = float(np.clip(lower_q, 0.0, 0.49))
    uq = float(np.clip(upper_q, lq + 1e-9, 1.0))
    lo = s.quantile(lq)
    hi = s.quantile(uq)
    return s.clip(lower=lo, upper=hi)


def daily_mar_from_settings() -> float:
    """Annual RF from settings → daily MAR (Sortino)."""
    if _HAS_SETTINGS:
        try:
            rf_annual = float(getattr(settings.options, "constant_r", 0.0))
        except Exception:
            rf_annual = 0.0
    else:
        rf_annual = 0.0
    return rf_annual / TRADING_DAYS_PER_YEAR


def nav_from_returns(returns: pd.Series, start_nav: float = 1.0) -> pd.Series:
    r = _to_float_series(returns)
    if r.empty:
        return pd.Series(dtype=float)
    nav = (1.0 + r).cumprod() * float(start_nav)
    return nav


def cagr_from_nav(nav: pd.Series) -> float:
    """CAGR based on NAV series with trading day approximation."""
    nav = _to_float_series(nav)
    if nav.empty:
        return 0.0
    total_return = float(nav.iloc[-1] / nav.iloc[0]) - 1.0
    n = len(nav)
    years = n / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drawdown series from NAV."""
    nav = _to_float_series(nav)
    if nav.empty:
        return pd.Series(dtype=float)
    peaks = nav.cummax()
    dd = nav / peaks - 1.0
    return dd


def max_drawdown(nav: pd.Series) -> float:
    dd = drawdown_series(nav)
    return float(dd.min()) if not dd.empty else 0.0


def average_drawdown(nav: pd.Series) -> float:
    dd = drawdown_series(nav)
    if dd.empty:
        return 0.0
    return float(dd.mean())


def ulcer_index(nav: pd.Series) -> float:
    """Ulcer Index: sqrt(mean(Drawdown%^2)). Drawdown% = 100*DD."""
    dd = drawdown_series(nav) * 100.0
    if dd.empty:
        return 0.0
    return float(np.sqrt(np.mean(np.square(dd))))


def ulcer_performance_index(nav: pd.Series, rf_annual: float = 0.0) -> float:
    """UPI = (CAGR - rf) / UlcerIndex."""
    ui = ulcer_index(nav)
    cagr = cagr_from_nav(nav)
    numer = cagr - rf_annual
    if ui <= _EPS:
        return np.inf if numer > 0 else 0.0
    return float(numer / ui)


def pain_index(nav: pd.Series) -> float:
    """Pain Index = mean absolute drawdown (0..1)."""
    dd = drawdown_series(nav)
    if dd.empty:
        return 0.0
    return float(np.mean(np.abs(dd)))


def pain_ratio(nav: pd.Series, rf_annual: float = 0.0) -> float:
    """Pain Ratio = (CAGR - rf)/PainIndex."""
    pi = pain_index(nav)
    cagr = cagr_from_nav(nav)
    numer = cagr - rf_annual
    if pi <= _EPS:
        return np.inf if numer > 0 else 0.0
    return float(numer / pi)


def burke_ratio(nav: pd.Series, rf_annual: float = 0.0) -> float:
    """Burke ratio ~ (CAGR - rf) / sqrt(sum(DD^2)/N)."""
    dd = drawdown_series(nav)
    if dd.empty:
        return 0.0
    denom = float(np.sqrt(np.mean(np.square(dd))))
    if denom <= _EPS:
        return np.inf if (cagr_from_nav(nav) - rf_annual) > 0 else 0.0
    return float((cagr_from_nav(nav) - rf_annual) / denom)


def sterling_ratio(nav: pd.Series, rf_annual: float = 0.0, top_k: int = 3) -> float:
    """
    Sterling ratio ~ (CAGR - rf) / average of top-k drawdowns (absolute).
    """
    dd = drawdown_series(nav)
    if dd.empty:
        return 0.0
    worst = np.sort(np.abs(dd.values))[-top_k:]
    denom = float(np.mean(worst)) if worst.size else 0.0
    if denom <= _EPS:
        return np.inf if (cagr_from_nav(nav) - rf_annual) > 0 else 0.0
    return float((cagr_from_nav(nav) - rf_annual) / denom)


# =====================================
# 2) RETURN DISTRIBUTION CORE METRICS
# =====================================
def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe using daily returns."""
    r = _to_float_series(returns)
    if len(r) < 2:
        return 0.0
    mu = float(r.mean()) * TRADING_DAYS_PER_YEAR
    sd = float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    if sd <= _EPS:
        return np.inf if mu > 0 else 0.0
    return float(mu / sd)


def sortino(returns: pd.Series, mar_daily: Optional[float] = None) -> float:
    """
    MAR-aware Sortino (zeros don't create downside).
    Annualized with daily approximation.
    """
    r = _to_float_series(returns)
    if len(r) < 2:
        return 0.0
    mar = daily_mar_from_settings() if mar_daily is None else float(mar_daily)
    excess = r - mar
    downside = excess[excess < 0.0]
    if downside.empty:
        return np.inf if excess.mean() > 0 else 0.0
    denom = float(downside.std(ddof=1))
    if denom <= _EPS:
        return np.inf if excess.mean() > 0 else 0.0
    s = float(excess.mean() / denom) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return s


def omega(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio via discrete sums around threshold."""
    r = _to_float_series(returns)
    if r.empty:
        return 0.0
    gains = np.clip(r - threshold, 0.0, None).sum()
    losses = np.clip(threshold - r, 0.0, None).sum()
    if losses <= _EPS:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def profit_factor(trade_returns: pd.Series) -> float:
    """Profit Factor = sum wins / sum |losses| (per trade series)."""
    t = _to_float_series(trade_returns)
    if t.empty:
        return 0.0
    wins = t[t > 0.0].sum()
    losses = -t[t < 0.0].sum()
    if losses <= _EPS:
        return np.inf if wins > 0 else 0.0
    return float(wins / losses)


def payoff_ratio(trade_returns: pd.Series) -> float:
    """Average win / average loss (absolute)."""
    t = _to_float_series(trade_returns)
    if t.empty:
        return 0.0
    wins = t[t > 0.0]
    losses = -t[t < 0.0]
    if len(losses) == 0:
        return np.inf if len(wins) > 0 else 0.0
    return float(wins.mean() / max(losses.mean(), _EPS))


def hit_rate(trade_returns: pd.Series) -> float:
    """Fraction of positive trades."""
    t = _to_float_series(trade_returns)
    if t.empty:
        return 0.0
    return float((t > 0.0).mean())


def calmar(nav: pd.Series) -> float:
    """Calmar = CAGR / |MaxDD|."""
    c = cagr_from_nav(nav)
    mdd = abs(max_drawdown(nav))
    if mdd <= _EPS:
        return np.inf if c > 0 else 0.0
    return float(c / mdd)


def mar_ratio(nav: pd.Series) -> float:
    """Synonym for Calmar in many contexts."""
    return calmar(nav)


def higher_moments(returns: pd.Series) -> Dict[str, float]:
    """Sample skewness and excess kurtosis."""
    r = _to_float_series(returns)
    if len(r) < 3:
        return {"skew": 0.0, "ex_kurt": 0.0}
    x = r.values
    m = x.mean()
    s = x.std(ddof=1)
    if s <= _EPS:
        return {"skew": 0.0, "ex_kurt": 0.0}
    z = (x - m) / s
    skew = float((np.mean(z**3)))
    ex_kurt = float((np.mean(z**4)) - 3.0)
    return {"skew": skew, "ex_kurt": ex_kurt}


# ===========================
# 3) TAIL RISK & DRAWDOWN TAIL
# ===========================
def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical VaR at alpha (loss is negative)."""
    r = _to_float_series(returns)
    if r.empty:
        return 0.0
    return float(np.quantile(r, alpha))


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical CVaR at alpha (expected loss in tail)."""
    r = _to_float_series(returns)
    if r.empty:
        return 0.0
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    if tail.empty:
        return q
    return float(tail.mean())


def cdar(nav: pd.Series, alpha: float = 0.05) -> float:
    """Conditional Drawdown at Risk at alpha on drawdown distribution."""
    dd = drawdown_series(nav)
    if dd.empty:
        return 0.0
    q = np.quantile(dd, alpha)  # negative
    tail = dd[dd <= q]
    if tail.empty:
        return q
    return float(tail.mean())


# =========================
# 4) BOOTSTRAP UTILITIES
# =========================
def _mbb_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Overlapping moving-block bootstrap indices of length n."""
    if n <= 0:
        return np.arange(0)
    b = int(max(1, block_size))
    n_blocks = max(1, int(np.ceil(n / b)))
    starts = rng.integers(0, max(1, n - b + 1), size=n_blocks)
    idx = []
    for s in starts:
        idx.extend(range(s, min(s + b, n)))
        if len(idx) >= n:
            break
    return np.array(idx[:n], dtype=int)


def _stationary_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary bootstrap indices (Politis-Romano)."""
    if n <= 0:
        return np.arange(0)
    p = 1.0 / max(1, mean_block)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t-1] + 1) % n
    return idx


def bootstrap_metric(
    series: pd.Series,
    metric_fn: Callable[[pd.Series], float],
    n_boot: int = 1000,
    block_size: int = 20,
    method: str = "stationary",  # "mbb" or "stationary"
    seed: Optional[int] = 42,
    small_sample_fallback: bool = True,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> Dict[str, float]:
    """
    Generic block bootstrap for a scalar metric.
    Returns dict with {"median", "lb", "ub"}.
    """
    s = _to_float_series(series)
    n = len(s)
    if n == 0:
        val = 0.0
        try:
            val = float(metric_fn(pd.Series([], dtype=float)))
        except Exception:
            pass
        return {"median": val, "lb": val, "ub": val}

    # Small sample: return point estimate deterministically
    if small_sample_fallback and n < max(10, block_size * 2):
        val = float(metric_fn(s))
        return {"median": val, "lb": val, "ub": val}

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        if method == "mbb":
            idx = _mbb_indices(n, block_size, rng)
        else:
            idx = _stationary_indices(n, max(2, block_size), rng)
        boot = s.iloc[idx]
        v = float(metric_fn(boot))
        vals[i] = v

    return {
        "median": float(np.nanmedian(vals)),
        "lb": float(np.nanpercentile(vals, q_low * 100.0)),
        "ub": float(np.nanpercentile(vals, q_high * 100.0)),
    }


# ==========================================
# 5) PROBABILISTIC & DEFLATED SHARPE (PSR/DSR)
# ==========================================
def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR): P(SR > sr_benchmark).
    Bailey & López de Prado (2012).

    PSR = Phi( ( (SR - SR*) * sqrt(n - 1) ) /
                sqrt( 1 - g1*SR + ((g2-1)/4)*SR^2 ) )
    where g1=skewness, g2=excess kurtosis + 3 (here we pass excess+3 -> g2).
    """
    r = _to_float_series(returns)
    n = len(r)
    if n < 3:
        return 0.0
    sr = sharpe(r)
    hm = higher_moments(r)
    g1 = hm["skew"]
    g2 = hm["ex_kurt"] + 3.0  # total kurtosis
    denom = math.sqrt(max(1e-12, 1.0 - g1 * sr + ((g2 - 1.0) / 4.0) * (sr ** 2)))
    z = (sr - float(sr_benchmark)) * math.sqrt(max(1.0, n - 1.0)) / denom
    # Normal CDF
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    sr_benchmark: float = 0.0
) -> float:
    """
    Deflated Sharpe Ratio (DSR) ~ PSR adjusted for selection bias
    from multiple trials (strategies, parameters).
    Uses an approximation via PSR with a higher effective benchmark.

    Reference: Bailey & López de Prado (2014).
    """
    # Effective benchmark is raised due to multiple comparisons.
    # Approximate inflation of SR* by sqrt(2*log(n_trials)/ (n-1))
    r = _to_float_series(returns)
    n = len(r)
    if n < 3:
        return 0.0
    sr_star = float(sr_benchmark)
    if n_trials > 1:
        sr_star = sr_benchmark + math.sqrt(max(0.0, 2.0 * math.log(n_trials) / max(1.0, n - 1.0)))
    return probabilistic_sharpe_ratio(r, sr_benchmark=sr_star)


# ===================================
# 6) BENCHMARK / TRACKING (OPTIONAL)
# ===================================
def beta_alpha_information_ratio(
    returns: pd.Series,
    benchmark: Optional[pd.Series]
) -> Dict[str, float]:
    """
    CAPM beta & alpha (daily to annual) and Information Ratio vs benchmark.
    Returns zeros if benchmark missing or ill-conditioned.
    """
    r = _to_float_series(returns)
    if benchmark is None:
        return {"beta": 0.0, "alpha_annual": 0.0, "tracking_error": 0.0, "info_ratio": 0.0}
    b = _to_float_series(benchmark)
    if r.empty or b.empty:
        return {"beta": 0.0, "alpha_annual": 0.0, "tracking_error": 0.0, "info_ratio": 0.0}
    df = pd.concat([r, b], axis=1).dropna()
    if df.shape[0] < 3:
        return {"beta": 0.0, "alpha_annual": 0.0, "tracking_error": 0.0, "info_ratio": 0.0}
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    vx = x.var(ddof=1)
    if vx <= _EPS:
        return {"beta": 0.0, "alpha_annual": 0.0, "tracking_error": 0.0, "info_ratio": 0.0}
    beta = float(np.cov(y, x, ddof=1)[0, 1] / vx)
    resid = y - beta * x
    alpha_daily = float(resid.mean())
    alpha_annual = alpha_daily * TRADING_DAYS_PER_YEAR
    te = float(resid.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
    ir = float(alpha_annual / te) if te > _EPS else (np.inf if alpha_annual > 0 else 0.0)
    return {"beta": beta, "alpha_annual": alpha_annual, "tracking_error": te, "info_ratio": ir}


# ======================
# 7) TRADE-LEVEL METRICS
# ======================
def expectancy(trade_ledger: Optional[pd.DataFrame]) -> float:
    """
    Expectancy per trade. Column detection tolerant: 'pnl_pct'|'ret_pct'|'return_pct'|'pnl%'.
    """
    if trade_ledger is None or len(trade_ledger) == 0:
        return 0.0
    cols = {c.lower(): c for c in trade_ledger.columns}
    for k in ("pnl_pct", "ret_pct", "return_pct", "pnl%"):
        if k in cols:
            s = pd.to_numeric(trade_ledger[cols[k]], errors="coerce").dropna().astype(float)
            return float(s.mean()) if len(s) else 0.0
    return 0.0


def avg_win_loss(trade_ledger: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Average win size, average loss size (absolute)."""
    if trade_ledger is None or len(trade_ledger) == 0:
        return {"avg_win": 0.0, "avg_loss": 0.0}
    cols = {c.lower(): c for c in trade_ledger.columns}
    col = None
    for k in ("pnl_pct", "ret_pct", "return_pct", "pnl%"):
        if k in cols:
            col = cols[k]
            break
    if col is None:
        return {"avg_win": 0.0, "avg_loss": 0.0}
    s = pd.to_numeric(trade_ledger[col], errors="coerce").dropna().astype(float)
    wins = s[s > 0.0]
    losses = -s[s < 0.0]
    return {
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
    }


def longest_win_loss_streak(trade_ledger: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Longest consecutive wins and losses."""
    if trade_ledger is None or len(trade_ledger) == 0:
        return {"longest_win_streak": 0, "longest_loss_streak": 0}
    cols = {c.lower(): c for c in trade_ledger.columns}
    col = None
    for k in ("pnl_pct", "ret_pct", "return_pct", "pnl%"):
        if k in cols:
            col = cols[k]
            break
    if col is None:
        return {"longest_win_streak": 0, "longest_loss_streak": 0}
    s = pd.to_numeric(trade_ledger[col], errors="coerce").dropna().astype(float)
    signs = np.sign(s.values)
    lw = ll = curw = curl = 0
    for v in signs:
        if v > 0:
            curw += 1
            lw = max(lw, curw)
            curl = 0
        elif v < 0:
            curl += 1
            ll = max(ll, curl)
            curw = 0
        else:
            curw = 0; curl = 0
    return {"longest_win_streak": int(lw), "longest_loss_streak": int(ll)}


# ============================
# 8) METRIC BUNDLE (ONE-STOP)
# ============================
def compute_portfolio_metrics_bundle(
    daily_returns: pd.Series,
    trade_ledger: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    do_winsorize: bool = True,
    winsor_q: Tuple[float, float] = (0.005, 0.995),
    bootstrap_B: int = 1000,
    bootstrap_block: int = 20,
    bootstrap_method: str = "stationary",
    seed: Optional[int] = 42,
    mar_daily: Optional[float] = None,
    n_trials_for_dsr: int = 1,
) -> Dict[str, Any]:
    """
    Comprehensive bundle for GA/reporting (point estimates + lower/upper bounds).

    Returns keys include (non-exhaustive):
    - support, cagr, max_drawdown, avg_drawdown, ulcer_index, upi, pain_index, pain_ratio,
      burke_ratio, sterling_ratio, calmar, mar_ratio.
    - sharpe, sortino, omega, var/cvar, cdar.
    - bootstrap_{sharpe,sortino,omega,calmar,profit_factor}_lb/median/ub.
    - psr, dsr.
    - trade metrics: expectancy, profit_factor, payoff_ratio, hit_rate, avg_win/avg_loss, streaks.
    - benchmark: beta, alpha_annual, tracking_error, info_ratio (if provided).

    Robustness conventions:
    - When a metric can be inf (e.g., no losses), we *also* compute a bootstrap LB (finite) for ranking.
    - For tiny samples (< ~block*2), bootstrap falls back to point estimate deterministically.
    """
    r = sanitize(daily_returns)
    if r.empty:
        # Empty stub with zeros (safe to merge)
        out = {k: 0.0 for k in [
            "support","cagr","max_drawdown","avg_drawdown","ulcer_index","upi","pain_index",
            "pain_ratio","burke_ratio","sterling_ratio","calmar","mar_ratio","sharpe","sortino",
            "omega","var_5","cvar_5","cdar_5","psr","dsr","expectancy","profit_factor","payoff_ratio",
            "hit_rate","beta","alpha_annual","tracking_error","info_ratio"
        ]}
        out.update({"avg_win":0.0,"avg_loss":0.0,"longest_win_streak":0,"longest_loss_streak":0})
        out.update({
            "bootstrap_sharpe_lb":0.0,"bootstrap_sharpe_median":0.0,"bootstrap_sharpe_ub":0.0,
            "bootstrap_sortino_lb":0.0,"bootstrap_sortino_median":0.0,"bootstrap_sortino_ub":0.0,
            "bootstrap_omega_lb":0.0,"bootstrap_omega_median":0.0,"bootstrap_omega_ub":0.0,
            "bootstrap_calmar_lb":0.0,"bootstrap_calmar_median":0.0,"bootstrap_calmar_ub":0.0,
            "bootstrap_profit_factor_lb":0.0,"bootstrap_profit_factor_median":0.0,"bootstrap_profit_factor_ub":0.0,
        })
        return out

    if do_winsorize and len(r) > 20:
        r = winsorize(r, winsor_q[0], winsor_q[1])

    nav = nav_from_returns(r)
    rf_annual = float(getattr(settings.options, "constant_r", 0.0)) if _HAS_SETTINGS else 0.0
    mar = daily_mar_from_settings() if mar_daily is None else float(mar_daily)

    # Point estimates
    cagr = cagr_from_nav(nav)
    mdd = max_drawdown(nav)
    add = average_drawdown(nav)
    ui = ulcer_index(nav)
    upi = ulcer_performance_index(nav, rf_annual=rf_annual)
    pain = pain_index(nav)
    painr = pain_ratio(nav, rf_annual=rf_annual)
    burke = burke_ratio(nav, rf_annual=rf_annual)
    sterling = sterling_ratio(nav, rf_annual=rf_annual, top_k=3)
    cal = calmar(nav)
    mar_ratio_val = cal  # synonym

    sh = sharpe(r)
    so = sortino(r, mar_daily=mar)
    om = omega(r, threshold=0.0)
    v5 = var_historical(r, alpha=0.05)
    cv5 = cvar_historical(r, alpha=0.05)
    cdar5 = cdar(nav, alpha=0.05)

    # Trade-level
    pf = profit_factor(trade_ledger[trade_ledger.columns[0]] if isinstance(trade_ledger, pd.Series) else
                       (pd.to_numeric(trade_ledger.iloc[:,0], errors="coerce") if (isinstance(trade_ledger, pd.DataFrame) and trade_ledger.shape[1]==1)
                        else None)) if isinstance(trade_ledger, (pd.Series, pd.DataFrame)) else 0.0
    # If ledger is a DataFrame with standard columns, use helpers instead:
    if isinstance(trade_ledger, pd.DataFrame):
        pf = profit_factor(_to_float_series(
            pd.to_numeric(
                trade_ledger.get('pnl_pct', trade_ledger.get('ret_pct', trade_ledger.get('return_pct', trade_ledger.get('pnl%', pd.Series(dtype=float))))),
                errors='coerce'
            )
        ))
        pr = payoff_ratio(_to_float_series(
            pd.to_numeric(
                trade_ledger.get('pnl_pct', trade_ledger.get('ret_pct', trade_ledger.get('return_pct', trade_ledger.get('pnl%', pd.Series(dtype=float))))),
                errors='coerce'
            )
        ))
        hr = hit_rate(_to_float_series(
            pd.to_numeric(
                trade_ledger.get('pnl_pct', trade_ledger.get('ret_pct', trade_ledger.get('return_pct', trade_ledger.get('pnl%', pd.Series(dtype=float))))),
                errors='coerce'
            )
        ))
        expct = expectancy(trade_ledger)
        awl = avg_win_loss(trade_ledger)
        streaks = longest_win_loss_streak(trade_ledger)
    else:
        pr = 0.0
        hr = 0.0
        expct = 0.0
        awl = {"avg_win": 0.0, "avg_loss": 0.0}
        streaks = {"longest_win_streak": 0, "longest_loss_streak": 0}

    # Benchmark (optional)
    bench = beta_alpha_information_ratio(r, benchmark_returns)

    # Bootstrap lower/upper bounds
    bs = bootstrap_B
    blk = bootstrap_block
    meth = bootstrap_method

    bs_sh = bootstrap_metric(r, sharpe, n_boot=bs, block_size=blk, method=meth, seed=seed)
    bs_so = bootstrap_metric(r, lambda x: sortino(x, mar_daily=mar), n_boot=bs, block_size=blk, method=meth, seed=seed)
    bs_om = bootstrap_metric(r, lambda x: omega(x, 0.0), n_boot=bs, block_size=blk, method=meth, seed=seed)
    bs_cal = bootstrap_metric(nav, calmar, n_boot=bs, block_size=blk, method=meth, seed=seed)

    # Profit Factor bootstrap on trades (if we have a series of trade returns)
    if isinstance(trade_ledger, pd.DataFrame):
        tser = _to_float_series(
            pd.to_numeric(
                trade_ledger.get('pnl_pct', trade_ledger.get('ret_pct', trade_ledger.get('return_pct', trade_ledger.get('pnl%', pd.Series(dtype=float))))),
                errors='coerce'
            )
        )
        bs_pf = bootstrap_metric(tser, profit_factor, n_boot=bs, block_size=max(2, min(blk, max(2, len(tser)//10))), method="mbb", seed=seed)
    else:
        bs_pf = {"median": pf, "lb": pf, "ub": pf}

    # Probabilistic/Deflated Sharpe
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=max(1, int(n_trials_for_dsr)), sr_benchmark=0.0)

    out = {
        # Basic support
        "support": float(len(r)),

        # Growth & drawdown family
        "cagr": float(cagr),
        "max_drawdown": float(mdd),
        "avg_drawdown": float(add),
        "ulcer_index": float(ui),
        "upi": float(upi),
        "pain_index": float(pain),
        "pain_ratio": float(painr),
        "burke_ratio": float(burke),
        "sterling_ratio": float(sterling),
        "calmar": float(cal),
        "mar_ratio": float(mar_ratio_val),

        # Distribution family
        "sharpe": float(sh),
        "sortino": float(so),
        "omega": float(om),

        # Tail risk
        "var_5": float(v5),
        "cvar_5": float(cv5),
        "cdar_5": float(cdar5),

        # Prob/deflated
        "psr": float(psr),
        "dsr": float(dsr),

        # Trade-level
        "expectancy": float(expct),
        "profit_factor": float(pf),
        "payoff_ratio": float(pr),
        "hit_rate": float(hr),
        "avg_win": float(awl.get("avg_win", 0.0)),
        "avg_loss": float(awl.get("avg_loss", 0.0)),
        "longest_win_streak": int(streaks.get("longest_win_streak", 0)),
        "longest_loss_streak": int(streaks.get("longest_loss_streak", 0)),

        # Benchmark
        "beta": float(bench["beta"]),
        "alpha_annual": float(bench["alpha_annual"]),
        "tracking_error": float(bench["tracking_error"]),
        "info_ratio": float(bench["info_ratio"]),

        # Bootstrap bands (rank with *_lb if you want ultra-conservatism)
        "bootstrap_sharpe_median": float(bs_sh["median"]),
        "bootstrap_sharpe_lb": float(bs_sh["lb"]),
        "bootstrap_sharpe_ub": float(bs_sh["ub"]),

        "bootstrap_sortino_median": float(bs_so["median"]),
        "bootstrap_sortino_lb": float(bs_so["lb"]),
        "bootstrap_sortino_ub": float(bs_so["ub"]),

        "bootstrap_omega_median": float(bs_om["median"]),
        "bootstrap_omega_lb": float(bs_om["lb"]),
        "bootstrap_omega_ub": float(bs_om["ub"]),

        "bootstrap_calmar_median": float(bs_cal["median"]),
        "bootstrap_calmar_lb": float(bs_cal["lb"]),
        "bootstrap_calmar_ub": float(bs_cal["ub"]),

        "bootstrap_profit_factor_median": float(bs_pf["median"]),
        "bootstrap_profit_factor_lb": float(bs_pf["lb"]),
        "bootstrap_profit_factor_ub": float(bs_pf["ub"]),
    }

    # Replace non-finite with finite placeholders ONLY in bundle if needed for vector math.
    # (Leave infinities in raw point metrics; the *_lb fields are finite for ranking.)
    for k, v in list(out.items()):
        if isinstance(v, float) and (not np.isfinite(v)):
            # map +/-inf to a very large/small finite sentinel without capping legit signals
            out[k] = 1e9 if v > 0 else -1e9

    return out
