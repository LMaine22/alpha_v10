from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from math import sqrt, isfinite
from scipy.stats import norm

TRADING_DAYS = 252

def daily_returns_from_ledger(ledger: pd.DataFrame, base_capital: float) -> pd.Series:
    """
    NAV-style daily returns using realized P&L on exit dates.
    """
    if ledger is None or ledger.empty:
        return pd.Series(dtype=float)
    df = ledger.copy()
    if "exit_time" not in df or "pnl_dollars" not in df:
        return pd.Series(dtype=float)
    df["exit_time"] = pd.to_datetime(df["exit_time"]).dt.normalize()
    df["entry_time"] = pd.to_datetime(df.get("entry_time", pd.NaT)).dt.normalize()
    start = min(df["exit_time"].min(), df["entry_time"].min())
    end   = max(df["exit_time"].max(), df["entry_time"].max())
    if pd.isna(start) or pd.isna(end):
        return pd.Series(dtype=float)
    idx = pd.date_range(start=start, end=end, freq="B")
    realized = df.groupby("exit_time")["pnl_dollars"].sum(min_count=1).reindex(idx, fill_value=0.0).astype(float)
    cum = realized.cumsum()
    nav_prev = (base_capital + cum.shift(1).fillna(0.0)).astype(float)
    denom = nav_prev.replace(0.0, np.nan)
    daily_ret = (realized / denom).fillna(0.0).astype(float)
    daily_ret.index.name = "date"
    return daily_ret

def ewma_mean_std(x: pd.Series, halflife: float) -> Tuple[float, float]:
    if x is None or x.empty:
        return 0.0, 0.0
    w = x.ewm(halflife=max(1.0, float(halflife)), adjust=False)
    m = float(w.mean().iloc[-1])
    s = float(w.std(bias=False).iloc[-1])
    return m, s

def ewma_sharpe(x: pd.Series, halflife: float) -> float:
    m, s = ewma_mean_std(x, halflife)
    if s <= 0 or not isfinite(s):
        return 0.0
    return (m / s) * sqrt(TRADING_DAYS)

def max_drawdown_from_returns(x: pd.Series) -> float:
    """Return positive magnitude of max drawdown from a returns series."""
    if x is None or x.empty:
        return 0.0
    nav = (1.0 + x).cumprod()
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return float(abs(dd.min()))

def estimate_block_len(n: int, method: str = "auto", lmin: int = 5, lmax: int = 50, series: Optional[pd.Series] = None) -> int:
    """
    Heuristic block length for MBB. If 'auto', use n**(1/3) clipped; if 'acf1', use lag-1 autocorr.
    """
    if n <= 10:
        return max(lmin, 3)
    if method == "acf1" and series is not None and len(series) >= 20:
        rho1 = float(series.autocorr(lag=1))
        rho1 = max(-0.99, min(0.99, rho1))
        # simple mapping: larger autocorr -> larger block
        L = int(round((1.0/(1.0 - abs(rho1))) * 10.0))
    else:
        L = int(round(n ** (1.0/3.0)))
    return max(lmin, min(lmax, L))

def mbb_resample(series: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving Block Bootstrap (overlapping blocks).
    """
    n = len(series)
    if n == 0:
        return series
    L = max(1, int(block_len))
    blocks = np.array([series[i:i+L] for i in range(0, n - L + 1)])
    k = int(np.ceil(n / L))
    idx = rng.integers(0, len(blocks), size=k)
    sample = np.concatenate(blocks[idx], axis=None)[:n]
    return sample

def sharpe_ratio(x: pd.Series) -> float:
    if x is None or x.empty:
        return 0.0
    m = float(x.mean())
    s = float(x.std(ddof=1))
    if s <= 0:
        return 0.0
    return (m / s) * sqrt(TRADING_DAYS)

def pvalue_sharpe_via_mbb(x: pd.Series, B: int, rng: np.random.Generator, block_len: int) -> float:
    """
    One-sided p-value for H0: SR<=0 vs SR>0 using MBB over returns.
    """
    x = x.dropna().astype(float)
    n = len(x)
    if n < 20:
        return 1.0
    sr_star = []
    base = x.to_numpy()
    for _ in range(int(B)):
        res = mbb_resample(base, block_len, rng)
        sr = sharpe_ratio(pd.Series(res, index=x.index))
        sr_star.append(sr)
    sr_star = np.asarray(sr_star, dtype=float)
    p = float(np.mean(sr_star <= 0.0))
    return p

def benjamini_hochberg(pvals: pd.Series, q: float) -> pd.Series:
    """
    Return boolean mask of discoveries under BH-FDR at level q.
    """
    s = pvals.sort_values()
    m = len(s)
    thresh = pd.Series([q * (i+1) / m for i in range(m)], index=s.index)
    ok = s <= thresh
    if not ok.any():
        return pd.Series(False, index=pvals.index)
    k_star = ok[ok].index[-1]
    cutoff = s.loc[k_star]
    return pvals <= cutoff

def dsr(sr: float, skew: float, kurt: float, T: int, N_eff: int) -> float:
    """
    Deflated Sharpe Ratio:
      sigma_SR^2 = 1/(T-1) * ( 1 - skew*SR + ((kurt-1)/4)*SR^2 )
      SR0 ~ expected max under null given N_eff (extreme value approx).
      DSR = Phi( (SR - SR0)/sqrt(sigma_SR^2) )
    """
    T = max(3, int(T))
    # moments guards
    if not np.isfinite(skew): skew = 0.0
    if not np.isfinite(kurt): kurt = 3.0
    var_sr = (1.0/(T-1.0)) * (1.0 - skew*sr + ((kurt - 1.0)/4.0)*(sr**2))
    var_sr = max(1e-12, float(var_sr))
    sigma_sr = sqrt(var_sr)

    # Effective trials → expected max Sharpe under null (approx)
    N_eff = max(1, int(N_eff))
    gamma = 0.5772156649  # Euler–Mascheroni
    # two-term Gumbel-ish approx for normal maxima (per LoP/BLP style)
    from scipy.stats import norm as _N
    SR0 = sigma_sr * ((1 - gamma) * _N.ppf(1 - 1.0/N_eff) + gamma * _N.ppf(1 - 1.0/(N_eff*np.e)))
    z = (sr - SR0) / sigma_sr
    return float(norm.cdf(z))

def effective_trials_from_corr(corr: np.ndarray) -> int:
    """
    Crude shrinkage of M by average absolute off-diagonal correlation.
    """
    M = corr.shape[0]
    if M <= 1:
        return 1
    mask = ~np.eye(M, dtype=bool)
    r = float(np.nanmean(np.abs(corr[mask])))
    r = max(0.0, min(0.99, r))
    # shrink M by average dependence
    N_eff = int(round(M / (1.0 + r * (M - 1.0))))
    return max(1, N_eff)
