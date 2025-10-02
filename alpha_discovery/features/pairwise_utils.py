import numpy as np
import pandas as pd

__all__ = [
    "safe_roll",
    "safe_rolling_corr",
    "safe_rolling_cov",
    "safe_zscore",
    "safe_leadlag_corr",
]

def safe_roll(series: pd.Series, window: int, min_periods: int):
    return series.rolling(int(window), min_periods=int(min_periods))

def safe_rolling_corr(s1: pd.Series, s2: pd.Series, window: int, min_periods: int):
    return safe_roll(s1, window, min_periods).corr(s2)

def safe_rolling_cov(s1: pd.Series, s2: pd.Series, window: int, min_periods: int):
    return safe_roll(s1, window, min_periods).cov(s2)

def safe_zscore(x: pd.Series, window: int, min_periods: int):
    r = safe_roll(x, window, min_periods)
    mu = r.mean()
    sd = r.std(ddof=1)
    return (x - mu) / sd.replace(0, np.nan)

def safe_leadlag_corr(s1: pd.Series, s2: pd.Series, lags: list[int], window: int, min_periods: int):
    """Return (best_lag, best_corr_series) maximizing absolute corr across given lags."""
    best_abs = -np.inf
    best_lag = 0
    best = None
    for L in lags:
        s2s = s2.shift(L)
        corr = safe_rolling_corr(s1, s2s, window, min_periods)
        cmax = np.nanmax(np.abs(corr.values))
        if np.isfinite(cmax) and cmax > best_abs:
            best_abs = float(cmax)
            best_lag = int(L)
            best = corr
    return best_lag, best
