# alpha_discovery/features/core.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# A small constant to prevent division by zero
EPSILON = 1e-9


# ===================================
# Section 1: Normalization Functions
# ===================================

def zscore_rolling(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates the rolling z-score of a series.
    The z-score measures how many standard deviations an element is from the mean.
    """
    if min_periods is None:
        min_periods = window // 2

    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()

    return (s - mean) / (std + EPSILON)


def mad_z_rolling(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates a z-score using the Median Absolute Deviation (MAD).
    This is more robust to outliers than a standard z-score.
    """
    if min_periods is None:
        min_periods = window // 2

    s = pd.to_numeric(series, errors="coerce")

    # The constant 1.4826 scales the MAD to be comparable to the standard deviation
    # for a normal distribution.
    c = 1.4826

    median = s.rolling(window, min_periods=min_periods).median()
    mad = (s - median).abs().rolling(window, min_periods=min_periods).median()

    return (s - median) / (c * mad + EPSILON)


# ===================================
# Section 2: Relational Functions
# ===================================

def align_series(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Aligns two series by their index and drops any rows with NaNs in either.
    Ensures that calculations like correlation or beta are performed on a common set of dates.
    """
    s1 = pd.to_numeric(s1, errors="coerce")
    s2 = pd.to_numeric(s2, errors="coerce")
    df = pd.concat([s1, s2], axis=1, join='inner').dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def rolling_corr_fisher(s1: pd.Series, s2: pd.Series, window: int, min_periods: int = None) -> Tuple[pd.Series, pd.Series]:
    """
    Calculates the rolling correlation and its Fisher transformation.
    The Fisher transform helps normalize the correlation's distribution, making it
    more suitable for subsequent z-scoring.
    """
    if min_periods is None:
        min_periods = window // 2

    s1_aligned, s2_aligned = align_series(s1, s2)

    # Calculate rolling correlation, clipped to avoid issues with log(0)
    corr = s1_aligned.rolling(window, min_periods=min_periods).corr(s2_aligned).clip(-0.9999, 0.9999)

    # Apply the Fisher transformation
    fisher_transform = 0.5 * np.log((1 + corr) / (1 - corr))

    return corr, fisher_transform


def rolling_beta(s1: pd.Series, s2: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates the rolling beta of series s1 with respect to series s2.
    Beta = Cov(s1, s2) / Var(s2)
    """
    if min_periods is None:
        min_periods = window // 2

    s1_aligned, s2_aligned = align_series(s1, s2)

    covariance = s1_aligned.rolling(window, min_periods=min_periods).cov(s2_aligned)
    variance = s2_aligned.rolling(window, min_periods=min_periods).var()

    return covariance / (variance + EPSILON)


# ===================================
# Section 3: Time Series Functions
# ===================================

def get_realized_vol(price_series: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculates the annualized realized volatility over a rolling window.
    Based on the standard deviation of daily log returns.
    """
    px = pd.to_numeric(price_series, errors="coerce").astype(float)
    # Robust log returns: use log(px).diff() with a tiny floor to avoid log(<=0)
    log_px = np.log(px.clip(lower=1e-12))
    log_returns = log_px.diff()

    # Multiply by sqrt(252) to annualize the daily standard deviation
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252)
    return realized_vol


def frac_diff(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """
    Computes fractional differentiation of a time series.
    This helps to make the series stationary while preserving more memory
    than traditional integer differencing.

    Args:
        series (pd.Series): The input time series.
        d (float): The order of differentiation, typically between [0, 1].
        window (int): The lookback window to compute weights. A larger window
                      provides a more accurate but slower calculation.
    """
    s = pd.to_numeric(series, errors="coerce")

    # Calculate weights
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])  # Reverse weights for dot product

    # Apply weights
    output = s.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)
    return output


# ===================================
# Section 4: Sentiment & Momentum Helpers
# ===================================

def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Coerce to numeric while preserving index."""
    return pd.to_numeric(series, errors="coerce")


def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    s = to_numeric_safe(series)
    # keep your explicit ffill, but silence the future deprecation by setting fill_method=None
    return s.ffill().pct_change(n, fill_method=None)


def diff_n(series: pd.Series, n: int) -> pd.Series:
    s = to_numeric_safe(series)
    return s.diff(n)


def rolling_mean(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    s = to_numeric_safe(series)
    return s.rolling(window, min_periods=min_periods).mean()


def deviation_from_mean(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """(current - rolling_mean(window))"""
    s = to_numeric_safe(series)
    rm = rolling_mean(s, window, min_periods)
    return s - rm


def rolling_std(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    s = to_numeric_safe(series)
    return s.rolling(window, min_periods=min_periods).std()


def momentum_trend(series: pd.Series, short_win: int, long_win: int) -> pd.Series:
    """Rolling mean short - rolling mean long."""
    s = to_numeric_safe(series)
    return rolling_mean(s, short_win) - rolling_mean(s, long_win)


def acceleration(series: pd.Series, k: int) -> pd.Series:
    """Second difference over k (captures turning points)."""
    s = to_numeric_safe(series)
    return s.diff().diff(k)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Classic RSI on arbitrary series (not just prices)."""
    s = to_numeric_safe(series)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + EPSILON)
    return 100 - (100 / (1 + rs))


def spike_z(series: pd.Series, window: int = 30, min_periods: Optional[int] = None) -> pd.Series:
    """Z-score of the level (useful for spike/exhaustion detection)."""
    return zscore_rolling(to_numeric_safe(series), window, min_periods=min_periods)


def spike_on_change_z(series: pd.Series, change_n: int = 1, window: int = 30) -> pd.Series:
    """Z-score of an n-day change (captures sudden shifts)."""
    chg = diff_n(series, change_n)
    return zscore_rolling(chg, window)


# ===================================
# Section 5: Convenience Helpers for This Project
# ===================================

def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Elementwise safe division with numeric coercion and EPSILON protection."""
    n = to_numeric_safe(numer)
    d = to_numeric_safe(denom)
    return n / (d.replace(0, np.nan) + EPSILON)


def range_polarity(px_last: pd.Series, px_low: pd.Series, px_high: pd.Series) -> pd.Series:
    """
    Position of close within the daily range, normalized to [-1, 1]-ish:
        (close - mid) / (high - low)
    """
    close = to_numeric_safe(px_last)
    low = to_numeric_safe(px_low)
    high = to_numeric_safe(px_high)
    rng = (high - low)
    mid = (high + low) / 2.0
    return (close - mid) / (rng + EPSILON)
