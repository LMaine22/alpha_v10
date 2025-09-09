import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional

# Suppress specific warnings that are expected and handled
warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='overflow encountered in exp', category=RuntimeWarning)

# --- numerics ---
EPSILON = 1e-9


# =========================
# Common helpers
# =========================
def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    n = to_numeric_safe(numerator)
    d = to_numeric_safe(denominator)
    d = d.where(d.abs() > EPSILON)
    return n / d


# =========================
# Normalizations
# =========================
def zscore_rolling(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    s = to_numeric_safe(series)
    
    # Suppress All-NaN slice warnings for rolling operations
    with pd.option_context('mode.chained_assignment', None):
        m = s.rolling(window, min_periods=min_periods).mean()
        sd = s.rolling(window, min_periods=min_periods).std()
    
    # Handle cases where std is NaN or 0
    safe_sd = sd.fillna(0) + EPSILON
    result = (s - m) / safe_sd
    
    # Return NaN where the original std was NaN (insufficient data)
    return result.where(sd.notna(), np.nan)


def mad_z_rolling(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    s = to_numeric_safe(series)
    c = 1.4826
    
    # Suppress All-NaN slice warnings for rolling operations
    with pd.option_context('mode.chained_assignment', None):
        med = s.rolling(window, min_periods=min_periods).median()
        mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    
    # Handle cases where mad is NaN or 0
    safe_mad = mad.fillna(0) + EPSILON
    result = (s - med) / (c * safe_mad)
    
    # Return NaN where the original mad was NaN (insufficient data)
    return result.where(mad.notna(), np.nan)


def robust_z(series: pd.Series) -> pd.Series:
    s = to_numeric_safe(series)
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad + EPSILON)


# =========================
# Time-series transforms
# =========================
def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    return to_numeric_safe(series).ffill().pct_change(n, fill_method=None)


def diff_n(series: pd.Series, n: int) -> pd.Series:
    return to_numeric_safe(series).diff(n)


def rolling_mean(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    return to_numeric_safe(series).rolling(window, min_periods=min_periods).mean()


def rolling_std(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(1, window // 2)
    return to_numeric_safe(series).rolling(window, min_periods=min_periods).std()


def momentum_trend(series: pd.Series, short_win: int, long_win: int) -> pd.Series:
    s = to_numeric_safe(series)
    return rolling_mean(s, short_win) - rolling_mean(s, long_win)


def acceleration(series: pd.Series, k: int) -> pd.Series:
    s = to_numeric_safe(series)
    return s.diff().diff(k)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    s = to_numeric_safe(series)
    d = s.diff()
    up = d.where(d > 0, 0).rolling(window).mean()
    dn = (-d.where(d < 0, 0)).rolling(window).mean()
    rs = up / (dn + EPSILON)
    return 100 - (100 / (1 + rs))


# =========================
# Correlation / beta
# =========================
def align_series(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
    s1 = to_numeric_safe(s1)
    s2 = to_numeric_safe(s2)
    df = pd.concat([s1, s2], axis=1, join="inner").dropna()
    if df.empty:
        idx = s1.index.union(s2.index)
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)
    return df.iloc[:, 0], df.iloc[:, 1]


def rolling_corr_fisher(s1: pd.Series, s2: pd.Series, window: int, min_periods: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
    if min_periods is None:
        min_periods = max(2, window // 2)
    a, b = align_series(s1, s2)
    if a.empty:
        idx = s1.index.union(s2.index)
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty
    corr = a.rolling(window, min_periods=min_periods).corr(b).clip(-0.9999, 0.9999)
    fisher = 0.5 * np.log((1 + corr) / (1 - corr))
    return corr, fisher


def rolling_beta(s1: pd.Series, s2: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = max(2, window // 2)
    a, b = align_series(s1, s2)
    cov = a.rolling(window, min_periods=min_periods).cov(b)
    var = b.rolling(window, min_periods=min_periods).var()
    return cov / (var + EPSILON)


def beta_hedged_return(ret: pd.Series, bench_ret: pd.Series, window: int = 63) -> pd.Series:
    r, rb = align_series(ret, bench_ret)
    beta = rolling_beta(r, rb, window)
    return r - beta * rb


# =========================
# Price/volatility stats
# =========================
def realized_vol(price_series: pd.Series, window: int = 21) -> pd.Series:
    px = to_numeric_safe(price_series).astype(float).clip(lower=1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log(px).diff()
    return lr.rolling(window).std() * np.sqrt(252)


def garman_klass_vol(px_open: pd.Series, px_high: pd.Series, px_low: pd.Series, px_close: pd.Series, window: int = 21) -> pd.Series:
    O = to_numeric_safe(px_open).clip(lower=1e-12)
    H = to_numeric_safe(px_high).clip(lower=1e-12)
    L = to_numeric_safe(px_low ).clip(lower=1e-12)
    C = to_numeric_safe(px_close).clip(lower=1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = 0.5 * (np.log(H / L) ** 2) - (2 * np.log(2) - 1) * (np.log(C / O) ** 2)
    ann = rs.rolling(window).sum().clip(lower=0).pow(0.5) * np.sqrt(252)
    return ann


def vol_of_vol(vol_series: pd.Series, window: int = 21) -> pd.Series:
    v = to_numeric_safe(vol_series)
    # avoid default fill_method deprecation
    dv = v.pct_change(fill_method=None)
    return dv.rolling(window, min_periods=max(2, window // 2)).std()


def up_down_vol_ratio(returns: pd.Series, window: int = 21) -> pd.Series:
    r = to_numeric_safe(returns)
    up = r.where(r > 0, 0.0).rolling(window).std()
    dn = (-r.where(r < 0, 0.0)).rolling(window).std()
    return safe_div(up, dn)


def realized_skew_kurt(returns: pd.Series, window: int = 63) -> Tuple[pd.Series, pd.Series]:
    r = to_numeric_safe(returns)
    return r.rolling(window).skew(), r.rolling(window).kurt()


def tail_probs(returns: pd.Series, window: int = 252, alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
    r = to_numeric_safe(returns)
    q_l = r.rolling(window).quantile(alpha)
    q_r = r.rolling(window).quantile(1 - alpha)
    left = (r <= q_l).rolling(window).mean()
    right = (r >= q_r).rolling(window).mean()
    return left, right


def drawdown_from_high(price_series: pd.Series, window: int = 63) -> pd.Series:
    s = to_numeric_safe(price_series)
    roll_max = s.rolling(window).max()
    return s / (roll_max + EPSILON) - 1.0


def time_in_drawdown(price_series: pd.Series, window: int = 63) -> pd.Series:
    px = to_numeric_safe(price_series).ffill()
    r = px.pct_change(fill_method=None)  # avoid deprecation
    eq = (1 + r.fillna(0)).cumprod()
    run_max = eq.cummax()
    dd = eq / (run_max + EPSILON) - 1.0
    return (dd < 0).astype(float).rolling(window, min_periods=max(2, window // 2)).mean()


# =========================
# Trend / breakouts
# =========================
def log_trend_slope(series: pd.Series, window: int = 63) -> pd.Series:
    s = to_numeric_safe(series).clip(lower=1e-12)
    y = np.log(s)
    x = np.arange(window)
    out = np.full(len(y), np.nan)
    for i in range(window - 1, len(y)):
        yi = y.iloc[i - window + 1 : i + 1].values
        mask = np.isfinite(yi)
        if mask.sum() >= max(5, window // 2):
            out[i] = np.polyfit(x[mask], yi[mask], 1)[0]
    return pd.Series(out, index=y.index)


def ma_gap(series: pd.Series, short_window: int = 50, long_window: int = 200) -> pd.Series:
    s = to_numeric_safe(series)
    short_ma = s.rolling(short_window).mean()
    long_ma  = s.rolling(long_window).mean()
    return (short_ma - long_ma) / (long_ma + EPSILON)


def crossover_flags(series: pd.Series, short_window: int = 50, long_window: int = 200) -> Tuple[pd.Series, pd.Series]:
    """Returns (golden_cross_flag, death_cross_flag)."""
    s = to_numeric_safe(series)
    sma = s.rolling(short_window).mean()
    lma = s.rolling(long_window).mean()

    # ensure a consistent boolean dtype and avoid .fillna on bools
    above = (sma > lma).astype("boolean")
    prev  = above.shift(1, fill_value=False)

    golden = (above & ~prev).astype(float)
    death  = (~above & prev).astype(float)
    return golden, death


def breakout_flags(series: pd.Series, window: int = 126) -> Tuple[pd.Series, pd.Series]:
    """Returns (breakout_high, breakout_low) vs prior-window extremes (exclude today)."""
    s = to_numeric_safe(series)
    prior_max = s.shift(1).rolling(window).max()
    prior_min = s.shift(1).rolling(window).min()
    brk_hi = (s >= prior_max).astype(float)
    brk_lo = (s <= prior_min).astype(float)
    return brk_hi, brk_lo


def trend_persistence(returns: pd.Series, window: int = 21) -> pd.Series:
    r = to_numeric_safe(returns)
    return r.rolling(window).apply(lambda x: np.mean(x > 0), raw=True)


def stochastic_k(close: pd.Series, low: pd.Series, high: pd.Series, window: int = 14) -> pd.Series:
    C = to_numeric_safe(close)
    L = to_numeric_safe(low)
    H = to_numeric_safe(high)
    LL = L.rolling(window).min()
    HH = H.rolling(window).max()
    return 100 * (C - LL) / (HH - LL + EPSILON)


def close_to_range_pos(close: pd.Series, low: pd.Series, high: pd.Series, window: int = 63) -> pd.Series:
    C = to_numeric_safe(close)
    L = to_numeric_safe(low)
    H = to_numeric_safe(high)
    rmin = C.rolling(window).min().combine(L.rolling(window).min(), np.minimum)
    rmax = C.rolling(window).max().combine(H.rolling(window).max(), np.maximum)
    return (C - rmin) / (rmax - rmin + EPSILON)


# =========================
# Liquidity / microstructure
# =========================
def amihud_illiquidity(ret: pd.Series, dollar_volume: pd.Series, window: int) -> pd.Series:
    r = to_numeric_safe(ret).abs()
    dv = to_numeric_safe(dollar_volume)
    daily = r / (dv + EPSILON)
    return daily.rolling(window).mean()


def corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    H = to_numeric_safe(high).clip(lower=1e-12)
    L = to_numeric_safe(low).clip(lower=1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = (np.log(H / L) ** 2).rolling(window=2).sum()
        gamma = (np.log(H.shift(-1) / L) * np.log(H / L.shift(-1))).rolling(window=1).sum()
    alpha = (beta - gamma).clip(lower=0)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.rolling(window).mean()


def percentile(series: pd.Series, window: int = 252) -> pd.Series:
    s = to_numeric_safe(series)
    return s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def ewma_halflife(series: pd.Series, halflife: int, min_periods: Optional[int] = None) -> pd.Series:
    s = to_numeric_safe(series)
    return s.ewm(halflife=halflife, min_periods=min_periods).mean()


def relevance_weighted_ewma(values: pd.Series, weights: pd.Series, halflife: int) -> pd.Series:
    v = to_numeric_safe(values)
    w = to_numeric_safe(weights).clip(lower=0)
    w_norm = w / (w.rolling(5).max() + EPSILON)
    return (v * w_norm).ewm(halflife=halflife, min_periods=2).mean()


def safe_dollar_volume(df: pd.DataFrame) -> pd.Series:
    dv = pd.to_numeric(df.get("TURNOVER", pd.Series(index=df.index, dtype=float)), errors="coerce")
    fallback = (
        pd.to_numeric(df.get("PX_LAST",  pd.Series(index=df.index, dtype=float)), errors="coerce")
        * pd.to_numeric(df.get("PX_VOLUME", pd.Series(index=df.index, dtype=float)), errors="coerce")
    )
    return dv.where(dv > 0, fallback)


# =========================
# Missing functions for registry compatibility
# =========================
def semivariance(returns: pd.Series, window: int = 21, downside: bool = True) -> pd.Series:
    """Rolling semi-variance (downside by default)"""
    r = to_numeric_safe(returns)
    if downside:
        r_masked = r.where(r < 0, 0.0)
    else:
        r_masked = r.where(r > 0, 0.0)
    return (r_masked ** 2).rolling(window).mean()


def roll_spread_proxy(returns: pd.Series, window: int = 21) -> pd.Series:
    """Roll spread proxy for liquidity"""
    r = to_numeric_safe(returns)
    return r.abs().rolling(window).mean()


def corr_abs_to_abs(s1: pd.Series, s2: pd.Series, window: int = 21) -> pd.Series:
    """Correlation of absolute values"""
    a1 = to_numeric_safe(s1).abs()
    a2 = to_numeric_safe(s2).abs()
    return a1.rolling(window).corr(a2)


# Convenience alias used in some places
safe_divide = safe_div
