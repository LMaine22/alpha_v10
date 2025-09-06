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
    Rolling z-score: (x - mean) / std
    """
    if min_periods is None:
        min_periods = window // 2
    s = pd.to_numeric(series, errors="coerce")
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / (std + EPSILON)


def mad_z_rolling(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Rolling robust z using MAD (median absolute deviation).
    """
    if min_periods is None:
        min_periods = window // 2
    s = pd.to_numeric(series, errors="coerce")
    c = 1.4826
    median = s.rolling(window, min_periods=min_periods).median()
    mad = (s - median).abs().rolling(window, min_periods=min_periods).median()
    return (s - median) / (c * mad + EPSILON)


def robust_z(series: pd.Series) -> pd.Series:
    """
    Static (non-rolling) robust z using MAD.
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    c = 1.4826
    return (s - med) / (c * mad + EPSILON)


# ===================================
# Section 2: Relational Functions
# ===================================

def align_series(s1: pd.Series, s2: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Align two series on common index and drop any NaNs.
    """
    s1 = pd.to_numeric(s1, errors="coerce")
    s2 = pd.to_numeric(s2, errors="coerce")
    df = pd.concat([s1, s2], axis=1, join='inner').dropna()
    if df.empty:
        # Return empty aligned series with the union index to avoid KeyErrors downstream
        idx = s1.index.union(s2.index)
        return pd.Series(index=idx, dtype=float), pd.Series(index=idx, dtype=float)
    return df.iloc[:, 0], df.iloc[:, 1]


def rolling_corr_fisher(s1: pd.Series, s2: pd.Series, window: int, min_periods: int = None) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling correlation and Fisher transform (stabilizes correlation variance).
    Returns: (corr, fisher_z)
    """
    if min_periods is None:
        min_periods = window // 2
    s1_aligned, s2_aligned = align_series(s1, s2)
    if s1_aligned.empty or s2_aligned.empty:
        idx = s1.index.union(s2.index)
        empty = pd.Series(index=idx, dtype=float)
        return empty, empty
    corr = s1_aligned.rolling(window, min_periods=min_periods).corr(s2_aligned).clip(-0.9999, 0.9999)
    fisher_transform = 0.5 * np.log((1 + corr) / (1 - corr))
    return corr, fisher_transform


def rolling_beta(s1: pd.Series, s2: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Rolling beta of s1 w.r.t. s2 = Cov(s1, s2) / Var(s2)
    """
    if min_periods is None:
        min_periods = window // 2
    s1_aligned, s2_aligned = align_series(s1, s2)
    if s1_aligned.empty or s2_aligned.empty:
        return pd.Series(index=s1.index.union(s2.index), dtype=float)
    covariance = s1_aligned.rolling(window, min_periods=min_periods).cov(s2_aligned)
    variance = s2_aligned.rolling(window, min_periods=min_periods).var()
    return covariance / (variance + EPSILON)


# ===================================
# Section 3: Time Series Functions
# ===================================

def get_realized_vol(price_series: pd.Series, window: int = 21) -> pd.Series:
    """
    Annualized realized vol via std of daily log returns over a rolling window.
    """
    px = pd.to_numeric(price_series, errors="coerce").astype(float)
    log_px = np.log(px.clip(lower=1e-12))
    log_returns = log_px.diff()
    return log_returns.rolling(window).std() * np.sqrt(252)



def garman_klass_vol(px_open, px_high, px_low, px_close, window: int = 21) -> pd.Series:
    O = pd.to_numeric(px_open, errors="coerce").clip(lower=1e-12)
    H = pd.to_numeric(px_high, errors="coerce").clip(lower=1e-12)
    L = pd.to_numeric(px_low,  errors="coerce").clip(lower=1e-12)
    C = pd.to_numeric(px_close,errors="coerce").clip(lower=1e-12)
    rs = 0.5 * (np.log(H / L) ** 2) - (2 * np.log(2) - 1) * (np.log(C / O) ** 2)
    return rs.rolling(window).sum().clip(lower=0).pow(0.5) * np.sqrt(252)



def frac_diff(series: pd.Series, d: float, window: int = 100) -> pd.Series:
    """
    Fractional differentiation to help stationarity while retaining memory.
    """
    s = pd.to_numeric(series, errors="coerce")
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1])
    return s.rolling(window).apply(lambda x: np.dot(weights, x), raw=True)


# ===================================
# Section 4: Sentiment & Momentum Helpers
# ===================================

def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Coerce to numeric while preserving index."""
    return pd.to_numeric(series, errors="coerce")


def pct_change_n(series: pd.Series, n: int) -> pd.Series:
    s = to_numeric_safe(series)
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
# Section 5: Convenience & Risk Helpers
# ===================================

def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Elementwise safe division with EPSILON protection."""
    n = to_numeric_safe(numer)
    d = to_numeric_safe(denom)
    return n / (d.replace(0, np.nan) + EPSILON)


def range_polarity(px_last: pd.Series, px_low: pd.Series, px_high: pd.Series) -> pd.Series:
    """
    Close vs mid-range: (close - mid) / (high - low)
    """
    close = to_numeric_safe(px_last)
    low = to_numeric_safe(px_low)
    high = to_numeric_safe(px_high)
    rng = (high - low)
    mid = (high + low) / 2.0
    return (close - mid) / (rng + EPSILON)


def semivariance(returns: pd.Series, window: int, downside: bool = True) -> pd.Series:
    """
    Rolling semi-variance (downside by default).
    """
    r = to_numeric_safe(returns)
    mask = r < 0 if downside else r > 0
    r_masked = r.where(mask, 0.0)
    return (r_masked ** 2).rolling(window).mean()


def drawdown_from_high(series: pd.Series, window: int) -> pd.Series:
    """
    Drawdown from rolling max over window: (px / max) - 1
    """
    s = to_numeric_safe(series)
    roll_max = s.rolling(window).max()
    return (s / (roll_max + EPSILON)) - 1.0


def amihud_illiquidity(ret: pd.Series, dollar_volume: pd.Series, window: int) -> pd.Series:
    """
    Amihud illiquidity: mean(|ret| / $volume) over window.
    """
    r = to_numeric_safe(ret).abs()
    dv = to_numeric_safe(dollar_volume)
    daily = r / (dv + EPSILON)
    return daily.rolling(window).mean()


def corwin_schultz_spread(px_high: pd.Series, px_low: pd.Series, window: int = 21) -> pd.Series:
    """
    Corwin–Schultz high–low spread proxy (rolling mean).
    """
    H, L = to_numeric_safe(px_high), to_numeric_safe(px_low)
    beta = (np.log(H / L) ** 2).rolling(window=2).sum()
    gamma = (np.log(H.shift(-1) / L) * np.log(H / L.shift(-1))).rolling(window=1).sum()
    alpha = (beta - gamma).clip(lower=0)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.rolling(window).mean()


def percentile(series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling percentile rank (0..1) of the latest value in the trailing window.
    """
    s = to_numeric_safe(series)
    return s.rolling(window).apply(lambda x: (pd.Series(x).rank(pct=True)).iloc[-1], raw=False)


def ewma_halflife(series: pd.Series, halflife: int, min_periods: Optional[int] = None) -> pd.Series:
    """
    EWMA with halflife parameter.
    """
    s = to_numeric_safe(series)
    return s.ewm(halflife=halflife, min_periods=min_periods).mean()


def relevance_weighted_ewma(values: pd.Series, weights: pd.Series, halflife: int) -> pd.Series:
    """
    Relevance-weighted EWMA: normalize weights locally, then EWMA.
    """
    v = to_numeric_safe(values)
    w = to_numeric_safe(weights).clip(lower=0)
    w_norm = w / (w.rolling(5).max() + EPSILON)
    return (v * w_norm).ewm(halflife=halflife, min_periods=2).mean()


# ===================================
# Section 6: Project-Specific Convenience
# ===================================

REQUIRED_COLS = {
    "Dates","PX_LAST","DAY_TO_DAY_TOT_RETURN_NET_DVDS","VWAP_VOLUME","TURNOVER","VOLATILITY_90D",
    "BETA_ADJ_OVERRIDABLE","PUT_CALL_VOLUME_RATIO_CUR_DAY","TOT_OPT_VOLUME_CUR_DAY",
    "OPEN_INT_TOTAL_CALL","OPEN_INT_TOTAL_PUT","PX_VOLUME","PX_OPEN","IVOL_MONEYNESS",
    "TWITTER_POS_SENTIMENT_COUNT","TWITTER_PUBLICATION_COUNT","TWITTER_NEG_SENTIMENT_COUNT",
    "TWITTER_SENTIMENT_DAILY_AVG","TWITTER_NEUTRAL_SENTIMENT_CNT","NEWS_PUBLICATION_COUNT",
    "CHINESE_NEWS_SENTMNT_DAILY_AVG","NEWS_HEAT_READ_DMAX","3MO_CALL_IMP_VOL","3MO_PUT_IMP_VOL",
    "PX_ASK","PX_HIGH","PX_LOW"
}

def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def safe_dollar_volume(df: pd.DataFrame) -> pd.Series:
    """
    TURNOVER preferred; fallback = PX_LAST * PX_VOLUME when TURNOVER is 0/NaN.
    """
    dv = pd.to_numeric(df.get("TURNOVER", pd.Series(index=df.index, dtype=float)), errors="coerce")
    fallback = pd.to_numeric(df.get("PX_LAST", pd.Series(index=df.index, dtype=float)), errors="coerce") * \
               pd.to_numeric(df.get("PX_VOLUME", pd.Series(index=df.index, dtype=float)), errors="coerce")
    return dv.where(dv > 0, fallback)
