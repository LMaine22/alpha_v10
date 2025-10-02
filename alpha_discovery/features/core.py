import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.stats import entropy
from collections import Counter

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
# Robust helpers & clipping
# =========================
def winsorize_robust(series: pd.Series, lower_pct: float = 0.005, upper_pct: float = 0.995) -> pd.Series:
    """Uniform winsorize to avoid rank explosions."""
    s = to_numeric_safe(series)
    lower = s.quantile(lower_pct)
    upper = s.quantile(upper_pct)
    return s.clip(lower=lower, upper=upper)


def huber_z(series: pd.Series, window: int = 60, k: float = 1.345) -> pd.Series:
    """Huberized z-score using M-estimator for mean/std."""
    s = to_numeric_safe(series)
    
    def huber_mean_std(x):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 3:
            return np.nan, np.nan
        
        # Initial estimates
        med = np.median(x_clean)
        mad = np.median(np.abs(x_clean - med))
        
        if mad == 0:
            return med, np.std(x_clean)
        
        # Huber weights
        u = (x_clean - med) / (k * mad + EPSILON)
        w = np.where(np.abs(u) <= 1, 1.0, 1.0 / np.abs(u))
        
        # Weighted mean
        huber_mean = np.sum(w * x_clean) / np.sum(w)
        
        # Robust std
        huber_std = mad * 1.4826
        
        return huber_mean, huber_std
    
    out = pd.Series(index=s.index, dtype=float)
    for i in range(window - 1, len(s)):
        window_data = s.iloc[i - window + 1:i + 1].values
        m, sd = huber_mean_std(window_data)
        if pd.notna(m) and pd.notna(sd) and sd > 0:
            out.iloc[i] = (s.iloc[i] - m) / sd
    
    return out


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


def acceleration(series: pd.Series, k: int = 1) -> pd.Series:
    """Standard second difference (fixed semantics)."""
    s = to_numeric_safe(series)
    if k == 1:
        return s.diff().diff()  # Standard second difference
    else:
        # For backward compatibility, keep k-lagged version but with clear naming
        return s.diff().diff(k)


def second_diff_k(series: pd.Series, k: int) -> pd.Series:
    """K-lagged second difference (explicit name)."""
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
# Advanced Volatility Estimators
# =========================
def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson volatility estimator (HL-only)."""
    H = to_numeric_safe(high).clip(lower=1e-12)
    L = to_numeric_safe(low).clip(lower=1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = (np.log(H / L) ** 2) / (4 * np.log(2))
    return rs.rolling(window).mean().pow(0.5) * np.sqrt(252)


def rogers_satchell_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """Rogers-Satchell volatility (OHLC directional)."""
    O = to_numeric_safe(open).clip(lower=1e-12)
    H = to_numeric_safe(high).clip(lower=1e-12)
    L = to_numeric_safe(low).clip(lower=1e-12)
    C = to_numeric_safe(close).clip(lower=1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
    return rs.rolling(window).mean().clip(lower=0).pow(0.5) * np.sqrt(252)


def yang_zhang_vol(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21, k: float = 0.34) -> pd.Series:
    """Yang-Zhang volatility (gap-aware)."""
    O = to_numeric_safe(open).clip(lower=1e-12)
    H = to_numeric_safe(high).clip(lower=1e-12)
    L = to_numeric_safe(low).clip(lower=1e-12)
    C = to_numeric_safe(close).clip(lower=1e-12)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Overnight component
        oc = np.log(O / C.shift(1))
        oc_var = oc.rolling(window).var()
        
        # Open-to-close component
        co = np.log(C / O)
        co_var = co.rolling(window).var()
        
        # Rogers-Satchell component
        rs = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
        rs_var = rs.rolling(window).mean()
    
    # Yang-Zhang combination
    yz_var = oc_var + k * co_var + (1 - k) * rs_var
    return yz_var.clip(lower=0).pow(0.5) * np.sqrt(252)


def vol_regime_flags(vol_series: pd.Series, window: int = 21, z_window: int = 60) -> Tuple[pd.Series, pd.Series]:
    """Vol regime flags for high vol and vol crush."""
    v = to_numeric_safe(vol_series)
    z = zscore_rolling(v, z_window)
    
    vol_regime_hi = (z > 1.5).astype(float)
    
    # For vol crush, we need IV vs RV comparison - simplified as rapid vol decline
    vol_change = v.diff(window)
    vol_change_z = zscore_rolling(vol_change, z_window)
    vol_regime_crush = (vol_change_z < -1.5).astype(float)
    
    return vol_regime_hi, vol_regime_crush


# =========================
# Tail & Distribution Shape
# =========================
def expected_shortfall(returns: pd.Series, window: int = 252, alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
    """Rolling expected shortfall (ES/CVaR) for left and right tails."""
    r = to_numeric_safe(returns)
    
    def calc_es(x, a):
        if len(x[~np.isnan(x)]) < 10:
            return np.nan
        q = np.quantile(x[~np.isnan(x)], a)
        tail = x[x <= q]
        return tail.mean() if len(tail) > 0 else np.nan
    
    es_left = r.rolling(window).apply(lambda x: calc_es(x, alpha), raw=True)
    es_right = r.rolling(window).apply(lambda x: -calc_es(-x, alpha), raw=True)
    
    return es_left, es_right


def partial_moments(returns: pd.Series, window: int = 63, threshold: float = 0.0) -> Dict[str, pd.Series]:
    """Lower and Upper Partial Moments (k=1,2,3)."""
    r = to_numeric_safe(returns)
    
    lpm = {}
    hpm = {}
    
    for k in [1, 2, 3]:
        # Lower partial moments
        lower_dev = (threshold - r).clip(lower=0)
        lpm[f'lpm_{k}'] = (lower_dev ** k).rolling(window).mean()
        
        # Upper partial moments  
        upper_dev = (r - threshold).clip(lower=0)
        hpm[f'hpm_{k}'] = (upper_dev ** k).rolling(window).mean()
    
    return {**lpm, **hpm}


# =========================
# Complexity & Memory
# =========================
def permutation_entropy(returns: pd.Series, window: int = 63, order: int = 3) -> pd.Series:
    """Rolling permutation entropy on return windows."""
    r = to_numeric_safe(returns)
    
    def perm_entropy(x, m):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < m + 1:
            return np.nan
        
        # Create order patterns
        patterns = []
        for i in range(len(x_clean) - m + 1):
            pattern = tuple(np.argsort(x_clean[i:i + m]))
            patterns.append(pattern)
        
        # Calculate entropy
        counts = Counter(patterns)
        probs = np.array(list(counts.values())) / len(patterns)
        return entropy(probs, base=2)
    
    return r.rolling(window).apply(lambda x: perm_entropy(x, order), raw=True)


def lempel_ziv_complexity(returns: pd.Series, window: int = 126, n_bins: int = 3) -> pd.Series:
    """Lempel-Ziv complexity on discretized return windows."""
    r = to_numeric_safe(returns)
    
    def lz_complexity(x):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 10:
            return np.nan
        
        # Discretize to symbols
        bins = np.linspace(x_clean.min(), x_clean.max() + EPSILON, n_bins + 1)
        symbols = np.digitize(x_clean, bins) - 1
        s = ''.join(str(s) for s in symbols)
        
        # LZ complexity
        i, k, l = 0, 1, 1
        k_max = 1
        n = len(s)
        c = 1
        
        while k + l <= n:
            if s[i:i + l] not in s[k:k + l]:
                c += 1
                i = k + l
                l = 1
                k = 1
                k_max = max(k, k_max)
            else:
                l += 1
            k += 1
        
        # Normalize
        return c / (n / np.log2(n_bins + 1)) if n > 0 else np.nan
    
    return r.rolling(window).apply(lz_complexity, raw=True)


def hurst_exponent(returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling Hurst exponent for long-memory detection."""
    r = to_numeric_safe(returns)
    
    def calc_hurst(x):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 20:
            return np.nan
        
        # R/S analysis
        lags = range(2, min(20, len(x_clean) // 2))
        tau = []
        
        for lag in lags:
            n_chunks = len(x_clean) // lag
            if n_chunks < 2:
                continue
            
            chunks = x_clean[:n_chunks * lag].reshape(n_chunks, lag)
            
            # Calculate R/S for each chunk
            rs_values = []
            for chunk in chunks:
                mean = chunk.mean()
                y = chunk - mean
                z = np.cumsum(y)
                R = z.max() - z.min()
                S = chunk.std()
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.mean(rs_values))
        
        if len(tau) < 2:
            return np.nan
        
        # Fit log-log regression
        poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        return poly[0]
    
    return r.rolling(window).apply(calc_hurst, raw=True)


def dfa_alpha(returns: pd.Series, window: int = 252) -> pd.Series:
    """Detrended Fluctuation Analysis exponent."""
    r = to_numeric_safe(returns)
    
    def calc_dfa(x):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 20:
            return np.nan
        
        # Cumulative sum
        y = np.cumsum(x_clean - x_clean.mean())
        
        # Scales
        scales = np.logspace(0.5, np.log10(len(x_clean) // 4), num=10, dtype=int)
        fluct = []
        
        for scale in scales:
            n_segments = len(y) // scale
            if n_segments < 1:
                continue
            
            # Detrend each segment
            f2 = []
            for i in range(n_segments):
                segment = y[i * scale:(i + 1) * scale]
                x_seg = np.arange(len(segment))
                coeffs = np.polyfit(x_seg, segment, 1)
                fit = np.polyval(coeffs, x_seg)
                f2.append(np.mean((segment - fit) ** 2))
            
            if f2:
                fluct.append(np.sqrt(np.mean(f2)))
        
        if len(fluct) < 2:
            return np.nan
        
        # Fit scaling
        valid_scales = scales[:len(fluct)]
        coeffs = np.polyfit(np.log(valid_scales), np.log(fluct), 1)
        return coeffs[0]
    
    return r.rolling(window).apply(calc_dfa, raw=True)


def state_persistence(returns: pd.Series, window: int = 63) -> Dict[str, pd.Series]:
    """Run-length encoding for sign persistence and entropy."""
    r = to_numeric_safe(returns)
    
    # Sign of returns
    signs = np.sign(r)
    
    # Run lengths
    def calc_run_stats(x):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 5:
            return np.nan, np.nan
        
        # Find runs
        runs = []
        current_run = 1
        
        for i in range(1, len(x_clean)):
            if x_clean[i] == x_clean[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Stats
        mean_run = np.mean(runs)
        
        # Entropy of run distribution
        run_counts = Counter(runs)
        probs = np.array(list(run_counts.values())) / len(runs)
        run_entropy = entropy(probs, base=2)
        
        return mean_run, run_entropy
    
    mean_run = pd.Series(index=r.index, dtype=float)
    run_entropy = pd.Series(index=r.index, dtype=float)
    
    for i in range(window - 1, len(r)):
        m, e = calc_run_stats(signs.iloc[i - window + 1:i + 1].values)
        mean_run.iloc[i] = m
        run_entropy.iloc[i] = e
    
    return {
        'state_mean_run_length': mean_run,
        'state_run_entropy': run_entropy
    }


# =========================
# Cross-asset structure & leverage
# =========================
def return_iv_correlation(returns: pd.Series, iv: pd.Series, window: int = 21) -> pd.Series:
    """Return-IV change correlation (leverage effect)."""
    r = to_numeric_safe(returns)
    iv_change = to_numeric_safe(iv).diff()
    return r.rolling(window).corr(iv_change)


def ewma_correlation(s1: pd.Series, s2: pd.Series, halflife: int = 21) -> pd.Series:
    """EWMA correlation."""
    r1 = to_numeric_safe(s1)
    r2 = to_numeric_safe(s2)
    
    # Center the series
    r1_centered = r1 - r1.ewm(halflife=halflife).mean()
    r2_centered = r2 - r2.ewm(halflife=halflife).mean()
    
    # EWMA covariance and variances
    cov = (r1_centered * r2_centered).ewm(halflife=halflife).mean()
    var1 = (r1_centered ** 2).ewm(halflife=halflife).mean()
    var2 = (r2_centered ** 2).ewm(halflife=halflife).mean()
    
    return cov / np.sqrt(var1 * var2 + EPSILON)


def dcc_lite(returns1: pd.Series, returns2: pd.Series, halflife: int = 21) -> pd.Series:
    """Simplified DCC-style dynamic correlation."""
    r1 = to_numeric_safe(returns1)
    r2 = to_numeric_safe(returns2)
    
    # Standardize returns
    r1_std = (r1 - r1.rolling(63).mean()) / (r1.rolling(63).std() + EPSILON)
    r2_std = (r2 - r2.rolling(63).mean()) / (r2.rolling(63).std() + EPSILON)
    
    # Dynamic correlation via EWMA
    q11 = (r1_std ** 2).ewm(halflife=halflife).mean()
    q22 = (r2_std ** 2).ewm(halflife=halflife).mean()
    q12 = (r1_std * r2_std).ewm(halflife=halflife).mean()
    
    return q12 / np.sqrt(q11 * q22 + EPSILON)


# =========================
# Lead-lag & Causality
# =========================
def max_leadlag_correlation(s1: pd.Series, s2: pd.Series, window: int = 21, max_lag: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Find max correlation within ±max_lag window."""
    r1 = to_numeric_safe(s1)
    r2 = to_numeric_safe(s2)
    
    best_corr = pd.Series(index=r1.index, dtype=float)
    best_lag = pd.Series(index=r1.index, dtype=float)
    
    for i in range(window + max_lag, len(r1)):
        corrs = []
        lags = []
        
        for lag in range(-max_lag, max_lag + 1):
            if i - window + 1 - lag >= 0 and i - lag < len(r2):
                s1_window = r1.iloc[i - window + 1:i + 1]
                s2_window = r2.iloc[i - window + 1 - lag:i + 1 - lag]
                
                if len(s1_window) == len(s2_window):
                    c = s1_window.corr(s2_window)
                    if pd.notna(c):
                        corrs.append(c)
                        lags.append(lag)
        
        if corrs:
            max_idx = np.argmax(np.abs(corrs))
            best_corr.iloc[i] = corrs[max_idx]
            best_lag.iloc[i] = lags[max_idx]
    
    return best_corr, best_lag


def cointegration_residual_z(s1: pd.Series, s2: pd.Series, window: int = 252) -> Tuple[pd.Series, pd.Series]:
    """Rolling cointegration residual z-score and half-life."""
    y = to_numeric_safe(s1)
    x = to_numeric_safe(s2)
    
    resid_z = pd.Series(index=y.index, dtype=float)
    half_life = pd.Series(index=y.index, dtype=float)
    
    for i in range(window - 1, len(y)):
        y_win = y.iloc[i - window + 1:i + 1].values
        x_win = x.iloc[i - window + 1:i + 1].values
        
        # Remove NaNs
        mask = ~(np.isnan(y_win) | np.isnan(x_win))
        if mask.sum() < window // 2:
            continue
        
        y_clean = y_win[mask]
        x_clean = x_win[mask]
        
        # OLS regression
        X = np.column_stack([np.ones(len(x_clean)), x_clean])
        try:
            beta = np.linalg.lstsq(X, y_clean, rcond=None)[0]
            
            # Residuals
            resid = y_clean - (beta[0] + beta[1] * x_clean)
            
            # Z-score of current residual
            current_resid = y.iloc[i] - (beta[0] + beta[1] * x.iloc[i])
            resid_z.iloc[i] = (current_resid - resid.mean()) / (resid.std() + EPSILON)
            
            # Half-life via AR(1)
            resid_lag = resid[:-1]
            resid_diff = resid[1:] - resid[:-1]
            
            if len(resid_lag) > 1:
                phi = np.corrcoef(resid_lag, resid[1:])[0, 1]
                if phi > 0 and phi < 1:
                    half_life.iloc[i] = -np.log(2) / np.log(phi)
        except:
            continue
    
    return resid_z, half_life


# =========================
# Fixed Corwin-Schultz
# =========================
def corwin_schultz_spread_fixed(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Fixed Corwin-Schultz spread estimator."""
    H = to_numeric_safe(high).clip(lower=1e-12)
    L = to_numeric_safe(low).clip(lower=1e-12)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Beta: sum of squared log ranges over 2 days
        beta = (np.log(H / L) ** 2).rolling(window=2).sum()
        
        # FIXED Gamma: use same-day H[t]/L[t] and previous day H[t-1]/L[t-1]
        gamma = np.log(H / L) * np.log(H.shift(1) / L.shift(1))
    
    # Alpha calculation
    alpha = (beta - gamma).clip(lower=0).pow(0.5)
    
    # Spread estimation
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    
    return spread.rolling(window).mean()


# =========================
# Correlation / beta (existing)
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
# Price/volatility stats (existing)
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
    r = px.pct_change(fill_method=None)
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
    """Original implementation - kept for backward compatibility."""
    return corwin_schultz_spread_fixed(high, low, window)


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
# Advanced Options Utilities
# =========================
def skew_dynamics(call_iv: pd.Series, put_iv: pd.Series, window: int = 21) -> pd.Series:
    """Risk reversal dynamics"""
    c = to_numeric_safe(call_iv)
    p = to_numeric_safe(put_iv)
    return (c - p).rolling(window).mean()

def flow_pressure(call_oi: pd.Series, put_oi: pd.Series, window: int = 5) -> pd.Series:
    """OI skew change pressure"""
    c = to_numeric_safe(call_oi)
    p = to_numeric_safe(put_oi)
    skew = c / (p + EPSILON)
    return skew.pct_change(window, fill_method=None)

def congestion_distance(price: pd.Series, oi_peak: pd.Series, window: int = 21) -> pd.Series:
    """Distance to OI congestion peaks"""
    px = to_numeric_safe(price)
    oi = to_numeric_safe(oi_peak)
    oi_rank = oi.rolling(window).rank(pct=True)
    return (px - px.rolling(window).mean()) / (px.rolling(window).std() + EPSILON) * (1 - oi_rank)

def iv_term_structure_slope(short_iv: pd.Series, long_iv: pd.Series) -> pd.Series:
    """IV term structure slope"""
    s = to_numeric_safe(short_iv)
    l = to_numeric_safe(long_iv)
    return (l - s) / (s + EPSILON)

# =========================
# Advanced Sentiment Utilities  
# =========================
def sentiment_consensus(sentiment_sources: List[pd.Series], weights: Optional[List[float]] = None) -> pd.Series:
    """Cross-source sentiment consensus"""
    if weights is None:
        weights = [1.0] * len(sentiment_sources)
    
    consensus = pd.Series(index=sentiment_sources[0].index, dtype=float)
    total_weight = 0
    
    for i, (sent, w) in enumerate(zip(sentiment_sources, weights)):
        s = to_numeric_safe(sent)
        consensus += s * w
        total_weight += w
    
    return consensus / (total_weight + EPSILON)

def sentiment_burst_detection(sentiment: pd.Series, window: int = 21, threshold: float = 2.0) -> pd.Series:
    """Detect sentiment bursts using z-score"""
    s = to_numeric_safe(sentiment)
    rolling_mean = s.rolling(window).mean()
    rolling_std = s.rolling(window).std()
    z_score = (s - rolling_mean) / (rolling_std + EPSILON)
    return (z_score.abs() > threshold).astype(float)

def sentiment_roc(sentiment: pd.Series, window: int = 5) -> pd.Series:
    """Rate of change in sentiment"""
    s = to_numeric_safe(sentiment)
    return s.pct_change(window, fill_method=None)

# =========================
# Advanced Macro Regime Utilities
# =========================
def trend_filter(series: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.Series:
    """Trend filter using moving averages"""
    s = to_numeric_safe(series)
    short_ma = s.rolling(short_window).mean()
    long_ma = s.rolling(long_window).mean()
    return (short_ma > long_ma).astype(float)

def momentum_regime(returns: pd.Series, window: int = 21) -> pd.Series:
    """Momentum regime detection"""
    r = to_numeric_safe(returns)
    rolling_mean = r.rolling(window).mean()
    rolling_std = r.rolling(window).std()
    return (rolling_mean / (rolling_std + EPSILON)).rolling(window).mean()

def beta_conditioned_return(asset_ret: pd.Series, market_ret: pd.Series, window: int = 63) -> pd.Series:
    """Beta-conditioned orthogonalized returns"""
    a, m = align_series(asset_ret, market_ret)
    beta = rolling_beta(a, m, window)
    return a - beta * m

def dollar_momentum(dxy: pd.Series, window: int = 21) -> pd.Series:
    """DXY momentum for macro regime"""
    s = to_numeric_safe(dxy)
    return s.pct_change(window, fill_method=None)

def rates_trend(yield_series: pd.Series, window: int = 21) -> pd.Series:
    """Interest rates trend"""
    s = to_numeric_safe(yield_series)
    return s.diff(window)

# =========================
# Advanced Sentiment Analysis
# =========================
def sentiment_regime_classifier(sentiment: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Classify sentiment regimes based on intensity and volume."""
    s = to_numeric_safe(sentiment)
    v = to_numeric_safe(volume)
    
    # Sentiment z-score
    s_z = (s - s.rolling(window).mean()) / (s.rolling(window).std() + EPSILON)
    
    # Volume z-score  
    v_z = (v - v.rolling(window).mean()) / (v.rolling(window).std() + EPSILON)
    
    # Regime classification
    regime = pd.Series(0, index=s.index, dtype=float)  # neutral
    regime[(s_z > 1) & (v_z > 1)] = 2  # euphoria
    regime[(s_z < -1) & (v_z > 1)] = -2  # panic
    regime[(s_z > 1) & (v_z <= 1)] = 1  # optimism
    regime[(s_z < -1) & (v_z <= 1)] = -1  # pessimism
    
    return regime

def news_flow_momentum(news_count: pd.Series, sentiment: pd.Series, window: int = 5) -> pd.Series:
    """Measure news flow momentum combining count and sentiment."""
    count = to_numeric_safe(news_count)
    sent = to_numeric_safe(sentiment)
    
    # Weighted momentum (count-weighted sentiment change)
    sent_change = sent.diff(window)
    count_weight = count / (count.rolling(window * 2).mean() + EPSILON)
    
    return sent_change * count_weight

def social_consensus_divergence(sources: List[pd.Series], window: int = 10) -> pd.Series:
    """Measure divergence across social sentiment sources."""
    if len(sources) < 2:
        return pd.Series(dtype=float)
    
    # Align all sources
    df = pd.DataFrame({f'source_{i}': s for i, s in enumerate(sources)}).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    
    # Rolling correlation matrix
    def _divergence(window_data):
        corr_mat = window_data.corr()
        # Average pairwise correlation as consensus measure
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        avg_corr = corr_mat.values[mask].mean()
        return 1 - avg_corr  # Higher divergence = lower correlation
    
    return df.rolling(window).apply(_divergence, raw=False).mean(axis=1)

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


# =========================
# Portfolio Risk Management
# =========================
def portfolio_var(returns_matrix: pd.DataFrame, weights: pd.Series, window: int = 252, alpha: float = 0.05) -> pd.Series:
    """Rolling portfolio VaR calculation."""
    def _calc_var(ret_window):
        if len(ret_window) < 20:
            return np.nan
        port_ret = (ret_window * weights.reindex(ret_window.columns, fill_value=0)).sum(axis=1)
        return port_ret.quantile(alpha)
    
    return returns_matrix.rolling(window).apply(_calc_var, raw=False)

def regime_stability_score(feature_matrix: pd.DataFrame, window: int = 63) -> pd.Series:
    """Measure how stable current regime is based on feature correlations."""
    def _stability(feat_window):
        corr_mat = feat_window.corr()
        # Eigenvalue concentration as stability proxy
        eigenvals = np.linalg.eigvals(corr_mat.fillna(0))
        eigenvals = eigenvals[eigenvals > 0]
        if len(eigenvals) == 0:
            return np.nan
        return eigenvals.max() / eigenvals.sum()
    
    # Sample subset of features to avoid computational explosion
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns[:50]
    sample_features = feature_matrix[numeric_cols]
    
    return sample_features.rolling(window).apply(_stability, raw=False).mean(axis=1)

def cross_asset_stress_indicator(returns_dict: Dict[str, pd.Series], window: int = 21) -> pd.Series:
    """Detect cross-asset stress periods."""
    if len(returns_dict) < 2:
        return pd.Series(dtype=float)
    
    # Get aligned returns
    ret_df = pd.DataFrame(returns_dict).dropna()
    if ret_df.empty:
        return pd.Series(dtype=float)
    
    # Tail correlation during stress
    def _stress_corr(window_data):
        # Focus on 5% tail events
        tail_threshold = window_data.quantile(0.05, axis=0)
        stress_periods = (window_data <= tail_threshold).any(axis=1)
        if stress_periods.sum() < 3:
            return np.nan
        stress_data = window_data[stress_periods]
        return stress_data.corr().values[np.triu_indices_from(stress_data.corr().values, k=1)].mean()
    
    return ret_df.rolling(window).apply(_stress_corr, raw=False).mean(axis=1)


# =========================
# Advanced Microstructure
# =========================
def order_flow_imbalance(buy_volume: pd.Series, sell_volume: pd.Series, window: int = 20) -> pd.Series:
    """Calculate order flow imbalance with smoothing."""
    buy = to_numeric_safe(buy_volume)
    sell = to_numeric_safe(sell_volume)
    
    # Raw imbalance
    imbalance = (buy - sell) / (buy + sell + EPSILON)
    
    # Smooth with EWMA
    return imbalance.ewm(span=window).mean()

def market_impact_estimate(returns: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Estimate market impact per unit volume."""
    ret = to_numeric_safe(returns).abs()
    vol = to_numeric_safe(volume)
    
    # Impact = |return| / sqrt(volume) (Almgren style)
    impact = ret / (vol.clip(lower=1).pow(0.5))
    
    return impact.rolling(window).mean()

def liquidity_risk_premium(bid_ask_spread: pd.Series, volatility: pd.Series, window: int = 63) -> pd.Series:
    """Calculate liquidity risk premium."""
    spread = to_numeric_safe(bid_ask_spread)
    vol = to_numeric_safe(volatility)
    
    # Risk-adjusted spread
    risk_adj_spread = spread / (vol + EPSILON)
    
    # Z-score relative to history
    return zscore_rolling(risk_adj_spread, window)

def price_pressure_indicator(price: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    """Detect price pressure from volume patterns."""
    px = to_numeric_safe(price)
    vol = to_numeric_safe(volume)
    
    ret = px.pct_change()
    vol_z = zscore_rolling(vol, window * 2)
    
    # High volume + directional price = pressure
    pressure = ret.rolling(window).mean() * vol_z.rolling(window).mean()
    
    return pressure


# Convenience alias
safe_divide = safe_div