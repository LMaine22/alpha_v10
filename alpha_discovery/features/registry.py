# registry.py - Enhanced version with all improvements
import numpy as np
import pandas as pd
import os
from typing import Dict, Callable, List, Tuple, Set
from functools import lru_cache
import gc
from joblib import Parallel, delayed, Memory
import inspect
from itertools import count

# --- Pairwise perf helpers: footprint, caching, slimming, warnings ---
import os as _os2, hashlib, warnings, contextlib, time
from joblib import Memory as _Memory2

# persistent cache directory for pairwise outputs
_PAIRWISE_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache", "pairwise"))
os.makedirs(_PAIRWISE_CACHE_DIR, exist_ok=True)
pair_mem = _Memory2(_PAIRWISE_CACHE_DIR, verbose=0, compress=3)

def _df_footprint_bytes(df) -> int:
    try:
        return int(df.memory_usage(deep=True).sum())
    except Exception:
        return 0

def _hash_series_fast(s):
    """Stable 64-bit hash for a pandas Series of numeric values + index."""
    try:
        a = s.values.view("i8") if str(s.dtype).startswith("datetime") else s.values
        raw = a.tobytes() + s.index.view("i8").tobytes()
    except Exception:
        raw = repr(s).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]

def _data_hash_for_pair(df, t1: str, t2: str, ret_col: str = "ret"):
    """Hash only the minimal inputs for this pair (returns or price columns)."""
    cand1 = [f"{t1}.ret", f"{t1}|ret", f"{t1}_{ret_col}", (t1, ret_col), t1]
    cand2 = [f"{t2}.ret", f"{t2}|ret", f"{t2}_{ret_col}", (t2, ret_col), t2]
    s1, s2 = None, None
    for c in cand1:
        if c in getattr(df, 'columns', []):
            s1 = df[c]; break
    for c in cand2:
        if c in getattr(df, 'columns', []):
            s2 = df[c]; break
    if s1 is None or s2 is None:
        try:
            s1 = df[(t1, ret_col)]
            s2 = df[(t2, ret_col)]
        except Exception:
            return hashlib.sha1(df.index.view("i8").tobytes()).hexdigest()[:16]
    h = hashlib.sha1()
    h.update(_hash_series_fast(s1).encode("utf-8"))
    h.update(_hash_series_fast(s2).encode("utf-8"))
    return h.hexdigest()[:16]

@contextlib.contextmanager
def suppress_pairwise_runtime_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        yield

def _slim_view_for_pairwise(df, tickers: List[str], ret_col: str = "ret"):
    """Return a narrow DataFrame containing only what pairwise needs (returns or price proxies)."""
    cols = []
    for t in tickers:
        for pat in (f"{t}.ret", f"{t}|ret", f"{t}_{ret_col}", (t, ret_col), t):
            if pat in getattr(df, 'columns', []):
                cols.append(pat); break
    view = df[cols].copy() if cols else df.copy()
    return view

# --- Memory threshold alert + adaptive job scaling ---
def decide_pairwise_jobs(df_footprint_bytes: int,
                         default_jobs: int,
                         warn_mb: int = None) -> tuple[int, str]:
    """
    Return (n_jobs, note). If footprint exceeds threshold, reduce jobs.
    Env:
      FEATURES_PAIRWISE_WARN_MB (int, MB)  -> warn threshold (default 800)
      FEATURES_PAIRWISE_MAX_JOBS_ON_WARN   -> clamp jobs when threshold exceeded (default 4)
    """
    mb = df_footprint_bytes / 1e6
    warn_mb = warn_mb or int(os.environ.get("FEATURES_PAIRWISE_WARN_MB", "800"))
    clamp_jobs = int(os.environ.get("FEATURES_PAIRWISE_MAX_JOBS_ON_WARN", "4"))
    if mb >= warn_mb:
        print(f"  [pairwise][WARN] DataFrame footprint {mb:.1f} MB ≥ {warn_mb} MB; clamping jobs to {clamp_jobs}.")
        return max(1, clamp_jobs), "clamped"
    return max(1, default_jobs), "default"

# joblib-cached wrapper that avoids pickling the heavy DataFrame
@pair_mem.cache(ignore=["view_df", "pair_specs"])
def _compute_pairwise_features_cached(target: str, bench: str, spec_version: str, data_hash: str, view_df, pair_specs):
    return _compute_pairwise_features(target, bench, view_df, pair_specs)

from ..config import settings
from . import core as f
from ..data.events import build_event_features  # daily EV_* features from your events.py
from .complexity_engine import get_fast_complexity_engine, compute_complexity_feature
try:  # pairwise helper safe imports
    from .pairwise_utils import safe_rolling_corr, safe_rolling_cov, safe_zscore, safe_leadlag_corr  # type: ignore
except Exception:  # fallback no-op stubs if import resolution lags
    def safe_rolling_corr(a,b,window,min_periods):
        return a.rolling(window, min_periods=min_periods).corr(b)
    def safe_rolling_cov(a,b,window,min_periods):
        return a.rolling(window, min_periods=min_periods).cov(b)
    def safe_zscore(x,window,min_periods):
        r = x.rolling(window, min_periods=min_periods)
        mu = r.mean(); sd = r.std(ddof=1)
        return (x - mu) / sd.replace(0, np.nan)
    def safe_leadlag_corr(a,b,lags,window,min_periods):
        best_abs=-np.inf; best=None; best_lag=0
        for L in lags:
            s2=b.shift(L)
            corr=a.rolling(window, min_periods=min_periods).corr(s2)
            cmax=np.nanmax(np.abs(corr.values))
            if np.isfinite(cmax) and cmax>best_abs:
                best_abs=float(cmax); best=corr; best_lag=L
        return best_lag, best

# Setup caching with improved configuration
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cache', 'features')
os.makedirs(cache_dir, exist_ok=True)
# Note: Cannot use both compression and mmap_mode together - we prioritize compression
memory = Memory(location=cache_dir, verbose=0, compress=3)


# ----------------------
# Column helpers
# ----------------------
def _col(df: pd.DataFrame, tkr: str, field: str) -> pd.Series:
    name = f"{tkr}_{field}"
    return pd.to_numeric(df.get(name, pd.Series(index=df.index, dtype=float)), errors="coerce")


def _ret1d(df: pd.DataFrame, t: str) -> pd.Series:
    return f.pct_change_n(_col(df, t, "PX_LAST"), 1)


def _dvol(df: pd.DataFrame, t: str) -> pd.Series:
    sub = pd.DataFrame({
        "TURNOVER": _col(df, t, "TURNOVER"),
        "PX_LAST": _col(df, t, "PX_LAST"),
        "PX_VOLUME": _col(df, t, "PX_VOLUME"),
    })
    return f.safe_dollar_volume(sub)


def _pcr(df: pd.DataFrame, t: str) -> pd.Series:
    return _col(df, t, "PUT_CALL_VOLUME_RATIO_CUR_DAY")


def _oi_skew(df: pd.DataFrame, t: str) -> pd.Series:
    c = _col(df, t, "OPEN_INT_TOTAL_CALL")
    p = _col(df, t, "OPEN_INT_TOTAL_PUT")
    return f.safe_div(c - p, c + p)


def _options_available(df: pd.DataFrame, t: str) -> pd.Series:
    has = (
        _col(df, t, "3MO_CALL_IMP_VOL").notna()
        | _col(df, t, "PUT_IMP_VOL_30D").notna()
        | _col(df, t, "CALL_IMP_VOL_30D").notna()
    ).astype(float)
    return has


# ----------------------
# Universe - EXPANDED
# ----------------------
TICKS_ALL: List[str] = [
    # Original tickers
    "CRM US Equity","QQQ US Equity","SPY US Equity","TSLA US Equity","AMZN US Equity","GOOGL US Equity",
    "MSFT US Equity","AAPL US Equity","LLY US Equity","CRWV US Equity","JPM US Equity","C US Equity",
    "PLTR US Equity","ARM US Equity","AMD US Equity","BMY US Equity","PEP US Equity","NKE US Equity",
    "WMT US Equity","DXY Curncy","JPY Curncy","EUR Curncy","XAU Curncy","XLE US Equity","CL1 Comdty",
    "XLK US Equity","XLRE US Equity","XLC US Equity","XLV US Equity","MSTR US Equity","COIN US Equity",
    "BTC Index","XLY US Equity","EFA US Equity","MXWO Index","EEM US Equity","USGG10YR Index",
    "USGG2YR Index","XLP US Equity","SPX Index","NDX Index","RTY Index","HG1 Comdty",
    # New additions
    "NVDA US Equity","AVGO US Equity","SMCI US Equity","META US Equity","TSM US Equity",
    "TLT US Equity","VIX Index"
]

# Macro aliases
SPY = "SPY US Equity"
SPX = "SPX Index"
NDX = "NDX Index"
RTY = "RTY Index"
QQQ = "QQQ US Equity"
DXY = "DXY Curncy"
EUR = "EUR Curncy"
JPY = "JPY Curncy"
XAU = "XAU Curncy"
CL1 = "CL1 Comdty"
HG1 = "HG1 Comdty"
BTC = "BTC Index"
US10 = "USGG10YR Index"
US2  = "USGG2YR Index"
VIX = "VIX Index"
TLT = "TLT US Equity"

# Sectors
XLK = "XLK US Equity"; XLE = "XLE US Equity"; XLV = "XLV US Equity"; XLY = "XLY US Equity"; XLP = "XLP US Equity"
XLRE = "XLRE US Equity"; XLC = "XLC US Equity"

# Sector map for single names - EXPANDED
SECTOR_MAP: Dict[str, str] = {
    # Tech
    "AAPL US Equity": XLK, "MSFT US Equity": XLK, "GOOGL US Equity": XLK,
    "AMD US Equity": XLK, "ARM US Equity": XLK, "PLTR US Equity": XLK,
    "CRM US Equity": XLK, "CRWV US Equity": XLK,
    "NVDA US Equity": XLK, "AVGO US Equity": XLK, "SMCI US Equity": XLK,
    "TSM US Equity": XLK,
    # Consumer Discretionary
    "AMZN US Equity": XLY, "TSLA US Equity": XLY, "NKE US Equity": XLY,
    # Consumer Staples
    "WMT US Equity": XLP, "PEP US Equity": XLP,
    # Health Care
    "LLY US Equity": XLV, "BMY US Equity": XLV,
    # Financials (fallback to SPY — no XLF in universe)
    "JPM US Equity": SPY, "C US Equity": SPY,
    # Communication Services
    "META US Equity": XLC,
    # Crypto beta — use QQQ as peer proxy
    "MSTR US Equity": QQQ, "COIN US Equity": QQQ,
}


# ----------------------
# Macro features - EXPANDED
# ----------------------
MACRO_SPECS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Original features
    "macro.2s10s_slope":      lambda D: _col(D, US10, "PX_LAST") - _col(D, US2, "PX_LAST"),
    "macro.2s10s_steepen_21": lambda D: f.diff_n(_col(D, US10, "PX_LAST") - _col(D, US2, "PX_LAST"), 21),
    "macro.dxy_pctile_252":   lambda D: f.percentile(_col(D, DXY, "PX_LAST"), 252),
    "macro.hgxau_mom_63":     lambda D: f.pct_change_n(f.safe_div(_col(D, HG1, "PX_LAST"), _col(D, XAU, "PX_LAST")), 63),
    "macro.risk_on_score":    lambda D: (
        -f.zscore_rolling(f.pct_change_n(_col(D, DXY, "PX_LAST"), 1), 63)
        -f.zscore_rolling(f.pct_change_n(_col(D, US10, "PX_LAST"), 1), 63)
        +f.zscore_rolling(f.pct_change_n(_col(D, HG1, "PX_LAST"), 1), 63)
        +f.zscore_rolling(f.pct_change_n(_col(D, CL1, "PX_LAST"), 1), 63)
        -f.zscore_rolling(f.pct_change_n(_col(D, XAU, "PX_LAST"), 1), 63)
        +f.zscore_rolling(f.pct_change_n(_col(D, BTC, "PX_LAST"), 1), 63)
        +f.zscore_rolling(f.pct_change_n((_col(D, RTY, "PX_LAST") - _col(D, SPX, "PX_LAST")), 1), 63)
    ),
    "macro.rty_over_spx":     lambda D: f.safe_div(_col(D, RTY, "PX_LAST"), _col(D, SPX, "PX_LAST")),
    "macro.eem_over_mxwo":    lambda D: f.safe_div(_col(D, "EEM US Equity", "PX_LAST"), _col(D, "MXWO Index", "PX_LAST")),
    
    # New macro features
    "macro.vix_z60":          lambda D: f.zscore_rolling(_col(D, VIX, "PX_LAST"), 60),
    "macro.rate_duration_pivot": lambda D: f.zscore_rolling(
        f.pct_change_n(_col(D, TLT, "PX_LAST"), 21) - f.pct_change_n(_col(D, SPY if SPY in TICKS_ALL else SPX, "PX_LAST"), 21), 60
    ),
    "macro.btc_beta_shock":   lambda D: f.zscore_rolling(
        f.rolling_corr_fisher(
            f.pct_change_n(_col(D, BTC, "PX_LAST"), 1),
            f.pct_change_n(_col(D, QQQ, "PX_LAST"), 1),
            20
        )[1], 60
    ),
    "macro.btc_corr_delta":   lambda D: f.diff_n(
        f.rolling_corr_fisher(
            f.pct_change_n(_col(D, BTC, "PX_LAST"), 1),
            f.pct_change_n(_col(D, QQQ, "PX_LAST"), 1),
            20
        )[1], 5
    ),
}


# ----------------------
# Single-asset feature specs - EXPANDED
# ----------------------
FEAT: Dict[str, Callable[[pd.DataFrame, str], pd.Series]] = {
    # L1: Trend / Momentum / Reversal
    "px.mom_5":               lambda D,t: f.pct_change_n(_col(D,t,"PX_LAST"),5),
    "px.mom_21":              lambda D,t: f.pct_change_n(_col(D,t,"PX_LAST"),21),
    "px.mom_63":              lambda D,t: f.pct_change_n(_col(D,t,"PX_LAST"),63),
    "px.mom_126":             lambda D,t: f.pct_change_n(_col(D,t,"PX_LAST"),126),
    "px.trend_slope_63":      lambda D,t: f.log_trend_slope(_col(D,t,"PX_LAST"),63),
    "px.ma_gap_50_200_z":     lambda D,t: f.zscore_rolling(f.ma_gap(_col(D,t,"PX_LAST"),50,200),60),
    "px.golden_cross":        lambda D,t: f.crossover_flags(_col(D,t,"PX_LAST"),50,200)[0],
    "px.death_cross":         lambda D,t: f.crossover_flags(_col(D,t,"PX_LAST"),50,200)[1],
    "px.breakout_high_126":   lambda D,t: f.breakout_flags(_col(D,t,"PX_LAST"),126)[0],
    "px.breakout_low_126":    lambda D,t: f.breakout_flags(_col(D,t,"PX_LAST"),126)[1],
    "px.trend_persist_21":    lambda D,t: f.trend_persistence(_ret1d(D,t),21),
    "px.rsi14_z":             lambda D,t: f.zscore_rolling(f.rsi(_col(D,t,"PX_LAST"),14),60),
    "px.stoch14_z":           lambda D,t: f.zscore_rolling(f.stochastic_k(_col(D,t,"PX_LAST"),_col(D,t,"PX_LOW"),_col(D,t,"PX_HIGH"),14),60),
    "px.range_pos_63":        lambda D,t: f.close_to_range_pos(_col(D,t,"PX_LAST"),_col(D,t,"PX_LOW"),_col(D,t,"PX_HIGH"),63),

    # L2: Volatility / Tails / Drawdown - ENHANCED
    "vol.rv_21":              lambda D,t: f.realized_vol(_col(D,t,"PX_LAST"),21),
    "vol.rv_gk_21":           lambda D,t: f.garman_klass_vol(_col(D,t,"PX_OPEN"),_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),_col(D,t,"PX_LAST"),21),
    "vol.parkinson_21":       lambda D,t: f.parkinson_vol(_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),21),
    "vol.rogers_satchell_21": lambda D,t: f.rogers_satchell_vol(_col(D,t,"PX_OPEN"),_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),_col(D,t,"PX_LAST"),21),
    "vol.yang_zhang_21":      lambda D,t: f.yang_zhang_vol(_col(D,t,"PX_OPEN"),_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),_col(D,t,"PX_LAST"),21),
    "vol.regime_hi":          lambda D,t: f.vol_regime_flags(f.yang_zhang_vol(_col(D,t,"PX_OPEN"),_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),_col(D,t,"PX_LAST"),21))[0],
    "vol.regime_crush":       lambda D,t: f.vol_regime_flags(f.yang_zhang_vol(_col(D,t,"PX_OPEN"),_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),_col(D,t,"PX_LAST"),21))[1],
    "vol.vol_of_vol_21":      lambda D,t: f.vol_of_vol(f.realized_vol(_col(D,t,"PX_LAST"),21),21),
    "vol.semivar_dn_21":      lambda D,t: f.semivariance(_ret1d(D,t),21,True),
    "vol.updown_ratio_21":    lambda D,t: f.up_down_vol_ratio(_ret1d(D,t),21),
    "vol.skew_63":            lambda D,t: f.realized_skew_kurt(_ret1d(D,t),63)[0],
    "vol.kurt_63":            lambda D,t: f.realized_skew_kurt(_ret1d(D,t),63)[1],
    "vol.left_tail_pr_252":   lambda D,t: f.tail_probs(_ret1d(D,t),252,0.05)[0],
    "vol.right_tail_pr_252":  lambda D,t: f.tail_probs(_ret1d(D,t),252,0.05)[1],
    "vol.es_left_5pct":       lambda D,t: f.expected_shortfall(_ret1d(D,t),252,0.05)[0],
    "vol.es_right_5pct":      lambda D,t: f.expected_shortfall(_ret1d(D,t),252,0.05)[1],
    "vol.es_left_1pct":       lambda D,t: f.expected_shortfall(_ret1d(D,t),252,0.01)[0],
    "vol.es_right_1pct":      lambda D,t: f.expected_shortfall(_ret1d(D,t),252,0.01)[1],
    "vol.lpm_1":              lambda D,t: f.partial_moments(_ret1d(D,t),63)['lpm_1'],
    "vol.lpm_2":              lambda D,t: f.partial_moments(_ret1d(D,t),63)['lpm_2'],
    "vol.lpm_3":              lambda D,t: f.partial_moments(_ret1d(D,t),63)['lpm_3'],
    "vol.hpm_1":              lambda D,t: f.partial_moments(_ret1d(D,t),63)['hpm_1'],
    "vol.hpm_2":              lambda D,t: f.partial_moments(_ret1d(D,t),63)['hpm_2'],
    "vol.hpm_3":              lambda D,t: f.partial_moments(_ret1d(D,t),63)['hpm_3'],
    "risk.drawdown_63":       lambda D,t: f.drawdown_from_high(_col(D,t,"PX_LAST"),63),
    "risk.time_in_dd_63":     lambda D,t: f.time_in_drawdown(_col(D,t,"PX_LAST"),63),

    # L3: Liquidity / Microstructure - ENHANCED
    "liq.amihud_21":          lambda D,t: f.amihud_illiquidity(_ret1d(D,t), _dvol(D,t), 21),
    "liq.cs_spread_21":       lambda D,t: f.corwin_schultz_spread_fixed(_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),21),
    "liq.roll_proxy_21":      lambda D,t: f.roll_spread_proxy(_ret1d(D,t),21),
    "liq.turnover_z_63":      lambda D,t: f.zscore_rolling(_col(D,t,"TURNOVER"),63),

    # L4: Options / IV / Flow - ENHANCED
    "opt.iv30_z_60":          lambda D,t: f.zscore_rolling(_col(D,t,"CALL_IMP_VOL_30D"),60),
    "opt.iv3m_z_60":          lambda D,t: f.zscore_rolling(_col(D,t,"3MO_CALL_IMP_VOL"),60),
    "opt.term_30d_3m":        lambda D,t: _col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"3MO_CALL_IMP_VOL"),
    "opt.skew_30d":           lambda D,t: _col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"PUT_IMP_VOL_30D"),
    "opt.rr25_1m_proxy":      lambda D,t: _col(D,t,"1M_CALL_IMP_VOL_25DELTA_DFLT") - _col(D,t,"1M_PUT_IMP_VOL_10DELTA_DFLT"),
    "opt.fly_1m_40v25":       lambda D,t: 0.5*(_col(D,t,"1M_CALL_IMP_VOL_40DELTA_DFLT")+_col(D,t,"1M_PUT_IMP_VOL_40DELTA_DFLT")) - _col(D,t,"1M_CALL_IMP_VOL_25DELTA_DFLT"),
    "opt.smile_slope":        lambda D,t: (_col(D,t,"1M_CALL_IMP_VOL_25DELTA_DFLT") - _col(D,t,"1M_PUT_IMP_VOL_25DELTA_DFLT")),
    "opt.smile_curvature":    lambda D,t: (
        0.5*(_col(D,t,"1M_CALL_IMP_VOL_10DELTA_DFLT")+_col(D,t,"1M_PUT_IMP_VOL_10DELTA_DFLT")) 
        - 0.5*(_col(D,t,"1M_CALL_IMP_VOL_40DELTA_DFLT")+_col(D,t,"1M_PUT_IMP_VOL_40DELTA_DFLT"))
    ),
    "opt.iv_carry":           lambda D,t: _col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"3MO_CALL_IMP_VOL"),
    "opt.iv_carry_mom":       lambda D,t: f.diff_n(_col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"3MO_CALL_IMP_VOL"), 5),
    "opt.iv_carry_abs":       lambda D,t: (_col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"3MO_CALL_IMP_VOL")).abs(),
    "opt.iv_mom_5_z60":       lambda D,t: f.zscore_rolling(f.diff_n(_col(D,t,"3MO_CALL_IMP_VOL"),5),60),
    "opt.vrp_21":             lambda D,t: _col(D,t,"3MO_CALL_IMP_VOL") - f.realized_vol(_col(D,t,"PX_LAST"),21) * 100.0,
    "opt.rv_minus_iv_21":     lambda D,t: f.realized_vol(_col(D,t,"PX_LAST"),21) * 100.0 - _col(D,t,"3MO_CALL_IMP_VOL"),
    "opt.return_iv_corr":     lambda D,t: f.return_iv_correlation(_ret1d(D,t), _col(D,t,"3MO_CALL_IMP_VOL"), 21),
    "flow.pcr_ema5_z30":      lambda D,t: f.zscore_rolling(_pcr(df=D,t=t).ewm(span=5,min_periods=2).mean(),30),
    "flow.opt_vol_z30":       lambda D,t: f.zscore_rolling(_col(D,t,"TOT_OPT_VOLUME_CUR_DAY"),30),
    "pos.oi_call_put_ratio":  lambda D,t: f.safe_div(_col(D,t,"OPEN_INT_TOTAL_CALL"), _col(D,t,"OPEN_INT_TOTAL_PUT")),
    "pos.oi_skew_z60":        lambda D,t: f.zscore_rolling(_oi_skew(D,t),60),

    # L8: Advanced Options Features
    "opt.skew_dynamics_21":   lambda D,t: f.skew_dynamics(_col(D,t,"CALL_IMP_VOL_30D"), _col(D,t,"PUT_IMP_VOL_30D"), 21),
    "opt.flow_pressure_5":    lambda D,t: f.flow_pressure(_col(D,t,"OPEN_INT_TOTAL_CALL"), _col(D,t,"OPEN_INT_TOTAL_PUT"), 5),
    "opt.congestion_dist_21": lambda D,t: f.congestion_distance(_col(D,t,"PX_LAST"), _col(D,t,"OPEN_INT_TOTAL_CALL"), 21),
    "opt.iv_term_slope":      lambda D,t: f.iv_term_structure_slope(_col(D,t,"CALL_IMP_VOL_30D"), _col(D,t,"3MO_CALL_IMP_VOL")),

    # L5: Sentiment / Media
    "sent.tw_pub_z30":        lambda D,t: f.zscore_rolling(_col(D,t,"TWITTER_PUBLICATION_COUNT"),30),
    "sent.tw_avg_z30":        lambda D,t: f.zscore_rolling(_col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"),30),
    "sent.tw_mom_21":         lambda D,t: f.diff_n(_col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"),21),
    "sent.news_pub_z30":      lambda D,t: f.zscore_rolling(_col(D,t,"NEWS_PUBLICATION_COUNT"),30),
    "sent.news_heat_z21":     lambda D,t: f.zscore_rolling(_col(D,t,"NEWS_HEAT_READ_DMAX"),21),
    "sent.cn_avg_z60":        lambda D,t: f.zscore_rolling(_col(D,t,"CHINESE_NEWS_SENTMNT_DAILY_AVG"),60),

    # L9: Advanced Sentiment Features
    "sent.consensus_z30":     lambda D,t: f.zscore_rolling(f.sentiment_consensus([
        _col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"),
        _col(D,t,"CHINESE_NEWS_SENTMNT_DAILY_AVG"),
        _col(D,t,"NEWS_HEAT_READ_DMAX")
    ]), 30),
    "sent.burst_detect_21":   lambda D,t: f.sentiment_burst_detection(_col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"), 21),
    "sent.roc_5":             lambda D,t: f.sentiment_roc(_col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"), 5),

    # L6: Cross-asset β & spreads - ENHANCED
    "x.beta_spy_63":          lambda D,t: f.rolling_beta(_ret1d(D,t), _ret1d(D, SPY if SPY in TICKS_ALL else SPX), 63),
    "x.bh_ret_spy_21":        lambda D,t: f.beta_hedged_return(_ret1d(D,t), _ret1d(D, SPY if SPY in TICKS_ALL else SPX), 63).rolling(21).sum(),
    "x.spread_vs_sector_21":  lambda D,t: f.pct_change_n(_col(D,t,"PX_LAST"),21) - f.pct_change_n(_col(D, SECTOR_MAP.get(t, SPY), "PX_LAST"),21),
    "x.ewma_corr_21":         lambda D,t: f.ewma_correlation(_ret1d(D,t), _ret1d(D, SPY if SPY in TICKS_ALL else SPX), 21),
    "x.dcc_corr_21":          lambda D,t: f.dcc_lite(_ret1d(D,t), _ret1d(D, SPY if SPY in TICKS_ALL else SPX), 21),

    # L7: Regime & Complexity - FAST ENGINE (TEMPORARILY DISABLED FOR DEBUGGING)
    "reg.vol90d_z120":        lambda D,t: f.zscore_rolling(_col(D,t,"VOLATILITY_90D"),120),
    # Complexity features temporarily disabled to prevent hanging during parallel computation
    #"cmplx.perm_entropy_63":  lambda D,t: compute_complexity_feature(D, t, "cmplx.perm_entropy_63"),
    #"cmplx.perm_entropy_126": lambda D,t: compute_complexity_feature(D, t, "cmplx.perm_entropy_126"),
    #"cmplx.lz_complexity_63": lambda D,t: compute_complexity_feature(D, t, "cmplx.lz_complexity_63"),
    #"cmplx.lz_complexity_126":lambda D,t: compute_complexity_feature(D, t, "cmplx.lz_complexity_126"),
    #"cmplx.hurst_252":        lambda D,t: compute_complexity_feature(D, t, "cmplx.hurst_252"),
    #"cmplx.dfa_alpha_252":    lambda D,t: compute_complexity_feature(D, t, "cmplx.dfa_alpha_252"),
    #"cmplx.run_length_mean":  lambda D,t: compute_complexity_feature(D, t, "cmplx.run_length_mean"),
    #"cmplx.run_entropy":      lambda D,t: compute_complexity_feature(D, t, "cmplx.run_entropy"),

    # L10: Advanced Macro Regime Features
    "macro.trend_filter_20_50": lambda D,t: f.trend_filter(_col(D,t,"PX_LAST"), 20, 50),
    "macro.momentum_regime_21": lambda D,t: f.momentum_regime(_ret1d(D,t), 21),
    "macro.beta_conditioned_63": lambda D,t: f.beta_conditioned_return(_ret1d(D,t), _ret1d(D, SPY), 63),
}


# ----------------------
# Pairwise vs benchmark - ENHANCED
# ----------------------
def _pw_corr_fisher20_z60(D,a,b):
    return f.zscore_rolling(f.rolling_corr_fisher(_ret1d(D,a), _ret1d(D,b),20)[1],60)
def _pw_corr_delta_20_60_z60(D,a,b):
    return f.zscore_rolling(
        f.rolling_corr_fisher(_ret1d(D,a), _ret1d(D,b),20)[1] - f.rolling_corr_fisher(_ret1d(D,a), _ret1d(D,b),60)[1],60
    )
def _pw_beta_60_z120(D,a,b):
    return f.zscore_rolling(f.rolling_beta(_ret1d(D,a), _ret1d(D,b),60),120)
def _pw_corr_abs_21(D,a,b):
    return f.corr_abs_to_abs(_ret1d(D,a), _ret1d(D,b),21)
def _pw_partial_corr_21(D,a,b):
    return _partial_correlation(D,a,b,21)
def _pw_leadlag_best_5(D,a,b):
    return f.max_leadlag_correlation(_ret1d(D,a), _ret1d(D,b),21,5)
def _pw_coint_resid_z_252(D,a,b):
    return f.cointegration_residual_z(_col(D,a,"PX_LAST"), _col(D,b,"PX_LAST"),252)

PAIR_SPECS: Dict[str, Callable[[pd.DataFrame, str, str], pd.Series]] = {
    "x.corr_fisher20_z60": _pw_corr_fisher20_z60,
    "x.corr_delta_20_60_z60": _pw_corr_delta_20_60_z60,
    "x.beta_60_z120": _pw_beta_60_z120,
    "x.corr_abs_21": _pw_corr_abs_21,
    "x.partial_corr_21": _pw_partial_corr_21,
    "x.leadlag_best_5": _pw_leadlag_best_5,
    "x.coint_resid_z_252": _pw_coint_resid_z_252,
}


def _partial_correlation(df: pd.DataFrame, asset: str, bench: str, window: int) -> pd.Series:
    """Partial correlation controlling for sector."""
    sector = SECTOR_MAP.get(asset, SPY)
    if sector == bench:
        # If sector is same as benchmark, return regular correlation
        return f.rolling_corr_fisher(_ret1d(df, asset), _ret1d(df, bench), window)[0]
    
    # Otherwise compute partial correlation
    r_asset = _ret1d(df, asset)
    r_bench = _ret1d(df, bench)
    r_sector = _ret1d(df, sector)
    
    # Residualize asset and bench on sector
    beta_asset = f.rolling_beta(r_asset, r_sector, window)
    beta_bench = f.rolling_beta(r_bench, r_sector, window)
    
    resid_asset = r_asset - beta_asset * r_sector
    resid_bench = r_bench - beta_bench * r_sector
    
    # Correlation of residuals (safe)
    minp = int(os.environ.get("PAIRWISE_MIN_PERIODS", "20"))
    return safe_rolling_corr(resid_asset, resid_bench, window, minp)


# ----------------------
# Event bundle / interactions
# ----------------------
@memory.cache
def _cached_build_event_features():
    """Cached version of build_event_features to avoid recomputation.
    Using optimized caching with compression and memory mapping."""
    return build_event_features()

def _ev_bundle(index_like) -> pd.DataFrame:
    EV = None
    try:
        EV = _cached_build_event_features()
    except Exception as e:
        # do NOT hide the error — surface it loudly
        print(f"[events] ERROR: {e}")
        EV = pd.DataFrame(index=index_like)

    if EV is None or EV.empty:
        # ensure we return a df aligned to index_like even if empty
        return pd.DataFrame(index=index_like)

    def _safe(k):
        s = EV[k] if k in EV.columns else pd.Series(index=EV.index, dtype=float)
        return pd.to_numeric(s, errors="coerce")

    keep_prefixes = ("EV", "days_to_", "COND.", "EXP.", "META.", "day_", "SEQ.", "INF.", "LAB.")
    ev_cols = {col: _safe(col) for col in EV.columns if any(col.startswith(p) for p in keep_prefixes)}
    out = pd.DataFrame(ev_cols, index=EV.index)
    return out.reindex(index_like)


EV_INTERACT: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]] = {
    "ev.gated_mom21":     lambda EV,D,t: f.pct_change_n(_col(D,t,"PX_LAST"),21) * (EV["EV_forward_calendar_heat_3"] < EV["EV_forward_calendar_heat_3"].rolling(7).median()).astype(float),
    "ev.tail_carry":      lambda EV,D,t: EV["EV_after_surprise_z"].ewm(span=5, min_periods=2).mean(),
    "ev.infl_beta_gate":  lambda EV,D,t: EV["EV.bucket_inflation_surp"] * f.rolling_beta(_ret1d(D,t), f.pct_change_n(_col(D,DXY,"PX_LAST"),1),63),
    "ev.growth_rotation": lambda EV,D,t: EV["EV.bucket_growth_surp"] * ( f.pct_change_n(_col(D,"XLY US Equity","PX_LAST"),21) - f.pct_change_n(_col(D,"XLP US Equity","PX_LAST"),21) ),
    
    # Advanced Event Interactions
    "ev.intensity_gated_mom": lambda EV,D,t: f.pct_change_n(_col(D,t,"PX_LAST"),21) * EV.get("EV_forward_calendar_heat_3", pd.Series(index=EV.index, dtype=float)),
    "ev.regime_sensitivity":  lambda EV,D,t: EV.get("EV.bucket_divergence", pd.Series(index=EV.index, dtype=float)) * f.rolling_beta(_ret1d(D,t), _ret1d(D,SPY), 63),
    "ev.vol_regime_gate":     lambda EV,D,t: f.pct_change_n(_col(D,t,"PX_LAST"),5) * (EV.get("EV_tail_intensity_21", pd.Series(index=EV.index, dtype=float)) > 0.5).astype(float),
}


def _compute_single_asset_features(ticker: str, df: pd.DataFrame, feature_specs: Dict) -> Dict[str, pd.Series]:
    """Compute all features for a single asset - parallelized."""
    feats = {}
    for name, fn in feature_specs.items():
        col = f"{ticker}_{name}"
        try:
            s = pd.to_numeric(fn(df, ticker), errors="coerce").shift(1)
            feats[col] = s
        except Exception as e:
            print(f"[feat warn] {col}: {e}")
    return feats

def _process_ticker_chunk(tickers_chunk: List[str], df: pd.DataFrame, feature_specs: Dict) -> Dict[str, pd.Series]:
    """Process a chunk of tickers to reduce memory pressure."""
    chunk_feats = {}
    # Skip expensive features for speed
    fast_specs = {k: v for k, v in feature_specs.items() 
                  if not any(slow in k for slow in ['hurst', 'dfa_alpha_252', 'lz_complexity', 'run_entropy', 'sentiment_consensus'])}
    
    for ticker in tickers_chunk:
        for name, fn in fast_specs.items():
            col = f"{ticker}_{name}"
            try:
                s = pd.to_numeric(fn(df, ticker), errors="coerce").shift(1)
                chunk_feats[col] = s
            except Exception as e:
                if getattr(settings.ga, "verbose", 0) >= 2:
                    print(f"[feat warn] {col}: {e}")
    return chunk_feats

def _compute_pairwise_features(ticker: str, bench: str, df: pd.DataFrame, pair_specs: Dict) -> Dict[str, pd.Series]:
    """Compute pairwise features for a single asset - parallelized."""
    # Safety defaults for rolling window operations
    DEFAULT_MIN_PERIODS = 20
    pd.options.mode.use_inf_as_na = True
    feats = {}
    MINP = int(os.environ.get("PAIRWISE_MIN_PERIODS", str(DEFAULT_MIN_PERIODS)))
    for name, fn in pair_specs.items():
        col = f"{ticker}_{name}"
        try:
            # centralized min_periods injection (only if supported) 
            # Skip expensive features during signal generation for speed
            if any(expensive in name for expensive in ['coint_resid', 'leadlag', 'partial_corr']):
                continue
            result = call_pair_spec(fn, df, ticker, bench, min_periods=MINP)
            if isinstance(result, tuple):
                s_corr, s_lag = result
                feats[col] = pd.to_numeric(s_corr, errors="coerce").shift(1)
                feats[f"{col}_lag"] = pd.to_numeric(s_lag, errors="coerce").shift(1)
            else:
                s = pd.to_numeric(result, errors="coerce").shift(1)
                feats[col] = s
        except Exception as e:
            print(f"[pair warn] {col}: {e}")
    return feats

# --- Safe spec invocation shim (inject min_periods when supported) ---
def call_with_supported_kwargs(func, /, *args, **kwargs):
    """
    Call func(*args, **filtered_kwargs) where filtered_kwargs are those present in func's signature.
    Prevents passing unsupported keywords to legacy spec functions.
    """
    try:
        sig = inspect.signature(func)
        allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(*args, **allowed)
    except Exception:
        # if inspection fails, just call without kwargs
        return func(*args)

def call_pair_spec(func, *args, min_periods: int | None = None, **kwargs):
    """
    Call pairwise spec function, injecting min_periods if the function supports it.
    """
    if min_periods is not None:
        kwargs = {**kwargs, "min_periods": min_periods}
    return call_with_supported_kwargs(func, *args, **kwargs)

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: wide panel with columns like '<TICKER>_<FIELD>' indexed by daily dates.
    Output: feature matrix X (all features shifted by 1 day; EV_* already leak-safe upstream).
    """
    print("Building feature matrix with parallelization...")
    
    tradables: List[str] = list(dict.fromkeys(getattr(settings.data, "tradable_tickers", TICKS_ALL)))
    bench = getattr(settings.data, "benchmark_ticker", SPY if SPY in TICKS_ALL else SPX)
    macro_ticks: List[str] = list(getattr(settings.data, "macro_tickers", []))
    all_ticks = list(dict.fromkeys(tradables + macro_ticks + [bench]))

    feats: Dict[str, pd.Series] = {}

    # ---- Macro once (sequential - small) ----
    print("  Computing macro features...")
    for name, fn in MACRO_SPECS.items():
        try:
            feats[name] = pd.to_numeric(fn(df), errors="coerce").shift(1)
        except Exception as e:
            print(f"[macro warn] {name}: {e}")

    # ---- Chunked Single-asset processing ----
    chunk_size = 10  # Process 10 tickers at a time
    print(f"  Computing single-asset features for {len(all_ticks)} tickers in chunks...")
    
    for i in range(0, len(all_ticks), chunk_size):
        chunk = all_ticks[i:i + chunk_size]
        print(f"    Processing chunk {i//chunk_size + 1}/{(len(all_ticks) + chunk_size - 1)//chunk_size}: {chunk}")
        chunk_results = _process_ticker_chunk(chunk, df, FEAT)
        feats.update(chunk_results)
        
        # Force garbage collection between chunks
        if i % (chunk_size * 3) == 0:
            gc.collect()
        if i % (chunk_size * 3) == 0:
            gc.collect()

    # ---- Pairwise vs benchmark (parallelized) ----
    print(f"  Computing pairwise features for {len(tradables)} tickers...")
    pairwise_tickers = [t for t in tradables if t != bench]

    # --- Pairwise config gates ---
    disable_pairwise = bool(int(os.environ.get("FEATURES_DISABLE_PAIRWISE", "0")))
    pairwise_cap = os.environ.get("FEATURES_PAIRWISE_MAX_TICKERS", "")
    pairwise_cap = int(pairwise_cap) if str(pairwise_cap).isdigit() else None

    if disable_pairwise:
        print("  [pairwise] DISABLED via FEATURES_DISABLE_PAIRWISE=1")
        pairwise_tickers = []
    else:
        if pairwise_cap is not None and pairwise_cap > 0:
            if len(pairwise_tickers) > pairwise_cap:
                print(f"  [pairwise] Limiting tickers from {len(pairwise_tickers)} -> {pairwise_cap}")
                pairwise_tickers = pairwise_tickers[:pairwise_cap]
        else:
            # Auto-limit for performance: cap at 32 tickers for speed
            if len(pairwise_tickers) > 32:
                print(f"  [pairwise] Auto-limiting from {len(pairwise_tickers)} -> 32 for performance")
                pairwise_tickers = pairwise_tickers[:32]
    # --- Optional micro-profiler hook ---
    def _maybe_profile_pairwise(enabled_env="FEATURES_PAIRWISE_PROFILE"):
        try:
            flag = os.environ.get(enabled_env, "0") == "1"
            if not flag:
                return contextlib.nullcontext()
            import cProfile, pstats, io
            class _Profiler:
                def __enter__(self):
                    self.pr = cProfile.Profile(); self.pr.enable(); return self
                def __exit__(self, exc_type, exc, tb):
                    self.pr.disable()
                    s = io.StringIO()
                    pstats.Stats(self.pr, stream=s).sort_stats("tottime").print_stats(25)
                    print("\n[top25 pairwise profile]\n", s.getvalue())
            return _Profiler()
        except Exception:
            return contextlib.nullcontext()

    if pairwise_tickers:
        with _maybe_profile_pairwise():
            #  Computing pairwise features for N tickers... (adaptive, streaming)
            print(f"  Computing pairwise features for {len(pairwise_tickers)} tickers...")
            t0_pair = time.time()
            foot = _df_footprint_bytes(df)
            use_threads = True  # heuristic: always threads to avoid pickling large df
            max_jobs = int(os.environ.get("FEATURES_PAIRWISE_JOBS", "8"))
            backend = "threading" if use_threads else "loky"

            # slim the input DF to just what's needed by pairwise logic
            view_df = _slim_view_for_pairwise(df, tickers=pairwise_tickers + [bench])
            spec_version = os.environ.get("PAIRWISE_SPEC_VERSION", "v1")  # bump when you change specs

            from joblib import Parallel as _P2, delayed as _d2
            with suppress_pairwise_runtime_warnings():
                if use_threads:
                    # heartbeat progress controls
                    _progress_ctr = count(start=1)
                    _progress_every = int(os.environ.get("FEATURES_PAIRWISE_PROGRESS_EVERY", "10"))

                    def _emit(target):
                        dh = _data_hash_for_pair(view_df, target, bench)
                        res = _compute_pairwise_features_cached(target, bench, spec_version, dh, view_df, PAIR_SPECS)
                        feats.update(res)
                        k = next(_progress_ctr)
                        if k % _progress_every == 0:
                            print(f"    [pairwise] completed {k}/{len(pairwise_tickers)}")

                    n_jobs_raw = min(max_jobs, max(1, len(pairwise_tickers)))
                    n_jobs, nj_note = decide_pairwise_jobs(foot, n_jobs_raw)
                    _P2(n_jobs=n_jobs, backend=backend, batch_size="auto")(
                        _d2(_emit)(t) for t in pairwise_tickers
                    )
                else:
                    for t in pairwise_tickers:
                        dh = _data_hash_for_pair(view_df, t, bench)
                        res = _compute_pairwise_features_cached(t, bench, spec_version, dh, view_df, PAIR_SPECS)
                        feats.update(res)

            dt_pair = time.time() - t0_pair
            try:
                print(f"    Pairwise stage: backend={backend}, jobs={n_jobs if use_threads else 1}, df_footprint={foot/1e6:.1f} MB, elapsed={dt_pair:.1f}s")
            except Exception:
                pass

    # Complexity features are now computed via individual lambdas above
    # No bulk computation needed - handled by the fast engine on-demand

    # ---- EV bundle & interactions (cached) ----
    print("  Computing event features...")
    EV = _ev_bundle(df.index)
    for c in EV.columns:
        feats[c] = EV[c]  # already leak-safe upstream
    
    print("  Computing event interactions...")
    for t in tradables:
        for name, fn in EV_INTERACT.items():
            col = f"{t}_{name}"
            try:
                feats[col] = pd.to_numeric(fn(EV, df, t), errors="coerce").shift(1)
            except Exception as e:
                print(f"[ev warn] {col}: {e}")

    # ---- Assemble (avoid fragmentation) ----
    print("  Assembling feature matrix...")
    X = pd.DataFrame(feats).sort_index()
    
    # Drop non-event columns that are all-NaN, keep EV/COND/EXP/META/day_* even if sparse
    keep_prefixes = ("EV", "days_to_", "COND.", "EXP.", "META.", "day_", "SEQ.", "INF.", "LAB.")
    event_cols = [c for c in X.columns if any(c.startswith(p) for p in keep_prefixes)]
    non_event_cols = [c for c in X.columns if c not in event_cols]
    
    # Drop all-NaN non-event columns
    if non_event_cols:
        X_non_event = X[non_event_cols].dropna(how="all", axis=1)
    else:
        X_non_event = pd.DataFrame(index=X.index)
    
    # Keep all event columns (even if mostly NaN - events are naturally sparse)
    if event_cols:
        X_event = X[event_cols]
    else:
        X_event = pd.DataFrame(index=X.index)
    
    # Combine back together
    X = pd.concat([X_non_event, X_event], axis=1)

    # ---- Quality masks / flags (parallelized) ----
    print("  Computing quality gates...")
    def _compute_gates(ticker):
        volz = f.zscore_rolling(_col(df,ticker,"PX_VOLUME"),63)
        toz  = f.zscore_rolling(_col(df,ticker,"TURNOVER"),63)
        liquid_flag = ((volz>0) & (toz>0)).astype(float).shift(1)
        opt_flag = _options_available(df,ticker).shift(1)
        return {
            f"{ticker}_gate.liquid_flag": liquid_flag,
            f"{ticker}_gate.opt_avail_flag": opt_flag
        }
    
    gate_results = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_compute_gates)(t) for t in tradables
    )
    
    for result in gate_results:
        feats.update(result)
        for col, series in result.items():
            X[col] = series

    # ---- Cross-sectional ranks - EXPANDED ----
    print("  Computing cross-sectional ranks...")
    def _cs_rank(stub: str, outname: str):
        nonlocal X
        cols = [f"{t}_{stub}" for t in tradables if f"{t}_{stub}" in X.columns]
        if not cols:
            return
        sub = X.loc[:, cols]
        R = sub.rank(axis=1, pct=True)
        
        # FIX: Ensure we extract proper 1D Series from ranked DataFrame
        new_cols = {}
        for t in tradables:
            col_name = f"{t}_{stub}"
            if col_name in R.columns:
                # Extract Series properly - R[col_name] should already be 1D
                series_data = R[col_name]
                # Ensure it's a proper Series (should already be from DataFrame column selection)
                if not isinstance(series_data, pd.Series):
                    series_data = pd.Series(series_data, index=R.index)
                new_cols[f"{t}_{outname}"] = series_data.astype(float)
        
        if new_cols:
            # Create DataFrame from properly formatted 1D Series
            new_df = pd.DataFrame(new_cols, index=X.index)
            X = pd.concat([X, new_df], axis=1)

    def _sector_neutral_cs_rank(stub: str, outname: str):
        nonlocal X
        by_sector = {}
        for t in tradables:
            sec = SECTOR_MAP.get(t, SPY)
            by_sector.setdefault(sec, []).append(t)
        all_new_cols = {}
        for sec, members in by_sector.items():
            cols = [f"{m}_{stub}" for m in members if f"{m}_{stub}" in X.columns]
            if not cols:
                continue
            sub = X.loc[:, cols]
            demean = sub.sub(sub.mean(axis=1), axis=0)
            R = demean.rank(axis=1, pct=True)
            for m in members:
                c = f"{m}_{stub}"
                if c in R.columns:
                    # Extract Series properly - R[c] should already be 1D
                    series_data = R[c]
                    # Ensure it's a proper Series (should already be from DataFrame column selection)
                    if not isinstance(series_data, pd.Series):
                        series_data = pd.Series(series_data, index=R.index)
                    all_new_cols[f"{m}_{outname}"] = series_data.astype(float)
        
        if all_new_cols:
            new_df = pd.DataFrame(all_new_cols, index=X.index)
            X = pd.concat([X, new_df], axis=1)

    def _sector_neutral_z(stub: str, outname: str):
        """Add sector-neutral z-scores."""
        nonlocal X
        by_sector = {}
        for t in tradables:
            sec = SECTOR_MAP.get(t, SPY)
            by_sector.setdefault(sec, []).append(t)
        
        all_new_cols = {}
        for sec, members in by_sector.items():
            cols = [f"{m}_{stub}" for m in members if f"{m}_{stub}" in X.columns]
            if not cols:
                continue
            sub = X.loc[:, cols]
            
            # Sector mean and std
            sec_mean = sub.mean(axis=1)
            sec_std = sub.std(axis=1)
            
            # Z-score within sector
            for m in members:
                c = f"{m}_{stub}"
                if c in sub.columns:
                    z_score = (sub[c] - sec_mean) / (sec_std + 1e-9)
                    all_new_cols[f"{m}_{outname}"] = z_score.astype(float)
        
        if all_new_cols:
            new_df = pd.DataFrame(all_new_cols, index=X.index)
            X = pd.concat([X, new_df], axis=1)

    # Extended CS ranks for more predictive stubs
    _cs_rank("px.mom_5", "cs.rank_mom_5")
    _cs_rank("px.mom_21", "cs.rank_mom_21")
    _cs_rank("px.mom_63", "cs.rank_mom_63")
    _cs_rank("px.range_pos_63", "cs.rank_range_pos_63")
    _cs_rank("vol.rv_21", "cs.rank_rv_21")
    _cs_rank("opt.vrp_21", "cs.rank_vrp_21")
    _cs_rank("x.beta_spy_63", "cs.rank_beta_spy_63")
    _cs_rank("risk.drawdown_63", "cs.rank_drawdown_63")
    _cs_rank("liq.amihud_21", "cs.rank_illiq_21")
    
    # Sector-neutral ranks
    _sector_neutral_cs_rank("x.spread_vs_sector_21", "sn.rank_spread_vs_sector_21")
    
    # Sector-neutral z-scores
    _sector_neutral_z("px.mom_21", "sn.z_mom_21")
    _sector_neutral_z("vol.rv_21", "sn.z_rv_21")

    print(f"Feature matrix complete: {X.shape[1]} features, {X.shape[0]} observations")
    return X


# =========================
# Adaptive Feature Selection System
# =========================
class AdaptiveFeatureSelector:
    def __init__(self, lookback: int = 252, min_correlation: float = 0.1):
        self.lookback = lookback
        self.min_correlation = min_correlation
        self._feature_scores: Dict[str, float] = {}
        
    def score_features(self, feature_matrix: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Score features based on rolling correlation with target."""
        scores = {}
        for col in feature_matrix.columns:
            if col.startswith(('liquid_flag', 'opt_avail_flag')):
                continue
                
            feat = feature_matrix[col].dropna()
            if len(feat) < self.lookback // 2:
                continue
                
            # Rolling correlation with target
            corr = feat.rolling(self.lookback).corr(target.reindex(feat.index))
            avg_abs_corr = corr.abs().mean()
            
            if pd.notna(avg_abs_corr):
                scores[col] = avg_abs_corr
        
        return scores
    
    def select_features(self, feature_matrix: pd.DataFrame, target: pd.Series, 
                       max_features: int = 200) -> List[str]:
        """Select top features based on adaptive scoring."""
        scores = self.score_features(feature_matrix, target)
        
        # Filter by minimum correlation
        filtered_scores = {k: v for k, v in scores.items() if v >= self.min_correlation}
        
        # Sort by score and take top N
        sorted_features = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [feat for feat, score in sorted_features[:max_features]]
        
        print(f"Selected {len(selected)} features (avg score: {np.mean([s for _, s in sorted_features[:max_features]]):.3f})")
        return selected

def build_adaptive_feature_matrix(df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
    """Build feature matrix with adaptive selection if target provided."""
    # Build full matrix first
    X = build_feature_matrix(df)
    
    if target is not None:
        selector = AdaptiveFeatureSelector()
        selected_features = selector.select_features(X, target)
        
        # Keep gating flags regardless
        gate_cols = [c for c in X.columns if 'flag' in c.lower()]
        final_cols = list(set(selected_features + gate_cols))
        
        X = X[final_cols]
        print(f"Adaptive selection reduced features from {len(X.columns)} to {len(final_cols)}")
    
    return X