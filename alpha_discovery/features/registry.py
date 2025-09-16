# registry.py  (drop-in, wired to your tickers/fields)
import numpy as np
import pandas as pd
from typing import Dict, Callable, List

from ..config import settings
from . import core as f
from ..data.events import build_event_features  # daily EV_* features from your events.py


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
# Universe
# ----------------------
TICKS_ALL: List[str] = [
    "CRM US Equity","QQQ US Equity","SPY US Equity","TSLA US Equity","AMZN US Equity","GOOGL US Equity",
    "MSFT US Equity","AAPL US Equity","LLY US Equity","CRWV US Equity","JPM US Equity","C US Equity",
    "PLTR US Equity","ARM US Equity","AMD US Equity","BMY US Equity","PEP US Equity","NKE US Equity",
    "WMT US Equity","DXY Curncy","JPY Curncy","EUR Curncy","XAU Curncy","XLE US Equity","CL1 Comdty",
    "XLK US Equity","XLRE US Equity","XLC US Equity","XLV US Equity","MSTR US Equity","COIN US Equity",
    "BTC Index","XLY US Equity","EFA US Equity","MXWO Index","EEM US Equity","USGG10YR Index",
    "USGG2YR Index","XLP US Equity","SPX Index","NDX Index","RTY Index","HG1 Comdty"
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
# Sectors
XLK = "XLK US Equity"; XLE = "XLE US Equity"; XLV = "XLV US Equity"; XLY = "XLY US Equity"; XLP = "XLP US Equity"
XLRE = "XLRE US Equity"; XLC = "XLC US Equity"

# Sector map for single names
SECTOR_MAP: Dict[str, str] = {
    # Tech
    "AAPL US Equity": XLK, "MSFT US Equity": XLK, "GOOGL US Equity": XLK,
    "AMD US Equity": XLK, "ARM US Equity": XLK, "PLTR US Equity": XLK,
    "CRM US Equity": XLK, "CRWV US Equity": XLK,
    # Consumer Discretionary
    "AMZN US Equity": XLY, "TSLA US Equity": XLY, "NKE US Equity": XLY,
    # Consumer Staples
    "WMT US Equity": XLP, "PEP US Equity": XLP,
    # Health Care
    "LLY US Equity": XLV, "BMY US Equity": XLV,
    # Financials (fallback to SPY — no XLF in universe)
    "JPM US Equity": SPY, "C US Equity": SPY,
    # Crypto beta — use QQQ as peer proxy
    "MSTR US Equity": QQQ, "COIN US Equity": QQQ,
}


# ----------------------
# Macro features (once)
# ----------------------
MACRO_SPECS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
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
}


# ----------------------
# Single-asset feature specs
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

    # L2: Volatility / Tails / Drawdown
    "vol.rv_21":              lambda D,t: f.realized_vol(_col(D,t,"PX_LAST"),21),
    "vol.rv_gk_21":           lambda D,t: f.garman_klass_vol(_col(D,t,"PX_OPEN"),_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),_col(D,t,"PX_LAST"),21),
    "vol.vol_of_vol_21":      lambda D,t: f.vol_of_vol(f.realized_vol(_col(D,t,"PX_LAST"),21),21),
    "vol.semivar_dn_21":      lambda D,t: f.semivariance(_ret1d(D,t),21,True),
    "vol.updown_ratio_21":    lambda D,t: f.up_down_vol_ratio(_ret1d(D,t),21),
    "vol.skew_63":            lambda D,t: f.realized_skew_kurt(_ret1d(D,t),63)[0],
    "vol.kurt_63":            lambda D,t: f.realized_skew_kurt(_ret1d(D,t),63)[1],
    "vol.left_tail_pr_252":   lambda D,t: f.tail_probs(_ret1d(D,t),252,0.05)[0],
    "vol.right_tail_pr_252":  lambda D,t: f.tail_probs(_ret1d(D,t),252,0.05)[1],
    "risk.drawdown_63":       lambda D,t: f.drawdown_from_high(_col(D,t,"PX_LAST"),63),
    "risk.time_in_dd_63":     lambda D,t: f.time_in_drawdown(_col(D,t,"PX_LAST"),63),

    # L3: Liquidity / Microstructure
    "liq.amihud_21":          lambda D,t: f.amihud_illiquidity(_ret1d(D,t), _dvol(D,t), 21),
    "liq.cs_spread_21":       lambda D,t: f.corwin_schultz_spread(_col(D,t,"PX_HIGH"),_col(D,t,"PX_LOW"),21),
    "liq.roll_proxy_21":      lambda D,t: f.roll_spread_proxy(_ret1d(D,t),21),
    "liq.turnover_z_63":      lambda D,t: f.zscore_rolling(_col(D,t,"TURNOVER"),63),

    # L4: Options / IV / Flow
    "opt.iv30_z_60":          lambda D,t: f.zscore_rolling(_col(D,t,"CALL_IMP_VOL_30D"),60),
    "opt.iv3m_z_60":          lambda D,t: f.zscore_rolling(_col(D,t,"3MO_CALL_IMP_VOL"),60),
    "opt.term_30d_3m":        lambda D,t: _col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"3MO_CALL_IMP_VOL"),
    "opt.skew_30d":           lambda D,t: _col(D,t,"CALL_IMP_VOL_30D") - _col(D,t,"PUT_IMP_VOL_30D"),
    "opt.rr25_1m_proxy":      lambda D,t: _col(D,t,"1M_CALL_IMP_VOL_25DELTA_DFLT") - _col(D,t,"1M_PUT_IMP_VOL_10DELTA_DFLT"),
    "opt.fly_1m_40v25":       lambda D,t: 0.5*(_col(D,t,"1M_CALL_IMP_VOL_40DELTA_DFLT")+_col(D,t,"1M_PUT_IMP_VOL_40DELTA_DFLT")) - _col(D,t,"1M_CALL_IMP_VOL_25DELTA_DFLT"),
    "opt.iv_mom_5_z60":       lambda D,t: f.zscore_rolling(f.diff_n(_col(D,t,"3MO_CALL_IMP_VOL"),5),60),
    "opt.vrp_21":             lambda D,t: _col(D,t,"3MO_CALL_IMP_VOL") - f.realized_vol(_col(D,t,"PX_LAST"),21) * 100.0,
    "opt.rv_minus_iv_21":     lambda D,t: f.realized_vol(_col(D,t,"PX_LAST"),21) * 100.0 - _col(D,t,"3MO_CALL_IMP_VOL"),
    "flow.pcr_ema5_z30":      lambda D,t: f.zscore_rolling(_pcr(df=D,t=t).ewm(span=5,min_periods=2).mean(),30),
    "flow.opt_vol_z30":       lambda D,t: f.zscore_rolling(_col(D,t,"TOT_OPT_VOLUME_CUR_DAY"),30),
    "pos.oi_call_put_ratio":  lambda D,t: f.safe_div(_col(D,t,"OPEN_INT_TOTAL_CALL"), _col(D,t,"OPEN_INT_TOTAL_PUT")),
    "pos.oi_skew_z60":        lambda D,t: f.zscore_rolling(_oi_skew(D,t),60),

    # L5: Sentiment / Media
    "sent.tw_pub_z30":        lambda D,t: f.zscore_rolling(_col(D,t,"TWITTER_PUBLICATION_COUNT"),30),
    "sent.tw_avg_z30":        lambda D,t: f.zscore_rolling(_col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"),30),
    "sent.tw_mom_21":         lambda D,t: f.diff_n(_col(D,t,"TWITTER_SENTIMENT_DAILY_AVG"),21),
    "sent.news_pub_z30":      lambda D,t: f.zscore_rolling(_col(D,t,"NEWS_PUBLICATION_COUNT"),30),
    "sent.news_heat_z21":     lambda D,t: f.zscore_rolling(_col(D,t,"NEWS_HEAT_READ_DMAX"),21),
    "sent.cn_avg_z60":        lambda D,t: f.zscore_rolling(_col(D,t,"CHINESE_NEWS_SENTMNT_DAILY_AVG"),60),

    # L6: Cross-asset β & spreads (vs SPY & sector)
    "x.beta_spy_63":          lambda D,t: f.rolling_beta(_ret1d(D,t), _ret1d(D, SPY if SPY in TICKS_ALL else SPX), 63),
    "x.bh_ret_spy_21":        lambda D,t: f.beta_hedged_return(_ret1d(D,t), _ret1d(D, SPY if SPY in TICKS_ALL else SPX), 63).rolling(21).sum(),
    "x.spread_vs_sector_21":  lambda D,t: f.pct_change_n(_col(D,t,"PX_LAST"),21) - f.pct_change_n(_col(D, SECTOR_MAP.get(t, SPY), "PX_LAST"),21),

    # L7: Regime (ticker)
    "reg.vol90d_z120":        lambda D,t: f.zscore_rolling(_col(D,t,"VOLATILITY_90D"),120),
}


# ----------------------
# Pairwise vs benchmark
# ----------------------
PAIR_SPECS: Dict[str, Callable[[pd.DataFrame, str, str], pd.Series]] = {
    "x.corr_fisher20_z60":    lambda D,a,b: f.zscore_rolling(f.rolling_corr_fisher(_ret1d(D,a), _ret1d(D,b),20)[1],60),
    "x.corr_delta_20_60_z60": lambda D,a,b: f.zscore_rolling(
        f.rolling_corr_fisher(_ret1d(D,a), _ret1d(D,b),20)[1] - f.rolling_corr_fisher(_ret1d(D,a), _ret1d(D,b),60)[1],60
    ),
    "x.beta_60_z120":         lambda D,a,b: f.zscore_rolling(f.rolling_beta(_ret1d(D,a), _ret1d(D,b),60),120),
    "x.corr_abs_21":          lambda D,a,b: f.corr_abs_to_abs(_ret1d(D,a), _ret1d(D,b),21),
}


# ----------------------
# Event bundle / interactions
# ----------------------
def _ev_bundle(index_like) -> pd.DataFrame:
    EV = None
    try:
        EV = build_event_features()
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

    keep_prefixes = ("EV", "days_to_", "COND.", "EXP.", "META.", "day_")
    ev_cols = {col: _safe(col) for col in EV.columns if any(col.startswith(p) for p in keep_prefixes)}
    out = pd.DataFrame(ev_cols, index=EV.index)
    return out.reindex(index_like)


EV_INTERACT: Dict[str, Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]] = {
    "ev.gated_mom21":     lambda EV,D,t: f.pct_change_n(_col(D,t,"PX_LAST"),21) * (EV["EV_forward_calendar_heat_3"] < EV["EV_forward_calendar_heat_3"].rolling(7).median()).astype(float),
    "ev.tail_carry":      lambda EV,D,t: EV["EV_after_surprise_z"].ewm(span=5, min_periods=2).mean(),
    "ev.infl_beta_gate":  lambda EV,D,t: EV["EV.bucket_inflation_surp"] * f.rolling_beta(_ret1d(D,t), f.pct_change_n(_col(D,DXY,"PX_LAST"),1),63),
    "ev.growth_rotation": lambda EV,D,t: EV["EV.bucket_growth_surp"] * ( f.pct_change_n(_col(D,"XLY US Equity","PX_LAST"),21) - f.pct_change_n(_col(D,"XLP US Equity","PX_LAST"),21) ),
}


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: wide panel with columns like '<TICKER>_<FIELD>' indexed by daily dates.
    Output: feature matrix X (all features shifted by 1 day; EV_* already leak-safe upstream).
    """
    tradables: List[str] = list(dict.fromkeys(getattr(settings.data, "tradable_tickers", TICKS_ALL)))
    bench = getattr(settings.data, "benchmark_ticker", SPY if SPY in TICKS_ALL else SPX)
    macro_ticks: List[str] = list(getattr(settings.data, "macro_tickers", []))
    all_ticks = list(dict.fromkeys(tradables + macro_ticks + [bench]))

    feats: Dict[str, pd.Series] = {}

    # ---- Macro once ----
    for name, fn in MACRO_SPECS.items():
        try:
            feats[name] = pd.to_numeric(fn(df), errors="coerce").shift(1)
        except Exception as e:
            print(f"[macro warn] {name}: {e}")

    # ---- Single-asset ----
    for t in all_ticks:
        for name, fn in FEAT.items():
            col = f"{t}_{name}"
            try:
                s = pd.to_numeric(fn(df, t), errors="coerce").shift(1)
                feats[col] = s
            except Exception as e:
                print(f"[feat warn] {col}: {e}")

    # ---- Pairwise vs benchmark ----
    for t in tradables:
        if t == bench:
            continue
        for name, fn in PAIR_SPECS.items():
            col = f"{t}_{name}"
            try:
                s = pd.to_numeric(fn(df, t, bench), errors="coerce").shift(1)
                feats[col] = s
            except Exception as e:
                print(f"[pair warn] {col}: {e}")

    # ---- EV bundle & interactions ----
    EV = _ev_bundle(df.index)
    for c in EV.columns:
        feats[c] = EV[c]  # already leak-safe upstream
    for t in tradables:
        for name, fn in EV_INTERACT.items():
            col = f"{t}_{name}"
            try:
                feats[col] = pd.to_numeric(fn(EV, df, t), errors="coerce").shift(1)
            except Exception as e:
                print(f"[ev warn] {col}: {e}")

    # ---- Assemble (avoid fragmentation) ----
    X = pd.DataFrame(feats).sort_index()
    
    # Drop non-event columns that are all-NaN, keep EV/COND/EXP/META/day_* even if sparse
    keep_prefixes = ("EV", "days_to_", "COND.", "EXP.", "META.", "day_")
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

    # ---- Quality masks / flags ----
    for t in tradables:
        volz = f.zscore_rolling(_col(df,t,"PX_VOLUME"),63)
        toz  = f.zscore_rolling(_col(df,t,"TURNOVER"),63)
        X[f"{t}_gate.liquid_flag"] = ((volz>0) & (toz>0)).astype(float).shift(1)
        X[f"{t}_gate.opt_avail_flag"] = _options_available(df,t).shift(1)

    # ---- Cross-sectional ranks (concat at once) ----
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

    _cs_rank("px.mom_21", "cs.rank_mom_21")
    _cs_rank("px.mom_63", "cs.rank_mom_63")
    _cs_rank("risk.drawdown_63", "cs.rank_drawdown_63")
    _cs_rank("liq.amihud_21", "cs.rank_illiq_21")
    _sector_neutral_cs_rank("x.spread_vs_sector_21", "sn.rank_spread_vs_sector_21")

    return X