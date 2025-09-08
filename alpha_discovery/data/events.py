# alpha_discovery/data/events.py
"""
Event calendar loader + daily EV_* feature builder (expanded).
- Robust to timezone mismatches (normalizes to ET midnight, tz-naive).
- Strictly one-business-day post-release lag to avoid lookahead.
- Preserves your original EV outputs and adds ~28 high-signal event features.
- Optional debug:
    * Recent 'event tape' with actual event_type names
    * Print exact EV_* column list
    * Write per-release and per-day inspection parquet tables
"""
from __future__ import annotations

import math
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

# Suppress pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ---------------------------------------------------------------------
# Project settings and helpers (safe fallbacks if imports missing)
# ---------------------------------------------------------------------
try:
    from ..config import settings  # type: ignore
except Exception:  # pragma: no cover
    settings = type("S", (), {})()  # minimal fallback container

try:
    # Use shared halflife EWMA if available to keep semantics consistent
    from ..features.core import ewma_halflife  # type: ignore
except Exception:  # pragma: no cover
    def ewma_halflife(s: pd.Series, halflife: float) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        return s.ewm(halflife=halflife, adjust=False).mean()

ET_TZ = "America/New_York"

# ---------------------------------------------------------------------
# Defaults (used only if settings.events.* not provided)
# ---------------------------------------------------------------------
DEFAULT_CAL_PATHS = (
    # Canonical path (your correction)
    "data_store/processed/economic_releases.parquet",
    # Fallbacks for resilience
    "data_store/processed/economic_releases.csv",
    "data_store/processed/economic_combined.parquet",
    "data_store/processed/economic_combined.csv",
    "data_store/economic_releases.parquet",
    "data_store/economic_releases.csv",
    "data_store/economic_combined.parquet",
    "data_store/economic_combined.csv",
)

DEFAULT_HIGH_RELEVANCE = 80.0
DEFAULT_PRE_WINDOW = 1       # calendar days before an event included in EV_in_window
DEFAULT_POST_WINDOW = 1      # calendar days after an event included in EV_in_window
DEFAULT_POST_LAG = 1         # business days to delay post-release signals
DEFAULT_TOP_TIER = (
    # Inflation & prices
    "CPI", "PCE", "PPI", "ISM Prices Paid",
    # Labor
    "Change in Nonfarm Payrolls", "Unemployment Rate", "Average Hourly Earnings",
    # Growth/Activity
    "GDP Annualized QoQ", "Retail Sales", "ISM Manufacturing", "ISM Services",
    # Policy
    "FOMC Rate Decision", "Fed Interest on Reserve Balances Rate",
)

INFLATION_KEYS = ("CPI", "PCE", "PPI", "Prices Paid")
GROWTH_KEYS    = ("Payrolls", "Unemployment", "GDP", "Retail Sales", "ISM", "JOLTS")

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _series_to_et_midnight_naive(s: pd.Series) -> pd.Series:
    """Convert timestamps to ET midnight; return tz-naive dates."""
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        return s.dt.tz_convert(ET_TZ).dt.normalize().dt.tz_localize(None)
    return s.dt.tz_localize(ET_TZ).dt.normalize().dt.tz_localize(None)

def _safe_numeric(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")

def _calendar_path() -> Optional[str]:
    # explicit override
    cal_path = getattr(getattr(settings, "events", object()), "calendar_path", None)
    if isinstance(cal_path, str):
        return cal_path
    # common fallbacks
    import os
    for p in DEFAULT_CAL_PATHS:
        if os.path.exists(p):
            return p
    return None

def _load_calendar() -> pd.DataFrame:
    """
    Load the combined economic calendar.
    Expected columns:
      release_datetime, event_type, country, survey, actual, prior, revised,
      relevance, bb_ticker, release_date
    """
    path = _calendar_path()
    if path is None:
        print("  [EV] No economic calendar found.")
        return pd.DataFrame(columns=[
            "release_datetime","event_type","country","survey","actual","prior","revised",
            "relevance","bb_ticker","release_date"
        ])

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Coerce numerics
    for c in ("survey","actual","prior","revised","relevance"):
        if c in df.columns:
            df[c] = _safe_numeric(df[c])

    # Normalize to ET midnight (tz-naive) for the release day
    if "release_datetime" in df.columns and df["release_datetime"].notna().any():
        dt = pd.to_datetime(df["release_datetime"], errors="coerce", utc=True)
        df["release_day"] = _series_to_et_midnight_naive(dt)
        # pre-open flag (09:30 ET gate)
        try:
            etdt = dt.dt.tz_convert(ET_TZ)
            df["preopen_release"] = ((etdt.dt.hour < 9) | ((etdt.dt.hour == 9) & (etdt.dt.minute < 30))).astype(int)
        except Exception:
            df["preopen_release"] = 0
    else:
        df["release_day"] = _series_to_et_midnight_naive(pd.to_datetime(df["release_date"], errors="coerce"))
        df["preopen_release"] = 0

    df["event_type"] = df["event_type"].fillna("").astype(str)
    df["relevance"] = _safe_numeric(df["relevance"]).clip(lower=0.0)

    # Keep US if we have it, else keep all
    if "country" in df.columns:
        has_us = df["country"].astype(str).str.upper().eq("US").any()
        if has_us:
            df = df[df["country"].astype(str).str.upper().eq("US")]

    df = df[df["release_day"].notna() & df["event_type"].ne("")]
    return df.reset_index(drop=True)

# ---- grouped stats helpers (with pandas 2.2+ compat) ----------------
def _gb_apply(df: pd.DataFrame, func, *args, **kwargs) -> pd.Series:
    """GroupBy.apply with pandas>=2.2 include_groups=False support, fallback otherwise."""
    g = df.groupby("event_type", group_keys=False, observed=True)
    try:
        return g.apply(func, *args, include_groups=False, **kwargs)
    except TypeError:
        return g.apply(func, *args, **kwargs)

def _std_grouped_surprise(df: pd.DataFrame,
                          col_actual="actual", col_survey="survey",
                          win: int = 252, min_obs: int = 12) -> pd.Series:
    """
    Standardized surprise per event_type (rolling z of (actual - survey)).
    Uses shift(1) within each type to avoid same-print lookahead.
    """
    x = _safe_numeric(df[col_actual]) - _safe_numeric(df[col_survey])
    df = df.copy()
    df["surprise_raw"] = x

    def _z(g: pd.DataFrame) -> pd.Series:
        s = g["surprise_raw"]
        mu = s.shift(1).rolling(win, min_periods=min_obs).mean()
        sd = s.shift(1).rolling(win, min_periods=min_obs).std(ddof=0)
        return (s - mu) / (sd.replace(0, np.nan))

    return _gb_apply(df, _z)

def _std_grouped_revision(df: pd.DataFrame, win: int = 252, min_obs: int = 8) -> pd.Series:
    """
    Standardized revision per event_type (rolling z of (revised - prior)).
    Shifted to avoid lookahead.
    """
    rev = _safe_numeric(df.get("revised", pd.Series(index=df.index, dtype=float))) \
        - _safe_numeric(df.get("prior", pd.Series(index=df.index, dtype=float)))

    def _rz(g: pd.DataFrame) -> pd.Series:
        r = rev.loc[g.index]
        mu = r.shift(1).rolling(win, min_periods=min_obs).mean()
        sd = r.shift(1).rolling(win, min_periods=min_obs).std(ddof=0)
        return (r - mu) / (sd.replace(0, np.nan))

    return _gb_apply(df, _rz)

def _expanding_signed_percentile(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    """
    Expanding signed percentile (−1..+1) within each event_type.
    Uses shift(1) to prohibit same-print information.
    """
    def _pct(g: pd.DataFrame) -> pd.Series:
        v = values.loc[g.index]
        ranks = v.rank(method="average", pct=True).shift(1)
        return (ranks * 2.0 - 1.0)

    return _gb_apply(df, _pct)

def _is_top_tier(event_type: str, top_list: Tuple[str, ...]) -> bool:
    et = event_type.lower()
    for key in top_list:
        if key.lower() in et:
            return True
    return False

def _bucket(event_type: str) -> str:
    et = event_type.lower()
    if any(k.lower() in et for k in INFLATION_KEYS):
        return "inflation"
    if any(k.lower() in et for k in GROWTH_KEYS):
        return "growth"
    return "other"

# ---------------------------------------------------------------------
# Debug printing: human-readable event tape (optional)
# ---------------------------------------------------------------------
def _print_recent_event_tape(cal: pd.DataFrame, recent_days: int = 5, max_rows: int = 40) -> None:
    """Pretty-print a recent 'event tape' with actual event_type names."""
    if cal.empty:
        print("  [EV][debug] No events to print.")
        return
    cutoff = cal["release_day"].max() - pd.Timedelta(days=recent_days)
    tape = (
        cal[cal["release_day"] >= cutoff]
        .loc[:, ["release_day", "event_type", "relevance", "surprise_z", "revision_z", "net_info_surprise",
                 "is_top_tier", "is_high_impact", "is_tail_1p5", "is_tail_2p5", "bucket", "preopen_release"]]
        .sort_values(["release_day", "relevance"], ascending=[False, False])
        .head(max_rows)
    )
    print("\n================ RECENT ECON EVENT TAPE ================")
    if tape.empty:
        print("  [EV][debug] No recent events in the chosen window.")
    else:
        display_cols = ["release_day", "event_type", "relevance", "surprise_z", "revision_z",
                        "net_info_surprise", "is_top_tier", "is_high_impact", "is_tail_2p5", "bucket", "preopen_release"]
        print(tape[display_cols].to_string(index=False))
    print("========================================================\n")

# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------
def build_event_features(full_data_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build daily EV_* features aligned to the master price index (tz-naive).

    Preserved originals:
      EV_days_to_high, EV_in_window, EV_pre_window, EV_is_event_week,
      EV_after_surprise_z, EV_after_pos, EV_after_neg,
      EV_surprise_ewma_21, EV_surprise_ewma_63,
      EV_tail_intensity_21, EV_dense_macro_window_7

    Additions (post-release, BDay(1) lag unless noted):
      EV_signed_surprise_percentile, EV_surprise_dispersion_day,
      EV_tail_flag_1p5, EV_tail_flag_2p5, EV_time_since_tailshock_60,
      EV_revision_z, EV_revision_tail_flag, EV_net_info_surprise,
      EV_revision_polarity_memory_3, EV_revision_vol_252,
      EV_shock_adjusted_surprise,
      EV_dense_macro_day_score, EV_top_tier_dominance_share,
      EV_forward_calendar_heat_3 (calendar-forward, no lag),
      EV_calendar_vacuum_7 (calendar-forward, no lag),
      EV_clustered_tail_count_5,
      EV_infl_vs_growth_divergence
    """
    # Building event features from economic calendar...

    # Settings / thresholds (with safe defaults)
    conf = getattr(settings, "events", object())
    pre  = int(getattr(conf, "pre_window_days", DEFAULT_PRE_WINDOW))
    post = int(getattr(conf, "post_window_days", DEFAULT_POST_WINDOW))
    lag  = int(getattr(conf, "post_release_lag_days", DEFAULT_POST_LAG))
    hi_rel = float(getattr(conf, "high_relevance_threshold", DEFAULT_HIGH_RELEVANCE))
    top_tier = tuple(getattr(conf, "top_tier_types", DEFAULT_TOP_TIER))

    # -----------------------------------------------------------------
    # Normalize trading index: work on a unique index internally
    # -----------------------------------------------------------------
    base_index = pd.DatetimeIndex(full_data_index).tz_localize(None)
    uni_index = pd.DatetimeIndex(base_index.drop_duplicates(keep="first"))
    # Build on the unique index; project back to base_index at the end.
    out = pd.DataFrame(index=uni_index)

    cal = _load_calendar()
    if cal.empty:
        return pd.DataFrame(index=base_index)

    # Per-release stats
    cal = cal.sort_values("release_day").reset_index(drop=True)
    cal["surprise_z"] = _std_grouped_surprise(cal)
    cal["revision_z"] = _std_grouped_revision(cal)
    cal["net_info_surprise"] = cal["surprise_z"] + 0.7 * cal["revision_z"].fillna(0.0)
    cal["signed_surprise_pct"] = _expanding_signed_percentile(cal, cal["surprise_z"])

    cal["is_top_tier"] = cal["event_type"].apply(lambda et: _is_top_tier(et, top_tier)).astype(int)
    cal["is_high_impact"] = (cal["relevance"] >= hi_rel).astype(int)
    cal["is_tail_1p5"] = (cal["surprise_z"].abs() > 1.5).astype(int)
    cal["is_tail_2p5"] = (cal["surprise_z"].abs() > 2.5).astype(int)
    cal["bucket"] = cal["event_type"].apply(_bucket)

    # Relevance sqrt-weight (cap at 100 to stabilize)
    w = cal["relevance"].clip(0, 100) ** 0.5
    cal["w_surprise"] = cal["surprise_z"] * w
    cal["w_net_info"] = cal["net_info_surprise"] * w
    cal["w_signed_pct"] = cal["signed_surprise_pct"] * w

    # -----------------------------------------------------------------
    # Calendar-forward features (no lag; based on scheduled dates)
    # -----------------------------------------------------------------
    high_dates = sorted(pd.to_datetime(cal.loc[cal["is_high_impact"].eq(1), "release_day"].dropna().unique()))
    if high_dates:
        # EV_days_to_high: calendar days until next high-impact date
        next_idx = pd.Series(pd.NaT, index=uni_index)
        hi = pd.DatetimeIndex(high_dates)
        j = 0
        for d in uni_index:
            while j < len(hi) and hi[j] < d:
                j += 1
            next_idx.loc[d] = hi[j] if j < len(hi) else pd.NaT
        td = pd.to_timedelta(next_idx - uni_index)
        out["EV_days_to_high"] = pd.Series(td, index=uni_index).dt.days.clip(lower=0).fillna(0).astype(float)
    else:
        out["EV_days_to_high"] = 0.0

    # EV_in_window / EV_pre_window
    out["EV_in_window"] = 0
    out["EV_pre_window"] = 0
    if high_dates:
        for d in high_dates:
            start_pre = d - pd.Timedelta(days=pre)
            end_post = d + pd.Timedelta(days=post)
            out.loc[start_pre:end_post, "EV_in_window"] = 1
            if pre > 0:
                out.loc[start_pre:(d - pd.Timedelta(days=1)), "EV_pre_window"] = 1

    # EV_is_event_week
    out["week_period"] = out.index.to_period("W")
    event_weeks = pd.Series(pd.DatetimeIndex(high_dates)).dt.to_period("W").unique()
    out["EV_is_event_week"] = out["week_period"].isin(event_weeks).astype(int)
    out.drop(columns=["week_period"], inplace=True)

    # EV_dense_macro_window_7: number of high-impact days in next 7 calendar days
    if high_dates:
        by_day = pd.Series(1, index=pd.DatetimeIndex(high_dates))
        dense7 = []
        # Ensure no duplicate index on by_day
        by_day = by_day[~by_day.index.duplicated(keep="last")]
        for dt in uni_index:
            horizon = [dt + pd.Timedelta(days=i) for i in range(1, 8)]
            dense7.append(sum(by_day.reindex(horizon).fillna(0).values))
        out["EV_dense_macro_window_7"] = pd.Series(dense7, index=uni_index, dtype=float)
    else:
        out["EV_dense_macro_window_7"] = 0.0

    # EV_forward_calendar_heat_3: sum of relevance next 3 calendar days
    rel_by_day = cal.groupby("release_day")["relevance"].sum()
    rel_by_day = rel_by_day[~rel_by_day.index.duplicated(keep="last")]
    rel_by_day = rel_by_day.reindex(uni_index, fill_value=0.0)
    heat3 = []
    # Use a plain dict for fast lookups
    rel_map = rel_by_day.to_dict()
    for dt in uni_index:
        tot = 0.0
        for i in (1, 2, 3):
            dti = dt + pd.Timedelta(days=i)
            tot += float(rel_map.get(dti, 0.0))
        heat3.append(tot)
    out["EV_forward_calendar_heat_3"] = pd.Series(heat3, index=uni_index, dtype=float)

    # EV_calendar_vacuum_7: 1 if no high-impact in next 7 days
    out["EV_calendar_vacuum_7"] = (out["EV_dense_macro_window_7"] == 0).astype(int)

    # -----------------------------------------------------------------
    # Post-release aggregated daily signals (will be lagged by BDay)
    # -----------------------------------------------------------------
    daily = cal.groupby("release_day").agg(
        w_surprise_sum=("w_surprise", "sum"),
        w_surprise_den=("relevance", lambda x: (x.clip(0, 100) ** 0.5).sum()),
        w_net_info_sum=("w_net_info", "sum"),
        w_signed_pct_sum=("w_signed_pct", "sum"),
        tails_1p5=("is_tail_1p5", "max"),
        tails_2p5=("is_tail_2p5", "max"),
        tail_any=("surprise_z", lambda s: int((s.abs() > 2.0).any())),
        preopen=("preopen_release", "max"),
        dense_score=("relevance", "sum"),
        top_share=("is_top_tier", "sum"),
        count=("event_type", "count"),
    )
    # Dedup any accidental duplicates (robustness)
    daily = daily[~daily.index.duplicated(keep="last")]
    den = daily["w_surprise_den"].replace(0, np.nan)
    daily["surprise_wavg"] = daily["w_surprise_sum"] / den
    daily["net_info_wavg"] = daily["w_net_info_sum"] / den
    daily["signed_pct_wavg"] = daily["w_signed_pct_sum"] / den

    # Dispersion of surprise across types (std) per day
    disp = cal.groupby("release_day")["surprise_z"].std()
    disp = disp[~disp.index.duplicated(keep="last")]
    daily["surprise_dispersion_day"] = disp.reindex(daily.index)

    # Growth vs Inflation divergence ( −1 if opposite signs; else 0 )
    grp = cal.groupby(["release_day", "bucket"])["surprise_z"].mean().unstack()
    grp = grp[~grp.index.duplicated(keep="last")]
    gv = grp.get("growth")
    inf = grp.get("inflation")
    infl_vs_growth_div = (np.sign(gv).fillna(0) * np.sign(inf).fillna(0))
    infl_vs_growth_div = infl_vs_growth_div.reindex(daily.index).fillna(0)
    daily["infl_vs_growth_div"] = (infl_vs_growth_div == -1).astype(int)

    # Reindex daily to unique trading calendar
    daily = daily.reindex(uni_index).fillna(0.0)

    # Business-day lag helper (drop dup index in returned series)
    def _lag_bday(s: pd.Series, k: int) -> pd.Series:
        s2 = pd.Series(s.values, index=pd.DatetimeIndex(s.index) + BDay(k))
        return s2[~s2.index.duplicated(keep="last")]

    # Apply lag to core aggregates
    s_t     = _lag_bday(daily["surprise_wavg"], lag).reindex(uni_index)
    n_t     = _lag_bday(daily["net_info_wavg"], lag).reindex(uni_index)
    p_t     = _lag_bday(np.sign(daily["surprise_wavg"]).clip(-1, 1), lag).reindex(uni_index)
    pct_t   = _lag_bday(daily["signed_pct_wavg"], lag).reindex(uni_index)
    disp_t  = _lag_bday(daily["surprise_dispersion_day"], lag).reindex(uni_index)
    tail1_t = _lag_bday(daily["tails_1p5"], lag).reindex(uni_index)
    tail2_t = _lag_bday(daily["tails_2p5"], lag).reindex(uni_index)
    tailany_t = _lag_bday(daily["tail_any"], lag).reindex(uni_index)
    infl_div_t = _lag_bday(daily["infl_vs_growth_div"], lag).reindex(uni_index)

    # Core post-release EV features
    out["EV_after_surprise_z"] = s_t.astype(float).fillna(0.0)
    out["EV_after_pos"] = (p_t > 0).fillna(False).astype(int)
    out["EV_after_neg"] = (p_t < 0).fillna(False).astype(int)
    out["EV_signed_surprise_percentile"] = pct_t.astype(float).fillna(0.0)
    out["EV_surprise_dispersion_day"] = disp_t.astype(float).fillna(0.0)
    out["EV_tail_flag_1p5"] = tail1_t.fillna(0).astype(int)
    out["EV_tail_flag_2p5"] = tail2_t.fillna(0).astype(int)
    out["EV_infl_vs_growth_divergence"] = infl_div_t.fillna(0).astype(int)

    # Time since last tail shock (cap at 60)
    tails_any = tailany_t.fillna(0).astype(int)
    ts = []
    since = math.inf
    for v in tails_any.reindex(uni_index).fillna(0).astype(int).values:
        if v > 0:
            since = 0
        else:
            since = since + 1 if since != math.inf else math.inf
        ts.append(0 if since is math.inf else min(float(since), 60.0))
    out["EV_time_since_tailshock_60"] = pd.Series(ts, index=uni_index, dtype=float)

    # Revisions (post-release)
    rev_daily = cal.groupby("release_day")["revision_z"].mean()
    rev_daily = rev_daily[~rev_daily.index.duplicated(keep="last")]
    out["EV_revision_z"] = _lag_bday(rev_daily, lag).reindex(uni_index).astype(float).fillna(0.0)
    rev_tail = cal.groupby("release_day")["revision_z"].apply(lambda s: int((s.abs() > 2.0).any()))
    rev_tail = rev_tail[~rev_tail.index.duplicated(keep="last")]
    out["EV_revision_tail_flag"] = _lag_bday(rev_tail, lag).reindex(uni_index).fillna(0).astype(int)
    out["EV_net_info_surprise"] = n_t.astype(float).fillna(0.0)

    # Revision polarity memory (last 3 per type → daily mean)
    cal_sorted = cal.sort_values("release_day")
    cal_sorted["rev_sign"] = np.sign(cal_sorted["revision_z"]).fillna(0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mem = cal_sorted.groupby("event_type").apply(
            lambda g: g.set_index("release_day")["rev_sign"].rolling(3, min_periods=1).mean().shift(1)
        ).reset_index(level=0, drop=True)
    # mem has duplicate dates by design; average them to a single daily value
    rev_mem_daily = mem.groupby(mem.index).mean()
    rev_mem_daily = rev_mem_daily[~rev_mem_daily.index.duplicated(keep="last")]
    out["EV_revision_polarity_memory_3"] = _lag_bday(rev_mem_daily, lag).reindex(uni_index).astype(float).fillna(0.0)

    # Revision volatility regime (rolling std per type → daily avg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        rev_vol = cal.groupby("event_type").apply(
            lambda g: g.set_index("release_day")["revision_z"].rolling(252, min_periods=30).std().shift(1)
        ).reset_index(level=0, drop=True)
    rev_vol_daily = rev_vol.groupby(rev_vol.index).mean()
    rev_vol_daily = rev_vol_daily[~rev_vol_daily.index.duplicated(keep="last")]
    out["EV_revision_vol_252"] = _lag_bday(rev_vol_daily, lag).reindex(uni_index).astype(float).fillna(0.0)

    # Surprise EWMAs & shock-adjusted composite
    out["EV_surprise_ewma_21"] = ewma_halflife(out["EV_after_surprise_z"], halflife=10.5)
    out["EV_surprise_ewma_63"] = ewma_halflife(out["EV_after_surprise_z"], halflife=31.5)
    out["EV_shock_adjusted_surprise"] = (
        out["EV_after_surprise_z"] * (1.0 + out["EV_forward_calendar_heat_3"].pow(0.5) / 10.0)
    ).astype(float)

    # Tail intensity: share of |surprise|>2 over past 21 trading days
    tails_2p5 = (out["EV_tail_flag_2p5"] > 0).astype(int)
    out["EV_tail_intensity_21"] = tails_2p5.rolling(21, min_periods=5).mean().fillna(0.0)

    # Dense macro day score & top-tier dominance (lagged)
    dense = daily["dense_score"]
    top = daily["top_share"]
    cnt = daily["count"].replace(0, np.nan)
    out["EV_dense_macro_day_score"] = _lag_bday(dense, lag).reindex(uni_index).fillna(0.0).astype(float)
    denom = _lag_bday(cnt, lag).reindex(uni_index)
    numer = _lag_bday(top, lag).reindex(uni_index)
    out["EV_top_tier_dominance_share"] = (numer / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    # Clustered tail count last 5 trading days
    out["EV_clustered_tail_count_5"] = tails_2p5.rolling(5, min_periods=1).sum().astype(float)

    # Final coercion on unique index
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Successfully built event features

    # -------------------------------------------------------------
    # Debug printing / optional dumps (OFF by default; enable via settings)
    # -------------------------------------------------------------
    dbg_conf = getattr(settings, "events", object())
    dbg_print = bool(getattr(dbg_conf, "debug_print_recent", False))
    dbg_days = int(getattr(dbg_conf, "debug_recent_days", 5))
    dbg_rows = int(getattr(dbg_conf, "debug_recent_max_rows", 40))
    dbg_dump  = bool(getattr(dbg_conf, "debug_write_tables", False))

    if dbg_print:
        _print_recent_event_tape(cal, recent_days=dbg_days, max_rows=dbg_rows)
        # Also print the exact EV columns we produced
        print(f"[EV][debug] Built {len(out.columns)} EV feature columns:")
        print(", ".join(list(out.columns)))

    if dbg_dump:
        try:
            # Per-release long table with names
            cal_out = cal.loc[:, ["release_day","event_type","relevance","surprise_z","revision_z",
                                  "net_info_surprise","is_top_tier","is_high_impact","is_tail_1p5","is_tail_2p5","bucket"]]
            cal_out.to_parquet("data_store/processed/ev_releases.parquet", index=False)
            # Per-day aggregates (aligned to release_day, pre-lag)
            daily_out = daily.reset_index()[["release_day","surprise_wavg","net_info_wavg","signed_pct_wavg",
                                             "surprise_dispersion_day","tails_1p5","tails_2p5","tail_any",
                                             "dense_score","top_share","count"]]
            daily_out.to_parquet("data_store/processed/ev_daily.parquet", index=False)
            print("  [EV][debug] Wrote ev_releases.parquet and ev_daily.parquet to data_store/processed/")
        except Exception as e:
            print(f"  [EV][debug] Failed to write debug tables: {e}")

    # -----------------------------------------------------------------
    # Project back to the original (possibly duplicated) base_index
    # -----------------------------------------------------------------
    return out.reindex(base_index)


def build_per_event_type_features(full_data_index: pd.DatetimeIndex, top_n_events: int = 20) -> pd.DataFrame:
    """
    Build per-event-type EV features for the top N most frequent event types.
    
    Creates features like:
    - CPI_tail_flag_1p5
    - Nonfarm Payrolls_surprise_dispersion_day
    - FOMC Rate Decision_revision_z
    etc.
    
    Args:
        full_data_index: Trading calendar index
        top_n_events: Number of top event types to include (default 20)
    
    Returns:
        DataFrame with per-event-type features for each ticker
    """
    # Building per-event-type features for top N event types
    
    # Get top event types
    cal = _load_calendar()
    if cal.empty:
        return pd.DataFrame(index=full_data_index)
    
    top_events = cal['event_type'].value_counts().head(top_n_events).index.tolist()
    
    # Normalize trading index
    base_index = pd.DatetimeIndex(full_data_index).tz_localize(None)
    uni_index = pd.DatetimeIndex(base_index.drop_duplicates(keep="first"))
    
    # Get all tradable tickers from config
    try:
        from ..config import settings
        tickers = getattr(settings.data, "tradable_tickers", [])
    except:
        tickers = ['AAPL US Equity', 'MSFT US Equity', 'QQQ US Equity']  # fallback
    
    # Build features for each ticker
    all_features = {}
    
    for ticker in tickers:
        ticker_features = {}
        
        for event_type in top_events:
            # Filter calendar for this event type
            event_cal = cal[cal['event_type'] == event_type].copy()
            if event_cal.empty:
                continue
                
            # Process this event type
            event_features = _build_single_event_type_features(
                event_cal, event_type, uni_index, ticker
            )
            ticker_features.update(event_features)
        
        # Event features are global - don't add ticker prefix
        # Only add if not already present (to avoid duplicates across tickers)
        for feature_name, feature_series in ticker_features.items():
            if feature_name not in all_features:
                all_features[feature_name] = feature_series
    
    # Combine all features
    result_df = pd.DataFrame(all_features, index=uni_index)
    return result_df.reindex(base_index).fillna(0.0)


def _build_single_event_type_features(event_cal: pd.DataFrame, event_type: str, 
                                    uni_index: pd.DatetimeIndex, ticker: str) -> dict:
    """Build EV features for a single event type."""
    
    # Clean event type name for feature naming
    clean_name = event_type.replace(" ", "_").replace(".", "").replace(",", "").replace("(", "").replace(")", "")
    
    # Settings
    conf = getattr(settings, "events", object())
    pre = int(getattr(conf, "pre_window_days", DEFAULT_PRE_WINDOW))
    post = int(getattr(conf, "post_window_days", DEFAULT_POST_WINDOW))
    lag = int(getattr(conf, "post_release_lag_days", DEFAULT_POST_LAG))
    hi_rel = float(getattr(conf, "high_relevance_threshold", DEFAULT_HIGH_RELEVANCE))
    
    # Process the event data
    event_cal = event_cal.sort_values("release_day").reset_index(drop=True)
    
    # For single event type, we need to handle the grouped functions differently
    if len(event_cal) > 0:
        # Calculate surprise_z for single event type
        surprise_series = _std_grouped_surprise(event_cal)
        if isinstance(surprise_series, pd.Series):
            event_cal["surprise_z"] = surprise_series
        else:
            event_cal["surprise_z"] = 0.0
            
        # Calculate revision_z for single event type  
        revision_series = _std_grouped_revision(event_cal)
        if isinstance(revision_series, pd.Series):
            event_cal["revision_z"] = revision_series
        else:
            event_cal["revision_z"] = 0.0
    else:
        event_cal["surprise_z"] = 0.0
        event_cal["revision_z"] = 0.0
    event_cal["net_info_surprise"] = event_cal["surprise_z"] + 0.7 * event_cal["revision_z"].fillna(0.0)
    
    # Handle signed_surprise_pct for single event type
    if len(event_cal) > 0:
        signed_pct_series = _expanding_signed_percentile(event_cal, event_cal["surprise_z"])
        if isinstance(signed_pct_series, pd.Series):
            event_cal["signed_surprise_pct"] = signed_pct_series
        else:
            event_cal["signed_surprise_pct"] = 0.0
    else:
        event_cal["signed_surprise_pct"] = 0.0
    
    event_cal["is_high_impact"] = (event_cal["relevance"] >= hi_rel).astype(int)
    event_cal["is_tail_1p5"] = (event_cal["surprise_z"].abs() > 1.5).astype(int)
    event_cal["is_tail_2p5"] = (event_cal["surprise_z"].abs() > 2.5).astype(int)
    
    # Relevance weighting
    w = event_cal["relevance"].clip(0, 100) ** 0.5
    event_cal["w_surprise"] = event_cal["surprise_z"] * w
    event_cal["w_net_info"] = event_cal["net_info_surprise"] * w
    event_cal["w_signed_pct"] = event_cal["signed_surprise_pct"] * w
    
    # Daily aggregation
    daily = event_cal.groupby("release_day").agg({
        "w_surprise": "sum",
        "w_net_info": "sum", 
        "w_signed_pct": "sum",
        "surprise_z": "mean",
        "is_high_impact": "sum",
        "is_tail_1p5": "sum",
        "is_tail_2p5": "sum",
        "relevance": "mean"
    }).fillna(0.0)
    
    # Normalize weights
    daily["surprise_wavg"] = daily["w_surprise"] / daily["relevance"].clip(1.0)
    daily["net_info_wavg"] = daily["w_net_info"] / daily["relevance"].clip(1.0)
    daily["signed_pct_wavg"] = daily["w_signed_pct"] / daily["relevance"].clip(1.0)
    
    # Additional features
    daily["surprise_dispersion_day"] = event_cal.groupby("release_day")["surprise_z"].std()
    daily["tail_any"] = (daily["is_tail_1p5"] > 0).astype(int)
    
    # Reindex to trading calendar
    daily = daily.reindex(uni_index).fillna(0.0)
    
    # Business-day lag helper
    def _lag_bday(s: pd.Series, k: int) -> pd.Series:
        s2 = pd.Series(s.values, index=pd.DatetimeIndex(s.index) + BDay(k))
        return s2[~s2.index.duplicated(keep="last")]
    
    # Apply lag and create features
    features = {}
    
    # Core features
    features[f"{clean_name}_after_surprise_z"] = _lag_bday(daily["surprise_wavg"], lag).reindex(uni_index).fillna(0.0)
    features[f"{clean_name}_after_pos"] = (_lag_bday(daily["surprise_wavg"], lag).reindex(uni_index) > 0).fillna(False).astype(int)
    features[f"{clean_name}_after_neg"] = (_lag_bday(daily["surprise_wavg"], lag).reindex(uni_index) < 0).fillna(False).astype(int)
    features[f"{clean_name}_signed_surprise_percentile"] = _lag_bday(daily["signed_pct_wavg"], lag).reindex(uni_index).fillna(0.0)
    features[f"{clean_name}_surprise_dispersion_day"] = _lag_bday(daily["surprise_dispersion_day"], lag).reindex(uni_index).fillna(0.0)
    features[f"{clean_name}_tail_flag_1p5"] = _lag_bday(daily["is_tail_1p5"], lag).reindex(uni_index).fillna(0).astype(int)
    features[f"{clean_name}_tail_flag_2p5"] = _lag_bday(daily["is_tail_2p5"], lag).reindex(uni_index).fillna(0).astype(int)
    features[f"{clean_name}_net_info_surprise"] = _lag_bday(daily["net_info_wavg"], lag).reindex(uni_index).fillna(0.0)
    
    # Revision features
    if "revision_z" in event_cal.columns:
        rev_daily = event_cal.groupby("release_day")["revision_z"].mean()
        rev_daily = rev_daily[~rev_daily.index.duplicated(keep="last")]
        features[f"{clean_name}_revision_z"] = _lag_bday(rev_daily, lag).reindex(uni_index).fillna(0.0)
        
        rev_tail = event_cal.groupby("release_day")["revision_z"].apply(lambda s: int((s.abs() > 2.0).any()))
        rev_tail = rev_tail[~rev_tail.index.duplicated(keep="last")]
        features[f"{clean_name}_revision_tail_flag"] = _lag_bday(rev_tail, lag).reindex(uni_index).fillna(0).astype(int)
    
    # Calendar features
    high_dates = sorted(pd.to_datetime(event_cal.loc[event_cal["is_high_impact"].eq(1), "release_day"].dropna().unique()))
    if high_dates:
        # Days to next high-impact event
        next_idx = pd.Series(pd.NaT, index=uni_index)
        hi = pd.DatetimeIndex(high_dates)
        j = 0
        for d in uni_index:
            while j < len(hi) and hi[j] < d:
                j += 1
            next_idx.loc[d] = hi[j] if j < len(hi) else pd.NaT
        td = pd.to_timedelta(next_idx - uni_index)
        features[f"{clean_name}_days_to_high"] = pd.Series(td, index=uni_index).dt.days.clip(lower=0).fillna(0).astype(float)
        
        # Event window flags
        in_window = pd.Series(0, index=uni_index)
        pre_window = pd.Series(0, index=uni_index)
        for d in high_dates:
            start_pre = d - pd.Timedelta(days=pre)
            end_post = d + pd.Timedelta(days=post)
            in_window.loc[start_pre:end_post] = 1
            if pre > 0:
                pre_window.loc[start_pre:(d - pd.Timedelta(days=1))] = 1
        features[f"{clean_name}_in_window"] = in_window
        features[f"{clean_name}_pre_window"] = pre_window
    else:
        features[f"{clean_name}_days_to_high"] = pd.Series(0.0, index=uni_index)
        features[f"{clean_name}_in_window"] = pd.Series(0, index=uni_index)
        features[f"{clean_name}_pre_window"] = pd.Series(0, index=uni_index)
    
    return features
