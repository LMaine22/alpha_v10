"""
Event calendar loader + daily EV_* feature builder.
Robust to timezone mismatches (normalizes to ET midnight, tz-naive).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from ..config import settings
from ..features.core import ewma_halflife  # for new EWMA surprise features

ET_TZ = "America/New_York"

# --------------------------
# Helpers
# --------------------------

def _to_et_midnight_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Convert any timestamp (naive or tz-aware) to America/New_York midnight (00:00),
    then drop timezone to return a tz-naive Timestamp.
    """
    if ts.tz is None:
        ts_et = ts.tz_localize(ET_TZ)
    else:
        ts_et = ts.tz_convert(ET_TZ)
    return ts_et.normalize().tz_localize(None)

def _series_to_et_midnight_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(ET_TZ).dt.normalize().dt.tz_localize(None)
    else:
        s = s.dt.tz_localize(ET_TZ).dt.normalize().dt.tz_localize(None)
    return s

def _load_and_normalize_events(path: str) -> pd.DataFrame | None:
    """Load merged parquet and create a tz-naive 'date_naive' column aligned to ET midnight."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"  Warning: Event calendar file not found at '{path}'.")
        return None

    # Required columns: release_datetime, event_type, country, relevance, actual, survey
    required = {"release_datetime", "event_type", "country", "relevance", "actual", "survey"}
    missing = required.difference(df.columns)
    if missing:
        print(f"  Warning: Missing columns in events file: {sorted(missing)}")
        return None

    rel = pd.to_datetime(df["release_datetime"], errors="coerce")
    if getattr(rel.dt, "tz", None) is not None:
        df["date_naive"] = rel.dt.tz_convert(ET_TZ).dt.normalize().dt.tz_localize(None)
    else:
        df["date_naive"] = rel.dt.tz_localize(ET_TZ).dt.normalize().dt.tz_localize(None)

    df = df.dropna(subset=["date_naive"]).copy()
    return df

def _filter_events(df: pd.DataFrame, conf) -> pd.DataFrame:
    """Apply country, type, and relevance filters."""
    out = df.copy()
    if getattr(conf, "countries", None):
        out = out[out["country"].isin(conf.countries)]
    include_types = getattr(conf, "include_types", None)
    if include_types:
        out = out[out["event_type"].isin(include_types)]
    thr = float(getattr(conf, "high_relevance_threshold", 70.0))
    out = out[out["relevance"] >= thr]
    return out

def _get_high_impact_dates_naive(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Unique ET-midnight (tz-naive) high impact event dates, sorted."""
    if df.empty:
        return []
    return sorted(pd.DatetimeIndex(df["date_naive"].unique()).tz_localize(None))

def _daily_surprise_z(df: pd.DataFrame) -> pd.Series:
    """
    Surprise z per event_type (within filtered set), then pick the max |z| per day.
    Index returned = tz-naive ET dates.
    """
    if df.empty:
        return pd.Series(dtype=float)
    work = df.copy()
    work["surprise"] = pd.to_numeric(work["actual"], errors="coerce") - pd.to_numeric(work["survey"], errors="coerce")
    work = work.dropna(subset=["surprise", "event_type", "date_naive"])
    if work.empty:
        return pd.Series(dtype=float)
    grp = work.groupby("event_type")["surprise"]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0.0, np.nan)
    work["surprise_z"] = (work["surprise"] - mu) / sd
    work = work.dropna(subset=["surprise_z"])
    if work.empty:
        return pd.Series(dtype=float)
    work["abs_z"] = work["surprise_z"].abs()
    best = work.sort_values("abs_z", ascending=False).drop_duplicates("date_naive")
    s = best.set_index("date_naive")["surprise_z"].sort_index()
    return s

def _dense_count_next_days(df: pd.DataFrame, by_day: pd.Series, days: int = 7) -> pd.Series:
    """
    For each day in the base index, count number of high-impact events in next N calendar days.
    """
    base_days = pd.DatetimeIndex(by_day.index).sort_values().unique()
    out = []
    for dt in base_days:
        horizon = [dt + pd.Timedelta(days=i) for i in range(1, days + 1)]
        out.append(sum(by_day.reindex(horizon).fillna(0).values))
    return pd.Series(out, index=base_days, dtype=float)

# --------------------------
# Public API
# --------------------------

def build_event_features(full_data_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build daily EV_* features aligned to the master index (tz-naive).
    - EV_days_to_high: days until next high-impact event (capped)
    - EV_in_window: 1 if in [T-pre, T+post] around any high-impact event
    - EV_pre_window: 1 if in [T-pre, T-1]
    - EV_is_event_week: 1 if any high-impact event in the same ISO week
    - EV_after_surprise_z: standardized surprise (posted on T+lag business days)
    - EV_after_pos/EV_after_neg: sign of surprise (T+lag)
    - EV_surprise_ewma_21 / EV_surprise_ewma_63: EWMAs of the daily surprise z (post-release)
    - EV_tail_intensity_21: share of days with |surprise_z|>2 over the past 21 days (post-release)
    - EV_dense_macro_window_7: count of high-impact events in next 7 calendar days (calendar-only)
    """
    print("Building event features from economic calendar...")

    conf = settings.events
    events_df = _load_and_normalize_events(conf.file_path)
    if events_df is None:
        return pd.DataFrame(index=full_data_index)

    # Apply filters
    filt = _filter_events(events_df, conf)
    if filt.empty:
        print("  Warning: No high-relevance events found after filtering. Returning empty DataFrame.")
        return pd.DataFrame(index=full_data_index)

    # Ensure master index is tz-naive
    if getattr(full_data_index, "tz", None) is not None:
        base_index = pd.DatetimeIndex(full_data_index.tz_localize(None))
    else:
        base_index = pd.DatetimeIndex(full_data_index)

    # High impact dates (tz-naive)
    high_dates = _get_high_impact_dates_naive(filt)

    out = pd.DataFrame(index=base_index)

    # --- EV_days_to_high via asof on sorted indices ---
    if high_dates:
        events_only = pd.DataFrame(index=pd.DatetimeIndex(high_dates), data={"event_date": high_dates})
        events_only.sort_index(inplace=True)
        out_sorted = out.sort_index()
        merged = pd.merge_asof(
            out_sorted,
            events_only,
            left_index=True,
            right_index=True,
            direction="forward",
            allow_exact_matches=True,
        )
        out["EV_days_to_high"] = (merged["event_date"] - merged.index).dt.days
        cap = int(getattr(conf, "max_countdown_cap_days", 10))
        out["EV_days_to_high"] = out["EV_days_to_high"].clip(upper=cap).fillna(cap)
    else:
        out["EV_days_to_high"] = int(getattr(conf, "max_countdown_cap_days", 10))

    # --- EV_in_window / EV_pre_window flags ---
    pre = int(getattr(conf, "pre_window_days", 2))
    post = int(getattr(conf, "post_window_days", 2))
    out["EV_in_window"] = 0
    out["EV_pre_window"] = 0
    if high_dates:
        for d in high_dates:
            start_pre = d - pd.Timedelta(days=pre)
            end_post = d + pd.Timedelta(days=post)
            out.loc[start_pre:end_post, "EV_in_window"] = 1
            if pre > 0:
                out.loc[start_pre : (d - pd.Timedelta(days=1)), "EV_pre_window"] = 1

    # --- EV_is_event_week ---
    out["week_period"] = out.index.to_period("W")
    event_weeks = pd.Series(pd.DatetimeIndex(high_dates)).dt.to_period("W").unique()
    out["EV_is_event_week"] = out["week_period"].isin(event_weeks).astype(int)
    out.drop(columns=["week_period"], inplace=True)

    # --- Post-release: surprise z at T+lag (business days) ---
    sz = _daily_surprise_z(filt)
    if not sz.empty:
        lag = int(getattr(conf, "post_release_lag_days", 1))
        sz_t1 = pd.Series(sz.values, index=sz.index + BDay(lag))
        # Align to base index
        sz_t1 = sz_t1.reindex(out.index)
        out["EV_after_surprise_z"] = sz_t1.fillna(0.0)
        out["EV_after_pos"] = (out["EV_after_surprise_z"] > 0).astype(int)
        out["EV_after_neg"] = (out["EV_after_surprise_z"] < 0).astype(int)

        # Surprise EWMAs (post-release)
        out["EV_surprise_ewma_21"] = ewma_halflife(out["EV_after_surprise_z"], halflife=21)
        out["EV_surprise_ewma_63"] = ewma_halflife(out["EV_after_surprise_z"], halflife=63)

        # Tail intensity over last 21 business days (post-release)
        absz = out["EV_after_surprise_z"].abs()
        out["EV_tail_intensity_21"] = (absz > 2.0).astype(float).rolling(21).mean().fillna(0.0)
    else:
        out["EV_after_surprise_z"] = 0.0
        out["EV_after_pos"] = 0
        out["EV_after_neg"] = 0
        out["EV_surprise_ewma_21"] = 0.0
        out["EV_surprise_ewma_63"] = 0.0
        out["EV_tail_intensity_21"] = 0.0

    # --- Dense macro window: count of high-impact events in next 7 calendar days (calendar-only) ---
    if high_dates:
        by_day = pd.Series(1, index=pd.DatetimeIndex(high_dates))
        dense7 = []
        # preserve full_data_index alignment
        for dt in out.index:
            horizon = [dt + pd.Timedelta(days=i) for i in range(1, 8)]
            dense7.append(sum(by_day.reindex(horizon).fillna(0).values))
        out["EV_dense_macro_window_7"] = pd.Series(dense7, index=out.index, dtype=float)
    else:
        out["EV_dense_macro_window_7"] = 0.0

    print(f"  Successfully built {len(out.columns)} event features.")
    return out
