from __future__ import annotations

import re
import math
import warnings
from typing import Optional, Tuple, Dict, List

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
    "data_store/processed/economic_releases.parquet",
    "data_store/processed/economic_releases.csv",
    "data_store/processed/economic_combined.parquet",
    "data_store/processed/economic_combined.csv",
    "data_store/economic_releases.parquet",
    "data_store/economic_releases.csv",
    "data_store/economic_combined.parquet",
    "data_store/economic_combined.csv",
)

# Core engine constants
DEFAULT_HALFLIFES: Tuple[float, float, float] = (10.5, 21, 63)
DEFAULT_ROLL_WINDOWS: Tuple[int, int, int, int] = (3, 5, 10, 21)
DEFAULT_Z_RELEASE_WINDOWS: Tuple[int, int, int] = (6, 12, 24)

# Allow project settings to override defaults (optional)
try:
    if hasattr(settings, "events"):
        if getattr(settings.events, "halflifes", None):
            DEFAULT_HALFLIFES = tuple(getattr(settings.events, "halflifes"))
        if getattr(settings.events, "roll_windows", None):
            DEFAULT_ROLL_WINDOWS = tuple(getattr(settings.events, "roll_windows"))
        if getattr(settings.events, "z_release_windows", None):
            DEFAULT_Z_RELEASE_WINDOWS = tuple(getattr(settings.events, "z_release_windows"))
        if getattr(settings.events, "calendar_paths", None):
            DEFAULT_CAL_PATHS = tuple(getattr(settings.events, "calendar_paths"))
except Exception:
    pass

# --- Top events for proximity comes next ---

# Top-tier events we’ll often reference in proximity counters
TOP_EVENTS_FOR_PROXIMITY = [
    "CPI MoM", "CPI YoY", "Core CPI MoM", "Core CPI YoY",
    "PCE Price Index MoM", "Core PCE Price Index MoM",
    "PPI Final Demand MoM", "PPI Final Demand YoY",
    "ADP Employment Change", "Change in Nonfarm Payrolls", "Change in Private Payrolls",
    "Unemployment Rate", "Average Hourly Earnings MoM", "Average Hourly Earnings YoY",
    "Retail Sales Advance MoM", "Retail Sales Control Group",
    "Durable Goods Orders", "Durables Ex Transportation",
    "Cap Goods Orders Nondef Ex Air", "Industrial Production MoM",
    "GDP Annualized QoQ", "GDP Price Index",
    "ISM Manufacturing", "ISM Services Index", "ISM Prices Paid", "ISM Services Prices Paid",
    "Housing Starts", "Building Permits", "Existing Home Sales", "New Home Sales",
    "FHFA House Price Index MoM",
    "FOMC Rate Decision (Upper Bound)", "FOMC Rate Decision (Lower Bound)",
]

# --------------------------------------------------------------------------------------
# Event category mapping (Inflation/Labor/Growth/Housing/Sentiment)
# Includes exact-name map + regex-based fallback classifier
# --------------------------------------------------------------------------------------

CATEGORY_MAP: Dict[str, str] = {
    # --- Inflation ---
    "CPI Core Index SA": "inflation",
    "CPI Ex Food and Energy MoM": "inflation",
    "CPI Ex Food and Energy YoY": "inflation",
    "CPI Index NSA": "inflation",
    "CPI MoM": "inflation",
    "CPI YoY": "inflation",
    "Core PCE Price Index MoM": "inflation",
    "Core PCE Price Index QoQ": "inflation",
    "Core PCE Price Index YoY": "inflation",
    "Export Price Index MoM": "inflation",
    "Export Price Index YoY": "inflation",
    "GDP Price Index": "inflation",
    "ISM Prices Paid": "inflation",
    "ISM Services Prices Paid": "inflation",
    "Import Price Index MoM": "inflation",
    "Import Price Index YoY": "inflation",
    "Import Price Index ex Petroleum MoM": "inflation",
    "PCE Price Index MoM": "inflation",
    "PCE Price Index YoY": "inflation",
    "PPI Ex Food and Energy MoM": "inflation",
    "PPI Ex Food and Energy YoY": "inflation",
    "PPI Ex Food, Energy, Trade MoM": "inflation",
    "PPI Ex Food, Energy, Trade YoY": "inflation",
    "PPI Final Demand MoM": "inflation",
    "PPI Final Demand YoY": "inflation",

    # --- Labor ---
    "ADP Employment Change": "labor",
    "Average Hourly Earnings MoM": "labor",
    "Average Hourly Earnings YoY": "labor",
    "Average Weekly Hours All Employees": "labor",
    "Challenger Job Cuts YoY": "labor",
    "Change in Manufact. Payrolls": "labor",
    "Change in Nonfarm Payrolls": "labor",
    "Change in Private Payrolls": "labor",
    "Continuing Claims": "labor",
    "Employment Cost Index (ECI)": "labor",
    "Initial Claims 4-Wk Moving Avg": "labor",
    "Initial Jobless Claims": "labor",
    "ISM Employment": "labor",
    "ISM Services Employment": "labor",
    "JOLTS Job Openings": "labor",
    "JOLTS Job Openings Rate": "labor",
    "JOLTS Layoffs Level": "labor",
    "JOLTS Layoffs Rate": "labor",
    "JOLTS Quits Level": "labor",
    "JOLTS Quits Rate": "labor",
    "Labor Force Participation Rate": "labor",
    "Nonfarm Payrolls 3-Mo Avg Chg": "labor",
    "Nonfarm Productivity": "labor",
    "Unemployment Rate": "labor",
    "Unit Labor Costs": "labor",

    # --- Growth / Activity ---
    "Advance Goods Exports MoM SA": "growth",
    "Advance Goods Imports MoM SA": "growth",
    "Advance Goods Trade Balance": "growth",
    "Auto Sales": "growth",
    "Business Inventories": "growth",
    "Capacity Utilization": "growth",
    "Chicago Fed Nat Activity Index": "growth",
    "Chicago PMI": "growth",
    "Consumer Credit": "growth",
    "Cap Goods Orders Nondef Ex Air": "growth",
    "Cap Goods Shipments Nondef Ex Air": "growth",
    "Current Account Balance": "growth",
    "Dallas Fed Manf. Activity": "growth",
    "Dallas Fed Services Activity": "growth",
    "Durable Goods Orders": "growth",
    "Durables Ex Transportation": "growth",
    "Factory Orders": "growth",
    "GDP Annualized QoQ": "growth",
    "Industrial Production MoM": "growth",
    "Inventories": "growth",
    "ISM Manufacturing": "growth",
    "ISM New Orders": "growth",
    "ISM Services Index": "growth",
    "ISM Services New Orders": "growth",
    "Kansas City Fed Manf. Activity": "growth",
    "Kansas City Fed Services Activity": "growth",
    "Leading Index": "growth",
    "NFIB Small Business Optimism": "growth",
    "NY Fed 2Q Report on Household Debt and Credit": "growth",
    "NY Fed Services Business Activity": "growth",
    "Personal Income": "growth",
    "Personal Spending": "growth",
    "Real Personal Spending": "growth",
    "Philadelphia Fed Business Outlook": "growth",
    "Philadelphia Fed Non-Manufacturing Activity": "growth",
    "PMI Manufacturing": "growth",
    "PMI Services": "growth",
    "PMI Composite": "growth",
    "Retail Sales Advance MoM": "growth",
    "Retail Sales Control Group": "growth",
    "Retail Sales Ex Auto MoM": "growth",
    "Retail Sales Ex Auto and Gas": "growth",
    "S&P Global PMI Manufacturing": "growth",
    "S&P Global PMI Services": "growth",
    "S&P Global PMI Composite": "growth",
    "Shipments": "growth",
    "Orders": "growth",
    "Trade Balance": "growth",
    "Vehicle Sales": "growth",
    "Wholesale Inventories": "growth",
    "Wholesale Sales": "growth",

    # --- Housing ---
    "Building Permits": "housing",
    "Building Permits MoM": "housing",
    "Construction Spending MoM": "housing",
    "Existing Home Sales": "housing",
    "Existing Home Sales MoM": "housing",
    "FHFA House Price Index MoM": "housing",
    "Housing Starts": "housing",
    "Housing Starts MoM": "housing",
    "NAHB Housing Market Index": "housing",
    "New Home Sales": "housing",
    "New Home Sales MoM": "housing",
    "Pending Home Sales": "housing",
    "Pending Home Sales MoM": "housing",
    "Pending Home Sales NSA YoY": "housing",
    "Residential Construction": "housing",
    "House Price Index": "housing",

    # --- Sentiment ---
    "Conf. Board Consumer Confidence": "sentiment",
    "Conf. Board Expectations": "sentiment",
    "Conf. Board Present Situation": "sentiment",
    "Langer Economic Expectations": "sentiment",
    "NY Fed 1-Yr Inflation Expectations": "sentiment",
    "U. of Mich. 1 Yr Inflation": "sentiment",
    "U. of Mich. 5–10 Yr Inflation": "sentiment",
    "U. of Mich. Current Conditions": "sentiment",
    "U. of Mich. Expectations": "sentiment",
    "U. of Mich. Sentiment": "sentiment",
}

# Regex fallback rules for unmapped names (order matters)
CATEGORY_REGEX_RULES: List[Tuple[str, str]] = [
    (r"\b(CPI|PCE|PPI|Import Price|Export Price|Prices Paid|GDP Price)\b", "inflation"),
    (r"\b(Nonfarm|Payroll|ADP|Unemployment|Claims|JOLTS|AHE|Employment)\b", "labor"),
    (r"\b(ISM|PMI|Retail Sales|Durable|Cap Goods|Industrial Production|GDP|Orders|Shipments|Inventor|Trade Balance|Vehicle Sales|NFIB|Chicago PMI|Dallas|Kansas|Philadelphia|Richmond|S&P Global)\b", "growth"),
    (r"\b(Housing|Home Sales|Building Permit|Starts|NAHB|Construction|HPI|FHFA)\b", "housing"),
    (r"\b(Consumer Confidence|Expectations|Sentiment|NY Fed .*Inflation)\b", "sentiment"),
]

POLICY_EVENTS = {
    "FOMC Rate Decision (Upper Bound)",
    "FOMC Rate Decision (Lower Bound)",
    "FOMC Meeting Minutes",
    "Fed Releases Beige Book",
    "Federal Budget Balance",
    "Fed Interest on Reserve Balances Rate",
    "Fed Reverse Repo Rate",
}

ADMIN_META_KEYWORDS = [
    "Revisions", "Annual Revision", "Released Live on the Web", "Postponed", "Shutdown",
    "Resumes publication", "Methodology", "Benchmark", "Special Note"
]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _cap(x, lo=None, hi=None):
    if lo is not None:
        x = np.maximum(x, lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x

def _safe_div(a, b, eps=1e-12):
    return a / np.where(np.abs(b) < eps, np.sign(b) * eps + eps, b)

def _to_date_et_midnight(ts: pd.Series) -> pd.Series:
    # Accept date-like strings/datetimes; normalize to naive dates (ET midnight)
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    if s.dt.tz is not None:
        s = s.dt.tz_convert(ET_TZ)
    s = s.dt.tz_localize(None)
    return s.dt.floor("D")

def _ewm_halflife(s: pd.Series, halflife: float) -> pd.Series:
    return s.ewm(halflife=halflife, adjust=False, ignore_na=True).mean()


def _z_by_event_release(df: pd.DataFrame, value_col: str, event_col: str, lookback_releases: int) -> pd.Series:
    """
    Z-score per event type across last N releases (counted by release index).
    Returns a Series aligned 1:1 with df.index.
    """
    s = pd.to_numeric(df[value_col], errors="coerce")
    groups = df[event_col]
    minp = max(3, min(lookback_releases, 3))

    def _f(g: pd.Series) -> pd.Series:
        r = g.rolling(lookback_releases, min_periods=minp)
        mu = r.mean()
        sd = r.std(ddof=0)
        return _safe_div(g - mu, sd)

    # group-by the series, not the whole DataFrame → Series out
    out = s.groupby(groups, sort=False, dropna=False).apply(_f)
    # drop the group level added by apply
    out.index = out.index.droplevel(0)
    return out.reindex(df.index)

def _pct_by_event_release(df: pd.DataFrame, value_col: str, event_col: str, lookback_releases: int) -> pd.Series:
    """
    Percentile per event over last N releases (0..1) for the latest value in each rolling window.
    Returns a Series aligned 1:1 with df.index.
    """
    s = pd.to_numeric(df[value_col], errors="coerce")
    groups = df[event_col]
    minp = max(3, min(lookback_releases, 3))

    def _last_pct(g: pd.Series) -> pd.Series:
        r = g.rolling(lookback_releases, min_periods=minp)
        return r.apply(
            lambda window: (pd.Series(window).rank(pct=True)).iloc[-1]
            if pd.notna(pd.Series(window).iloc[-1]) else np.nan,
            raw=False
        )

    out = s.groupby(groups, sort=False, dropna=False).apply(_last_pct)
    out.index = out.index.droplevel(0)
    return out.reindex(df.index)

def _label_category(name: str) -> Optional[str]:
    if name in CATEGORY_MAP:
        return CATEGORY_MAP[name]
    # regex fallbacks
    for pat, cat in CATEGORY_REGEX_RULES:
        if re.search(pat, str(name), flags=re.IGNORECASE):
            return cat
    return None

def _is_admin_meta(name: str) -> bool:
    s = str(name) if name is not None else ""
    return any(k.lower() in s.lower() for k in ADMIN_META_KEYWORDS)

def _calendar_path() -> Optional[str]:
    for p in DEFAULT_CAL_PATHS:
        try:
            import os
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return None

# --------------------------------------------------------------------------------------
# Load & normalize calendar
# --------------------------------------------------------------------------------------

REQUIRED_COLS = [
    "Date",           # release datetime (string or datetime)
    "Event",          # event type name
    "Actual",
    "Survey",         # median
    "Prior",
    "Revised",        # optional
    "Relevance",      # numeric (0..100)
    "Country",        # 'US' expected; not strictly required
    # Optional but very useful:
    "Survey High", "Survey Low", "Survey Mean",
    "Release Time",   # "08:30", "10:00", etc. (local ET or with tz)
    "SA/NSA", "Unit", "Frequency", "MoM/YoY/Level",
    "Release Status", "Methodology Change", "Benchmark Revision", "Special Note",
]

def _load_calendar_df(calendar_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if calendar_df is None:
        p = _calendar_path()
        if p is None:
            raise FileNotFoundError("No calendar file found in DEFAULT_CAL_PATHS and no DataFrame provided.")
        if p.lower().endswith(".parquet"):
            calendar_df = pd.read_parquet(p)
        else:
            calendar_df = pd.read_csv(p)

    df = calendar_df.copy()

    # Normalize columns if users have different headers
    rename_map = {
        "Event Name": "Event",
        "ReleaseDate": "Date",
        "Release Time (ET)": "Release Time",
        "Survey Median": "Survey",
        "SurveyHigh": "Survey High",
        "SurveyLow": "Survey Low",
        "SurveyMean": "Survey Mean",
        "Country Code": "Country",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Required minimal set
    for col in ["Date", "Event", "Actual", "Survey", "Prior", "Relevance"]:
        if col not in df.columns:
            df[col] = np.nan

    # Normalize date to ET-midnight naive date (for daily grouping)
    df["ReleaseDate"] = _to_date_et_midnight(df["Date"])

    # Categorize events
    df["Category"] = df["Event"].apply(_label_category)

    # Filter admin/meta-only lines to a META flag
    df["IsAdminMeta"] = df["Event"].apply(_is_admin_meta)

    # Numeric casting
    for c in ["Actual", "Survey", "Prior", "Revised", "Relevance",
              "Survey High", "Survey Low", "Survey Mean"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure relevance bounded and sqrt-weight ready
    df["Relevance"] = _cap(df["Relevance"], 0, 100)
    df["w"] = np.sqrt(df["Relevance"].fillna(0.0))

    # Basic per-release fields
    df["raw_surp.as"] = df["Actual"] - df["Survey"]
    df["raw_surp.ap"] = df["Actual"] - df["Prior"]
    df["raw_surp.ar"] = df["Actual"] - df["Revised"] if "Revised" in df.columns else np.nan

    # Survey band features (use quantile fallback if High/Low missing)
    have_bands = ("Survey High" in df.columns) and ("Survey Low" in df.columns)
    if have_bands:
        width = (df["Survey High"] - df["Survey Low"]).replace(0, np.nan)
        df["band_pos"] = (df["Actual"] - df["Survey"]) / width
        df["band_pos"] = df["band_pos"].clip(-5, 5)
        df["tail_hit_low_flag"] = ((df["Actual"] < df["Survey Low"]).astype(float))
        df["tail_hit_high_flag"] = ((df["Actual"] > df["Survey High"]).astype(float))
        df["share_of_range"] = (df["Actual"] - df["Survey"]) / width
        df["share_of_range"] = df["share_of_range"].clip(-5, 5)
    else:
        df["band_pos"] = np.nan
        df["tail_hit_low_flag"] = np.nan
        df["tail_hit_high_flag"] = np.nan
        df["share_of_range"] = np.nan

    # Per-event z & percentile over last N releases
    for k in DEFAULT_Z_RELEASE_WINDOWS:
        df[f"z_surp.N{k}"] = _z_by_event_release(df, "raw_surp.as", "Event", k)
        df[f"pct_surp.N{k}"] = _pct_by_event_release(df, "raw_surp.as", "Event", k)

    # A conservative polarity map placeholder (extend/replace if you maintain an asset-specific mapping)
    # +1 if positive surprise is typically risk-on for equities; -1 if risk-off.
    POLARITY_DEFAULTS = {
        "Unemployment Rate": -1,
        "Initial Jobless Claims": -1,
        "Continuing Claims": -1,
        "Average Hourly Earnings MoM": -1,  # wage inflation
        "Average Hourly Earnings YoY": -1,
        "CPI MoM": -1,
        "CPI YoY": -1,
        "Core CPI MoM": -1,
        "Core CPI YoY": -1,
        "PCE Price Index MoM": -1,
        "Core PCE Price Index MoM": -1,
        "PPI Final Demand MoM": -1,
        "PPI Final Demand YoY": -1,
        "GDP Price Index": -1,
        "ISM Prices Paid": -1,
    }
    df["polarity"] = df["Event"].map(POLARITY_DEFAULTS).fillna(1.0)
    df["polarity_adjusted_z"] = df["polarity"] * df["z_surp.N12"]

    # Revision features
    df["rev.first"] = df["Revised"] - df["Prior"] if "Revised" in df.columns else np.nan
    for k in DEFAULT_Z_RELEASE_WINDOWS:
        if "rev.first" in df.columns:
            df[f"rev.abs_z.N{k}"] = _z_by_event_release(df, "rev.first", "Event", k).abs()

    return df


# --------------------------------------------------------------------------------------
# Daily aggregation & feature construction
# --------------------------------------------------------------------------------------

def _wavg(x, w) -> float:
    xs = pd.to_numeric(pd.Series(x).squeeze(), errors="coerce")
    ws = pd.to_numeric(pd.Series(w).squeeze(), errors="coerce")
    xs, ws = xs.align(ws, join="inner")
    m = xs.notna() & ws.notna() & (ws > 0)
    if not m.any():
        return np.nan
    sw = ws[m].sum()
    if not np.isfinite(sw) or sw <= 0:
        return np.nan
    return float(np.dot(xs[m].values, ws[m].values) / sw)

def _wavg_by_mask(g: pd.DataFrame, mask_global: pd.Series, value_col: str, w_col: str = "w") -> float:
    m = mask_global.reindex(g.index).fillna(False)
    if not m.any():
        return np.nan
    return _wavg(g.loc[m, value_col], g.loc[m, w_col])

def _daily_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-release features to daily level (no lag applied inside;
    BDay+1 shift is applied in the final compositor).
    """
    # Only non-admin events contribute to EV signals
    # Apply infer_objects to avoid the future downcast warning
    admin_mask = df["IsAdminMeta"].infer_objects(copy=False).fillna(False)
    d = df[~admin_mask].copy()

    # per-day book-keeping
    grp = d.groupby("ReleaseDate", sort=True)

    out = pd.DataFrame(index=grp.size().index)

    # Breadth & dispersion (same-day)
    out["EV_dense_macro_day_score"] = grp["w"].sum()
    # Weighted means of surprise z and polarity-adjusted
    # Use a simple aggregation approach that returns scalars
    z_vals = []
    pol_vals = []
    for date, group in grp:
        z_vals.append(_wavg(group["z_surp.N12"], group["w"]))
        pol_vals.append(_wavg(group["polarity_adjusted_z"], group["w"]))
    
    out["EV_after_surprise_z"] = pd.Series(z_vals, index=out.index)
    out["EV_polarity_adjusted"] = pd.Series(pol_vals, index=out.index)
    # Signed percentile: map percentile to [-1, +1] using sign of raw surprise
    signed_pct = np.where(d["raw_surp.as"]>=0, d["pct_surp.N12"], 1 - d["pct_surp.N12"])
    d["_signed_pct"] = signed_pct
    # Use manual iteration for signed percentile too
    signed_pct_vals = []
    for date, group in grp:
        signed_pct_vals.append(_wavg(group["_signed_pct"], group["w"]))
    out["EV_signed_surprise_percentile"] = pd.Series(signed_pct_vals, index=out.index)

    # Dispersion of same-day z
    out["EV_surprise_dispersion_day"] = grp["z_surp.N12"].std(ddof=0)

    # Revision composites
    if "rev.abs_z.N12" in d.columns:
        # Manual iteration for revision z
        rev_z_vals = []
        for date, group in grp:
            rev_z_vals.append(_wavg(group["rev.abs_z.N12"], group["w"]))
        out["EV_revision_z"] = pd.Series(rev_z_vals, index=out.index)
        # revision conflict: sign(rev) != sign(surp) -> +1; same -> -1; undefined -> 0
        rc = np.sign(d["rev.first"]) * np.sign(d["raw_surp.as"])
        rc = np.where(np.isnan(rc), 0, np.where(rc < 0, 1, -1))
        d["_rev_conflict"] = rc
        # Manual iteration for revision conflict
        rev_conflict_vals = []
        for date, group in grp:
            rev_conflict_vals.append(_wavg(group["_rev_conflict"], group["w"]))
        out["EV_revision_conflict"] = pd.Series(rev_conflict_vals, index=out.index)
    else:
        out["EV_revision_z"] = np.nan
        out["EV_revision_conflict"] = np.nan

    # Tail flags: share of |z| >= 2
    d["_tail_flag"] = (d["z_surp.N12"].abs() >= 2).astype(float)
    out["EV_tail_share"] = grp["_tail_flag"].mean()

    # Top-tier dominance: if one event dominates weight
    # (we can approximate by Herfindahl or max weight share)
    def _max_weight_share(g):
        w = g["w"].fillna(0.0)
        tot = w.sum()
        return (w.max() / tot) if tot > 0 else np.nan
    # Manual iteration for top tier dominance
    dominance_vals = []
    for date, group in grp:
        dominance_vals.append(_max_weight_share(group))
    out["EV_top_tier_dominance_share"] = pd.Series(dominance_vals, index=out.index)

    # Buckets/composites by category
    for cat in ["inflation", "labor", "growth", "housing", "sentiment"]:
        m = d["Category"] == cat
        key = f"EV.bucket_{cat}_surp"
        if m.any():
            # Manual iteration for category buckets
            cat_vals = []
            for date, group in grp:
                cat_mask = m.reindex(group.index).fillna(False)
                if cat_mask.any():
                    cat_vals.append(_wavg(group.loc[cat_mask, "z_surp.N12"], group.loc[cat_mask, "w"]))
                else:
                    cat_vals.append(np.nan)
            out[key] = pd.Series(cat_vals, index=out.index)
        else:
            out[key] = np.nan

    # Bucket divergence/conflict: sign(mean inflation) * sign(mean growth)
    def _bucket_sign(g, cat):
        mg = g["Category"] == cat
        if mg.any():
            v = _wavg(g.loc[mg, "z_surp.N12"], g.loc[mg, "w"])
            return np.sign(v) if pd.notna(v) and v != 0 else 0.0
        return 0.0
    def _diverge(g):
        s_infl = _bucket_sign(g, "inflation")
        s_grow = _bucket_sign(g, "growth")
        prod = s_infl * s_grow
        if pd.isna(prod):
            return np.nan
        return -1.0 if prod < 0 else (1.0 if prod > 0 else 0.0)
    # Manual iteration for bucket divergence
    divergence_vals = []
    for date, group in grp:
        divergence_vals.append(_diverge(group))
    out["EV.bucket_divergence"] = pd.Series(divergence_vals, index=out.index)

    # Tail share within inflation bucket
    def _infl_tail_share(g):
        mg = g["Category"] == "inflation"
        if mg.any():
            return (g.loc[mg, "z_surp.N12"].abs() >= 2).mean()
        return np.nan
    # Manual iteration for inflation tail share
    infl_tail_vals = []
    for date, group in grp:
        infl_tail_vals.append(_infl_tail_share(group))
    out["EV.bucket_inflation_tail_share"] = pd.Series(infl_tail_vals, index=out.index)

    return out.sort_index()


# --------------------------------------------------------------------------------------
# Sequencing & calendar proximity layers
# --------------------------------------------------------------------------------------

def _forward_calendar_features(df_rel: pd.DataFrame) -> pd.DataFrame:
    """
    Build pre-event forward heat / vacuum / proximity counters.
    Unshifted by design (no leak).
    """
    # Build a daily schedule table with relevance by day
    # Apply infer_objects to avoid the future downcast warning
    admin_mask = df_rel["IsAdminMeta"].infer_objects(copy=False).fillna(False)
    sch = df_rel[~admin_mask].copy()
    day = sch.groupby("ReleaseDate")["w"].sum().rename("day_weight")
    
    # Check if we have valid dates
    if day.empty or day.index.isnull().all():
        # Return empty DataFrame with expected structure
        cal = pd.DataFrame(index=pd.date_range('2018-01-01', '2025-01-01', freq='D'))
        cal["day_weight"] = 0.0
    else:
        valid_dates = day.index.dropna()
        if len(valid_dates) == 0:
            # Return empty DataFrame with expected structure
            cal = pd.DataFrame(index=pd.date_range('2018-01-01', '2025-01-01', freq='D'))
            cal["day_weight"] = 0.0
        else:
            idx = pd.date_range(valid_dates.min(), valid_dates.max(), freq="D")
            cal = pd.DataFrame(index=idx)
            cal["day_weight"] = day.reindex(idx).fillna(0.0)

    # forward heat over {1,3,5,10} calendar days
    for k in (1, 3, 5, 10):
        cal[f"EV_forward_calendar_heat_{k}"] = cal["day_weight"].rolling(k, min_periods=1).sum().shift(-0)

    # vacuum flags when no high-weight events in horizon (threshold=0 here; you can raise to 5, etc.)
    for k in (5, 7, 10):
        cal[f"EV_calendar_vacuum_{k}"] = (cal["day_weight"].rolling(k, min_periods=1).sum() == 0).astype(float)

    # ISO-week density: count of high-weight days within the same week (Mon-Sun)
    # We'll approximate with 7-day rolling sum centered on each day
    cal["EV_week_density"] = cal["day_weight"].rolling(7, min_periods=1).apply(lambda s: (s > 0).sum(), raw=True)

    # Days-to-next primary releases
    cal["days_to_next_CPI"] = _days_to_next_event(df_rel, "CPI")
    cal["days_to_next_NFP"] = _days_to_next_event(df_rel, "Payrolls|Nonfarm")
    cal["days_to_next_FOMC"] = _days_to_next_event(df_rel, "FOMC")
    cal["within_payrolls_week_flag"] = _is_payrolls_week(df_rel).reindex(cal.index).fillna(0.0)

    return cal


def _days_to_next_event(df_rel: pd.DataFrame, pattern: str) -> pd.Series:
    # Mark event days matching pattern
    match = df_rel["Event"].astype(str).str.contains(pattern, case=False, regex=True)
    event_days = df_rel.loc[match, "ReleaseDate"].drop_duplicates().sort_values()
    if event_days.empty:
        return pd.Series(index=pd.Index([], dtype="datetime64[ns]"), dtype=float)
    # For each calendar day in span, compute days until next event day
    idx = pd.date_range(df_rel["ReleaseDate"].min(), df_rel["ReleaseDate"].max(), freq="D")
    s = pd.Series(index=idx, dtype=float)
    next_idx = 0
    next_day = event_days.iloc[next_idx]
    for d in idx:
        while next_idx < len(event_days) and next_day < d:
            next_idx += 1
            if next_idx >= len(event_days):
                next_day = pd.NaT
                break
            next_day = event_days.iloc[next_idx]
        s.loc[d] = (next_day - d).days if pd.notna(next_day) else np.nan
        if s.loc[d] < 0:
            s.loc[d] = 0.0
    return s


def _is_payrolls_week(df_rel: pd.DataFrame) -> pd.Series:
    # Payrolls week: Thursday claims + Friday NFP in same ISO week (approx)
    
    # Check if we have valid release dates
    release_dates = df_rel["ReleaseDate"].dropna()
    if len(release_dates) == 0:
        # Return empty series with default date range
        idx = pd.date_range('2018-01-01', '2025-01-01', freq='D')
        return pd.Series(0.0, index=idx)
    
    idx = pd.date_range(release_dates.min(), release_dates.max(), freq="D")
    s = pd.Series(0.0, index=idx)

    is_claims = df_rel["Event"].astype(str).str.contains("Jobless Claims|Initial Claims", case=False, regex=True)
    is_nfp = df_rel["Event"].astype(str).str.contains("Nonfarm Payrolls|Change in Nonfarm Payrolls", case=False, regex=True)

    claims_days = set(df_rel.loc[is_claims, "ReleaseDate"].dt.date.tolist())
    nfp_days = set(df_rel.loc[is_nfp, "ReleaseDate"].dt.date.tolist())

    for d in idx:
        # ISO week alignment
        week = d.isocalendar().week
        year = d.isocalendar().year
        # find Thu & Fri of this week
        monday = d - pd.Timedelta(days=d.weekday())
        thursday = monday + pd.Timedelta(days=3)
        friday = monday + pd.Timedelta(days=4)
        if thursday.date() in claims_days and friday.date() in nfp_days:
            s.loc[d] = 1.0
    return s


def _sequence_flags(df_rel: pd.DataFrame) -> pd.DataFrame:
    """
    Daily sequence/conditional flags realized after the second event occurs (BDay+1 applied later).
    """
    # Check if we have valid release dates
    release_dates = df_rel["ReleaseDate"].dropna()
    if len(release_dates) == 0:
        # Return empty DataFrame with default date range
        idx = pd.date_range('2018-01-01', '2025-01-01', freq='D')
        out = pd.DataFrame(index=idx)
    else:
        idx = pd.date_range(release_dates.min(), release_dates.max(), freq="D")
        out = pd.DataFrame(index=idx)

    def _pair(A_pat: str, B_pat: str, horizon_days: int, name: str):
        A = df_rel["Event"].astype(str).str.contains(A_pat, case=False, regex=True)
        B = df_rel["Event"].astype(str).str.contains(B_pat, case=False, regex=True)
        relA = df_rel.loc[A, ["ReleaseDate", "z_surp.N12"]].dropna()
        relB = df_rel.loc[B, ["ReleaseDate", "z_surp.N12"]].dropna()

        # Build flags: agree/diverge, neg→pos, pos→neg when B occurs within horizon of A
        flags = pd.Series(0.0, index=idx)
        for _, brow in relB.iterrows():
            bday = brow["ReleaseDate"]
            window_start = bday - pd.Timedelta(days=horizon_days)
            a_in_win = relA[(relA["ReleaseDate"] >= window_start) & (relA["ReleaseDate"] < bday)]
            if a_in_win.empty:
                continue
            a = a_in_win.sort_values("ReleaseDate").iloc[-1]["z_surp.N12"]
            b = brow["z_surp.N12"]
            if np.isnan(a) or np.isnan(b):
                continue
            sA, sB = np.sign(a), np.sign(b)
            if sA == 0 or sB == 0:
                continue
            # Encode: +1 agree, -1 diverge; also we record directional transitions using extra columns
            flags.loc[bday] = 1.0 if sA == sB else -1.0
            out.loc[bday, f"{name}.negpos"] = 1.0 if (sA < 0 and sB > 0) else 0.0
            out.loc[bday, f"{name}.posneg"] = 1.0 if (sA > 0 and sB < 0) else 0.0

        out[name] = flags

    # Canonical sequences
    for T in (3, 5, 10):
        _pair("PPI", "CPI", T, f"SEQ.PPI_to_CPI.T{T}")
    for T in (3, 5):
        _pair(r"\bADP\b", r"Nonfarm Payrolls|Change in Nonfarm Payrolls|NFP", T, f"SEQ.ADP_to_NFP.T{T}")
    for T in (5, 10):
        _pair(r"ISM(?!.*Services).*(?<!Services)", r"ISM Services", T, f"SEQ.ISM_M_to_ISM_S.T{T}")

    # Expectation-vs-previous example scenarios (binary flags):
    out["COND.cpi_exp_below_prev_and_adp_neg_within5"] = 0.0
    # CPI expectation below prior?
    cpi = df_rel["Event"].astype(str).str.contains(r"\bCPI\b", case=False, regex=True)
    cpi_rows = df_rel.loc[cpi, ["ReleaseDate", "Survey", "Prior"]].dropna()
    adp = df_rel["Event"].astype(str).str.contains(r"\bADP\b", case=False, regex=True)
    adp_rows = df_rel.loc[adp, ["ReleaseDate", "z_surp.N12"]].dropna()
    for _, crow in cpi_rows.iterrows():
        dt = crow["ReleaseDate"]
        exp_below_prev = (crow["Survey"] - crow["Prior"]) < 0 if pd.notna(crow["Survey"]) and pd.notna(crow["Prior"]) else False
        if not exp_below_prev:
            continue
        wstart = dt - pd.Timedelta(days=5)
        adp_win = adp_rows[(adp_rows["ReleaseDate"] >= wstart) & (adp_rows["ReleaseDate"] <= dt)]
        if not adp_win.empty and np.sign(adp_win.iloc[-1]["z_surp.N12"]) < 0:
            out.loc[dt, "COND.cpi_exp_below_prev_and_adp_neg_within5"] = 1.0

    # Hawkish/dovish FOMC setup flags using schedule proximity:
    # (No direct dots/OIS in this file; we just use days_to_next_FOMC later.)
    return out


# --------------------------------------------------------------------------------------
# Put it all together and apply leak-safe shifts & variants
# --------------------------------------------------------------------------------------

def _add_decay_variants(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        for h in DEFAULT_HALFLIFES:
            out[f"{c}.ewm_hf{h}"] = _ewm_halflife(out[c], h)
        for rw in DEFAULT_ROLL_WINDOWS:
            out[f"{c}.roll{rw}"] = out[c].rolling(rw, min_periods=max(2, min(rw, 2))).mean()
    return out

def _shift_realized(df: pd.DataFrame, realized_cols: List[str]) -> pd.DataFrame:
    # Ensure no duplicate index before shifting
    out = df.groupby(df.index).last()
    
    for c in realized_cols:
        if c in out.columns:
            # Use simple numeric shift to avoid freq=BDay() duplicate issues
            shifted_series = out[c].shift(1)
            out[c] = shifted_series
    
    # Clean up any duplicates that might have been introduced
    out = out.groupby(out.index).last()
    return out

def _combine_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    out = pd.DataFrame()
    for f in frames:
        if f is None or f.empty:
            continue
        out = out.join(f, how="outer")
    return out.sort_index()

def build_event_features(calendar_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Main entry: returns a daily DataFrame indexed by date (tz-naive ET-midnight),
    containing EV_* and related feature families. All realized release-derived
    features are shifted by BDay(+1); schedule/proximity features are unshifted.
    """
    rel = _load_calendar_df(calendar_df)

    # Daily realized aggregates
    day_realized = _daily_group(rel)

    # Forward calendar (pre-event, unshifted)
    day_forward = _forward_calendar_features(rel)

    # Sequence/scenario flags (realized on second event's day)
    day_seq = _sequence_flags(rel)

    # Merge all day-level pieces
    # First, let's create a simple merge function since _combine_frames doesn't exist
    def _combine_frames(frames):
        if not frames:
            return pd.DataFrame()
        result = frames[0]
        for frame in frames[1:]:
            if not frame.empty:
                result = result.join(frame, how='outer')
        return result
    
    daily = _combine_frames([day_realized, day_forward, day_seq])
    
    # Keep the last observation if any duplicate dates sneak in
    daily = daily.groupby(daily.index).last()

    # Build additional realized composites that depend on trailing history (tail intensity, cooldown)
    # Tail intensity over 21 trading days (approx daily rolling)
    daily["EV_tail_intensity_21"] = daily["EV_tail_share"].rolling(21, min_periods=5).mean()
    # Cooldown: time since last tail day (EW decays)
    # We approximate "time since last tail" as cumulative days since EV_tail_share>0.2
    # Apply infer_objects to avoid the future downcast warning
    tail_share = daily["EV_tail_share"].infer_objects(copy=False).fillna(0)
    tail_day = (tail_share > 0.2).astype(int)
    cooldown = []
    c = 0
    for v in tail_day:
        c = 0 if v == 1 else c + 1
        cooldown.append(float(c))
    daily["EV_time_since_tailshock_60"] = pd.Series(cooldown, index=daily.index)
    for h in (5, 21, 63):
        daily[f"EV_tail_cooldown_{h}"] = _ewm_halflife(daily["EV_time_since_tailshock_60"], h)

    # Conflict persistence
    if "EV.bucket_divergence" in daily.columns:
        is_conflict = (daily["EV.bucket_divergence"] == -1).astype(float)
        daily["EV.conflict_persist_5"] = is_conflict.rolling(5, min_periods=2).mean()
        daily["EV.conflict_persist_21"] = is_conflict.rolling(21, min_periods=5).mean()

    # Shadow indices (anticipators) using realized releases (BDay+1 applied later)
    # We proxy by using bucket components available in rel df.
    # Inflation shadow: PPI ex-F&E, Import ex-Petrol, ISM Prices Paid (M/S)
    infl_mask = rel["Event"].astype(str).str.contains(
        r"PPI Ex Food.*Energy|Import Price Index ex Petroleum|Prices Paid", case=False, regex=True
    )
    infl_shadow = rel.loc[infl_mask, ["ReleaseDate", "z_surp.N12", "w"]].copy()
    if not infl_shadow.empty:
        infl_shadow_day_result = infl_shadow.groupby("ReleaseDate").apply(lambda g: _wavg(g["z_surp.N12"], g["w"]))
        infl_shadow_day = infl_shadow_day_result.iloc[:, 0] if isinstance(infl_shadow_day_result, pd.DataFrame) else infl_shadow_day_result
        daily["INF.shadow_cpi_z"] = infl_shadow_day.reindex(daily.index)

    # Labor shadow: ADP, ISM Employment, Claims, JOLTS
    lab_mask = rel["Event"].astype(str).str.contains(
        r"ADP|Employment(?!.*Prices)|Jobless Claims|JOLTS", case=False, regex=True
    )
    lab_shadow = rel.loc[lab_mask, ["ReleaseDate", "z_surp.N12", "w"]].copy()
    if not lab_shadow.empty:
        lab_shadow_day_result = lab_shadow.groupby("ReleaseDate").apply(lambda g: _wavg(g["z_surp.N12"], g["w"]))
        lab_shadow_day = lab_shadow_day_result.iloc[:, 0] if isinstance(lab_shadow_day_result, pd.DataFrame) else lab_shadow_day_result
        daily["LAB.shadow_nfp_z"] = lab_shadow_day.reindex(daily.index)

    # Expectations dispersion & crowding (daily)
    # Rebuild a minimal day-level survey width weighted by relevance (if bands exist)
    if "Survey High" in rel.columns and "Survey Low" in rel.columns:
        rel["_survey_width"] = (rel["Survey High"] - rel["Survey Low"]).replace(0, np.nan)
        sw_result = rel.groupby("ReleaseDate").apply(lambda g: _wavg(g["_survey_width"], g["w"]))
        sw = sw_result.iloc[:, 0] if isinstance(sw_result, pd.DataFrame) else sw_result
        daily["EXP.survey_width"] = sw.reindex(daily.index)
        daily["EXP.confidence_proxy"] = _safe_div(1.0, daily["EXP.survey_width"])
    else:
        daily["EXP.survey_width"] = np.nan
        daily["EXP.confidence_proxy"] = np.nan

    # Reliability & integrity (daily)
    # A simple reliability index from available components:
    comp = []
    if "EV_revision_z" in daily.columns:
        # Use explicit dtype conversion to avoid future downcast warning
        revision_z = pd.to_numeric(daily["EV_revision_z"], errors='coerce').fillna(0.0)
        comp.append(1.0 / (1.0 + revision_z))
    if "EXP.survey_width" in daily.columns:
        # Use explicit dtype conversion to avoid future downcast warning
        survey_width = pd.to_numeric(daily["EXP.survey_width"], errors='coerce').fillna(0.0)
        comp.append(1.0 / (1.0 + survey_width))
    if len(comp) > 0:
        daily["META.day_reliability_index"] = np.vstack([c.values for c in comp]).mean(axis=0)
        daily["META.day_reliability_index"] = pd.Series(daily["META.day_reliability_index"], index=daily.index)
    else:
        daily["META.day_reliability_index"] = np.nan

    # Apply leak-safe BDay(+1) shift to realized features
    realized_cols = [
        "EV_after_surprise_z", "EV_polarity_adjusted", "EV_signed_surprise_percentile",
        "EV_surprise_dispersion_day", "EV_tail_share", "EV_top_tier_dominance_share",
        "EV_revision_z", "EV_revision_conflict",
        "EV.bucket_inflation_surp", "EV.bucket_labor_surp", "EV.bucket_growth_surp",
        "EV.bucket_housing_surp", "EV.bucket_sentiment_surp", "EV.bucket_divergence",
        "EV.bucket_inflation_tail_share",
        "EV_tail_intensity_21", "EV_time_since_tailshock_60",
        "EV_tail_cooldown_5", "EV_tail_cooldown_21", "EV_tail_cooldown_63",
        "INF.shadow_cpi_z", "LAB.shadow_nfp_z",
        "SEQ.PPI_to_CPI.T3", "SEQ.PPI_to_CPI.T5", "SEQ.PPI_to_CPI.T10",
        "SEQ.ADP_to_NFP.T3", "SEQ.ADP_to_NFP.T5",
        "SEQ.ISM_M_to_ISM_S.T5", "SEQ.ISM_M_to_ISM_S.T10",
        "SEQ.PPI_to_CPI.T3.negpos", "SEQ.PPI_to_CPI.T3.posneg",
        "SEQ.PPI_to_CPI.T5.negpos", "SEQ.PPI_to_CPI.T5.posneg",
        "SEQ.PPI_to_CPI.T10.negpos", "SEQ.PPI_to_CPI.T10.posneg",
        "SEQ.ADP_to_NFP.T3.negpos", "SEQ.ADP_to_NFP.T3.posneg",
        "SEQ.ADP_to_NFP.T5.negpos", "SEQ.ADP_to_NFP.T5.posneg",
        "SEQ.ISM_M_to_ISM_S.T5.negpos", "SEQ.ISM_M_to_ISM_S.T5.posneg",
        "SEQ.ISM_M_to_ISM_S.T10.negpos", "SEQ.ISM_M_to_ISM_S.T10.posneg",
        "COND.cpi_exp_below_prev_and_adp_neg_within5",
        "EXP.survey_width", "EXP.confidence_proxy",
        "META.day_reliability_index",
    ]
    
    # Ensure no duplicate labels before shifting
    daily = daily.groupby(daily.index).last()
    
    daily = _shift_realized(daily, realized_cols)

    # Add decay/rolling variants for the main scalar EV tapes
    decay_cols = [
        "EV_after_surprise_z",
        "EV_signed_surprise_percentile",
        "EV_surprise_dispersion_day",
        "EV_tail_intensity_21",
        "EV_top_tier_dominance_share",
        "EV.bucket_inflation_surp",
        "EV.bucket_labor_surp",
        "EV.bucket_growth_surp",
        "EV.bucket_housing_surp",
        "EV.bucket_sentiment_surp",
        "INF.shadow_cpi_z",
        "LAB.shadow_nfp_z",
    ]
    daily = _add_decay_variants(daily, decay_cols)

    # Final sorting & clean-up
    daily = daily.sort_index()
    return daily


# --------------------------------------------------------------------------------------
# CLI & quick inspect
# --------------------------------------------------------------------------------------

def _auto():
    try:
        df = build_event_features()  # auto-load
    except Exception as e:
        print("Auto-load failed:", e)
        return
    print("EV daily shape:", df.shape)
    print("Sample columns (first 40):", list(df.columns)[:40])
    print(df.tail(10))


if __name__ == "__main__":
    _auto()
