# alpha_discovery/data/merge_economic_releases.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

# ---- Anchor to repo root regardless of CWD ----
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data_store" / "raw"
PROC_DIR = REPO_ROOT / "data_store" / "processed"

INPUT_CANDIDATES = [
    "Economic_Releases.xlsx",
    "Economic_Releases2.xlsx",
    "Economic_Releases3.xlsx",
    "Economic_Releases4.xlsx",

    # fallbacks in case files were named singular
    "Economic_Release.xlsx",
    "Economic_Release2.xlsx",
    "Economic_Release3.xlsx",
    "Economic_Release4.xlsx",

]

OUTPUT_XLSX = RAW_DIR / "Economic_Releases_combined.xlsx"
OUTPUT_PARQUET = PROC_DIR / "economic_releases.parquet"

ET_TZ = "America/New_York"


def _existing_inputs() -> List[Path]:
    return [RAW_DIR / n for n in INPUT_CANDIDATES if (RAW_DIR / n).exists()]


def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_relevance(col: pd.Series) -> pd.Series:
    if col is None:
        return pd.Series(dtype="float64")
    x = col.astype(str).str.strip()
    x = x.str.replace("%", "", regex=False)
    x = x.str.title()
    map_text = {"High": "90", "Medium": "60", "Med": "60", "Low": "30"}
    x = x.replace(map_text)
    return pd.to_numeric(x, errors="coerce")


def _first_match(cols: List[str], *candidates: str) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_one_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    dt_col = _first_match(df.columns, "Date Time", "Datetime", "Release Datetime", "DateTime")
    if dt_col is None:
        date_col = _first_match(df.columns, "Date", "Release Date")
        time_col = _first_match(df.columns, "Time")
        if date_col:
            if time_col:
                dt = pd.to_datetime(
                    df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
                    errors="coerce",
                )
            else:
                dt = pd.to_datetime(df[date_col], errors="coerce")
        else:
            dt = pd.NaT
    else:
        dt = pd.to_datetime(df[dt_col], errors="coerce")

    # localize/convert to ET
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(ET_TZ)
        else:
            dt = dt.dt.tz_convert(ET_TZ)
    except Exception:
        pass

    event_col = _first_match(df.columns, "Event", "Event Name", "Indicator")
    country_col = _first_match(df.columns, "Country Code", "Country")
    survey_col = _first_match(df.columns, "Survey", "Consensus")
    actual_col = _first_match(df.columns, "Actual")
    prior_col = _first_match(df.columns, "Prior", "Previous")
    revised_col = _first_match(df.columns, "Revised")
    rel_col = _first_match(df.columns, "Relevance", "Importance")
    ticker_col = _first_match(df.columns, "Ticker", "BB Ticker")

    out = pd.DataFrame({
        "release_datetime": dt,
        "event_type": df[event_col].astype(str).str.strip() if event_col else pd.Series(dtype="string"),
        "country": df[country_col].astype(str).str.strip() if country_col else pd.Series(dtype="string"),
        "survey": _coerce_float(df[survey_col]) if survey_col else pd.Series(dtype="float64"),
        "actual": _coerce_float(df[actual_col]) if actual_col else pd.Series(dtype="float64"),
        "prior": _coerce_float(df[prior_col]) if prior_col else pd.Series(dtype="float64"),
        "revised": _coerce_float(df[revised_col]) if revised_col else pd.Series(dtype="float64"),
        "relevance": _parse_relevance(df[rel_col]) if rel_col else pd.Series(dtype="float64"),
        "bb_ticker": df[ticker_col].astype(str).str.strip() if ticker_col else pd.Series(dtype="string"),
    })

    out = out.dropna(subset=["release_datetime"])
    out["event_type"] = out["event_type"].fillna("").str.strip()
    out["country"] = out["country"].fillna("").str.strip()
    out.loc[out["bb_ticker"].isin(["", "nan", "NaN"]), "bb_ticker"] = pd.NA
    out = out[out["event_type"] != ""]
    return out


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    inputs = _existing_inputs()
    if not inputs:
        expected = ", ".join(INPUT_CANDIDATES)
        print(f"No inputs found in {RAW_DIR}. Expected one of: {expected}")
        sys.exit(1)

    print("\nMerging the following files:")
    for p in inputs:
        print(f"  - {p}")

    frames = [load_one_excel(p) for p in inputs]
    combined = pd.concat(frames, ignore_index=True)

    # convenience daily date (keeps tz in datetime)
    try:
        combined["release_date"] = combined["release_datetime"].dt.tz_convert(ET_TZ).dt.date
    except Exception:
        combined["release_date"] = pd.to_datetime(combined["release_datetime"]).dt.date

    combined = (
        combined.drop_duplicates(subset=["release_datetime", "event_type", "country"])
        .sort_values(["release_datetime", "event_type", "country"])
        .reset_index(drop=True)
    )

    by_event = combined["event_type"].value_counts().head(10)
    print("\nTop event types (head):")
    print(by_event.to_string())

    print(f"\nTotal rows after merge: {len(combined):,}")
    print(f"Earliest: {combined['release_datetime'].min()}  Latest: {combined['release_datetime'].max()}")

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    # --- Write Parquet with tz-aware datetime (preferred for pipeline) ---
    combined.to_parquet(OUTPUT_PARQUET, index=False)

    # --- Excel export must be tz-naive ---
    excel_df = combined.copy()
    try:
        excel_df["release_datetime"] = excel_df["release_datetime"].dt.tz_convert(ET_TZ).dt.tz_localize(None)
    except Exception:
        # if already naive, just ensure datetime dtype
        excel_df["release_datetime"] = pd.to_datetime(excel_df["release_datetime"]).dt.tz_localize(None)

    excel_df.to_excel(OUTPUT_XLSX, index=False)

    print(f"\nWrote compact Parquet → {OUTPUT_PARQUET}")
    print(f"Wrote combined Excel  → {OUTPUT_XLSX}")
    print("Done.")


if __name__ == "__main__":
    main()
