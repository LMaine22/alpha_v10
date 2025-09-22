from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
from pathlib import Path

# Ensure repo root is importable when running file directly from scripts/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Project settings and helpers
try:
    from alpha_discovery.config import settings  # type: ignore
except Exception:
    # Lightweight fallback if project deps aren't installed or package import fails
    @dataclass
    class _EventsCfg:
        file_path: str = str(_REPO_ROOT / "data_store" / "processed" / "economic_releases.parquet")

    @dataclass
    class _DataCfg:
        parquet_file_path: str = str(_REPO_ROOT / "data_store" / "processed" / "bb_data.parquet")
        tradable_tickers: List[str] = field(default_factory=lambda: [
        'MSTR US Equity', 'SNOW US Equity', 'LLY US Equity', 'COIN US Equity',
        'QCOM US Equity', 'ULTA US Equity', 'CRM US Equity', 'AAPL US Equity',
        'AMZN US Equity', 'MSFT US Equity', 'QQQ US Equity', 'SPY US Equity',
        'TSM US Equity', 'META US Equity', 'TSLA US Equity', 'CRWV US Equity',
        'VIX Index', 'GOOGL US Equity', 'AMD US Equity',  'ARM US Equity',
        'PLTR US Equity', 'VIX Index', 'JPM US Equity', 'C US Equity',
        'BMY US Equity','NKE US Equity', 'AVGO US Equity'
        ])

    @dataclass
    class _FallbackSettings:
        events: _EventsCfg = field(default_factory=_EventsCfg)
        data: _DataCfg = field(default_factory=_DataCfg)

    settings = _FallbackSettings()


# -----------------------------
# Data loading
# -----------------------------

def _load_calendar() -> pd.DataFrame:
    """Load merged economic releases and normalize column names/types.

    We read the canonical parquet written by merge_economic_releases.py and
    coerce to a uniform schema used in this script.
    """
    path = settings.events.file_path
    try:
        df = pd.read_parquet(path)
    except Exception:
        # Fallback to CSV if available
        import os as _os
        csv_guess = _os.path.splitext(path)[0] + ".csv"
        if _os.path.exists(csv_guess):
            df = pd.read_csv(csv_guess)
        else:
            raise

    # Normalize names to match expectations
    rename_map = {
        "release_datetime": "Date",
        "event_type": "Event",
        "country": "Country",
        "survey": "Survey",
        "actual": "Actual",
        "prior": "Prior",
        "revised": "Revised",
        "relevance": "Relevance",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Date handling (keep tz-naive ET date for daily joins)
    dt = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert("America/New_York").dt.tz_localize(None)
    except Exception:
        dt = pd.to_datetime(df["Date"], errors="coerce")
    df["DateTime"] = dt
    df["ReleaseDate"] = df["DateTime"].dt.floor("D")

    # Numerics
    for c in ["Survey", "Actual", "Prior", "Revised", "Relevance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _load_prices() -> pd.DataFrame:
    """Load processed panel data parquet with price columns."""
    p = settings.data.parquet_file_path
    try:
        return pd.read_parquet(p)
    except Exception:
        # Fallback to CSV if present
        import os as _os
        csv_guess = _os.path.splitext(p)[0] + ".csv"
        if _os.path.exists(csv_guess):
            return pd.read_csv(csv_guess, parse_dates=[0], index_col=0)
        raise


# -----------------------------
# Ticker listing & selection
# -----------------------------

def list_tradable_tickers() -> List[str]:
    try:
        tickers = list(getattr(settings.data, "tradable_tickers", []))
        return tickers if tickers else []
    except Exception:
        return []


# Edit this if you'd like to set a default without passing --ticker
SELECTED_TICKER: str = "AMZN"


# -----------------------------
# FOMC cut logic
# -----------------------------

def _is_fomc_rate_decision(event: str) -> bool:
    return isinstance(event, str) and "FOMC Rate Decision" in event


@dataclass
class FomcCut:
    date: pd.Timestamp
    prior: float
    survey: float
    actual: float
    expected_cut: bool
    realized_cut: bool


def find_expected_and_realized_cuts(cal: pd.DataFrame) -> List[FomcCut]:
    """Return list of dates where the market expected a cut (Survey < Prior)
    and the Fed cut (Actual < Prior). We use the Upper Bound series if present,
    falling back to Lower Bound.
    """
    df = cal.copy()
    df = df[df["Country"].astype(str).str.upper().eq("US")]
    df = df[df["Event"].apply(_is_fomc_rate_decision)]

    # Prefer Upper Bound rows to avoid double counting
    upper = df[df["Event"].str.contains("Upper Bound", na=False)]
    base = upper if len(upper) else df

    rows: List[FomcCut] = []
    for _, r in base.dropna(subset=["ReleaseDate"]).iterrows():
        prior = float(r.get("Prior", np.nan))
        survey = float(r.get("Survey", np.nan))
        actual = float(r.get("Actual", np.nan))
        if not np.isfinite(prior) or not np.isfinite(survey) or not np.isfinite(actual):
            continue
        exp_cut = survey < prior - 1e-9
        did_cut = actual < prior - 1e-9
        rows.append(FomcCut(
            date=pd.Timestamp(r["ReleaseDate"]),
            prior=prior, survey=survey, actual=actual,
            expected_cut=bool(exp_cut), realized_cut=bool(did_cut)
        ))
    return rows


# -----------------------------
# Price helpers
# -----------------------------

def _resolve_ticker_base(df: pd.DataFrame, user_ticker: str) -> Optional[str]:
    """Resolve a sheet/ticker base used in column prefixes.

    Example: user_ticker="QQQ" -> "QQQ US Equity" if those columns exist.
    """
    # Try config list first
    for t in settings.data.tradable_tickers:
        if t.upper().startswith(user_ticker.upper()):
            if f"{t}_PX_LAST" in df.columns or f"{t}_PX_OPEN" in df.columns:
                return t

    # Heuristic scan: look for columns ending with _PX_LAST that start with ticker
    suffix = "_PX_LAST"
    for c in df.columns:
        if c.endswith(suffix) and c.upper().startswith(user_ticker.upper() + " "):
            return c[: -len(suffix)]

    # Fallback common pattern
    guess = f"{user_ticker} US Equity"
    if f"{guess}_PX_LAST" in df.columns or f"{guess}_PX_OPEN" in df.columns:
        return guess
    return None


def _open_close_return(panel: pd.DataFrame, base: str, d: pd.Timestamp) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ocol = f"{base}_PX_OPEN"
    ccol = f"{base}_PX_LAST"
    if ocol not in panel.columns and ccol not in panel.columns:
        return None, None, None
    # Coerce index and pick values as-of the day
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("Price panel index must be DatetimeIndex")
    dd = pd.Timestamp(d)
    try:
        o = panel[ocol].reindex(panel.index).loc[dd]
        c = panel[ccol].reindex(panel.index).loc[dd]
    except Exception:
        # If missing exact date, try nearest previous/next business day
        idx = panel.index
        if dd not in idx:
            prev = idx[idx <= dd]
            if len(prev):
                dd = prev.max()
        o = panel[ocol].loc[dd] if ocol in panel.columns and dd in panel.index else np.nan
        c = panel[ccol].loc[dd] if ccol in panel.columns and dd in panel.index else np.nan
    if not np.isfinite(o) or not np.isfinite(c) or o == 0:
        return None, None, None
    ret = float(c / o - 1.0)
    return float(o), float(c), ret


# -----------------------------
# Main routine
# -----------------------------

def run(ticker: str = "QQQ", only_expected_and_realized: bool = True) -> pd.DataFrame:
    cal = _load_calendar()
    panel = _load_prices()

    base = _resolve_ticker_base(panel, ticker)
    if base is None:
        raise ValueError(f"Could not resolve price columns for '{ticker}'.")

    cuts = find_expected_and_realized_cuts(cal)
    if only_expected_and_realized:
        cuts = [c for c in cuts if c.expected_cut and c.realized_cut]

    rows = []
    for c in cuts:
        o, cl, r = _open_close_return(panel, base, c.date)
        rows.append({
            "date": c.date.normalize(),
            "prior": c.prior,
            "survey": c.survey,
            "actual": c.actual,
            "expected_cut": c.expected_cut,
            "realized_cut": c.realized_cut,
            f"{base}_open": o,
            f"{base}_close": cl,
            f"{base}_oc_ret": r,
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def scan_all_tickers(only_expected_and_realized: bool = True) -> pd.DataFrame:
    """Compute per-ticker win-rate on cut days across settings.data.tradable_tickers."""
    cal = _load_calendar()
    panel = _load_prices()
    tickers = list_tradable_tickers()
    results = []
    for t in tickers:
        # Accept short name as well
        short = t.split(" ")[0]
        base = _resolve_ticker_base(panel, short)
        if base is None:
            continue
        cuts = find_expected_and_realized_cuts(cal)
        if only_expected_and_realized:
            cuts = [c for c in cuts if c.expected_cut and c.realized_cut]
        if not cuts:
            continue
        rets = []
        for c in cuts:
            o, cl, r = _open_close_return(panel, base, c.date)
            if r is not None and np.isfinite(r):
                rets.append(float(r))
        if not rets:
            continue
        arr = np.array(rets, dtype=float)
        win_rate = float((arr > 0).mean())
        mean_ret = float(arr.mean())
        median_ret = float(np.median(arr))
        n = int(len(arr))
        results.append({
            "ticker": base,
            "n_obs": n,
            "win_rate": win_rate,
            "mean_oc": mean_ret,
            "median_oc": median_ret,
        })
    return pd.DataFrame(results).sort_values(["win_rate", "mean_oc", "n_obs"], ascending=[False, False, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Analyze market reaction on FOMC rate cut days")
    ap.add_argument("--ticker", default=None, help="Base ticker to analyze (e.g., QQQ or 'QQQ US Equity')")
    ap.add_argument("--all", action="store_true", help="Include all FOMC decisions, not only expected+realized cuts")
    ap.add_argument("--out", default=None, help="Optional CSV path to write results")
    ap.add_argument("--list", action="store_true", help="List tradable tickers and exit")
    ap.add_argument("--rank", action="store_true", help="Scan all tradable tickers and rank by win-rate on cut days")
    ap.add_argument("--top", type=int, default=10, help="Show top N and bottom N in rank mode")
    ap.add_argument("--summary_out", default=None, help="Optional CSV path for rank-mode summary table")
    args = ap.parse_args()

    if args.list:
        tickers = list_tradable_tickers()
        if not tickers:
            print("No tradable tickers configured in settings.data.tradable_tickers.")
        else:
            print("Tradable tickers:")
            for t in tickers:
                print(f"  - {t}")
        return

    if args.rank:
        table = scan_all_tickers(only_expected_and_realized=(not args.all))
        if table.empty:
            print("No results.")
            return
        k = max(1, int(args.top))
        print("\nBest by win-rate on cut days:\n")
        print(table.head(k))
        print("\nWorst by win-rate on cut days:\n")
        print(table.sort_values(["win_rate", "mean_oc", "n_obs"], ascending=[True, True, False]).head(k))
        if args.summary_out:
            import os
            os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
            table.to_csv(args.summary_out, index=False)
            print(f"\nWrote summary → {args.summary_out}")
        return

    ticker = args.ticker or SELECTED_TICKER
    print(f"Using ticker: {ticker}")

    df = run(ticker=ticker, only_expected_and_realized=(not args.all))
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 160)

    print("\nFOMC rate cut reaction days (open→close)\n")
    print(df)

    # Summary
    ret_col = [c for c in df.columns if c.endswith("_oc_ret")]
    if ret_col:
        s = pd.to_numeric(df[ret_col[0]], errors="coerce").dropna()
        if len(s):
            print("\nSummary stats:")
            print(f"  N = {len(s)}")
            print(f"  Mean O→C = {s.mean()*100:.2f}%")
            print(f"  Median O→C = {s.median()*100:.2f}%")
            print(f"  Hit-rate (>0) = {(s>0).mean()*100:.1f}%")
        else:
            print("\nNo valid return observations.")
    else:
        print("\nNo return column computed.")

    # Optional CSV output
    if args.out:
        import os
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nWrote results → {out_path}")


if __name__ == "__main__":
    main()


