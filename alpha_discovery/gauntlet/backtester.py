# alpha_discovery/gauntlet/backtester.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from ..config import Settings  # type: ignore
from ..engine.bt_core import run_setup_backtest_options
from ..engine.bt_runtime import _enforce_exclusivity_by_setup, _parse_bt_env_flag

# ----------------- small helpers -----------------

def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    for cand in ["date", "Date", "timestamp", "ts", "datetime", "Datetime"]:
        if cand in df.columns:
            tmp = df.copy()
            tmp[cand] = pd.to_datetime(tmp[cand], errors="coerce")
            tmp = tmp.set_index(cand).sort_index()
            return tmp
    return df

def _filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in ["ticker", "symbol", "underlying", "SpecializedTicker", "asset"]:
        if col in df.columns:
            return df.loc[df[col] == ticker]
    return df

def _infer_last_price(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    for col in ["px_last", "close", "Close", "adj_close", "Adj Close", "price"]:
        if col in df.columns and df[col].notna().any():
            try:
                return float(df[col].dropna().iloc[-1])
            except Exception:
                pass
    return None

def _mark_open_positions_at_eod(
    ledger: pd.DataFrame,
    oos_end: pd.Timestamp,
    last_mark: Optional[float] = None,
) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return ledger

    led = ledger.copy()
    defaults = {
        "status": None,
        "exit_date": pd.NaT,
        "exit_reason": None,
        "exit_price": np.nan,
        "unrealized_pnl": np.nan,
        "realized_pnl": np.nan,
        "entry_price": np.nan,
        "contracts": np.nan,
        "direction": None,
        "trigger_date": pd.NaT,
        "entry_date": pd.NaT,
    }
    for k, v in defaults.items():
        if k not in led.columns:
            led[k] = v

    for dc in ("exit_date", "entry_date", "trigger_date"):
        if dc in led.columns:
            led[dc] = pd.to_datetime(led[dc], errors="coerce")

    is_open_eod = led["exit_date"].isna() | (led["exit_date"] > oos_end)
    if is_open_eod.any():
        led.loc[is_open_eod, "status"] = "OPEN"
        led.loc[is_open_eod, "exit_date"] = pd.NaT
        led.loc[is_open_eod, "exit_reason"] = "EOD_OPEN"
        led.loc[is_open_eod, "exit_price"] = np.nan

        if last_mark is not None:
            entry_price = pd.to_numeric(led.loc[is_open_eod, "entry_price"], errors="coerce")
            contracts = pd.to_numeric(led.loc[is_open_eod, "contracts"], errors="coerce").fillna(0.0)
            direction = led.loc[is_open_eod, "direction"].astype(str).str.upper()
            sign = np.where(direction.str.contains("SHORT"), -1.0, 1.0)
            led.loc[is_open_eod, "unrealized_pnl"] = (last_mark - entry_price) * contracts * sign

        entry_dt = led.loc[is_open_eod, "entry_date"]
        led.loc[is_open_eod, "holding_days_open"] = (
            oos_end.normalize() - entry_dt.dt.normalize()
        ).dt.days.clip(lower=0)

    return led

# ----------------- schema alignment -----------------

def _align_to_pareto_schema_auto(ledger: pd.DataFrame, origin_fold: int) -> pd.DataFrame:
    """
    Align Gauntlet OOS ledger to Pareto TRAIN ledger columns for the same fold,
    but always preserve critical fields needed for Gauntlet/QA.
    """
    if ledger is None or ledger.empty:
        return ledger

    df = ledger.copy()
    if "origin_fold" in df.columns:
        df = df.rename(columns={"origin_fold": "fold"})
    if "solution_rank" not in df.columns:
        df["solution_rank"] = np.nan

    # Locate pareto_ledger.csv for this fold under the most recent run
    candidates: List[str] = []
    cwd_runs = os.path.join(os.getcwd(), "runs")
    if os.path.isdir(cwd_runs):
        candidates.append(cwd_runs)
    here = os.path.abspath(os.path.dirname(__file__))  # .../alpha_discovery/gauntlet
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    root_runs = os.path.join(project_root, "runs")
    if os.path.isdir(root_runs) and root_runs not in candidates:
        candidates.append(root_runs)

    def _run_dirs(runs_root: str) -> List[str]:
        try:
            entries = [
                os.path.join(runs_root, d)
                for d in os.listdir(runs_root)
                if os.path.isdir(os.path.join(runs_root, d))
            ]
            entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return entries
        except Exception:
            return []

    target_header: Optional[List[str]] = None
    fold_dirname = f"fold_{int(origin_fold):02d}"
    for runs_root in candidates:
        for run_dir in _run_dirs(runs_root):
            candidate = os.path.join(run_dir, "folds", fold_dirname, "pareto_ledger.csv")
            if os.path.exists(candidate):
                try:
                    target_header = list(pd.read_csv(candidate, nrows=0).columns)
                    break
                except Exception:
                    target_header = None
        if target_header is not None:
            break

    # Always preserve these critical columns even if Pareto didn't include them
    must_keep = [
        "setup_id", "signal_ids", "specialized_ticker", "exit_reason",
        "first_exit_reason", "status", "unrealized_pnl", "holding_days_actual",
        "direction",
    ]
    for c in must_keep:
        if c not in df.columns:
            df[c] = np.nan

    if not target_header:
        # No pareto template found; return minimally harmonized DF (keep everything)
        return df

    # Build final column order = pareto header + must_keep (append if missing)
    final_cols = list(target_header)
    for c in must_keep:
        if c not in final_cols:
            final_cols.append(c)

    # Ensure all target columns exist; drop extras by reindexing
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[final_cols]
    return df

# ----------------- OOS backtest wrapper -----------------

def run_gauntlet_backtest(
    setup_id: str,
    specialized_ticker: str,
    signal_ids: List[str],
    direction: str,
    oos_start_date: pd.Timestamp,
    master_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict[str, Any]],
    exit_policy: Optional[Dict[str, Any]],
    settings: Settings,
    origin_fold: int,
) -> Optional[pd.DataFrame]:
    """
    OOS backtest for a single setup on its specialized ticker. Ensures
    end-of-data open trades stay OPEN (no fabricated exits).
    """
    if master_df is None or master_df.empty or signals_df is None or signals_df.empty:
        return None

    md = _as_dt_index(master_df)
    sd = _as_dt_index(signals_df)

    if not isinstance(md.index, pd.DatetimeIndex) or not isinstance(sd.index, pd.DatetimeIndex):
        oos_master_df = _filter_ticker(master_df, specialized_ticker)
        oos_signals_df = _filter_ticker(signals_df, specialized_ticker)
    else:
        md = md.sort_index()
        sd = sd.sort_index()
        oos_master_df = md.loc[md.index >= pd.Timestamp(oos_start_date)]
        oos_signals_df = sd.loc[sd.index >= pd.Timestamp(oos_start_date)]
        oos_master_df = _filter_ticker(oos_master_df, specialized_ticker)
        oos_signals_df = _filter_ticker(oos_signals_df, specialized_ticker)

    if oos_master_df is None or oos_master_df.empty or oos_signals_df is None or oos_signals_df.empty:
        return None

    ledger = run_setup_backtest_options(
        setup_signals=signal_ids,
        signals_df=oos_signals_df,
        master_df=oos_master_df,
        direction=direction,
        exit_policy=exit_policy,
        tickers_to_run=[specialized_ticker],
    )

    if ledger is None or len(ledger) == 0:
        return ledger

    oos_end = (
        oos_master_df.index.max()
        if isinstance(oos_master_df.index, pd.DatetimeIndex)
        else pd.Timestamp.max.tz_localize(None)
    )
    last_mark = _infer_last_price(oos_master_df)

    ledger = _mark_open_positions_at_eod(ledger, oos_end=oos_end, last_mark=last_mark).copy()
    ledger["setup_id"] = setup_id
    ledger["origin_fold"] = int(origin_fold)
    ledger["signal_ids"] = ", ".join(sorted(signal_ids))
    ledger["specialized_ticker"] = specialized_ticker

    if _parse_bt_env_flag("BT_ENFORCE_EXCLUSIVITY", True):
        ledger = _enforce_exclusivity_by_setup(ledger)

    ledger = _align_to_pareto_schema_auto(ledger, origin_fold=int(origin_fold))
    return ledger
