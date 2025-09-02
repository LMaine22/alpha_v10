# alpha_discovery/gauntlet/backtester.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from ..config import Settings  # type: ignore
from ..engine.backtester import run_setup_backtest_options


def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the frame has a DatetimeIndex. Prefer index if already datetime,
    otherwise try 'date' column; leave as-is if neither works.
    """
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
    """
    Filter a DF to a specific underlying/specialized_ticker.
    Tries common column names; if none present, returns df unchanged.
    """
    if df is None or df.empty:
        return df
    for col in ["ticker", "symbol", "underlying", "SpecializedTicker", "asset"]:
        if col in df.columns:
            return df.loc[df[col] == ticker]
    return df


def _infer_last_price(df: pd.DataFrame) -> Optional[float]:
    """
    Try to get a last mark for unrealized PnL purposes.
    """
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
    """
    Keep positions that are still open at end-of-data as OPEN (no fabricated exits).
      - exit_date -> NaT
      - status -> 'OPEN'
      - exit_price -> NaN
      - exit_reason -> 'EOD_OPEN'
      - unrealized_pnl -> based on last_mark if provided (optional)
    """
    if ledger is None or ledger.empty:
        return ledger

    led = ledger.copy()

    # Ensure columns exist
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

    # Coerce dates
    for dc in ("exit_date", "entry_date", "trigger_date"):
        if dc in led.columns:
            led[dc] = pd.to_datetime(led[dc], errors="coerce")

    # Identify trades improperly closed after the data end, or never closed
    is_open_eod = led["exit_date"].isna() | (led["exit_date"] > oos_end)

    if is_open_eod.any():
        led.loc[is_open_eod, "status"] = "OPEN"
        led.loc[is_open_eod, "exit_date"] = pd.NaT
        led.loc[is_open_eod, "exit_reason"] = "EOD_OPEN"
        led.loc[is_open_eod, "exit_price"] = np.nan

        if last_mark is not None:
            # Simple mark-to-market; adjust if you have option multipliers/greeks
            entry_price = pd.to_numeric(led.loc[is_open_eod, "entry_price"], errors="coerce")
            contracts = pd.to_numeric(led.loc[is_open_eod, "contracts"], errors="coerce").fillna(0.0)
            direction = led.loc[is_open_eod, "direction"].astype(str).str.upper()
            sign = np.where(direction.str.contains("SHORT"), -1.0, 1.0)
            led.loc[is_open_eod, "unrealized_pnl"] = (last_mark - entry_price) * contracts * sign

        # Handy: how long has it been open
        entry_dt = led.loc[is_open_eod, "entry_date"]
        led.loc[is_open_eod, "holding_days_open"] = (
            oos_end.normalize() - entry_dt.dt.normalize()
        ).dt.days.clip(lower=0)

    return led


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
    end-of-data open trades stay OPEN (no fabricated exit dates).
    """
    if master_df is None or master_df.empty or signals_df is None or signals_df.empty:
        return None

    # Normalize, slice OOS, filter to ticker
    md = _as_dt_index(master_df)
    sd = _as_dt_index(signals_df)

    # If index isn't datetime after this, bail fast
    if not isinstance(md.index, pd.DatetimeIndex) or not isinstance(sd.index, pd.DatetimeIndex):
        # Still attempt the backtest without slicing to be robust
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

    # Run the backtest on the OOS slice, ONLY for the specialized ticker
    ledger = run_setup_backtest_options(
        setup_signals=signal_ids,
        signals_df=oos_signals_df,
        master_df=oos_master_df,
        direction=direction,
        exit_policy=exit_policy,
        tickers_to_run=[specialized_ticker],  # <<< IMPORTANT: restrict to the intended ticker
    )

    if ledger is None or len(ledger) == 0:
        return ledger

    # Determine true OOS end and last mark
    oos_end = (
        oos_master_df.index.max()
        if isinstance(oos_master_df.index, pd.DatetimeIndex)
        else pd.Timestamp.max.tz_localize(None)
    )
    last_mark = _infer_last_price(oos_master_df)

    # Ensure end-of-data open trades remain OPEN
    ledger = _mark_open_positions_at_eod(ledger, oos_end=oos_end, last_mark=last_mark)

    return ledger
