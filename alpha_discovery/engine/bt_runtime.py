# alpha_discovery/engine/bt_runtime.py
from __future__ import annotations
from typing import Optional, Dict, Tuple, List
import numpy as np
import pandas as pd
import os


from .bt_common import (
    _ensure_df_token, _get_underlier_eod_raw, _get_iv3m_eod_raw,
    _map_sigma_from_3m_to_T_raw, _price_entry_exit, _add_bdays, _build_option_strike
)

# --------- caches keyed by a per-DataFrame token ----------
_CURRENT_TOKEN: Optional[str] = None

_UNDERLIER_CACHE: Dict[Tuple[str, str, pd.Timestamp], Optional[float]] = {}
_IV3M_CACHE: Dict[Tuple[str, str, pd.Timestamp, str], Optional[float]] = {}
_SIGMA_MAP_CACHE: Dict[Tuple[str, str, pd.Timestamp, int, float], Optional[float]] = {}
_PRICE_ENTRY_EXIT_CACHE: Dict[Tuple[float, float, float, float, int, float, str, float, float], Optional[dict]] = {}
_PATH_CACHE: Dict[Tuple, pd.Series] = {}

# --- Phase 1: Exclusivity & cooldown helpers ---
def _parse_bt_env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}

def _parse_bt_env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _enforce_exclusivity_by_setup(ledger: pd.DataFrame, cooldown_days: int | None = None) -> pd.DataFrame:
    """
    Enforce non-overlapping trades per setup_id using `trigger_date` as the entry timestamp.

    Logic:
      - For each setup_id, sort trades by trigger_date.
      - Keep the first trade; suppress any subsequent trade whose trigger_date is
        <= last_eligible_time.
      - last_eligible_time = exit_date + cooldown_days (if exit_date present),
        otherwise a far-future sentinel if the trade is still open (NaT exit).

    Notes:
      - Requires columns: ['setup_id', 'trigger_date']; 'exit_date' is optional.
      - Cooldown_days is measured in calendar days (set via BT_COOLDOWN_DAYS env or passed in).
    """
    if ledger is None or len(ledger) == 0:
        return ledger
    if "setup_id" not in ledger.columns:
        return ledger
    if "trigger_date" not in ledger.columns:
        # Hard guardrail: your system defines entry time via trigger_date only
        raise KeyError("Ledger is missing required 'trigger_date' column for exclusivity enforcement.")

    # Normalize timestamps
    ledger = ledger.copy()
    ledger["trigger_date"] = pd.to_datetime(ledger["trigger_date"], errors="coerce")
    has_exit = "exit_date" in ledger.columns
    if has_exit:
        ledger["exit_date"] = pd.to_datetime(ledger["exit_date"], errors="coerce")

    if cooldown_days is None:
        # Use config cooldown_days as default instead of 0
        from ..config import settings
        cooldown_days = _parse_bt_env_int("BT_COOLDOWN_DAYS", getattr(settings.options, "cooldown_days", 3))

    keep_idx: list[int] = []
    for _sid, grp in ledger.groupby("setup_id", group_keys=False):
        g = grp.sort_values("trigger_date").copy()
        last_eligible = pd.Timestamp.min
        for idx, row in g.iterrows():
            t = row.get("trigger_date")
            if pd.isna(t):
                # Keep rows with missing trigger_date so downstream can decide; they won't open the gate anyway
                keep_idx.append(idx)
                continue

            # Block if not yet eligible
            if t <= last_eligible:
                continue

            # Keep this trade
            keep_idx.append(idx)

            # Advance eligibility to exit + cooldown (or block indefinitely if no exit)
            if has_exit:
                exit_t = row.get("exit_date")
                if pd.isna(exit_t):
                    last_eligible = t + pd.Timedelta(days=3650)  # treat as open; block reentry for a long time
                else:
                    last_eligible = exit_t + pd.Timedelta(days=int(cooldown_days))
            else:
                # No exit column -> treat as immediate eligibility plus cooldown
                last_eligible = t + pd.Timedelta(days=int(cooldown_days))

    return ledger.loc[keep_idx].sort_values(["setup_id", "trigger_date"])


def _rf(x: float, nd: int = 8) -> float:
    return float(round(float(x), nd))

def _maybe_reset_caches(df: pd.DataFrame) -> str:
    global _CURRENT_TOKEN, _UNDERLIER_CACHE, _IV3M_CACHE, _SIGMA_MAP_CACHE, _PRICE_ENTRY_EXIT_CACHE, _PATH_CACHE
    tok = _ensure_df_token(df)
    if tok != _CURRENT_TOKEN:
        _CURRENT_TOKEN = tok
        _UNDERLIER_CACHE.clear()
        _IV3M_CACHE.clear()
        _SIGMA_MAP_CACHE.clear()
        _PRICE_ENTRY_EXIT_CACHE.clear()
        _PATH_CACHE.clear()
    return tok

def _cached_underlier(df: pd.DataFrame, ticker: str, date: pd.Timestamp) -> Optional[float]:
    key = (_CURRENT_TOKEN, ticker, date)
    if key in _UNDERLIER_CACHE:
        return _UNDERLIER_CACHE[key]
    v = _get_underlier_eod_raw(ticker, date, df)
    _UNDERLIER_CACHE[key] = v
    return v

def _cached_iv3m(df: pd.DataFrame, ticker: str, date: pd.Timestamp, direction: str, fallback: Optional[float]) -> Optional[float]:
    key = (_CURRENT_TOKEN, ticker, date, direction)
    if key in _IV3M_CACHE:
        return _IV3M_CACHE[key]
    v = _get_iv3m_eod_raw(ticker, date, direction, df, fallback_sigma=fallback)
    _IV3M_CACHE[key] = v
    return v

def _cached_sigma_map(df: pd.DataFrame, ticker: str, date: pd.Timestamp, T_bd: int, sigma3m: Optional[float]) -> Optional[float]:
    if sigma3m is None or not np.isfinite(sigma3m):
        return None
    key = (_CURRENT_TOKEN, ticker, date, int(T_bd), _rf(float(sigma3m), 6))
    if key in _SIGMA_MAP_CACHE:
        return _SIGMA_MAP_CACHE[key]
    v = _map_sigma_from_3m_to_T_raw(float(sigma3m), int(T_bd), ticker, date, df)
    _SIGMA_MAP_CACHE[key] = v
    return v

def _cached_price_entry_exit(S0: float, S1: float, K: float, T0_years: float, h_days: int, r: float,
                             direction: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Optional[dict]:
    key = (_rf(S0), _rf(S1), _rf(K), _rf(T0_years), int(h_days), _rf(r), direction,
           _rf(entry_sigma), _rf(exit_sigma))
    if key in _PRICE_ENTRY_EXIT_CACHE:
        return _PRICE_ENTRY_EXIT_CACHE[key]
    v = _price_entry_exit(S0, S1, K, T0_years, int(h_days), r, direction, entry_sigma, exit_sigma, q)
    _PRICE_ENTRY_EXIT_CACHE[key] = v
    return v

def _get_or_build_price_path(
    df: pd.DataFrame,
    ticker: str,
    trigger_date: pd.Timestamp,
    horizon_days: int,
    tenor_bd: int,
    direction: str,
    r: float,
    S0: float,
    entry_iv3m: float,
    entry_sigma_T0: float,
    T0_years: float,
) -> pd.Series:
    """Series of mid option prices per business day from trigger_date to horizon_date (cached)."""
    horizon_date = _add_bdays(trigger_date, horizon_days)
    key = (
        _CURRENT_TOKEN, ticker, trigger_date, int(horizon_days), int(tenor_bd), str(direction),
        _rf(r), _rf(S0), _rf(entry_iv3m, 6), _rf(entry_sigma_T0, 6), _rf(T0_years, 8)
    )
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]

    dates = pd.bdate_range(trigger_date, horizon_date, freq="C")
    mids: List[float] = []
    K = _build_option_strike(S0, direction)

    for di, d in enumerate(dates):
        St = _cached_underlier(df, ticker, d)
        if St is None:
            mids.append(np.nan); continue
        iv3m_t = _cached_iv3m(df, ticker, d, direction, fallback=entry_iv3m)
        if iv3m_t is None:
            mids.append(np.nan); continue
        remaining_bd = max(1, int(tenor_bd) - di)
        sigma_t = _cached_sigma_map(df, ticker, d, remaining_bd, iv3m_t)
        if sigma_t is None:
            mids.append(np.nan); continue
        pt = _cached_price_entry_exit(
            S0=S0, S1=St, K=K, T0_years=T0_years, h_days=di, r=r,
            direction=("long" if direction == "long" else "short"),
            entry_sigma=entry_sigma_T0, exit_sigma=sigma_t, q=0.0
        )
        mids.append(float(pt["exit_price"]) if pt is not None else np.nan)

    path = pd.Series(mids, index=dates).ffill()
    _PATH_CACHE[key] = path
    return path
