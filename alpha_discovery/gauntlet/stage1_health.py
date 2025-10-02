# alpha_discovery/gauntlet/stage1_health.py
"""
Gauntlet 2.0 — Stage 1: Health & Sanity (fast hard-gates)

Purpose
-------
Kill pathological or data-broken candidates in milliseconds before heavy validation.
This stage checks market plumbing and data feasibility, not alpha quality.

Inputs
------
- fold_summary: single-row DF with setup metadata (should include setup_id/rank if available)
- fold_ledger : trade-level DF for the candidate on the evaluated fold (trigger/entry/exit, pnl, strikes, iv, oi/volume, etc.)
- config/settings: optional dict-like with thresholds (see Config keys below)

Outputs
-------
Single-row DataFrame with:
  setup_id, rank, pass_stage1 (bool), reject_code (str|None), reason (str),
  plus diagnostic fields for auditability.

Config keys (defaults shown)
----------------------------
s1_recent_window_days: int = 7
s1_min_recent_trades : int = 1
s1_min_total_trades  : int = 5
s1_momentum_window_days : int = 30
s1_min_momentum_trades  : int = 3
s1_iv_availability_min  : float = 0.98   # share of days with at least one IV field present
s1_strike_success_min   : float = 0.97   # if strike_selected / selected_strike exists
s1_missing_data_tolerance: float = 0.01  # key columns missing share
s1_mean_holding_ratio_min: float = 0.40  # mean holding >= ratio * mean tenor
s1_daily_trade_cap      : int = 12
# Liquidity floors (aggregate proxies since per-option OI/vol not always present)
s1_min_tot_opt_volume   : int = 1000     # TOT_OPT_VOLUME_CUR_DAY >= this on >=95% days (if present)
s1_min_open_interest_sum: int = 5000     # (OPEN_INT_TOTAL_CALL+PUT) >= this on >=95% days (if present)
# Quote sanity
s1_quote_sanity_min_ok_share: float = 0.99  # share of days with BID<=ASK and OHLC in-range

Notes
-----
- All checks are non-fatal if a required column is absent; we record 'skipped' via NaNs and do not fail purely due to missing columns.
- We return early only if inputs are missing or timestamps are unusable.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


# ------------------------------ Column Aliases (tailored to your data) ------------------------------

IV_COLUMNS = [
    "PUT_IMP_VOL_30D", "CALL_IMP_VOL_30D",
    "3MO_CALL_IMP_VOL", "3MO_PUT_IMP_VOL",
    "1M_CALL_IMP_VOL_10DELTA_DFLT", "1M_CALL_IMP_VOL_25DELTA_DFLT",
    "1M_CALL_IMP_VOL_40DELTA_DFLT", "1M_PUT_IMP_VOL_25DELTA_DFLT",
    "1M_PUT_IMP_VOL_40DELTA_DFLT",
    # keep flexible: any column containing 'IMP_VOL' also counts as IV
]

LIQ_VOLUME_COL = "TOT_OPT_VOLUME_CUR_DAY"
LIQ_OI_COLS = ["OPEN_INT_TOTAL_CALL", "OPEN_INT_TOTAL_PUT"]

BID_COL, ASK_COL = "PX_BID", "PX_ASK"
OPEN_COL, HIGH_COL, LOW_COL = "PX_OPEN", "PX_HIGH", "PX_LOW"

KEY_MARKET_COLS = [OPEN_COL, HIGH_COL, LOW_COL, BID_COL, ASK_COL, "PX_VOLUME"]


# ------------------------------ Helpers ------------------------------

def _infer_setup_col(df: pd.DataFrame) -> str:
    for c in ["setup_id", "Setup", "strategy_id", "StrategyID"]:
        if c in df.columns:
            return c
    if "setup_id" not in df.columns:
        df = df.copy()
        df["setup_id"] = "unknown"
    return "setup_id"


def _data_end_from_ledger(ledger: pd.DataFrame) -> Optional[pd.Timestamp]:
    candidates = []
    for c in ("trigger_date", "entry_date", "exit_date"):
        if c in ledger.columns:
            s = pd.to_datetime(ledger[c], errors="coerce")
            m = s.max()
            if pd.notnull(m):
                candidates.append(m)
    return max(candidates) if candidates else None


def _get_recent_mask(dates: pd.Series, data_end: pd.Timestamp, window_days: int) -> pd.Series:
    return (data_end.normalize() - dates.dt.normalize()).dt.days <= int(window_days)


def _safe_win_rate(ledger: pd.DataFrame, mask: pd.Series, min_trades: int) -> Tuple[bool, float, int]:
    """Return (has_momentum, win_rate, n_recent_evaluable) using pnl columns if present."""
    recent = ledger.loc[mask]
    n = len(recent)
    if n < min_trades:
        return True, 0.0, n  # not enough data to judge, treat as pass-by-insufficient-sample
    if "pnl_pct" in recent.columns:
        ser = pd.to_numeric(recent["pnl_pct"], errors="coerce")
    elif "realized_pnl" in recent.columns and "entry_exec" in recent.columns:
        denom = pd.to_numeric(recent["entry_exec"], errors="coerce").replace(0, np.nan)
        ser = pd.to_numeric(recent["realized_pnl"], errors="coerce") / denom
    else:
        return True, 0.0, 0  # cannot compute; do not fail here
    ser = ser.replace([np.inf, -np.inf], np.nan).dropna()
    if ser.empty:
        return True, 0.0, 0
    wins = (ser > 0).sum()
    wr = float(wins) / float(len(ser)) if len(ser) else 0.0
    return wr >= 0.40, wr, len(ser)


def _nan_share(df: pd.DataFrame, cols: List[str]) -> float:
    present = [c for c in cols if c in df.columns]
    if not present:
        return 0.0
    sub = df[present]
    return float(sub.isna().mean().mean())


def _daily_trade_cap_exceeded(dates: pd.Series, cap: int) -> Tuple[bool, int]:
    if dates.empty:
        return False, 0
    days = dates.dt.normalize().value_counts()
    peak = int(days.max()) if not days.empty else 0
    return peak > cap, peak


def _holding_days(ledger: pd.DataFrame) -> pd.Series:
    if "entry_date" in ledger.columns and "exit_date" in ledger.columns:
        ed = pd.to_datetime(ledger["entry_date"], errors="coerce")
        xd = pd.to_datetime(ledger["exit_date"], errors="coerce")
        return (xd - ed).dt.days
    return pd.Series(index=ledger.index, dtype=float)


def _mean_tenor_days(ledger: pd.DataFrame) -> Optional[float]:
    for c in ["tenor_days", "option_tenor_days", "tenor", "days_to_expiry"]:
        if c in ledger.columns:
            s = pd.to_numeric(ledger[c], errors="coerce")
            m = float(s.dropna().mean()) if s.notna().any() else None
            if m is not None:
                return m
    return None


def _iv_availability_share(mkt: pd.DataFrame) -> float:
    if mkt is None or mkt.empty:
        return np.nan
    # consider explicit IV columns + any '*IMP_VOL*'
    cols = list(IV_COLUMNS) + [c for c in mkt.columns if "IMP_VOL" in c.upper() and c not in IV_COLUMNS]
    present = [c for c in cols if c in mkt.columns]
    if not present:
        return np.nan
    any_iv_present = mkt[present].notna().any(axis=1)
    return float(any_iv_present.mean())


def _quote_sanity_share(mkt: pd.DataFrame) -> float:
    """Share of rows with sane quotes and OHLC ordering when present."""
    if mkt is None or mkt.empty:
        return np.nan
    ok = pd.Series(True, index=mkt.index)
    if BID_COL in mkt.columns and ASK_COL in mkt.columns:
        bid = pd.to_numeric(mkt[BID_COL], errors="coerce")
        ask = pd.to_numeric(mkt[ASK_COL], errors="coerce")
        ok = ok & (bid <= ask)
    if all(c in mkt.columns for c in (OPEN_COL, HIGH_COL, LOW_COL)):
        op = pd.to_numeric(mkt[OPEN_COL], errors="coerce")
        hi = pd.to_numeric(mkt[HIGH_COL], errors="coerce")
        lo = pd.to_numeric(mkt[LOW_COL], errors="coerce")
        ok = ok & (lo <= op) & (op <= hi) & (lo <= hi)
    return float(ok.mean()) if len(ok) else np.nan


def _liquidity_proxies_ok(mkt: pd.DataFrame, min_vol: int, min_oi_sum: int) -> Tuple[Optional[bool], Optional[float], Optional[float]]:
    """Return (ok?, share_ok_vol, share_ok_oi) using aggregate proxies if present."""
    if mkt is None or mkt.empty:
        return None, None, None
    share_ok_vol = None
    if LIQ_VOLUME_COL in mkt.columns:
        ok_vol = pd.to_numeric(mkt[LIQ_VOLUME_COL], errors="coerce") >= int(min_vol)
        share_ok_vol = float(ok_vol.mean())
    share_ok_oi = None
    oi_present = [c for c in LIQ_OI_COLS if c in mkt.columns]
    if oi_present:
        oi_sum = sum(pd.to_numeric(mkt[c], errors="coerce").fillna(0) for c in oi_present)
        ok_oi = oi_sum >= int(min_oi_sum)
        share_ok_oi = float(ok_oi.mean())
    if share_ok_vol is None and share_ok_oi is None:
        return None, None, None
    # Combine rule: if both present, require both ≥ 0.95; if one present, require it ≥ 0.95.
    return (
        ((share_ok_vol is None) or (share_ok_vol >= 0.95)) and
        ((share_ok_oi  is None) or (share_ok_oi  >= 0.95)),
        share_ok_vol, share_ok_oi
    )


# ------------------------------ Main ------------------------------

def run_stage1_health_check(
    run_dir: Optional[str] = None,
    fold_num: Optional[int] = None,
    settings: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    fold_summary: Optional[pd.DataFrame] = None,
    fold_ledger: Optional[pd.DataFrame] = None,
    # Optional: market daily dataframe (same period/asset) for IV/liquidity/quotes checks
    market_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Stage 1: Health & Sanity hard-gates.
    See module docstring for details and config keys.
    """
    cfg = dict(config or {})
    # Defaults
    rolling_window_days      = int(cfg.get("s1_recent_window_days", 7))
    min_recent_trades        = int(cfg.get("s1_min_recent_trades", 1))
    min_total_trades         = int(cfg.get("s1_min_total_trades", 5))
    momentum_window_days     = int(cfg.get("s1_momentum_window_days", 30))
    min_momentum_trades      = int(cfg.get("s1_min_momentum_trades", 3))
    iv_availability_min      = float(cfg.get("s1_iv_availability_min", 0.98))
    strike_success_min       = float(cfg.get("s1_strike_success_min", 0.97))
    missing_tol              = float(cfg.get("s1_missing_data_tolerance", 0.01))
    mean_hold_ratio_min      = float(cfg.get("s1_mean_holding_ratio_min", 0.40))
    daily_cap                = int(cfg.get("s1_daily_trade_cap", 12))
    min_tot_opt_volume       = int(cfg.get("s1_min_tot_opt_volume", 1000))
    min_open_interest_sum    = int(cfg.get("s1_min_open_interest_sum", 5000))
    quote_sanity_min_share   = float(cfg.get("s1_quote_sanity_min_ok_share", 0.99))

    # Defensive: ensure inputs
    if fold_summary is None or fold_summary.empty or fold_ledger is None or fold_ledger.empty:
        return pd.DataFrame([{
            "setup_id": None, "rank": None, "pass_stage1": False,
            "reject_code": "S1_INPUT_MISSING",
            "reason": "missing_ledger_or_summary",
        }])

    ledger = fold_ledger.copy()
    # Normalize dates
    for dc in [c for c in ledger.columns if ("date" in c.lower() or "time" in c.lower())]:
        ledger[dc] = pd.to_datetime(ledger[dc], errors="coerce")

    setup_col = _infer_setup_col(ledger)
    sid  = str(fold_summary["setup_id"].iloc[0]) if "setup_id" in fold_summary.columns else None
    rank = fold_summary["rank"].iloc[0] if "rank" in fold_summary.columns else None

    # Focus ledger on this setup if possible
    led = ledger[ledger[setup_col].astype(str) == str(sid)].copy() if (sid is not None and setup_col in ledger.columns) else ledger.copy()
    data_end = _data_end_from_ledger(led)
    if data_end is None:
        return pd.DataFrame([{
            "setup_id": sid, "rank": rank, "pass_stage1": False,
            "reject_code": "S1_NO_DATA_END",
            "reason": "cannot_determine_data_end",
        }])

    # ------------------------------ Checks ------------------------------
    failures: List[str] = []
    reject_code: Optional[str] = None

    # 1) Recency & activity
    recency_col = "trigger_date" if "trigger_date" in led.columns and led["trigger_date"].notna().any() \
                  else "entry_date" if "entry_date" in led.columns and led["entry_date"].notna().any() \
                  else None
    if recency_col is None:
        return pd.DataFrame([{
            "setup_id": sid, "rank": rank, "pass_stage1": False,
            "reject_code": "S1_NO_TIMESTAMP",
            "reason": "no_trigger_or_entry_dates",
        }])
    recency_mask = _get_recent_mask(pd.to_datetime(led[recency_col], errors="coerce"), data_end, rolling_window_days)
    recent_trades = int(recency_mask.sum())
    total_trades  = int(len(led))
    pass_recency  = recent_trades >= min_recent_trades
    pass_activity = total_trades  >= min_total_trades
    if not pass_recency:
        failures.append(f"recent_trades={recent_trades}<{min_recent_trades}")
        reject_code = reject_code or "S1_RECENCY"
    if not pass_activity:
        failures.append(f"total_trades={total_trades}<{min_total_trades}")
        reject_code = reject_code or "S1_ACTIVITY"

    # 2) Momentum sanity (only if enough recent trades in a longer window)
    momentum_mask = _get_recent_mask(pd.to_datetime(led[recency_col], errors="coerce"), data_end, momentum_window_days)
    pass_mom, recent_wr, n_recent_eval = _safe_win_rate(led, momentum_mask, min_momentum_trades)
    if not pass_mom and n_recent_eval >= min_momentum_trades:
        failures.append(f"recent_win_rate={recent_wr:.2f}<0.40")
        reject_code = reject_code or "S1_MOMENTUM"

    # 3) IV availability on market data (soft-hard gate depending on presence)
    iv_availability = np.nan
    if market_df is not None and not market_df.empty:
        iv_availability = _iv_availability_share(market_df)
        # Gate only if we could actually compute availability (i.e., IV fields exist)
        if not np.isnan(iv_availability) and iv_availability < iv_availability_min:
            failures.append(f"iv_availability={iv_availability:.3f}<{iv_availability_min:.2f}")
            reject_code = reject_code or "S1_IV_AVAIL"

    # 4) Strike selection success (if columns exist on ledger)
    strike_success = None
    if "strike_selected" in led.columns:
        strike_success = float(pd.Series(led["strike_selected"]).astype(float).mean())
    elif "selected_strike" in led.columns:
        strike_success = float(led["selected_strike"].notna().mean())
    if strike_success is not None and strike_success < strike_success_min:
        failures.append(f"strike_success={strike_success:.3f}<{strike_success_min:.2f}")
        reject_code = reject_code or "S1_STRIKE"

    # 5) Missing-data tolerance on key market cols (if present)
    missing_share_key = _nan_share(market_df, KEY_MARKET_COLS) if (market_df is not None and not market_df.empty) else np.nan
    if not np.isnan(missing_share_key) and missing_share_key > missing_tol:
        failures.append(f"missing_share_keycols={missing_share_key:.3f}>{missing_tol:.2f}")
        reject_code = reject_code or "S1_MISSING"

    # 6) Tenor coherence: mean holding >= ratio * mean tenor (if both available on ledger)
    mean_hold_days  = float(_holding_days(led).dropna().mean()) if not _holding_days(led).dropna().empty else np.nan
    mean_tenor_days = _mean_tenor_days(led)
    hold_ratio = np.nan
    if not np.isnan(mean_hold_days) and (mean_tenor_days is not None) and mean_tenor_days > 0:
        hold_ratio = float(mean_hold_days) / float(mean_tenor_days)
        if hold_ratio < mean_hold_ratio_min:
            failures.append(f"mean_holding_ratio={hold_ratio:.2f}<{mean_hold_ratio_min:.2f}")
            reject_code = reject_code or "S1_HOLDING"

    # 7) Daily trade cap (ledger timestamps)
    cap_exceeded, peak_daily = _daily_trade_cap_exceeded(pd.to_datetime(led[recency_col], errors="coerce"), daily_cap)
    if cap_exceeded:
        failures.append(f"peak_daily_trades={peak_daily}>{daily_cap}")
        reject_code = reject_code or "S1_DAILY_CAP"

    # 8) Liquidity proxies (market aggregates)
    liq_ok, share_ok_vol, share_ok_oi = (None, None, None)
    if market_df is not None and not market_df.empty:
        liq_ok, share_ok_vol, share_ok_oi = _liquidity_proxies_ok(market_df, min_tot_opt_volume, min_open_interest_sum)
        if liq_ok is False:  # only gate if we could evaluate
            parts = []
            if share_ok_vol is not None and share_ok_vol < 0.95:
                parts.append(f"{LIQ_VOLUME_COL} ok_share={share_ok_vol:.2f}<0.95 (min {min_tot_opt_volume})")
            if share_ok_oi  is not None and share_ok_oi  < 0.95:
                parts.append(f"OPEN_INT_TOTAL_* ok_share={share_ok_oi:.2f}<0.95 (min {min_open_interest_sum})")
            if parts:
                failures.append("liquidity_proxy_fail: " + "; ".join(parts))
                reject_code = reject_code or "S1_LIQ_PROXY"

    # 9) Quote sanity (market)
    quote_sanity_share = np.nan
    if market_df is not None and not market_df.empty:
        quote_sanity_share = _quote_sanity_share(market_df)
        if not np.isnan(quote_sanity_share) and quote_sanity_share < quote_sanity_min_share:
            failures.append(f"quote_sanity_share={quote_sanity_share:.3f}<{quote_sanity_min_share:.2f}")
            reject_code = reject_code or "S1_QUOTES"

    # Aggregate decision
    passed = len(failures) == 0
    reason = "ok" if passed else ";".join(failures) or "failed"

    # Output row
    return pd.DataFrame([{
        "setup_id": sid,
        "rank": rank,
        "pass_stage1": bool(passed),
        "reject_code": None if passed else (reject_code or "S1_FAIL"),
        "reason": reason,

        # Recency/activity/momentum diagnostics
        "recent_window_days": int(rolling_window_days),
        "recent_trades_count": int(recent_trades),
        "min_recent_trades": int(min_recent_trades),
        "total_trades_count": int(total_trades),
        "min_total_trades": int(min_total_trades),
        "momentum_window_days": int(momentum_window_days),
        "min_momentum_trades": int(min_momentum_trades),
        "recent_win_rate": float(recent_wr) if not np.isnan(recent_wr) else np.nan,
        "recent_win_n": int(n_recent_eval),

        # IV availability
        "iv_availability": float(iv_availability) if not np.isnan(iv_availability) else np.nan,
        "iv_availability_min": float(iv_availability_min),

        # Strike success
        "strike_success": float(strike_success) if strike_success is not None else np.nan,
        "strike_success_min": float(strike_success_min),

        # Missing data on key market cols
        "missing_share_keycols": float(missing_share_key) if not np.isnan(missing_share_key) else np.nan,
        "missing_share_tol": float(missing_tol),

        # Tenor/holding coherence
        "mean_holding_days": float(mean_hold_days) if not np.isnan(mean_hold_days) else np.nan,
        "mean_tenor_days": float(mean_tenor_days) if (mean_tenor_days is not None) else np.nan,
        "mean_holding_ratio": float(hold_ratio) if not np.isnan(hold_ratio) else np.nan,
        "mean_holding_ratio_min": float(mean_hold_ratio_min),

        # Daily cap
        "peak_daily_trades": int(peak_daily),
        "daily_trade_cap": int(daily_cap),

        # Liquidity proxies
        "tot_opt_volume_min": int(min_tot_opt_volume),
        "open_interest_sum_min": int(min_open_interest_sum),
        "liquidity_ok_share_vol": float(share_ok_vol) if share_ok_vol is not None else np.nan,
        "liquidity_ok_share_oi": float(share_ok_oi) if share_ok_oi is not None else np.nan,
        "liquidity_proxies_ok": bool(liq_ok) if liq_ok is not None else np.nan,

        # Quote sanity
        "quote_sanity_share": float(quote_sanity_share) if not np.isnan(quote_sanity_share) else np.nan,
        "quote_sanity_min_ok_share": float(quote_sanity_min_share),
    }])
