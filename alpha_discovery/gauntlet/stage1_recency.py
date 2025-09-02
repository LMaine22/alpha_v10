# alpha_discovery/gauntlet/stage1_recency.py
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def _infer_setup_col(df: pd.DataFrame) -> str:
    for c in ["setup_id", "Setup", "strategy_id", "StrategyID"]:
        if c in df.columns:
            return c
    if "setup_id" not in df.columns:
        df["setup_id"] = "unknown"
    return "setup_id"


def _data_end_from_ledger(ledger: pd.DataFrame) -> Optional[pd.Timestamp]:
    """
    End-of-data defined STRICTLY by trigger/entry timestamps.
    exit_date is ignored to prevent phantom exits from skewing recency.
    """
    candidates = []
    for c in ("trigger_date", "entry_date"):
        if c in ledger.columns:
            s = pd.to_datetime(ledger[c], errors="coerce")
            m = s.max()
            if pd.notnull(m):
                candidates.append(m)
    return max(candidates) if candidates else None


def _short_window_mask(dates: pd.Series, data_end: pd.Timestamp, window_days: int) -> pd.Series:
    return (data_end.normalize() - dates.dt.normalize()).dt.days <= int(window_days)


def _compute_short_window_dd(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)
    return float(dd.fillna(0.0).max())


def _build_equity_from_ledger(ledger: pd.DataFrame) -> pd.Series:
    pnl_cols = [c for c in ["realized_pnl", "unrealized_pnl", "pnl", "PnL"] if c in ledger.columns]
    if pnl_cols:
        sort_col = "trigger_date" if "trigger_date" in ledger.columns else (
            "entry_date" if "entry_date" in ledger.columns else None
        )
        if sort_col is None:
            eq = ledger[pnl_cols].sum(axis=1).cumsum()
            eq.index = pd.RangeIndex(len(eq))
            return eq
        tmp = ledger.copy()
        tmp[sort_col] = pd.to_datetime(tmp[sort_col], errors="coerce")
        tmp = tmp.sort_values(sort_col)
        eq = tmp[pnl_cols].sum(axis=1).cumsum()
        eq.index = tmp[sort_col].values
        return eq
    return pd.Series(dtype=float)


def run_stage1_recency_liveness(
    run_dir: Optional[str] = None,
    fold_num: Optional[int] = None,
    settings: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    fold_summary: Optional[pd.DataFrame] = None,
    fold_ledger: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Stage-1 Recency/Liveness gate operating directly on the provided ledger+summary row.
    Output: single-row DataFrame with pass flag and diagnostics.
    """
    cfg = dict(config or {})

    # STRICT defaults (tune in config if you like)
    recency_max_days = int(cfg.get("s1_recency_max_days", 7))
    short_window_days = int(cfg.get("s1_short_window_days", 20))
    min_trades_short = int(cfg.get("s1_min_trades_short", 2))
    max_dd_short_cap = float(cfg.get("s1_max_drawdown_short", 0.15))  # 15%

    # Require both overrides
    if fold_summary is None or fold_summary.empty or fold_ledger is None or fold_ledger.empty:
        return pd.DataFrame([{
            "setup_id": None, "rank": None, "pass_stage1": False,
            "days_since_last_trigger": None, "short_window_days": short_window_days,
            "trades_in_short_window": 0, "max_dd_short": None,
            "recency_max_days": recency_max_days, "min_trades_short": min_trades_short,
            "max_dd_short_cap": max_dd_short_cap, "reason": "missing_ledger_or_summary",
        }])

    ledger = fold_ledger.copy()
    for dc in [c for c in ledger.columns if "date" in c.lower() or "time" in c.lower()]:
        ledger[dc] = pd.to_datetime(ledger[dc], errors="coerce")

    setup_col = _infer_setup_col(ledger)
    sid = str(fold_summary["setup_id"].iloc[0]) if "setup_id" in fold_summary.columns else None
    rank = fold_summary["rank"].iloc[0] if "rank" in fold_summary.columns else None

    # Recency is based on trigger_date when available, else entry_date
    if "trigger_date" in ledger.columns and ledger["trigger_date"].notna().any():
        recency_col = "trigger_date"
    elif "entry_date" in ledger.columns and ledger["entry_date"].notna().any():
        recency_col = "entry_date"
    else:
        return pd.DataFrame([{
            "setup_id": sid, "rank": rank, "pass_stage1": False,
            "days_since_last_trigger": None, "short_window_days": short_window_days,
            "trades_in_short_window": 0, "max_dd_short": None,
            "recency_max_days": recency_max_days, "min_trades_short": min_trades_short,
            "max_dd_short_cap": max_dd_short_cap, "reason": "no_trigger_or_entry_dates",
        }])

    data_end = _data_end_from_ledger(ledger)

    # Use only this setup's rows when the ledger carries setup ids
    led_this = ledger[ledger[setup_col].astype(str) == str(sid)].copy() if sid is not None and setup_col in ledger.columns else ledger.copy()

    last_trig = pd.to_datetime(led_this[recency_col], errors="coerce").max()
    days_since_last = int((data_end.normalize() - last_trig.normalize()).days) if pd.notnull(last_trig) and data_end is not None else None
    if days_since_last is None:
        return pd.DataFrame([{
            "setup_id": sid, "rank": rank, "pass_stage1": False,
            "days_since_last_trigger": None, "short_window_days": short_window_days,
            "trades_in_short_window": 0, "max_dd_short": None,
            "recency_max_days": recency_max_days, "min_trades_short": min_trades_short,
            "max_dd_short_cap": max_dd_short_cap, "reason": "cannot_compute_recency",
        }])

    mask_short = _short_window_mask(pd.to_datetime(led_this[recency_col], errors="coerce"), data_end, short_window_days)
    trades_in_short = int(mask_short.sum())

    equity = _build_equity_from_ledger(led_this)
    if isinstance(equity.index, pd.DatetimeIndex) and data_end is not None:
        idx_norm = pd.to_datetime(equity.index).normalize()
        ages_days = (data_end.normalize() - idx_norm) / pd.Timedelta(days=1)
        equity = equity[ages_days <= short_window_days]
    max_dd_short = _compute_short_window_dd(equity)

    pass_recency = days_since_last <= recency_max_days
    pass_trades = trades_in_short >= min_trades_short
    pass_dd = (max_dd_short is None) or (max_dd_short <= max_dd_short_cap)

    passed = bool(pass_recency and pass_trades and pass_dd)
    reason = "ok" if passed else ";".join([
        s for s in [
            (None if pass_recency else f"recency={days_since_last}d>max{recency_max_days}"),
            (None if pass_trades else f"trades_short={trades_in_short}<min{min_trades_short}"),
            (None if pass_dd else f"dd_short={max_dd_short:.4f}>cap{max_dd_short_cap:.4f}"),
        ] if s
    ]) or "failed"

    return pd.DataFrame([{
        "setup_id": sid, "rank": rank, "pass_stage1": passed,
        "days_since_last_trigger": days_since_last,
        "short_window_days": short_window_days,
        "trades_in_short_window": trades_in_short,
        "max_dd_short": float(max_dd_short) if max_dd_short is not None else None,
        "recency_max_days": recency_max_days,
        "min_trades_short": min_trades_short,
        "max_dd_short_cap": max_dd_short_cap,
        "reason": reason,
    }])
