# alpha_discovery/gauntlet/stage1_health.py
"""
Stage 1: Signal Health Check
- Rolling window recency (has setup traded recently?)
- Minimum activity (enough trades in recent period)
- Positive momentum (not all recent trades are losers)
- Basic heartbeat check: is the signal alive and firing?
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def _infer_setup_col(df: pd.DataFrame) -> str:
    """Find the setup identifier column."""
    for c in ["setup_id", "Setup", "strategy_id", "StrategyID"]:
        if c in df.columns:
            return c
    if "setup_id" not in df.columns:
        df["setup_id"] = "unknown"
    return "setup_id"


def _data_end_from_ledger(ledger: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Get the end date of the data from trigger/entry timestamps."""
    candidates = []
    for c in ("trigger_date", "entry_date"):
        if c in ledger.columns:
            s = pd.to_datetime(ledger[c], errors="coerce")
            m = s.max()
            if pd.notnull(m):
                candidates.append(m)
    return max(candidates) if candidates else None


def _get_recent_trades_mask(dates: pd.Series, data_end: pd.Timestamp, window_days: int) -> pd.Series:
    """Get mask for trades within the rolling window."""
    return (data_end.normalize() - dates.dt.normalize()).dt.days <= int(window_days)


def _check_positive_momentum(ledger: pd.DataFrame, recent_mask: pd.Series, min_recent_trades: int) -> tuple[bool, float]:
    """
    Check if recent trades show positive momentum.
    Returns: (has_momentum, win_rate)
    """
    recent_ledger = ledger[recent_mask]
    if len(recent_ledger) < min_recent_trades:
        return False, 0.0
    
    # Use PnL column if available, otherwise compute from entry/exit prices
    if 'pnl_pct' in recent_ledger.columns:
        pnl_series = recent_ledger['pnl_pct']
    elif 'realized_pnl' in recent_ledger.columns:
        # Convert to percentage if needed
        pnl_series = recent_ledger['realized_pnl'] / recent_ledger.get('entry_exec', 1.0) * 100
    else:
        # Fallback: assume no momentum if we can't determine PnL
        return False, 0.0
    
    wins = (pnl_series > 0).sum()
    total = len(pnl_series)
    win_rate = wins / total if total > 0 else 0.0
    
    # Require at least 40% win rate for positive momentum
    has_momentum = win_rate >= 0.4
    return has_momentum, win_rate


def run_stage1_health_check(
    run_dir: Optional[str] = None,
    fold_num: Optional[int] = None,
    settings: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    fold_summary: Optional[pd.DataFrame] = None,
    fold_ledger: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Stage 1: Signal Health Check
    
    Checks:
    1. Rolling window recency: Has setup traded in last N days?
    2. Minimum activity: At least X trades in last Y days
    3. Positive momentum: Recent trades not all losers
    4. Basic heartbeat: Signal is alive and firing
    """
    cfg = dict(config or {})
    
    # Configuration with sensible defaults
    rolling_window_days = int(cfg.get("s1_rolling_window_days", 7))  # Last 7 days
    min_recent_trades = int(cfg.get("s1_min_recent_trades", 1))      # At least 1 trade in window
    min_total_trades = int(cfg.get("s1_min_total_trades", 5))        # At least 5 trades total
    momentum_window_days = int(cfg.get("s1_momentum_window_days", 30))  # Check momentum in last 30 days
    min_momentum_trades = int(cfg.get("s1_min_momentum_trades", 3))  # Need 3 trades for momentum check
    
    # Require both inputs
    if fold_summary is None or fold_summary.empty or fold_ledger is None or fold_ledger.empty:
        return pd.DataFrame([{
            "setup_id": None, "rank": None, "pass_stage1": False,
            "rolling_window_days": rolling_window_days, "min_recent_trades": min_recent_trades,
            "min_total_trades": min_total_trades, "recent_trades_count": 0,
            "total_trades_count": 0, "has_positive_momentum": False, "recent_win_rate": 0.0,
            "reason": "missing_ledger_or_summary",
        }])

    ledger = fold_ledger.copy()
    for dc in [c for c in ledger.columns if "date" in c.lower() or "time" in c.lower()]:
        ledger[dc] = pd.to_datetime(ledger[dc], errors="coerce")

    setup_col = _infer_setup_col(ledger)
    sid = str(fold_summary["setup_id"].iloc[0]) if "setup_id" in fold_summary.columns else None
    rank = fold_summary["rank"].iloc[0] if "rank" in fold_summary.columns else None

    # Get recency column (trigger_date preferred, entry_date fallback)
    if "trigger_date" in ledger.columns and ledger["trigger_date"].notna().any():
        recency_col = "trigger_date"
    elif "entry_date" in ledger.columns and ledger["entry_date"].notna().any():
        recency_col = "entry_date"
    else:
        return pd.DataFrame([{
            "setup_id": sid, "rank": rank, "pass_stage1": False,
            "rolling_window_days": rolling_window_days, "min_recent_trades": min_recent_trades,
            "min_total_trades": min_total_trades, "recent_trades_count": 0,
            "total_trades_count": 0, "has_positive_momentum": False, "recent_win_rate": 0.0,
            "reason": "no_trigger_or_entry_dates",
        }])

    data_end = _data_end_from_ledger(ledger)
    if data_end is None:
        return pd.DataFrame([{
            "setup_id": sid, "rank": rank, "pass_stage1": False,
            "rolling_window_days": rolling_window_days, "min_recent_trades": min_recent_trades,
            "min_total_trades": min_total_trades, "recent_trades_count": 0,
            "total_trades_count": 0, "has_positive_momentum": False, "recent_win_rate": 0.0,
            "reason": "cannot_determine_data_end",
        }])

    # Filter to this setup's trades
    led_this = ledger[ledger[setup_col].astype(str) == str(sid)].copy() if sid is not None and setup_col in ledger.columns else ledger.copy()
    
    # Check 1: Rolling window recency
    recent_mask = _get_recent_trades_mask(
        pd.to_datetime(led_this[recency_col], errors="coerce"), 
        data_end, 
        rolling_window_days
    )
    recent_trades_count = int(recent_mask.sum())
    
    # Check 2: Total activity
    total_trades_count = len(led_this)
    
    # Check 3: Positive momentum (in longer window)
    momentum_mask = _get_recent_trades_mask(
        pd.to_datetime(led_this[recency_col], errors="coerce"), 
        data_end, 
        momentum_window_days
    )
    has_momentum, recent_win_rate = _check_positive_momentum(led_this, momentum_mask, min_momentum_trades)
    
    # Apply all checks
    pass_recency = recent_trades_count >= min_recent_trades
    pass_activity = total_trades_count >= min_total_trades
    pass_momentum = has_momentum or recent_trades_count < min_momentum_trades  # Skip momentum if not enough trades
    
    passed = bool(pass_recency and pass_activity and pass_momentum)
    
    # Build reason string
    reasons = []
    if not pass_recency:
        reasons.append(f"recent_trades={recent_trades_count}<{min_recent_trades}")
    if not pass_activity:
        reasons.append(f"total_trades={total_trades_count}<{min_total_trades}")
    if not pass_momentum and recent_trades_count >= min_momentum_trades:
        reasons.append(f"win_rate={recent_win_rate:.2f}<0.40")
    
    reason = "ok" if passed else ";".join(reasons) or "failed"

    return pd.DataFrame([{
        "setup_id": sid, "rank": rank, "pass_stage1": passed,
        "rolling_window_days": rolling_window_days,
        "min_recent_trades": min_recent_trades,
        "min_total_trades": min_total_trades,
        "recent_trades_count": recent_trades_count,
        "total_trades_count": total_trades_count,
        "has_positive_momentum": has_momentum,
        "recent_win_rate": float(recent_win_rate),
        "reason": reason,
    }])
