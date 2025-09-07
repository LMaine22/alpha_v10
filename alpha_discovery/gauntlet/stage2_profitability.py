# alpha_discovery/gauntlet/stage2_profitability.py
"""
Stage 2: OOS Profitability Check
- Total OOS NAV growth: require positive overall NAV
- Cumulative $ PnL: sum of PnL across OOS trades must be positive
- Hit rate / payoff balance: win% or payoff ratio thresholds
- Rolling PnL check: recent OOS trades also profitable
- Drawdown sanity: cap max drawdown relative to total NAV
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

try:
    from ..config import Settings
except Exception:
    class Settings:  # type: ignore
        pass

from ..eval.nav import nav_daily_returns_from_ledger, sharpe


def _compute_nav_metrics(ledger: pd.DataFrame, base_capital: float = 100000.0) -> Dict[str, float]:
    """Compute NAV-based metrics from trade ledger."""
    try:
        returns = nav_daily_returns_from_ledger(ledger, base_capital=base_capital)
        if len(returns) == 0:
            return {"nav_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 1.0, "volatility": 0.0}
        
        # NAV return
        nav_return_pct = (returns.cumsum().iloc[-1] / base_capital) * 100
        
        # Sharpe ratio
        sharpe_ratio = float(sharpe(returns))
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(abs(drawdown.min()))
        
        # Volatility
        volatility = float(returns.std() * np.sqrt(252))
        
        return {
            "nav_return_pct": nav_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility
        }
    except Exception:
        return {"nav_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 1.0, "volatility": 0.0}


def _compute_trade_metrics(ledger: pd.DataFrame) -> Dict[str, float]:
    """Compute trade-based metrics from ledger."""
    if len(ledger) == 0:
        return {"total_pnl": 0.0, "win_rate": 0.0, "payoff_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    
    # Get PnL column
    pnl_col = None
    for col in ["pnl_pct", "realized_pnl", "pnl", "PnL"]:
        if col in ledger.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        return {"total_pnl": 0.0, "win_rate": 0.0, "payoff_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    
    pnl_series = pd.to_numeric(ledger[pnl_col], errors="coerce").fillna(0.0)
    
    # Total PnL
    total_pnl = float(pnl_series.sum())
    
    # Win rate and payoff ratio
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    
    win_rate = len(wins) / len(pnl_series) if len(pnl_series) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0.0
    
    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }


def _compute_recent_metrics(ledger: pd.DataFrame, recent_days: int = 30) -> Dict[str, float]:
    """Compute metrics for recent trades only."""
    if len(ledger) == 0:
        return {"recent_nav_return": 0.0, "recent_trades": 0}
    
    # Get date column
    date_col = None
    for col in ["trigger_date", "entry_date", "exit_date"]:
        if col in ledger.columns:
            date_col = col
            break
    
    if date_col is None:
        return {"recent_nav_return": 0.0, "recent_trades": 0}
    
    # Filter to recent trades
    ledger[date_col] = pd.to_datetime(ledger[date_col], errors="coerce")
    cutoff_date = ledger[date_col].max() - pd.Timedelta(days=recent_days)
    recent_ledger = ledger[ledger[date_col] >= cutoff_date]
    
    if len(recent_ledger) == 0:
        return {"recent_nav_return": 0.0, "recent_trades": 0}
    
    # Compute recent NAV return
    nav_metrics = _compute_nav_metrics(recent_ledger)
    
    return {
        "recent_nav_return": nav_metrics["nav_return_pct"],
        "recent_trades": len(recent_ledger)
    }


def run_stage2_profitability(
    run_dir: str,
    fold_num: int,
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Stage 2: OOS Profitability Check
    
    Checks:
    1. Total OOS NAV growth: positive overall NAV
    2. Cumulative $ PnL: positive total PnL
    3. Hit rate / payoff balance: win% or payoff ratio thresholds
    4. Rolling PnL check: recent trades also profitable
    5. Drawdown sanity: reasonable max drawdown
    """
    cfg = dict(config or {})
    
    # Configuration
    min_nav_return_pct = float(cfg.get("s2_min_nav_return_pct", 0.0))      # Must be profitable
    min_total_pnl = float(cfg.get("s2_min_total_pnl", 0.0))                # Positive PnL
    min_win_rate = float(cfg.get("s2_min_win_rate", 0.0))                  # Win rate threshold
    min_payoff_ratio = float(cfg.get("s2_min_payoff_ratio", 0.0))          # Payoff ratio threshold
    max_drawdown_pct = float(cfg.get("s2_max_drawdown_pct", 0.50))         # Max 50% drawdown
    recent_days = int(cfg.get("s2_recent_days", 30))                       # Recent window
    min_recent_nav_return = float(cfg.get("s2_min_recent_nav_return", 0.0)) # Recent profitability
    min_recent_trades = int(cfg.get("s2_min_recent_trades", 1))            # Min recent trades
    
    if stage1_df is None or stage1_df.empty:
        return pd.DataFrame(columns=[
            "setup_id", "rank", "nav_return_pct", "total_pnl", "win_rate", 
            "payoff_ratio", "max_drawdown", "recent_nav_return", "recent_trades",
            "pass_stage2", "reason"
        ])

    setup_id = stage1_df["setup_id"].iloc[0]
    rank = stage1_df.get("rank", pd.Series([None])).iloc[0]

    # Load OOS ledger
    import os
    cand_paths = [
        os.path.join(run_dir, f"fold_{fold_num}", "pareto_ledger.csv"),
        os.path.join(run_dir, f"fold_{fold_num}", "front", "pareto_ledger.csv"),
    ]
    fold_ledger = None
    for p in cand_paths:
        if os.path.exists(p):
            try:
                fold_ledger = pd.read_csv(p)
                break
            except Exception:
                pass
    
    if fold_ledger is None or not isinstance(fold_ledger, pd.DataFrame) or fold_ledger.empty:
        return pd.DataFrame([{
            "setup_id": setup_id, "rank": rank, "nav_return_pct": 0.0, "total_pnl": 0.0,
            "win_rate": 0.0, "payoff_ratio": 0.0, "max_drawdown": 1.0,
            "recent_nav_return": 0.0, "recent_trades": 0, "pass_stage2": False,
            "reason": "no_oos_ledger"
        }])

    # Get base capital
    try:
        base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_cap = 100_000.0

    # Compute all metrics
    nav_metrics = _compute_nav_metrics(fold_ledger, base_cap)
    trade_metrics = _compute_trade_metrics(fold_ledger)
    recent_metrics = _compute_recent_metrics(fold_ledger, recent_days)
    
    # Apply checks
    pass_nav = nav_metrics["nav_return_pct"] >= min_nav_return_pct
    pass_pnl = trade_metrics["total_pnl"] >= min_total_pnl
    pass_winrate = trade_metrics["win_rate"] >= min_win_rate
    pass_payoff = trade_metrics["payoff_ratio"] >= min_payoff_ratio
    pass_drawdown = nav_metrics["max_drawdown"] <= max_drawdown_pct
    pass_recent_nav = recent_metrics["recent_nav_return"] >= min_recent_nav_return
    pass_recent_trades = recent_metrics["recent_trades"] >= min_recent_trades
    
    # Overall pass (all checks must pass)
    passed = bool(pass_nav and pass_pnl and pass_drawdown and pass_recent_nav and pass_recent_trades and 
                  (pass_winrate or pass_payoff))  # Either win rate OR payoff ratio must pass
    
    # Build reason string
    reasons = []
    if not pass_nav:
        reasons.append(f"nav_return={nav_metrics['nav_return_pct']:.2f}%<{min_nav_return_pct}%")
    if not pass_pnl:
        reasons.append(f"total_pnl={trade_metrics['total_pnl']:.2f}<{min_total_pnl}")
    if not pass_winrate and not pass_payoff:
        reasons.append(f"win_rate={trade_metrics['win_rate']:.2f}<{min_win_rate} OR payoff={trade_metrics['payoff_ratio']:.2f}<{min_payoff_ratio}")
    if not pass_drawdown:
        reasons.append(f"max_dd={nav_metrics['max_drawdown']:.2f}>{max_drawdown_pct}")
    if not pass_recent_nav:
        reasons.append(f"recent_nav={recent_metrics['recent_nav_return']:.2f}%<{min_recent_nav_return}%")
    if not pass_recent_trades:
        reasons.append(f"recent_trades={recent_metrics['recent_trades']}<{min_recent_trades}")
    
    reason = "ok" if passed else ";".join(reasons) or "failed"

    return pd.DataFrame([{
        "setup_id": setup_id, "rank": rank,
        "nav_return_pct": nav_metrics["nav_return_pct"],
        "total_pnl": trade_metrics["total_pnl"],
        "win_rate": trade_metrics["win_rate"],
        "payoff_ratio": trade_metrics["payoff_ratio"],
        "max_drawdown": nav_metrics["max_drawdown"],
        "recent_nav_return": recent_metrics["recent_nav_return"],
        "recent_trades": recent_metrics["recent_trades"],
        "pass_stage2": passed,
        "reason": reason,
    }])


def run_stage2_profitability_on_ledger(
    fold_ledger: pd.DataFrame,
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Stage 2 for an arbitrary ledger (not tied to a run_dir/fold)."""
    cfg = dict(config or {})
    
    # Same configuration as above
    min_nav_return_pct = float(cfg.get("s2_min_nav_return_pct", 0.0))
    min_total_pnl = float(cfg.get("s2_min_total_pnl", 0.0))
    min_win_rate = float(cfg.get("s2_min_win_rate", 0.0))
    min_payoff_ratio = float(cfg.get("s2_min_payoff_ratio", 0.0))
    max_drawdown_pct = float(cfg.get("s2_max_drawdown_pct", 0.50))
    recent_days = int(cfg.get("s2_recent_days", 30))
    min_recent_nav_return = float(cfg.get("s2_min_recent_nav_return", 0.0))
    min_recent_trades = int(cfg.get("s2_min_recent_trades", 1))

    if stage1_df is None or stage1_df.empty:
        return pd.DataFrame(columns=[
            "setup_id", "rank", "nav_return_pct", "total_pnl", "win_rate", 
            "payoff_ratio", "max_drawdown", "recent_nav_return", "recent_trades",
            "pass_stage2", "reason"
        ])

    setup_id = str(stage1_df["setup_id"].iloc[0])
    rank = stage1_df.get("rank", pd.Series([None])).iloc[0]

    # Get base capital
    try:
        base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_cap = 100_000.0

    # Compute all metrics
    nav_metrics = _compute_nav_metrics(fold_ledger, base_cap)
    trade_metrics = _compute_trade_metrics(fold_ledger)
    recent_metrics = _compute_recent_metrics(fold_ledger, recent_days)
    
    # Apply checks (same logic as above)
    pass_nav = nav_metrics["nav_return_pct"] >= min_nav_return_pct
    pass_pnl = trade_metrics["total_pnl"] >= min_total_pnl
    pass_winrate = trade_metrics["win_rate"] >= min_win_rate
    pass_payoff = trade_metrics["payoff_ratio"] >= min_payoff_ratio
    pass_drawdown = nav_metrics["max_drawdown"] <= max_drawdown_pct
    pass_recent_nav = recent_metrics["recent_nav_return"] >= min_recent_nav_return
    pass_recent_trades = recent_metrics["recent_trades"] >= min_recent_trades
    
    passed = bool(pass_nav and pass_pnl and pass_drawdown and pass_recent_nav and pass_recent_trades and 
                  (pass_winrate or pass_payoff))
    
    # Build reason string
    reasons = []
    if not pass_nav:
        reasons.append(f"nav_return={nav_metrics['nav_return_pct']:.2f}%<{min_nav_return_pct}%")
    if not pass_pnl:
        reasons.append(f"total_pnl={trade_metrics['total_pnl']:.2f}<{min_total_pnl}")
    if not pass_winrate and not pass_payoff:
        reasons.append(f"win_rate={trade_metrics['win_rate']:.2f}<{min_win_rate} OR payoff={trade_metrics['payoff_ratio']:.2f}<{min_payoff_ratio}")
    if not pass_drawdown:
        reasons.append(f"max_dd={nav_metrics['max_drawdown']:.2f}>{max_drawdown_pct}")
    if not pass_recent_nav:
        reasons.append(f"recent_nav={recent_metrics['recent_nav_return']:.2f}%<{min_recent_nav_return}%")
    if not pass_recent_trades:
        reasons.append(f"recent_trades={recent_metrics['recent_trades']}<{min_recent_trades}")
    
    reason = "ok" if passed else ";".join(reasons) or "failed"

    return pd.DataFrame([{
        "setup_id": setup_id, "rank": rank,
        "nav_return_pct": nav_metrics["nav_return_pct"],
        "total_pnl": trade_metrics["total_pnl"],
        "win_rate": trade_metrics["win_rate"],
        "payoff_ratio": trade_metrics["payoff_ratio"],
        "max_drawdown": nav_metrics["max_drawdown"],
        "recent_nav_return": recent_metrics["recent_nav_return"],
        "recent_trades": recent_metrics["recent_trades"],
        "pass_stage2": passed,
        "reason": reason,
    }])
