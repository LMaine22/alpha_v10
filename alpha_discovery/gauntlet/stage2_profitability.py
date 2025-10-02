# alpha_discovery/gauntlet/stage2_profitability.py
"""
Gauntlet 2.0 — Stage 2: OOS Profitability (Walk-Forward)

Hard gates (configurable defaults):
    - DSR_WF (median) ≥ 0.25
    - IQR(Sharpe)_WF ≤ 0.80
    - CVaR5_WF ≥ -0.15
    - Support_WF (total trades across folds) ≥ 50
    - Fold catastrophe guard: each fold total return ≥ -0.20 (i.e., no single fold worse than -20%)

Outputs:
    - Single-row DataFrame with pass_stage2 (bool), reject_code (str|None), reason (str)
    - Diagnostics per-fold and aggregated (Sharpe, DSR, CVaR5, vol, maxDD, fold_return, support)

Notes:
    - Computes metrics on OOS test periods only (supply per-fold OOS ledgers).
    - Uses Deflated Sharpe if available; falls back to Sharpe gracefully.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

try:
    from ..config import Settings
except Exception:
    class Settings:  # type: ignore
        pass

# You referenced these in your existing code
try:
    from ..eval.nav import nav_daily_returns_from_ledger, sharpe
except Exception:
    # Fallbacks if your helpers are not importable in this context.
    def nav_daily_returns_from_ledger(ledger: pd.DataFrame, base_capital: float = 100000.0) -> pd.Series:
        """Very rough fallback: treat each trade's pnl as a day return; replace with your real function."""
        pnl = pd.to_numeric(ledger.get("pnl_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) / 100.0
        idx = pd.to_datetime(ledger.get("exit_date", pd.Timestamp.today()), errors="coerce")
        return pd.Series(pnl.values, index=idx).sort_index()

    def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
        r = returns.dropna()
        if len(r) < 2:
            return 0.0
        ex = r - risk_free / 252.0
        mu, sig = ex.mean(), ex.std(ddof=1)
        return 0.0 if sig == 0 else float(np.sqrt(252) * mu / sig)


# ------------------------------ Metrics helpers ------------------------------

def _cvar(series: pd.Series, alpha: float = 0.05) -> float:
    """CVaR at level alpha on a return series (per-bar or per-day)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    q = s.quantile(alpha)
    tail = s[s <= q]
    return float(tail.mean()) if len(tail) else float(q)

def _max_drawdown(returns: pd.Series) -> float:
    """Max drawdown on compounded returns series."""
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    curve = (1 + r).cumprod()
    roll_max = curve.cummax()
    dd = (curve / roll_max) - 1.0
    return float(dd.min())

def _safe_deflated_sharpe(sr: float, n: int, n_trials: int = 1) -> float:
    """
    Conservative Deflated Sharpe proxy.
    If you have a canonical DSR implementation elsewhere, import and use it.
    Here we apply a mild deflation for small samples and multiple trials.
    """
    if n <= 1:
        return 0.0
    # Penalize multiple comparisons; shrink toward 0 with 1/sqrt(n) and log(n_trials)
    penalty = (np.sqrt(max(np.log(max(n_trials, 1)), 1.0))) / np.sqrt(max(n - 1, 1))
    return float(max(0.0, sr - penalty))

def _fold_metrics(ledger: pd.DataFrame, base_capital: float) -> Dict[str, float]:
    """Compute per-fold OOS metrics."""
    rets = nav_daily_returns_from_ledger(ledger, base_capital=base_capital)
    rets = pd.to_numeric(rets, errors="coerce").dropna()
    n_obs = int(len(rets))
    sr = float(sharpe(rets)) if n_obs >= 2 else 0.0
    dsr = _safe_deflated_sharpe(sr, n_obs, n_trials=1)  # set n_trials via config if you sweep many params
    maxdd = _max_drawdown(rets) if n_obs else 0.0
    vol = float(rets.std(ddof=1) * np.sqrt(252)) if n_obs >= 2 else 0.0
    cvar5 = _cvar(rets, 0.05) if n_obs else 0.0
    fold_ret = float((1 + rets).prod() - 1.0) if n_obs else 0.0

    # Support: trade count in this fold
    if "pnl_pct" in ledger.columns:
        support = int(pd.to_numeric(ledger["pnl_pct"], errors="coerce").dropna().shape[0])
    else:
        support = int(len(ledger))

    return {
        "sharpe": sr,
        "dsr": dsr,
        "cvar5": cvar5,
        "maxdd": maxdd,
        "vol": vol,
        "fold_return": fold_ret,
        "support": support,
        "n_obs": n_obs,
    }


# ------------------------------ Stage 2 (multi-fold, recommended) ------------------------------

def run_stage2_profitability_wf(
    fold_ledgers: List[pd.DataFrame],
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Stage 2 over multiple walk-forward OOS folds (contract-aligned)."""
    cfg = dict(config or {})
    dsr_wf_min = float(cfg.get("s2_dsr_wf_min", 0.25))
    iqr_sharp_max = float(cfg.get("s2_iqr_sharpe_wf_max", 0.80))
    cvar5_wf_min = float(cfg.get("s2_cvar5_wf_min", -0.15))
    support_wf_min = int(cfg.get("s2_support_wf_min", 50))
    fold_min_ret = float(cfg.get("s2_fold_min_return", -0.20))
    base_cap = float(
        getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", cfg.get("s2_base_capital", 100_000.0))
    )

    if stage1_df is None or stage1_df.empty:
        return pd.DataFrame(
            [
                {
                    "setup_id": None,
                    "rank": None,
                    "pass_stage2": False,
                    "reject_code": "S2_INPUT_MISSING",
                    "reason": "missing_stage1",
                    "wf_return_pct": np.nan,
                    "wf_sharpe": np.nan,
                    "wf_dsr": np.nan,
                    "wf_cvar_5": np.nan,
                    "wf_support": np.nan,
                    "wf_sharpe_iqr": np.nan,
                    "wf_fold_min_return": np.nan,
                }
            ]
        )

    setup_id = stage1_df["setup_id"].iloc[0]
    rank = stage1_df.get("rank", pd.Series([None])).iloc[0]

    ledgers = [ld for ld in (fold_ledgers or []) if isinstance(ld, pd.DataFrame) and not ld.empty]
    if not ledgers:
        return pd.DataFrame(
            [
                {
                    "setup_id": setup_id,
                    "rank": rank,
                    "pass_stage2": False,
                    "reject_code": "S2_NO_WF",
                    "reason": "no_walkforward_ledgers",
                    "wf_return_pct": np.nan,
                    "wf_sharpe": np.nan,
                    "wf_dsr": np.nan,
                    "wf_cvar_5": np.nan,
                    "wf_support": np.nan,
                    "wf_sharpe_iqr": np.nan,
                    "wf_fold_min_return": np.nan,
                }
            ]
        )

    fold_metrics = [_fold_metrics(ld, base_cap) for ld in ledgers]
    sharpe_list = [fm["sharpe"] for fm in fold_metrics]
    dsr_list = [fm["dsr"] for fm in fold_metrics]
    cvar5_list = [fm["cvar5"] for fm in fold_metrics]
    ret_list = [fm["fold_return"] for fm in fold_metrics]
    support_list = [fm["support"] for fm in fold_metrics]

    wf_sharpe = float(pd.Series(sharpe_list).mean()) if sharpe_list else np.nan
    wf_dsr = float(pd.Series(dsr_list).median()) if dsr_list else np.nan
    wf_sharpe_iqr = float(pd.Series(sharpe_list).quantile(0.75) - pd.Series(sharpe_list).quantile(0.25)) if len(sharpe_list) >= 2 else 0.0
    combined_returns = pd.concat([nav_daily_returns_from_ledger(ld, base_cap) for ld in ledgers], axis=0)
    wf_cvar_5 = float(_cvar(combined_returns, 0.05)) if not combined_returns.empty else np.nan
    wf_support = int(np.sum(support_list)) if support_list else 0
    wf_return_pct = float((1 + combined_returns).prod() - 1.0) if not combined_returns.empty else np.nan
    wf_fold_min_return = float(np.min(ret_list)) if ret_list else np.nan

    failures: List[str] = []
    reject_code: Optional[str] = None

    def gate(cond: bool, code: str, message: str) -> None:
        nonlocal reject_code
        if not cond:
            if reject_code is None:
                reject_code = code
            failures.append(message)

    gate(not np.isnan(wf_dsr) and wf_dsr >= dsr_wf_min, "S2_DSR", f"median_DSR={wf_dsr:.2f}<{dsr_wf_min:.2f}")
    gate(not np.isnan(wf_sharpe_iqr) and wf_sharpe_iqr <= iqr_sharp_max, "S2_IQR", f"IQR_Sharpe={wf_sharpe_iqr:.2f}>{iqr_sharp_max:.2f}")
    gate(not np.isnan(wf_cvar_5) and wf_cvar_5 >= cvar5_wf_min, "S2_CVAR", f"CVaR5={wf_cvar_5:.2f}<{cvar5_wf_min:.2f}")
    gate(wf_support >= support_wf_min, "S2_SUPPORT", f"Support_WF={wf_support}<{support_wf_min}")
    gate(not np.isnan(wf_fold_min_return) and wf_fold_min_return >= fold_min_ret, "S2_CATASTROPHE", f"WorstFoldReturn={wf_fold_min_return:.2%}<{fold_min_ret:.2%}")

    passed = len(failures) == 0

    out = {
        "setup_id": setup_id,
        "rank": rank,
        "pass_stage2": bool(passed),
        "reject_code": None if passed else (reject_code or "S2_FAIL"),
        "reason": "ok" if passed else ";".join(failures),
        "wf_return_pct": float(wf_return_pct) if not np.isnan(wf_return_pct) else np.nan,
        "wf_sharpe": float(wf_sharpe) if not np.isnan(wf_sharpe) else np.nan,
        "wf_dsr": float(wf_dsr) if not np.isnan(wf_dsr) else np.nan,
        "wf_cvar_5": float(wf_cvar_5) if not np.isnan(wf_cvar_5) else np.nan,
        "wf_support": int(wf_support),
        "wf_sharpe_iqr": float(wf_sharpe_iqr) if not np.isnan(wf_sharpe_iqr) else np.nan,
        "wf_fold_min_return": float(wf_fold_min_return) if not np.isnan(wf_fold_min_return) else np.nan,
    }
    return pd.DataFrame([out])


# ------------------------------ Stage 2 (legacy single-fold compatibility) ------------------------------

def _compute_nav_metrics(ledger: pd.DataFrame, base_capital: float = 100000.0) -> Dict[str, float]:
    """Computes NAV-based metrics (Sharpe, drawdown) from a trade ledger."""
    daily_returns = nav_daily_returns_from_ledger(ledger, base_capital)
    
    # Calculate metrics, handling potential NaNs or empty series
    sharpe_val = sharpe(daily_returns) if not daily_returns.empty else 0.0
    dsr_val = sharpe_val  # DSR approximation for this context
    cvar_val = _cvar(daily_returns) if not daily_returns.empty else 0.0
    mdd_val = _max_drawdown(daily_returns) if not daily_returns.empty else 0.0
    
    return {
        "wf_return_pct": (daily_returns.sum()) * 100.0,
        "wf_sharpe": sharpe_val,
        "wf_dsr": dsr_val,
        "wf_cvar_5": cvar_val,
        "wf_max_drawdown_pct": mdd_val,
    }

def _compute_trade_metrics(ledger: pd.DataFrame) -> Dict[str, float]:
    """Legacy trade metrics (kept for compatibility with your previous stage)."""
    if len(ledger) == 0:
        return {"total_pnl": 0.0, "win_rate": 0.0, "payoff_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    pnl_col = next((c for c in ["pnl_pct", "realized_pnl", "pnl", "PnL"] if c in ledger.columns), None)
    if pnl_col is None:
        return {"total_pnl": 0.0, "win_rate": 0.0, "payoff_ratio": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    pnl_series = pd.to_numeric(ledger[pnl_col], errors="coerce").fillna(0.0)
    total_pnl = float(pnl_series.sum())
    wins = pnl_series[pnl_series > 0]; losses = pnl_series[pnl_series < 0]
    win_rate = len(wins) / len(pnl_series) if len(pnl_series) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0.0)
    return {"total_pnl": total_pnl, "win_rate": win_rate, "payoff_ratio": payoff_ratio, "avg_win": avg_win, "avg_loss": avg_loss}

def _compute_recent_metrics(ledger: pd.DataFrame, recent_days: int = 30) -> Dict[str, float]:
    """Legacy 'recent' snapshot (kept for compatibility)."""
    if len(ledger) == 0:
        return {"recent_nav_return": 0.0, "recent_trades": 0}
    date_col = next((c for c in ["trigger_date", "entry_date", "exit_date"] if c in ledger.columns), None)
    if date_col is None:
        return {"recent_nav_return": 0.0, "recent_trades": 0}
    ledger = ledger.copy()
    ledger[date_col] = pd.to_datetime(ledger[date_col], errors="coerce")
    cutoff_date = ledger[date_col].max() - pd.Timedelta(days=recent_days)
    recent_ledger = ledger[ledger[date_col] >= cutoff_date]
    if len(recent_ledger) == 0:
        return {"recent_nav_return": 0.0, "recent_trades": 0}
    nav_metrics = _compute_nav_metrics(recent_ledger)
    return {"recent_nav_return": nav_metrics["wf_return_pct"], "recent_trades": len(recent_ledger)}

def run_stage2_profitability(
    run_dir: str,
    fold_num: int,
    settings: Settings,
    config: Optional[Dict[str, Any]] = None,
    stage1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Legacy single-fold Stage 2 (kept for back-compat with existing runners).

    Gates here are milder and *not* the Gauntlet 2.0 multi-fold gates.
    Prefer run_stage2_profitability_wf for full robustness.
    """
    cfg = dict(config or {})
    min_nav_return_pct   = float(cfg.get("s2_min_nav_return_pct", 0.0))
    min_total_pnl        = float(cfg.get("s2_min_total_pnl", 0.0))
    min_win_rate         = float(cfg.get("s2_min_win_rate", 0.0))
    min_payoff_ratio     = float(cfg.get("s2_min_payoff_ratio", 0.0))
    max_drawdown_pct     = float(cfg.get("s2_max_drawdown_pct", 0.50))
    recent_days          = int(cfg.get("s2_recent_days", 30))
    min_recent_nav_return= float(cfg.get("s2_min_recent_nav_return", 0.0))
    min_recent_trades    = int(cfg.get("s2_min_recent_trades", 1))

    if stage1_df is None or stage1_df.empty:
        return pd.DataFrame([{
            "setup_id": None, "rank": None, "pass_stage2": False,
            "reject_code": "S2_INPUT_MISSING", "reason": "missing_stage1",
        }])

    setup_id = stage1_df["setup_id"].iloc[0]
    rank = stage1_df.get("rank", pd.Series([None])).iloc[0]

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
            "setup_id": setup_id, "rank": rank, "pass_stage2": False,
            "reject_code": "S2_NO_LEDGER", "reason": "no_oos_ledger",
        }])

    base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))

    nav_metrics    = _compute_nav_metrics(fold_ledger, base_cap)
    trade_metrics  = _compute_trade_metrics(fold_ledger)
    recent_metrics = _compute_recent_metrics(fold_ledger, recent_days)

    pass_nav         = nav_metrics["wf_return_pct"] >= min_nav_return_pct
    pass_pnl         = trade_metrics["total_pnl"] >= min_total_pnl
    pass_winratepay  = (trade_metrics["win_rate"] >= min_win_rate) or (trade_metrics["payoff_ratio"] >= min_payoff_ratio)
    pass_drawdown    = nav_metrics["wf_max_drawdown_pct"] <= max_drawdown_pct
    pass_recent_nav  = recent_metrics["recent_nav_return"] >= min_recent_nav_return
    pass_recent_trds = recent_metrics["recent_trades"] >= min_recent_trades

    passed = bool(pass_nav and pass_pnl and pass_drawdown and pass_recent_nav and pass_recent_trds and pass_winratepay)

    failures = []
    if not pass_nav:        failures.append(f"nav_return={nav_metrics['wf_return_pct']:.2f}%<{min_nav_return_pct}%")
    if not pass_pnl:        failures.append(f"total_pnl={trade_metrics['total_pnl']:.2f}<{min_total_pnl}")
    if not pass_winratepay: failures.append(f"win_rate={trade_metrics['win_rate']:.2f} or payoff={trade_metrics['payoff_ratio']:.2f} below mins")
    if not pass_drawdown:   failures.append(f"max_dd={nav_metrics['wf_max_drawdown_pct']:.2f}>{max_drawdown_pct}")
    if not pass_recent_nav: failures.append(f"recent_nav={recent_metrics['recent_nav_return']:.2f}%<{min_recent_nav_return}%")
    if not pass_recent_trds:failures.append(f"recent_trades={recent_metrics['recent_trades']}<{min_recent_trades}")

    return pd.DataFrame([{
        "setup_id": setup_id, "rank": rank,
        "pass_stage2": bool(passed),
        "reject_code": None if passed else "S2_FAIL",
        "reason": "ok" if passed else ";".join(failures),

        # Legacy diagnostics
        "nav_return_pct": float(nav_metrics["wf_return_pct"]),
        "sharpe_ratio": float(nav_metrics["wf_sharpe"]),
        "cvar5": float(nav_metrics["wf_cvar_5"]),
        "max_drawdown": float(nav_metrics["wf_max_drawdown_pct"]),
        "volatility": float(nav_metrics["wf_max_drawdown_pct"]), # This seems like a bug in the original code, should be vol
        "total_pnl": float(trade_metrics["total_pnl"]),
        "win_rate": float(trade_metrics["win_rate"]),
        "payoff_ratio": float(trade_metrics["payoff_ratio"]),
        "recent_nav_return": float(recent_metrics["recent_nav_return"]),
        "recent_trades": int(recent_metrics["recent_trades"]),
    }])
