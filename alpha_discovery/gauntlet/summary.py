# alpha_discovery/gauntlet/summary.py
from __future__ import annotations

import os
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from ..config import settings
from .io import read_global_artifacts
from .reporting import write_csv


def write_gauntlet_outputs(
        run_dir: str,
        stage_results: List[Dict[str, Any]],
        full_ledger: pd.DataFrame,
        setup_ledgers: Dict[str, pd.DataFrame],
        settings: Any,
        diagnostics: bool = False,
) -> Dict[str, str]:
    """
    STREAMLINED GAUNTLET OUTPUTS: Generate only the essential files for trading decisions.

    This function creates the consolidated 3-4 files that traders actually need:
    1. gauntlet_results.csv - Pass/fail decisions with key metrics
    2. open_positions.csv - Current open trades requiring attention
    3. gauntlet_ledger.csv - Complete trade history for analysis
    4. stage_diagnostics.csv - Detailed metrics (optional, if diagnostics=True)

    The key improvement is properly detecting open positions where exit_date is NaT.
    """
    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    print("[Output Generation] Creating consolidated gauntlet files...")

    # 1. Main results file - the decision matrix
    results_path = _write_gauntlet_results(gaunt_dir, stage_results, full_ledger, setup_ledgers, settings)

    # 2. Open positions file - current risk exposure (FIXED)
    positions_path = _write_open_positions(gaunt_dir, full_ledger, stage_results)

    # 3. Full ledger - complete audit trail
    ledger_path = _write_gauntlet_ledger(gaunt_dir, full_ledger)

    output_paths = {
        "gauntlet_results": results_path,
        "open_positions": positions_path,
        "gauntlet_ledger": ledger_path
    }

    # 4. Optional detailed diagnostics
    if diagnostics:
        diag_path = _write_stage_diagnostics(gaunt_dir, stage_results, setup_ledgers, settings)
        output_paths["stage_diagnostics"] = diag_path

    print(f"[Output Generation] Successfully generated {len(output_paths)} files")
    return output_paths


def _write_gauntlet_results(
        gaunt_dir: str,
        stage_results: List[Dict[str, Any]],
        full_ledger: pd.DataFrame,
        setup_ledgers: Dict[str, pd.DataFrame],
        settings: Any,
) -> str:
    """
    TRADING DECISION MATRIX: One row per setup with pass/fail status and key metrics.

    This is the primary file traders use to decide what to deploy, monitor, or retire.
    """

    if not stage_results:
        empty_df = pd.DataFrame(columns=[
            "setup_id", "specialized_ticker", "direction", "signal_ids", "description",
            "stage1_pass", "stage1_reason", "stage2_pass", "stage2_reason",
            "stage3_pass", "stage3_reason", "final_decision",
            "total_trades", "open_trades", "days_since_last",
            "total_pnl", "nav_return_pct", "sharpe_ratio", "win_rate", "max_drawdown"
        ])
        path = os.path.join(gaunt_dir, "gauntlet_results.csv")
        return write_csv(path, empty_df)

    # Build results DataFrame with trading metrics
    results_data = []

    for result in stage_results:
        setup_id = result["setup_id"]
        ledger = setup_ledgers.get(setup_id, pd.DataFrame())

        # Stage results
        row = {
            "setup_id": setup_id,
            "stage1_pass": result["stage1_pass"],
            "stage1_reason": result["stage1_reason"],
            "stage2_pass": result["stage2_pass"],
            "stage2_reason": result["stage2_reason"],
            "stage3_pass": result["stage3_pass"],
            "stage3_reason": result["stage3_reason"],
            "final_decision": result["final_decision"]
        }

        # Add setup metadata
        if not ledger.empty:
            row["specialized_ticker"] = ledger.get("ticker", pd.Series(["Unknown"])).iloc[0]
            row["direction"] = ledger.get("direction", pd.Series(["long"])).iloc[0]
            row["signal_ids"] = ledger.get("signal_ids", pd.Series([""])).iloc[0]
        else:
            row["specialized_ticker"] = "Unknown"
            row["direction"] = "long"
            row["signal_ids"] = ""

        # Calculate trading metrics
        metrics = _compute_setup_metrics(ledger, settings)
        row.update(metrics)

        results_data.append(row)

    results_df = pd.DataFrame(results_data)

    # Add descriptions from Pareto front if available
    try:
        pareto_df, _ = read_global_artifacts(os.path.dirname(gaunt_dir))
        if not pareto_df.empty and "description" in pareto_df.columns:
            pareto_desc = pareto_df[["setup_id", "description"]].drop_duplicates()
            results_df = results_df.merge(pareto_desc, on="setup_id", how="left")
    except Exception:
        pass

    if "description" not in results_df.columns:
        results_df["description"] = "Setup: " + results_df["signal_ids"].astype(str)

    # Sort by trading priority: Deploy first, then by performance
    decision_priority = {"Deploy": 1, "Monitor": 2, "Retire": 3}
    results_df["_sort_priority"] = results_df["final_decision"].map(decision_priority)
    results_df = results_df.sort_values([
        "_sort_priority", "total_pnl", "days_since_last"
    ], ascending=[True, False, True])
    results_df = results_df.drop(columns=["_sort_priority"])

    path = os.path.join(gaunt_dir, "gauntlet_results.csv")
    return write_csv(path, results_df)


def _write_open_positions(gaunt_dir: str, full_ledger: pd.DataFrame, stage_results: List[Dict[str, Any]]) -> str:
    """
    ENHANCED OPEN POSITIONS TRACKER: Current trades with setup performance context.

    This now includes both position-specific data (days open, unrealized P&L) AND
    setup-level performance metrics (win rate, total return, max drawdown) to help
    traders quickly identify which open positions represent the best opportunities.
    """

    if full_ledger.empty:
        empty_df = pd.DataFrame(columns=[
            "setup_id", "specialized_ticker", "direction", "entry_date",
            "days_open", "unrealized_pnl", "contracts", "strike", "option_type",
            # NEW: Setup performance metrics for trading decisions
            "total_pnl", "nav_return_pct", "win_rate", "max_drawdown"
        ])
        path = os.path.join(gaunt_dir, "open_positions.csv")
        return write_csv(path, empty_df)

    ledger = full_ledger.copy()

    # Normalize date columns
    for col in ["trigger_date", "exit_date", "entry_date"]:
        if col in ledger.columns:
            ledger[col] = pd.to_datetime(ledger[col], errors="coerce")

    # CRITICAL FIX: Only show open positions from setups that passed all gauntlet stages
    if stage_results:
        # Get setup IDs that passed all stages (final_decision is "Deploy" or "Monitor")
        passed_setups = set()
        for result in stage_results:
            if result.get("final_decision") in ["Deploy", "Monitor"]:
                passed_setups.add(result["setup_id"])
        
        print(f"[Open Positions] Filtering to {len(passed_setups)} setups that passed all stages")
        
        # Filter ledger to only include trades from passed setups
        if "setup_id" in ledger.columns:
            ledger = ledger[ledger["setup_id"].isin(passed_setups)]
        else:
            print("[Open Positions] Warning: No setup_id column found, cannot filter by passed setups")
            ledger = pd.DataFrame()  # Empty if we can't filter

    # CRITICAL FIX: Properly identify open trades (exit_date is NaT)
    if "exit_date" in ledger.columns:
        is_open = ledger["exit_date"].isna()  # This catches pd.NaT values
    else:
        # Fallback: if no exit_date column, check if trade was triggered recently
        if "trigger_date" in ledger.columns:
            recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            is_open = ledger["trigger_date"] >= recent_cutoff
        else:
            is_open = pd.Series([False] * len(ledger), index=ledger.index)

    open_trades = ledger[is_open].copy()

    if open_trades.empty:
        print("[Open Positions] No open positions found from setups that passed all stages")
        empty_df = pd.DataFrame(columns=[
            "setup_id", "specialized_ticker", "direction", "entry_date",
            "days_open", "unrealized_pnl", "contracts", "strike", "option_type",
            "total_pnl", "nav_return_pct", "win_rate", "max_drawdown"
        ])
        path = os.path.join(gaunt_dir, "open_positions.csv")
        return write_csv(path, empty_df)

    print(f"[Open Positions] Found {len(open_trades)} open positions from setups that passed all stages")

    # Calculate position-specific metrics (same as before)
    entry_col = "trigger_date" if "trigger_date" in open_trades.columns else "entry_date"
    if entry_col in open_trades.columns:
        today = pd.Timestamp.now().normalize()
        open_trades["days_open"] = (today - open_trades[entry_col].dt.normalize()).dt.days
    else:
        open_trades["days_open"] = 0

    # Use the more descriptive ticker column name if available
    if "ticker" in open_trades.columns and "specialized_ticker" not in open_trades.columns:
        open_trades["specialized_ticker"] = open_trades["ticker"]

    # NEW: Compute setup-level performance metrics for setups with open positions
    unique_setups = open_trades["setup_id"].unique()
    setup_metrics = {}

    for setup_id in unique_setups:
        # Get all trades for this setup (both open and closed)
        setup_ledger = ledger[ledger["setup_id"] == setup_id]

        # Compute the same metrics used in gauntlet_results
        from ..config import settings as default_settings
        metrics = _compute_setup_metrics(setup_ledger, default_settings)

        setup_metrics[setup_id] = {
            "total_pnl": metrics["total_pnl"],
            "nav_return_pct": metrics["nav_return_pct"],
            "win_rate": metrics["win_rate"],
            "max_drawdown": metrics["max_drawdown"]
        }

    # Convert setup metrics to DataFrame for merging
    setup_metrics_df = pd.DataFrame.from_dict(setup_metrics, orient="index")
    setup_metrics_df.index.name = "setup_id"
    setup_metrics_df = setup_metrics_df.reset_index()

    # Merge setup performance metrics with open positions
    enhanced_positions = open_trades.merge(
        setup_metrics_df,
        on="setup_id",
        how="left"
    )

    # Select and organize columns for maximum trading value
    # Position-specific columns first, then setup performance context
    position_cols = [
        # Core position identification
        "setup_id", "specialized_ticker", "direction",

        # Position timing and status
        "entry_date", "days_open",

        # Position economics
        "unrealized_pnl", "contracts", "strike", "option_type",

        # Setup performance context (NEW - for trading decisions)
        "total_pnl", "nav_return_pct", "win_rate", "max_drawdown",

        # Additional position details
        "entry_underlying", "entry_iv", "entry_option_price"
    ]

    available_cols = [col for col in position_cols if col in enhanced_positions.columns]
    positions_df = enhanced_positions[available_cols].copy()

    # Rename trigger_date to entry_date for clarity if needed
    if "trigger_date" in positions_df.columns and "entry_date" not in available_cols:
        positions_df = positions_df.rename(columns={"trigger_date": "entry_date"})

    # Smart sorting: Best setups with longest-held positions first
    # This prioritizes positions that are both high-quality setups AND need attention
    sort_columns = []
    if "nav_return_pct" in positions_df.columns:
        sort_columns.append(("nav_return_pct", False))  # Best performing setups first
    if "days_open" in positions_df.columns:
        sort_columns.append(("days_open", False))  # Longest held positions first
    if "unrealized_pnl" in positions_df.columns:
        sort_columns.append(("unrealized_pnl", False))  # Most profitable positions first

    if sort_columns:
        sort_cols = [col for col, _ in sort_columns]
        sort_ascending = [asc for _, asc in sort_columns]
        positions_df = positions_df.sort_values(sort_cols, ascending=sort_ascending)

    # Data quality validation
    _validate_open_positions_data(positions_df)

    path = os.path.join(gaunt_dir, "open_positions.csv")
    return write_csv(path, positions_df)


def _write_gauntlet_ledger(gaunt_dir: str, full_ledger: pd.DataFrame) -> str:
    """
    COMPLETE TRADE HISTORY: Full audit trail with data quality validation.

    This includes both open and closed positions, with additional validation
    to catch and flag any data quality issues. Now applies consistent schema
    ordering to ensure predictable column structure across runs.
    """

    if not full_ledger.empty:
        # Apply data quality validation (same as before)
        validated_ledger = _validate_ledger_data_quality(full_ledger)

        # NEW: Apply consistent schema for the ledger file
        from .reporting import GAUNTLET_LEDGER_SCHEMA
        path = os.path.join(gaunt_dir, "gauntlet_ledger.csv")
        return write_csv(path, validated_ledger, apply_schema=GAUNTLET_LEDGER_SCHEMA)
    else:
        # For empty ledgers, still apply schema to get consistent headers
        from .reporting import GAUNTLET_LEDGER_SCHEMA
        empty_ledger = pd.DataFrame(columns=GAUNTLET_LEDGER_SCHEMA)
        path = os.path.join(gaunt_dir, "gauntlet_ledger.csv")
        return write_csv(path, empty_ledger, apply_schema=GAUNTLET_LEDGER_SCHEMA)


def _write_stage_diagnostics(
        gaunt_dir: str,
        stage_results: List[Dict[str, Any]],
        setup_ledgers: Dict[str, pd.DataFrame],
        settings: Any,
) -> str:
    """DETAILED DIAGNOSTICS: Stage-by-stage metrics for deep analysis (optional)."""

    diag_data = []

    for result in stage_results:
        setup_id = result["setup_id"]
        ledger = setup_ledgers.get(setup_id, pd.DataFrame())

        # Compute detailed metrics
        detailed_metrics = _compute_detailed_metrics(ledger, settings)

        row = {
            "setup_id": setup_id,
            "final_decision": result["final_decision"],
            **result,  # All stage results
            **detailed_metrics  # Detailed metrics
        }

        diag_data.append(row)

    diag_df = pd.DataFrame(diag_data)
    path = os.path.join(gaunt_dir, "stage_diagnostics.csv")
    return write_csv(path, diag_df)


def _compute_setup_metrics(ledger: pd.DataFrame, settings: Any) -> Dict[str, Any]:
    """
    CORE TRADING METRICS: Essential performance indicators for each setup.

    These metrics help traders quickly assess setup performance and risk.
    """

    if ledger.empty:
        return {
            "total_trades": 0, "open_trades": 0, "days_since_last": 999,
            "total_pnl": 0.0, "nav_return_pct": 0.0, "sharpe_ratio": 0.0,
            "win_rate": 0.0, "max_drawdown": 0.0
        }

    # Basic trade counts
    total_trades = len(ledger)

    # FIXED: Properly count open trades (exit_date is NaT)
    if "exit_date" in ledger.columns:
        open_trades = int(ledger["exit_date"].isna().sum())
    else:
        open_trades = 0

    # Days since last activity
    date_cols = ["trigger_date", "entry_date"]
    last_date = None
    for col in date_cols:
        if col in ledger.columns:
            dates = pd.to_datetime(ledger[col], errors="coerce")
            max_date = dates.max()
            if pd.notnull(max_date):
                last_date = max_date
                break

    if last_date:
        days_since_last = (pd.Timestamp.now() - last_date).days
    else:
        days_since_last = 999

    # P&L analysis
    pnl_col = None
    for col in ["pnl_dollars", "realized_pnl", "pnl"]:
        if col in ledger.columns:
            pnl_col = col
            break

    if pnl_col:
        pnl_series = pd.to_numeric(ledger[pnl_col], errors="coerce").fillna(0.0)

        # Include unrealized P&L for open positions in total
        if "unrealized_pnl" in ledger.columns:
            unrealized_series = pd.to_numeric(ledger["unrealized_pnl"], errors="coerce").fillna(0.0)
            total_pnl = float(pnl_series.sum() + unrealized_series.sum())
        else:
            total_pnl = float(pnl_series.sum())

        # Win rate (only on closed positions)
        closed_pnl = pnl_series[~ledger["exit_date"].isna()] if "exit_date" in ledger.columns else pnl_series
        wins = (closed_pnl > 0).sum()
        total_closed = len(closed_pnl)
        win_rate = float(wins / total_closed) if total_closed > 0 else 0.0
    else:
        total_pnl = 0.0
        win_rate = 0.0

    # NAV return percentage
    try:
        base_capital = float(getattr(settings.reporting, "base_capital_for_portfolio", 100000.0))
        nav_return_pct = (total_pnl / base_capital) * 100.0
    except Exception:
        nav_return_pct = 0.0

    # Simplified risk metrics (avoiding complex dependencies)
    sharpe_ratio = 0.0
    max_drawdown = 0.0

    if pnl_col and len(ledger) > 5:
        try:
            # Simple approximation for Sharpe ratio
            returns = pnl_series / base_capital if 'base_capital' in locals() else pnl_series / 100000
            if returns.std() > 0:
                sharpe_ratio = float((returns.mean() / returns.std()) * np.sqrt(252))

            # Simple drawdown from cumulative P&L
            cum_pnl = pnl_series.cumsum()
            running_max = cum_pnl.cummax()
            drawdown_series = (cum_pnl - running_max) / (running_max + base_capital)
            max_drawdown = float(abs(drawdown_series.min())) if len(drawdown_series) > 0 else 0.0
        except Exception:
            pass

    return {
        "total_trades": int(total_trades),
        "open_trades": int(open_trades),
        "days_since_last": int(days_since_last),
        "total_pnl": float(total_pnl),
        "nav_return_pct": float(nav_return_pct),
        "sharpe_ratio": float(sharpe_ratio),
        "win_rate": float(win_rate),
        "max_drawdown": float(max_drawdown)
    }


def _compute_detailed_metrics(ledger: pd.DataFrame, settings: Any) -> Dict[str, Any]:
    """EXTENDED METRICS: Additional analysis for diagnostic purposes."""

    basic_metrics = _compute_setup_metrics(ledger, settings)

    detailed = {
        "ledger_rows": len(ledger),
        "date_range_days": 0,
        "avg_holding_days": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "regime_exits_count": 0,
    }

    if not ledger.empty:
        # Date range analysis
        date_cols = ["trigger_date", "entry_date"]
        dates = []
        for col in date_cols:
            if col in ledger.columns:
                col_dates = pd.to_datetime(ledger[col], errors="coerce").dropna()
                dates.extend(col_dates.tolist())

        if dates:
            date_range_days = (max(dates) - min(dates)).days
            detailed["date_range_days"] = int(date_range_days)

        # Holding period analysis
        if "holding_days_actual" in ledger.columns:
            holding_days = pd.to_numeric(ledger["holding_days_actual"], errors="coerce")
            detailed["avg_holding_days"] = float(holding_days.mean()) if holding_days.notna().any() else 0.0

        # Win/loss extremes
        pnl_col = None
        for col in ["pnl_dollars", "realized_pnl", "pnl"]:
            if col in ledger.columns:
                pnl_col = col
                break

        if pnl_col:
            pnl_series = pd.to_numeric(ledger[pnl_col], errors="coerce").fillna(0.0)
            detailed["largest_win"] = float(pnl_series.max())
            detailed["largest_loss"] = float(pnl_series.min())

        # Regime-aware exit analysis
        if "exit_reason" in ledger.columns:
            regime_exits = ledger["exit_reason"].str.contains(
                "pt_hit_legA|volatility_spike_profit|time_decay_protection|stop_loss|atr_trail_hit",
                case=False, na=False
            ).sum()
            detailed["regime_exits_count"] = int(regime_exits)

    return {**basic_metrics, **detailed}


def _validate_open_positions_data(positions_df: pd.DataFrame) -> None:
    """
    DATA QUALITY VALIDATION: Check open positions for potential issues.

    This helps catch problems like positions that should have expired or
    unrealistic holding periods.
    """
    if positions_df.empty:
        return

    issues = []

    # Check for unrealistic holding periods
    if "days_open" in positions_df.columns:
        very_old = positions_df["days_open"] > 90  # More than 90 days
        if very_old.any():
            old_count = very_old.sum()
            issues.append(f"{old_count} positions open >90 days (check for expiration)")

    # Check for missing critical data
    critical_cols = ["setup_id", "specialized_ticker", "direction"]
    for col in critical_cols:
        if col in positions_df.columns:
            missing = positions_df[col].isna().sum()
            if missing > 0:
                issues.append(f"{missing} positions missing {col}")

    if issues:
        print(f"[Open Positions Validation] Issues found: {'; '.join(issues)}")


def _validate_ledger_data_quality(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    COMPREHENSIVE DATA VALIDATION: Check the full ledger for quality issues.

    This implements additional safeguards against the data quality problems
    identified in Fix #2.
    """
    validated = ledger.copy()
    original_count = len(validated)

    # Flag phantom same-day trades
    if all(col in validated.columns for col in ["trigger_date", "exit_date", "entry_underlying", "exit_underlying"]):
        same_day = (
                           validated["trigger_date"].dt.normalize() == validated["exit_date"].dt.normalize()
                   ) & validated["exit_date"].notna()

        identical_underlying = (
                abs(validated["entry_underlying"] - validated["exit_underlying"]) < 1e-6
        )

        phantom_trades = same_day & identical_underlying
        if phantom_trades.any():
            print(f"[Validation] Warning: {phantom_trades.sum()} phantom same-day trades detected")
            # Add a flag column instead of removing (for transparency)
            validated["data_quality_flag"] = phantom_trades.map({True: "phantom_same_day", False: ""})

    # Flag suspicious entry/exit dates
    if "trigger_date" in validated.columns and "exit_date" in validated.columns:
        # Flag trades where exit is before trigger (impossible)
        invalid_sequence = (
                (validated["exit_date"] < validated["trigger_date"]) &
                validated["exit_date"].notna()
        )
        if invalid_sequence.any():
            print(f"[Validation] Warning: {invalid_sequence.sum()} trades with exit before trigger")

    print(f"[Validation] Ledger validation complete: {len(validated)} trades processed")

    return validated