# alpha_discovery/gauntlet/run.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import Settings, gauntlet_cfg
from .io import find_latest_run_dir, read_global_artifacts, read_oos_artifacts
from .backtester import run_gauntlet_backtest
from .stage1_health import run_stage1_health_check
from .stage2_profitability import run_stage2_profitability_on_ledger
from .stage3_robustness import run_stage3_robustness_on_ledger
from .summary import write_gauntlet_outputs


def run_gauntlet(
        run_dir: Optional[str] = None,
        settings: Optional[Settings] = None,
        config: Optional[Dict[str, Any]] = None,
        master_df: Optional[pd.DataFrame] = None,
        signals_df: Optional[pd.DataFrame] = None,
        signals_metadata: Optional[List[Dict[str, Any]]] = None,
        splits: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None,
        mode: str = "legacy",  # "legacy" or "strict_oos"
        diagnostics: bool = False,
) -> Dict[str, str]:
    """
    UNIFIED GAUNTLET ENTRY POINT

    This function implements the streamlined gauntlet with only regime-aware exits
    and proper handling of open positions. It supports two modes:

    Legacy Mode: Runs fresh backtests from train-end dates to TODAY (September 15, 2025)
                 - Best for live trading decisions and current actionable signals

    Strict OOS Mode: Uses existing OOS artifacts within historical test windows
                     - Best for research validation and historical performance analysis

    Returns dict with paths to core output files:
    - gauntlet_results.csv: Pass/fail decisions for all setups
    - open_positions.csv: Current open trades only
    - gauntlet_ledger.csv: Full trade history
    - stage_diagnostics.csv: Detailed breakdowns (if diagnostics=True)
    """
    # Setup configuration
    from ..config import settings as default_settings
    actual_settings = settings or default_settings
    cfg = gauntlet_cfg(actual_settings)
    if config:
        cfg.update(config)

    if run_dir is None:
        run_dir = find_latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("Could not resolve a valid run_dir for Gauntlet.")

    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    print(f"[Gauntlet] Running in {mode} mode with diagnostics={diagnostics}")
    print(f"[Gauntlet] Target data end date: September 15, 2025")

    if mode == "legacy":
        return _run_legacy_mode(run_dir, actual_settings, cfg, master_df, signals_df,
                                signals_metadata, splits, diagnostics)
    else:
        return _run_strict_oos_mode(run_dir, actual_settings, cfg, splits, diagnostics)


def _run_legacy_mode(
        run_dir: str,
        settings: Settings,
        config: Dict[str, Any],
        master_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        signals_metadata: List[Dict[str, Any]],
        splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
        diagnostics: bool,
) -> Dict[str, str]:
    """
    LEGACY MODE: Fresh backtests from train-end dates to TODAY

    This mode provides the most current and actionable trading signals by:
    1. Taking Pareto candidates from each fold's training period
    2. Running fresh backtests from train-end date to September 15, 2025
    3. Using regime-aware exits to catch big runners while managing risk
    4. Properly handling open positions that haven't hit exit conditions
    """

    if not all([master_df is not None, signals_df is not None,
                signals_metadata is not None, splits is not None]):
        raise ValueError("Legacy mode requires master_df, signals_df, signals_metadata, and splits")

    print("[Legacy Mode] Loading Pareto candidates from training periods...")

    # Get candidates from Pareto front
    pareto_summary, _ = read_global_artifacts(run_dir)
    if pareto_summary is None or pareto_summary.empty:
        print("No Pareto candidates found")
        return _write_empty_outputs(run_dir)

    # Map each fold to its train end date
    fold_date_map = {i + 1: train_idx.max() for i, (train_idx, _) in enumerate(splits)}
    candidates = pareto_summary.copy()
    candidates["train_end_date"] = candidates["fold"].map(fold_date_map)

    # Clean up candidates and ensure required columns exist
    required_cols = ["specialized_ticker", "direction", "signal_ids", "train_end_date"]
    candidates = candidates.dropna(subset=required_cols)

    print(f"Found {len(candidates)} candidates from Pareto front across {len(fold_date_map)} folds")

    # Run fresh OOS backtests from train-end to TODAY
    print("[Legacy Mode] Running fresh backtests to current date...")
    backtest_results = _run_fresh_backtests(
        candidates, master_df, signals_df, signals_metadata, settings
    )

    if not backtest_results:
        print("No successful backtests generated")
        return _write_empty_outputs(run_dir)

    # Combine and deduplicate all trades
    print("[Legacy Mode] Combining and deduplicating trade results...")
    full_ledger = _combine_and_deduplicate_trades(backtest_results)

    print(f"Generated {len(full_ledger)} total trades after deduplication")
    print(f"Open positions: {full_ledger['exit_date'].isna().sum()}")

    # Run stages and generate outputs
    return _run_stages_and_output(run_dir, backtest_results, full_ledger, settings, config, diagnostics)


def _run_strict_oos_mode(
        run_dir: str,
        settings: Settings,
        config: Dict[str, Any],
        splits: Optional[List],
        diagnostics: bool,
) -> Dict[str, str]:
    """
    STRICT OOS MODE: Use existing OOS artifacts within historical test windows

    This mode provides research-quality validation by using pre-computed
    OOS results that respect the original cross-validation boundaries.
    """

    print("[Strict OOS Mode] Loading existing OOS artifacts...")

    try:
        # Load existing OOS data
        oos_summary, oos_ledger = read_oos_artifacts(run_dir)
        if oos_ledger is None or oos_ledger.empty:
            print("No OOS ledger data found")
            return _write_empty_outputs(run_dir)

        print(f"Loaded {len(oos_ledger)} OOS trades from existing artifacts")

        # Group by setup for stage processing
        setup_groups = {}
        setup_col = _find_setup_col(oos_ledger)
        for setup_id, group in oos_ledger.groupby(setup_col):
            setup_groups[str(setup_id)] = group

        # Apply deduplication to existing data as well
        full_ledger = _deduplicate_trades(oos_ledger)

        return _run_stages_and_output(run_dir, setup_groups, full_ledger, settings, config, diagnostics)

    except Exception as e:
        print(f"Error in strict OOS mode: {e}")
        return _write_empty_outputs(run_dir)


def _run_fresh_backtests(
        candidates: pd.DataFrame,
        master_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        signals_metadata: List[Dict[str, Any]],
        settings: Settings,
) -> Dict[str, pd.DataFrame]:
    """
    Run fresh backtests for all candidates from their train-end dates to TODAY.

    This is the core of Legacy mode - it generates current, actionable signals
    by testing each setup from when it was discovered to the present moment.
    """

    tasks = []
    for _, row in candidates.iterrows():
        signal_ids = [s.strip() for s in str(row["signal_ids"]).split(",")] if pd.notnull(row["signal_ids"]) else []

        # Start OOS period the day after training ended
        oos_start = pd.to_datetime(row["train_end_date"]) + pd.Timedelta(days=1)

        # Ensure the OOS start date is reasonable
        if oos_start > pd.Timestamp("2025-09-15"):
            continue  # Skip if train period extends beyond our data

        tasks.append(delayed(run_gauntlet_backtest)(
            setup_id=str(row["setup_id"]),
            specialized_ticker=str(row["specialized_ticker"]),
            signal_ids=signal_ids,
            direction=str(row["direction"]),
            oos_start_date=oos_start,
            master_df=master_df,
            signals_df=signals_df,
            signals_metadata=signals_metadata,
            exit_policy=None,  # Use default regime-aware exits
            settings=settings,
            origin_fold=int(row["fold"]),
        ))

    results = {}
    if tasks:
        print(f"[Fresh Backtests] Processing {len(tasks)} setups...")
        with tqdm(total=len(tasks), desc="Running fresh backtests", dynamic_ncols=True) as pbar:
            with Parallel(n_jobs=4, backend="loky", timeout=300) as parallel:
                backtest_outputs = parallel(tasks)
                for i, ledger in enumerate(backtest_outputs):
                    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
                        setup_id = str(candidates.iloc[i]["setup_id"])
                        results[setup_id] = ledger
                        print(f"Setup {setup_id}: {len(ledger)} trades, {ledger['exit_date'].isna().sum()} open")
                    pbar.update(1)

    return results


def _combine_and_deduplicate_trades(setup_ledgers: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all setup ledgers and apply robust deduplication.

    This addresses Fix #2 by preventing phantom trades and exact duplicates
    from polluting the results.
    """
    if not setup_ledgers:
        return pd.DataFrame()

    # Combine all ledgers
    all_ledgers = list(setup_ledgers.values())
    combined = pd.concat(all_ledgers, ignore_index=True)

    return _deduplicate_trades(combined)


def _deduplicate_trades(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Apply robust deduplication logic to prevent data quality issues.

    This implements comprehensive deduplication based on:
    1. Exact duplicates across all key fields
    2. Same-day entry/exit with identical underlying prices (phantom trades)
    3. Multiple entries for the same setup on the same day
    """
    if ledger.empty:
        return ledger

    original_count = len(ledger)
    df = ledger.copy()

    # Ensure proper date types
    for col in ["trigger_date", "exit_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Step 1: Remove exact duplicates based on core trading fields
    key_cols = ["setup_id", "ticker", "direction", "trigger_date", "strike", "horizon_days"]
    available_key_cols = [col for col in key_cols if col in df.columns]

    if available_key_cols:
        df = df.drop_duplicates(subset=available_key_cols, keep="first")
        print(f"[Deduplication] Removed {original_count - len(df)} exact duplicates")

    # Step 2: Remove phantom same-day trades with identical underlying prices
    if "entry_underlying" in df.columns and "exit_underlying" in df.columns:
        phantom_mask = (
                (df["trigger_date"].dt.normalize() == df["exit_date"].dt.normalize()) &
                (abs(df["entry_underlying"] - df["exit_underlying"]) < 1e-6) &
                (df["exit_date"].notna())  # Only check closed positions
        )
        phantom_count = phantom_mask.sum()
        if phantom_count > 0:
            df = df[~phantom_mask]
            print(f"[Deduplication] Removed {phantom_count} phantom same-day trades")

    # Step 3: For each setup+ticker+direction, keep only one trade per day
    if "setup_id" in df.columns and "ticker" in df.columns:
        df["trade_date"] = df["trigger_date"].dt.normalize()
        dedup_cols = ["setup_id", "ticker", "direction", "trade_date"]
        available_dedup_cols = [col for col in dedup_cols if col in df.columns]

        if len(available_dedup_cols) >= 3:  # Need at least setup, ticker, date
            pre_daily_count = len(df)
            df = df.sort_values("trigger_date").drop_duplicates(
                subset=available_dedup_cols, keep="first"
            )
            daily_dedup_count = pre_daily_count - len(df)
            if daily_dedup_count > 0:
                print(f"[Deduplication] Removed {daily_dedup_count} same-day duplicates")

        df = df.drop(columns=["trade_date"], errors="ignore")

    final_count = len(df)
    print(f"[Deduplication] Final: {final_count} trades ({original_count - final_count} total removed)")

    return df.sort_values(["trigger_date", "ticker"], ignore_index=True)


def _run_stages_and_output(
        run_dir: str,
        setup_ledgers: Dict[str, pd.DataFrame],
        full_ledger: pd.DataFrame,
        settings: Settings,
        config: Dict[str, Any],
        diagnostics: bool,
) -> Dict[str, str]:
    """
    Run the 3-stage gauntlet on regime-aware backtests and generate consolidated outputs.

    The stages are designed to identify setups that are:
    1. Healthy and active (Stage 1: Health Check)
    2. Profitable and stable (Stage 2: Profitability Check)
    3. Statistically robust (Stage 3: Robustness Check)
    """

    print(f"[Gauntlet Stages] Processing {len(setup_ledgers)} setups through 3-stage evaluation...")

    stage_results = []

    # Run stages for each setup
    for setup_id, ledger in setup_ledgers.items():

        # Stage 1: Health Check (recency, activity, momentum)
        mock_summary = pd.DataFrame([{"setup_id": setup_id, "rank": 1}])
        s1_result = run_stage1_health_check(
            fold_summary=mock_summary,
            fold_ledger=ledger,
            config=config
        )

        stage1_passed = bool(s1_result["pass_stage1"].iloc[0]) if not s1_result.empty else False
        stage1_reason = str(s1_result["reason"].iloc[0]) if not s1_result.empty else "no_data"

        if not stage1_passed:
            stage_results.append({
                "setup_id": setup_id,
                "stage1_pass": False, "stage1_reason": stage1_reason,
                "stage2_pass": False, "stage2_reason": "skipped_stage1_fail",
                "stage3_pass": False, "stage3_reason": "skipped_stage1_fail",
                "final_decision": "Retire"
            })
            continue

        # Stage 2: Profitability Check (NAV, PnL, hit rate, drawdown)
        s2_result = run_stage2_profitability_on_ledger(
            fold_ledger=ledger,
            settings=settings,
            config=config,
            stage1_df=s1_result
        )

        stage2_passed = bool(s2_result["pass_stage2"].iloc[0]) if not s2_result.empty else False
        stage2_reason = str(s2_result["reason"].iloc[0]) if not s2_result.empty else "no_data"

        if not stage2_passed:
            stage_results.append({
                "setup_id": setup_id,
                "stage1_pass": True, "stage1_reason": "ok",
                "stage2_pass": False, "stage2_reason": stage2_reason,
                "stage3_pass": False, "stage3_reason": "skipped_stage2_fail",
                "final_decision": "Retire"
            })
            continue

        # Stage 3: Robustness Check (DSR, bootstrap CI, stability)
        s3_result = run_stage3_robustness_on_ledger(
            fold_ledger=ledger,
            settings=settings,
            config=config,
            stage2_df=s2_result
        )

        stage3_passed = bool(s3_result["pass_stage3"].iloc[0]) if not s3_result.empty else False
        stage3_reason = str(s3_result["reason"].iloc[0]) if not s3_result.empty else "no_data"

        # Determine final decision based on stage results
        if stage3_passed:
            final_decision = "Deploy"  # Passed all stages - ready for live trading
        elif stage2_passed:
            final_decision = "Monitor"  # Profitable but not robust - watch carefully
        else:
            final_decision = "Retire"  # Failed early stages - not viable

        stage_results.append({
            "setup_id": setup_id,
            "stage1_pass": True, "stage1_reason": "ok",
            "stage2_pass": True, "stage2_reason": "ok",
            "stage3_pass": stage3_passed, "stage3_reason": stage3_reason,
            "final_decision": final_decision
        })

    # Print stage summary
    deploy_count = len([r for r in stage_results if r['final_decision'] == 'Deploy'])
    monitor_count = len([r for r in stage_results if r['final_decision'] == 'Monitor'])
    retire_count = len([r for r in stage_results if r['final_decision'] == 'Retire'])

    print(f"[Gauntlet Results] Deploy: {deploy_count}, Monitor: {monitor_count}, Retire: {retire_count}")

    # Generate consolidated outputs
    return write_gauntlet_outputs(
        run_dir=run_dir,
        stage_results=stage_results,
        full_ledger=full_ledger,
        setup_ledgers=setup_ledgers,
        settings=settings,
        diagnostics=diagnostics
    )


def _write_empty_outputs(run_dir: str) -> Dict[str, str]:
    """Write empty output files when no data is available."""
    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    # Create empty DataFrames with correct schemas
    empty_results = pd.DataFrame(columns=[
        "setup_id", "specialized_ticker", "direction", "signal_ids",
        "stage1_pass", "stage1_reason", "stage2_pass", "stage2_reason",
        "stage3_pass", "stage3_reason", "final_decision",
        "total_trades", "open_trades", "days_since_last",
        "total_pnl", "nav_return_pct", "sharpe_ratio"
    ])

    empty_positions = pd.DataFrame(columns=[
        "setup_id", "specialized_ticker", "direction", "entry_date",
        "days_open", "unrealized_pnl", "contracts"
    ])

    empty_ledger = pd.DataFrame()

    paths = {
        "gauntlet_results": os.path.join(gaunt_dir, "gauntlet_results.csv"),
        "open_positions": os.path.join(gaunt_dir, "open_positions.csv"),
        "gauntlet_ledger": os.path.join(gaunt_dir, "gauntlet_ledger.csv")
    }

    empty_results.to_csv(paths["gauntlet_results"], index=False)
    empty_positions.to_csv(paths["open_positions"], index=False)
    empty_ledger.to_csv(paths["gauntlet_ledger"], index=False)

    print("Generated empty gauntlet outputs")
    return paths


def _find_setup_col(df: pd.DataFrame) -> str:
    """Find setup identifier column in the dataframe."""
    for col in ["setup_id", "Setup", "strategy_id"]:
        if col in df.columns:
            return col
    return df.columns[0] if len(df.columns) > 0 else "setup_id"