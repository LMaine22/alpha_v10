# main.py
import os
import argparse
from datetime import datetime

# Tame BLAS/joblib threads for stability/repro
os.environ.update(
    VECLIB_MAXIMUM_THREADS='1',
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    JOBLIB_TEMP_FOLDER=os.getenv('JOBLIB_TEMP_FOLDER', os.getenv('TMPDIR', '/tmp')),
)

import pandas as pd

from alpha_discovery.config import settings, gauntlet_cfg
from alpha_discovery.gauntlet.run import run_gauntlet
from alpha_discovery.data.loader import load_data_from_parquet
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.utils.split_patch import extend_last_test_window
from alpha_discovery.signals.compiler import compile_signals
from alpha_discovery.search.nsga import evolve
from alpha_discovery.reporting.artifacts import save_results, materialize_per_fold_artifacts, _portfolio_daily_returns_from_ledger, _compound_total_return, _get_base_portfolio_capital, _signal_ids_str, _desc_from_meta, _build_signal_meta_map, _format_date, _format_setup_description
from alpha_discovery.eval.validation import create_walk_forward_splits
from alpha_discovery.engine.bt_common import TRADE_HORIZONS_DAYS
from alpha_discovery.engine import backtester
from alpha_discovery.search.ga_core import _infer_direction_from_metadata
from alpha_discovery.eval import metrics
from alpha_discovery.eval.selection import portfolio_daily_returns

# Global setup counter for unique SETUP_XXXX IDs
_setup_counter = 0
_setup_id_map = {}  # Maps (ticker, sorted_signals) to SETUP_XXXX

def _generate_unique_setup_id(ticker: str, signals: list) -> str:
    """Generate unique SETUP_XXXX identifiers instead of concatenated signal IDs"""
    global _setup_counter, _setup_id_map
    
    key = (ticker, tuple(sorted(signals)))
    if key not in _setup_id_map:
        _setup_counter += 1
        _setup_id_map[key] = f"SETUP_{_setup_counter:04d}"
    
    return _setup_id_map[key]


def run_oos_backtesting(
    pareto_front_solutions: list,
    signals_metadata: list,
    test_master_df: pd.DataFrame,
    test_signals_df: pd.DataFrame,
    fold_num: int
) -> list:
    """
    Run out-of-sample backtesting on the test window for evolved solutions.
    Mirrors the training evaluation but on unseen test data.
    """
    print(f"    Running OOS backtesting on {len(pareto_front_solutions)} solutions...")
    
    oos_results = []
    
    for i, solution in enumerate(pareto_front_solutions):
        # Extract solution details
        individual = solution.get('individual')
        if not individual or len(individual) != 2:
            continue
            
        ticker, setup_signals = individual
        if not setup_signals:
            continue
            
        direction = _infer_direction_from_metadata(setup_signals, signals_metadata)
        
        # Run OOS backtest using the same backtesting engine
        # No max_open_days for OOS - trades should be open based on horizon only
        oos_ledger = backtester.run_setup_backtest_options(
            setup_signals=setup_signals,
            signals_df=test_signals_df,
            master_df=test_master_df,
            direction=direction,
            exit_policy=solution.get('exit_policy'),
            tickers_to_run=[ticker]
        )
        
        if oos_ledger is None or oos_ledger.empty:
            continue
            
        # Create unique setup_id using the new system
        setup_id = _generate_unique_setup_id(ticker, setup_signals)
        oos_ledger = oos_ledger.copy()
        oos_ledger["setup_id"] = setup_id
        oos_ledger["fold"] = fold_num
        oos_ledger["solution_rank"] = solution.get('rank', i)
        
        # Store OOS result with original solution info
        oos_result = solution.copy()
        oos_result["oos_trade_ledger"] = oos_ledger
        oos_result["oos_fold"] = fold_num
        
        oos_results.append(oos_result)
    
    print(f"    OOS backtesting complete: {len(oos_results)} solutions with trades")
    return oos_results


def save_oos_results(
    oos_fold_results: list,
    signals_metadata: list,
    settings,
    output_dir: str,
    fold_num: int
) -> None:
    """
    Save OOS results with simplified format matching gauntlet summary structure.
    """
    if not oos_fold_results:
        print(f"    No OOS results to save for fold {fold_num}")
        return
        
    print(f"    Saving OOS artifacts for fold {fold_num}...")
    
    # Create OOS-specific directory
    oos_folds_dir = os.path.join(output_dir, "oos_folds")
    os.makedirs(oos_folds_dir, exist_ok=True)
    oos_dir = os.path.join(oos_folds_dir, f"fold_{fold_num:02d}_oos")
    os.makedirs(oos_dir, exist_ok=True)
    
    # Build signal metadata map for descriptions
    signal_meta_map = _build_signal_meta_map(signals_metadata)
    
    summary_rows = []
    all_oos_ledgers = []
    
    for i, solution in enumerate(oos_fold_results):
        individual = solution.get('individual', (None, []))
        if isinstance(individual, tuple) and len(individual) == 2:
            specialized_ticker, setup_signal_items = individual
        else:
            specialized_ticker = 'UNKNOWN'
            setup_signal_items = []
            
        # Use the unique setup ID that was already generated in run_oos_backtesting
        setup_id = solution.get('oos_trade_ledger', pd.DataFrame()).get('setup_id', pd.Series()).iloc[0] if not solution.get('oos_trade_ledger', pd.DataFrame()).empty else _generate_unique_setup_id(specialized_ticker, setup_signal_items)
        direction = solution.get('direction', 'N/A')
        
        # Get signal description and signal IDs string
        sig_str = _signal_ids_str(setup_signal_items)
        desc = _desc_from_meta(setup_signal_items, signal_meta_map)
        if not desc:
            desc = _format_setup_description(solution)
        
        oos_ledger = solution.get('oos_trade_ledger', pd.DataFrame())
        
        if isinstance(oos_ledger, pd.DataFrame) and not oos_ledger.empty:
            # Calculate basic OOS metrics
            oos_total_trades = len(oos_ledger)
            oos_open_trades = oos_ledger['exit_date'].isna().sum() if 'exit_date' in oos_ledger.columns else 0
            oos_sum_pnl_dollars = float(pd.to_numeric(oos_ledger.get('pnl_dollars', 0), errors="coerce").fillna(0).sum())
            
            # Portfolio NAV calculation
            daily_returns = _portfolio_daily_returns_from_ledger(oos_ledger, settings)
            oos_nav_total_return_pct = _compound_total_return(daily_returns)
            oos_final_nav = float(_get_base_portfolio_capital(settings) * (1.0 + oos_nav_total_return_pct))
            
            # Calculate expectancy (average PnL per trade)
            oos_expectancy = float(oos_sum_pnl_dollars / oos_total_trades) if oos_total_trades > 0 else 0.0
            
            # Calculate OOS Sortino ratio (lower bound)
            try:
                oos_portfolio_metrics = metrics.calculate_portfolio_metrics(
                    daily_returns=daily_returns,
                    portfolio_ledger=oos_ledger
                )
                oos_sortino_lb = oos_portfolio_metrics.get('sortino_lb', 0.0)
            except Exception as e:
                print(f"Warning: Could not calculate OOS sortino for {setup_id}: {e}")
                oos_sortino_lb = 0.0
            
            # Format dates
            oos_first_trigger = _format_date(oos_ledger['trigger_date'].min())
            oos_last_trigger = _format_date(oos_ledger['trigger_date'].max())
            
            # Build simplified OOS record (without days_since_last_trigger, with sortino_lb)
            flat_record = {
                'setup_id': setup_id,
                'specialized_ticker': specialized_ticker,
                'direction': direction,
                'description': desc,
                'signal_ids': sig_str,
                'oos_first_trigger': oos_first_trigger,
                'oos_last_trigger': oos_last_trigger,
                'oos_total_trades': oos_total_trades,
                'oos_open_trades': oos_open_trades,
                'oos_sum_pnl_dollars': oos_sum_pnl_dollars,
                'oos_final_nav': oos_final_nav,
                'oos_nav_total_return_pct': oos_nav_total_return_pct,
                'oos_sortino_lb': oos_sortino_lb,
                'expectancy': oos_expectancy
            }
            summary_rows.append(flat_record)
            all_oos_ledgers.append(oos_ledger)
            
        else:  # Handle case where there is no OOS ledger
            flat_record = {
                'setup_id': setup_id,
                'specialized_ticker': specialized_ticker,
                'direction': direction,
                'description': desc,
                'signal_ids': sig_str,
                'oos_first_trigger': '',
                'oos_last_trigger': '',
                'oos_total_trades': 0,
                'oos_open_trades': 0,
                'oos_sum_pnl_dollars': 0.0,
                'oos_final_nav': float(_get_base_portfolio_capital(settings)),
                'oos_nav_total_return_pct': 0.0,
                'oos_sortino_lb': 0.0,
                'expectancy': 0.0
            }
            summary_rows.append(flat_record)
    
    # Save simplified OOS summary
    if summary_rows:
        oos_summary_df = pd.DataFrame(summary_rows)
        
        # Format nav_total_return_pct as percentage
        if 'oos_nav_total_return_pct' in oos_summary_df.columns:
            oos_summary_df['oos_nav_total_return_pct'] = (
                pd.to_numeric(oos_summary_df['oos_nav_total_return_pct'], errors="coerce").fillna(0.0) * 100.0
            ).round(4).map(lambda x: f"{x:.4f}")
        
        oos_summary_path = os.path.join(oos_dir, 'oos_pareto_front_summary.csv')
        oos_summary_df.to_csv(oos_summary_path, index=False, float_format='%.4f')
        print(f"    OOS summary saved to: {oos_summary_path}")
    
    # Save OOS trade ledger
    if all_oos_ledgers:
        full_oos_ledger_df = pd.concat(all_oos_ledgers, ignore_index=True)
        oos_ledger_path = os.path.join(oos_dir, 'oos_pareto_front_trade_ledger.csv')
        full_oos_ledger_df.to_csv(oos_ledger_path, index=False, float_format='%.6f')
        print(f"    OOS trade ledger saved to: {oos_ledger_path}")


def save_combined_oos_results(output_dir: str) -> None:
    """
    Aggregate all OOS results across folds into combined summary files.
    """
    print("\n--- Saving Combined Out-of-Sample Results ---")
    
    all_oos_summaries = []
    all_oos_ledgers = []
    
    # Find all OOS directories - check both old structure and new oos_folds/ structure
    oos_folds_dir = os.path.join(output_dir, 'oos_folds')
    
    if os.path.exists(oos_folds_dir):
        # New structure: look in oos_folds/ subdirectory
        oos_dirs = [d for d in os.listdir(oos_folds_dir) if d.endswith('_oos') and os.path.isdir(os.path.join(oos_folds_dir, d))]
        base_oos_path = oos_folds_dir
    else:
        # Old structure: look in main output directory
        oos_dirs = [d for d in os.listdir(output_dir) if d.endswith('_oos') and os.path.isdir(os.path.join(output_dir, d))]
        base_oos_path = output_dir
    
    oos_dirs.sort()  # Ensure consistent ordering
    
    if not oos_dirs:
        print("No OOS results found to aggregate.")
        return
    
    print(f"Aggregating OOS results from {len(oos_dirs)} folds...")
    
    for oos_dir in oos_dirs:
        oos_path = os.path.join(base_oos_path, oos_dir)
        
        # Load OOS summary
        summary_file = os.path.join(oos_path, 'oos_pareto_front_summary.csv')
        if os.path.exists(summary_file):
            try:
                oos_summary = pd.read_csv(summary_file)
                all_oos_summaries.append(oos_summary)
            except Exception as e:
                print(f"Warning: Could not load {summary_file}: {e}")
        
        # Load OOS ledger
        ledger_file = os.path.join(oos_path, 'oos_pareto_front_trade_ledger.csv')
        if os.path.exists(ledger_file):
            try:
                oos_ledger = pd.read_csv(ledger_file)
                all_oos_ledgers.append(oos_ledger)
            except Exception as e:
                print(f"Warning: Could not load {ledger_file}: {e}")
    
    # Save combined OOS summary
    if all_oos_summaries:
        combined_oos_summary = pd.concat(all_oos_summaries, ignore_index=True)
        # Create pareto directory for OOS files
        pareto_dir = os.path.join(output_dir, 'pareto')
        os.makedirs(pareto_dir, exist_ok=True)
        
        combined_summary_path = os.path.join(pareto_dir, 'oos_pareto_front_summary_combined.csv')
        combined_oos_summary.to_csv(combined_summary_path, index=False, float_format='%.4f')
        print(f"Combined OOS summary saved to: {combined_summary_path}")
        
        # Generate final report message
        total_setups = len(combined_oos_summary)
        total_trades = combined_oos_summary['oos_total_trades'].sum() if 'oos_total_trades' in combined_oos_summary.columns else 0
        print(f"Final reports from winning setups across all folds (OOS): {total_setups} setups, {total_trades} trades")
    
    # Save combined OOS ledger  
    if all_oos_ledgers:
        combined_oos_ledger = pd.concat(all_oos_ledgers, ignore_index=True)
        combined_ledger_path = os.path.join(pareto_dir, 'oos_pareto_front_trade_ledger_combined.csv')
        combined_oos_ledger.to_csv(combined_ledger_path, index=False, float_format='%.6f')
        print(f"Combined OOS trade ledger saved to: {combined_ledger_path}")


def run_pipeline(args=None):
    """
    Walk-forward pipeline with TRAINING + OOS testing per fold, followed by optional Gauntlet validation.
    
    For each fold:
    1. Train GA on train window (unchanged - same artifacts as before)
    2. Test evolved solutions on test window (NEW - generates oos_* artifacts)
    3. Save both training and OOS results separately
    
    Then optionally run Gauntlet on final holdout period.
    """
    print("--- Loading data ---")
    master_df = load_data_from_parquet()
    if master_df is None or master_df.empty:
        print("Master dataframe is empty. Exiting.")
        return

    print("--- Building features ---")
    feature_matrix = build_feature_matrix(master_df)
    if feature_matrix is None or feature_matrix.empty:
        print("Feature matrix is empty. Exiting.")
        return

    print("--- Compiling signals ---")
    signals_df, signals_metadata = compile_signals(feature_matrix)
    if signals_df is None or signals_df.empty:
        print("Signals dataframe is empty. Exiting.")
        return

    if not isinstance(signals_df.index, pd.DatetimeIndex):
        signals_df = signals_df.copy()
        signals_df.index = pd.to_datetime(signals_df.index)

    print("--- Creating walk-forward splits ---")
    splits = create_walk_forward_splits(data_index=signals_df.index)
    
    # Extend last test window to include recent/live data
    if args and hasattr(args, 'extend_last_test_to'):
        extend_arg = args.extend_last_test_to
        if extend_arg and extend_arg.lower() != "none":
            if extend_arg.lower() == "end-of-data":
                cap_date = None  # will default to full_index.max()
            else:
                cap_date = pd.to_datetime(extend_arg).normalize()

            # Use the same calendar index used by split generator
            full_index = signals_df.index

            # Apply the extension
            splits = extend_last_test_window(splits, full_index=full_index, as_of=cap_date)
            print(f"[Splits] Extended last fold TEST to {cap_date or full_index.max().date()} (inclusive).")
    
    # Keep only the most recent (last) split for quick tests
    #splits = [splits[-1]]
    #print(
    #   f"[Quick Test] Using latest split only: Train={splits[0][0][0].date()}..{splits[0][0][-1].date()}  Test={splits[0][1][0].date()}..{splits[0][1][-1].date()}")

    print(f"Total folds created: {len(splits)}")

    # Embargo sanity guard vs. holding horizon
    min_needed = max(
        int(getattr(settings.options, "exit_time_cap_days", 0) or 0),
        max(TRADE_HORIZONS_DAYS) if isinstance(TRADE_HORIZONS_DAYS, (list, tuple)) else int(TRADE_HORIZONS_DAYS),
    )
    if int(getattr(settings.validation, "embargo_days", 0)) < min_needed:
        print(
            f"Warning: embargo_days ({settings.validation.embargo_days}) < max holding horizon ({min_needed}). "
            f"Increase embargo to avoid boundary leakage."
        )

    all_fold_results = []

    # Create a single run directory for this entire pipeline execution
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"run_seed{settings.ga.seed}_{run_timestamp}"
    output_dir = os.path.join('runs', folder_name)
    os.makedirs(output_dir, exist_ok=True)

    # ===== Walk-forward training AND OOS testing across folds =====
    for i, (train_idx, test_idx) in enumerate(splits):
        fold_num = i + 1
        print(f"\n==================== RUNNING FOLD {fold_num}/{len(splits)} ====================")
        print(f"Training Period: {train_idx.min().date()} to {train_idx.max().date()}")
        print(f"Testing  Period: {test_idx.min().date()} to {test_idx.max().date()}")

        # ===== TRAINING PHASE (unchanged) =====
        train_master_df = master_df.loc[train_idx]
        train_signals_df = signals_df.reindex(train_idx).fillna(False)

        print(f"  [TRAIN] Running GA evolution...")
        pareto_front_for_fold = evolve(train_signals_df, signals_metadata, train_master_df)
        for solution in pareto_front_for_fold:
            solution["fold"] = fold_num

        all_fold_results.extend(pareto_front_for_fold)

        # ===== OOS TESTING PHASE (NEW) =====
        print(f"  [OOS] Preparing test data...")
        test_master_df = master_df.loc[test_idx]
        test_signals_df = signals_df.reindex(test_idx).fillna(False)
        
        # Run OOS backtesting on the evolved solutions
        oos_results = run_oos_backtesting(
            pareto_front_solutions=pareto_front_for_fold,
            signals_metadata=signals_metadata,
            test_master_df=test_master_df,
            test_signals_df=test_signals_df,
            fold_num=fold_num
        )
        
        # Save OOS artifacts for this fold
        save_oos_results(
            oos_fold_results=oos_results,
            signals_metadata=signals_metadata,
            settings=settings,
            output_dir=output_dir,
            fold_num=fold_num
        )

    print(f"\n{'=' * 20} WALK-FORWARD (TRAIN + OOS) COMPLETE {'=' * 20}")

    print("\n--- Saving In-Sample Training Results ---")
    save_results(all_fold_results, signals_metadata, settings, output_dir=output_dir)
    print("Global results saved.")

    try:
        materialize_per_fold_artifacts(base_dir=output_dir)
        print("Per-fold TRAIN artifacts materialized.")
    except Exception as e:
        print(f"Warning: could not materialize per-fold artifacts: {e}")

    # Save combined OOS results
    save_combined_oos_results(output_dir)

    # ===== Gauntlet: true OOS validation =====
    if settings.ga.run_gauntlet:
        print("\n--- Launching Out-of-Sample Gauntlet ---")
        try:
            run_gauntlet(
                run_dir=output_dir,              # Use this run's directory
                settings=settings,
                config=gauntlet_cfg(settings),   # Flattened Stage1/2/3 knobs
                master_df=master_df,
                signals_df=signals_df,
                signals_metadata=signals_metadata,
                splits=splits                    # Critical: to compute train_end per fold
            )
        except Exception as e:
            print(f"ERROR: Gauntlet failed to run: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Gauntlet SKIPPED (run_gauntlet = False) ---")

    # --- NEW: Strict-OOS Gauntlet (Phase 1-3) ---
    if settings.ga.run_strict_oos_gauntlet:
        print("\n--- Launching Strict-OOS Gauntlet (OOS Test Data) ---")
        try:
            from alpha_discovery.gauntlet.run import run_gauntlet_strict_oos
            strict_oos_results = run_gauntlet_strict_oos(
                run_dir=output_dir,
                splits=splits,
                outdir=None,  # Uses default: runs/<run>/gauntlet/strict_oos/
                settings=settings,
                config=gauntlet_cfg(settings),
                stage1_recency_days=14,  # Keep 14-day recency window
                stage1_min_trades=0      # Remove minimum trades filter
            )
            print(f"Strict-OOS Gauntlet completed: {strict_oos_results}")
        except Exception as e:
            print(f"ERROR: Strict-OOS Gauntlet failed to run: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Strict-OOS Gauntlet SKIPPED (run_strict_oos_gauntlet = False) ---")

    # --- NEW: Diagnostic Replay + Portfolio Analysis (Phase 4-6) ---
    if settings.ga.run_diagnostic_replay:
        print("\n--- Launching Diagnostic Replay + Portfolio Analysis ---")
        try:
            from alpha_discovery.gauntlet.diagnostic_replay import build_diagnostic_replay, summarize_diagnostic_replay  # pyright: ignore[reportMissingImports]
            from alpha_discovery.gauntlet.portfolio_diag import simulate_portfolio  # pyright: ignore[reportMissingImports]
            from alpha_discovery.gauntlet.run import _ensure_dir
            
            # Build diagnostic replay (survivors-only by default)
            replay_df = build_diagnostic_replay(
                run_dir=output_dir,
                splits=splits,
                survivors_only=True  # Only Strict-OOS survivors, like --default mode
            )
            
            # Save replay artifacts
            replay_base = _ensure_dir(os.path.join(output_dir, 'gauntlet', 'diagnostic_replay'))
            replay_df.to_csv(os.path.join(replay_base, 'diag_replay_ledger.csv'), index=False)
            
            replay_summary = summarize_diagnostic_replay(replay_df)
            replay_summary.to_csv(os.path.join(replay_base, 'diag_replay_summary.csv'), index=False)
            
            # Run portfolio diagnostics with default settings
            portfolio_results = simulate_portfolio(
                df=replay_df,
                out_base=replay_base,
                starting_capital=100_000.0,
                position_size=1_000.0,
                max_concurrent=5,
                since_knowledge=False  # Include all trades by default
            )
            
            print(f"Diagnostic Replay completed: ledger={len(replay_df)} rows, portfolio={portfolio_results}")
        except Exception as e:
            print(f"ERROR: Diagnostic Replay failed to run: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Diagnostic Replay SKIPPED (run_diagnostic_replay = False) ---")


def main():
    parser = argparse.ArgumentParser(description="Alpha Discovery Pipeline")
    parser.add_argument("--extend-last-test-to", type=str, default="end-of-data",
                        help="Either 'end-of-data' or an explicit YYYY-MM-DD date to extend the LAST fold's TEST window to.")
    args = parser.parse_args()
    
    print("--- Starting Full Alpha Discovery Pipeline ---")
    run_pipeline(args)
    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    main()
