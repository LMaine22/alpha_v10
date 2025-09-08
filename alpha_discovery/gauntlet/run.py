# alpha_discovery/gauntlet/run.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import Settings  # type: ignore
from .io import find_latest_run_dir, read_global_artifacts
from .stage1_health import run_stage1_health_check
from .stage2_profitability import run_stage2_profitability_on_ledger
from .stage3_robustness import run_stage3_robustness_on_ledger
from .summary import write_gauntlet_summary, write_all_setups_summary, write_open_trades_summary
from .backtester import run_gauntlet_backtest
from .reporting import write_stage_csv
from .io import ensure_dir
from .backtester import _align_to_pareto_schema_auto
from .config_new import get_permissive_gauntlet_config
from .medium import run_medium_gauntlet, SetupResults, write_gauntlet_artifacts
from ..meta_labeling import MetaLabelingSystem



def run_gauntlet(
    run_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
    config: Optional[Dict[str, Any]] = None,
    master_df: Optional[pd.DataFrame] = None,
    signals_df: Optional[pd.DataFrame] = None,
    signals_metadata: Optional[List[Dict[str, Any]]] = None,
    splits: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None,
    mode: str = "legacy",  # "legacy", "medium", "heavy"
) -> None:
    """
    Gauntlet pipeline:
      1) Collect all Pareto-front candidates across training folds.
      2) Run out-of-sample (OOS) backtests per candidate to build the gauntlet ledger.
      3) Stage-1/Stage-2 per setup, then Stage-3 cohort across survivors.
      4) Write summary & artifacts.
    """
    # Use permissive configuration by default to allow more strategies through
    default_config = get_permissive_gauntlet_config()
    cfg = dict(default_config)
    if config:
        cfg.update(config)

    # Resolve run_dir
    if run_dir is None:
        run_dir = find_latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("Could not resolve a valid run_dir for Gauntlet.")

    # Required inputs
    if master_df is None or signals_df is None or signals_metadata is None or splits is None:
        raise ValueError("master_df, signals_df, signals_metadata, and splits must be provided.")

    # Ensure gauntlet output dir
    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    # Base capital for NAV/returns aggregation
    try:
        base_capital = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_capital = 100_000.0

    # -----------------------
    # Phase 1: Candidate pool
    # -----------------------
    print("\n--- Phase 1: Collecting All In-Sample Candidates ---")
    pareto_summary, _ = read_global_artifacts(run_dir)
    if pareto_summary is None or pareto_summary.empty:
        print("Global pareto_front_summary.csv is empty. Cannot identify candidates.")
        return

    # Map fold -> last train date (OOS starts next day)
    fold_date_map = {i + 1: train_idx.max() for i, (train_idx, _) in enumerate(splits)}
    pareto_summary = pareto_summary.copy()
    pareto_summary["train_end_date"] = pareto_summary["fold"].map(fold_date_map)

    cols_needed = ["setup_id", "fold", "specialized_ticker", "direction", "signal_ids", "train_end_date"]
    missing = [c for c in cols_needed if c not in pareto_summary.columns]
    if missing:
        raise KeyError(f"pareto_front_summary.csv is missing columns: {missing}")

    candidates = pareto_summary.dropna(
        subset=["specialized_ticker", "direction", "signal_ids", "train_end_date"]
    ).copy()
    print(f"Found {len(candidates)} total candidates from all training folds.")

    # ---------------------------------------
    # Phase 2: Run out-of-sample backtesting
    # ---------------------------------------
    print("\n--- Phase 2: Running Out-of-Sample Backtests ---")
    tasks = []
    for _, row in candidates.iterrows():
        signal_ids = [s.strip() for s in str(row["signal_ids"]).split(",")] if pd.notnull(row["signal_ids"]) else []
        oos_start = pd.to_datetime(row["train_end_date"]) + pd.Timedelta(days=1)
        tasks.append(delayed(run_gauntlet_backtest)(
            setup_id=str(row["setup_id"]),
            specialized_ticker=str(row["specialized_ticker"]),
            signal_ids=signal_ids,
            direction=str(row["direction"]),
            oos_start_date=oos_start,
            master_df=master_df,
            signals_df=signals_df,
            signals_metadata=signals_metadata,
            exit_policy=None,       # allow backtester to use defaults/GA policy
            settings=settings,
            origin_fold=int(row["fold"]),
        ))

    oos_ledgers: Dict[str, pd.DataFrame] = {}
    if tasks:
        with tqdm(total=len(tasks), desc="Running OOS Backtests", dynamic_ncols=True) as pbar:
            with Parallel(n_jobs=4, backend="loky", timeout=300) as parallel:
                results = parallel(tasks)
                for i, ledger in enumerate(results):
                    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
                        setup_id = str(candidates.iloc[i]["setup_id"])
                        oos_ledgers[setup_id] = ledger
                    pbar.update(1)

    if not oos_ledgers:
        print("No trades were generated during out-of-sample backtesting. Gauntlet finished.")
        return

    # Concatenate all OOS ledgers and write with an explicit, forward-compatible schema
    full_oos_ledger = pd.concat(oos_ledgers.values(), ignore_index=True)
    # Align the combined OOS ledger to the Pareto (TRAIN) header, then write directly.
    try:
        # Choose any present origin_fold for the template (columns are the same across folds)
        fold_for_template = int(pd.to_numeric(full_oos_ledger.get("origin_fold")).dropna().iloc[0])
    except Exception:
        fold_for_template = 1

    aligned_ledger = _align_to_pareto_schema_auto(full_oos_ledger, origin_fold=fold_for_template)

    # Write directly to avoid the legacy LEDGER_BASE_SCHEMA coercion
    gaunt_dir = os.path.join(run_dir, "gauntlet")
    ensure_dir(gaunt_dir)
    aligned_ledger.to_csv(os.path.join(gaunt_dir, "gauntlet_ledger.csv"), index=False)
    print(f"\nOut-of-sample backtesting complete. Generated OOS ledgers for {len(oos_ledgers)} strategies.")
    
    # Generate summary for ALL setups in the trade ledger (regardless of gauntlet performance)
    write_all_setups_summary(run_dir, full_oos_ledger)
    
    # Generate summary for setups with open trades (filtered from all setups)
    write_open_trades_summary(run_dir, full_oos_ledger)

    # -----------------------------------------------------------------
    # Phase 3: Mode-specific gauntlet execution
    # -----------------------------------------------------------------
    if mode == "medium":
        print("\n--- Phase 3: Medium Gauntlet (Statistical Rigor) ---")
        _run_medium_gauntlet_mode(run_dir, oos_ledgers, settings, cfg)
        return
    elif mode == "heavy":
        print("\n--- Phase 3: Heavy Gauntlet (Academic Rigor) ---")
        # TODO: Implement heavy mode with DSR, BH-FDR, SPA, etc.
        print("Heavy mode not yet implemented, falling back to legacy mode")
        mode = "legacy"
    
    # Legacy mode (existing implementation)
    print("\n--- Phase 3: Legacy Gauntlet (Stage-1/2/3 per setup) ---")
    print("\n--- Phase 3: OOS Stage-1/2/3 per-setup ---")
    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    stage3_rows: List[Dict[str, Any]] = []

    for setup_id, oos_ledger in oos_ledgers.items():
        # Stage-1 per setup: health check (recency + activity + momentum)
        s1 = run_stage1_health_check(
            run_dir=run_dir,
            fold_num=0,
            settings=settings,
            config=cfg,
            fold_summary=pd.DataFrame([{"setup_id": setup_id, "rank": None}]),
            fold_ledger=oos_ledger,
        )
        stage1_rows.append(s1.iloc[0].to_dict())
        if not bool(s1["pass_stage1"].iloc[0]):
            continue

        # Stage-2 per setup: profitability check (NAV + PnL + win rate)
        s2 = run_stage2_profitability_on_ledger(
            fold_ledger=oos_ledger,
            settings=settings,
            config=cfg,
            stage1_df=s1,
        )
        if s2 is None or s2.empty:
            continue
        rec = s2.iloc[0].to_dict()
        rec["setup_id"] = setup_id
        stage2_rows.append(rec)

        # Stage-3 per setup: robustness check (DSR + bootstrap + stability)
        s3 = run_stage3_robustness_on_ledger(
            fold_ledger=oos_ledger,
            settings=settings,
            config=cfg,
            stage2_df=s2,
        )
        if s3 is None or s3.empty:
            continue
        s3_rec = s3.iloc[0].to_dict()
        s3_rec["setup_id"] = setup_id
        stage3_rows.append(s3_rec)

    # Write Stage diagnostics
    if stage1_rows:
        write_stage_csv(run_dir, "stage1_oos", pd.DataFrame(stage1_rows))
    if stage2_rows:
        write_stage_csv(run_dir, "stage2_oos", pd.DataFrame(stage2_rows))
    if stage3_rows:
        write_stage_csv(run_dir, "stage3_oos", pd.DataFrame(stage3_rows))

    if not stage3_rows:
        print("No candidates reached Stage-3 on OOS; gauntlet complete.")
        return

    # Survivors & summary (all Stage-3 passers are survivors)
    survivor_ids = set([str(d["setup_id"]) for d in stage3_rows if d.get("pass_stage3", False)])
    print(f"Stage-3 survivors: {len(survivor_ids)}")

    by_id_s1 = {str(d["setup_id"]): d for d in stage1_rows if "setup_id" in d}
    by_id_s2 = {str(d["setup_id"]): d for d in stage2_rows if "setup_id" in d}
    by_id_s3 = {str(d["setup_id"]): d for d in stage3_rows if "setup_id" in d}

    final_survivors: List[Dict[str, Any]] = []
    for sid in survivor_ids:
        final_survivors.append({
            "setup_id": sid,
            "oos_s1": by_id_s1.get(sid, {}),
            "oos_s2": by_id_s2.get(sid, {}),
            "oos_s3": by_id_s3.get(sid, {}),
        })

    if final_survivors:
        write_gauntlet_summary(run_dir, final_survivors, full_oos_ledger)
    else:
        print("No strategies survived Stage-3.")


def _run_medium_gauntlet_mode(run_dir: str, oos_ledgers: Dict[str, pd.DataFrame], 
                             settings: Settings, config: Dict[str, Any]) -> None:
    """Run Medium Gauntlet mode."""
    from .medium import get_medium_gauntlet_config
    
    # Get medium gauntlet config
    medium_config = get_medium_gauntlet_config()
    if config:
        medium_config.update(config)
    
    # Convert OOS ledgers to SetupResults
    setup_results = []
    for setup_id, ledger in oos_ledgers.items():
        if ledger.empty:
            continue
            
        # Get base capital
        try:
            base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
        except Exception:
            base_cap = 100_000.0
        
        # Compute daily returns and equity curve
        daily_returns = nav_daily_returns_from_ledger(ledger, base_capital=base_cap)
        equity_curve = (1 + daily_returns).cumprod() * base_cap
        
        # Extract setup info
        ticker = ledger.get('specialized_ticker', ['Unknown']).iloc[0] if not ledger.empty else 'Unknown'
        direction = ledger.get('direction', ['long']).iloc[0] if not ledger.empty else 'long'
        
        # Create SetupResults
        setup_result = SetupResults(
            setup_id=setup_id,
            ticker=ticker,
            direction=direction,
            oos_trades=ledger,
            oos_daily_returns=daily_returns,
            oos_equity_curve=equity_curve,
            historical_median_drawdown=None  # Would need historical data
        )
        setup_results.append(setup_result)
    
    if not setup_results:
        print("No valid setups found for Medium Gauntlet.")
        return
    
    print(f"Running Medium Gauntlet on {len(setup_results)} setups...")
    
    # Run medium gauntlet
    results = run_medium_gauntlet(setup_results, settings, medium_config)
    
    # Write artifacts
    output_dir = os.path.join(run_dir, "gauntlet_medium")
    write_gauntlet_artifacts(results, output_dir, medium_config)
    
    # Run meta-labeling on survivors
    if settings.meta_labeling.enabled:
        print("\n--- Phase 4: Meta-Labeling (Post-Gauntlet Filter) ---")
        _run_meta_labeling(run_dir, results, oos_ledgers, settings, config)
    
    # Print summary
    deploy_count = len([r for r in results if r.final_decision == "Deploy"])
    monitor_count = len([r for r in results if r.final_decision == "Monitor"])
    retire_count = len([r for r in results if r.final_decision == "Retire"])
    
    print(f"\nMedium Gauntlet Results:")
    print(f"  Deploy: {deploy_count}")
    print(f"  Monitor: {monitor_count}")
    print(f"  Retire: {retire_count}")
    print(f"  Artifacts written to: {output_dir}")


def _run_meta_labeling(run_dir: str, gauntlet_results: List, oos_ledgers: Dict[str, pd.DataFrame],
                      settings: Any, config: Dict[str, Any]) -> None:
    """Run meta-labeling on gauntlet survivors."""
    from ..meta_labeling import MetaLabelingSystem
    
    # Get gauntlet survivors (Deploy and Monitor decisions)
    survivors = []
    for result in gauntlet_results:
        if result.final_decision in ["Deploy", "Monitor"]:
            survivors.append({
                'setup_id': result.setup_id,
                'ticker': result.ticker,
                'direction': result.direction,
                'final_decision': result.final_decision
            })
    
    if not survivors:
        print("No gauntlet survivors for meta-labeling")
        return
    
    print(f"Running meta-labeling on {len(survivors)} gauntlet survivors...")
    
    # Initialize meta-labeling system
    meta_system = MetaLabelingSystem(config.get('meta_labeling', {}))
    
    # Load required data
    from ..data.loader import load_master_data
    from ..features.registry import build_feature_matrix
    from ..signals.compiler import compile_signals
    
    print("Loading data for meta-labeling...")
    master_df = load_master_data()
    feature_matrix = build_feature_matrix(master_df)
    signals_df, signals_metadata = compile_signals(feature_matrix)
    
    # Run meta-labeling
    meta_results = meta_system.run_meta_labeling(
        survivors, oos_ledgers, master_df, signals_df, signals_metadata
    )
    
    # Generate artifacts
    meta_output_dir = os.path.join(run_dir, "meta_labeling")
    meta_system.artifact_generator.generate_summary_artifacts(meta_results, meta_output_dir)
    
    # Print summary
    summary_stats = meta_system.get_summary_statistics()
    print(f"\nMeta-Labeling Results:")
    print(f"  Total setups: {summary_stats.get('total_setups', 0)}")
    print(f"  Successfully trained: {summary_stats.get('trained_count', 0)}")
    print(f"  Average EV improvement: {summary_stats.get('avg_ev_improvement', 0):.4f}")
    print(f"  Average Sharpe improvement: {summary_stats.get('avg_sharpe_improvement', 0):.4f}")
    print(f"  Average retention rate: {summary_stats.get('avg_retention_rate', 0):.4f}")
    print(f"  Average model accuracy: {summary_stats.get('avg_accuracy', 0):.4f}")
    print(f"  Artifacts written to: {meta_output_dir}")
