# main_phase_a.py - TEMPORARY: Backtesting-based discovery
"""
PHASE A: Simple walk-forward + GA + OPTIONS BACKTESTING

This is a simplified main that uses backtesting fitness (like OLD_VERSION) 
but with improved signal caching and GA infrastructure.

Run this to get trades TODAY while we refine the full system.
"""
from __future__ import annotations
import os
os.environ.update(
    VECLIB_MAXIMUM_THREADS='1',
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    JOBLIB_TEMP_FOLDER=os.getenv('JOBLIB_TEMP_FOLDER', os.getenv('TMPDIR', '/tmp')),
)

import pandas as pd
from datetime import datetime
from pathlib import Path

from alpha_discovery.config import settings
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import check_signals_cache, compile_signals
from alpha_discovery.search.nsga import evolve
from alpha_discovery.reporting.artifacts import save_results, materialize_per_fold_artifacts
from alpha_discovery.data.loader import load_data_from_parquet


def create_walk_forward_splits(
    data_index: pd.DatetimeIndex,
    train_years: float = 3,
    test_years: float = 1,
    step_months: int = 12
):
    """Simple walk-forward splits (copied from OLD_VERSION validation.py)."""
    splits = []
    if data_index.empty:
        return splits
    
    start_date = data_index.min()
    end_date = data_index.max()
    
    train_months = int(round(train_years * 12))
    test_months = int(round(test_years * 12))
    
    train_period = pd.DateOffset(months=train_months)
    test_period = pd.DateOffset(months=test_months)
    step_period = pd.DateOffset(months=step_months)
    embargo_period = pd.DateOffset(days=int(settings.validation.embargo_days))
    
    current_start = start_date
    
    print("\n--- Creating Walk-Forward Splits ---")
    
    while True:
        train_end = current_start + train_period
        test_start = train_end + embargo_period
        test_end = test_start + test_period
        
        if test_end > end_date:
            break
        
        train_indices = data_index[(data_index >= current_start) & (data_index < train_end)]
        test_indices = data_index[(data_index >= test_start) & (data_index < test_end)]
        
        if not train_indices.empty and not test_indices.empty:
            splits.append((train_indices, test_indices))
            print(
                f"Created Split {len(splits)}: "
                f"Train ({train_indices.min().date()} to {train_indices.max().date()}), "
                f"Test ({test_indices.min().date()} to {test_indices.max().date()})"
            )
        
        current_start += step_period
    
    print(f"Generated {len(splits)} walk-forward splits.")
    return splits


def main():
    print("="*80)
    print(" "*20 + "ALPHA DISCOVERY - PHASE A")
    print(" "*15 + "Backtesting-Based Discovery Mode")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration Seed: {settings.ga.seed}")
    print("-"*80)
    
    # Load data
    print("\n--- Loading Data ---")
    master_df = load_data_from_parquet()
    if master_df is None or master_df.empty:
        print("Master dataframe is empty. Exiting.")
        return
    
    # Build features & signals (with caching)
    print("\n--- Building Features & Signals ---")
    signals_df, signals_meta = check_signals_cache(master_df)
    
    if signals_df is None:
        print("  Cache miss - building features...")
        feature_matrix = build_feature_matrix(master_df)
        signals_df, signals_meta = compile_signals(feature_matrix)
    
    # Ensure DatetimeIndex
    if not isinstance(signals_df.index, pd.DatetimeIndex):
        signals_df = signals_df.copy()
        signals_df.index = pd.to_datetime(signals_df.index)
    
    # Create run directory
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"run_phase_a_seed{settings.ga.seed}_{run_timestamp}"
    output_dir = os.path.join('runs', folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Run Directory: {output_dir} ---")
    
    # Create walk-forward splits
    splits = create_walk_forward_splits(
        data_index=signals_df.index,
        train_years=3,
        test_years=1,
        step_months=12
    )
    
    print(f"Total folds created: {len(splits)}")
    
    # Run GA evolution on each fold
    all_fold_results = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        fold_num = i + 1
        print(f"\n{'='*80}")
        print(f"RUNNING FOLD {fold_num}/{len(splits)}")
        print(f"{'='*80}")
        print(f"Training Period: {train_idx.min().date()} to {train_idx.max().date()}")
        print(f"Testing  Period: {test_idx.min().date()} to {test_idx.max().date()}")
        
        # Prepare fold data
        train_master_df = master_df.loc[train_idx]
        train_signals_df = signals_df.reindex(train_idx).fillna(False)
        
        print(f"\n[TRAIN] Running GA evolution with OPTIONS BACKTESTING fitness...")
        
        # Run GA evolution (uses backtesting evaluation from ga_core.py)
        pareto_front_for_fold = evolve(train_signals_df, signals_meta, train_master_df)
        
        # Tag fold number
        for solution in pareto_front_for_fold:
            solution["fold"] = fold_num
        
        all_fold_results.extend(pareto_front_for_fold)
        
        print(f"\n[FOLD {fold_num}] Discovered {len(pareto_front_for_fold)} Pareto-optimal setups")
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD COMPLETE - {len(all_fold_results)} solutions discovered")
    print(f"{'='*80}")
    
    # Save results using existing artifacts system
    print("\n--- Saving Training Results ---")
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_fold_results)
    # Call save_results with correct signature: (df, metadata, run_dir, splits, settings)
    save_results(results_df, signals_meta, output_dir, splits, settings)
    print("Results saved.")
    
    try:
        materialize_per_fold_artifacts(base_dir=output_dir)
        print("Per-fold artifacts materialized.")
    except Exception as e:
        print(f"Warning: could not materialize per-fold artifacts: {e}")
    
    print(f"\n{'='*80}")
    print(f"✅ PHASE A COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Results in: {output_dir}")
    print(f"   - Pareto front CSV: {output_dir}/pareto/pareto_front_summary.csv")
    print(f"   - Trade ledger: {output_dir}/pareto/pareto_front_trade_ledger.csv")
    print(f"   - Per-fold results: {output_dir}/fold_*/")
    print(f"\n🎯 Check the trade ledger for setups to trade TODAY!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

