from __future__ import annotations
import os
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings but keep errors visible
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._function_base_impl')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy._core._methods')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._nanfunctions_impl')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*Persisting input arguments.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='alpha_discovery.search.ga_core')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='alpha_discovery.eval.info_metrics.robustness')
warnings.filterwarnings('ignore', message='Field name "validate" in "Settings" shadows an attribute', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r".*alpha_.*\.data\.events")

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Set numpy error handling
np.seterr(all='ignore')

from alpha_discovery.config import settings
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import compile_signals, check_signals_cache

# Import loader
try:
    from alpha_discovery.data.loader import convert_excel_to_parquet
except Exception:
    convert_excel_to_parquet = None

# Thread caps
os.environ.update(
    VECLIB_MAXIMUM_THREADS='4',
    OMP_NUM_THREADS='4',
    OPENBLAS_NUM_THREADS='4',
    MKL_NUM_THREADS='4',
    NUMEXPR_NUM_THREADS='4',
    FEATURES_PAIRWISE_MAX_TICKERS='32',
    FEATURES_PAIRWISE_JOBS='8',
    PAIRWISE_MIN_PERIODS='10',
)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and prepare data from parquet."""
    pq = settings.data.parquet_file_path
    excel = settings.data.excel_file_path
    
    if not os.path.exists(excel):
        print(f"ERROR: Excel file not found: {excel}")
        return pd.DataFrame()
    
    # Refresh parquet from Excel
    if convert_excel_to_parquet is not None:
        print(f"Loading data from '{pq}' (refreshed from Excel)...")
        convert_excel_to_parquet()
    
    if not os.path.exists(pq):
        print(f"ERROR: Parquet not found: {pq}")
        return pd.DataFrame()
    
    df = pd.read_parquet(pq)
    
    # Normalize index
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Filter date range
    df = df.loc[(df.index.date >= settings.data.start_date) & (df.index.date <= settings.data.end_date)]
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df


def create_run_dir() -> str:
    """Create timestamped run directory."""
    rd_base = settings.reporting.runs_dir
    _ensure_dir(rd_base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(rd_base, f"pawf_npwf_seed{settings.ga.seed}_{ts}")
    _ensure_dir(run_dir)
    return run_dir


def print_header():
    """Print stylized header."""
    print("=" * 80)
    print("         ALPHA DISCOVERY - PAWF + NPWF + Options Backtesting")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration Seed: {settings.ga.seed}")
    print("-" * 80)


def main():
    """Main entry point."""
    print_header()
    main_discover_workflow()


def main_discover_workflow():
    """
    Full PAWF + NPWF workflow with options backtesting.
    
    Steps:
    1. Load data and build signals (with caching)
    2. Build PAWF outer splits (4 folds with purging)
    3. For each outer fold:
       - Build NPWF inner folds from train data
       - Run Island GA with NPWF-based backtesting fitness
       - Collect Pareto-optimal setups
    4. Aggregate and save all results
    """
    
    # Load data
    print("\n--- Loading Data ---")
    master_df = load_data()
    if master_df.empty:
        print("No data; exiting.")
        return
    
    # Build/load signals (with caching)
    print("\n--- Building Features & Signals ---")
    signals_df, signals_meta = check_signals_cache(master_df)
    
    if signals_df is None:
        print("  Cache miss - building features...")
        feature_matrix = build_feature_matrix(master_df)
        signals_df, signals_meta = compile_signals(feature_matrix)
    else:
        print(f"  ✅ Loaded {len(signals_df.columns)} cached signals")
    
    # Create run directory
    run_dir = create_run_dir()
    print(f"\n--- Run Directory: {run_dir} ---")
    
    # Build PAWF outer splits
    print("\n--- Building PAWF Outer Splits ---")
    from alpha_discovery.splits.pawf import build_pawf_splits
    from alpha_discovery.splits.ids import generate_split_id
    from alpha_discovery.adapters.features import FeatureAdapter, calculate_max_lookback_from_list
    
    adapter = FeatureAdapter()
    feature_names = adapter.list_features()
    lookback_tail = calculate_max_lookback_from_list(feature_names)
    label_horizon_days = 5  # 1-week horizon for backtesting
    
    pawf_splits = build_pawf_splits(
        df=master_df,
        label_horizon_days=label_horizon_days,
        feature_lookback_tail=lookback_tail,
        min_train_months=36,        # 3 years minimum training
        test_window_days=180,       # 6-month test windows
        step_months=1,              # 1-month step forward
        regime_version="R1"
    )
    
    print(f"Created {len(pawf_splits)} PAWF outer folds")
    print(f"  Label horizon: {label_horizon_days} days")
    print(f"  Feature lookback: {lookback_tail} days")
    
    # Get tradable tickers
    tradable_tickers = settings.data.effective_tradable_tickers
    print(f"\n--- Tradable Tickers: {len(tradable_tickers)} ---")
    print(f"  {', '.join(tradable_tickers[:10])}{'...' if len(tradable_tickers) > 10 else ''}")
    
    # Run GA discovery for each outer fold
    print(f"\n{'='*80}")
    print(f"RUNNING PAWF + NPWF DISCOVERY")
    print(f"{'='*80}")
    
    all_fold_results = []
    
    from alpha_discovery.splits.npwf import make_inner_folds
    from alpha_discovery.search.island_model import IslandManager, ExitPolicy
    
    for fold_idx, split_spec in enumerate(pawf_splits, 1):
        # Extract train/test indices from SplitSpec and intersect with actual trading days
        outer_train_idx = master_df.index.intersection(split_spec.train_index)
        outer_test_idx = master_df.index.intersection(split_spec.test_index)
        split_id = generate_split_id(split_spec)
        
        print(f"\n{'='*80}")
        print(f"OUTER FOLD {fold_idx}/{len(pawf_splits)}: {split_id}")
        print(f"{'='*80}")
        print(f"Training Period: {outer_train_idx[0]} to {outer_train_idx[-1]} ({len(outer_train_idx)} days)")
        print(f"Testing  Period: {outer_test_idx[0]} to {outer_test_idx[-1]} ({len(outer_test_idx)} days)")
        
        # Build NPWF inner folds from outer train data
        print(f"\n[Fold {fold_idx}] Building NPWF inner folds...")
        df_train_outer = master_df.loc[outer_train_idx]
        inner_folds = make_inner_folds(
            df_train_outer=df_train_outer,
            label_horizon_days=label_horizon_days,
            feature_lookback_tail=lookback_tail,
            k_folds=3           # 3-fold inner CV
        )
        print(f"  Created {len(inner_folds)} NPWF inner folds for GA evaluation")
        
        # Get train/test data slices
        train_master_df = master_df.loc[outer_train_idx]
        train_signals_df = signals_df.loc[outer_train_idx]
        
        # Create ExitPolicy for GA (includes NPWF inner folds)
        exit_policy = ExitPolicy(
            train_indices=outer_train_idx,
            splits=inner_folds
        )
        
        # Run Island Model GA
        print(f"\n[Fold {fold_idx}] Running Island Model GA with NPWF-based backtesting fitness...")
        print(f"  Population: {settings.ga.population_size}, Generations: {settings.ga.generations}")
        print(f"  Islands: {settings.ga.n_islands}, Migration Interval: {settings.ga.migration_interval}")
        
        island_manager = IslandManager(
            n_islands=settings.ga.n_islands,
            n_individuals=settings.ga.population_size,
            n_generations=settings.ga.generations,
            signals_df=train_signals_df,
            signals_metadata=signals_meta,
            master_df=train_master_df,
            exit_policy=exit_policy,
            migration_interval=settings.ga.migration_interval,
            seed=settings.ga.seed + fold_idx  # Different seed per fold
        )
        
        pareto_front = island_manager.evolve()
        
        # Tag results with fold number
        for setup in pareto_front:
            setup['fold'] = fold_idx
            setup['train_start'] = str(outer_train_idx[0])
            setup['train_end'] = str(outer_train_idx[-1])
            setup['test_start'] = str(outer_test_idx[0])
            setup['test_end'] = str(outer_test_idx[-1])
        
        all_fold_results.extend(pareto_front)
        
        print(f"\n[Fold {fold_idx}] Discovered {len(pareto_front)} Pareto-optimal setups")
    
    print(f"\n{'='*80}")
    print(f"PAWF + NPWF DISCOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"Total setups discovered: {len(all_fold_results)}")
    
    # Save results
    print("\n--- Saving Results ---")
    from alpha_discovery.reporting.artifacts import save_results
    
    results_df = pd.DataFrame(all_fold_results)
    
    # Build splits object for save_results (simple dict representation)
    splits_dict = {
        'n_splits': len(pawf_splits),
        'type': 'PAWF',
        'inner_type': 'NPWF',
        'inner_folds': 3,
        'splits': [
            {
                'id': split_spec.split_id,
                'train_start': str(train_idx[0]),
                'train_end': str(train_idx[-1]),
                'test_start': str(test_idx[0]),
                'test_end': str(test_idx[-1])
            }
            for split_spec, train_idx, test_idx in pawf_splits
        ]
    }
    
    save_results(results_df, signals_meta, run_dir, splits_dict, settings)
    
    print(f"\n{'='*80}")
    print(f"✅ DISCOVERY COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Results saved to: {run_dir}")
    print(f"   - Pareto front: {run_dir}/pareto_front_elv.csv")
    print(f"   - Forecast slate: {run_dir}/forecast_slate.csv")
    print(f"   - Config: {run_dir}/config.json")
    print(f"\n🎯 Check the forecast slate for today's trade recommendations!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
