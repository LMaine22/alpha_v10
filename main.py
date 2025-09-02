# main.py
import os
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
from alpha_discovery.signals.compiler import compile_signals
from alpha_discovery.search.nsga import evolve
from alpha_discovery.reporting.artifacts import save_results, materialize_per_fold_artifacts
from alpha_discovery.eval.validation import create_walk_forward_splits
from alpha_discovery.engine.bt_common import TRADE_HORIZONS_DAYS


def run_pipeline():
    """
    TRAIN-only pipeline followed by a true out-of-sample Gauntlet validation.
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

    # ===== Walk-forward training across folds =====
    for i, (train_idx, test_idx) in enumerate(splits):
        fold_num = i + 1
        print(f"\n==================== RUNNING FOLD {fold_num}/{len(splits)} ====================")
        print(f"Training Period: {train_idx.min().date()} to {train_idx.max().date()}")
        print(f"Testing  Period: {test_idx.min().date()} to {test_idx.max().date()}")

        train_master_df = master_df.loc[train_idx]
        train_signals_df = signals_df.reindex(train_idx).fillna(False)

        pareto_front_for_fold = evolve(train_signals_df, signals_metadata, train_master_df)
        for solution in pareto_front_for_fold:
            solution["fold"] = fold_num

        all_fold_results.extend(pareto_front_for_fold)

    print(f"\n{'=' * 20} WALK-FORWARD (TRAIN artifacts) COMPLETE {'=' * 20}")

    print("\n--- Saving In-Sample Training Results ---")
    save_results(all_fold_results, signals_metadata, settings, output_dir=output_dir)
    print("Global results saved.")

    try:
        materialize_per_fold_artifacts(base_dir=output_dir)
        print("Per-fold TRAIN artifacts materialized.")
    except Exception as e:
        print(f"Warning: could not materialize per-fold artifacts: {e}")

    # ===== Gauntlet: true OOS validation =====
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


if __name__ == '__main__':
    print("--- Starting Full Alpha Discovery Pipeline ---")
    run_pipeline()
    print("\n--- Pipeline Finished ---")
