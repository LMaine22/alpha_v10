from __future__ import annotations
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from alpha_discovery.config import settings
from alpha_discovery.search import nsga as nsga_mod
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import compile_signals
from alpha_discovery.reporting.artifacts import save_results # Will be enhanced in Step 7
from alpha_discovery.core.splits import create_hybrid_splits, HybridSplits
from alpha_discovery.eval.validation import run_full_pipeline
from alpha_discovery.eval.elv import calculate_elv_and_labels
from alpha_discovery.eval.regime import RegimeModel
from alpha_discovery.reporting.tradeable_setups import write_tradeable_setups


# Try to import the loader to build parquet if missing
try:
    from alpha_discovery.data.loader import convert_excel_to_parquet
except Exception:
    convert_excel_to_parquet = None

# ------------------------------------------------------------------
# Optional: de-noise console. Comment out any of these if you prefer.
# ------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message='Field name "validate" in "Settings" shadows an attribute',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r".*alpha_.*\.data\.events",
)
warnings.filterwarnings(
    "ignore",
    message="All-NaN slice encountered",
    category=RuntimeWarning,
)

# Thread caps for reproducibility
os.environ.update(
    VECLIB_MAXIMUM_THREADS='1',
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def load_data() -> pd.DataFrame:
    # If parquet missing but Excel exists and loader available, build it
    pq = settings.data.parquet_file_path
    if not os.path.exists(pq) and convert_excel_to_parquet is not None:
        convert_excel_to_parquet()
    if not os.path.exists(pq):
        print(f"[warn] Parquet not found: {pq}")
        return pd.DataFrame()
    df = pd.read_parquet(pq)
    # normalize index to DatetimeIndex
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # clip date range
    df = df.loc[(df.index.date >= settings.data.start_date) & (df.index.date <= settings.data.end_date)]
    return df


def build_signals(master_df: pd.DataFrame):
    # Build feature matrix with your registry (events included, leak-safe shifts inside)
    X = build_feature_matrix(master_df)
    # Compile primitive signals from features
    signals_df, signals_meta = compile_signals(X)
    # Ensure boolean dtype & drop ultra-sparse signals (unfireable setups)
    fires = {}
    for c in list(signals_df.columns):
        if signals_df[c].dtype != bool:
            signals_df[c] = signals_df[c].astype(bool)
        fires[c] = int(signals_df[c].sum())
    min_fires = int(settings.validation.min_signal_fires)
    keep_cols = [c for c in signals_df.columns if fires.get(c, 0) >= min_fires]
    dropped = len(signals_df.columns) - len(keep_cols)
    if dropped:
        print(f"[signals] Dropping {dropped} ultra-sparse signals (<{min_fires} fires)")
    signals_df = signals_df[keep_cols]
    # prune metadata to match (best-effort if dicts with 'id')
    try:
        signals_meta = [m for m in signals_meta if m.get('signal_id') in keep_cols]
    except Exception:
        pass
    return signals_df, signals_meta




# -----------------------------
# Evolution + Reporting
# -----------------------------

def create_run_dir() -> str:
    """Creates a unique run directory based on current time and seed."""
    rd_base = settings.reporting.runs_dir
    _ensure_dir(rd_base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(rd_base, f"pivot_forecast_seed{settings.ga.seed}_{ts}")
    _ensure_dir(run_dir)
    return run_dir


def discovery_phase(splits: HybridSplits, signals_df: pd.DataFrame, signals_meta: List[Dict], master_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], Optional[RegimeModel]]:
    """Runs the GA over each discovery fold and returns unique candidates and all fold results."""
    if settings.run_mode not in ['discover', 'full']:
        return [], [], None

    all_fold_results = []
    for fold_num, (train_idx, _) in enumerate(splits.discovery_cv, 1):
        print(f"\n==================== RUNNING DISCOVERY FOLD {fold_num}/{len(splits.discovery_cv)} ====================")
        train_signals_df = signals_df.loc[train_idx]
        train_master_df = master_df.loc[train_idx]
        pareto_front = nsga_mod.evolve(train_signals_df, signals_meta, train_master_df)
        for sol in pareto_front:
            sol['fold'] = fold_num
        all_fold_results.extend(pareto_front)
    
    # Convert signal lists to tuples for hashing
    unique_dict = {}
    for sol in all_fold_results:
        ticker, signals = sol['individual']
        hashable_key = (ticker, tuple(signals))  # Convert list to tuple for hashing
        unique_dict[hashable_key] = sol
    unique_candidates = list(unique_dict.values())
    print(f"\n--- Discovery complete. Found {len(unique_candidates)} unique candidates across {len(splits.discovery_cv)} folds. ---")

    # (Inside discovery_phase, after CV folds have run and models are trained/aligned)
    # anchor_regime_model = # ... (logic to get the anchor model)
    # return unique_candidates, all_fold_results, anchor_regime_model
    return unique_candidates, all_fold_results, None # Placeholder for now

def main():
    print("--- Alpha Discovery v3.0: ELV Forecast Framework ---")
    master_df = load_data()
    if master_df.empty:
        print("No data; exiting.")
        return
        
    print("--- Building all signals & features ---")
    feature_matrix = build_feature_matrix(master_df) # Need features for eligibility
    signals_df, signals_meta = compile_signals(feature_matrix)

    print("\n--- Creating hybrid validation splits ---")
    splits = create_hybrid_splits(data_index=signals_df.index)

    # --- Discovery Phase ---
    unique_candidates, all_discovery_results, anchor_regime_model = discovery_phase(splits, signals_df, signals_meta, master_df)

    # --- OOS & Gauntlet Evaluation Phase ---
    pre_elv_df = run_full_pipeline(unique_candidates, all_discovery_results, splits, signals_df, master_df, feature_matrix, signals_meta)

    # Debug: Check pre_elv_df before ELV calculation
    print("\n--- Debug: pre_elv_df info ---")
    print(f"Shape: {pre_elv_df.shape}")
    print(f"Columns: {pre_elv_df.columns.tolist()}")
    print(f"Sample individual values: {pre_elv_df['individual'].head()}")
    print(f"\nNon-null counts by column:")
    null_info = pre_elv_df.notna().sum().sort_values()
    for col, count in null_info.items():
        if count == 0:
            print(f"  {col}: {count} (ALL NULL)")
        elif count < len(pre_elv_df):
            print(f"  {col}: {count} ({len(pre_elv_df) - count} null)")
    
    # Check specifically for edge metrics
    edge_cols = [col for col in pre_elv_df.columns if 'edge_' in col and '_raw' in col]
    if edge_cols:
        print(f"\nEdge metrics check ({len(edge_cols)} columns):")
        for col in edge_cols:
            non_null = pre_elv_df[col].notna().sum()
            print(f"  {col}: {non_null} non-null values")

    # --- ELV Scoring & Labeling Phase ---
    final_results_df = calculate_elv_and_labels(pre_elv_df)
    
    # --- Reporting Phase ---
    print("\n--- All folds complete. Saving final ELV-scored results. ---")
    run_dir = create_run_dir()
    
    # Save the main artifacts (pareto front, forecast slate, etc.)
    # The save_results function now returns the path to the forecast_slate.csv
    forecast_slate_path = save_results(final_results_df, signals_meta, run_dir, splits, settings, anchor_regime_model)
    
    # --- Generate Actionable Trade Report ---
    if forecast_slate_path and os.path.exists(forecast_slate_path):
        print("\n--- Generating Actionable Trade Report ---")
        forecast_df = pd.read_csv(forecast_slate_path)
        end_of_data_date = master_df.index.max()
        write_tradeable_setups(forecast_df, end_of_data_date, run_dir)
    else:
        print("\n--- Skipping Actionable Trade Report (forecast slate not found) ---")

    print(f"\nSaved final aggregated results to: {run_dir}")


if __name__ == "__main__":
    main()
