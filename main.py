from __future__ import annotations
import os
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings but keep errors visible
warnings.simplefilter('ignore')

# Suppress specific numpy warnings that occur during feature computation in pandas/numpy operations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._function_base_impl')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy._core._methods')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy.lib._nanfunctions_impl')
# Suppress joblib memory warnings
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*Persisting input arguments.*')
# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='All-NaN slice encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
# Suppress specific warnings from our modules
warnings.filterwarnings('ignore', category=RuntimeWarning, module='alpha_discovery.search.ga_core')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='alpha_discovery.eval.metrics.robustness')

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

# Set numpy error handling to ignore warnings
np.seterr(all='ignore')

from alpha_discovery.config import settings
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import compile_signals


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

# Thread caps for reproducibility + performance tuning
os.environ.update(
    VECLIB_MAXIMUM_THREADS='4',  # Allow more threads for numpy
    OMP_NUM_THREADS='4',
    OPENBLAS_NUM_THREADS='4',
    MKL_NUM_THREADS='4',
    NUMEXPR_NUM_THREADS='4',
    # Feature performance tuning
    FEATURES_PAIRWISE_MAX_TICKERS='32',  # Limit pairwise to 32 tickers
    FEATURES_PAIRWISE_JOBS='8',  # Use 8 parallel jobs
    PAIRWISE_MIN_PERIODS='10',   # Reduce min periods for speed
)


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def load_data() -> pd.DataFrame:
    # Always refresh the parquet from Excel to ensure we have the latest data
    pq = settings.data.parquet_file_path
    excel = settings.data.excel_file_path
    
    # Check if Excel file exists
    if not os.path.exists(excel):
        print(f"[error] Excel file not found: {excel}")
        return pd.DataFrame()
    
    # Always convert Excel to Parquet to ensure fresh data
    if convert_excel_to_parquet is not None:
        print(f"[info] Refreshing parquet data from Excel source: {excel}")
        convert_excel_to_parquet()
    else:
        print(f"[warn] Excel conversion function not available. Using existing parquet if available.")
    
    # Check if Parquet exists after potential conversion
    if not os.path.exists(pq):
        print(f"[error] Parquet not found: {pq}")
        return pd.DataFrame()
    
    # Load the parquet data
    print(f"[info] Loading data from parquet: {pq}")
    df = pd.read_parquet(pq)
    
    # normalize index to DatetimeIndex
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.set_index('DATE')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # clip date range according to config
    date_filtered_df = df.loc[(df.index.date >= settings.data.start_date) & (df.index.date <= settings.data.end_date)]
    
    # Print data range info
    print(f"[info] Data loaded with date range: {date_filtered_df.index.min().date()} to {date_filtered_df.index.max().date()}")
    print(f"[info] Config date range: {settings.data.start_date} to {settings.data.end_date}")
    
    return date_filtered_df


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
    run_dir = os.path.join(rd_base, f"forecast_first_seed{settings.ga.seed}_{ts}")
    _ensure_dir(run_dir)
    return run_dir


def print_header():
    """Prints a stylized header to the console."""
    print("=" * 80)
    print("         ALPHA DISCOVERY ENGINE v10 - Forecast-First")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration Seed: {settings.ga.seed}")
    print("-" * 80)


def main():
    """Forecast-first discovery and validation workflow."""
    print_header()
    
    # Run the forecast-first workflow
    main_discover_workflow()


def main_discover_workflow():
    """
    Forecast-first discovery mode.
    
    Runs:
    1. Load data and build features
    2. Create PAWF splits for outer validation
    3. Run GA with NPWF inner folds
    4. Validate all candidates on outer PAWF folds
    5. Generate EligibilityMatrix with skill/calibration/drift metrics
    6. Save eligibility matrix for selection phase
    """
    # Header already printed by main()
    
    # Load data
    master_df = load_data()
    if master_df.empty:
        print("No data; exiting.")
        return
    
    print("--- Building features and signals ---")
    feature_matrix = build_feature_matrix(master_df)
    signals_df, signals_meta = compile_signals(feature_matrix)
    
    # Create run directory
    run_dir = create_run_dir()
    print(f"\n--- Run directory: {run_dir} ---")
    
    # Import forecast-first orchestrator
    from alpha_discovery.eval.orchestrator import ForecastOrchestrator
    from alpha_discovery.adapters.features import FeatureAdapter, calculate_max_lookback
    
    print("\n--- Initializing Forecast-First Orchestrator ---")
    
    # Setup feature adapter
    adapter = FeatureAdapter()
    lookback_tail = calculate_max_lookback(adapter.features, adapter.pairwise)
    print(f"Feature lookback tail: {lookback_tail} days")
    
    # Create orchestrator
    orchestrator = ForecastOrchestrator(
        master_df=master_df,
        signals_df=signals_df,
        signals_meta=signals_meta,
        output_dir=Path(run_dir),
        feature_lookback_tail=lookback_tail,
        seed=settings.ga.seed
    )
    
    # Run validation with PAWF + NPWF
    print("\n--- Running PAWF validation with NPWF inner folds ---")
    eligibility_matrix = orchestrator.run_validation(
        n_outer_splits=4,           # PAWF outer splits
        test_size_months=6,          # 6-month test windows
        purge_days=5,                # 5-day purge
        n_inner_folds=3,             # NPWF inner folds for GA
        n_regimes=5,                 # GMM regimes
        run_ga=True,                 # Run GA for discovery
        n_ga_generations=settings.ga.generations,
        ga_population=settings.ga.population_size
    )
    
    # Generate reports
    print("\n--- Generating Eligibility Reports ---")
    from alpha_discovery.reporting.eligibility_report import (
        generate_eligibility_report,
        print_eligibility_summary
    )
    
    reports_dir = Path(run_dir) / "reports"
    eligibility_path = Path(run_dir) / "eligibility_matrix.json"
    
    report_outputs = generate_eligibility_report(
        eligibility_matrix_path=eligibility_path,
        output_dir=reports_dir,
        min_skill_vs_marginal=0.01,
        max_calibration_mae=0.15,
        drift_gate=True,
        top_n=50
    )
    
    # Print summary
    print_eligibility_summary(report_outputs['summary'])
    
    print(f"\n--- Discovery Complete ---")
    print(f"Eligibility matrix: {eligibility_path}")
    print(f"Reports directory: {reports_dir}")
    print(f"\nNext step: Run with --mode select --eligibility {eligibility_path}")


def main_select(eligibility_path: Optional[Path] = None):
    """
    Selection mode: Load eligibility matrix and generate forecast slate.
    
    Runs:
    1. Load eligibility matrix
    2. Filter by skill/calibration/drift thresholds
    3. Rank by skill_vs_marginal
    4. Apply portfolio constraints (max per ticker, diversification)
    5. Generate actionable forecast slate
    """
    print_header(mode="select")
    
    # Find eligibility matrix
    if eligibility_path is None:
        # Look for most recent run
        runs_dir = Path(settings.reporting.runs_dir)
        if not runs_dir.exists():
            print(f"No runs directory found: {runs_dir}")
            return
        
        # Find most recent eligibility matrix
        eligible_runs = list(runs_dir.glob("*/eligibility_matrix.json"))
        if not eligible_runs:
            print(f"No eligibility matrices found in {runs_dir}")
            print("Run with --mode discover first!")
            return
        
        eligibility_path = max(eligible_runs, key=lambda p: p.stat().st_mtime)
        print(f"Using most recent eligibility matrix: {eligibility_path}")
    
    if not eligibility_path.exists():
        print(f"Eligibility matrix not found: {eligibility_path}")
        return
    
    # Load eligibility matrix
    import json
    import pandas as pd
    
    with open(eligibility_path, 'r') as f:
        data = json.load(f)
    
    results_df = pd.DataFrame(data['results'])
    metadata = data['metadata']
    
    print(f"\n--- Loaded {len(results_df)} validated setups ---")
    print(f"Validation date: {metadata.get('timestamp', 'unknown')}")
    
    # Apply selection criteria
    print("\n--- Applying Selection Criteria ---")
    
    MIN_SKILL = 0.01  # Must beat marginal by 1% CRPS
    MAX_CALIB_MAE = 0.15  # Max 15% calibration error
    DRIFT_GATE = True
    MAX_PER_TICKER = 2  # Max 2 setups per ticker
    TOP_N = 20  # Final portfolio size
    
    eligible = results_df[
        (results_df['skill_vs_marginal'] >= MIN_SKILL) &
        (results_df['calibration_mae'] <= MAX_CALIB_MAE)
    ]
    
    if DRIFT_GATE:
        eligible = eligible[eligible['drift_passed'] == True]
    
    print(f"  Eligible setups: {len(eligible)}/{len(results_df)}")
    
    # Rank by skill
    eligible = eligible.sort_values('skill_vs_marginal', ascending=False)
    
    # Apply max-per-ticker constraint
    selected = []
    ticker_counts = {}
    
    for idx, row in eligible.iterrows():
        ticker = row['ticker']
        if ticker_counts.get(ticker, 0) < MAX_PER_TICKER:
            selected.append(row)
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        
        if len(selected) >= TOP_N:
            break
    
    final_df = pd.DataFrame(selected)
    
    print(f"  Final portfolio: {len(final_df)} setups across {len(ticker_counts)} tickers")
    
    # Generate forecast slate
    output_dir = eligibility_path.parent / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    forecast_slate_path = output_dir / "forecast_slate.csv"
    final_df.to_csv(forecast_slate_path, index=False)
    
    print(f"\n--- Forecast Slate Generated ---")
    print(f"Output: {forecast_slate_path}")
    
    # Print top 10
    print(f"\n🏆 Top 10 Setups:")
    for i, row in final_df.head(10).iterrows():
        print(f"  {i+1}. {row['ticker']} H{row['horizon']}: "
              f"skill={row['skill_vs_marginal']:.4f}, "
              f"CRPS={row['crps']:.4f}, "
              f"calib_mae={row['calibration_mae']:.4f}")
    
    print(f"\n--- Selection Complete ---")


# Simple CLI: Run forecast-first discovery
if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
