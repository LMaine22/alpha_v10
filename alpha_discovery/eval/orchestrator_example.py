"""
Example usage of the ForecastOrchestrator for validation.

This shows how to use the new forecast-first validation workflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from alpha_discovery.eval.orchestrator import ForecastOrchestrator, EligibilityMatrix
from alpha_discovery.adapters import FeatureAdapter


def run_validation_example(
    master_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    discovered_setups: list,
    output_dir: Path
):
    """
    Complete validation workflow example.
    
    Args:
        master_df: DataFrame with price/market data
        signals_df: DataFrame with signal triggers
        discovered_setups: List of (ticker, signals) from GA discovery
        output_dir: Directory for validation artifacts
    
    Returns:
        EligibilityMatrix with validation results
    """
    print("=" * 80)
    print("FORECAST-FIRST VALIDATION WORKFLOW")
    print("=" * 80)
    
    # Step 1: Initialize orchestrator
    print("\n[1] Initializing orchestrator...")
    orchestrator = ForecastOrchestrator(
        df=master_df,
        signals_df=signals_df,
        feature_adapter=FeatureAdapter()  # Optional
    )
    
    # Step 2: Run validation
    print(f"\n[2] Validating {len(discovered_setups)} discovered setups...")
    eligibility = orchestrator.run_validation(
        discovered_setups=discovered_setups,
        output_dir=output_dir,
        n_jobs=-1
    )
    
    # Step 3: Analyze results
    print("\n[3] Analyzing results...")
    df_results = eligibility.to_dataframe()
    
    print(f"\nTotal validations: {len(df_results)}")
    print(f"Mean CRPS: {df_results['crps'].mean():.4f}")
    print(f"Mean skill vs marginal: {df_results['skill_vs_marginal'].mean():.4f}")
    print(f"Drift pass rate: {df_results['drift_passed'].mean():.1%}")
    
    # Step 4: Filter to eligible setups
    print("\n[4] Filtering to eligible setups...")
    eligible = eligibility.filter_eligible(
        min_skill_vs_marginal=0.01,  # Must beat marginal by 1% CRPS
        max_calibration_mae=0.15,    # Max 15% calibration error
        drift_gate=True               # Must pass drift test
    )
    
    print(f"Eligible setups: {len(eligible)}/{len(df_results)}")
    
    # Step 5: Show top setups
    if eligible:
        print("\n[5] Top 5 eligible setups by skill:")
        eligible_df = pd.DataFrame([e.to_dict() for e in eligible])
        top5 = eligible_df.nlargest(5, 'skill_vs_marginal')
        
        for idx, row in top5.iterrows():
            print(f"\n  {row['ticker']} | H:{row['horizon']} | Signals: {len(row['setup'])}")
            print(f"    Skill vs marginal: {row['skill_vs_marginal']:.4f}")
            print(f"    CRPS: {row['crps']:.4f}")
            print(f"    Calibration MAE: {row['calibration_mae']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"Artifacts saved to: {output_dir}")
    print("=" * 80)
    
    return eligibility


def load_and_analyze_results(eligibility_path: Path):
    """
    Load and analyze saved eligibility matrix.
    
    Args:
        eligibility_path: Path to saved eligibility_matrix.json
    """
    print(f"Loading eligibility matrix from {eligibility_path}...")
    eligibility = EligibilityMatrix.load(eligibility_path)
    
    df = eligibility.to_dataframe()
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("ELIGIBILITY MATRIX ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal validations: {len(df)}")
    print(f"Unique setups: {df.groupby(['ticker', 'horizon']).ngroups}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    
    # Metrics distribution
    print("\nMetrics Distribution:")
    print(df[['crps', 'skill_vs_marginal', 'calibration_mae', 'drift_auc']].describe())
    
    # By ticker
    print("\nBy Ticker:")
    ticker_stats = df.groupby('ticker').agg({
        'crps': 'mean',
        'skill_vs_marginal': 'mean',
        'calibration_mae': 'mean',
        'drift_passed': 'mean'
    }).round(4)
    print(ticker_stats)
    
    # Regime analysis
    print("\nRegime Distribution:")
    print(df['regime_test'].value_counts())
    
    # Eligibility
    eligible = eligibility.filter_eligible()
    print(f"\nEligible setups: {len(eligible)}/{len(df)} ({len(eligible)/len(df):.1%})")
    
    return eligibility


# Example: Minimal synthetic data
if __name__ == "__main__":
    # Create synthetic data for demonstration
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
    
    # Master DataFrame with prices
    master_df = pd.DataFrame({
        'AAPL_PX_LAST': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01),
        'SPY_PX_LAST': 300 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    }, index=dates)
    
    # Signals DataFrame
    signals_df = pd.DataFrame({
        'signal_mom_positive': np.random.rand(len(dates)) > 0.7,
        'signal_vol_low': np.random.rand(len(dates)) > 0.8
    }, index=dates)
    
    # Discovered setups from GA (example)
    discovered_setups = [
        ('AAPL', ['signal_mom_positive']),
        ('AAPL', ['signal_mom_positive', 'signal_vol_low']),
    ]
    
    # Run validation
    output_dir = Path('runs/validation_example')
    eligibility = run_validation_example(
        master_df=master_df,
        signals_df=signals_df,
        discovered_setups=discovered_setups,
        output_dir=output_dir
    )
    
    # Load and analyze
    print("\n\nReloading saved results...")
    load_and_analyze_results(output_dir / "eligibility_matrix.json")
