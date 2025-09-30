"""
Example usage of eligibility reporting for forecast-first validation.

Shows how to generate comprehensive reports from EligibilityMatrix artifacts.
"""

from pathlib import Path
from alpha_discovery.reporting.eligibility_report import (
    generate_eligibility_report,
    print_eligibility_summary,
    export_reliability_data
)


def example_generate_reports():
    """
    Example: Generate all reports from an eligibility matrix.
    
    Assumes you've already run validation and have:
    - runs/validation_001/eligibility_matrix.json
    """
    # Paths
    validation_dir = Path("runs/validation_001")
    eligibility_path = validation_dir / "eligibility_matrix.json"
    reports_dir = validation_dir / "reports"
    
    print("=" * 80)
    print("GENERATING ELIGIBILITY REPORTS")
    print("=" * 80)
    
    # Generate all reports
    print("\n[1] Generating comprehensive reports...")
    outputs = generate_eligibility_report(
        eligibility_matrix_path=eligibility_path,
        output_dir=reports_dir,
        min_skill_vs_marginal=0.01,  # Must beat marginal by 1% CRPS
        max_calibration_mae=0.15,     # Max 15% calibration error
        drift_gate=True,               # Enforce drift test
        top_n=50                       # Top 50 setups
    )
    
    print(f"\n✓ Generated {len(outputs)} report files:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")
    
    # Print summary to console
    print("\n[2] Printing summary...")
    print_eligibility_summary(outputs['summary'])
    
    # Export reliability data for visualization
    print("\n[3] Exporting reliability curve data...")
    reliability_path = reports_dir / "reliability_curves.csv"
    export_reliability_data(
        eligibility_matrix_path=eligibility_path,
        output_path=reliability_path,
        top_n=20
    )
    print(f"✓ Exported to: {reliability_path}")
    
    print("\n" + "=" * 80)
    print("REPORTS COMPLETE")
    print("=" * 80)
    
    return outputs


def example_load_and_filter():
    """
    Example: Load eligibility matrix and apply custom filtering.
    """
    import pandas as pd
    import json
    
    # Load eligibility matrix
    eligibility_path = Path("runs/validation_001/eligibility_matrix.json")
    
    with open(eligibility_path, 'r') as f:
        data = json.load(f)
    
    results_df = pd.DataFrame(data['results'])
    
    # Custom filtering logic
    print("\n📊 Custom Filtering Example:")
    
    # Filter 1: High-skill, well-calibrated setups
    high_quality = results_df[
        (results_df['skill_vs_marginal'] >= 0.02) &  # Strong skill
        (results_df['calibration_mae'] <= 0.10) &     # Excellent calibration
        (results_df['drift_passed'] == True)          # Passed drift test
    ]
    
    print(f"\n  High-quality setups: {len(high_quality)}/{len(results_df)}")
    
    # Filter 2: By specific ticker
    ticker_setups = results_df[results_df['ticker'] == 'AAPL']
    print(f"  AAPL setups: {len(ticker_setups)}")
    
    # Filter 3: By horizon
    short_horizon = results_df[results_df['horizon'] <= 5]
    print(f"  Short-horizon (<= 5d) setups: {len(short_horizon)}")
    
    # Top 10 by skill
    top10 = results_df.nlargest(10, 'skill_vs_marginal')
    
    print(f"\n🏆 Top 10 Setups by Skill:")
    for i, row in top10.iterrows():
        print(f"  {row['ticker']} H{row['horizon']}: "
              f"skill={row['skill_vs_marginal']:.4f}, "
              f"CRPS={row['crps']:.4f}, "
              f"calib_mae={row['calibration_mae']:.4f}")
    
    return results_df


def example_multi_run_comparison():
    """
    Example: Compare eligibility across multiple validation runs.
    """
    import pandas as pd
    import json
    
    runs = [
        Path("runs/validation_001/eligibility_matrix.json"),
        Path("runs/validation_002/eligibility_matrix.json"),
    ]
    
    comparison = []
    
    for run_path in runs:
        if not run_path.exists():
            continue
        
        with open(run_path, 'r') as f:
            data = json.load(f)
        
        results_df = pd.DataFrame(data['results'])
        
        comparison.append({
            'run': run_path.parent.name,
            'total_validations': len(results_df),
            'mean_skill': results_df['skill_vs_marginal'].mean(),
            'mean_crps': results_df['crps'].mean(),
            'mean_calib_mae': results_df['calibration_mae'].mean(),
            'drift_pass_rate': results_df['drift_passed'].mean()
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\n📊 Multi-Run Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


# Main example
if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ELIGIBILITY REPORTING EXAMPLES                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script demonstrates how to:
1. Generate comprehensive reports from EligibilityMatrix
2. Load and filter validation results
3. Compare multiple validation runs

Note: This assumes you have already run validation and have 
      eligibility_matrix.json files in your runs/ directory.
    """)
    
    # Check if example data exists
    example_path = Path("runs/validation_001/eligibility_matrix.json")
    
    if not example_path.exists():
        print(f"⚠️  Example data not found at: {example_path}")
        print("\nTo generate example data:")
        print("  1. Run the orchestrator: see alpha_discovery/eval/orchestrator_example.py")
        print("  2. Or create synthetic eligibility matrix")
        sys.exit(0)
    
    # Run examples
    try:
        # Example 1: Generate reports
        outputs = example_generate_reports()
        
        # Example 2: Custom filtering
        results_df = example_load_and_filter()
        
        # Example 3: Multi-run comparison (if multiple runs exist)
        # comparison = example_multi_run_comparison()
        
    except FileNotFoundError as e:
        print(f"\n⚠️  File not found: {e}")
        print("Make sure you've run validation first!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
