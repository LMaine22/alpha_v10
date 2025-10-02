#!/usr/bin/env python3
"""
Validation script to test that all pipeline fixes are working correctly.
Run this after applying the fixes to check coverage improvements.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alpha_discovery.search.ga_core import _calculate_robust_replacement_metrics


def validate_pipeline_fixes(
    final_results_df: pd.DataFrame,
    run_dir: str = "runs"
) -> None:
    """
    Validation script to verify all fixes are working correctly.
    Run this after applying the fixes to check coverage improvements.
    """
    print("\n=== PIPELINE FIXES VALIDATION ===")
    
    # 1. Check metric coverage improvements
    key_metrics = [
        "edge_crps_raw", "edge_pin_q10_raw", "edge_pin_q90_raw", "edge_ig_raw", 
        "edge_w1_raw", "edge_calib_mae_raw", "sensitivity_delta_edge_raw",
        "bootstrap_p_value_raw", "redundancy_mi_raw", "complexity_metric_raw",
        "transfer_entropy", "tr_cv_reg", "tr_fg", "n_trig_oos"
    ]
    
    print("METRIC COVERAGE ANALYSIS:")
    coverage_results = {}
    
    for metric in key_metrics:
        if metric in final_results_df.columns:
            non_null = final_results_df[metric].notna().sum()
            total = len(final_results_df)
            coverage_pct = (non_null / total) * 100
            coverage_results[metric] = coverage_pct
            
            # Expected improvements
            expected_min = {
                'edge_crps_raw': 70,
                'redundancy_mi_raw': 90,
                'tr_cv_reg': 80,
                'tr_fg': 90,
                'complexity_metric_raw': 85
            }.get(metric, 50)
            
            status = "✓ GOOD" if coverage_pct >= expected_min else "✗ LOW"
            print(f"  {metric}: {coverage_pct:.1f}% ({non_null}/{total}) - {status}")
        else:
            print(f"  {metric}: MISSING COLUMN")
            coverage_results[metric] = 0
    
    # 2. Check for robust metric alternatives
    print("\nROBUST ALTERNATIVES CHECK:")
    robust_alternatives = [
        'volatility_persistence', 'sample_entropy_robust', 'permutation_entropy_robust',
        'tscv_mean_stability', 'h0_total_persistence', 'regime_stability', 'composite_robustness'
    ]
    
    for alt_metric in robust_alternatives:
        if alt_metric in final_results_df.columns:
            non_null = final_results_df[alt_metric].notna().sum()
            total = len(final_results_df)
            coverage_pct = (non_null / total) * 100
            print(f"  {alt_metric}: {coverage_pct:.1f}% ({non_null}/{total})")
        else:
            print(f"  {alt_metric}: NOT PRESENT (may not be implemented yet)")
    
    # 3. Check ELV component health
    print("\nELV COMPONENTS HEALTH:")
    elv_components = ['edge_oos', 'live_tr_prior', 'coverage_factor', 'penalty_adj', 'elv']
    
    for comp in elv_components:
        if comp in final_results_df.columns:
            non_null = final_results_df[comp].notna().sum()
            finite_vals = final_results_df[comp].replace([np.inf, -np.inf], np.nan).notna().sum()
            total = len(final_results_df)
            
            print(f"  {comp}: {finite_vals}/{total} finite ({finite_vals/total*100:.1f}%)")
            
            if finite_vals > 0:
                mean_val = final_results_df[comp].mean()
                print(f"    Mean: {mean_val:.4f}")
    
    # 4. Check Hart Index distribution  
    if 'hart_index' in final_results_df.columns:
        print("\nHART INDEX DISTRIBUTION:")
        hart_scores = final_results_df['hart_index'].dropna()
        
        if not hart_scores.empty:
            print(f"  Count: {len(hart_scores)}")
            print(f"  Mean: {hart_scores.mean():.1f}")
            print(f"  Median: {hart_scores.median():.1f}")
            print(f"  Max: {hart_scores.max():.1f}")
            
            # Count by label
            if 'hart_label' in final_results_df.columns:
                label_counts = final_results_df['hart_label'].value_counts()
                for label, count in label_counts.items():
                    print(f"    {label}: {count}")
    
    # 5. Simulate individual parsing test
    print("\nINDIVIDUAL PARSING TEST:")
    if 'individual' in final_results_df.columns:
        sample_individuals = final_results_df['individual'].head(3)
        
        for i, ind in enumerate(sample_individuals):
            try:
                if isinstance(ind, str):
                    parsed = eval(ind)
                elif isinstance(ind, tuple):
                    parsed = ind
                else:
                    parsed = None
                    
                # Test string normalization
                if parsed:
                    def _normalize_strings(obj):
                        if isinstance(obj, np.str_):
                            return str(obj)
                        elif isinstance(obj, tuple):
                            return tuple(_normalize_strings(x) for x in obj)
                        elif isinstance(obj, list):
                            return [_normalize_strings(x) for x in obj]
                        else:
                            return obj
                    
                    normalized = _normalize_strings(parsed)
                    print(f"  Sample {i+1}: ✓ PARSED - {type(normalized[0])}, {type(normalized[1])}")
                else:
                    print(f"  Sample {i+1}: ✗ FAILED - {type(ind)}")
                    
            except Exception as e:
                print(f"  Sample {i+1}: ✗ ERROR - {e}")
    
    # 6. Write validation report
    os.makedirs(run_dir, exist_ok=True)
    report_path = os.path.join(run_dir, "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("PIPELINE FIXES VALIDATION REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("METRIC COVERAGE:\n")
        for metric, coverage in coverage_results.items():
            f.write(f"  {metric}: {coverage:.1f}%\n")
        
        f.write(f"\nTOTAL CANDIDATES: {len(final_results_df)}\n")
        
        if 'hart_index' in final_results_df.columns:
            hart_scores = final_results_df['hart_index'].dropna()
            f.write(f"HART INDEX STATS:\n")
            f.write(f"  Mean: {hart_scores.mean():.1f}\n")
            f.write(f"  Candidates >= 55: {(hart_scores >= 55).sum()}\n")
            f.write(f"  Candidates >= 70: {(hart_scores >= 70).sum()}\n")
    
    print(f"\nValidation report saved to: {report_path}")
    
    # 7. Overall assessment
    critical_metrics = ['edge_crps_raw', 'redundancy_mi_raw', 'tr_cv_reg', 'tr_fg']
    critical_coverage = [coverage_results.get(m, 0) for m in critical_metrics]
    avg_critical_coverage = np.mean(critical_coverage)
    
    print(f"\nOVERALL ASSESSMENT:")
    if avg_critical_coverage >= 75:
        print("✓ EXCELLENT - Critical metrics well covered")
    elif avg_critical_coverage >= 50:
        print("⚠ GOOD - Most critical metrics covered, some room for improvement")  
    else:
        print("✗ NEEDS WORK - Many critical metrics still sparse")
    
    print(f"Average critical metric coverage: {avg_critical_coverage:.1f}%")


def quick_test_fixes():
    """Quick test of individual components during development."""
    
    # Test metric replacement logic
    print("Testing robust metric replacements...")
    
    # Create sample data
    np.random.seed(42)
    sample_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    sample_dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_returns.index = sample_dates
    
    # Create trigger dates (random subset)
    trigger_indices = np.random.choice(100, size=20, replace=False)
    trigger_dates = sample_dates[trigger_indices]
    
    unconditional_returns = {4: sample_returns}
    
    # Test robust replacements
    try:
        robust_metrics = _calculate_robust_replacement_metrics(
            trigger_dates, unconditional_returns, [4], 
            np.array([-0.1, -0.05, 0, 0.05, 0.1])
        )
        print("✓ Robust metrics calculation works")
        for key, value in robust_metrics.items():
            if np.isfinite(value):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Robust metrics failed: {e}")
    
    print("Quick test complete.")


def validate_latest_run():
    """Validate the most recent run automatically."""
    
    # Find the latest run directory
    runs_dir = os.path.join(project_root, 'runs')
    if not os.path.exists(runs_dir):
        print("No runs directory found")
        return
    
    # Get the latest run folder
    run_folders = [f for f in os.listdir(runs_dir) if f.startswith('pivot_forecast_')]
    if not run_folders:
        print("No forecast run folders found")
        return
    
    latest_run = sorted(run_folders)[-1]
    latest_run_path = os.path.join(runs_dir, latest_run)
    
    # Look for the main results CSV
    possible_files = [
        'final_results.csv',
        'consolidated_results.csv', 
        'hart_index_results.csv'
    ]
    
    results_file = None
    for filename in possible_files:
        filepath = os.path.join(latest_run_path, filename)
        if os.path.exists(filepath):
            results_file = filepath
            break
    
    if not results_file:
        print(f"No results file found in {latest_run_path}")
        print("Available files:", os.listdir(latest_run_path))
        return
    
    print(f"Validating results from: {results_file}")
    
    try:
        results_df = pd.read_csv(results_file)
        validate_pipeline_fixes(results_df, latest_run_path)
    except Exception as e:
        print(f"Error loading results: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate pipeline fixes')
    parser.add_argument('--quick', action='store_true', help='Run quick component test')
    parser.add_argument('--latest', action='store_true', help='Validate latest run automatically')
    parser.add_argument('--file', type=str, help='Path to specific results CSV file')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test_fixes()
    elif args.latest:
        validate_latest_run()
    elif args.file:
        try:
            results_df = pd.read_csv(args.file)
            validate_pipeline_fixes(results_df, os.path.dirname(args.file))
        except Exception as e:
            print(f"Error loading file {args.file}: {e}")
    else:
        # Default: run quick test
        print("Running quick test. Use --latest to validate latest run or --file <path> for specific file.")
        quick_test_fixes()
