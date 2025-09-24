#!/usr/bin/env python3
"""
Hart Index Audit Script - Find problematic values affecting the calculation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from alpha_discovery.eval.hart_index import calculate_hart_index

def audit_hart_index_data(csv_path: str):
    """Audit Hart Index calculation to find problematic values."""
    
    print("=== HART INDEX AUDIT ===\n")
    
    # Load the data
    df = pd.read_csv(csv_path)
    print(f"Loaded data with {len(df)} setups and {len(df.columns)} columns")
    
    # Key metrics used in Hart Index calculation
    key_metrics = [
        'edge_crps_raw', 'edge_pin_q10_raw', 'edge_pin_q90_raw', 'edge_w1_raw', 
        'edge_calib_mae_raw', 'edge_ig_raw', 'E_move', 'P_up', 'P_down',
        'bootstrap_p_value_raw', 'n_trig_oos', 'sensitivity_delta_edge_raw',
        'redundancy_mi_raw', 'complexity_metric_raw', 'dfa_alpha_raw', 
        'transfer_entropy_raw', 'live_tr_prior', 'coverage_factor',
        'page_hinkley_alarm', 'elv', 'hart_index'
    ]
    
    print("\n=== COLUMN AVAILABILITY CHECK ===")
    missing_metrics = []
    available_metrics = []
    
    for metric in key_metrics:
        if metric in df.columns:
            available_metrics.append(metric)
            print(f"✅ {metric}")
        else:
            missing_metrics.append(metric)
            print(f"❌ {metric} - MISSING")
    
    print(f"\nAvailable: {len(available_metrics)}/{len(key_metrics)} metrics")
    
    if missing_metrics:
        print(f"\nMISSING METRICS:")
        for metric in missing_metrics:
            print(f"  - {metric}")
    
    print("\n=== VALUE ANALYSIS FOR AVAILABLE METRICS ===")
    
    problematic_metrics = {}
    
    for metric in available_metrics:
        if metric in df.columns:
            series = df[metric]
            
            n_total = len(series)
            n_nan = series.isna().sum()
            n_zero = (series == 0.0).sum()
            n_inf = np.isinf(series).sum()
            n_negative = (series < 0).sum() if metric not in ['E_move'] else 0  # E_move can be negative
            
            mean_val = series.mean()
            median_val = series.median()
            min_val = series.min()
            max_val = series.max()
            std_val = series.std()
            
            print(f"\n📊 {metric}:")
            print(f"  Range: {min_val:.4f} to {max_val:.4f}")
            print(f"  Mean: {mean_val:.4f}, Median: {median_val:.4f}, Std: {std_val:.4f}")
            print(f"  NaN values: {n_nan} ({n_nan/n_total*100:.1f}%)")
            print(f"  Zero values: {n_zero} ({n_zero/n_total*100:.1f}%)")
            print(f"  Infinite values: {n_inf} ({n_inf/n_total*100:.1f}%)")
            
            if metric not in ['E_move', 'page_hinkley_alarm']:  # These can legitimately be negative/zero
                print(f"  Negative values: {n_negative} ({n_negative/n_total*100:.1f}%)")
            
            # Flag problematic metrics
            if n_nan > n_total * 0.1:  # More than 10% NaN
                problematic_metrics[metric] = problematic_metrics.get(metric, []) + ['High NaN rate']
            
            if n_zero > n_total * 0.5 and metric not in ['page_hinkley_alarm']:  # More than 50% zeros
                problematic_metrics[metric] = problematic_metrics.get(metric, []) + ['High zero rate']
                
            if n_inf > 0:
                problematic_metrics[metric] = problematic_metrics.get(metric, []) + ['Contains infinite values']
            
            if std_val == 0 or pd.isna(std_val):
                problematic_metrics[metric] = problematic_metrics.get(metric, []) + ['No variance (constant values)']
    
    print("\n=== PROBLEMATIC METRICS SUMMARY ===")
    if problematic_metrics:
        for metric, issues in problematic_metrics.items():
            print(f"\n🚨 {metric}:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        print("✅ No major issues detected in available metrics")
    
    print("\n=== HART INDEX DISTRIBUTION ===")
    if 'hart_index' in df.columns:
        hart_scores = df['hart_index']
        print(f"Hart Index Statistics:")
        print(f"  Count: {len(hart_scores)}")
        print(f"  Mean: {hart_scores.mean():.1f}")
        print(f"  Median: {hart_scores.median():.1f}")
        print(f"  Std: {hart_scores.std():.1f}")
        print(f"  Min: {hart_scores.min():.1f}")
        print(f"  Max: {hart_scores.max():.1f}")
        print(f"  25th percentile: {hart_scores.quantile(0.25):.1f}")
        print(f"  75th percentile: {hart_scores.quantile(0.75):.1f}")
        
        # Distribution by ranges
        print(f"\nDistribution:")
        print(f"  Exceptional (≥85): {(hart_scores >= 85).sum()} setups")
        print(f"  Strong (70-84): {((hart_scores >= 70) & (hart_scores < 85)).sum()} setups")
        print(f"  Moderate (55-69): {((hart_scores >= 55) & (hart_scores < 70)).sum()} setups")
        print(f"  Marginal (40-54): {((hart_scores >= 40) & (hart_scores < 55)).sum()} setups")
        print(f"  Weak (<40): {(hart_scores < 40).sum()} setups")
    else:
        print("❌ Hart Index column not found")
    
    print("\n=== ALTERNATIVE METRICS AVAILABLE ===")
    print("Looking for alternative metrics in the dataset...")
    
    # Look for alternative metrics that might be better
    alternative_patterns = [
        'sharpe', 'sortino', 'calmar', 'max_drawdown', 'win_rate', 
        'avg_win', 'avg_loss', 'profit_factor', 'recovery_factor',
        'var_', 'cvar_', 'skewness', 'kurtosis', 'alpha', 'beta',
        'correlation', 'information_ratio', 'tracking_error'
    ]
    
    alternative_metrics = []
    for col in df.columns:
        for pattern in alternative_patterns:
            if pattern.lower() in col.lower():
                alternative_metrics.append(col)
                break
    
    if alternative_metrics:
        print("Found alternative metrics:")
        for metric in alternative_metrics:
            series = df[metric]
            n_nan = series.isna().sum()
            n_zero = (series == 0.0).sum()
            print(f"  ✓ {metric}: {len(series)-n_nan-n_zero} usable values ({(len(series)-n_nan-n_zero)/len(series)*100:.1f}%)")
    else:
        print("No obvious alternative metrics found in current columns")
    
    print(f"\n=== RECOMMENDATIONS ===")
    if problematic_metrics:
        print("🔧 Suggested fixes:")
        for metric, issues in problematic_metrics.items():
            print(f"\n{metric}:")
            if 'High NaN rate' in issues:
                print(f"  - Consider using median imputation or removing this metric")
            if 'High zero rate' in issues:
                print(f"  - Investigate why so many zeros - may need different calculation")
            if 'No variance' in issues:
                print(f"  - Replace with a metric that has actual variation")
            if 'Contains infinite values' in issues:
                print(f"  - Add bounds checking and clipping to calculation")
    
    print(f"\n💡 To improve Hart Index scores:")
    print(f"  - Focus on metrics with good coverage (>90% non-null, non-zero)")
    print(f"  - Consider log-transforming highly skewed metrics")
    print(f"  - Add robust outlier handling (winsorization)")
    print(f"  - Verify calculation formulas are producing expected ranges")
    
    return problematic_metrics


if __name__ == "__main__":
    # Use the provided CSV file
    csv_path = "/Users/lutherhart/PycharmProjects/alpha_v10/runs/pivot_forecast_seed182_20250923_100314/tradeable_setups_today_20250923_1003.csv"
    
    if os.path.exists(csv_path):
        problematic_metrics = audit_hart_index_data(csv_path)
    else:
        print(f"CSV file not found: {csv_path}")
        print("Please provide the correct path to your tradeable setups CSV")