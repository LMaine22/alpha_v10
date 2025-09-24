#!/usr/bin/env python3
"""
Test script to demonstrate Hart Index calculation with sample data.
"""

import pandas as pd
import numpy as np
from alpha_discovery.eval.hart_index import calculate_hart_index, get_hart_index_summary

# Create sample data with various metric values
sample_data = {
    'individual': [
        ('AAPL US Equity', ['SIG_001', 'SIG_002']),
        ('MSFT US Equity', ['SIG_003', 'SIG_004']),
        ('GOOGL US Equity', ['SIG_005', 'SIG_006']),
        ('TSLA US Equity', ['SIG_007', 'SIG_008']),
        ('NVDA US Equity', ['SIG_009', 'SIG_010']),
    ],
    'edge_crps_raw': [0.15, 0.25, 0.20, 0.35, 0.18],
    'edge_pin_q10_raw': [0.10, 0.15, 0.12, 0.20, 0.11],
    'edge_pin_q90_raw': [0.12, 0.18, 0.14, 0.22, 0.13],
    'edge_ig_raw': [0.30, 0.15, 0.25, 0.10, 0.28],
    'edge_w1_raw': [0.08, 0.12, 0.10, 0.15, 0.09],
    'edge_calib_mae_raw': [0.05, 0.08, 0.06, 0.10, 0.04],
    'bootstrap_p_value_raw': [0.02, 0.15, 0.05, 0.25, 0.01],
    'n_trig_oos': [25, 10, 18, 5, 30],
    'stab_crps_mad': [0.02, 0.05, 0.03, 0.08, 0.015],
    'sensitivity_delta_edge_raw': [0.05, 0.15, 0.08, 0.20, 0.03],
    'regime_breadth': [0.8, 0.5, 0.7, 0.3, 0.9],
    'fold_coverage': [0.9, 0.6, 0.8, 0.4, 0.95],
    'redundancy_mi_raw': [0.2, 0.5, 0.3, 0.7, 0.15],
    'complexity_metric_raw': [0.5, 0.3, 0.6, 0.2, 0.55],
    'dfa_alpha_raw': [0.65, 0.8, 0.6, 0.9, 0.68],
    'transfer_entropy_raw': [0.15, 0.05, 0.12, 0.02, 0.18],
    'live_tr_prior': [0.08, 0.04, 0.06, 0.02, 0.10],
    'coverage_factor': [0.7, 0.4, 0.6, 0.2, 0.8],
    'page_hinkley_alarm': [0, 0, 1, 0, 0],
    'dormant_qualified_flag': [False, False, False, True, False],
    'specialist_flag': [False, False, False, False, True],
    'pass_cv_gates': [True, True, False, True, True],
    'elv': [0.25, 0.10, 0.15, 0.05, 0.30],
    'E_move': [0.02, -0.01, 0.015, -0.02, 0.025],
    'P_up': [0.65, 0.45, 0.58, 0.40, 0.70],
    'P_down': [0.35, 0.55, 0.42, 0.60, 0.30],
}

# Create DataFrame
df = pd.DataFrame(sample_data)

print("Sample Data Overview:")
print(df[['individual', 'edge_crps_raw', 'edge_ig_raw', 'n_trig_oos', 'elv']].to_string())
print("\n" + "="*80 + "\n")

# Calculate Hart Index
print("Calculating Hart Index...")
result_df = calculate_hart_index(df)

# Display results
print("\nHart Index Results:")
display_cols = ['individual', 'hart_index', 'hart_index_label', 'elv']
print(result_df[display_cols].to_string())

print("\n" + "="*80 + "\n")

# Show component breakdown for the top setup
top_setup = result_df.loc[result_df['hart_index'].idxmax()]
print(f"Component Breakdown for Top Setup: {top_setup['individual']}")
print(f"  Hart Index: {top_setup['hart_index']:.1f} ({top_setup['hart_index_label']})")
print(f"  Performance Total: {top_setup['hart_performance_total']:.1f}/40")
print(f"  Robustness Total: {top_setup['hart_robustness_total']:.1f}/30")
print(f"  Complexity Total: {top_setup['hart_complexity_total']:.1f}/15")
print(f"  Readiness Total: {top_setup['hart_readiness_total']:.1f}/15")

print("\n" + "="*80 + "\n")

# Get summary statistics
summary = get_hart_index_summary(result_df)
print("Hart Index Distribution Summary:")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

print("\n" + "="*80 + "\n")

# Demonstrate how adjustments work
print("Impact of Quality Gates:")
for idx, row in result_df.iterrows():
    ticker, signals = row['individual']
    adjustments = []
    if not row['pass_cv_gates']:
        adjustments.append("Failed CV gates (-30%)")
    if row['elv'] > 0.25:  # High ELV bonus
        adjustments.append("High ELV bonus (+10%)")
    
    if adjustments:
        print(f"{ticker}: {', '.join(adjustments)}")

print("\nTest completed successfully!")
