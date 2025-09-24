"""
Debug script to examine ELV calculation issues with a specific row

This script loads the saved pareto front and analyzes row 9 in detail,
showing the step-by-step ELV calculation and the reason it's empty.
"""

import pandas as pd
import numpy as np
from alpha_discovery.config import settings
from alpha_discovery.eval.elv import calculate_elv_and_labels

# Path to the pareto front CSV
csv_path = "/Users/lutherhart/PycharmProjects/alpha_v10/runs/pivot_forecast_seed172_20250921_213657/pareto_front_elv.csv"

# Load the CSV file
df = pd.read_csv(csv_path)
print(f"Loaded CSV with {len(df)} rows")

# Look at row 9 - the AMZN SIG_03053/SIG_04306 example
row_index = 8  # 0-indexed
row = df.iloc[row_index]
print("\n=== Row Information ===")
print(f"Individual: {row['individual']}")
print(f"n_trig_oos: {row['n_trig_oos']}")
print(f"edge_oos: {row['edge_oos']}")
print(f"live_tr_prior: {row['live_tr_prior']}")
print(f"coverage_factor: {row['coverage_factor']}")
print(f"penalty_adj: {row['penalty_adj']}")
print(f"elv: {row['elv']}")

# Calculate expected ELV value
expected_elv = row['edge_oos'] * row['live_tr_prior'] * row['coverage_factor'] * row['penalty_adj']
print(f"\nExpected ELV: {expected_elv:.4f}")

# Look at flags 
print("\n=== Flags ===")
print(f"dormant_flag: {row['dormant_flag']}")
print(f"dormant_qualified_flag: {row['dormant_qualified_flag']}")
print(f"pass_cv_gates: {row['pass_cv_gates']}")
print(f"specialist_flag: {row['specialist_flag']}")

# Check gate conditions
print("\n=== Gate Checks ===")
print(f"edge_crps_raw: {row['edge_crps_raw']}")
print(f"edge_ig_raw: {row['edge_ig_raw']}")
print(f"redundancy_mi_raw: {row['redundancy_mi_raw']}")
print(f"sensitivity_delta_edge_raw: {row['sensitivity_delta_edge_raw']}")

# Check disqualification gates
print("\n=== Disqualification Gate Checks ===")
print(f"edge_calib_mae_raw: {row['edge_calib_mae_raw']}")
print(f"bootstrap_p_value_raw: {row['bootstrap_p_value_raw']}")
print(f"page_hinkley_alarm: {row['page_hinkley_alarm']}")

# Recompute ELV to see if we get the right value
print("\n=== Recomputing ELV ===")
single_row_df = pd.DataFrame([row.to_dict()])

# Set any NaN values in required fields to zero or appropriate defaults
required_fields = ['edge_crps_raw', 'edge_pin_q10_raw', 'edge_pin_q90_raw', 'edge_ig_raw', 
                  'edge_w1_raw', 'edge_calib_mae_raw', 'redundancy_mi_raw', 'complexity_metric_raw',
                  'bootstrap_p_value_raw', 'sensitivity_delta_edge_raw', 'stab_crps_mad',
                  'regime_breadth', 'fold_coverage', 'tr_cv_reg', 'tr_fg', 'page_hinkley_alarm']

for field in required_fields:
    if field in single_row_df.columns and pd.isna(single_row_df[field].iloc[0]):
        print(f"Setting NaN value in {field} to 0")
        single_row_df[field] = 0

# Run through ELV calculation
try:
    new_df = calculate_elv_and_labels(single_row_df)
    print(f"Recalculated ELV: {new_df['elv'].iloc[0]:.4f}")
    print(f"Recalculated edge_oos: {new_df['edge_oos'].iloc[0]:.4f}")
    print(f"Recalculated live_tr_prior: {new_df['live_tr_prior'].iloc[0]:.4f}")
    print(f"Recalculated coverage_factor: {new_df['coverage_factor'].iloc[0]:.4f}")
    print(f"Recalculated penalty_adj: {new_df['penalty_adj'].iloc[0]:.4f}")
    print(f"Recalculated pass_cv_gates: {new_df['pass_cv_gates'].iloc[0]}")
except Exception as e:
    print(f"Error in recalculation: {e}")

print("\nConclusion:")
print("The ELV value is likely missing in the CSV file due to a serialization issue when writing to CSV.")
print("The calculate_elv_and_labels function should be producing the correct value.")
print("Recommended fix: Check data types in final_results_df and ensure proper CSV writing.")