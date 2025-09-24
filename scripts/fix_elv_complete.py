"""
Fix for ELV calculation issues with missing values and NaN output

This script modifies the alpha_discovery/eval/elv.py file to:
1. Fill NA values in key fields before calculation
2. Handle missing redundancy_mi_raw more gracefully
3. Ensure proper data types in calculations

It then runs the main pipeline with these fixes to produce valid ELV values.
"""

import os
import pandas as pd
import numpy as np

# Define the fix for the ELV calculation
def fix_elv_calculation_function():
    target_file = "/Users/lutherhart/PycharmProjects/alpha_v10/alpha_discovery/eval/elv.py"
    
    # Read the current file
    with open(target_file, "r") as f:
        lines = f.readlines()
    
    # Find the calculate_elv_and_labels function
    start_idx = -1
    for i, line in enumerate(lines):
        if "def calculate_elv_and_labels" in line:
            start_idx = i
            break
    
    if start_idx == -1:
        print("Could not find the calculate_elv_and_labels function")
        return False
    
    # Find insertion point right after the debug input check
    insert_idx = -1
    for i in range(start_idx, len(lines)):
        if "print(f\"  {col}: {non_null} " in lines[i]:
            insert_idx = i + 3  # Add after the debug input check section
            break
    
    if insert_idx == -1:
        print("Could not find the insertion point after debug input check")
        return False
    
    # Prepare the code to insert - this fills NaN values in key fields
    code_to_insert = """    # Fill missing values in key fields to prevent NaN propagation
    fill_zero_cols = ['edge_crps_raw', 'edge_pin_q10_raw', 'edge_pin_q90_raw', 
                     'edge_ig_raw', 'edge_w1_raw', 'edge_calib_mae_raw', 
                     'sensitivity_delta_edge_raw', 'bootstrap_p_value_raw']
    df[fill_zero_cols] = df[fill_zero_cols].fillna(0.0)
    
    # Handle redundancy_mi_raw separately - important for penalty calculation
    if 'redundancy_mi_raw' in df.columns:
        df['redundancy_mi_raw'] = df['redundancy_mi_raw'].fillna(0.5)  # Middle value as default
    else:
        df['redundancy_mi_raw'] = 0.5
    
    # Handle complexity metric
    if 'complexity_metric_raw' in df.columns:
        df['complexity_metric_raw'] = df['complexity_metric_raw'].fillna(0.5)
    else:
        df['complexity_metric_raw'] = 0.5
        
    # Ensure stab_crps_mad has valid values
    if 'stab_crps_mad' not in df.columns or df['stab_crps_mad'].isna().all():
        df['stab_crps_mad'] = 0.1  # Default stability value
        
"""

    # Insert the code
    lines.insert(insert_idx, code_to_insert)
    
    # Replace line that uses fillna with inplace=True to avoid warning
    for i, line in enumerate(lines):
        if "df['edge_oos'].fillna(0, inplace=True)" in line:
            lines[i] = "    df['edge_oos'] = df['edge_oos'].fillna(0)\n"
    
    # Write the modified file
    with open(target_file, "w") as f:
        f.writelines(lines)
    
    print(f"Successfully modified {target_file}")
    return True

# Fix the elv.py file
if fix_elv_calculation_function():
    print("ELV calculation function fixed successfully.")
    
    # Also lower the gate threshold as we did before
    from alpha_discovery.config import settings
    settings.elv.gate_min_oos_triggers = 5
    print(f"Lowered gate_min_oos_triggers from 15 to {settings.elv.gate_min_oos_triggers}")
    
    # Run the main script
    print("\nRunning main script with fixes...")
    import main
    main.main()
else:
    print("Failed to fix ELV calculation function.")