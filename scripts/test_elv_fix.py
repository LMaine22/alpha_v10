#!/usr/bin/env python3
"""
Test script to verify the ELV calculation fix.
"""
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from alpha_discovery.eval.elv import calculate_elv_and_labels
from alpha_discovery.config import settings

def main():
    # Load the CSV file
    csv_path = "runs/pivot_forecast_seed172_20250921_223848/pareto_front_elv.csv"
    df = pd.read_csv(csv_path)
    
    # Print original dataframe shape and elv stats
    print(f"Original dataframe shape: {df.shape}")
    print(f"Original ELV stats: {df['elv'].describe()}")
    print(f"Original ELV non-null count: {df['elv'].notna().sum()}")
    print(f"Original ELV > 0 count: {(df['elv'] > 0).sum()}")
    
    # Extract row 9 for detailed examination
    row_9 = df.iloc[9]
    print("\nRow 9 components:")
    for key in ['edge_oos', 'live_tr_prior', 'coverage_factor', 'penalty_adj', 'elv']:
        print(f"  {key}: {row_9[key]}")
    
    # Run through calculation again
    print("\nRecalculating ELV...")
    recalculated_df = calculate_elv_and_labels(df)
    
    # Print recalculated dataframe shape and elv stats
    print(f"Recalculated dataframe shape: {recalculated_df.shape}")
    print(f"Recalculated ELV stats: {recalculated_df['elv'].describe()}")
    print(f"Recalculated ELV non-null count: {recalculated_df['elv'].notna().sum()}")
    print(f"Recalculated ELV > 0 count: {(recalculated_df['elv'] > 0).sum()}")
    
    # Extract recalculated row 9
    recalc_row_9 = recalculated_df.iloc[9]
    print("\nRecalculated Row 9 components:")
    for key in ['edge_oos', 'live_tr_prior', 'coverage_factor', 'penalty_adj', 'elv']:
        print(f"  {key}: {recalc_row_9[key]}")
        
    # Count recommendation types
    orig_descriptions = df['description'].value_counts()
    print("\nOriginal recommendations:")
    print(orig_descriptions)
    
    # Count recommended structures
    structure_counts = df['band_probs'].apply(lambda x: suggest_option_structure(x)).value_counts()
    print("\nOptions structure recommendations:")
    print(structure_counts)

def suggest_option_structure(band_probs_str):
    """
    Mock function to mimic the behavior of suggest_option_structure
    """
    if isinstance(band_probs_str, str) and len(band_probs_str) > 5:
        # Parse the array-like string into a list of floats
        try:
            band_probs_str = band_probs_str.strip("[]")
            band_probs = [float(x.strip()) for x in band_probs_str.split(",")]
            
            # Simple logic to mimic the original function
            if sum(band_probs) < 0.01:
                return "no-structure / invalid edge format"
            elif band_probs[6] + band_probs[7] + band_probs[8] > 0.5:
                return "call_ratio or OTM calls (skew monetization)"
            elif band_probs[3] + band_probs[4] > 0.4:
                return "debit_call_spread (target 3–5% band, width ≈ 2–3%)"
            elif band_probs[0] + band_probs[1] + band_probs[2] > 0.4:
                return "debit_put_spread (express −3% to −5%)"
            elif band_probs[0] + band_probs[1] > 0.2:
                return "OTM puts or put ratio (tail hedge)"
            elif band_probs[3] + band_probs[4] + band_probs[5] > 0.6:
                return "calendar/iron_condor (mean-revert/low vol)"
            else:
                return "no-structure / wait (insufficient edge)"
        except:
            return "parsing-error"
    return "invalid-format"

if __name__ == "__main__":
    main()