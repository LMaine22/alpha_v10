# alpha_discovery/reporting/tradeable_setups.py
"""
Generates a filtered report of the most actionable, tradeable setups for the current day.
"""

from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

def write_tradeable_setups(
    forecast_df: pd.DataFrame,
    end_of_data_date: pd.Timestamp,
    run_dir: str = "runs"
) -> str:
    """
    Filters the main forecast slate to produce a list of currently actionable trades.

    Args:
        forecast_df: The full DataFrame from the forecast_slate.csv.
        end_of_data_date: The last date available in the master dataset.
        run_dir: The directory to write the output CSV.

    Returns:
        Path to the written CSV file.
    """
    if forecast_df.empty:
        print("Tradeable Setups: Input forecast slate is empty. Skipping.")
        return ""

    # --- 1. Recency Filter ---
    # Ensure date columns are in datetime format
    forecast_df['last_trigger'] = pd.to_datetime(forecast_df['last_trigger'], errors='coerce')

    # Calculate how many days have passed since the last trigger
    # Force end_of_data_date to be a Timestamp if it's not already
    if not isinstance(end_of_data_date, pd.Timestamp):
        end_of_data_date = pd.Timestamp(end_of_data_date)
    
    forecast_df['days_since_trigger'] = (end_of_data_date - forecast_df['last_trigger']).dt.days
    
    # Log for debugging
    print(f"Tradeable Setups: End of data date is {end_of_data_date}")
    print(f"Tradeable Setups: Calculated days_since_trigger for sample setups:")
    if not forecast_df.empty:
        sample = forecast_df.head(3)
        for _, row in sample.iterrows():
            print(f"  {row['ticker']} last_trigger={row['last_trigger']}, days_since={row['days_since_trigger']}")

    # Keep only setups where the forecast horizon has not yet expired or is within a significantly extended window
    # We use 'best_horizon' as it represents the most reliable forecast window for the setup
    # Extending by a factor of 2.0 to allow signals past their optimal window but still potentially useful
    live_setups_df = forecast_df[
        forecast_df['days_since_trigger'] <= forecast_df['best_horizon'] * 2.0
    ].copy()

    if live_setups_df.empty:
        print("Tradeable Setups: No live setups found after applying recency filter.")
        # Create an empty file to indicate no trades are active
        tradeable_path = os.path.join(run_dir, "tradeable_setups_today.csv")
        pd.DataFrame(
            columns=forecast_df.columns
        ).to_csv(tradeable_path, index=False)
        print(f"Wrote empty tradeable setups report to: {tradeable_path}")
        return tradeable_path

    # --- 2. Single best per ticker (recency -> HartIndex -> ELV) ---
    # For each ticker, keep only the most recent last_trigger; tie-break by hart_index, then elv
    best_rows = []
    for ticker, group in live_setups_df.groupby('ticker'):
        # Build sort keys present in the data
        sort_cols = ['last_trigger']
        if 'hart_index' in group.columns:
            sort_cols.append('hart_index')
        if 'elv' in group.columns:
            sort_cols.append('elv')

        ascending = [False] * len(sort_cols)
        chosen = group.sort_values(by=sort_cols, ascending=ascending).head(1)
        best_rows.append(chosen)

    final_setups_df = pd.concat(best_rows, ignore_index=False)

    # Global sort for readability: prioritize HartIndex, then recency
    if 'hart_index' in final_setups_df.columns:
        final_setups_df = final_setups_df.sort_values(by=['hart_index', 'last_trigger'], ascending=[False, False])
    elif 'elv' in final_setups_df.columns:
        final_setups_df = final_setups_df.sort_values(by=['elv', 'last_trigger'], ascending=[False, False])

    # --- 3. Write the Report ---
    # Include timestamp in the filename to track when the setup was generated
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tradeable_path = os.path.join(run_dir, f"tradeable_setups_today_{timestamp}.csv")
    
    # Select and reorder columns for clarity with extended information
    output_cols = [
        "ticker", "setup_desc", "suggested_structure", "elv", "edge_oos", "info_gain", 
        "w1_effect", "transfer_entropy", "n_trig_oos", "fold_coverage", "regime_breadth",
        "first_trigger", "last_trigger", "best_horizon", "days_since_trigger",
        "E_move", "P_up", "P_down", "P(>5%)", "P(<-5%)"
    ]
    # Ensure all selected columns exist, and if not, they are ignored
    final_cols = [col for col in output_cols if col in final_setups_df.columns]
    
    final_setups_df.to_csv(tradeable_path, index=False, columns=final_cols, float_format="%.4f")

    print(f"Wrote {len(final_setups_df)} actionable setups to: {tradeable_path}")
    
    # Also create an extended forecast report with more signals
    extended_setups_path = write_extended_forecasts(forecast_df, end_of_data_date, run_dir)

    return tradeable_path


def write_extended_forecasts(
    forecast_df: pd.DataFrame,
    end_of_data_date: pd.Timestamp,
    run_dir: str = "runs"
) -> str:
    """
    Creates a more comprehensive report of potential forecasts with relaxed filtering.
    This includes signals that don't meet all the strict tradeable criteria but may still be valuable.
    
    Args:
        forecast_df: The full DataFrame from the forecast_slate.csv.
        end_of_data_date: The last date available in the master dataset.
        run_dir: The directory to write the output CSV.
    
    Returns:
        Path to the written CSV file.
    """
    if forecast_df.empty:
        print("Extended Forecasts: Input forecast slate is empty. Skipping.")
        return ""

    # --- 1. Apply Extended Recency Filter ---
    # Use a much longer window - 2.5x the best horizon
    forecast_df['last_trigger'] = pd.to_datetime(forecast_df['last_trigger'], errors='coerce')
    forecast_df['days_since_trigger'] = (end_of_data_date - forecast_df['last_trigger']).dt.days
    
    extended_setups_df = forecast_df[
        forecast_df['days_since_trigger'] <= forecast_df['best_horizon'] * 2.5
    ].copy()

    if extended_setups_df.empty:
        print("Extended Forecasts: No setups found after applying extended recency filter.")
        return ""
    
    # --- 2. Apply Minimum ELV Filter but keep more setups per ticker ---
    # Filter by minimum ELV score, but even more permissive
    min_elv_score = 0.005  # Even lower threshold to catch more signals
    extended_filtered_df = extended_setups_df[extended_setups_df['elv'] >= min_elv_score]
    
    # Keep top 5 per ticker
    top_n_per_ticker = 5
    best_setups_indices = []
    
    for ticker, group in extended_filtered_df.groupby('ticker'):
        # Get indices of top N setups for this ticker, sorted by ELV
        top_indices = group.sort_values('elv', ascending=False).head(top_n_per_ticker).index
        best_setups_indices.extend(top_indices)
    
    final_extended_df = extended_filtered_df.loc[best_setups_indices]

    # --- 3. Write the Report ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    extended_path = os.path.join(run_dir, f"extended_forecasts_{timestamp}.csv")
    
    # Select and reorder columns for clarity
    output_cols = [
        "ticker", "setup_desc", "suggested_structure", "elv", "edge_oos", "info_gain",
        "w1_effect", "transfer_entropy", "n_trig_oos", "fold_coverage", "regime_breadth",
        "first_trigger", "last_trigger", "best_horizon", "days_since_trigger",
        "E_move", "P_up", "P_down", "P(>5%)", "P(<-5%)"
    ]
    # Ensure all selected columns exist
    final_cols = [col for col in output_cols if col in final_extended_df.columns]
    
    final_extended_df.to_csv(extended_path, index=False, columns=final_cols, float_format="%.4f")
    
    print(f"Wrote {len(final_extended_df)} extended forecasts to: {extended_path}")
    
    return extended_path
