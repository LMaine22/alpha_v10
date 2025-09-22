# alpha_discovery/reporting/tradeable_setups.py
"""
Generates a filtered report of the most actionable, tradeable setups for the current day.
"""

from __future__ import annotations
import os
import pandas as pd
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
    forecast_df['days_since_trigger'] = (end_of_data_date - forecast_df['last_trigger']).dt.days

    # Keep only setups where the forecast horizon has not yet expired
    # We use 'best_horizon' as it represents the most reliable forecast window for the setup
    live_setups_df = forecast_df[
        forecast_df['days_since_trigger'] <= forecast_df['best_horizon']
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

    # --- 2. Best-of-Ticker Filter ---
    # For each ticker, find the setup with the highest ELV score
    # This resolves conflicting forecasts by picking the one with the highest overall quality
    best_setups_indices = live_setups_df.groupby('ticker')['elv'].idxmax()
    final_setups_df = live_setups_df.loc[best_setups_indices]

    # Sort by ELV score for a clean, prioritized list
    final_setups_df = final_setups_df.sort_values("elv", ascending=False)

    # --- 3. Write the Report ---
    tradeable_path = os.path.join(run_dir, "tradeable_setups_today.csv")
    
    # Select and reorder columns for clarity
    output_cols = [
        "ticker", "setup_desc", "suggested_structure", "elv", "edge_oos", "n_trig_oos",
        "first_trigger", "last_trigger", "best_horizon", "days_since_trigger",
        "E_move", "P_up", "P_down", "P(>5%)", "P(<-5%)"
    ]
    # Ensure all selected columns exist, and if not, they are ignored
    final_cols = [col for col in output_cols if col in final_setups_df.columns]
    
    final_setups_df.to_csv(tradeable_path, index=False, columns=final_cols, float_format="%.4f")

    print(f"Wrote {len(final_setups_df)} actionable setups to: {tradeable_path}")

    return tradeable_path
