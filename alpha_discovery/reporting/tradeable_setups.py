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
    Filters the main forecast slate to produce a list of the single best, most actionable
    trade for each ticker for the current day.
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

    # --- 2. Best-of-Breed Per-Ticker Filter ---
    # For each ticker, select only the single best setup based on a hierarchy:
    # 1. Most recent `last_trigger` date.
    # 2. Highest `hart_index` as a tie-breaker.
    best_setups_indices = []
    
    for ticker, group in live_setups_df.groupby('ticker'):
        # Find the most recent trigger date in the group
        most_recent_date = group['last_trigger'].max()
        
        # Filter for all setups that triggered on that most recent date
        recent_setups = group[group['last_trigger'] == most_recent_date]
        
        # From that group, find the one with the highest HartIndex as the tie-breaker
        # .name attribute of the series returned by idxmax() is the index label
        best_setup_index = recent_setups['hart_index'].idxmax()
        best_setups_indices.append(best_setup_index)
    
    final_setups_df = live_setups_df.loc[best_setups_indices]

    # Sort the final, unique-per-ticker list by HartIndex for a clean, prioritized list
    final_setups_df = final_setups_df.sort_values("hart_index", ascending=False)

    # --- 3. Write the Report ---
    # Include timestamp in the filename to track when the setup was generated
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tradeable_path = os.path.join(run_dir, f"tradeable_setups_today_{timestamp}.csv")
    
    # Select and reorder columns for clarity with extended information
    output_cols = [
        "ticker", "setup_desc", "hart_index", "hart_index_label", "suggested_structure", 
        "elv", "edge_oos", "info_gain", 
        "w1_effect", "transfer_entropy", "n_trig_oos", "fold_coverage", "regime_breadth",
        "first_trigger", "last_trigger", "best_horizon", "days_since_trigger",
        "E_move", "P_up", "P_down", "P(>5%)", "P(<-5%)",
        # Hart Index raw components for diagnostics
        "edge_crps_raw", "edge_pin_q10_raw", "edge_pin_q90_raw", "edge_ig_raw", 
        "edge_w1_raw", "edge_calib_mae_raw", "bootstrap_p_value_raw", 
        "sensitivity_delta_edge_raw", "redundancy_mi_raw", "complexity_metric_raw",
        "dfa_alpha_raw", "transfer_entropy_raw", "live_tr_prior", "coverage_factor",
        "page_hinkley_alarm",
        # Hart Index component scores for transparency
        "hart_edge_performance", "hart_information_quality", "hart_prediction_accuracy", 
        "hart_risk_reward", "hart_statistical_significance", "hart_stability_consistency", 
        "hart_sensitivity_resilience", "hart_signal_quality", "hart_complexity_balance", 
        "hart_trigger_reliability", "hart_regime_coverage",
        "hart_performance_total", "hart_robustness_total", "hart_complexity_total", "hart_readiness_total"
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
        "ticker", "setup_desc", "hart_index", "hart_index_label", "suggested_structure", 
        "elv", "edge_oos", "info_gain",
        "w1_effect", "transfer_entropy", "n_trig_oos", "fold_coverage", "regime_breadth",
        "first_trigger", "last_trigger", "best_horizon", "days_since_trigger",
        "E_move", "P_up", "P_down", "P(>5%)", "P(<-5%)",
        # Hart Index raw components for diagnostics  
        "edge_crps_raw", "edge_pin_q10_raw", "edge_pin_q90_raw", "edge_ig_raw", 
        "edge_w1_raw", "edge_calib_mae_raw", "bootstrap_p_value_raw", 
        "sensitivity_delta_edge_raw", "redundancy_mi_raw", "complexity_metric_raw",
        "dfa_alpha_raw", "transfer_entropy_raw", "live_tr_prior", "coverage_factor",
        "page_hinkley_alarm",
        # Hart Index component scores for transparency
        "hart_edge_performance", "hart_information_quality", "hart_prediction_accuracy", 
        "hart_risk_reward", "hart_statistical_significance", "hart_stability_consistency", 
        "hart_sensitivity_resilience", "hart_signal_quality", "hart_complexity_balance", 
        "hart_trigger_reliability", "hart_regime_coverage", 
        "hart_performance_total", "hart_robustness_total", "hart_complexity_total", "hart_readiness_total"
    ]
    # Ensure all selected columns exist
    final_cols = [col for col in output_cols if col in final_extended_df.columns]
    
    final_extended_df.to_csv(extended_path, index=False, columns=final_cols, float_format="%.4f")
    
    print(f"Wrote {len(final_extended_df)} extended forecasts to: {extended_path}")
    
    return extended_path
