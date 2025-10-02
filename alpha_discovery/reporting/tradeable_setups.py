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
    
    Now enhanced with Gauntlet 2.0 integration:
    - Uses gauntlet decisions (Deploy/Monitor) instead of Hart Index thresholds
    - Falls back to Hart Index if gauntlet results are unavailable
    - Maintains backward compatibility

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

    # --- 0. Gauntlet Integration ---
    gauntlet_approved_setups = _load_gauntlet_results(run_dir)
    use_gauntlet = gauntlet_approved_setups is not None and not gauntlet_approved_setups.empty
    
    if use_gauntlet:
        print(f"Tradeable Setups: Using Gauntlet 2.0 decisions ({len(gauntlet_approved_setups)} approved setups)")
    else:
        print("Tradeable Setups: Gauntlet results not available, using Hart Index fallback")

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
    # Extending by a factor of 21.0 to allow recent signals (temporary fix for debugging)
    live_setups_df = forecast_df[
        forecast_df['days_since_trigger'] <= forecast_df['best_horizon'] * 21.0
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

    # --- 1.5. Gauntlet Quality Filter ---
    if use_gauntlet:
        # Filter to only setups approved by gauntlet (Deploy or Monitor status)
        initial_count = len(live_setups_df)
        live_setups_df = _apply_gauntlet_filter(live_setups_df, gauntlet_approved_setups)
        gauntlet_count = len(live_setups_df)
        print(f"Tradeable Setups: Gauntlet filter: {initial_count} → {gauntlet_count} setups")
        
        if live_setups_df.empty:
            print("Tradeable Setups: No setups passed gauntlet evaluation.")
            # Create an empty file to indicate no trades are active
            tradeable_path = os.path.join(run_dir, "tradeable_setups_today.csv")
            pd.DataFrame(
                columns=forecast_df.columns
            ).to_csv(tradeable_path, index=False)
            print(f"Wrote empty tradeable setups report to: {tradeable_path}")
            return tradeable_path
    else:
        # Legacy Hart Index filter (as fallback)
        if 'hart_index' in live_setups_df.columns:
            hart_threshold = 50.0  # Configurable threshold
            initial_count = len(live_setups_df) 
            live_setups_df = live_setups_df[live_setups_df['hart_index'] >= hart_threshold]
            hart_count = len(live_setups_df)
            print(f"Tradeable Setups: Hart Index filter (≥{hart_threshold}): {initial_count} → {hart_count} setups")
            
            if live_setups_df.empty:
                print(f"Tradeable Setups: No setups met Hart Index threshold (≥{hart_threshold}).")
                # Create an empty file to indicate no trades are active
                tradeable_path = os.path.join(run_dir, "tradeable_setups_today.csv")
                pd.DataFrame(
                    columns=forecast_df.columns
                ).to_csv(tradeable_path, index=False)
                print(f"Wrote empty tradeable setups report to: {tradeable_path}")
                return tradeable_path

    # --- 2. Enhance with Gauntlet Scores ---
    if use_gauntlet and gauntlet_approved_setups is not None:
        live_setups_df = _merge_gauntlet_scores(live_setups_df, gauntlet_approved_setups)

    # --- 3. Single best per ticker (recency -> Quality Score -> ELV) ---
    # For each ticker, keep only the most recent last_trigger; tie-break by quality score, then elv
    best_rows = []
    for ticker, group in live_setups_df.groupby('ticker'):
        # Build sort keys present in the data - prioritize gauntlet scores if available
        sort_cols = ['last_trigger']
        
        if use_gauntlet and 'promotion_score' in group.columns:
            sort_cols.append('promotion_score')
        elif 'hart_index' in group.columns:
            sort_cols.append('hart_index')
            
        if 'elv' in group.columns:
            sort_cols.append('elv')

        ascending = [False] * len(sort_cols)
        chosen = group.sort_values(by=sort_cols, ascending=ascending).head(1)
        best_rows.append(chosen)

    final_setups_df = pd.concat(best_rows, ignore_index=False)

    # Global sort for readability: prioritize quality scores, then recency
    if use_gauntlet and 'promotion_score' in final_setups_df.columns:
        final_setups_df = final_setups_df.sort_values(by=['promotion_score', 'last_trigger'], ascending=[False, False])
    elif 'hart_index' in final_setups_df.columns:
        final_setups_df = final_setups_df.sort_values(by=['hart_index', 'last_trigger'], ascending=[False, False])
    elif 'elv' in final_setups_df.columns:
        final_setups_df = final_setups_df.sort_values(by=['elv', 'last_trigger'], ascending=[False, False])

    # --- 3. Write the Report ---
    # Include timestamp in the filename to track when the setup was generated
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    tradeable_path = os.path.join(run_dir, f"tradeable_setups_today_{timestamp}.csv")
    
    # Select and reorder columns for clarity with extended information
    # Include gauntlet scores if available
    output_cols = [
        "ticker", "setup_desc", "suggested_structure", 
        # Quality scores (gauntlet first, then legacy)
        "promotion_score", "final_decision", "hart_index", "elv", 
        # Core metrics
        "edge_oos", "info_gain", "w1_effect", "transfer_entropy", 
        # Coverage & robustness
        "n_trig_oos", "fold_coverage", "regime_breadth",
        # Timing
        "first_trigger", "last_trigger", "best_horizon", "days_since_trigger",
        # Forecast probabilities
        "E_move", "P_up", "P_down", "P(>5%)", "P(<-5%)",
        # Gauntlet stage scores (detailed)
        "s1_health_score", "s2_profitability_score", 
        "s3_hart_full_score", "s4_portfolio_score"
    ]
    # Ensure all selected columns exist, and if not, they are ignored
    final_cols = [col for col in output_cols if col in final_setups_df.columns]
    
    final_setups_df.to_csv(tradeable_path, index=False, columns=final_cols, float_format="%.4f")

    # Enhanced logging
    decision_source = "Gauntlet 2.0" if use_gauntlet else "Hart Index (fallback)"
    print(f"Wrote {len(final_setups_df)} actionable setups to: {tradeable_path}")
    print(f"Decision source: {decision_source}")
    
    if not final_setups_df.empty:
        # Show quality score distribution
        if use_gauntlet and 'promotion_score' in final_setups_df.columns:
            scores = final_setups_df['promotion_score'].dropna()
            if not scores.empty:
                print(f"Promotion scores: {scores.min():.2f} to {scores.max():.2f} (mean: {scores.mean():.2f})")
        elif 'hart_index' in final_setups_df.columns:
            scores = final_setups_df['hart_index'].dropna()
            if not scores.empty:
                print(f"Hart Index scores: {scores.min():.1f} to {scores.max():.1f} (mean: {scores.mean():.1f})")
    
        # Show final decisions if available
        if 'final_decision' in final_setups_df.columns:
            decisions = final_setups_df['final_decision'].value_counts()
            print(f"Final decisions: {dict(decisions)}")
    
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


def _load_gauntlet_results(run_dir: str) -> Optional[pd.DataFrame]:
    """
    Load gauntlet results from the run directory.
    
    Returns:
        DataFrame with gauntlet decisions or None if not available
    """
    gauntlet_path = os.path.join(run_dir, "gauntlet", "gauntlet_results.csv")
    
    if not os.path.exists(gauntlet_path):
        return None
    
    try:
        gauntlet_df = pd.read_csv(gauntlet_path)
        
        # Filter to only approved setups (Deploy or Monitor status)
        if 'passed_gauntlet' in gauntlet_df.columns:
            # Use boolean column if available
            approved = gauntlet_df[gauntlet_df['passed_gauntlet'] == True]
        elif 'final_decision' in gauntlet_df.columns:
            # Use final decision column
            approved = gauntlet_df[gauntlet_df['final_decision'].isin(['Deploy', 'Monitor'])]
        elif 'pass_stage5' in gauntlet_df.columns:
            # Use stage 5 pass column as fallback
            approved = gauntlet_df[gauntlet_df['pass_stage5'] == True]
        else:
            print("Tradeable Setups: Warning - gauntlet results missing expected decision columns")
            return None
        
        if approved.empty:
            print("Tradeable Setups: No setups approved by gauntlet")
            return pd.DataFrame()  # Return empty but valid DataFrame
        
        print(f"Tradeable Setups: Loaded {len(approved)} gauntlet-approved setups")
        return approved
        
    except Exception as e:
        print(f"Tradeable Setups: Error loading gauntlet results: {e}")
        return None


def _apply_gauntlet_filter(live_setups_df: pd.DataFrame, gauntlet_approved: pd.DataFrame) -> pd.DataFrame:
    """
    Filter live setups to only include those approved by gauntlet.
    
    Args:
        live_setups_df: Setups that passed recency filter
        gauntlet_approved: Setups approved by gauntlet
        
    Returns:
        Filtered DataFrame with only gauntlet-approved setups
    """
    if gauntlet_approved.empty:
        return pd.DataFrame(columns=live_setups_df.columns)
    
    # Try to match on setup_id first (most reliable)
    if 'setup_id' in live_setups_df.columns and 'setup_id' in gauntlet_approved.columns:
        approved_ids = set(gauntlet_approved['setup_id'].astype(str))
        live_setups_df['setup_id_str'] = live_setups_df['setup_id'].astype(str)
        filtered = live_setups_df[live_setups_df['setup_id_str'].isin(approved_ids)]
        filtered = filtered.drop('setup_id_str', axis=1)
        
        if not filtered.empty:
            print(f"Tradeable Setups: Matched {len(filtered)} setups by setup_id")
            return filtered
    
    # Fallback: try matching on individual (tuple representation)
    if 'individual' in live_setups_df.columns and 'setup_id' in gauntlet_approved.columns:
        approved_individuals = set()
        for _, row in gauntlet_approved.iterrows():
            # Extract individual from setup_id if possible
            setup_id = str(row['setup_id'])
            approved_individuals.add(setup_id)
        
        live_setups_df['individual_str'] = live_setups_df['individual'].astype(str)
        filtered = live_setups_df[live_setups_df['individual_str'].isin(approved_individuals)]
        filtered = filtered.drop('individual_str', axis=1)
        
        if not filtered.empty:
            print(f"Tradeable Setups: Matched {len(filtered)} setups by individual")
            return filtered
    
    # If no matches found, return empty
    print("Tradeable Setups: Warning - could not match any setups between forecast and gauntlet results")
    return pd.DataFrame(columns=live_setups_df.columns)


def _merge_gauntlet_scores(live_setups_df: pd.DataFrame, gauntlet_approved: pd.DataFrame) -> pd.DataFrame:
    """
    Merge gauntlet scores (promotion_score, stage scores) into live setups.
    
    Args:
        live_setups_df: Setups that passed recency filter
        gauntlet_approved: Setups approved by gauntlet with scores
        
    Returns:
        Enhanced DataFrame with gauntlet scores
    """
    # Score columns to merge from gauntlet
    score_cols = [
        'promotion_score', 'final_decision',
        's1_health_score', 's2_profitability_score', 
        's3_hart_full_score', 's4_portfolio_score'
    ]
    
    # Available score columns in gauntlet data
    available_scores = [col for col in score_cols if col in gauntlet_approved.columns]
    
    if not available_scores:
        print("Tradeable Setups: No gauntlet scores available for merging")
        return live_setups_df
    
    # Prepare merge columns
    merge_cols = ['setup_id'] + available_scores
    gauntlet_scores = gauntlet_approved[merge_cols].copy()
    
    # Merge on setup_id
    if 'setup_id' in live_setups_df.columns:
        enhanced = live_setups_df.merge(
            gauntlet_scores,
            on='setup_id',
            how='left',
            suffixes=('', '_gauntlet')
        )
        
        merged_count = enhanced[available_scores[0]].notna().sum()
        print(f"Tradeable Setups: Merged gauntlet scores for {merged_count}/{len(enhanced)} setups")
        return enhanced
    else:
        print("Tradeable Setups: Cannot merge gauntlet scores - no setup_id column")
        return live_setups_df
