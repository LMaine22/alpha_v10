# alpha_discovery/signals/compiler.py

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple
from ..features import core as fcore

# Suppress pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
# ===================================================================
# EVENT FEATURE MAPPING
# Maps generic EV_* feature names to more interpretable event types
# ===================================================================

EVENT_FEATURE_MAPPING = {
    # Tail flags and extreme events
    'EV__EV_tail_flag_1p5': 'Extreme Event (1.5σ)',
    'EV__EV_tail_flag_2p5': 'Extreme Event (2.5σ)',
    'EV__EV_revision_tail_flag': 'Revision Shock',
    'EV__EV_time_since_tailshock_60': 'Time Since Shock',
    
    # Surprise and dispersion
    'EV__EV_surprise_dispersion_day': 'Surprise Dispersion',
    'EV__EV_signed_surprise_percentile': 'Surprise Percentile',
    'EV__EV_net_info_surprise': 'Net Info Surprise',
    'EV__EV_shock_adjusted_surprise': 'Shock-Adjusted Surprise',
    
    # Revisions and memory
    'EV__EV_revision_z': 'Revision Z-Score',
    'EV__EV_revision_polarity_memory_3': 'Revision Memory',
    'EV__EV_revision_vol_252': 'Revision Volatility',
    
    # Calendar and timing
    'EV__EV_forward_calendar_heat_3': 'Calendar Heat',
    'EV__EV_calendar_vacuum_7': 'Calendar Vacuum',
    'EV__EV_dense_macro_day_score': 'Macro Day Score',
    'EV__EV_top_tier_dominance_share': 'Top Tier Dominance',
    
    # Clustering and divergence
    'EV__EV_clustered_tail_count_5': 'Clustered Tails',
    'EV__EV_infl_vs_growth_divergence': 'Inflation vs Growth',
    
    # Legacy features
    'EV__EV_days_to_high': 'Days to High Impact',
    'EV__EV_in_window': 'In Event Window',
    'EV__EV_pre_window': 'Pre-Event Window',
    'EV__EV_is_event_week': 'Event Week',
    'EV__EV_after_surprise_z': 'Post-Surprise Z',
    'EV__EV_after_pos': 'Positive Surprise',
    'EV__EV_after_neg': 'Negative Surprise',
    'EV__EV_surprise_ewma_21': 'Surprise EWMA (21d)',
    'EV__EV_surprise_ewma_63': 'Surprise EWMA (63d)',
    'EV__EV_tail_intensity_21': 'Tail Intensity (21d)',
    'EV__EV_dense_macro_window_7': 'Dense Macro Window',
}

def _get_interpretable_event_name(feature_name: str) -> str:
    """Convert generic EV_* feature names to more interpretable event types."""
    # Check if it's an event feature (look for EV__EV_ anywhere in the name)
    if '_EV__EV_' in feature_name:
        # Extract the base feature name (after EV__EV_)
        base_name = feature_name.split('_EV__EV_')[-1]
        # Create the full feature key for lookup
        full_key = f"EV__EV_{base_name}"
        mapped_name = EVENT_FEATURE_MAPPING.get(full_key, f"Event: {base_name}")
        # Return with ticker prefix if present
        if '_EV__EV_' in feature_name:
            ticker_part = feature_name.split('_EV__EV_')[0]
            return f"{ticker_part}: {mapped_name}"
        return mapped_name
    return feature_name

# ===================================================================
# PRIMITIVE SIGNAL GRAMMAR
# This defines the set of rules for converting a continuous feature
# into a binary (True/False) signal.
# ===================================================================

PRIMITIVE_SIGNAL_DEFINITIONS = [
    # Rule 1: Is the feature's value high/low relative to its history?
    {'type': 'percentile', 'operator': '>', 'threshold': 0.80, 'name': 'is_high'},
    {'type': 'percentile', 'operator': '<', 'threshold': 0.20, 'name': 'is_low'},

    # Rule 2: Is the feature experiencing a z-score breakout?
    # We apply this to features that are NOT already z-scores.
    {'type': 'z_score', 'operator': '>', 'threshold': 1.5, 'name': 'z_breakout_pos'},
    {'type': 'z_score', 'operator': '<', 'threshold': -1.5, 'name': 'z_breakout_neg'},

    # Rule 3: For features that ARE z-scores, we can use a simpler threshold.
    {'type': 'value', 'operator': '>', 'threshold': 1.75, 'name': 'is_very_high_z'},
    {'type': 'value', 'operator': '<', 'threshold': -1.75, 'name': 'is_very_low_z'},
]


# ===================================================================
# THE SIGNAL COMPILER
# This function orchestrates the creation of all primitive signals.
# ===================================================================

def compile_signals(feature_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Compiles a feature matrix into a set of binary primitive signals.

    Iterates through each feature and applies the rules defined in the
    PRIMITIVE_SIGNAL_DEFINITIONS grammar to generate a large pool of
    boolean Series.

    Returns:
        A tuple containing:
        - A DataFrame where each column is a binary signal Series.
        - A list of metadata dictionaries, one for each signal, describing how it was created.
    """
    print("Compiling primitive signals from feature matrix...")

    all_signal_series: Dict[str, pd.Series] = {}
    all_signal_metadata: List[Dict] = []
    signal_id_counter = 0

    for feature_name in feature_matrix.columns:
        feature_series = feature_matrix[feature_name].dropna()
        if feature_series.empty:
            continue

        for rule in PRIMITIVE_SIGNAL_DEFINITIONS:
            signal_series = None

            # --- Apply the rule based on its type ---

            if rule['type'] == 'percentile':
                ranks = feature_series.rank(pct=True)
                if rule['operator'] == '>':
                    signal_series = (ranks > rule['threshold'])
                else:
                    signal_series = (ranks < rule['threshold'])

            elif rule['type'] == 'z_score':
                # Only apply this rule to features that aren't already z-scores
                if 'zscore' not in feature_name:
                    z_scores = fcore.zscore_rolling(feature_series, window=60)
                    if rule['operator'] == '>':
                        signal_series = (z_scores > rule['threshold'])
                    else:
                        signal_series = (z_scores < rule['threshold'])

            elif rule['type'] == 'value':
                # Only apply this rule to features that ARE z-scores
                if 'zscore' in feature_name:
                    if rule['operator'] == '>':
                        signal_series = (feature_series > rule['threshold'])
                    else:
                        signal_series = (feature_series < rule['threshold'])

            # --- If a signal was generated, store it ---
            if signal_series is not None and signal_series.any():
                signal_id = f"SIG_{signal_id_counter:05d}"

                # Store the actual boolean data
                all_signal_series[signal_id] = signal_series

                # Store the descriptive metadata
                interpretable_name = _get_interpretable_event_name(feature_name)
                all_signal_metadata.append({
                    'signal_id': signal_id,
                    'feature_name': feature_name,
                    'rule_type': rule['type'],
                    'condition': f"{rule['operator']} {rule['threshold']}",
                    'description': f"{interpretable_name} {rule['name']}"
                })
                signal_id_counter += 1

    # Combine all boolean series into a single dataframe, filling non-triggers with False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        signals_df = pd.DataFrame(all_signal_series).fillna(False)
    signals_df = signals_df.infer_objects(copy=False)

    print(f" Signal compilation complete. Generated {len(all_signal_metadata)} primitive signals.")

    return signals_df, all_signal_metadata