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
    # New mapping based on events.py
    'EV_tail_share': 'Tail Event Share',
    'EV_revision_z': 'Revision Shock Z-Score',
    'EV_time_since_tailshock_60': 'Time Since Tail Shock (60d)',
    'EV_surprise_dispersion_day': 'Surprise Dispersion',
    'EV_signed_surprise_percentile': 'Surprise Percentile (Signed)',
    'EV_revision_conflict': 'Revision Conflict',
    'EV_forward_calendar_heat': 'Calendar Heat',
    'EV_calendar_vacuum': 'Calendar Vacuum',
    'EV_dense_macro_day_score': 'Macro Day Score',
    'EV_top_tier_dominance_share': 'Top Tier Dominance',
    'EV.bucket_inflation_surp': 'Inflation Surprise',
    'EV.bucket_labor_surp': 'Labor Surprise',
    'EV.bucket_growth_surp': 'Growth Surprise',
    'EV.bucket_housing_surp': 'Housing Surprise',
    'EV.bucket_sentiment_surp': 'Sentiment Surprise',
    'EV.bucket_divergence': 'Inflation/Growth Divergence',
    'EV.bucket_inflation_tail_share': 'Inflation Tail Share',
    'EV_after_surprise_z': 'Post-Surprise Z-Score',
    'EV_tail_intensity': 'Tail Intensity',
    'EV_tail_cooldown': 'Tail Cooldown',
    'INF.shadow_cpi_z': 'CPI Shadow Indicator',
    'LAB.shadow_nfp_z': 'NFP Shadow Indicator',
    'EXP.confidence_proxy': 'Expectations Confidence',
    'META.day_reliability_index': 'Data Reliability Index',
}

def _get_interpretable_event_name(feature_name: str) -> str:
    """Convert feature names to more interpretable event types."""
    # Check for event features first
    if feature_name.startswith(('EV_', 'EV.', 'INF.', 'LAB.', 'EXP.', 'META.')):
        # Try to find a matching prefix in our map
        for prefix, readable_name in EVENT_FEATURE_MAPPING.items():
            if feature_name.startswith(prefix):
                # Append any suffixes (like .ewm_hf10.5)
                suffix = feature_name[len(prefix):]
                # Clean up suffix for display
                suffix = suffix.replace("_", " ").replace(".", " ").strip()
                return f"{readable_name} ({suffix})" if suffix else readable_name
        # Fallback for unmapped event features
        return feature_name.replace("_", " ").title()

    # For non-event features, the full feature name is the description.
    # This preserves the ticker prefix for inter-market signals.
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
                    z_scores = fcore.zscore_rolling(feature_series, 60)
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