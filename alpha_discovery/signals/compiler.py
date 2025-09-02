# alpha_discovery/signals/compiler.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from ..features import core as fcore
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
                all_signal_metadata.append({
                    'signal_id': signal_id,
                    'feature_name': feature_name,
                    'rule_type': rule['type'],
                    'condition': f"{rule['operator']} {rule['threshold']}",
                    'description': f"{feature_name} {rule['name']}"
                })
                signal_id_counter += 1

    # Combine all boolean series into a single dataframe, filling non-triggers with False
    signals_df = pd.DataFrame(all_signal_series).fillna(False)
    signals_df = signals_df.infer_objects(copy=False)
    # (or opt in globally: pd.set_option('future.no_silent_downcasting', True))

    print(f" Signal compilation complete. Generated {len(all_signal_metadata)} primitive signals.")

    return signals_df, all_signal_metadata