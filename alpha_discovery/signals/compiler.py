# alpha_discovery/signals/compiler.py

import pandas as pd
import numpy as np
import warnings
import re
from typing import List, Dict, Tuple, Optional
from joblib import Parallel, delayed
from ..features import core as fcore

# Suppress pandas FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# ===================================================================
# UTILITY & HELPER FUNCTIONS
# ===================================================================

def _is_z_like(name: str) -> bool:
    """
    Enhanced check to see if a feature is a z-score based on naming conventions.
    Z-like if: ends with _z or ._z, contains _z/ or .z, or contains 'zscore'.
    """
    name_lower = name.lower()
    return (
        name_lower.endswith(('_z', '._z')) or
        '_z/' in name_lower or
        '.z' in name_lower or
        'zscore' in name_lower
    )

def _is_cs_rank(name: str) -> bool:
    """Checks if a feature is a cross-sectional rank."""
    return name.lower().startswith('cs.rank')

def _parse_window_from_name(name: str) -> Optional[int]:
    """Parses a lookback window (e.g., _63, _21d) from a feature name."""
    matches = re.findall(r'[._](\d+)[d]?', name)
    if matches:
        # Take the last number found as it's most likely the window
        return int(matches[-1])
    return None

def apply_hysteresis_and_dwell(
    series: pd.Series,
    enter_threshold: float,
    exit_threshold: float,
    min_dwell: int,
    cooldown: int,
    is_upper_band: bool
) -> pd.Series:
    """
    Applies two-sided hysteresis, minimum dwell time, and a cooldown period to a raw series.

    Args:
        series: The continuous input series (e.g., percentiles or z-scores).
        enter_threshold: The level the series must cross to activate the signal.
        exit_threshold: The level the series must cross to deactivate the signal.
        min_dwell: The minimum number of consecutive bars the signal must be active.
        cooldown: The number of bars the signal must remain off after deactivating.
        is_upper_band: True for signals like "is_high" (> enter), False for "is_low" (< enter).

    Returns:
        A boolean pd.Series representing the final, stabilized signal.
    """
    if series.empty:
        return pd.Series(dtype=bool)

    # 1. Hysteresis State Machine
    if is_upper_band:
        enter_signal = series > enter_threshold
        exit_signal = series < exit_threshold
    else:  # Lower band
        enter_signal = series < enter_threshold
        exit_signal = series > exit_threshold

    state = pd.Series(np.nan, index=series.index)
    state[enter_signal] = 1.0
    state[exit_signal] = 0.0
    state = state.ffill().fillna(0).astype(bool)

    if not state.any():
        return state

    # 2. Dwell Time Enforcement
    # A block of 'True' values is kept only if its length is >= min_dwell.
    blocks = (state != state.shift()).cumsum()
    block_sizes = state.groupby(blocks).transform('size')
    dwell_ok = (block_sizes >= min_dwell) & state

    if not dwell_ok.any():
        return dwell_ok

    # 3. Cooldown Enforcement
    # Find where signal blocks end.
    exits = dwell_ok & ~dwell_ok.shift(-1, fill_value=False)
    exit_indices = np.where(exits)[0]

    if exit_indices.size == 0:
        return dwell_ok # No exits means no cooldowns to apply

    # Create a mask to suppress signals during cooldown periods.
    # True means allowed, False means suppressed.
    cooldown_mask = np.full(len(series), True)
    for idx in exit_indices:
        # The cooldown starts on the bar AFTER the exit.
        start_cool = idx + 1
        end_cool = start_cool + cooldown
        if start_cool < len(cooldown_mask):
            cooldown_mask[start_cool:end_cool] = False

    return dwell_ok & pd.Series(cooldown_mask, index=series.index)


def _deduplicate_signals(
    signals_df: pd.DataFrame,
    metadata_list: List[Dict]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Finds and removes identical boolean signal columns, keeping the one
    with the shortest description.
    """
    if signals_df.empty:
        return signals_df, metadata_list

    print("  Deduplicating signal set...")
    metadata_map = {m['signal_id']: m for m in metadata_list}
    hashes = signals_df.apply(lambda s: pd.util.hash_pandas_object(s.values, index=False).sum(), axis=0)

    unique_signals_metadata = {}
    duplicates_removed = 0

    # Group by hash and select the best candidate from each group
    for _, group_ids in hashes.groupby(hashes).groups.items():
        if len(group_ids) == 1:
            signal_id = group_ids[0]
            unique_signals_metadata[signal_id] = metadata_map[signal_id]
        else:
            # Duplicate hash found, find the best one to keep (shortest description)
            metadatas_in_group = [metadata_map[sid] for sid in group_ids]
            best_signal = min(metadatas_in_group, key=lambda m: len(m.get('description', '')))
            unique_signals_metadata[best_signal['signal_id']] = best_signal
            duplicates_removed += (len(group_ids) - 1)

    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate signal columns.")

    final_metadata = list(unique_signals_metadata.values())
    final_signal_ids = list(unique_signals_metadata.keys())
    final_signals_df = signals_df[final_signal_ids]

    return final_signals_df, final_metadata

# ===================================================================
# EVENT FEATURE MAPPING
# ===================================================================
EVENT_FEATURE_MAPPING = {
    'EV_tail_share': 'Tail Event Share', 'EV_revision_z': 'Revision Shock Z-Score',
    'EV_time_since_tailshock_60': 'Time Since Tail Shock (60d)',
    'EV_surprise_dispersion_day': 'Surprise Dispersion',
    'EV_signed_surprise_percentile': 'Surprise Percentile (Signed)',
    'EV_revision_conflict': 'Revision Conflict', 'EV_forward_calendar_heat': 'Calendar Heat',
    'EV_calendar_vacuum': 'Calendar Vacuum', 'EV_dense_macro_day_score': 'Macro Day Score',
    'EV_top_tier_dominance_share': 'Top Tier Dominance', 'EV.bucket_inflation_surp': 'Inflation Surprise',
    'EV.bucket_labor_surp': 'Labor Surprise', 'EV.bucket_growth_surp': 'Growth Surprise',
    'EV.bucket_housing_surp': 'Housing Surprise', 'EV.bucket_sentiment_surp': 'Sentiment Surprise',
    'EV.bucket_divergence': 'Inflation/Growth Divergence',
    'EV.bucket_inflation_tail_share': 'Inflation Tail Share',
    'EV_after_surprise_z': 'Post-Surprise Z-Score', 'EV_tail_intensity': 'Tail Intensity',
    'EV_tail_cooldown': 'Tail Cooldown', 'INF.shadow_cpi_z': 'CPI Shadow Indicator',
    'LAB.shadow_nfp_z': 'NFP Shadow Indicator', 'EXP.confidence_proxy': 'Expectations Confidence',
    'META.day_reliability_index': 'Data Reliability Index',
}

def _get_interpretable_event_name(feature_name: str) -> str:
    """Convert feature names to more interpretable event types."""
    if any(feature_name.startswith(p) for p in ['EV_', 'EV.', 'INF.', 'LAB.', 'EXP.', 'META.']):
        for prefix, readable_name in EVENT_FEATURE_MAPPING.items():
            if feature_name.startswith(prefix):
                suffix = feature_name[len(prefix):].replace("_", " ").replace(".", " ").strip()
                return f"{readable_name} ({suffix})" if suffix else readable_name
        return feature_name.replace("_", " ").title()
    return feature_name

# ===================================================================
# PRIMITIVE SIGNAL GRAMMAR & CONFIGURATION
# ===================================================================

# -- Signal Stability Parameters --
MIN_DWELL_TIME = 3  # A signal must be active for at least this many bars
COOLDOWN_PERIOD = 3 # A signal cannot re-fire for this many bars after ending

# -- Rule Definitions --
PRIMITIVE_RULES = {
    # Rules for percentile-ranked features
    'percentile': [
        {'name': 'is_high', 'enter': 0.80, 'exit': 0.70},
        {'name': 'is_low', 'enter': 0.20, 'exit': 0.30},
        {'name': 'is_very_high_p99', 'enter': 0.99, 'exit': 0.95},
        {'name': 'is_very_low_p01', 'enter': 0.01, 'exit': 0.05},
    ],
    # Rules for features that are already z-scores
    'z_score_native': [
        {'name': 'is_very_high_z', 'enter': 1.75, 'exit': 1.25},
        {'name': 'is_very_low_z', 'enter': -1.75, 'exit': -1.25},
    ],
    # Rules for features we convert to z-scores
    'z_score_derived': [
        {'name': 'z_breakout_pos', 'enter': 1.5, 'exit': 1.0},
        {'name': 'z_breakout_neg', 'enter': -1.5, 'exit': -1.0},
    ],
    # Rules for cross-sectional rank features
    'cs_quantile': [
        {'name': 'is_top_quintile', 'operator': '>=', 'threshold': 0.8},
        {'name': 'is_bottom_quintile', 'operator': '<=', 'threshold': 0.2},
    ],
    # Rules for mean-reversion events
    'mean_reversion': [
        {'name': 'reverts_from_high', 'extreme_thresh': 0.99},
        {'name': 'reverts_from_low', 'extreme_thresh': 0.01},
    ]
}

# ===================================================================
# THE SIGNAL COMPILER
# ===================================================================

def _process_single_feature(feature_name: str, feature_series: pd.Series, 
                          liquid_flag: pd.Series, opt_avail_flag: pd.Series,
                          macro_vacuum_gate: Optional[pd.Series], 
                          macro_momentum_gate: Optional[pd.Series]) -> List[Tuple[str, pd.Series, Dict]]:
    """Process a single feature to generate signals - parallelized."""
    if feature_series.empty or pd.api.types.is_bool_dtype(feature_series.dtype):
        return []

    generated_signals: List[Tuple[str, pd.Series, Dict]] = []
    is_z = _is_z_like(feature_name)
    is_cs = _is_cs_rank(feature_name)

    # 1. Percentile and Mean-Reversion Rules
    if not is_cs:
        ranks = feature_series.rank(pct=True)
        for rule in PRIMITIVE_RULES['percentile']:
            is_upper = rule['enter'] > 0.5
            sig = apply_hysteresis_and_dwell(
                ranks, rule['enter'], rule['exit'], MIN_DWELL_TIME, COOLDOWN_PERIOD, is_upper
            )
            meta = {'rule_type': 'percentile', 'condition': f"enter_{rule['enter']}_exit_{rule['exit']}", 'name': rule['name']}
            generated_signals.append((feature_name, sig, meta))

        for rule in PRIMITIVE_RULES['mean_reversion']:
            if rule['name'] == 'reverts_from_high':
                extreme = ranks > rule['extreme_thresh']
                sig = extreme.shift(1, fill_value=False) & ~extreme
            else: # reverts_from_low
                extreme = ranks < rule['extreme_thresh']
                sig = extreme.shift(1, fill_value=False) & ~extreme
            meta = {'rule_type': 'mean_reversion', 'condition': f"cross_back_{rule['extreme_thresh']}", 'name': rule['name']}
            generated_signals.append((feature_name, sig, meta))

    # 2. Z-Score Rules
    if is_z: # Native Z-scores
        for rule in PRIMITIVE_RULES['z_score_native']:
            is_upper = rule['enter'] > 0
            sig = apply_hysteresis_and_dwell(
                feature_series, rule['enter'], rule['exit'], MIN_DWELL_TIME, COOLDOWN_PERIOD, is_upper
            )
            meta = {'rule_type': 'z_score_native', 'condition': f"enter_{rule['enter']}_exit_{rule['exit']}", 'name': rule['name']}
            generated_signals.append((feature_name, sig, meta))
    elif not is_cs: # Derived Z-scores (don't z-score a rank)
        z_scores = fcore.zscore_rolling(feature_series, 60)
        for rule in PRIMITIVE_RULES['z_score_derived']:
            is_upper = rule['enter'] > 0
            sig = apply_hysteresis_and_dwell(
                z_scores, rule['enter'], rule['exit'], MIN_DWELL_TIME, COOLDOWN_PERIOD, is_upper
            )
            meta = {'rule_type': 'z_score_derived', 'condition': f"enter_{rule['enter']}_exit_{rule['exit']}", 'name': rule['name']}
            generated_signals.append((feature_name, sig, meta))

    # 3. Cross-Sectional Quantile Rules
    if is_cs:
        for rule in PRIMITIVE_RULES['cs_quantile']:
            sig = feature_series >= rule['threshold'] if rule['operator'] == '>=' else feature_series <= rule['threshold']
            meta = {'rule_type': 'cs_quantile', 'condition': f"{rule['operator']} {rule['threshold']}", 'name': rule['name']}
            generated_signals.append((feature_name, sig, meta))

    # Process and gate the signals
    processed_signals = []
    for feature_name, signal_series, base_meta in generated_signals:
        if not signal_series.any():
            continue

        # Apply base liquidity gate
        gated_signal = signal_series & liquid_flag
        applied_gates = ['liquid']

        # Apply options availability gate for options-driven features
        if 'opt.' in feature_name.lower():
            gated_signal &= opt_avail_flag
            applied_gates.append('options')
        
        # Store the baseline gated signal
        if gated_signal.any():
            interpretable_name = _get_interpretable_event_name(feature_name)
            meta = {
                'feature_name': feature_name,
                'rule_type': base_meta['rule_type'],
                'condition': base_meta['condition'],
                'gates': ','.join(applied_gates),
                'window': _parse_window_from_name(feature_name),
                'description': f"{interpretable_name} {base_meta['name']}"
            }
            processed_signals.append((gated_signal, meta))

        # Create macro-gated versions
        if macro_vacuum_gate is not None and (gated_signal & macro_vacuum_gate).any():
            macro_gates = applied_gates + ['macro_vacuum']
            meta = {
                'feature_name': feature_name,
                'rule_type': base_meta['rule_type'],
                'condition': base_meta['condition'],
                'gates': ','.join(macro_gates),
                'window': _parse_window_from_name(feature_name),
                'description': f"{interpretable_name} {base_meta['name']} (in Macro Vacuum)"
            }
            processed_signals.append((gated_signal & macro_vacuum_gate, meta))

        if macro_momentum_gate is not None and (gated_signal & macro_momentum_gate).any():
            macro_gates = applied_gates + ['macro_momentum']
            meta = {
                'feature_name': feature_name,
                'rule_type': base_meta['rule_type'],
                'condition': base_meta['condition'],
                'gates': ','.join(macro_gates),
                'window': _parse_window_from_name(feature_name),
                'description': f"{interpretable_name} {base_meta['name']} (with Macro Momentum)"
            }
            processed_signals.append((gated_signal & macro_momentum_gate, meta))

    return processed_signals

def compile_signals(feature_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Compiles a feature matrix into a rich set of binary primitive signals using
    an expanded grammar including hysteresis, dwell times, and contextual gates.
    """
    print("Compiling primitive signals from feature matrix...")

    # --- Prepare Gating Series ---
    liquid_flag = feature_matrix.get('liquid_flag', pd.Series(True, index=feature_matrix.index))
    opt_avail_flag = feature_matrix.get('opt_avail_flag', pd.Series(True, index=feature_matrix.index))

    # Optional macro gates
    ev_density = feature_matrix.get('EV_week_density')
    macro_vacuum_gate = ev_density < ev_density.median() if ev_density is not None else None
    
    infl_mem = feature_matrix.get('infl_mem_hf21')
    macro_momentum_gate = infl_mem > infl_mem.median() if infl_mem is not None else None

    # --- Prepare features for parallel processing ---
    features_to_process = []
    for feature_name in feature_matrix.columns:
        if feature_name in ['liquid_flag', 'opt_avail_flag', 'EV_week_density', 'infl_mem_hf21']:
            continue
        feature_series = feature_matrix[feature_name].dropna()
        if not feature_series.empty and not pd.api.types.is_bool_dtype(feature_series.dtype):
            features_to_process.append((feature_name, feature_series))

    print(f"  Processing {len(features_to_process)} features in parallel...")

    # --- Parallel processing ---
    all_processed_signals = Parallel(n_jobs=-1, batch_size="auto")(
        delayed(_process_single_feature)(
            feature_name, feature_series, liquid_flag, opt_avail_flag, 
            macro_vacuum_gate, macro_momentum_gate
        ) for feature_name, feature_series in features_to_process
    )

    # --- Flatten and collect results ---
    all_signals: Dict[str, pd.Series] = {}
    all_metadata: List[Dict] = []
    signal_id_counter = 0

    for processed_signals in all_processed_signals:
        for signal_series, meta in processed_signals:
            signal_id = f"SIG_{signal_id_counter:05d}"
            all_signals[signal_id] = signal_series
            meta['signal_id'] = signal_id
            all_metadata.append(meta)
            signal_id_counter += 1

    # --- Finalize and Clean Up ---
    if not all_signals:
        print("Warning: No signals were generated. Check feature matrix and rules.")
        return pd.DataFrame(), []

    # Combine all boolean series into a single dataframe
    signals_df = pd.DataFrame(all_signals).fillna(False).astype(bool)

    # Deduplicate identical signals
    final_signals_df, final_metadata = _deduplicate_signals(signals_df, all_metadata)

    print(f"Signal compilation complete. Generated {len(final_metadata)} unique primitive signals.")
    return final_signals_df, final_metadata