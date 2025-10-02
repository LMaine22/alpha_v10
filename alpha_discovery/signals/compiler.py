# alpha_discovery/signals/compiler.py

import pandas as pd
import numpy as np
import warnings
import re
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from joblib import Parallel, delayed
from alpha_discovery.features import core as fcore

# Suppress pandas FutureWarnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Cache directory for compiled signals
SIGNALS_CACHE_DIR = Path("cache/signals")
SIGNALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================================
# SOFT-AND INFRASTRUCTURE
# ===================================================================

def _clip01(x: pd.Series) -> pd.Series:
    """Clip values to [0,1] range"""
    return x.clip(lower=0.0, upper=1.0)

def _z_to_prob_tanh(z: pd.Series, scale: float = 2.0) -> pd.Series:
    """
    Map z-score to probability in [0,1] using tanh with adjustable softness.
    Higher scale => softer mapping.
    """
    z = pd.to_numeric(z, errors="coerce")
    return _clip01(0.5 * (1.0 + np.tanh(z / scale)))

def _z_to_prob_sigmoid(z: pd.Series, scale: float = 1.0) -> pd.Series:
    """
    Alternative z->prob using logistic sigmoid.
    """
    z = pd.to_numeric(z, errors="coerce")
    return _clip01(1.0 / (1.0 + np.exp(-z / max(1e-9, scale))))

def _rank_to_prob(cs_rank: pd.Series) -> pd.Series:
    """
    Convert cross-sectional rank in [0,1] or percent [0,100] into [0,1].
    """
    s = pd.to_numeric(cs_rank, errors="coerce")
    if s.max(skipna=True) is not None and s.max(skipna=True) > 1.5:
        s = s / 100.0
    return _clip01(s)

def _soft_and(probs: List[pd.Series]) -> pd.Series:
    """
    Soft-AND via minimum across probabilities. Stable and monotone.
    """
    if len(probs) == 0:
        return pd.Series(dtype=float)
    out = probs[0]
    for p in probs[1:]:
        out = np.minimum(out, p)
    return _clip01(out)

def _soft_or(probs: List[pd.Series]) -> pd.Series:
    """
    Soft-OR via maximum across probabilities.
    """
    if len(probs) == 0:
        return pd.Series(dtype=float)
    out = probs[0]
    for p in probs[1:]:
        out = np.maximum(out, p)
    return _clip01(out)

def _within_window(signal: pd.Series, window: int = 5) -> pd.Series:
    """
    Check if signal was active within the last N bars.
    """
    return signal.rolling(window, min_periods=1).max().astype(float)

def _temporal_cooccur_score(signals: List[pd.Series], window: int = 5) -> pd.Series:
    """
    Temporal co-occurrence score: fraction of signals active within window.
    """
    if not signals:
        return pd.Series(dtype=float)
    
    active_count = sum(_within_window(s, window) for s in signals)
    return active_count / len(signals)

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

def _rolling_pct_rank(s: pd.Series, w: int = 252) -> pd.Series:
    """
    FAST vectorized rolling percentile rank using only information up to time t.
    """
    minp = max(20, w // 5)
    # Vectorized approach: much faster than apply
    result = pd.Series(index=s.index, dtype=float)
    for i in range(minp - 1, len(s)):
        window_data = s.iloc[max(0, i - w + 1):i + 1]
        if len(window_data) >= minp:
            result.iloc[i] = (window_data < window_data.iloc[-1]).sum() / len(window_data)
    return result

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
    Finds and removes identical boolean signal columns that have the same source feature.
    Signals from different tickers or features are kept separate even if they have identical patterns.
    """
    if signals_df.empty:
        return signals_df, metadata_list

    print("  Deduplicating signal set...")
    metadata_map = {m['signal_id']: m for m in metadata_list}
    
    # Create a compound key that includes both the pattern hash AND the source feature
    compound_keys = {}
    unique_features = set()
    
    for signal_id in signals_df.columns:
        # Get the signal's boolean pattern hash
        pattern_hash = pd.util.hash_pandas_object(signals_df[signal_id], index=False).sum()
        
        # Get the source feature name from metadata
        meta = metadata_map.get(signal_id, {})
        feature_name = meta.get('feature_name', 'unknown')
        unique_features.add(feature_name)
        
        # Create compound key: (feature_name, pattern_hash)
        # This ensures signals from different tickers/features are never deduplicated
        compound_key = (feature_name, pattern_hash)
        compound_keys[signal_id] = compound_key
    
    print(f"    Found signals from {len(unique_features)} unique features")
    
    # Group signals by their compound keys
    key_groups = {}
    for signal_id, key in compound_keys.items():
        if key not in key_groups:
            key_groups[key] = []
        key_groups[key].append(signal_id)
    
    unique_signals_metadata = {}
    duplicates_removed = 0
    
    # Process each group - only deduplicate within the same feature
    for compound_key, group_ids in key_groups.items():
        feature_name, _ = compound_key
        
        if len(group_ids) == 1:
            signal_id = group_ids[0]
            unique_signals_metadata[signal_id] = metadata_map[signal_id]
        else:
            # Multiple signals from the SAME feature with the SAME pattern
            # This is a true duplicate - keep the one with the shortest description
            metadatas_in_group = [metadata_map[sid] for sid in group_ids]
            best_signal = min(metadatas_in_group, key=lambda m: len(m.get('description', '')))
            unique_signals_metadata[best_signal['signal_id']] = best_signal
            duplicates_removed += (len(group_ids) - 1)

    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate signal columns (same feature, same pattern).")

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
        {'name': 'is_very_high_z', 'enter': 1.5, 'exit': 1.25},
        {'name': 'is_very_low_z', 'enter': -1.5, 'exit': -1.25},
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
                          macro_momentum_gate: Optional[pd.Series],
                          rank_window: int = 252) -> List[Tuple[str, pd.Series, Dict]]:
    """Process a single feature to generate signals - parallelized."""
    if feature_series.empty or pd.api.types.is_bool_dtype(feature_series.dtype):
        return []

    generated_signals: List[Tuple[str, pd.Series, Dict]] = []
    is_z = _is_z_like(feature_name)
    is_cs = _is_cs_rank(feature_name)

    # 1. Percentile and Mean-Reversion Rules
    if not is_cs:
        ranks = _rolling_pct_rank(feature_series, w=rank_window)
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


# ===================================================================
# CACHING INFRASTRUCTURE
# ===================================================================

def _hash_feature_matrix(feature_matrix: pd.DataFrame) -> str:
    """
    Generate a deterministic hash of the feature matrix for caching.
    
    Hash is based on:
    - Column names (features) - deterministic from registry
    - Index (dates) - from master_df
    
    Note: We don't include sample values because feature computation is 
    deterministic from master_df. This allows us to check cache without
    building features first (huge performance win).
    
    Args:
        feature_matrix: Feature DataFrame to hash
        
    Returns:
        Hexadecimal hash string
    """
    hash_components = []
    
    # 1. Column names (sorted for determinism)
    cols_str = ",".join(sorted(feature_matrix.columns))
    hash_components.append(cols_str.encode())
    
    # 2. Index dates (first, last, length)
    idx_str = f"{feature_matrix.index[0]}_{feature_matrix.index[-1]}_{len(feature_matrix.index)}"
    hash_components.append(idx_str.encode())
    
    # Combine and hash
    combined = b"||".join(hash_components)
    return hashlib.sha256(combined).hexdigest()[:16]  # Use first 16 chars for readability


def _hash_columns_only(feature_matrix: pd.DataFrame) -> str:
    """
    Generate hash from columns only (ignoring index/dates).
    Used to find compatible caches for incremental updates.
    """
    cols_str = ",".join(sorted(feature_matrix.columns))
    return hashlib.sha256(cols_str.encode()).hexdigest()[:16]


def _find_compatible_cache(feature_columns: List[str], current_index: pd.Index) -> Tuple[Optional[str], Optional[pd.Index]]:
    """
    Find most recent compatible cache (same feature columns, overlapping dates).
    Returns (cache_key, cached_index) if found, else (None, None).
    """
    compatible_caches = []
    
    sorted_cols = sorted(feature_columns)
    cols_hash = hashlib.sha256(",".join(sorted_cols).encode()).hexdigest()[:16]
    
    print(f"  🔍 Searching for compatible caches (cols_hash={cols_hash[:8]}...)...")
    
    for cache_file in SIGNALS_CACHE_DIR.glob(f"signals_*.pkl"):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Get cached feature columns (if stored)
            cached_feature_cols = cached_data.get('feature_columns')
            cached_feature_idx = cached_data.get('feature_index')
            
            if cached_feature_cols is None or cached_feature_idx is None:
                print(f"    Checking {cache_file.name}: OLD CACHE FORMAT (no feature metadata)")
                continue
            
            # Check if columns match
            cached_cols_hash = hashlib.sha256(",".join(sorted(cached_feature_cols)).encode()).hexdigest()[:16]
            
            print(f"    Checking {cache_file.name}: cols_match={cached_cols_hash == cols_hash}, rows={len(cached_feature_idx)}")
            
            if cached_cols_hash == cols_hash:
                # Check if cache is a prefix of current data
                if len(cached_feature_idx) < len(current_index):
                    # Check if all cached dates exist in current index
                    if cached_feature_idx[0] == current_index[0] and cached_feature_idx[-1] in current_index:
                        print(f"      ✅ Compatible! ({len(cached_feature_idx)} rows, ending {cached_feature_idx[-1]})")
                        compatible_caches.append((cached_data['cache_key'], cached_feature_idx, cache_file.stat().st_mtime))
                    else:
                        print(f"      ❌ Index mismatch: first={cached_feature_idx[0] == current_index[0]}, last_in_current={cached_feature_idx[-1] in current_index}")
                else:
                    print(f"      ❌ Not a prefix (cached={len(cached_feature_idx)} >= current={len(current_index)})")
        except Exception as e:
            print(f"    ⚠️ Error reading {cache_file.name}: {e}")
            continue
    
    if compatible_caches:
        # Sort by modification time, return most recent
        compatible_caches.sort(key=lambda x: x[2], reverse=True)
        print(f"  ✅ Found {len(compatible_caches)} compatible cache(s), using most recent")
        return compatible_caches[0][0], compatible_caches[0][1]
    
    print(f"  ❌ No compatible caches found")
    return None, None


def _load_cached_signals(cache_key: str) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]]]:
    """
    Load cached signals from disk.
    
    Args:
        cache_key: Hash key for the cached signals
        
    Returns:
        (signals_df, signals_meta) if cache exists, else (None, None)
    """
    cache_file = SIGNALS_CACHE_DIR / f"signals_{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            signals_df = cached_data['signals_df']
            signals_meta = cached_data['signals_meta']
            
            print(f"  ✅ Loaded cached signals from: {cache_file.name}")
            print(f"     Cache key: {cache_key}")
            print(f"     Signals: {len(signals_df.columns)}, Observations: {len(signals_df)}")
            
            return signals_df, signals_meta
        except Exception as e:
            print(f"  ⚠️  Error loading cache: {e}")
            return None, None
    
    return None, None


def _save_cached_signals(cache_key: str, signals_df: pd.DataFrame, signals_meta: List[Dict], feature_columns: List[str] = None, feature_index: pd.Index = None):
    """
    Save compiled signals to cache.
    
    Args:
        cache_key: Hash key for the cached signals
        signals_df: Compiled signals DataFrame
        signals_meta: Metadata for signals
        feature_columns: List of feature column names used to generate signals (for incremental cache)
        feature_index: Date index of feature matrix (for incremental cache)
    """
    cache_file = SIGNALS_CACHE_DIR / f"signals_{cache_key}.pkl"
    
    try:
        cached_data = {
            'signals_df': signals_df,
            'signals_meta': signals_meta,
            'cache_key': cache_key,
            'feature_columns': feature_columns,  # NEW: Store input feature columns
            'feature_index': feature_index        # NEW: Store input date index
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"  💾 Saved signals to cache: {cache_file.name}")
    except Exception as e:
        print(f"  ⚠️  Error saving cache: {e}")


def check_signals_cache(master_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]]]:
    """
    Check if signals are cached for this master_df WITHOUT building features.
    
    Generates a lightweight hash from master_df metadata to check cache,
    avoiding expensive feature computation when cache exists.
    
    Args:
        master_df: Raw master DataFrame
        
    Returns:
        (signals_df, signals_meta) if cache hit, else (None, None)
    """
    # Build lightweight feature matrix metadata (column names + index only)
    # This is MUCH faster than computing actual feature values
    print("Checking signal cache...")
    
    # Get expected feature columns (fast - just registry lookup)
    from alpha_discovery.features.registry import FEAT
    try:
        from alpha_discovery.features.registry import PAIR_SPECS
        pairwise_features = list(PAIR_SPECS.keys())
    except ImportError:
        pairwise_features = []
    
    expected_columns = sorted(list(FEAT.keys()) + pairwise_features)
    
    # Build quick hash from metadata (no actual computation)
    hash_components = []
    hash_components.append(",".join(expected_columns).encode())
    idx_str = f"{master_df.index[0]}_{master_df.index[-1]}_{len(master_df.index)}"
    hash_components.append(idx_str.encode())
    
    combined = b"||".join(hash_components)
    cache_key = hashlib.sha256(combined).hexdigest()[:16]
    
    print(f"  Quick hash: {cache_key}")
    
    # Try to load from cache (exact match)
    cached_signals, cached_meta = _load_cached_signals(cache_key)
    if cached_signals is not None:
        return cached_signals, cached_meta
    
    # Try incremental cache (same features, but different dates)
    compat_key, compat_idx = _find_compatible_cache(expected_columns, master_df.index)
    
    if compat_key is not None:
        print(f"  📦 Found compatible cache - will use incremental update")
        print(f"  🔄 Need to process {len(master_df) - len(compat_idx)} new rows")
        # Return None to trigger feature building, but compile_signals will use incremental
        return None, None
    
    print("  No cache found - will build features and compile signals")
    return None, None


def compile_signals(feature_matrix: pd.DataFrame, use_cache: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Compiles a feature matrix into a rich set of binary primitive signals using
    an expanded grammar including hysteresis, dwell times, and contextual gates.
    
    Args:
        feature_matrix: DataFrame with features
        use_cache: If True, check cache before recomputing (default: True)
        
    Returns:
        (signals_df, signals_meta) - DataFrame of boolean signals and metadata list
    """
    print("Compiling primitive signals from feature matrix...")
    
    # Generate cache key from feature matrix
    cache_key = _hash_feature_matrix(feature_matrix)
    print(f"  Feature matrix hash: {cache_key}")
    
    # Try to load from cache
    if use_cache:
        # First try exact match
        cached_signals, cached_meta = _load_cached_signals(cache_key)
        if cached_signals is not None:
            return cached_signals, cached_meta
        
        # No exact match - try incremental cache (same features, older dates)
        compat_key, compat_idx = _find_compatible_cache(list(feature_matrix.columns), feature_matrix.index)
        
        if compat_key is not None:
            print(f"  📦 Found compatible cache with {len(compat_idx)} rows")
            print(f"  🔄 Computing signals for {len(feature_matrix) - len(compat_idx)} new rows...")
            
            # Load compatible cache
            old_signals, old_meta = _load_cached_signals(compat_key)
            
            # Compute signals only for new dates
            new_rows = feature_matrix.loc[feature_matrix.index > compat_idx[-1]]
            
            if len(new_rows) > 0:
                # Recursively call compile_signals for new rows only (without cache)
                new_signals, new_meta = compile_signals(new_rows, use_cache=False)
                
                # Merge old + new
                combined_signals = pd.concat([old_signals, new_signals], axis=0)
                
                # Keep metadata from old cache (signal_ids are already assigned)
                # Note: new_meta will have same signal structure, just new dates
                combined_meta = old_meta  # Metadata doesn't change with new dates
                
                # Save combined result with new cache key
                _save_cached_signals(cache_key, combined_signals, combined_meta, 
                                    feature_columns=list(feature_matrix.columns),
                                    feature_index=feature_matrix.index)
                
                print(f"  ✅ Incremental update complete: {len(combined_signals)} total rows")
                return combined_signals, combined_meta
    
    print("  Cache miss - computing signals from scratch...")

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

    # --- Parallel processing with chunking for better performance ---
    chunk_size = max(1, len(features_to_process) // 16)  # 16 chunks
    feature_chunks = [features_to_process[i:i+chunk_size] for i in range(0, len(features_to_process), chunk_size)]
    
    def _process_feature_chunk(chunk):
        chunk_results = []
        for feature_name, feature_series in chunk:
            result = _process_single_feature(
                feature_name, feature_series, liquid_flag, opt_avail_flag, 
                macro_vacuum_gate, macro_momentum_gate, 252
            )
            chunk_results.extend(result)
        return chunk_results
    
    # Parallel processing with joblib - using threading backend for pandas compatibility
    all_processed_signals = Parallel(n_jobs=-1, backend='threading', verbose=5)(
        delayed(_process_feature_chunk)(chunk) for chunk in feature_chunks
    )

    # --- Flatten and collect results (optimized) ---
    all_signals: Dict[str, pd.Series] = {}
    all_metadata: List[Dict] = []
    signal_id_counter = 0

    # Flatten in one pass instead of nested loops
    for chunk_results in all_processed_signals:
        for signal_series, meta in chunk_results:
            if signal_series.any():  # Skip empty signals early
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

    print(f"  Generated {len(all_signals)} total signals before deduplication.")

    # Deduplicate identical signals
    final_signals_df, final_metadata = _deduplicate_signals(signals_df, all_metadata)

    print(f"Signal compilation complete. Generated {len(final_metadata)} unique primitive signals.")
    
    # Save to cache for future runs
    if use_cache:
        _save_cached_signals(cache_key, final_signals_df, final_metadata,
                            feature_columns=list(feature_matrix.columns),
                            feature_index=feature_matrix.index)
    
    return final_signals_df, final_metadata


# ===================================================================
# COMPOSITE SIGNAL BUILDERS
# ===================================================================

def build_event_soft_and_signal(
    feature_matrix: pd.DataFrame,
    fold_id: str = "default",
    event_intensity_threshold: float = 0.3,
    temporal_window: int = 5
) -> Tuple[pd.Series, Dict]:
    """
    Build event-driven soft-AND composite signal.
    """
    # Event intensity features
    event_features = [col for col in feature_matrix.columns if col.startswith("EV_")]
    if not event_features:
        return pd.Series(False, index=feature_matrix.index), {"type": "event_soft_and", "status": "no_events"}
    
    # Convert event features to probabilities
    event_probs = []
    for feat in event_features:
        if _is_z_like(feat):
            prob = _z_to_prob_tanh(feature_matrix[feat], 2.0)
        else:
            prob = _rank_to_prob(feature_matrix[feat])
        event_probs.append(prob)
    
    # Soft-AND of event probabilities
    event_consensus = _soft_and(event_probs)
    
    # Apply threshold
    signal = event_consensus > event_intensity_threshold
    
    meta = {
        "type": "event_soft_and",
        "event_features": len(event_features),
        "threshold": event_intensity_threshold,
        "temporal_window": temporal_window
    }
    
    return signal, meta

def build_options_soft_and_signal(
    feature_matrix: pd.DataFrame,
    fold_id: str = "default",
    options_threshold: float = 0.4,
    temporal_window: int = 3
) -> Tuple[pd.Series, Dict]:
    """
    Build options-driven soft-AND composite signal.
    """
    # Options features
    options_features = [col for col in feature_matrix.columns if "opt." in col.lower()]
    if not options_features:
        return pd.Series(False, index=feature_matrix.index), {"type": "options_soft_and", "status": "no_options"}
    
    # Convert options features to probabilities
    options_probs = []
    for feat in options_features:
        if _is_z_like(feat):
            prob = _z_to_prob_tanh(feature_matrix[feat], 1.5)
        else:
            prob = _rank_to_prob(feature_matrix[feat])
        options_probs.append(prob)
    
    # Soft-AND of options probabilities
    options_consensus = _soft_and(options_probs)
    
    # Apply threshold
    signal = options_consensus > options_threshold
    
    meta = {
        "type": "options_soft_and",
        "options_features": len(options_features),
        "threshold": options_threshold,
        "temporal_window": temporal_window
    }
    
    return signal, meta

def build_adaptive_cooldown_signal(
    base_signal: pd.Series,
    volatility: pd.Series,
    base_cooldown: int = 5,
    volatility_multiplier: float = 2.0
) -> pd.Series:
    """
    Build adaptive cooldown signal based on volatility.
    """
    # Calculate adaptive cooldown based on volatility
    vol_z = volatility.rolling(21).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    adaptive_cooldown = base_cooldown * (1 + vol_z.abs() * volatility_multiplier)
    
    # Apply adaptive cooldown
    signal = base_signal.copy()
    for i in range(1, len(signal)):
        if signal.iloc[i-1] and not signal.iloc[i]:
            # Signal turned off, apply cooldown
            cooldown_period = int(adaptive_cooldown.iloc[i]) if not pd.isna(adaptive_cooldown.iloc[i]) else base_cooldown
            end_idx = min(i + cooldown_period, len(signal))
            signal.iloc[i:end_idx] = False
    
    return signal