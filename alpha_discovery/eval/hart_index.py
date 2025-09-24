"""
Hart Index: A comprehensive 0-100 trust score for trading setups.

This module calculates a final cumulative score that represents how trustworthy
a trading setup is based on all available metrics from the evaluation pipeline.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

from ..config import settings


@dataclass
class HartIndexComponents:
    """Components that make up the Hart Index with their weights."""
    # Performance components (40% total weight)
    edge_performance: float = 0.20  # CRPS & pinball loss
    information_quality: float = 0.15  # Information gain
    risk_reward: float = 0.05  # Expected returns, P_up/P_down ratios
    
    # Robustness components (30% total weight)
    statistical_significance: float = 0.12  # Bootstrap p-values
    support: float = 0.10             # Out-of-sample trigger count
    sensitivity_resilience: float = 0.08  # Sensitivity to perturbations
    
    # Complexity & Causality (15% total weight)
    causality: float = 0.08           # Transfer Entropy & Granger Causality
    signal_quality: float = 0.04      # Redundancy (MI)
    complexity_balance: float = 0.03  # Sample Entropy & DFA
    
    # Live Trading Readiness (15% total weight)
    trigger_reliability: float = 0.08  # Trigger rates, dormancy flags
    regime_coverage: float = 0.07  # Works across different market conditions


def _sigmoid_transform(x: float, center: float = 0.5, steepness: float = 10) -> float:
    """Apply sigmoid transformation to map values to (0,1) range."""
    return 1 / (1 + np.exp(-steepness * (x - center)))


def _percentile_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize series to 0-1 using percentile ranks."""
    if series.empty or series.isna().all():
        return pd.Series(0.5, index=series.index)
    
    # Use rank with method='average' to handle ties
    ranks = series.rank(pct=True, method='average', ascending=not invert)
    return ranks.fillna(0.5)


def _get_metric_series(df: pd.DataFrame, key: str, default: float = 0.0) -> pd.Series:
    """Safely get a metric series, filling NaNs with the median of the series."""
    if key not in df.columns:
        return pd.Series(default, index=df.index)
    
    series = df[key]
    if series.isna().any():
        median_val = series.median()
        # If the entire series is NaN, median will be NaN. Use the provided default.
        fill_value = median_val if pd.notna(median_val) else default
        return series.fillna(fill_value)
    return series


def _calculate_performance_score(df: pd.DataFrame) -> pd.Series:
    """Calculate performance component of Hart Index."""
    # Edge metrics (lower is better for CRPS)
    crps_score = _percentile_normalize(_get_metric_series(df, 'edge_crps_raw', 0.1), invert=True)
    pinball_score = _percentile_normalize(
        (_get_metric_series(df, 'edge_pin_q10_raw', 0.1) + _get_metric_series(df, 'edge_pin_q90_raw', 0.1)) / 2, 
        invert=True
    )
    
    # Information metrics (higher is better)
    ig_score = _percentile_normalize(_get_metric_series(df, 'edge_ig_raw', 0.0))
    
    # Risk-reward metrics
    e_move = _get_metric_series(df, 'E_move', 0.0)
    p_up = _get_metric_series(df, 'P_up', 0.5)
    p_down = _get_metric_series(df, 'P_down', 0.5)
    
    # Calculate directional confidence
    directional_confidence = np.abs(p_up - p_down) / (p_up + p_down + 1e-9)
    risk_reward_score = _sigmoid_transform(directional_confidence, center=0.2, steepness=8)
    
    # Combine performance metrics (weights sum to 1.0)
    performance_score = (
        0.40 * crps_score +
        0.30 * pinball_score +
        0.20 * ig_score +
        0.10 * risk_reward_score
    )
    
    return performance_score


def _calculate_robustness_score(df: pd.DataFrame) -> pd.Series:
    """Calculate robustness component of Hart Index."""
    # Statistical significance
    bootstrap_p = _get_metric_series(df, 'bootstrap_p_value_raw', 0.5)
    significance_score = _sigmoid_transform(1 - bootstrap_p, center=0.9, steepness=20)
    
    # Support metrics
    n_triggers = _get_metric_series(df, 'n_trig_oos', 10)
    support_score = _sigmoid_transform(n_triggers / 50, center=0.5, steepness=3)
    
    # Sensitivity metrics (lower is better)
    sensitivity_delta = _get_metric_series(df, 'sensitivity_delta_edge_raw', 0.1)
    sensitivity_score = _sigmoid_transform(1 / (1 + sensitivity_delta), center=0.7, steepness=5)
    
    # Combine robustness metrics (weights sum to 1.0)
    robustness_score = (
        0.40 * significance_score +
        0.35 * support_score +
        0.25 * sensitivity_score
    )
    
    return robustness_score


def _calculate_complexity_score(df: pd.DataFrame) -> pd.Series:
    """Calculate complexity & causality component of Hart Index."""
    # 1. Signal Quality (Redundancy) - lower is better
    redundancy_mi = _get_metric_series(df, 'redundancy_mi_raw', 0.5)
    redundancy_score = _sigmoid_transform(1 - redundancy_mi, center=0.7, steepness=5)

    # 2. Causality - higher TE is better, lower Granger p-value is better
    te_score = _percentile_normalize(_get_metric_series(df, 'transfer_entropy', 0.0))
    granger_p_value = _get_metric_series(df, 'granger_p_value', 1.0)
    granger_score = _sigmoid_transform(1 - granger_p_value, center=0.9, steepness=20) # Reward p < 0.1
    causality_score = 0.6 * te_score + 0.4 * granger_score

    # 3. Complexity Balance - DFA near 0.65 is good, Sample Entropy is a direct measure
    dfa_alpha = _get_metric_series(df, 'dfa_alpha_raw', 0.65)
    dfa_score = (1 - np.abs(dfa_alpha - 0.65) / 0.35).clip(0, 1) # Penalize distance from 0.65
    
    sampen_score = _percentile_normalize(_get_metric_series(df, 'sample_entropy', 0.5))
    complexity_balance_score = 0.5 * dfa_score + 0.5 * sampen_score

    # Combine complexity & causality metrics (weights sum to 1.0)
    final_score = (
        0.25 * redundancy_score +
        0.45 * causality_score +
        0.30 * complexity_balance_score
    )
    return final_score


def _calculate_live_readiness_score(df: pd.DataFrame) -> pd.Series:
    """Calculate live trading readiness component of Hart Index."""
    # Trigger rate metrics
    live_tr = _get_metric_series(df, 'live_tr_prior', 0.05)
    tr_saturation = settings.elv.trigger_rate_saturation
    
    # Ideal trigger rate is not too low, not too high
    tr_score = _sigmoid_transform(live_tr / tr_saturation, center=0.5, steepness=3)
    
    # Dormancy and specialist flags
    dormant_qualified = _get_metric_series(df, 'dormant_qualified_flag', 0).astype(bool)
    specialist = _get_metric_series(df, 'specialist_flag', 0).astype(bool)
    
    # Bonus for special qualifications
    qualification_bonus = pd.Series(0.0, index=df.index)
    qualification_bonus[dormant_qualified] = 0.1
    qualification_bonus[specialist] = 0.15
    
    # Coverage factor from ELV
    coverage = _get_metric_series(df, 'coverage_factor', 0.5)
    
    # Page-Hinkley alarm (no alarm is good)
    ph_alarm = _get_metric_series(df, 'page_hinkley_alarm', 0)
    alarm_penalty = pd.Series(1.0, index=df.index)
    alarm_penalty[ph_alarm > 0] = 0.7
    
    # Combine readiness metrics
    readiness_score = (
        0.40 * tr_score +
        0.30 * coverage +
        0.20 * alarm_penalty +
        0.10 * qualification_bonus
    )
    
    return readiness_score


def calculate_hart_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hart Index for all setups in the DataFrame.
    
    Args:
        df: DataFrame with all evaluation metrics (from ELV calculation)
        
    Returns:
        DataFrame with added hart_index and component columns
    """
    result_df = df.copy()
    
    # Ensure we have minimum required columns
    required_cols = ['edge_crps_raw', 'edge_ig_raw', 'n_trig_oos']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns for Hart Index: {missing_cols}")
        # Add missing columns with default values
        for col in missing_cols:
            result_df[col] = 0.0
    
    # Calculate component scores
    perf_score = _calculate_performance_score(result_df)
    robust_score = _calculate_robustness_score(result_df)
    complex_score = _calculate_complexity_score(result_df)
    ready_score = _calculate_live_readiness_score(result_df)
    
    # Apply component weights
    components = HartIndexComponents()
    
    # Performance sub-components
    result_df['hart_edge_performance'] = perf_score * components.edge_performance
    result_df['hart_information_quality'] = perf_score * components.information_quality
    result_df['hart_risk_reward'] = perf_score * components.risk_reward
    
    # Robustness sub-components
    result_df['hart_statistical_significance'] = robust_score * components.statistical_significance
    result_df['hart_support'] = robust_score * components.support
    result_df['hart_sensitivity_resilience'] = robust_score * components.sensitivity_resilience
    
    # Complexity sub-components
    result_df['hart_causality'] = complex_score * components.causality
    result_df['hart_signal_quality'] = complex_score * components.signal_quality
    result_df['hart_complexity_balance'] = complex_score * components.complexity_balance
    
    # Readiness sub-components
    result_df['hart_trigger_reliability'] = ready_score * components.trigger_reliability
    result_df['hart_regime_coverage'] = ready_score * components.regime_coverage
    
    # Calculate raw Hart Index (0-1)
    hart_index_raw = (
        result_df['hart_edge_performance'] +
        result_df['hart_information_quality'] +
        result_df['hart_risk_reward'] +
        result_df['hart_statistical_significance'] +
        result_df['hart_support'] +
        result_df['hart_sensitivity_resilience'] +
        result_df['hart_causality'] +
        result_df['hart_signal_quality'] +
        result_df['hart_complexity_balance'] +
        result_df['hart_trigger_reliability'] +
        result_df['hart_regime_coverage']
    )
    
    # Apply final adjustments
    # Penalty for setups that failed ELV gates
    elv_gate_penalty = pd.Series(1.0, index=result_df.index)
    if 'pass_cv_gates' in result_df.columns:
        elv_gate_penalty[~result_df['pass_cv_gates']] = 0.7
    
    # Bonus for high ELV scores (if available)
    elv_bonus = pd.Series(1.0, index=result_df.index)
    if 'elv' in result_df.columns:
        high_elv_threshold = _get_metric_series(df, 'elv').quantile(0.8)
        high_elv = result_df['elv'] > high_elv_threshold
        elv_bonus[high_elv] = 1.1
    
    # Apply adjustments
    hart_index_adjusted = hart_index_raw * elv_gate_penalty * elv_bonus
    
    # Scale to 0-100 and clip
    result_df['hart_index'] = (hart_index_adjusted * 100).clip(0, 100).round(1)
    
    # Add interpretation labels
    def get_hart_index_label(score):
        if score >= 85:
            return "Exceptional"
        elif score >= 70:
            return "Strong"
        elif score >= 55:
            return "Moderate"
        elif score >= 40:
            return "Marginal"
        else:
            return "Weak"
    
    result_df['hart_label'] = result_df['hart_index'].apply(get_hart_index_label)
    
    # Add component breakdown for transparency
    result_df['hart_performance_total'] = (
        result_df['hart_edge_performance'] +
        result_df['hart_information_quality'] +
        result_df['hart_risk_reward']
    ) * 100
    
    result_df['hart_robustness_total'] = (
        result_df['hart_statistical_significance'] +
        result_df['hart_support'] +
        result_df['hart_sensitivity_resilience']
    ) * 100
    
    result_df['hart_complexity_total'] = (
        result_df['hart_causality'] +
        result_df['hart_signal_quality'] +
        result_df['hart_complexity_balance']
    ) * 100
    
    result_df['hart_readiness_total'] = (
        result_df['hart_trigger_reliability'] +
        result_df['hart_regime_coverage']
    ) * 100
    
    return result_df


def get_hart_index_summary(df: pd.DataFrame) -> Dict[str, any]:
    """
    Get a summary of Hart Index distribution and statistics.
    
    Args:
        df: DataFrame with hart_index calculated
        
    Returns:
        Dictionary with summary statistics
    """
    if 'hart_index' not in df.columns:
        return {"error": "Hart Index not calculated"}
    
    hart_scores = df['hart_index']
    
    summary = {
        "mean": hart_scores.mean(),
        "median": hart_scores.median(),
        "std": hart_scores.std(),
        "min": hart_scores.min(),
        "max": hart_scores.max(),
        "q25": hart_scores.quantile(0.25),
        "q75": hart_scores.quantile(0.75),
        "n_exceptional": (hart_scores >= 85).sum(),
        "n_strong": ((hart_scores >= 70) & (hart_scores < 85)).sum(),
        "n_moderate": ((hart_scores >= 55) & (hart_scores < 70)).sum(),
        "n_marginal": ((hart_scores >= 40) & (hart_scores < 55)).sum(),
        "n_weak": (hart_scores < 40).sum(),
        "top_10_avg": hart_scores.nlargest(10).mean() if len(hart_scores) >= 10 else hart_scores.mean()
    }
    
    return summary
