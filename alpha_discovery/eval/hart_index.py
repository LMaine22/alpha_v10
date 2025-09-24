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


def _sigmoid_transform(x, center: float = 0.5, steepness: float = 10):
    """Apply sigmoid transformation to map values to (0,1) range. NaN-safe for Series/arrays."""
    return 1 / (1 + np.exp(-steepness * (x - center)))


def _percentile_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize series to 0-1 using percentile ranks. Do not inject defaults; keep NaNs."""
    if series.empty:
        return pd.Series(np.nan, index=series.index)
    ranks = series.rank(pct=True, method='average', ascending=not invert)
    return ranks


def _get_metric_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Return series if present; otherwise a NaN series. No default numeric fills."""
    if key not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[key]


def _rowwise_weighted_mean(values: List[pd.Series], weights: List[float]) -> pd.Series:
    """Compute row-wise weighted mean across component series with NaN-aware weight renormalization."""
    if not values:
        return pd.Series(dtype=float)
    # Stack into DataFrame
    mat = pd.concat(values, axis=1)
    w = np.array(weights, dtype=float)
    # Indicator of available components per row
    present = ~mat.isna()
    present_weights = present.dot(w)
    weighted_sum = (mat.fillna(0).values * w).sum(axis=1)
    out = pd.Series(np.nan, index=mat.index)
    nonzero = present_weights > 0
    out.loc[nonzero] = weighted_sum[nonzero] / present_weights[nonzero]
    return out


def _calculate_performance_score(df: pd.DataFrame) -> pd.Series:
    """Calculate performance component of Hart Index."""
    # Edge metrics (lower is better for CRPS)
    crps_series = _get_metric_series(df, 'edge_crps_raw')
    pin_q10 = _get_metric_series(df, 'edge_pin_q10_raw')
    pin_q90 = _get_metric_series(df, 'edge_pin_q90_raw')
    crps_score = _percentile_normalize(crps_series, invert=True)
    pinball_score = _percentile_normalize((pin_q10 + pin_q90) / 2, invert=True)
    
    # Information metrics (higher is better)
    ig_score = _percentile_normalize(_get_metric_series(df, 'edge_ig_raw'))
    
    # Risk-reward metrics
    e_move = _get_metric_series(df, 'E_move')
    p_up = _get_metric_series(df, 'P_up')
    p_down = _get_metric_series(df, 'P_down')
    
    # Calculate directional confidence
    directional_confidence = np.abs(p_up - p_down) / (p_up + p_down + 1e-9)
    risk_reward_score = _sigmoid_transform(directional_confidence, center=0.2, steepness=8)
    
    # Combine performance metrics (weights sum to 1.0)
    performance_score = _rowwise_weighted_mean(
        [crps_score, pinball_score, ig_score, risk_reward_score],
        [0.40, 0.30, 0.20, 0.10]
    )
    
    return performance_score


def _calculate_robustness_score(df: pd.DataFrame) -> pd.Series:
    """Calculate robustness component of Hart Index."""
    # Statistical significance
    bootstrap_p = _get_metric_series(df, 'bootstrap_p_value_raw')
    significance_score = _sigmoid_transform(1 - bootstrap_p, center=0.9, steepness=20)
    
    # Support metrics
    n_triggers = _get_metric_series(df, 'n_trig_oos')
    support_score = _sigmoid_transform(n_triggers / 50, center=0.5, steepness=3)
    
    # Sensitivity metrics (lower is better)
    sensitivity_delta = _get_metric_series(df, 'sensitivity_delta_edge_raw')
    sensitivity_score = _sigmoid_transform(1 / (1 + sensitivity_delta), center=0.7, steepness=5)
    
    # Combine robustness metrics (weights sum to 1.0)
    robustness_score = _rowwise_weighted_mean(
        [significance_score, support_score, sensitivity_score],
        [0.40, 0.35, 0.25]
    )
    
    return robustness_score


def _calculate_complexity_score(df: pd.DataFrame) -> pd.Series:
    """Calculate complexity & causality component of Hart Index."""
    # 1. Signal Quality (Redundancy) - lower is better
    redundancy_mi = _get_metric_series(df, 'redundancy_mi_raw')
    redundancy_score = _sigmoid_transform(1 - redundancy_mi, center=0.7, steepness=5)

    # 2. Causality - higher TE is better, lower Granger p-value is better
    te_score = _percentile_normalize(_get_metric_series(df, 'transfer_entropy'))
    granger_p_value = _get_metric_series(df, 'granger_p_value')
    granger_score = _sigmoid_transform(1 - granger_p_value, center=0.9, steepness=20) # Reward p < 0.1
    causality_score = _rowwise_weighted_mean([te_score, granger_score], [0.6, 0.4])

    # 3. Complexity Balance - DFA near 0.65 is good, Sample Entropy is a direct measure
    dfa_alpha = _get_metric_series(df, 'dfa_alpha_raw')
    dfa_score = (1 - np.abs(dfa_alpha - 0.65) / 0.35).clip(0, 1)
    
    sampen_score = _percentile_normalize(_get_metric_series(df, 'sample_entropy'))
    complexity_balance_score = _rowwise_weighted_mean([dfa_score, sampen_score], [0.5, 0.5])

    # Combine complexity & causality metrics (weights sum to 1.0)
    final_score = _rowwise_weighted_mean(
        [redundancy_score, causality_score, complexity_balance_score],
        [0.25, 0.45, 0.30]
    )
    return final_score


def _calculate_live_readiness_score(df: pd.DataFrame) -> pd.Series:
    """Calculate live trading readiness component of Hart Index."""
    # Trigger rate metrics
    live_tr = _get_metric_series(df, 'live_tr_prior')
    tr_saturation = settings.elv.trigger_rate_saturation
    
    # Ideal trigger rate is not too low, not too high
    tr_score = _sigmoid_transform(live_tr / tr_saturation, center=0.5, steepness=3)
    
    # Dormancy and specialist flags
    dormant_qualified = _get_metric_series(df, 'dormant_qualified_flag').astype('boolean')
    specialist = _get_metric_series(df, 'specialist_flag').astype('boolean')
    
    # Bonus for special qualifications
    qualification_bonus = pd.Series(np.nan, index=df.index)
    if dormant_qualified.notna().any():
        qualification_bonus = qualification_bonus.fillna(0.0)
        qualification_bonus[dormant_qualified.fillna(False)] += 0.1
    if specialist.notna().any():
        qualification_bonus = qualification_bonus.fillna(0.0)
        qualification_bonus[specialist.fillna(False)] += 0.15
    
    # Coverage factor from ELV
    coverage = _get_metric_series(df, 'coverage_factor')
    
    # Page-Hinkley alarm (no alarm is good)
    ph_alarm = _get_metric_series(df, 'page_hinkley_alarm')
    alarm_penalty = pd.Series(np.nan, index=df.index)
    if ph_alarm.notna().any():
        alarm_penalty = pd.Series(1.0, index=df.index)
        alarm_penalty[ph_alarm > 0] = 0.7
    
    # Combine readiness metrics
    readiness_score = _rowwise_weighted_mean(
        [tr_score, coverage, alarm_penalty, qualification_bonus],
        [0.40, 0.30, 0.20, 0.10]
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
    
    # Ensure we have minimum required columns (do not default-fill)
    required_cols = ['edge_crps_raw', 'edge_ig_raw', 'n_trig_oos']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns for Hart Index: {missing_cols}")
    
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
    
    # Calculate raw Hart Index (0-1) with renormalization over available subcomponent weights
    component_cols = [
        ('hart_edge_performance', components.edge_performance),
        ('hart_information_quality', components.information_quality),
        ('hart_risk_reward', components.risk_reward),
        ('hart_statistical_significance', components.statistical_significance),
        ('hart_support', components.support),
        ('hart_sensitivity_resilience', components.sensitivity_resilience),
        ('hart_causality', components.causality),
        ('hart_signal_quality', components.signal_quality),
        ('hart_complexity_balance', components.complexity_balance),
        ('hart_trigger_reliability', components.trigger_reliability),
        ('hart_regime_coverage', components.regime_coverage),
    ]
    comp_df = result_df[[c for c, _ in component_cols]]
    weights = np.array([w for _, w in component_cols], dtype=float)
    present = ~comp_df.isna()
    present_weight_sum = present.dot(weights)
    weighted_sum = (comp_df.fillna(0).values).sum(axis=1)
    hart_index_raw = pd.Series(np.nan, index=result_df.index)
    nonzero = present_weight_sum > 0
    # weighted_sum already includes weights within each component column
    hart_index_raw.loc[nonzero] = weighted_sum[nonzero] / present_weight_sum[nonzero]
    
    # Apply final adjustments
    # Penalty for setups that failed ELV gates
    elv_gate_penalty = pd.Series(1.0, index=result_df.index)
    if 'pass_cv_gates' in result_df.columns:
        elv_gate_penalty[~result_df['pass_cv_gates']] = 0.7
    
    # Bonus for high ELV scores (if available)
    elv_bonus = pd.Series(1.0, index=result_df.index)
    if 'elv' in result_df.columns:
        high_elv_threshold = result_df['elv'].quantile(0.8)
        high_elv = result_df['elv'] > high_elv_threshold
        elv_bonus.loc[high_elv.fillna(False)] = 1.1
    
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
