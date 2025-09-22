from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from ..config import settings
from .regime import tau_cv_reg_weighted


def _robust_scaler(series: pd.Series) -> pd.Series:
    """
    Applies a robust scaling: winsorize at 5/95 percentiles, then min-max to [0, 1].
    """
    if series.empty or series.isna().all():
        return series
    
    p05 = series.quantile(0.05)
    p95 = series.quantile(0.95)
    clipped = series.clip(lower=p05, upper=p95)
    
    min_val, max_val = clipped.min(), clipped.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
        
    return (clipped - min_val) / (max_val - min_val)


def calculate_elv_and_labels(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the calculation of the ELV score and its components for a full cohort.
    
    Args:
        cohort_df: A DataFrame containing the aggregated, per-setup metrics from all folds.

    Returns:
        The input DataFrame with added columns for ELV and all its components and labels.
    """
    elv_cfg = settings.elv
    df = cohort_df.copy()

    # Guard for missing column from orchestrator
    if 'regime_overlap_today' not in df:
        df['regime_overlap_today'] = 0.0

    # --- Gates and Flag Calculation (moved up) ---
    df['pass_cv_gates'] = (
        (df['edge_crps_raw'].rank(pct=True, ascending=False) >= elv_cfg.gate_crps_percentile) |
        (df['edge_ig_raw'].rank(pct=True) >= elv_cfg.gate_ig_percentile)
    ) & (df['redundancy_mi_raw'] <= elv_cfg.gate_mi_max) & \
      (df['sensitivity_delta_edge_raw'] <= elv_cfg.gate_sensitivity_max_drop)

    df['dormant_flag'] = (df['n_trig_oos'] < elv_cfg.gate_min_oos_triggers) & \
                           (df['eligibility_rate_oos'] < elv_cfg.dormancy_eligibility_threshold)

    df['dormant_qualified_flag'] = df['dormant_flag'] & df['pass_cv_gates'] & \
                                     (df['regime_overlap_today'] >= 0.3) & (df['page_hinkley_alarm'] == 0)

    # --- 1. Edge_OOS Component ---
    rz_crps = 1 - _robust_scaler(df['edge_crps_raw'])
    rz_pinball = 1 - _robust_scaler((df['edge_pin_q10_raw'] + df['edge_pin_q90_raw']) / 2)
    rz_ig = _robust_scaler(df['edge_ig_raw'])
    rz_w1 = _robust_scaler(df['edge_w1_raw'])
    rz_calib = 1 - _robust_scaler(df['edge_calib_mae_raw'])
    
    df['edge_oos'] = (
        elv_cfg.edge_crps_weight * rz_crps +
        elv_cfg.edge_pinball_weight * rz_pinball +
        elv_cfg.edge_info_gain_weight * rz_ig +
        elv_cfg.edge_w1_weight * rz_w1 +
        elv_cfg.edge_calibration_weight * rz_calib
    )
    df['edge_oos'].fillna(0, inplace=True)
    
    # Apply Edge_OOS gate
    mask_gate = (df['n_trig_oos'] < elv_cfg.gate_min_oos_triggers) & (~df['dormant_qualified_flag'])
    df.loc[mask_gate, 'edge_oos'] = 0.0

    # --- 2. LiveTriggerRate_Prior Component ---
    df['live_tr_prior'] = (
        elv_cfg.live_trigger_rate_cv_weight * df['tr_cv_reg'] +
        elv_cfg.live_trigger_rate_fg_weight * df['tr_fg']
    )
    df['live_tr_prior'] = np.minimum(1.0, df['live_tr_prior'] / elv_cfg.trigger_rate_saturation)

    # --- 3. CoverageFactor Component ---
    df['coverage_factor'] = (
        elv_cfg.coverage_regime_breadth_weight * df['regime_breadth'] +
        elv_cfg.coverage_fold_coverage_weight * df['fold_coverage'] +
        elv_cfg.coverage_stability_weight * (1 - _robust_scaler(df['stab_crps_mad']))
    )

    # --- 4. PenaltyAdj Component ---
    df['pen_sens'] = np.exp(-elv_cfg.penalty_sensitivity_k * df['sensitivity_delta_edge_raw']).clip(0.6, 1.0)
    df['pen_mbb'] = pd.cut(df['bootstrap_p_value_raw'], bins=[-1, 0.05, 0.10, 2], labels=[0.6, 0.8, 1.0]).astype(float)
    df['pen_ph'] = df['page_hinkley_alarm'].apply(lambda x: elv_cfg.penalty_page_hinkley_adj if x else 1.0)
    df['pen_red'] = 1 - elv_cfg.penalty_redundancy_factor * _robust_scaler(df['redundancy_mi_raw'])
    df['pen_cx'] = 1 - elv_cfg.penalty_complexity_factor * _robust_scaler(df['complexity_metric_raw'])
    df['pen_mat'] = np.minimum(1.0, df['n_trig_oos'] / elv_cfg.maturity_n_triggers)

    df['penalty_adj'] = df['pen_sens'] * df['pen_mbb'] * df['pen_ph'] * df['pen_red'] * df['pen_cx'] * df['pen_mat']
    
    # --- 5. Final Adjustments and Labels ---
    # Apply Specialist floor
    df['specialist_flag'] = (df['edge_oos'] >= elv_cfg.specialist_edge_threshold) & \
                          (df['live_tr_prior'] * elv_cfg.trigger_rate_saturation <= elv_cfg.specialist_trigger_rate_max)
    df.loc[df['specialist_flag'], 'coverage_factor'] = np.maximum(df.loc[df['specialist_flag'], 'coverage_factor'], elv_cfg.specialist_coverage_floor)

    # Adjust priors for dormant-qualified setups
    df.loc[df['dormant_qualified_flag'], 'live_tr_prior'] = df.loc[df['dormant_qualified_flag'], 'tr_cv_reg'] / elv_cfg.trigger_rate_saturation
    df.loc[df['dormant_qualified_flag'], 'pen_mat'] = np.maximum(df.loc[df['dormant_qualified_flag'], 'pen_mat'], elv_cfg.maturity_dormant_floor)

    # --- 6. Final ELV Score ---
    df['elv'] = df['edge_oos'] * df['live_tr_prior'] * df['coverage_factor'] * df['penalty_adj']
    
    # Apply disqualification gates to zero out ELV
    disqualify_calib = (df['edge_calib_mae_raw'].rank(pct=True) > elv_cfg.disqualify_calib_mae_percentile) & \
                       (df['edge_crps_raw'].rank(pct=True, ascending=False) < elv_cfg.disqualify_crps_percentile)
    disqualify_robust = (df['bootstrap_p_value_raw'] < elv_cfg.disqualify_mbb_p_value) & \
                        (df['sensitivity_delta_edge_raw'] > elv_cfg.disqualify_sensitivity_drop)
    disqualify_redundancy = df['redundancy_mi_raw'] > elv_cfg.disqualify_mi_max
    
    # Gate on minimum OOS triggers unless it's a qualified dormant setup
    disqualify_support = (df['n_trig_oos'] < elv_cfg.gate_min_oos_triggers) & (~df['dormant_qualified_flag'])

    df.loc[disqualify_calib | disqualify_robust | disqualify_redundancy | disqualify_support, 'elv'] = 0.0

    return df
