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

    # --- Pre-computation: Ensure all raw metrics exist ---
    # The validation pipeline sometimes produces metrics without the `_raw` suffix.
    # We will create the `_raw` versions here if they don't exist, ensuring
    # consistency for downstream consumers like Hart Score.
    raw_metric_map = {
        'transfer_entropy': 'transfer_entropy_raw',
        'redundancy_mi': 'redundancy_mi_raw',
        'permutation_entropy': 'complexity_metric_raw', # Assuming permutation is default
        'complexity_index': 'complexity_metric_raw'
    }
    for source, target in raw_metric_map.items():
        if source in df.columns and target not in df.columns:
            df[target] = df[source]
            print(f"Created '{target}' from source column '{source}'.")

    # Debug input metrics
    print("\n=== ELV Calculation Input Check ===")
    key_metrics = [
        "edge_crps_raw", "edge_pin_q10_raw", "edge_pin_q90_raw", "edge_ig_raw", 
        "edge_w1_raw", "edge_calib_mae_raw", "sensitivity_delta_edge_raw",
        "bootstrap_p_value_raw", "redundancy_mi_raw", "complexity_metric_raw",
        "transfer_entropy_raw"
    ]
    
    print("Non-null counts by column:")
    for col in key_metrics:
        if col in df.columns:
            non_null = df[col].notna().sum()
            total = len(df)
            print(f"  {col}: {non_null} " + 
                 ("(ALL NULL)" if non_null == 0 else f"({non_null/total:.1%})"))
        else:
            print(f"  {col}: MISSING")
    
    # Fill missing values in key fields to prevent NaN propagation
    fill_zero_cols = ['edge_crps_raw', 'edge_pin_q10_raw', 'edge_pin_q90_raw', 
                     'edge_ig_raw', 'edge_w1_raw', 'edge_calib_mae_raw', 
                     'sensitivity_delta_edge_raw', 'bootstrap_p_value_raw',
                     'transfer_entropy_raw']
    
    # First, ensure all required columns exist
    for col in fill_zero_cols:
        if col not in df.columns:
            print(f"Adding missing column '{col}' with default value 0.0")
            df[col] = 0.0
        elif df[col].isna().any():
            null_count = df[col].isna().sum()
            print(f"Filling {null_count} NaN values in '{col}' with 0.0")
            df[col] = df[col].fillna(0.0)
    
    # Handle redundancy_mi_raw separately - important for penalty calculation
    if 'redundancy_mi_raw' not in df.columns:
        print(f"Adding missing column 'redundancy_mi_raw' with default value 0.5")
        df['redundancy_mi_raw'] = 0.5  # Middle value as default
    elif df['redundancy_mi_raw'].isna().any():
        null_count = df['redundancy_mi_raw'].isna().sum()
        print(f"Filling {null_count} NaN values in 'redundancy_mi_raw' with 0.5")
        df['redundancy_mi_raw'] = df['redundancy_mi_raw'].fillna(0.5)
    
    # Handle complexity metric
    if 'complexity_metric_raw' not in df.columns:
        print(f"Adding missing column 'complexity_metric_raw' with default value 0.5")
        df['complexity_metric_raw'] = 0.5
    elif df['complexity_metric_raw'].isna().any():
        null_count = df['complexity_metric_raw'].isna().sum()
        print(f"Filling {null_count} NaN values in 'complexity_metric_raw' with 0.5")
        df['complexity_metric_raw'] = df['complexity_metric_raw'].fillna(0.5)
        
    # Ensure stab_crps_mad has valid values
    if 'stab_crps_mad' not in df.columns:
        print(f"Adding missing column 'stab_crps_mad' with default value 0.1")
        df['stab_crps_mad'] = 0.1  # Default stability value
    elif df['stab_crps_mad'].isna().any():
        null_count = df['stab_crps_mad'].isna().sum()
        print(f"Filling {null_count} NaN values in 'stab_crps_mad' with 0.1")
        df['stab_crps_mad'] = df['stab_crps_mad'].fillna(0.1)
    
    # Guard for missing column from orchestrator
    if 'regime_overlap_today' not in df:
        df['regime_overlap_today'] = 0.0

    # --- Gates and Flag Calculation (moved up) ---
    # Ensure required columns for gates exist
    if 'n_trig_oos' not in df.columns:
        print("Missing critical column 'n_trig_oos', defaulting to 0")
        df['n_trig_oos'] = 0
        
    if 'eligibility_rate_oos' not in df.columns:
        print("Missing critical column 'eligibility_rate_oos', defaulting to 0.01")
        df['eligibility_rate_oos'] = 0.01

    # Calculate pass_cv_gates with robust handling of missing data
    df['pass_cv_gates'] = (
        (df['edge_crps_raw'].rank(pct=True, ascending=False) >= elv_cfg.gate_crps_percentile) |
        (df['edge_ig_raw'].rank(pct=True) >= elv_cfg.gate_ig_percentile)
    ) & (df['redundancy_mi_raw'] <= elv_cfg.gate_mi_max) & \
      (df['sensitivity_delta_edge_raw'] <= elv_cfg.gate_sensitivity_max_drop)
    
    # Flag dormant setups (low triggers, low eligibility rate)
    df['dormant_flag'] = (df['n_trig_oos'] < elv_cfg.gate_min_oos_triggers) & \
                           (df['eligibility_rate_oos'] < elv_cfg.dormancy_eligibility_threshold)

    df['dormant_qualified_flag'] = df['dormant_flag'] & df['pass_cv_gates'] & \
                                     (df['regime_overlap_today'] >= 0.3) & (df['page_hinkley_alarm'] == 0)

    # --- 1. Edge_OOS Component ---
    # Use robust scalers with extra protection against empty series
    try:
        rz_crps = 1 - _robust_scaler(df['edge_crps_raw'])
        rz_pinball = 1 - _robust_scaler((df['edge_pin_q10_raw'] + df['edge_pin_q90_raw']) / 2)
        rz_ig = _robust_scaler(df['edge_ig_raw'])
        rz_w1 = _robust_scaler(df['edge_w1_raw'])
        rz_calib = 1 - _robust_scaler(df['edge_calib_mae_raw'])
    except Exception as e:
        print(f"Warning: Error in robust scaling: {e}")
        # Provide defaults if scaling fails
        rz_crps = pd.Series(0.5, index=df.index)
        rz_pinball = pd.Series(0.5, index=df.index)
        rz_ig = pd.Series(0.5, index=df.index)
        rz_w1 = pd.Series(0.5, index=df.index)
        rz_calib = pd.Series(0.5, index=df.index)
    
    # Calculate edge_oos with weights
    df['edge_oos'] = (
        elv_cfg.edge_crps_weight * rz_crps +
        elv_cfg.edge_pinball_weight * rz_pinball +
        elv_cfg.edge_info_gain_weight * rz_ig +
        elv_cfg.edge_w1_weight * rz_w1 +
        elv_cfg.edge_calibration_weight * rz_calib
    )
    df['edge_oos'] = df['edge_oos'].fillna(0)
    
    # Apply Edge_OOS gate - lower threshold to 5 for more coverage
    adjusted_gate = 5  # Lower threshold from default 15
    mask_gate = (df['n_trig_oos'] < adjusted_gate) & (~df['dormant_qualified_flag'])
    df.loc[mask_gate, 'edge_oos'] = 0.0

    # --- 2. LiveTriggerRate_Prior Component (recency-aware) ---
    # Required inputs
    if 'tr_cv_reg' not in df.columns:
        print("Warning: 'tr_cv_reg' column missing, using default value")
        df['tr_cv_reg'] = 0.0
    if 'tr_fg' not in df.columns:
        print("Warning: 'tr_fg' column missing, using default value")
        df['tr_fg'] = 0.0
    
    # Base prior from regime-weighted CV and foreground estimates
    base_prior = (
        elv_cfg.live_trigger_rate_cv_weight * df['tr_cv_reg'] +
        elv_cfg.live_trigger_rate_fg_weight * df['tr_fg']
    )
    
    # Short-term rate proxy: use rolling/foreground if available, else fall back to base
    short_term = df.get('tr_short_term', df['tr_fg']).fillna(0.0)
    blended_rate = elv_cfg.trigger_rate_blend_base_weight * base_prior + (1 - elv_cfg.trigger_rate_blend_base_weight) * short_term
    
    # Recency override using last_trigger and current end date
    # Estimate days_since_trigger if available
    if 'last_trigger' in df.columns:
        last_trigger_dt = pd.to_datetime(df['last_trigger'], errors='coerce')
        # Use end_date from settings if available; else use max date in cohort (if provided)
        end_date = pd.to_datetime(settings.data.end_date)
        days_since = (end_date - last_trigger_dt).dt.days
    else:
        days_since = pd.Series(np.inf, index=df.index)
    
    # Recent boost decays with days_since; if very recent, force to strong value
    tau_days = elv_cfg.recency_tau_days_default
    recent_boost = np.exp(-np.clip(days_since, 0, None) / max(tau_days, 1))
    recent_override = (np.clip(days_since, 0, None) <= elv_cfg.recency_override_days)
    
    # Trigger prior is max of recent activity and blended base rate
    live_tr_prior_raw = np.maximum(recent_boost, blended_rate)
    # If within override window, push to 1.0 before saturation
    live_tr_prior_raw[recent_override] = 1.0
    
    # Apply saturation
    df['live_tr_prior'] = np.minimum(1.0, live_tr_prior_raw / max(elv_cfg.trigger_rate_saturation, 1e-9))

    # --- 3. CoverageFactor Component (composite) ---
    # Regime coverage: share of regimes with positive OOS edge (if available), else use regime_breadth
    if 'regime_positive_share' in df.columns:
        cover_reg = df['regime_positive_share'].clip(0, 1).fillna(0.5)
    else:
        if 'regime_breadth' not in df.columns:
            print("Warning: 'regime_breadth' column missing, using default value 0.5")
            df['regime_breadth'] = 0.5
        cover_reg = df['regime_breadth'].fillna(0.5)
    
    # Support coverage: normalize recent OOS triggers; prefer last 126 days if present
    if 'n_trig_oos_recent' in df.columns:
        cover_sup = np.minimum(1.0, df['n_trig_oos_recent'] / max(settings.elv.maturity_n_triggers / 2, 1))
    else:
        cover_sup = np.minimum(1.0, df['n_trig_oos'] / max(settings.elv.maturity_n_triggers, 1))
    cover_sup = pd.Series(cover_sup, index=df.index).fillna(0.5)
    
    # Band certainty: 1 - normalized entropy if band_probs available; else fallback
    if 'band_probs' in df.columns:
        def entropy_row(probs):
            try:
                p = np.array(probs, dtype=float)
                p = p / (p.sum() + 1e-12)
                h = -(p * np.log(p + 1e-12)).sum()
                h_max = np.log(len(p)) if len(p) > 0 else 1.0
                return 1.0 - (h / max(h_max, 1e-9))
            except Exception:
                return 0.5
        cover_band = df['band_probs'].apply(entropy_row)
    else:
        cover_band = pd.Series(0.5, index=df.index)
    cover_band = cover_band.fillna(0.5)
    
    df['coverage_factor'] = (
        elv_cfg.coverage_regime_breadth_weight * cover_reg +
        elv_cfg.coverage_fold_coverage_weight * cover_sup +
        elv_cfg.coverage_stability_weight * cover_band
    ).clip(0, 1)
    df['coverage_factor'] = df['coverage_factor'].fillna(0.5)

    # --- 4. PenaltyAdj Component ---
    # Handle penalties with NaN protection
    try:
        df['pen_sens'] = np.exp(-elv_cfg.penalty_sensitivity_k * df['sensitivity_delta_edge_raw']).clip(0.6, 1.0)
    except Exception as e:
        print(f"Warning: Error calculating pen_sens: {e}")
        df['pen_sens'] = 0.8  # Default mid-penalty
    df['pen_sens'] = df['pen_sens'].fillna(0.8)
    
    try:
        df['pen_mbb'] = pd.cut(df['bootstrap_p_value_raw'], bins=[-1, 0.05, 0.10, 2], labels=[0.6, 0.8, 1.0]).astype(float)
    except Exception as e:
        print(f"Warning: Error calculating pen_mbb: {e}")
        df['pen_mbb'] = 0.8  # Default mid-penalty
    df['pen_mbb'] = df['pen_mbb'].fillna(0.8)
    
    # Handle page_hinkley_alarm column if missing
    if 'page_hinkley_alarm' not in df.columns:
        print("Warning: 'page_hinkley_alarm' column missing, using default value 0")
        df['page_hinkley_alarm'] = 0
        
    df['pen_ph'] = df['page_hinkley_alarm'].apply(lambda x: elv_cfg.penalty_page_hinkley_adj if x else 1.0)
    df['pen_ph'] = df['pen_ph'].fillna(1.0)
    
    try:
        df['pen_red'] = 1 - elv_cfg.penalty_redundancy_factor * _robust_scaler(df['redundancy_mi_raw'])
    except Exception as e:
        print(f"Warning: Error calculating pen_red: {e}")
        df['pen_red'] = 0.7  # Default penalty
    df['pen_red'] = df['pen_red'].fillna(0.7)
    
    try:
        df['pen_cx'] = 1 - elv_cfg.penalty_complexity_factor * _robust_scaler(df['complexity_metric_raw'])
    except Exception as e:
        print(f"Warning: Error calculating pen_cx: {e}")
        df['pen_cx'] = 0.7  # Default penalty
    df['pen_cx'] = df['pen_cx'].fillna(0.7)
    
    # Maturity penalty based on number of triggers
    df['pen_mat'] = np.minimum(1.0, df['n_trig_oos'] / elv_cfg.maturity_n_triggers)
    df['pen_mat'] = df['pen_mat'].fillna(0.5)  # Default half-maturity if n_trig_oos is NaN

    # Calculate final penalty adjustment with NaN protection
    df['penalty_adj'] = df['pen_sens'] * df['pen_mbb'] * df['pen_ph'] * df['pen_red'] * df['pen_cx'] * df['pen_mat']
    df['penalty_adj'] = df['penalty_adj'].fillna(0.5)  # Default mid-penalty if any component is NaN
    
    # --- 5. Final Adjustments and Labels ---
    # Apply Specialist floor
    df['specialist_flag'] = (df['edge_oos'] >= elv_cfg.specialist_edge_threshold) & \
                          (df['live_tr_prior'] * elv_cfg.trigger_rate_saturation <= elv_cfg.specialist_trigger_rate_max)
    df.loc[df['specialist_flag'], 'coverage_factor'] = np.maximum(df.loc[df['specialist_flag'], 'coverage_factor'], elv_cfg.specialist_coverage_floor)

    # Adjust priors for dormant-qualified setups
    df.loc[df['dormant_qualified_flag'], 'live_tr_prior'] = df.loc[df['dormant_qualified_flag'], 'tr_cv_reg'] / elv_cfg.trigger_rate_saturation
    df.loc[df['dormant_qualified_flag'], 'pen_mat'] = np.maximum(df.loc[df['dormant_qualified_flag'], 'pen_mat'], elv_cfg.maturity_dormant_floor)

    # --- 6. Final ELV Score ---
    # Make sure all components have valid values before final calculation
    df['edge_oos'] = df['edge_oos'].fillna(0)
    df['live_tr_prior'] = df['live_tr_prior'].fillna(0.5)
    df['coverage_factor'] = df['coverage_factor'].fillna(0.5)
    df['penalty_adj'] = df['penalty_adj'].fillna(0.5)
    
    # Calculate the final ELV score
    df['elv'] = df['edge_oos'] * df['live_tr_prior'] * df['coverage_factor'] * df['penalty_adj']
    
    # Debug the ELV calculation for important rows
    print("\n=== ELV Calculation Debug for First 10 Rows ===")
    debug_cols = ['edge_oos', 'live_tr_prior', 'coverage_factor', 'penalty_adj', 'elv', 'n_trig_oos']
    print(df[debug_cols].head(10).to_string())
    
    # Apply disqualification gates with robust handling
    try:
        disqualify_calib = (df['edge_calib_mae_raw'].rank(pct=True) > elv_cfg.disqualify_calib_mae_percentile) & \
                           (df['edge_crps_raw'].rank(pct=True, ascending=False) < elv_cfg.disqualify_crps_percentile)
    except Exception as e:
        print(f"Warning: Error calculating disqualify_calib: {e}")
        disqualify_calib = pd.Series(False, index=df.index)
        
    try:
        disqualify_robust = (df['bootstrap_p_value_raw'] < elv_cfg.disqualify_mbb_p_value) & \
                            (df['sensitivity_delta_edge_raw'] > elv_cfg.disqualify_sensitivity_drop)
    except Exception as e:
        print(f"Warning: Error calculating disqualify_robust: {e}")
        disqualify_robust = pd.Series(False, index=df.index)
        
    try:
        disqualify_redundancy = df['redundancy_mi_raw'] > elv_cfg.disqualify_mi_max
    except Exception as e:
        print(f"Warning: Error calculating disqualify_redundancy: {e}")
        disqualify_redundancy = pd.Series(False, index=df.index)
    
    # Use adjusted gate threshold for minimum OOS triggers
    adjusted_gate = 5  # Lower threshold from default 15
    disqualify_support = (df['n_trig_oos'] < adjusted_gate) & (~df['dormant_qualified_flag'])

    # Apply disqualification
    df.loc[disqualify_calib | disqualify_robust | disqualify_redundancy | disqualify_support, 'elv'] = 0.0
    
    # Final check to ensure no NaN values in ELV
    df['elv'] = df['elv'].fillna(0.0)

    return df
