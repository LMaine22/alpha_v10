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
        # No discriminative variation; return NaN to honor no-defaults policy
        return pd.Series(np.nan, index=series.index)
        
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
    
    # NOTE: Do not default-fill discriminative metrics. Keep NaNs for auditability.
    
    # Keep redundancy_mi_raw as-is (no default-fills)
    
    # Keep complexity_metric_raw as-is (no default-fills)
        
    # Keep stab_crps_mad as-is (no default-fills). It will be computed upstream; if missing, stays NaN.
    
    # Guard for missing column from orchestrator (keep as NaN if missing)
    if 'regime_overlap_today' not in df:
        df['regime_overlap_today'] = np.nan

    # --- Gates and Flag Calculation (moved up) ---
    # Ensure required columns for gates exist (no numeric defaults)
    if 'n_trig_oos' not in df.columns:
        df['n_trig_oos'] = np.nan
    if 'eligibility_rate_oos' not in df.columns:
        df['eligibility_rate_oos'] = np.nan

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
        # Respect no-defaults policy on failure
        rz_crps = pd.Series(np.nan, index=df.index)
        rz_pinball = pd.Series(np.nan, index=df.index)
        rz_ig = pd.Series(np.nan, index=df.index)
        rz_w1 = pd.Series(np.nan, index=df.index)
        rz_calib = pd.Series(np.nan, index=df.index)
    
    # Calculate edge_oos with weights
    df['edge_oos'] = (
        elv_cfg.edge_crps_weight * rz_crps +
        elv_cfg.edge_pinball_weight * rz_pinball +
        elv_cfg.edge_info_gain_weight * rz_ig +
        elv_cfg.edge_w1_weight * rz_w1 +
        elv_cfg.edge_calibration_weight * rz_calib
    )
    # Keep edge_oos NaN if inputs are NaN; do not coerce to 0
    
    # Apply Edge_OOS gate - lower threshold to 5 for more coverage
    adjusted_gate = 5  # Lower threshold from default 15
    mask_gate = (df['n_trig_oos'] < adjusted_gate) & (~df['dormant_qualified_flag'])
    # Honor no-defaults: mark as NaN when unsupported rather than forcing 0
    df.loc[mask_gate, 'edge_oos'] = np.nan

    # --- 2. LiveTriggerRate_Prior Component (recency-aware) ---
    # Required inputs (no numeric defaults)
    if 'tr_cv_reg' not in df.columns:
        df['tr_cv_reg'] = np.nan
    if 'tr_fg' not in df.columns:
        df['tr_fg'] = np.nan
    
    # Base prior from regime-weighted CV and foreground estimates
    base_prior = (
        elv_cfg.live_trigger_rate_cv_weight * df['tr_cv_reg'] +
        elv_cfg.live_trigger_rate_fg_weight * df['tr_fg']
    )
    
    # Short-term rate proxy: use rolling/foreground if available, else fall back to base
    short_term = df['tr_short_term'] if 'tr_short_term' in df.columns else pd.Series(np.nan, index=df.index)
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
    # Regime coverage: share of regimes with positive OOS edge (if available), else use regime_breadth; no defaults
    if 'regime_positive_share' in df.columns:
        cover_reg = df['regime_positive_share'].clip(0, 1)
    else:
        cover_reg = df['regime_breadth'] if 'regime_breadth' in df.columns else pd.Series(np.nan, index=df.index)
    
    # Support coverage: normalize recent OOS triggers; prefer last 126 days if present
    if 'n_trig_oos_recent' in df.columns:
        cover_sup = np.minimum(1.0, df['n_trig_oos_recent'] / max(settings.elv.maturity_n_triggers / 2, 1))
    else:
        cover_sup = np.minimum(1.0, df['n_trig_oos'] / max(settings.elv.maturity_n_triggers, 1)) if 'n_trig_oos' in df.columns else pd.Series(np.nan, index=df.index)
    cover_sup = pd.Series(cover_sup, index=df.index)
    
    # Band certainty: 1 - normalized entropy if band_probs available; else fallback
    if 'band_probs' in df.columns:
        def entropy_row(probs):
            try:
                p = np.array(probs, dtype=float)
                p = p / (p.sum() + 1e-12)
                h = -(p * np.log(p + 1e-12)).sum()
                h_max = np.log(len(p)) if len(p) > 0 else np.nan
                return 1.0 - (h / h_max) if np.isfinite(h_max) and h_max > 0 else np.nan
            except Exception:
                return np.nan
        cover_band = df['band_probs'].apply(entropy_row)
    else:
        cover_band = pd.Series(np.nan, index=df.index)
    
    # Renormalize weights per-row across available components (NaN-safe)
    wr, ws, wb = elv_cfg.coverage_regime_breadth_weight, elv_cfg.coverage_fold_coverage_weight, elv_cfg.coverage_stability_weight
    comp_df = pd.DataFrame({'reg': cover_reg, 'sup': cover_sup, 'band': cover_band})
    w_arr = np.array([wr, ws, wb], dtype=float)
    mask = comp_df.notna().astype(float)
    denom = (mask * w_arr).sum(axis=1)
    numer = comp_df.fillna(0.0).multiply(w_arr, axis=1).sum(axis=1)
    cov = numer / denom
    cov[denom == 0] = np.nan
    df['coverage_factor'] = cov.clip(0, 1)

    # --- 4. PenaltyAdj Component ---
    # Handle penalties with NaN protection (no numeric defaults)
    try:
        df['pen_sens'] = np.exp(-elv_cfg.penalty_sensitivity_k * df['sensitivity_delta_edge_raw']).clip(0.6, 1.0)
    except Exception as e:
        print(f"Warning: Error calculating pen_sens: {e}")
        df['pen_sens'] = pd.Series(np.nan, index=df.index)
    
    try:
        df['pen_mbb'] = pd.cut(df['bootstrap_p_value_raw'], bins=[-1, 0.05, 0.10, 2], labels=[0.6, 0.8, 1.0]).astype(float)
    except Exception as e:
        print(f"Warning: Error calculating pen_mbb: {e}")
        df['pen_mbb'] = pd.Series(np.nan, index=df.index)
    
    # Handle page_hinkley_alarm column if present; no defaults
    if 'page_hinkley_alarm' in df.columns:
        df['pen_ph'] = df['page_hinkley_alarm'].apply(lambda x: elv_cfg.penalty_page_hinkley_adj if x else 1.0)
    else:
        df['pen_ph'] = pd.Series(np.nan, index=df.index)
    
    try:
        df['pen_red'] = 1 - elv_cfg.penalty_redundancy_factor * _robust_scaler(df['redundancy_mi_raw'])
    except Exception as e:
        print(f"Warning: Error calculating pen_red: {e}")
        df['pen_red'] = pd.Series(np.nan, index=df.index)
    
    try:
        df['pen_cx'] = 1 - elv_cfg.penalty_complexity_factor * _robust_scaler(df['complexity_metric_raw'])
    except Exception as e:
        print(f"Warning: Error calculating pen_cx: {e}")
        df['pen_cx'] = pd.Series(np.nan, index=df.index)
    
    # Maturity penalty based on number of triggers
    df['pen_mat'] = np.minimum(1.0, df['n_trig_oos'] / elv_cfg.maturity_n_triggers)

    # Calculate final penalty adjustment; honor no-defaults (propagate NaN if any component is NaN)
    pen_cols = ['pen_sens', 'pen_mbb', 'pen_ph', 'pen_red', 'pen_cx', 'pen_mat']
    present = [c for c in pen_cols if c in df.columns]
    if present:
        df['penalty_adj'] = df[present].prod(axis=1)
    else:
        df['penalty_adj'] = pd.Series(np.nan, index=df.index)
    
    # --- 5. Final Adjustments and Labels ---
    # Apply Specialist floor
    df['specialist_flag'] = (df['edge_oos'] >= elv_cfg.specialist_edge_threshold) & \
                          (df['live_tr_prior'] * elv_cfg.trigger_rate_saturation <= elv_cfg.specialist_trigger_rate_max)
    df.loc[df['specialist_flag'], 'coverage_factor'] = np.maximum(df.loc[df['specialist_flag'], 'coverage_factor'], elv_cfg.specialist_coverage_floor)

    # Adjust priors for dormant-qualified setups
    df.loc[df['dormant_qualified_flag'], 'live_tr_prior'] = df.loc[df['dormant_qualified_flag'], 'tr_cv_reg'] / elv_cfg.trigger_rate_saturation
    df.loc[df['dormant_qualified_flag'], 'pen_mat'] = np.maximum(df.loc[df['dormant_qualified_flag'], 'pen_mat'], elv_cfg.maturity_dormant_floor)

    # --- 6. Final ELV Score ---
    # Calculate the final ELV score (compute-or-NaN). Do not fill missing components.
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
    
    return df
