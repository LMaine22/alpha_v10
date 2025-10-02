# alpha_discovery/gauntlet/stage5_decision.py
"""
Gauntlet 2.0 — Stage 5: Final Decision

Aggregates results from all previous stages and makes the final
promotion/rejection decision. Computes a composite score and 
applies final veto conditions.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def run_stage5_final(
    stage1_df: pd.DataFrame,
    stage2_df: pd.DataFrame, 
    stage3_df: pd.DataFrame,
    stage4_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Stage 5: Final promotion decision.
    
    Aggregates all stage results and computes a final promotion score.
    Applies any final veto conditions.
    
    Args:
        stage1_df: Stage 1 results
        stage2_df: Stage 2 results  
        stage3_df: Stage 3 results
        stage4_df: Stage 4 results
        config: Configuration for final decision
        
    Returns:
        Single-row DataFrame with final decision
    """
    if not all(df is not None and not df.empty for df in [stage1_df, stage2_df, stage3_df, stage4_df]):
        return pd.DataFrame([{
            "setup_id": "unknown",
            "rank": None,
            "pass_stage5": False,
            "reject_code": "S5_MISSING_INPUTS",
            "reason": "missing_stage_results",
            "promotion_score": 0.0
        }])
    
    # Get candidate info
    setup_id = stage1_df["setup_id"].iloc[0]
    rank = stage1_df["rank"].iloc[0] if "rank" in stage1_df.columns else None
    
    # Check if all previous stages passed
    all_passed = (
        _safe_bool(stage1_df, "pass_stage1") and
        _safe_bool(stage2_df, "pass_stage2") and  
        _safe_bool(stage3_df, "pass_stage3") and
        _safe_bool(stage4_df, "pass_stage4")
    )
    
    if not all_passed:
        # Find which stage failed
        failed_stages = []
        if not _safe_bool(stage1_df, "pass_stage1"):
            failed_stages.append("S1")
        if not _safe_bool(stage2_df, "pass_stage2"):
            failed_stages.append("S2")
        if not _safe_bool(stage3_df, "pass_stage3"):
            failed_stages.append("S3")
        if not _safe_bool(stage4_df, "pass_stage4"):
            failed_stages.append("S4")
        
        return pd.DataFrame([{
            "setup_id": setup_id,
            "rank": rank,
            "pass_stage5": False,
            "reject_code": f"S5_UPSTREAM_FAIL_{'+'.join(failed_stages)}",
            "reason": f"upstream_stage_failures: {', '.join(failed_stages)}",
            "promotion_score": 0.0
        }])
    
    # All stages passed - compute composite promotion score
    try:
        promotion_score = _compute_promotion_score(
            stage1_df, stage2_df, stage3_df, stage4_df, config
        )
        
        # Apply final gates if configured
        min_promotion_score = config.get("min_promotion_score", 0.0) if config else 0.0
        
        passed = promotion_score >= min_promotion_score
        
        return pd.DataFrame([{
            "setup_id": setup_id,
            "rank": rank,
            "pass_stage5": bool(passed),
            "reject_code": None if passed else "S5_LOW_SCORE",
            "reason": "approved_for_promotion" if passed else f"promotion_score_{promotion_score:.3f}_below_min_{min_promotion_score:.3f}",
            "promotion_score": float(promotion_score),
            
            # Include key metrics from each stage for reference
            "s1_health_score": _safe_float(stage1_df, "health_score", 0.0),
            "s2_profitability_score": _safe_float(stage2_df, "profitability_score", 0.0),
            "s3_hart_full_score": _safe_float(stage3_df, "hart_full_score", 0.0),
            "s4_portfolio_score": _safe_float(stage4_df, "portfolio_fit_score", 0.0),
        }])
        
    except Exception as e:
        return pd.DataFrame([{
            "setup_id": setup_id,
            "rank": rank,
            "pass_stage5": False,
            "reject_code": "S5_COMPUTATION_ERROR",
            "reason": f"promotion_score_computation_failed: {e}",
            "promotion_score": 0.0
        }])


def _compute_promotion_score(
    stage1_df: pd.DataFrame,
    stage2_df: pd.DataFrame,
    stage3_df: pd.DataFrame, 
    stage4_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute composite promotion score from all stage results.
    
    Returns:
        Promotion score (higher is better)
    """
    # Default weights
    weights = {
        "stage1_weight": 0.10,  # Health is important but basic
        "stage2_weight": 0.25,  # Profitability is key
        "stage3_weight": 0.45,  # CPCV robustness is most important
        "stage4_weight": 0.20,  # Portfolio fit matters for deployment
    }
    
    if config:
        weights.update({k: v for k, v in config.items() if k in weights})
    
    # Extract normalized scores from each stage
    s1_score = _normalize_score(_safe_float(stage1_df, "health_score", 0.5))
    s2_score = _normalize_score(_safe_float(stage2_df, "profitability_score", 0.5))
    s3_score = _normalize_score(_safe_float(stage3_df, "hart_full_score", 0.5))
    s4_score = _normalize_score(_safe_float(stage4_df, "portfolio_fit_score", 0.5))
    
    # Weighted combination
    promotion_score = (
        weights["stage1_weight"] * s1_score +
        weights["stage2_weight"] * s2_score +
        weights["stage3_weight"] * s3_score +
        weights["stage4_weight"] * s4_score
    )
    
    return float(promotion_score)


def _normalize_score(score: float, min_val: float = 0.0, max_val: float = 2.0) -> float:
    """Normalize score to [0, 1] range."""
    if pd.isna(score):
        return 0.0
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))


def _safe_bool(df: pd.DataFrame, col: str) -> bool:
    """Safely extract boolean from DataFrame column."""
    if df is None or df.empty or col not in df.columns:
        return False
    val = df[col].iloc[0]
    return bool(val) if pd.notna(val) else False


def _safe_float(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Safely extract float from DataFrame column."""
    if df is None or df.empty or col not in df.columns:
        return default
    val = df[col].iloc[0]
    return float(val) if pd.notna(val) else default
