"""
Objective transformations for GA optimization.

Defines how raw metrics are transformed into optimization objectives.
All objectives are defined in "maximize" space for DEAP.

Forecast-first design:
- Proper scoring rules (CRPS, Brier, Log Loss, Pinball) for GA selection
- Legacy P&L objectives retained for reporting only
- Calibration metrics for model quality assessment
"""

from __future__ import annotations
import math
from typing import Dict, Tuple, List, Set
import numpy as np


# ============================================================================
# PROPER SCORING RULES (for GA selection in forecast-first mode)
# ============================================================================

PROPER_SCORING_RULES: Set[str] = {
    "crps",           # Continuous Ranked Probability Score
    "brier_score",    # Brier Score
    "log_loss",       # Logarithmic loss (cross-entropy)
    "pinball_q10",    # Pinball loss at 10th quantile
    "pinball_q25",    # Pinball loss at 25th quantile
    "pinball_q50",    # Pinball loss at median
    "pinball_q75",    # Pinball loss at 75th quantile
    "pinball_q90",    # Pinball loss at 90th quantile
}


# ============================================================================
# OBJECTIVE TRANSFORMS
# ============================================================================

# Transform each metric into "maximize" space for DEAP.
# Format: metric_name -> (transform_function, label)
OBJECTIVE_TRANSFORMS: Dict[str, Tuple] = {
    # ------------------------------------------------------------------
    # PROPER SCORING RULES (loss-like: lower is better → negate)
    # ------------------------------------------------------------------
    "crps": (lambda x: -x, "negate"),
    "brier_score": (lambda x: -x, "negate"),
    "log_loss": (lambda x: -x, "negate"),
    "pinball_q10": (lambda x: -x, "negate"),
    "pinball_q25": (lambda x: -x, "negate"),
    "pinball_q50": (lambda x: -x, "negate"),
    "pinball_q75": (lambda x: -x, "negate"),
    "pinball_q90": (lambda x: -x, "negate"),
    
    # ------------------------------------------------------------------
    # CALIBRATION METRICS (loss-like: lower is better → negate)
    # ------------------------------------------------------------------
    "calibration_mae": (lambda x: -x, "negate"),
    "calibration_ece": (lambda x: -x, "negate"),  # Expected Calibration Error
    "calibration_mce": (lambda x: -x, "negate"),  # Maximum Calibration Error
    
    # ------------------------------------------------------------------
    # INFORMATION METRICS (score-like: higher is better → identity)
    # ------------------------------------------------------------------
    "info_gain": (lambda x: x, "identity"),           # Information Gain (bits)
    "kl_divergence": (lambda x: -x, "negate"),        # KL divergence (loss-like)
    "js_divergence": (lambda x: -x, "negate"),        # Jensen-Shannon divergence
    
    # ------------------------------------------------------------------
    # DISTRIBUTIONAL METRICS
    # ------------------------------------------------------------------
    "wasserstein1": (lambda x: -x, "negate"),         # Earth Mover's Distance (loss-like)
    "energy_score": (lambda x: -x, "negate"),         # Multivariate proper score
    
    # ------------------------------------------------------------------
    # SKILL METRICS (higher is better)
    # ------------------------------------------------------------------
    "skill_vs_uniform": (lambda x: x, "identity"),    # CRPS improvement over uniform
    "skill_vs_marginal": (lambda x: x, "identity"),   # CRPS improvement over marginal
    "skill_vs_persistence": (lambda x: x, "identity"), # vs last-value baseline
    
    # ------------------------------------------------------------------
    # ROBUSTNESS METRICS (higher is better)
    # ------------------------------------------------------------------
    "regime_consistency": (lambda x: x, "identity"),   # Regime similarity train→test
    "drift_tolerance": (lambda x: x, "identity"),      # 1 - drift_auc (higher = less drift)
    "bootstrap_stability": (lambda x: x, "identity"),  # Bootstrap CI width (inverted)
    
    # ------------------------------------------------------------------
    # COVERAGE METRICS (for interval forecasts)
    # ------------------------------------------------------------------
    "coverage_80": (lambda x: -abs(x - 0.80), "target_0.80"),  # Target 80% coverage
    "coverage_90": (lambda x: -abs(x - 0.90), "target_0.90"),  # Target 90% coverage
    "coverage_95": (lambda x: -abs(x - 0.95), "target_0.95"),  # Target 95% coverage
    
    # ------------------------------------------------------------------
    # LEGACY P&L OBJECTIVES (retained for reporting only, not for GA selection)
    # ------------------------------------------------------------------
    "ig_sharpe": (lambda x: x, "identity_LEGACY"),     # Info Gain Sharpe (P&L-based)
    "min_ig": (lambda x: x, "identity_LEGACY"),        # Min Info Gain (P&L-based)
    "sharpe_ratio": (lambda x: x, "identity_LEGACY"),  # Sharpe ratio (P&L-based)
    "sortino_ratio": (lambda x: x, "identity_LEGACY"), # Sortino ratio (P&L-based)
    
    # ------------------------------------------------------------------
    # OTHER METRICS
    # ------------------------------------------------------------------
    "novelty_score": (lambda x: x, "identity"),
    "psr": (lambda x: x, "identity"),  # Probabilistic Sharpe Ratio
    "dsr": (lambda x: x, "identity"),  # Deflated Sharpe Ratio
    "hart_index": (lambda x: x, "identity"),
    "mae": (lambda x: -x, "negate"),   # Mean Absolute Error
    "rmse": (lambda x: -x, "negate"),  # Root Mean Squared Error
}


# ============================================================================
# VALIDATION & FILTERING
# ============================================================================

def audit_objectives(objective_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if all objective names are registered.
    
    Args:
        objective_names: List of objective names to validate
        
    Returns:
        (all_valid, missing_names)
    """
    missing = [name for name in objective_names if name not in OBJECTIVE_TRANSFORMS]
    return (len(missing) == 0, missing)


def is_proper_scoring_rule(objective_name: str) -> bool:
    """
    Check if an objective is a proper scoring rule.
    
    Proper scoring rules are theoretically grounded forecasting metrics
    that incentivize honest probability assessments.
    
    Args:
        objective_name: Name of the objective
        
    Returns:
        True if it's a proper scoring rule, False otherwise
    """
    return objective_name in PROPER_SCORING_RULES


def filter_to_proper_scoring_rules(objective_names: List[str]) -> List[str]:
    """
    Filter objective list to only proper scoring rules.
    
    Used in NPWF mode to ensure GA selection uses only forecast-quality metrics.
    Legacy P&L objectives (ig_sharpe, min_ig) are excluded.
    
    Args:
        objective_names: List of all objective names
        
    Returns:
        List of objective names that are proper scoring rules
    """
    return [name for name in objective_names if is_proper_scoring_rule(name)]


def is_legacy_pnl_objective(objective_name: str) -> bool:
    """
    Check if an objective is a legacy P&L-based metric.
    
    Legacy objectives are retained for backward compatibility and reporting
    but should not be used for GA selection in forecast-first mode.
    
    Args:
        objective_name: Name of the objective
        
    Returns:
        True if it's a legacy P&L objective, False otherwise
    """
    if objective_name not in OBJECTIVE_TRANSFORMS:
        return False
    
    _, label = OBJECTIVE_TRANSFORMS[objective_name]
    return "LEGACY" in label


def get_recommended_objectives(mode: str = "forecast") -> List[str]:
    """
    Get recommended objective combinations for different modes.
    
    Args:
        mode: One of "forecast", "balanced", "legacy"
            - "forecast": Pure forecast-first (proper scoring rules only)
            - "balanced": Mix of forecast and calibration
            - "legacy": Backward compatible P&L-based
            
    Returns:
        List of recommended objective names
    """
    if mode == "forecast":
        # Pure forecast-first: CRPS + tail coverage
        return ["crps", "pinball_q10", "pinball_q90"]
    
    elif mode == "balanced":
        # Forecast quality + calibration + skill
        return ["crps", "calibration_mae", "skill_vs_marginal"]
    
    elif mode == "legacy":
        # Backward compatible
        return ["ig_sharpe", "min_ig"]
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'forecast', 'balanced', or 'legacy'")


def apply_objective_transforms(
    metrics: Dict[str, float],
    objective_names: List[str],
    allow_legacy: bool = True
) -> Tuple[List[float], Dict[str, str]]:
    """
    Transform raw metrics into objective vector (maximize space).
    
    Args:
        metrics: Dictionary of raw metric values
        objective_names: List of objectives to extract and transform
        allow_legacy: If False, raises error on legacy P&L objectives
        
    Returns:
        (objective_values, transform_labels)
        - objective_values: List of transformed values (all in maximize space)
        - transform_labels: Dict mapping objective_name -> transform_type
        
    Raises:
        KeyError: If objective name not in OBJECTIVE_TRANSFORMS
        ValueError: If metric value is missing or non-finite
        ValueError: If legacy objective used when allow_legacy=False
    """
    objs: List[float] = []
    labels: Dict[str, str] = {}
    
    for name in objective_names:
        # Check if objective exists
        if name not in OBJECTIVE_TRANSFORMS:
            raise KeyError(
                f"Unknown objective '{name}'. Add to OBJECTIVE_TRANSFORMS or use "
                f"get_recommended_objectives() to see valid options."
            )
        
        # Check if legacy objective is allowed
        if not allow_legacy and is_legacy_pnl_objective(name):
            raise ValueError(
                f"Legacy P&L objective '{name}' not allowed in forecast-first mode. "
                f"Use proper scoring rules: {list(PROPER_SCORING_RULES)}"
            )
        
        # Get transform function and label
        transform_fn, label = OBJECTIVE_TRANSFORMS[name]
        
        # Extract metric value
        value = metrics.get(name, None)
        
        # Validate metric value
        if value is None:
            raise ValueError(
                f"Objective '{name}' missing from metrics dict. "
                f"Available metrics: {list(metrics.keys())}"
            )
        
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(
                f"Objective '{name}' has non-finite value: {value}. "
                f"Cannot optimize non-finite objectives."
            )
        
        # Apply transform and add to output
        transformed_value = float(transform_fn(value))
        objs.append(transformed_value)
        labels[name] = label
    
    return objs, labels


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main transforms
    "OBJECTIVE_TRANSFORMS",
    "PROPER_SCORING_RULES",
    
    # Validation functions
    "audit_objectives",
    "is_proper_scoring_rule",
    "is_legacy_pnl_objective",
    
    # Filtering functions
    "filter_to_proper_scoring_rules",
    "get_recommended_objectives",
    
    # Transform application
    "apply_objective_transforms",
]