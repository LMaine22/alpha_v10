"""
Splits package for forecast-first discovery.

Provides:
- PAWF: Purged Anchored Walk-Forward (outer splits)
- NPWF: Nested Purged Walk-Forward (inner GA selection)
- Regime detection (GMM-based)
- Calendar orthogonality (day-of-week, month-end)
- Bootstrap methods for robustness
- Adversarial drift detection
- SplitSpec dataclass and ID generation
"""

from .ids import SplitSpec, generate_split_id
from .pawf import build_pawf_splits, summarize_pawf_splits
from .npwf import make_inner_folds, summarize_inner_folds
from .regime import fit_regimes, assign_regime, similarity
from .orthogonals import (
    make_horizon_holdouts,
    make_calendar_holdouts,
    is_month_end,
)
from .bootstrap import (
    stationary_bootstrap,
    heavy_block_bootstrap,
    bootstrap_skill_delta,
)
from .adversarial import (
    compute_adversarial_auc,
    check_drift_gate,
    compute_feature_importance_drift,
)

__all__ = [
    "SplitSpec", "generate_split_id",
    "build_pawf_splits", "summarize_pawf_splits",
    "make_inner_folds", "summarize_inner_folds",
    "fit_regimes", "assign_regime", "similarity",
    "make_horizon_holdouts", "make_calendar_holdouts", "is_month_end",
    "stationary_bootstrap", "heavy_block_bootstrap", "bootstrap_skill_delta",
    "compute_adversarial_auc", "check_drift_gate", "compute_feature_importance_drift",
]
