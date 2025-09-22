# alpha_discovery/eval/metrics/__init__.py
"""
Convenience re-exports for metrics package.

Usage:
    from alpha_discovery.eval import metrics as M
    M.crps(...)
    M.entropy(...)
    M.detect_regimes(...)

All modules remain importable individually (e.g., metrics.distribution).
"""

# Distributional metrics
from .distribution import (
    crps,
    pinball_loss,
    calibration_mae,
    wasserstein1,
)

# Information theory
from .info_theory import (
    entropy,
    conditional_entropy,
    info_gain,
    mutual_information,
    transfer_entropy,
)

# Dynamical systems
from .dynamics import (
    dfa_alpha,
    rqa_metrics,
)

# Complexity
from .complexity import (
    sample_entropy,
    approximate_entropy,
    permutation_entropy,
    multiscale_entropy,
    complexity_index,
)

# Causality
from .causality import (
    granger_causality,
    ccm,
    transfer_entropy_causality,
)

# Regimes
from .regime import (
    fit_hmm_gaussian,
    detect_regimes,
    regime_metrics,
    worst_regime,
)

# Robustness
from .robustness import (
    moving_block_bootstrap,
    sensitivity_scan,
    tscv_robustness,
    page_hinkley,
)

# Topological Data Analysis (H0)
from .tda import (
    persistent_homology_h0,
    h0_landscape_vector,
    wasserstein1_h0,
    bottleneck_h0,
)

# Aggregation & ranks
from .aggregate import (
    aggregate,
    median_mad,
    trimmed_mean,
    huber_mean,
    hodges_lehmann,
    rank_stability,
    jackknife_leave_one_out,
)

__all__ = [
    # distribution
    "crps", "pinball_loss", "calibration_mae", "wasserstein1",
    # info theory
    "entropy", "conditional_entropy", "info_gain", "mutual_information", "transfer_entropy",
    # dynamics
    "dfa_alpha", "rqa_metrics",
    # complexity
    "sample_entropy", "approximate_entropy", "permutation_entropy", "multiscale_entropy", "complexity_index",
    # causality
    "granger_causality", "ccm", "transfer_entropy_causality",
    # regimes
    "fit_hmm_gaussian", "detect_regimes", "regime_metrics", "worst_regime",
    # robustness
    "moving_block_bootstrap", "sensitivity_scan", "tscv_robustness", "page_hinkley",
    # tda
    "persistent_homology_h0", "h0_landscape_vector", "wasserstein1_h0", "bottleneck_h0",
    # aggregate
    "aggregate", "median_mad", "trimmed_mean", "huber_mean", "hodges_lehmann",
    "rank_stability", "jackknife_leave_one_out",
]

# Optional version tag (update if you version your metrics)
__version__ = "1.0.0"
