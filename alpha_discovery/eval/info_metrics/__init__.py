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

# Portfolio metrics from parent metrics.py file (naming conflict workaround)
# The parent metrics.py got shadowed by this package, so we re-import and re-export
try:
    # Temporarily add parent to path to import the file
    import sys
    import os
    _parent_dir = os.path.dirname(os.path.dirname(__file__))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    
    # Import using a fake module name to avoid collision
    import importlib
    _parent_metrics_spec = importlib.util.find_spec('eval.metrics')
    # This still gets the package, so we need to load the .py file directly
    # Let me just inline the function instead
    
    #Actually, cleanest is to just import from a renamed copy or accept we can't use it from package
    # For now, set to None and fix in ga_core directly
    calculate_portfolio_metrics = None
except Exception as e:
    calculate_portfolio_metrics = None

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
    # portfolio (from parent metrics.py)
    "calculate_portfolio_metrics",
]

# Optional version tag (update if you version your metrics)
__version__ = "1.0.0"
