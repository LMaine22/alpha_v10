"""
Adapters package for forecast-first discovery.

Provides read-only interfaces to existing feature/signal infrastructure
without modifying legacy code. Enables clean separation between discovery
and production systems.

Public API:
- FeatureAdapter: Read-only access to feature registry
- FeatureSubspace: Feature subset sampling for robustness
- calculate_max_lookback: Determine embargo periods from features
"""

from .features import FeatureAdapter, calculate_max_lookback
from .subspace import FeatureSubspace, sample_feature_subspaces

__all__ = [
    "FeatureAdapter",
    "calculate_max_lookback",
    "FeatureSubspace",
    "sample_feature_subspaces",
]
