"""Feature subspace sampling for robustness testing."""

from __future__ import annotations
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import numpy as np
import hashlib


@dataclass
class FeatureSubspace:
    """
    A specific subset of features for robustness testing.
    
    Used to test if a signal/setup maintains skill when evaluated
    on different feature subsets (feature durability).
    """
    
    subspace_id: str
    features: List[str]
    sampling_method: str
    fraction: float
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate subspace."""
        assert 0.0 < self.fraction <= 1.0, "Fraction must be in (0, 1]"
        assert len(self.features) > 0, "Feature list cannot be empty"
    
    @property
    def size(self) -> int:
        """Number of features in this subspace."""
        return len(self.features)
    
    def __hash__(self):
        """Make hashable for caching."""
        return hash(self.subspace_id)
    
    def __eq__(self, other):
        """Equality based on subspace_id."""
        if not isinstance(other, FeatureSubspace):
            return False
        return self.subspace_id == other.subspace_id


def sample_feature_subspaces(
    all_features: List[str],
    n_subspaces: int = 5,
    fraction: float = 0.7,
    method: str = "stratified",
    seed: Optional[int] = None,
    feature_categories: Optional[Dict[str, List[str]]] = None
) -> List[FeatureSubspace]:
    """
    Generate multiple feature subspaces for robustness testing.
    
    Tests whether a signal's predictive power depends on specific features
    or is robust across feature subsets.
    
    Args:
        all_features: Full list of available features
        n_subspaces: Number of subspaces to generate
        fraction: Fraction of features to include in each subspace (0.5-0.9 typical)
        method: Sampling method:
            - "random": Uniform random sampling
            - "stratified": Sample proportionally from each category (px., vol., etc.)
            - "complementary": Create non-overlapping subspaces
        seed: Random seed for reproducibility
        feature_categories: Optional dict of category → features for stratified sampling
        
    Returns:
        List of FeatureSubspace objects
        
    Example:
        >>> subspaces = sample_feature_subspaces(
        ...     all_features=['px.mom_5', 'px.mom_21', 'vol.rv_21'],
        ...     n_subspaces=3,
        ...     fraction=0.7,
        ...     method="random",
        ...     seed=42
        ... )
        >>> for subspace in subspaces:
        ...     print(f"{subspace.subspace_id}: {subspace.size} features")
    """
    rng = np.random.default_rng(seed)
    
    if method == "random":
        return _sample_random_subspaces(
            all_features, n_subspaces, fraction, rng
        )
    elif method == "stratified":
        return _sample_stratified_subspaces(
            all_features, n_subspaces, fraction, rng, feature_categories
        )
    elif method == "complementary":
        return _sample_complementary_subspaces(
            all_features, n_subspaces, fraction, rng
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _sample_random_subspaces(
    all_features: List[str],
    n_subspaces: int,
    fraction: float,
    rng: np.random.Generator
) -> List[FeatureSubspace]:
    """Random uniform sampling of feature subsets."""
    subspaces = []
    n_features = len(all_features)
    subset_size = max(1, int(n_features * fraction))
    
    for i in range(n_subspaces):
        # Random sample without replacement
        sampled = rng.choice(all_features, size=subset_size, replace=False).tolist()
        
        subspace_id = _generate_subspace_id(
            sampled, method="random", fraction=fraction, iteration=i
        )
        
        subspaces.append(FeatureSubspace(
            subspace_id=subspace_id,
            features=sorted(sampled),
            sampling_method="random",
            fraction=fraction,
            seed=None  # Don't store seed in individual subspaces
        ))
    
    return subspaces


def _sample_stratified_subspaces(
    all_features: List[str],
    n_subspaces: int,
    fraction: float,
    rng: np.random.Generator,
    feature_categories: Optional[Dict[str, List[str]]] = None
) -> List[FeatureSubspace]:
    """
    Stratified sampling: maintain category proportions.
    
    If categories not provided, infer from feature name prefixes (e.g., 'px.', 'vol.').
    """
    # Infer categories if not provided
    if feature_categories is None:
        feature_categories = _infer_feature_categories(all_features)
    
    subspaces = []
    
    for i in range(n_subspaces):
        sampled = []
        
        # Sample from each category proportionally
        for category, cat_features in feature_categories.items():
            cat_size = max(1, int(len(cat_features) * fraction))
            cat_sample = rng.choice(cat_features, size=cat_size, replace=False).tolist()
            sampled.extend(cat_sample)
        
        subspace_id = _generate_subspace_id(
            sampled, method="stratified", fraction=fraction, iteration=i
        )
        
        subspaces.append(FeatureSubspace(
            subspace_id=subspace_id,
            features=sorted(sampled),
            sampling_method="stratified",
            fraction=fraction
        ))
    
    return subspaces


def _sample_complementary_subspaces(
    all_features: List[str],
    n_subspaces: int,
    fraction: float,
    rng: np.random.Generator
) -> List[FeatureSubspace]:
    """
    Complementary sampling: minimize overlap between subspaces.
    
    Creates subspaces with minimal feature intersection, useful for
    testing independence of signal components.
    """
    subspaces = []
    n_features = len(all_features)
    subset_size = max(1, int(n_features * fraction))
    
    # Shuffle features
    shuffled = all_features.copy()
    rng.shuffle(shuffled)
    
    # Create non-overlapping chunks (with wrap-around if needed)
    for i in range(n_subspaces):
        start_idx = (i * subset_size) % n_features
        
        # Collect features in circular fashion
        sampled = []
        for j in range(subset_size):
            idx = (start_idx + j) % n_features
            sampled.append(shuffled[idx])
        
        subspace_id = _generate_subspace_id(
            sampled, method="complementary", fraction=fraction, iteration=i
        )
        
        subspaces.append(FeatureSubspace(
            subspace_id=subspace_id,
            features=sorted(sampled),
            sampling_method="complementary",
            fraction=fraction
        ))
    
    return subspaces


def _infer_feature_categories(features: List[str]) -> Dict[str, List[str]]:
    """
    Infer feature categories from name prefixes.
    
    Groups features by prefix before first dot (e.g., 'px.mom_5' → 'px').
    """
    categories = {}
    
    for feat in features:
        if '.' in feat:
            category = feat.split('.')[0]
        else:
            category = 'other'
        
        if category not in categories:
            categories[category] = []
        categories[category].append(feat)
    
    return categories


def _generate_subspace_id(
    features: List[str],
    method: str,
    fraction: float,
    iteration: int
) -> str:
    """
    Generate deterministic subspace ID.
    
    Format: "{method}_{fraction:.0%}_n{size}_{hash[:8]}"
    
    Example: "random_70%_n42_a3b5c7d9"
    """
    # Sort features for deterministic hash
    feat_str = "|".join(sorted(features))
    feat_hash = hashlib.sha1(feat_str.encode()).hexdigest()[:8]
    
    pct = int(fraction * 100)
    n_feat = len(features)
    
    return f"{method}_{pct}%_n{n_feat}_{feat_hash}"


def calculate_subspace_overlap(
    subspaces: List[FeatureSubspace]
) -> np.ndarray:
    """
    Calculate pairwise Jaccard similarity between subspaces.
    
    Args:
        subspaces: List of FeatureSubspace objects
        
    Returns:
        NxN matrix where entry [i,j] is Jaccard similarity between
        subspaces i and j
    """
    n = len(subspaces)
    overlap = np.zeros((n, n))
    
    for i in range(n):
        set_i = set(subspaces[i].features)
        for j in range(i, n):
            set_j = set(subspaces[j].features)
            
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            
            jaccard = intersection / union if union > 0 else 0.0
            overlap[i, j] = jaccard
            overlap[j, i] = jaccard
    
    return overlap
