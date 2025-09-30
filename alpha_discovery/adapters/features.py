"""Read-only adapter for existing feature infrastructure."""

from __future__ import annotations
from typing import List, Dict, Optional, Set
import pandas as pd
import re
from functools import lru_cache


class FeatureAdapter:
    """
    Read-only adapter for accessing existing feature registry.
    
    Provides a clean interface without importing or modifying the legacy
    features.registry module. Prevents accidental mutations during discovery.
    
    Usage:
        adapter = FeatureAdapter()
        available = adapter.list_features()
        values = adapter.compute_features(df, ticker, feature_names)
        lookback = adapter.get_max_lookback(feature_names)
    """
    
    def __init__(self):
        """Initialize adapter with lazy loading of feature registry."""
        self._feat_dict = None
        self._pairwise_dict = None
    
    @property
    def feat(self) -> Dict:
        """Lazy load FEAT dictionary from registry."""
        if self._feat_dict is None:
            from alpha_discovery.features.registry import FEAT
            self._feat_dict = FEAT
        return self._feat_dict
    
    @property
    def pairwise(self) -> Dict:
        """Lazy load PAIR_SPECS dictionary from registry."""
        if self._pairwise_dict is None:
            try:
                from alpha_discovery.features.registry import PAIR_SPECS
                self._pairwise_dict = PAIR_SPECS
            except ImportError:
                # Fallback if pairwise features not available
                self._pairwise_dict = {}
        return self._pairwise_dict
    
    def list_features(
        self,
        pattern: Optional[str] = None,
        exclude_pairwise: bool = False
    ) -> List[str]:
        """
        List all available single-asset features.
        
        Args:
            pattern: Optional regex pattern to filter feature names
            exclude_pairwise: If True, exclude pairwise features
            
        Returns:
            Sorted list of feature names
        """
        features = list(self.feat.keys())
        
        if not exclude_pairwise:
            features.extend(self.pairwise.keys())
        
        if pattern:
            regex = re.compile(pattern)
            features = [f for f in features if regex.search(f)]
        
        return sorted(features)
    
    def compute_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        feature_names: List[str],
        handle_errors: bool = True
    ) -> pd.DataFrame:
        """
        Compute requested features for a single ticker.
        
        Args:
            df: Master DataFrame with all data
            ticker: Target ticker symbol
            feature_names: List of feature names to compute
            handle_errors: If True, fill failed features with NaN
            
        Returns:
            DataFrame with computed features (index aligned to df.index)
        """
        result = pd.DataFrame(index=df.index)
        
        for fname in feature_names:
            try:
                if fname in self.feat:
                    # Single-asset feature
                    result[fname] = self.feat[fname](df, ticker)
                elif fname in self.pairwise:
                    # Pairwise feature (requires benchmark parameter)
                    # For now, skip pairwise features unless explicitly handled
                    if handle_errors:
                        result[fname] = pd.Series(index=df.index, dtype=float)
                else:
                    if handle_errors:
                        result[fname] = pd.Series(index=df.index, dtype=float)
                    else:
                        raise ValueError(f"Feature '{fname}' not found in registry")
            except Exception as e:
                if handle_errors:
                    result[fname] = pd.Series(index=df.index, dtype=float)
                else:
                    raise RuntimeError(f"Failed to compute '{fname}': {e}")
        
        return result
    
    def get_max_lookback(self, feature_names: List[str]) -> int:
        """
        Calculate maximum lookback window from feature names.
        
        Used for embargo calculation in splits.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Maximum lookback window in days (minimum 20)
        """
        return calculate_max_lookback(feature_names)
    
    def validate_features(self, feature_names: List[str]) -> Dict[str, bool]:
        """
        Check which features exist in the registry.
        
        Args:
            feature_names: List of feature names to validate
            
        Returns:
            Dict mapping feature_name → exists (bool)
        """
        all_features = set(self.feat.keys()) | set(self.pairwise.keys())
        return {fname: fname in all_features for fname in feature_names}
    
    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Group features by category prefix (e.g., 'px.', 'vol.', 'opt.').
        
        Returns:
            Dict mapping category → list of feature names
        """
        categories = {}
        for fname in self.list_features(exclude_pairwise=True):
            if '.' in fname:
                category = fname.split('.')[0]
                if category not in categories:
                    categories[category] = []
                categories[category].append(fname)
            else:
                if 'other' not in categories:
                    categories['other'] = []
                categories['other'].append(fname)
        
        return {k: sorted(v) for k, v in sorted(categories.items())}


@lru_cache(maxsize=256)
def calculate_max_lookback(feature_tuple: tuple) -> int:
    """
    Calculate maximum lookback window from feature names.
    
    Parses lookback periods from feature naming conventions:
    - Numbers after underscores: _21, _63, _252
    - Numbers after dots: .ma_50_200
    - Common patterns: iv30, 3m, 90d
    
    Args:
        feature_tuple: Tuple of feature names (tuple for caching)
        
    Returns:
        Maximum lookback in days (minimum 20, default 60 if no pattern found)
    """
    max_lookback = 20  # Minimum default
    
    for fname in feature_tuple:
        lookbacks = []
        
        # Extract all numbers that look like lookback periods
        # Pattern 1: _NUMBER or .NUMBER
        matches = re.findall(r'[._](\d+)', fname)
        lookbacks.extend([int(m) for m in matches])
        
        # Pattern 2: NUMBERd (e.g., 90d, 30d)
        matches = re.findall(r'(\d+)d', fname.lower())
        lookbacks.extend([int(m) for m in matches])
        
        # Pattern 3: NUMBERm for months (convert to days)
        matches = re.findall(r'(\d+)m', fname.lower())
        lookbacks.extend([int(m) * 21 for m in matches])
        
        # Take the maximum for this feature
        if lookbacks:
            max_lookback = max(max_lookback, max(lookbacks))
    
    # If no patterns found, use conservative default
    if max_lookback == 20 and len(feature_tuple) > 0:
        max_lookback = 60  # Conservative default for unknown features
    
    return max_lookback


# Wrapper for list input
def calculate_max_lookback_from_list(feature_names: List[str]) -> int:
    """
    Calculate maximum lookback from a list of feature names.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Maximum lookback in days
    """
    return calculate_max_lookback(tuple(sorted(feature_names)))
