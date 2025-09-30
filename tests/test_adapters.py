"""
Tests for feature adapters and subspace sampling.

Coverage:
- FeatureAdapter read-only behavior
- Lookback calculation
- Subspace sampling methods (random, stratified, complementary)
- Subspace determinism
"""

import pytest
import numpy as np
import pandas as pd
from alpha_discovery.adapters.features import FeatureAdapter, calculate_max_lookback
from alpha_discovery.adapters.subspace import sample_feature_subspaces, FeatureSubspace


class TestFeatureAdapter:
    """Test FeatureAdapter read-only interface."""
    
    def test_adapter_loads_features(self):
        """Test that adapter can load feature registry."""
        adapter = FeatureAdapter()
        
        # Should load features lazily
        assert adapter._feat_dict is None
        features = adapter.features
        assert features is not None
        assert isinstance(features, dict)
        assert adapter._feat_dict is not None  # Cached after first access
    
    def test_adapter_loads_pairwise(self):
        """Test that adapter can load pairwise features."""
        adapter = FeatureAdapter()
        
        # Should load pairwise lazily
        assert adapter._pairwise_dict is None
        pairwise = adapter.pairwise
        assert pairwise is not None
        assert isinstance(pairwise, dict)
        assert adapter._pairwise_dict is not None  # Cached after first access
    
    def test_adapter_is_readonly(self):
        """Test that adapter does not modify underlying registries."""
        adapter = FeatureAdapter()
        
        # Get references
        feat_before = adapter.features
        pair_before = adapter.pairwise
        
        # Access again
        feat_after = adapter.features
        pair_after = adapter.pairwise
        
        # Should be same objects (no copies)
        assert feat_after is feat_before
        assert pair_after is pair_before
    
    def test_list_features(self):
        """Test listing feature names."""
        adapter = FeatureAdapter()
        names = adapter.list_features()
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)
    
    def test_list_pairwise(self):
        """Test listing pairwise feature names."""
        adapter = FeatureAdapter()
        names = adapter.list_pairwise()
        
        assert isinstance(names, list)
        # May be empty if pairwise features not available
        assert all(isinstance(name, str) for name in names)


class TestLookbackCalculation:
    """Test lookback period calculation."""
    
    def test_calculate_lookback_simple(self):
        """Test lookback calculation with simple feature specs."""
        features = {
            'feat_a': {'lookback': 10},
            'feat_b': {'lookback': 20},
            'feat_c': {'lookback': 5}
        }
        
        lookback = calculate_max_lookback(features, {})
        assert lookback == 20
    
    def test_calculate_lookback_with_pairwise(self):
        """Test lookback calculation including pairwise features."""
        features = {
            'feat_a': {'lookback': 10}
        }
        pairwise = {
            'pair_x': {'lookback': 30},
            'pair_y': {'lookback': 15}
        }
        
        lookback = calculate_max_lookback(features, pairwise)
        assert lookback == 30
    
    def test_calculate_lookback_missing_key(self):
        """Test lookback calculation with features missing 'lookback' key."""
        features = {
            'feat_a': {'lookback': 10},
            'feat_b': {},  # Missing lookback
            'feat_c': {'lookback': 20}
        }
        
        # Should use default of 60 for missing lookback
        lookback = calculate_max_lookback(features, {})
        assert lookback == 60
    
    def test_calculate_lookback_empty(self):
        """Test lookback calculation with empty registries."""
        lookback = calculate_max_lookback({}, {})
        assert lookback == 60  # Default


class TestSubspaceSampling:
    """Test feature subspace sampling."""
    
    def test_sample_random(self):
        """Test random feature subspace sampling."""
        features = [f'feat_{i}' for i in range(100)]
        
        subspaces = sample_feature_subspaces(
            features,
            n_subspaces=5,
            method='random',
            feature_frac=0.5,
            seed=42
        )
        
        assert len(subspaces) == 5
        assert all(isinstance(s, FeatureSubspace) for s in subspaces)
        
        # Check sizes
        for s in subspaces:
            assert len(s.features) == 50  # 50% of 100
            assert s.n_features == 50
            assert s.feature_frac == 0.5
            assert s.method == 'random'
    
    def test_sample_stratified(self):
        """Test stratified feature subspace sampling."""
        features = [f'feat_{i}' for i in range(100)]
        
        subspaces = sample_feature_subspaces(
            features,
            n_subspaces=3,
            method='stratified',
            feature_frac=0.7,
            seed=42
        )
        
        assert len(subspaces) == 3
        
        # Stratified should partition features
        for s in subspaces:
            assert len(s.features) == 70
            assert s.method == 'stratified'
    
    def test_sample_complementary(self):
        """Test complementary feature subspace sampling."""
        features = [f'feat_{i}' for i in range(100)]
        
        subspaces = sample_feature_subspaces(
            features,
            n_subspaces=2,
            method='complementary',
            feature_frac=0.6,
            seed=42
        )
        
        assert len(subspaces) == 2
        
        # Complementary should have disjoint features
        feat_set_0 = set(subspaces[0].features)
        feat_set_1 = set(subspaces[1].features)
        
        # No overlap
        assert len(feat_set_0 & feat_set_1) == 0
    
    def test_sample_deterministic(self):
        """Test that same seed produces same subspaces."""
        features = [f'feat_{i}' for i in range(50)]
        
        subspaces_1 = sample_feature_subspaces(
            features, n_subspaces=3, method='random', feature_frac=0.5, seed=123
        )
        
        subspaces_2 = sample_feature_subspaces(
            features, n_subspaces=3, method='random', feature_frac=0.5, seed=123
        )
        
        # Should be identical
        assert len(subspaces_1) == len(subspaces_2)
        for s1, s2 in zip(subspaces_1, subspaces_2):
            assert s1.features == s2.features
            assert s1.subspace_id == s2.subspace_id
    
    def test_sample_different_seeds(self):
        """Test that different seeds produce different subspaces."""
        features = [f'feat_{i}' for i in range(50)]
        
        subspaces_1 = sample_feature_subspaces(
            features, n_subspaces=3, method='random', feature_frac=0.5, seed=123
        )
        
        subspaces_2 = sample_feature_subspaces(
            features, n_subspaces=3, method='random', feature_frac=0.5, seed=456
        )
        
        # Should be different
        assert len(subspaces_1) == len(subspaces_2)
        # At least one should differ
        any_different = any(
            s1.features != s2.features 
            for s1, s2 in zip(subspaces_1, subspaces_2)
        )
        assert any_different
    
    def test_subspace_coverage(self):
        """Test that subspaces cover the feature space."""
        features = [f'feat_{i}' for i in range(20)]
        
        subspaces = sample_feature_subspaces(
            features, n_subspaces=10, method='random', feature_frac=0.5, seed=42
        )
        
        # Union of all subspace features should cover most of original features
        all_subspace_features = set()
        for s in subspaces:
            all_subspace_features.update(s.features)
        
        # With 10 subspaces at 50% coverage, should get good coverage
        coverage = len(all_subspace_features) / len(features)
        assert coverage >= 0.8  # At least 80% coverage
    
    def test_invalid_feature_frac(self):
        """Test that invalid feature_frac raises error."""
        features = [f'feat_{i}' for i in range(10)]
        
        # feature_frac must be in (0, 1]
        with pytest.raises((ValueError, AssertionError)):
            sample_feature_subspaces(features, n_subspaces=2, feature_frac=0.0)
        
        with pytest.raises((ValueError, AssertionError)):
            sample_feature_subspaces(features, n_subspaces=2, feature_frac=1.5)
    
    def test_too_few_features(self):
        """Test behavior with very few features."""
        features = ['feat_1', 'feat_2']
        
        # Should still work
        subspaces = sample_feature_subspaces(
            features, n_subspaces=2, method='random', feature_frac=0.5, seed=42
        )
        
        assert len(subspaces) == 2
        # Each should have at least 1 feature
        assert all(len(s.features) >= 1 for s in subspaces)


class TestSubspaceDataclass:
    """Test FeatureSubspace dataclass."""
    
    def test_subspace_creation(self):
        """Test creating a FeatureSubspace."""
        features = ['feat_a', 'feat_b', 'feat_c']
        
        subspace = FeatureSubspace(
            subspace_id='sub_001',
            features=features,
            n_features=3,
            feature_frac=0.5,
            method='random',
            seed=42
        )
        
        assert subspace.subspace_id == 'sub_001'
        assert subspace.features == features
        assert subspace.n_features == 3
        assert subspace.feature_frac == 0.5
        assert subspace.method == 'random'
        assert subspace.seed == 42
    
    def test_subspace_immutable(self):
        """Test that FeatureSubspace is frozen (immutable)."""
        subspace = FeatureSubspace(
            subspace_id='sub_001',
            features=['feat_a'],
            n_features=1,
            feature_frac=0.5,
            method='random',
            seed=42
        )
        
        # Should not be able to modify
        with pytest.raises(AttributeError):
            subspace.features = ['feat_b']
