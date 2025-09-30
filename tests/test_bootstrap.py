"""
Tests for bootstrap methods.

Coverage:
- Stationary bootstrap
- Heavy-tailed block bootstrap
- Bootstrap skill delta
- Determinism
"""

import pytest
import numpy as np
import pandas as pd
from alpha_discovery.splits.bootstrap import (
    stationary_bootstrap,
    heavy_block_bootstrap,
    bootstrap_skill_delta
)


class TestStationaryBootstrap:
    """Test stationary bootstrap method."""
    
    def test_stationary_bootstrap_basic(self):
        """Test basic stationary bootstrap."""
        data = np.arange(100)
        
        bootstrap_samples = stationary_bootstrap(
            data, n_bootstrap=10, avg_block_length=5, seed=42
        )
        
        assert len(bootstrap_samples) == 10
        assert all(len(sample) == len(data) for sample in bootstrap_samples)
    
    def test_stationary_bootstrap_deterministic(self):
        """Test that stationary bootstrap is deterministic with seed."""
        data = np.arange(50)
        
        samples1 = stationary_bootstrap(data, n_bootstrap=5, avg_block_length=5, seed=123)
        samples2 = stationary_bootstrap(data, n_bootstrap=5, avg_block_length=5, seed=123)
        
        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert np.array_equal(s1, s2)
    
    def test_stationary_bootstrap_different_seeds(self):
        """Test that different seeds produce different samples."""
        data = np.arange(50)
        
        samples1 = stationary_bootstrap(data, n_bootstrap=5, avg_block_length=5, seed=123)
        samples2 = stationary_bootstrap(data, n_bootstrap=5, avg_block_length=5, seed=456)
        
        # Should be different
        any_different = any(
            not np.array_equal(s1, s2)
            for s1, s2 in zip(samples1, samples2)
        )
        assert any_different
    
    def test_stationary_bootstrap_coverage(self):
        """Test that bootstrap samples cover original data."""
        data = np.arange(20)
        
        samples = stationary_bootstrap(data, n_bootstrap=100, avg_block_length=3, seed=42)
        
        # Union of all samples should cover most of original data
        all_values = set()
        for sample in samples:
            all_values.update(sample)
        
        coverage = len(all_values) / len(data)
        assert coverage >= 0.8  # At least 80% coverage


class TestHeavyBlockBootstrap:
    """Test heavy-tailed block bootstrap."""
    
    def test_heavy_block_basic(self):
        """Test basic heavy block bootstrap."""
        data = np.arange(100)
        
        bootstrap_samples = heavy_block_bootstrap(
            data, n_bootstrap=10, min_block_length=2, seed=42
        )
        
        assert len(bootstrap_samples) == 10
        assert all(len(sample) == len(data) for sample in bootstrap_samples)
    
    def test_heavy_block_deterministic(self):
        """Test that heavy block bootstrap is deterministic with seed."""
        data = np.arange(50)
        
        samples1 = heavy_block_bootstrap(data, n_bootstrap=5, min_block_length=3, seed=123)
        samples2 = heavy_block_bootstrap(data, n_bootstrap=5, min_block_length=3, seed=123)
        
        # Should be identical
        for s1, s2 in zip(samples1, samples2):
            assert np.array_equal(s1, s2)
    
    def test_heavy_block_coverage(self):
        """Test that heavy block samples cover original data."""
        data = np.arange(20)
        
        samples = heavy_block_bootstrap(data, n_bootstrap=100, min_block_length=2, seed=42)
        
        # Union of all samples should cover most of original data
        all_values = set()
        for sample in samples:
            all_values.update(sample)
        
        coverage = len(all_values) / len(data)
        assert coverage >= 0.7  # At least 70% coverage (may be lower due to heavy tails)


class TestBootstrapSkillDelta:
    """Test bootstrap skill delta calculation."""
    
    def test_bootstrap_skill_delta_positive(self):
        """Test skill delta with clearly superior model."""
        np.random.seed(42)
        
        # Model 1: Better predictions
        y_true = np.random.randint(0, 2, 200)
        model1_probs = np.where(y_true == 1, 
                                np.random.uniform(0.6, 0.9, 200),
                                np.random.uniform(0.1, 0.4, 200))
        
        # Model 2: Worse predictions
        model2_probs = np.where(y_true == 1,
                                np.random.uniform(0.4, 0.7, 200),
                                np.random.uniform(0.3, 0.6, 200))
        
        mean_delta, p_value, ci_lower, ci_upper = bootstrap_skill_delta(
            y_true, model1_probs, model2_probs,
            metric='brier',
            n_bootstrap=100,
            block_length=10,
            seed=42
        )
        
        # Model 1 should be better (negative Brier delta means model1 < model2)
        assert mean_delta < 0
        # P-value should be small (significant difference)
        assert p_value < 0.1
    
    def test_bootstrap_skill_delta_no_difference(self):
        """Test skill delta with equivalent models."""
        np.random.seed(42)
        
        y_true = np.random.randint(0, 2, 200)
        model_probs = np.random.uniform(0, 1, 200)
        
        # Same model twice
        mean_delta, p_value, ci_lower, ci_upper = bootstrap_skill_delta(
            y_true, model_probs, model_probs,
            metric='brier',
            n_bootstrap=100,
            seed=42
        )
        
        # Delta should be near zero
        assert abs(mean_delta) < 0.01
        # P-value should be high (not significant)
        assert p_value > 0.5
    
    def test_bootstrap_skill_delta_deterministic(self):
        """Test that skill delta is deterministic with seed."""
        np.random.seed(42)
        
        y_true = np.random.randint(0, 2, 100)
        model1_probs = np.random.uniform(0, 1, 100)
        model2_probs = np.random.uniform(0, 1, 100)
        
        result1 = bootstrap_skill_delta(
            y_true, model1_probs, model2_probs,
            n_bootstrap=50, seed=123
        )
        
        result2 = bootstrap_skill_delta(
            y_true, model1_probs, model2_probs,
            n_bootstrap=50, seed=123
        )
        
        # Results should be identical
        assert result1[0] == result2[0]  # mean_delta
        assert result1[1] == result2[1]  # p_value
        assert result1[2] == result2[2]  # ci_lower
        assert result1[3] == result2[3]  # ci_upper
    
    def test_bootstrap_skill_delta_log_loss(self):
        """Test skill delta with log loss metric."""
        np.random.seed(42)
        
        y_true = np.random.randint(0, 2, 100)
        model1_probs = np.clip(np.random.uniform(0.1, 0.9, 100), 0.01, 0.99)
        model2_probs = np.clip(np.random.uniform(0.1, 0.9, 100), 0.01, 0.99)
        
        mean_delta, p_value, ci_lower, ci_upper = bootstrap_skill_delta(
            y_true, model1_probs, model2_probs,
            metric='log_loss',
            n_bootstrap=50,
            seed=42
        )
        
        # Should return valid results
        assert np.isfinite(mean_delta)
        assert 0 <= p_value <= 1
        assert ci_lower <= mean_delta <= ci_upper
    
    def test_bootstrap_skill_delta_confidence_interval(self):
        """Test that confidence interval brackets mean delta."""
        np.random.seed(42)
        
        y_true = np.random.randint(0, 2, 100)
        model1_probs = np.random.uniform(0, 1, 100)
        model2_probs = np.random.uniform(0, 1, 100)
        
        mean_delta, p_value, ci_lower, ci_upper = bootstrap_skill_delta(
            y_true, model1_probs, model2_probs,
            n_bootstrap=100,
            alpha=0.05,  # 95% CI
            seed=42
        )
        
        # Mean should be within CI
        assert ci_lower <= mean_delta <= ci_upper
        
        # CI should be reasonable width
        assert ci_upper - ci_lower > 0
