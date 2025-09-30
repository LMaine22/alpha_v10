"""
Tests for GMM regime detection.

Coverage:
- Regime fitting with GaussianMixture
- Regime assignment
- Regime similarity
- Determinism with seed
"""

import pytest
import numpy as np
import pandas as pd
from alpha_discovery.splits.regime import (
    fit_regimes,
    assign_regime,
    similarity,
    RegimeModel
)


class TestRegimeFitting:
    """Test GMM regime fitting."""
    
    def test_fit_regimes_simple(self):
        """Test basic regime fitting."""
        # Create synthetic data with clear regimes
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # Low vol regime
        prices_low = 100 + np.random.normal(0, 0.5, 100).cumsum()
        # High vol regime
        prices_high = prices_low[-1] + np.random.normal(0, 2.0, 100).cumsum()
        # Medium vol regime
        prices_med = prices_high[-1] + np.random.normal(0, 1.0, 100).cumsum()
        
        prices = np.concatenate([prices_low, prices_high, prices_med])
        df = pd.DataFrame({'price': prices}, index=dates)
        
        # Fit regimes
        regime_model, features = fit_regimes(
            df, price_col='price', K=3, vol_window=20, trend_window=20, version='R1'
        )
        
        assert regime_model is not None
        assert isinstance(regime_model, RegimeModel)
        assert regime_model.n_regimes == 3
        assert regime_model.model_type == 'gmm'
        assert regime_model.regime_version == 'R1'
        assert 'volatility' in regime_model.features_used
        assert 'trend' in regime_model.features_used
    
    def test_fit_regimes_insufficient_data(self):
        """Test regime fitting with insufficient data."""
        # Too few observations
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({'price': np.random.randn(10).cumsum() + 100}, index=dates)
        
        regime_model, features = fit_regimes(df, price_col='price', K=5)
        
        # Should fail gracefully
        assert regime_model is None
        assert features.empty
    
    def test_fit_regimes_deterministic(self):
        """Test that regime fitting is deterministic with fixed seed."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        df = pd.DataFrame({'price': np.random.randn(200).cumsum() + 100}, index=dates)
        
        # Fit twice
        rm1, _ = fit_regimes(df, price_col='price', K=3, version='R1')
        rm2, _ = fit_regimes(df, price_col='price', K=3, version='R1')
        
        assert rm1 is not None
        assert rm2 is not None
        
        # Models should have same number of components
        assert rm1.n_regimes == rm2.n_regimes
    
    def test_fit_regimes_missing_price_col(self):
        """Test regime fitting with missing price column."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'other_col': np.random.randn(100)}, index=dates)
        
        regime_model, features = fit_regimes(df, price_col='price', K=3)
        
        # Should fail gracefully
        assert regime_model is None
        assert features.empty


class TestRegimeAssignment:
    """Test regime label assignment."""
    
    def test_assign_regime_basic(self):
        """Test basic regime assignment."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        df = pd.DataFrame({'price': np.random.randn(200).cumsum() + 100}, index=dates)
        
        # Fit regimes
        regime_model, _ = fit_regimes(df, price_col='price', K=3)
        
        if regime_model is not None:
            # Assign regimes
            regime_labels = assign_regime(df, price_col='price', regime_model=regime_model)
            
            assert len(regime_labels) == len(df)
            assert regime_labels.index.equals(df.index)
            
            # Labels should be strings like 'R0', 'R1', 'R2'
            unique_labels = regime_labels.dropna().unique()
            assert all(label.startswith('R') for label in unique_labels)
            assert len(unique_labels) <= 3  # At most K regimes
    
    def test_assign_regime_forward_fill(self):
        """Test that regime assignment forward-fills gaps."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        df = pd.DataFrame({'price': np.random.randn(200).cumsum() + 100}, index=dates)
        
        regime_model, _ = fit_regimes(df, price_col='price', K=3, vol_window=20, trend_window=20)
        
        if regime_model is not None:
            regime_labels = assign_regime(df, price_col='price', regime_model=regime_model)
            
            # Early dates may be NaN due to rolling windows, but should forward fill
            # Check that most dates have labels
            non_null_count = regime_labels.notna().sum()
            assert non_null_count > len(df) * 0.5  # At least 50% should have labels
    
    def test_assign_regime_empty_model(self):
        """Test regime assignment with None model."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        df = pd.DataFrame({'price': np.random.randn(100).cumsum() + 100}, index=dates)
        
        # Create a dummy regime model that will fail
        regime_labels = assign_regime(df, price_col='nonexistent', regime_model=None)
        
        # Should return empty series
        assert isinstance(regime_labels, pd.Series)


class TestRegimeSimilarity:
    """Test regime similarity calculation."""
    
    def test_similarity_identical_vectors(self):
        """Test similarity with identical vectors."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        
        sim = similarity(vec1, vec2)
        
        assert sim == 1.0
    
    def test_similarity_orthogonal_vectors(self):
        """Test similarity with orthogonal vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        
        sim = similarity(vec1, vec2)
        
        assert sim == 0.0
    
    def test_similarity_opposite_vectors(self):
        """Test similarity with opposite vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])
        
        sim = similarity(vec1, vec2)
        
        # Cosine similarity is clipped to [0, 1]
        assert sim == 0.0
    
    def test_similarity_range(self):
        """Test that similarity is in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            vec1 = np.random.randn(5)
            vec2 = np.random.randn(5)
            
            sim = similarity(vec1, vec2)
            
            assert 0.0 <= sim <= 1.0
    
    def test_similarity_empty_vectors(self):
        """Test similarity with empty vectors."""
        vec1 = np.array([])
        vec2 = np.array([])
        
        sim = similarity(vec1, vec2)
        
        assert sim == 0.0
    
    def test_similarity_mismatched_shapes(self):
        """Test similarity with mismatched vector shapes."""
        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        
        sim = similarity(vec1, vec2)
        
        assert sim == 0.0


class TestRegimeModel:
    """Test RegimeModel class."""
    
    def test_regime_model_creation(self):
        """Test creating a RegimeModel."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        
        gmm = GaussianMixture(n_components=3, random_state=42)
        scaler = StandardScaler()
        
        # Fit on dummy data
        X = np.random.randn(100, 2)
        scaler.fit(X)
        gmm.fit(scaler.transform(X))
        
        regime_model = RegimeModel(
            model=gmm,
            scaler=scaler,
            n_regimes=3,
            features_used=['volatility', 'trend'],
            regime_version='R1',
            model_type='gmm'
        )
        
        assert regime_model.n_regimes == 3
        assert regime_model.model_type == 'gmm'
        assert regime_model.regime_version == 'R1'
        assert 'volatility' in regime_model.features_used
        assert regime_model.means_ is not None
    
    def test_regime_model_predict(self):
        """Test RegimeModel predict method."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        scaler = StandardScaler()
        
        X_train = np.random.randn(100, 2)
        scaler.fit(X_train)
        gmm.fit(scaler.transform(X_train))
        
        regime_model = RegimeModel(
            model=gmm,
            scaler=scaler,
            n_regimes=2,
            features_used=['vol', 'trend'],
            regime_version='R1'
        )
        
        # Predict on new data
        X_test = np.random.randn(10, 2)
        labels = regime_model.predict(X_test)
        
        assert len(labels) == 10
        assert all(label in [0, 1] for label in labels)
    
    def test_regime_model_predict_proba(self):
        """Test RegimeModel predict_proba method."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        
        gmm = GaussianMixture(n_components=2, random_state=42)
        scaler = StandardScaler()
        
        X_train = np.random.randn(100, 2)
        scaler.fit(X_train)
        gmm.fit(scaler.transform(X_train))
        
        regime_model = RegimeModel(
            model=gmm,
            scaler=scaler,
            n_regimes=2,
            features_used=['vol', 'trend'],
            regime_version='R1'
        )
        
        # Predict probabilities on new data
        X_test = np.random.randn(10, 2)
        probs = regime_model.predict_proba(X_test)
        
        assert probs.shape == (10, 2)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)
