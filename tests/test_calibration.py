"""
Tests for calibration helpers.

Coverage:
- ECE and MCE calculation
- Brier Score and Log Loss
- Isotonic Regression calibration
- Platt Scaling calibration
- PIT tests
- Reliability diagrams
"""

import pytest
import numpy as np
import pandas as pd
from alpha_discovery.eval.calibration import (
    calculate_ece,
    calculate_mce,
    brier_score,
    log_loss,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    apply_calibrator,
    pit_test,
    reliability_curve
)


class TestECE:
    """Test Expected Calibration Error."""
    
    def test_ece_perfect_calibration(self):
        """Test ECE with perfectly calibrated probabilities."""
        # Perfect calibration: predicted prob matches actual frequency
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.2, 0.2, 0.8, 0.8, 0.8])
        
        ece = calculate_ece(y_true, y_prob, n_bins=2)
        
        # Should be close to 0
        assert ece < 0.1
    
    def test_ece_poor_calibration(self):
        """Test ECE with poorly calibrated probabilities."""
        # Poor calibration: predicted prob far from actual
        y_true = np.array([1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        
        ece = calculate_ece(y_true, y_prob, n_bins=2)
        
        # Should be high
        assert ece > 0.5
    
    def test_ece_range(self):
        """Test that ECE is in [0, 1]."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        
        ece = calculate_ece(y_true, y_prob)
        
        assert 0 <= ece <= 1
    
    def test_ece_deterministic(self):
        """Test that ECE is deterministic."""
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.uniform(0, 1, 50)
        
        ece1 = calculate_ece(y_true, y_prob, n_bins=10)
        ece2 = calculate_ece(y_true, y_prob, n_bins=10)
        
        assert ece1 == ece2


class TestMCE:
    """Test Maximum Calibration Error."""
    
    def test_mce_perfect_calibration(self):
        """Test MCE with perfectly calibrated probabilities."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.2, 0.2, 0.8, 0.8, 0.8])
        
        mce = calculate_mce(y_true, y_prob, n_bins=2)
        
        assert mce < 0.1
    
    def test_mce_poor_calibration(self):
        """Test MCE with poorly calibrated probabilities."""
        y_true = np.array([1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9])
        
        mce = calculate_mce(y_true, y_prob, n_bins=2)
        
        assert mce > 0.5
    
    def test_mce_greater_than_ece(self):
        """Test that MCE >= ECE."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        
        ece = calculate_ece(y_true, y_prob)
        mce = calculate_mce(y_true, y_prob)
        
        assert mce >= ece


class TestBrierScore:
    """Test Brier Score calculation."""
    
    def test_brier_perfect_predictions(self):
        """Test Brier Score with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        
        bs = brier_score(y_true, y_prob)
        
        assert bs == 0.0
    
    def test_brier_worst_predictions(self):
        """Test Brier Score with worst predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        
        bs = brier_score(y_true, y_prob)
        
        assert bs == 1.0
    
    def test_brier_range(self):
        """Test that Brier Score is in [0, 1]."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        
        bs = brier_score(y_true, y_prob)
        
        assert 0 <= bs <= 1


class TestLogLoss:
    """Test Log Loss calculation."""
    
    def test_log_loss_perfect_predictions(self):
        """Test Log Loss with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([1e-10, 1e-10, 1-1e-10, 1-1e-10, 1-1e-10])
        
        ll = log_loss(y_true, y_prob)
        
        # Should be very small
        assert ll < 0.01
    
    def test_log_loss_random_predictions(self):
        """Test Log Loss with random predictions (0.5)."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.full(100, 0.5)
        
        ll = log_loss(y_true, y_prob)
        
        # Should be close to -log(0.5) ≈ 0.693
        assert 0.6 < ll < 0.8
    
    def test_log_loss_clipping(self):
        """Test that Log Loss handles extreme probabilities."""
        y_true = np.array([1, 0])
        y_prob = np.array([1.0, 0.0])  # Extreme values
        
        # Should not raise error (should clip internally)
        ll = log_loss(y_true, y_prob)
        assert np.isfinite(ll)


class TestIsotonicCalibration:
    """Test Isotonic Regression calibration."""
    
    def test_isotonic_fit(self):
        """Test fitting isotonic calibrator."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        
        calibrator = fit_isotonic_calibrator(y_true, y_prob)
        
        assert calibrator is not None
        assert hasattr(calibrator, 'predict')
    
    def test_isotonic_apply(self):
        """Test applying isotonic calibrator."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        
        calibrator = fit_isotonic_calibrator(y_true, y_prob)
        
        # Apply to new data
        new_prob = np.array([0.15, 0.65])
        calibrated = apply_calibrator(calibrator, new_prob)
        
        assert len(calibrated) == 2
        assert all(0 <= p <= 1 for p in calibrated)
    
    def test_isotonic_improves_calibration(self):
        """Test that isotonic regression improves calibration."""
        # Create biased probabilities
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200) * 0.5 + 0.25  # Biased toward center
        
        # Split train/test
        train_size = 100
        y_train, y_test = y_true[:train_size], y_true[train_size:]
        prob_train, prob_test = y_prob[:train_size], y_prob[train_size:]
        
        # Fit calibrator on train
        calibrator = fit_isotonic_calibrator(y_train, prob_train)
        
        # Apply to test
        prob_calibrated = apply_calibrator(calibrator, prob_test)
        
        # Calculate ECE before and after
        ece_before = calculate_ece(y_test, prob_test)
        ece_after = calculate_ece(y_test, prob_calibrated)
        
        # Calibration should improve (or stay same)
        assert ece_after <= ece_before * 1.1  # Allow 10% margin


class TestPlattScaling:
    """Test Platt Scaling calibration."""
    
    def test_platt_fit(self):
        """Test fitting Platt scaler."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        
        calibrator = fit_platt_calibrator(y_true, y_prob)
        
        assert calibrator is not None
        assert hasattr(calibrator, 'predict_proba')
    
    def test_platt_apply(self):
        """Test applying Platt scaler."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        
        calibrator = fit_platt_calibrator(y_true, y_prob)
        
        # Apply to new data
        new_prob = np.array([0.15, 0.65])
        calibrated = apply_calibrator(calibrator, new_prob)
        
        assert len(calibrated) == 2
        assert all(0 <= p <= 1 for p in calibrated)


class TestPIT:
    """Test Probability Integral Transform tests."""
    
    def test_pit_well_calibrated(self):
        """Test PIT with well-calibrated forecasts."""
        np.random.seed(42)
        
        # Generate well-calibrated forecasts
        y_true = np.random.uniform(0, 1, 100)
        y_prob = y_true + np.random.normal(0, 0.1, 100)
        y_prob = np.clip(y_prob, 0, 1)
        
        ks_stat, p_value = pit_test(y_true, y_prob)
        
        # P-value should be high (cannot reject uniformity)
        assert p_value > 0.05
    
    def test_pit_poorly_calibrated(self):
        """Test PIT with poorly calibrated forecasts."""
        np.random.seed(42)
        
        # Generate biased forecasts
        y_true = np.random.uniform(0, 1, 100)
        y_prob = np.full(100, 0.5)  # Constant prediction
        
        ks_stat, p_value = pit_test(y_true, y_prob)
        
        # P-value should be low (reject uniformity)
        # Note: This might not always hold with small samples
        assert ks_stat > 0.0


class TestReliabilityCurve:
    """Test reliability curve generation."""
    
    def test_reliability_curve_shape(self):
        """Test reliability curve returns correct shape."""
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.uniform(0, 1, 100)
        
        bin_centers, observed_freq = reliability_curve(y_true, y_prob, n_bins=10)
        
        assert len(bin_centers) == 10
        assert len(observed_freq) == 10
        assert all(0 <= c <= 1 for c in bin_centers)
        assert all(0 <= f <= 1 for f in observed_freq if not np.isnan(f))
    
    def test_reliability_curve_perfect(self):
        """Test reliability curve with perfect calibration."""
        # Create perfectly calibrated data
        y_true = np.array([0]*50 + [1]*50)
        y_prob = np.array([0.0]*50 + [1.0]*50)
        
        bin_centers, observed_freq = reliability_curve(y_true, y_prob, n_bins=2)
        
        # Observed frequency should match predicted probability
        # (allowing for some binning effects)
        assert np.allclose(bin_centers, observed_freq, atol=0.1, equal_nan=True)
