"""
Calibration tools for probabilistic forecasts.

Provides methods for:
- Calibration assessment (ECE, MCE, reliability diagrams)
- Calibration correction (Isotonic Regression, Platt Scaling)
- Distributional tests (PIT, coverage diagnostics)
- Sharpness vs calibration trade-offs

References:
- Gneiting & Raftery (2007) - Strictly Proper Scoring Rules
- Kuleshov et al. (2018) - Accurate Uncertainties for Deep Learning
- Naeini et al. (2015) - Obtaining Well Calibrated Probabilities
"""

from __future__ import annotations
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Calibration assessment metrics."""
    
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error  
    brier_score: float  # Brier score (proper scoring rule)
    log_loss: float  # Logarithmic loss (proper scoring rule)
    sharpness: float  # Average confidence (higher = less uncertain)
    n_bins: int  # Number of bins used
    bin_accs: List[float]  # Accuracy per bin
    bin_confs: List[float]  # Confidence per bin
    bin_counts: List[int]  # Sample count per bin
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ece': self.ece,
            'mce': self.mce,
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'sharpness': self.sharpness,
            'n_bins': self.n_bins,
            'bin_accs': self.bin_accs,
            'bin_confs': self.bin_confs,
            'bin_counts': self.bin_counts
        }


def compute_ece_mce(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> Tuple[float, float, Dict]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    ECE measures average calibration error across probability bins.
    MCE measures worst-case calibration error.
    
    Args:
        probabilities: Predicted probabilities (N,) or (N, K) for K classes
        outcomes: True outcomes (N,) - binary 0/1 or class indices
        n_bins: Number of bins for calibration assessment
        strategy: Binning strategy - "uniform" (equal width) or "quantile" (equal freq)
        
    Returns:
        (ece, mce, diagnostics) where diagnostics contains per-bin statistics
    """
    # Handle multiclass by taking predicted class probability
    if probabilities.ndim == 2:
        # For multiclass, use probability of predicted class
        predicted_class = probabilities.argmax(axis=1)
        probs = probabilities[np.arange(len(probabilities)), predicted_class]
        correct = (predicted_class == outcomes).astype(float)
    else:
        # Binary case
        probs = probabilities
        correct = outcomes.astype(float)
    
    # Create bins
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(probs, np.linspace(0, 100, n_bins + 1))
        bins[0] = 0.0  # Ensure bounds
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    
    # Assign samples to bins
    bin_indices = np.digitize(probs, bins[1:-1])
    
    # Compute per-bin statistics
    bin_accs = []
    bin_confs = []
    bin_counts = []
    calibration_errors = []
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        count = mask.sum()
        
        if count > 0:
            # Accuracy = fraction of correct predictions in bin
            acc = correct[mask].mean()
            # Confidence = average predicted probability in bin
            conf = probs[mask].mean()
            
            bin_accs.append(float(acc))
            bin_confs.append(float(conf))
            bin_counts.append(int(count))
            calibration_errors.append(abs(acc - conf) * count)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)
            calibration_errors.append(0.0)
    
    # ECE: weighted average of calibration errors
    ece = float(np.sum(calibration_errors) / len(probs))
    
    # MCE: maximum calibration error across bins
    bin_errors = [abs(acc - conf) for acc, conf, cnt in zip(bin_accs, bin_confs, bin_counts) if cnt > 0]
    mce = float(max(bin_errors)) if bin_errors else 0.0
    
    diagnostics = {
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'bins': bins.tolist()
    }
    
    return ece, mce, diagnostics


def compute_brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray
) -> float:
    """
    Compute Brier score (mean squared error of probabilities).
    
    Proper scoring rule for binary and multiclass forecasts.
    Lower is better (0 = perfect, 1 = worst for binary).
    
    Args:
        probabilities: Predicted probabilities (N,) or (N, K)
        outcomes: True outcomes (N,) - binary 0/1 or class indices
        
    Returns:
        Brier score
    """
    if probabilities.ndim == 1:
        # Binary case
        targets = outcomes.astype(float)
        return float(np.mean((probabilities - targets) ** 2))
    else:
        # Multiclass case: one-hot encode outcomes
        n_classes = probabilities.shape[1]
        targets = np.zeros_like(probabilities)
        targets[np.arange(len(outcomes)), outcomes.astype(int)] = 1
        return float(np.mean(np.sum((probabilities - targets) ** 2, axis=1)))


def compute_log_loss(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Compute logarithmic loss (cross-entropy).
    
    Proper scoring rule. Lower is better.
    
    Args:
        probabilities: Predicted probabilities (N,) or (N, K)
        outcomes: True outcomes (N,) - binary 0/1 or class indices
        eps: Small constant to avoid log(0)
        
    Returns:
        Log loss
    """
    # Clip probabilities to avoid log(0)
    probs = np.clip(probabilities, eps, 1 - eps)
    
    if probs.ndim == 1:
        # Binary case
        targets = outcomes.astype(float)
        loss = -(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
        return float(np.mean(loss))
    else:
        # Multiclass case
        # Take log probability of true class
        true_class_probs = probs[np.arange(len(outcomes)), outcomes.astype(int)]
        return float(-np.mean(np.log(true_class_probs)))


def compute_sharpness(probabilities: np.ndarray) -> float:
    """
    Compute sharpness (average confidence).
    
    Measures how confident/concentrated the forecasts are.
    Higher = more confident (less uncertain).
    
    Should be interpreted alongside calibration - sharp forecasts
    are only useful if they're also calibrated.
    
    Args:
        probabilities: Predicted probabilities (N,) or (N, K)
        
    Returns:
        Sharpness score
    """
    if probabilities.ndim == 1:
        # Binary: sharpness = how far from 0.5 (maximum uncertainty)
        return float(np.mean(np.abs(probabilities - 0.5) * 2))
    else:
        # Multiclass: sharpness = max probability (1 = certain, 1/K = uniform)
        return float(np.mean(probabilities.max(axis=1)))


def assess_calibration(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Comprehensive calibration assessment.
    
    Args:
        probabilities: Predicted probabilities
        outcomes: True outcomes
        n_bins: Number of bins for ECE/MCE
        
    Returns:
        CalibrationMetrics object with all metrics
    """
    ece, mce, diagnostics = compute_ece_mce(probabilities, outcomes, n_bins)
    brier = compute_brier_score(probabilities, outcomes)
    logloss = compute_log_loss(probabilities, outcomes)
    sharpness = compute_sharpness(probabilities)
    
    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier_score=brier,
        log_loss=logloss,
        sharpness=sharpness,
        n_bins=n_bins,
        bin_accs=diagnostics['bin_accs'],
        bin_confs=diagnostics['bin_confs'],
        bin_counts=diagnostics['bin_counts']
    )


# ============================================================================
# Calibration Methods
# ============================================================================

def fit_isotonic_calibrator(
    probabilities: np.ndarray,
    outcomes: np.ndarray
) -> 'IsotonicCalibrator':
    """
    Fit isotonic regression calibrator.
    
    Non-parametric monotonic calibration that preserves ranking.
    Works well when you have enough data (>1000 samples recommended).
    
    Args:
        probabilities: Training probabilities
        outcomes: Training outcomes (binary 0/1)
        
    Returns:
        Fitted IsotonicCalibrator
    """
    from sklearn.isotonic import IsotonicRegression
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probabilities, outcomes)
    
    return IsotonicCalibrator(iso)


def fit_platt_calibrator(
    probabilities: np.ndarray,
    outcomes: np.ndarray
) -> 'PlattCalibrator':
    """
    Fit Platt scaling calibrator (logistic regression).
    
    Parametric calibration using logistic sigmoid.
    Works well with smaller datasets, assumes sigmoidal relationship.
    
    Args:
        probabilities: Training probabilities (or raw scores)
        outcomes: Training outcomes (binary 0/1)
        
    Returns:
        Fitted PlattCalibrator
    """
    from sklearn.linear_model import LogisticRegression
    
    # Convert to log-odds (inverse sigmoid)
    eps = 1e-15
    logits = np.log((probabilities + eps) / (1 - probabilities + eps))
    
    # Fit logistic regression
    lr = LogisticRegression()
    lr.fit(logits.reshape(-1, 1), outcomes)
    
    return PlattCalibrator(lr)


class IsotonicCalibrator:
    """Isotonic regression calibrator."""
    
    def __init__(self, model):
        self.model = model
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to new probabilities."""
        return self.model.predict(probabilities)


class PlattCalibrator:
    """Platt scaling calibrator."""
    
    def __init__(self, model):
        self.model = model
    
    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to new probabilities."""
        eps = 1e-15
        logits = np.log((probabilities + eps) / (1 - probabilities + eps))
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]


# ============================================================================
# Distributional Tests
# ============================================================================

def pit_histogram(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probability Integral Transform (PIT) histogram.
    
    For calibrated forecasts, PIT values should be uniform [0,1].
    Deviations indicate miscalibration patterns.
    
    Args:
        probabilities: Predicted probabilities (N,)
        outcomes: True outcomes (N,) - binary 0/1
        n_bins: Number of histogram bins
        
    Returns:
        (hist_counts, bin_edges) - histogram of PIT values
    """
    # For binary outcomes, PIT = predicted prob if outcome=1, else 1-prob
    pit_values = np.where(outcomes == 1, probabilities, 1 - probabilities)
    
    # Histogram
    counts, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    
    return counts, edges


def coverage_diagnostics(
    lower: np.ndarray,
    upper: np.ndarray,
    outcomes: np.ndarray,
    target_coverage: float = 0.9
) -> Dict[str, float]:
    """
    Assess interval forecast coverage.
    
    For well-calibrated intervals, empirical coverage should match
    target coverage (e.g., 90% of outcomes in 90% interval).
    
    Args:
        lower: Lower bounds of prediction intervals
        upper: Upper bounds of prediction intervals
        outcomes: True outcomes
        target_coverage: Target coverage level (e.g., 0.9 for 90%)
        
    Returns:
        Dict with coverage metrics
    """
    # Empirical coverage
    covered = (outcomes >= lower) & (outcomes <= upper)
    empirical_coverage = float(covered.mean())
    
    # Coverage error
    coverage_error = abs(empirical_coverage - target_coverage)
    
    # Interval width (sharpness)
    avg_width = float((upper - lower).mean())
    
    # Winkler score (proper scoring rule for intervals)
    alpha = 1 - target_coverage
    below = outcomes < lower
    above = outcomes > upper
    
    winkler = (upper - lower).copy()
    winkler[below] += (2 / alpha) * (lower[below] - outcomes[below])
    winkler[above] += (2 / alpha) * (outcomes[above] - upper[above])
    avg_winkler = float(winkler.mean())
    
    return {
        'empirical_coverage': empirical_coverage,
        'target_coverage': target_coverage,
        'coverage_error': coverage_error,
        'avg_interval_width': avg_width,
        'winkler_score': avg_winkler
    }


# ============================================================================
# Utilities
# ============================================================================

def reliability_curve(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability curve (calibration plot).
    
    Returns predicted probability vs observed frequency for each bin.
    Perfect calibration = diagonal line (predicted = observed).
    
    Args:
        probabilities: Predicted probabilities
        outcomes: True outcomes
        n_bins: Number of bins
        strategy: Binning strategy
        
    Returns:
        (mean_predicted_probs, observed_frequencies, bin_counts)
    """
    _, _, diagnostics = compute_ece_mce(probabilities, outcomes, n_bins, strategy)
    
    mean_probs = np.array(diagnostics['bin_confs'])
    obs_freqs = np.array(diagnostics['bin_accs'])
    counts = np.array(diagnostics['bin_counts'])
    
    # Filter out empty bins
    mask = counts > 0
    
    return mean_probs[mask], obs_freqs[mask], counts[mask]


__all__ = [
    'CalibrationMetrics',
    'compute_ece_mce',
    'compute_brier_score',
    'compute_log_loss',
    'compute_sharpness',
    'assess_calibration',
    'fit_isotonic_calibrator',
    'fit_platt_calibrator',
    'IsotonicCalibrator',
    'PlattCalibrator',
    'pit_histogram',
    'coverage_diagnostics',
    'reliability_curve',
]
