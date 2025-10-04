"""Calibration metrics and simple probability calibrators."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _ensure_numpy(a: Iterable[float]) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _bin_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    bin_true = np.full(n_bins, np.nan)
    bin_pred = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    successes = np.zeros(n_bins)

    for idx in range(n_bins):
        mask = bin_ids == idx
        counts[idx] = mask.sum()
        if counts[idx] > 0:
            bin_true[idx] = y_true[mask].mean()
            bin_pred[idx] = y_prob[mask].mean()
            successes[idx] = y_true[mask].sum()
        else:
            bin_pred[idx] = 0.5 * (bins[idx] + bins[idx + 1])

    return bin_pred, bin_true, counts, successes


def calculate_ece(y_true: Iterable[float], y_prob: Iterable[float], n_bins: int = 10) -> float:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = np.clip(_ensure_numpy(y_prob), 1e-12, 1 - 1e-12)
    bin_pred, _, counts, successes = _bin_predictions(y_true_arr, y_prob_arr, n_bins)
    non_empty = counts > 0
    if not np.any(non_empty):
        return 0.0
    observed = (successes[non_empty] + 0.5) / (counts[non_empty] + 1.0)
    abs_diff = np.abs(observed - bin_pred[non_empty])
    weights = counts[non_empty] / counts.sum()
    return float(np.sum(abs_diff * weights))


def calculate_mce(y_true: Iterable[float], y_prob: Iterable[float], n_bins: int = 10) -> float:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = np.clip(_ensure_numpy(y_prob), 1e-12, 1 - 1e-12)
    bin_pred, _, counts, successes = _bin_predictions(y_true_arr, y_prob_arr, n_bins)
    non_empty = counts > 0
    if not np.any(non_empty):
        return 0.0
    observed = (successes[non_empty] + 0.5) / (counts[non_empty] + 1.0)
    return float(np.nanmax(np.abs(observed - bin_pred[non_empty])))


def brier_score(y_true: Iterable[float], y_prob: Iterable[float]) -> float:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = np.clip(_ensure_numpy(y_prob), 0.0, 1.0)
    return float(np.mean(np.square(y_prob_arr - y_true_arr)))


def log_loss(y_true: Iterable[float], y_prob: Iterable[float], eps: float = 1e-12) -> float:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = np.clip(_ensure_numpy(y_prob), eps, 1 - eps)
    loss = - (y_true_arr * np.log(y_prob_arr) + (1.0 - y_true_arr) * np.log(1.0 - y_prob_arr))
    return float(np.mean(loss))


def fit_isotonic_calibrator(y_true: Iterable[float], y_prob: Iterable[float]) -> IsotonicRegression:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = _ensure_numpy(y_prob)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_prob_arr, y_true_arr)
    calibrator.calibrator_type = 'isotonic'  # type: ignore[attr-defined]
    return calibrator


def fit_platt_calibrator(y_true: Iterable[int], y_prob: Iterable[float]) -> LogisticRegression:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = _ensure_numpy(y_prob)
    model = LogisticRegression(solver='lbfgs')
    model.fit(y_prob_arr.reshape(-1, 1), y_true_arr.astype(int))
    model.calibrator_type = 'platt'  # type: ignore[attr-defined]
    return model


def apply_calibrator(calibrator, y_prob: Iterable[float]) -> np.ndarray:
    y_prob_arr = _ensure_numpy(y_prob)
    calib_type = getattr(calibrator, 'calibrator_type', None)
    if calib_type == 'isotonic':
        calibrated = calibrator.predict(y_prob_arr)
    elif calib_type == 'platt':
        calibrated = calibrator.predict_proba(y_prob_arr.reshape(-1, 1))[:, 1]
    else:
        raise ValueError("Unsupported calibrator provided")
    return np.clip(calibrated, 0.0, 1.0)


def pit_test(y_true: Iterable[float], y_prob: Iterable[float]) -> Tuple[float, float]:
    pit_values = np.clip(_ensure_numpy(y_prob), 0.0, 1.0)
    ks_stat, p_value = stats.kstest(pit_values, 'uniform')
    return float(ks_stat), float(p_value)


def reliability_curve(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    y_true_arr = _ensure_numpy(y_true)
    y_prob_arr = np.clip(_ensure_numpy(y_prob), 0.0, 1.0)
    bin_pred, bin_true, counts, _ = _bin_predictions(y_true_arr, y_prob_arr, n_bins)
    bin_centers = bin_pred
    observed = np.where(counts > 0, bin_true, np.nan)
    return bin_centers, observed
