from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Dict

from sklearn.metrics import roc_auc_score, mean_absolute_error, brier_score_loss

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # For degenerate single-class windows, return 0.5 (uninformative)
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5

def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))

def ece_bin(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error on [0,1] probabilities.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        conf = y_prob[sel].mean()
        acc = y_true[sel].mean()
        w = sel.mean()
        ece += w * abs(acc - conf)
    return float(ece)

def weighted_interval_score(y_true: np.ndarray,
                            q_low_pred: np.ndarray,
                            q_med_pred: np.ndarray,
                            q_high_pred: np.ndarray,
                            alpha_low: float = 0.10,
                            alpha_high: float = 0.90) -> float:
    """
    WIS ≈ CRPS-like proper scoring for quantile intervals. Lower is better.
    """
    y = y_true
    l = q_low_pred
    m = q_med_pred
    u = q_high_pred
    alpha = alpha_high - alpha_low  # typically 0.8 for 10–90
    # interval score part
    width = u - l
    under = 2.0 / alpha * (l - y) * (y < l)
    over = 2.0 / alpha * (y - u) * (y > u)
    iscore = width + under + over
    # add median absolute error component
    mae_med = np.abs(y - m)
    wis = np.mean(0.5 * mae_med + 0.5 * iscore)
    return float(wis)

def summarize_regression_scores(y_true: np.ndarray,
                                q10: np.ndarray,
                                q50: np.ndarray,
                                q90: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, q50),
        "WIS": weighted_interval_score(y_true, q10, q50, q90, 0.10, 0.90),
    }
