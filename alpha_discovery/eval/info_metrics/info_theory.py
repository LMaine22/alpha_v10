# alpha_discovery/eval/metrics/info_theory.py
"""
Information-theoretic metrics (no placeholders).
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd

EPS = 1e-12

def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()

def _normalize(p: np.ndarray) -> np.ndarray:
    s = np.nansum(p)
    if s <= 0 or not np.isfinite(s):
        return np.full_like(p, np.nan)
    return np.clip(p / s, 0.0, 1.0)

def entropy(probabilities: np.ndarray, base: float = 2.0) -> float:
    """
    Shannon entropy H(P) = -∑ p log p (with finite-sample stabilization).
    """
    p = _normalize(_as_1d(probabilities))
    if p.size == 0 or np.any(~np.isfinite(p)):
        return np.nan
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum(p * (np.log(p) / np.log(base))))

def conditional_entropy(joint_probs: np.ndarray,
                        marginal_probs: np.ndarray,
                        base: float = 2.0) -> float:
    """
    Conditional entropy H(Y|X) from joint P(X,Y) and P(X).
    """
    Pxy = np.asarray(joint_probs, dtype=float)
    Px = np.asarray(marginal_probs, dtype=float).ravel()
    if Px.ndim != 1:
        raise ValueError("marginal_probs must be 1D (P(X)).")
    if Pxy.shape[0] != Px.size:
        raise ValueError("P(X,Y) first dim must equal len(P(X)).")

    Px = _normalize(Px)
    if np.any(~np.isfinite(Px)):
        return np.nan
    Pxy = np.clip(Pxy, 0.0, np.inf)
    s = Pxy.sum()
    if s <= 0:
        return np.nan
    Pxy = Pxy / s

    # H(Y|X) = -∑_x ∑_y P(x,y) log P(y|x)
    with np.errstate(divide='ignore', invalid='ignore'):
        Py_given_x = np.where(Px[:, None] > 0, Pxy / (Px[:, None] + EPS), 0.0)
        terms = np.where(Py_given_x > 0, Pxy * (np.log(Py_given_x) / np.log(base)), 0.0)
    return float(-np.sum(terms))

def info_gain(forecast_probs: np.ndarray,
              observed_values: np.ndarray,
              band_edges: np.ndarray,
              base: float = 2.0) -> float:
    """
    Information gain = KL(P_emp || P_forecast). Laplace-smoothed and symmetric-safe.
    """
    fp = _normalize(_as_1d(forecast_probs))
    ys = _as_1d(observed_values)
    be = _as_1d(band_edges)
    if fp.size == 0 or ys.size == 0:
        return np.nan
    counts, _ = np.histogram(ys[np.isfinite(ys)], bins=be)
    emp = counts.astype(float)
    if emp.sum() == 0:
        return np.nan
    # Laplace smoothing (Jeffreys 1/2 also possible)
    emp = (emp + 1.0) / (emp.sum() + emp.size)
    fp = np.clip(fp, EPS, 1.0)
    return float(np.sum(emp * (np.log(emp) - np.log(fp))) / np.log(base))

def mutual_information(x_values: np.ndarray,
                       y_values: np.ndarray,
                       x_bins: int = 16,
                       y_bins: int = 16) -> float:
    """
    MI(X;Y) via bivariate histogram with Laplace smoothing.
    """
    x = _as_1d(x_values); y = _as_1d(y_values)
    if x.size == 0 or y.size == 0:
        return np.nan
    # Joint histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    H = H.astype(float)
    H += 1.0  # Laplace
    Pxy = H / H.sum()
    Px = Pxy.sum(axis=1, keepdims=True)
    Py = Pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = Pxy / (Px @ Py)
        logfrac = np.where(frac > 0, np.log2(frac), 0.0)
    return float(np.sum(Pxy * logfrac))

def transfer_entropy(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray], lag: int = 1, bins: int = 3) -> float:
    """
    Calculate the Transfer Entropy from series x to series y using discrete bins.
    TE_{X→Y} ≈ ∑ p(y_t, y_{t-1}, x_{t-1}) log [ p(y_t | y_{t-1}, x_{t-1}) / p(y_t | y_{t-1}) ]
    Returns 0.0 on insufficient data.
    """
    try:
        L = int(max(1, lag))
        B = int(max(3, bins))
    except Exception:
        L = 1
        B = 3

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    N = min(x.size, y.size)
    if N <= L + 5:
        return np.nan

    x = x[:N]
    y = y[:N]

    # Build lagged arrays
    y_t = y[L:]
    y_p = y[:-L]
    x_p = x[:-L]

    # Drop any NaNs
    mask = np.isfinite(y_t) & np.isfinite(y_p) & np.isfinite(x_p)
    if mask.sum() <= 5:
        return np.nan
    y_t = y_t[mask]
    y_p = y_p[mask]
    x_p = x_p[mask]

    # Discretize each variable into B bins using histogram edges
    try:
        y_t_edges = np.histogram_bin_edges(y_t, bins=B)
        y_p_edges = np.histogram_bin_edges(y_p, bins=B)
        x_p_edges = np.histogram_bin_edges(x_p, bins=B)

        y_t_bins = np.clip(np.digitize(y_t, y_t_edges) - 1, 0, B - 1)
        y_p_bins = np.clip(np.digitize(y_p, y_p_edges) - 1, 0, B - 1)
        x_p_bins = np.clip(np.digitize(x_p, x_p_edges) - 1, 0, B - 1)
    except Exception:
        return np.nan

    # Joint counts with Laplace smoothing
    counts = np.ones((B, B, B), dtype=float)  # +1 Laplace
    np.add.at(counts, (y_t_bins, y_p_bins, x_p_bins), 1.0)

    Pypx = counts / counts.sum()
    Pyx = Pypx.sum(axis=0)          # sum over y_t -> P(y_p, x_p)
    Py = Pypx.sum(axis=(0, 2))      # sum over y_t,x_p -> P(y_p)
    Py_t_y_p = Pypx.sum(axis=2)     # P(y_t, y_p)

    # Conditional probabilities with EPS guards
    with np.errstate(divide='ignore', invalid='ignore'):
        P_y_t_given_y_p_x_p = Pypx / (Pyx[None, :, :] + EPS)
        P_y_t_given_y_p = Py_t_y_p / (Py[None, :] + EPS)
        ratio = P_y_t_given_y_p_x_p / (P_y_t_given_y_p[:, :, None] + EPS)
        log_ratio = np.where(ratio > 0, np.log2(ratio), 0.0)
        te = np.sum(Pypx * log_ratio)

    if not np.isfinite(te):
        return np.nan
    return float(te)


__all__ = ["entropy", "conditional_entropy", "info_gain", "mutual_information", "transfer_entropy"]
