# alpha_discovery/eval/metrics/info_theory.py
"""
Information-theoretic metrics (no placeholders).
"""
from __future__ import annotations
from typing import Optional, Tuple
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

def transfer_entropy(x_values: np.ndarray,
                     y_values: np.ndarray,
                     lag: int = 1,
                     bins: int = 10) -> float:
    """
    TE_{X→Y} using discretization:
    I(Y_t ; X_{t-1} | Y_{t-1})
    """
    x = _as_1d(x_values); y = _as_1d(y_values)
    if x.size <= lag or y.size <= lag:
        return np.nan

    # Align series
    Yt = y[lag:]
    Ylag = y[:-lag]
    Xlag = x[:-lag]
    n = min(Yt.size, Ylag.size, Xlag.size)
    Yt = Yt[:n]; Ylag = Ylag[:n]; Xlag = Xlag[:n]

    # Discretize with quantile bins for robustness
    def qbin(v, k):
        qs = np.quantile(v, np.linspace(0, 1, k+1))
        # Ensure strictly increasing (dedupe)
        qs = np.unique(qs)
        # If collapse, fallback to uniform bins
        if qs.size < k+1:
            qs = np.linspace(v.min()-1e-9, v.max()+1e-9, k+1)
        idx = np.clip(np.digitize(v, qs[1:-1], right=False), 0, k-1)
        return idx

    Yi = qbin(Yt, bins)
    Yl = qbin(Ylag, bins)
    Xl = qbin(Xlag, bins)

    # Joint counts P(Yt, Ylag, Xlag)
    J = np.zeros((bins, bins, bins), dtype=float)
    for a, b, c in zip(Yi, Yl, Xl):
        J[a, b, c] += 1.0
    J += 1.0  # Laplace
    P = J / J.sum()

    # Marginals
    Py_y = P.sum(axis=2)                      # P(Yt, Ylag)
    Py = Py_y.sum(axis=1, keepdims=True)      # P(Yt)
    Py_lag = Py_y.sum(axis=0, keepdims=True)  # P(Ylag)
    Px_lag = P.sum(axis=(0, 1), keepdims=True)  # P(Xlag)

    # TE = sum P(y_t,y_{t-1},x_{t-1}) log [ P(y_t | y_{t-1}, x_{t-1}) / P(y_t | y_{t-1}) ]
    with np.errstate(divide='ignore', invalid='ignore'):
        P_y_given_yl_xl = P / np.maximum(Py_lag * Px_lag, EPS)
        P_y_given_yl = Py_y / np.maximum(Py_lag, EPS)
        ratio = P_y_given_yl_xl / np.maximum(P_y_given_yl[:, :, None], EPS)
        logratio = np.where(ratio > 0, np.log2(ratio), 0.0)
    return float(np.sum(P * logratio))

__all__ = ["entropy", "conditional_entropy", "info_gain", "mutual_information", "transfer_entropy"]
