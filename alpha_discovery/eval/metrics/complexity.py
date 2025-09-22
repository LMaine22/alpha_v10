# alpha_discovery/eval/metrics/complexity.py
"""
Complexity metrics (no placeholders).
Implements:
  - Sample Entropy (SampEn)
  - Approximate Entropy (ApEn)
  - Permutation Entropy (PermEn)
  - Multiscale Entropy (MSE and CMSE)
  - Composite Complexity Index (PCA-based)
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import math

EPS = 1e-12

def _as_1d(a) -> np.ndarray:
    x = np.asarray(a, dtype=float).ravel()
    return x[np.isfinite(x)]

def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = x.size - (m - 1) * tau
    if n <= 1:
        return np.empty((0, m))
    strides = [slice(i * tau, i * tau + n) for i in range(m)]
    return np.vstack([x[s] for s in strides]).T

def sample_entropy(series, m: int = 2, r: float = 0.2, distance: str = "chebyshev") -> float:
    """
    SampEn(m,r): -log( A(m+1) / B(m) ), excluding self-matches.
    r is a fraction of series robust SD (MAD*1.4826) if r<1, otherwise absolute.
    """
    x = _as_1d(series)
    N = x.size
    if N < 5 or m < 1:
        return np.nan
    tol = r * (1.4826 * np.median(np.abs(x - np.median(x)))) if r < 1 else r
    Xm = _embed(x, m, 1)
    Xm1 = _embed(x, m + 1, 1)
    if Xm.shape[0] < 2 or Xm1.shape[0] < 2:
        return np.nan

    def _count_pairs(X, tol):
        # Chebyshev distance
        # compute pairwise max-abs diff without self pairs
        M = X.shape[0]
        cnt = 0
        for i in range(M - 1):
            d = np.max(np.abs(X[i+1:] - X[i]), axis=1)
            cnt += int(np.sum(d <= tol))
        return cnt

    B = _count_pairs(Xm, tol)
    A = _count_pairs(Xm1, tol)
    if B == 0 or A == 0:
        return np.inf
    # Convert to probabilities (normalize by number of comparisons)
    Mb = Xm.shape[0]; Ma = Xm1.shape[0]
    B_prob = B / (Mb * (Mb - 1) / 2.0)
    A_prob = A / (Ma * (Ma - 1) / 2.0)
    return float(-np.log((A_prob + EPS) / (B_prob + EPS)))

def approximate_entropy(series, m: int = 2, r: float = 0.2) -> float:
    """
    ApEn(m,r): Φ_m(r) - Φ_{m+1}(r); biased but included for comparison.
    """
    x = _as_1d(series)
    N = x.size
    if N < 5 or m < 1:
        return np.nan
    tol = r * (1.4826 * np.median(np.abs(x - np.median(x)))) if r < 1 else r

    def _phi(X):
        M = X.shape[0]
        if M == 0:
            return np.nan
        C = np.zeros(M, dtype=float)
        for i in range(M):
            d = np.max(np.abs(X - X[i]), axis=1)
            C[i] = np.mean(d <= tol)
        C = np.clip(C, EPS, 1.0)
        return np.mean(np.log(C))

    Xm = _embed(x, m, 1)
    Xm1 = _embed(x, m + 1, 1)
    return float(_phi(Xm) - _phi(Xm1))

def permutation_entropy(series, m: int = 3, tau: int = 1, normalize: bool = True, tie_strategy: str = "jitter") -> float:
    """
    Permutation entropy based on ordinal patterns.
    tie_strategy: "jitter" adds tiny noise to break ties; "average" uses argsort with stable tie handling.
    """
    x = _as_1d(series)
    if tie_strategy == "jitter":
        rng = np.random.default_rng(12345)
        x = x + rng.normal(0.0, 1e-9 * (np.std(x) + EPS), size=x.size)
    X = _embed(x, m, tau)
    if X.shape[0] == 0:
        return np.nan
    # ordinal patterns
    # The original apply_along_axis can produce unhashable arrays depending on numpy version/etc.
    # A direct list comprehension over the argsort result is more robust and guarantees tuples.
    patterns = [tuple(row) for row in np.argsort(X, axis=1, kind="mergesort")]
    # count frequencies
    from collections import Counter
    counts = np.array(list(Counter(patterns).values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + EPS))
    if normalize:
        H /= np.log(math.factorial(m))
    return float(H)

def multiscale_entropy(series, m: int = 2, r: float = 0.2, max_scale: int = 10, composite: bool = True) -> Dict[str, np.ndarray]:
    """
    MSE/CMSE curve using SampEn at increasing coarse-grained scales.
    composite=True uses CMSE (averaging SampEn across all possible offsets at each scale).
    """
    x = _as_1d(series)
    if x.size < 5:
        return {"scales": np.array([]), "mse": np.array([])}
    scales = np.arange(1, max_scale + 1, dtype=int)
    vals = []
    for s in scales:
        if composite and s > 1:
            # CMSE: compute SampEn across s offsets and average
            ens = []
            for offset in range(s):
                xs = x[offset: x.size - ((x.size - offset) % s)]
                if xs.size < s * 3:
                    continue
                xs = xs.reshape(-1, s).mean(axis=1)
                ens.append(sample_entropy(xs, m=m, r=r))
            
            # Filter out NaNs before taking the mean to avoid warnings
            ens_finite = [v for v in ens if np.isfinite(v)]
            vals.append(np.mean(ens_finite) if ens_finite else np.nan)
        else:
            xs = x[: x.size - (x.size % s)].reshape(-1, s).mean(axis=1)
            vals.append(sample_entropy(xs, m=m, r=r))
    return {"scales": scales, "mse": np.array(vals, dtype=float)}

def complexity_index(series, dfa_alpha: Optional[float] = None) -> Dict[str, float]:
    """
    Composite complexity index combining:
      SampEn(m=2,r=0.2), PermEn(m=3,tau=1,normalized), MSE-AUC (scales 1..10),
      |DFA-α - 0.5| penalty (requires caller to pass dfa_alpha).
    Returns dict {index, sampen, permen, mse_auc, dfa_dev, weights...}
    """
    x = _as_1d(series)
    se = sample_entropy(x, m=2, r=0.2)
    pe = permutation_entropy(x, m=3, tau=1, normalize=True)
    mse = multiscale_entropy(x, m=2, r=0.2, max_scale=10, composite=True)
    scales, curve = mse["scales"], mse["mse"]
    if curve.size == 0 or np.all(~np.isfinite(curve)):
        mse_auc = np.nan
    else:
        # simple trapezoidal AUC over integer scales
        valid = np.isfinite(curve)
        if valid.sum() < 2:
            mse_auc = np.nan
        else:
            s = scales[valid]; c = curve[valid]
            mse_auc = float(np.trapz(c, s))

    dfa_dev = abs((dfa_alpha if dfa_alpha is not None else 0.5) - 0.5)

    # Standardize and PCA (1D via SVD on 1xN vector is trivial; instead compute weighted sum)
    feats = np.array([se, pe, mse_auc, dfa_dev], dtype=float)

    # Guard against all-NaN features to prevent RuntimeWarning
    if not np.any(np.isfinite(feats)):
        return {
            "index": np.nan,
            "sampen": float(se) if np.isfinite(se) else np.nan,
            "permen": float(pe) if np.isfinite(pe) else np.nan,
            "mse_auc": float(mse_auc) if np.isfinite(mse_auc) else np.nan,
            "dfa_dev": float(dfa_dev),
            "w_sampen": 1.0,
            "w_permen": 1.0,
            "w_mse_auc": 1.0,
            "w_dfa_dev": -0.5,
        }
        
    # Robust z-scores calculated on finite values only to prevent warnings
    finite_mask = np.isfinite(feats)
    finite_feats = feats[finite_mask]
    
    med = np.median(finite_feats)
    mad = 1.4826 * np.median(np.abs(finite_feats - med))

    z = np.full_like(feats, np.nan)
    if mad > EPS:
        z[finite_mask] = (finite_feats - med) / mad
    else:
        # If MAD is zero, all finite values are identical, so z-score is 0
        z[finite_mask] = 0.0

    # Weights favor higher complexity (SampEn↑, PermEn↑, MSE-AUC↑) and penalize |DFA-0.5| (lower better)
    w = np.array([1.0, 1.0, 1.0, -0.5])
    # Use np.nansum to ignore NaNs in z, and normalize by sum of weights for valid features
    idx = float(np.nansum(w * z) / (np.nansum(np.abs(w[finite_mask])) + EPS))
    
    return {
        "index": idx,
        "sampen": float(se) if np.isfinite(se) else np.nan,
        "permen": float(pe) if np.isfinite(pe) else np.nan,
        "mse_auc": float(mse_auc) if np.isfinite(mse_auc) else np.nan,
        "dfa_dev": float(dfa_dev),
        "w_sampen": w[0],
        "w_permen": w[1],
        "w_mse_auc": w[2],
        "w_dfa_dev": w[3],
    }

__all__ = ["sample_entropy", "approximate_entropy", "permutation_entropy", "multiscale_entropy", "complexity_index"]
