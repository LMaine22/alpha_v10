# alpha_discovery/eval/metrics/aggregate.py
"""
Cross-fold aggregation & rank stability (no placeholders).
Implements:
  - Robust aggregators: median-MAD, trimmed mean, Huber M-estimator, Hodges–Lehmann
  - Rank stability: Kendall tau, Spearman rho, flip-rate
  - Jackknife-after-bootstrap style influence (leave-one-out)
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:
    kendalltau = None
    spearmanr = None

EPS = 1e-12

def median_mad(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = 1.4826 * np.nanmedian(np.abs(v - med))
    return {"agg": float(med), "dispersion": float(mad)}

def trimmed_mean(values: np.ndarray, trim: float = 0.1) -> Dict[str, float]:
    v = np.sort(v for v in np.asarray(values, dtype=float) if np.isfinite(v))
    if len(v) == 0:
        return {"agg": np.nan, "dispersion": np.nan}
    k = int(len(v) * trim)
    w = v[k:len(v)-k] if len(v) - 2*k > 0 else v
    return {"agg": float(np.mean(w)), "dispersion": float(np.std(w, ddof=1))}

def huber_mean(values: np.ndarray, c: float = 1.345, tol: float = 1e-6, max_iter: int = 50) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    mu = np.nanmedian(v)
    s = 1.4826 * np.nanmedian(np.abs(v - mu)) + EPS
    for _ in range(max_iter):
        r = (v - mu) / s
        w = np.clip(c / np.maximum(np.abs(r), EPS), 0.0, 1.0)
        mu_new = np.nansum(w * v) / (np.nansum(w) + EPS)
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new
    return {"agg": float(mu), "dispersion": float(np.nanstd(v - mu))}

def hodges_lehmann(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"agg": np.nan, "dispersion": np.nan}
    # all pairwise midpoints
    mids = []
    for i in range(v.size):
        for j in range(i, v.size):
            mids.append(0.5 * (v[i] + v[j]))
    mids = np.array(mids, dtype=float)
    return {"agg": float(np.median(mids)), "dispersion": float(1.4826 * np.median(np.abs(mids - np.median(mids))))}

def aggregate(values: np.ndarray, method: str = "median_mad", **kwargs) -> Dict[str, float]:
    method = method.lower()
    if method == "median_mad":
        return median_mad(values)
    if method == "trimmed_mean":
        return trimmed_mean(values, **kwargs)
    if method == "huber":
        return huber_mean(values, **kwargs)
    if method == "hodges_lehmann":
        return hodges_lehmann(values)
    raise ValueError(f"Unknown method: {method}")

def rank_stability(ranks_matrix: np.ndarray) -> Dict[str, float]:
    """
    ranks_matrix: shape (n_folds, n_items), smaller rank = better.
    Returns mean Kendall tau and Spearman rho across all fold pairs, and flip-rate.
    """
    R = np.asarray(ranks_matrix, dtype=float)
    F, M = R.shape
    taus = []
    rhos = []
    flips = 0
    pairs = 0
    for i in range(F):
        for j in range(i+1, F):
            a, b = R[i], R[j]
            # compute pairwise order flips
            for u in range(M):
                for v in range(u+1, M):
                    pairs += 1
                    if (a[u] - a[v]) * (b[u] - b[v]) < 0:
                        flips += 1
            # correlation
            if kendalltau is not None:
                taus.append(kendalltau(a, b, nan_policy="omit")[0])
            else:
                # fallback approximate kendall
                concord = 0; discord = 0
                for u in range(M):
                    for v in range(u+1, M):
                        s = np.sign(a[u]-a[v]) * np.sign(b[u]-b[v])
                        if s > 0: concord += 1
                        elif s < 0: discord += 1
                denom = concord + discord + EPS
                taus.append((concord - discord) / denom)
            if spearmanr is not None:
                rhos.append(spearmanr(a, b, nan_policy="omit")[0])
            else:
                a_ = (a - np.nanmean(a)) / (np.nanstd(a) + EPS)
                b_ = (b - np.nanmean(b)) / (np.nanstd(b) + EPS)
                rhos.append(float(np.nanmean(a_ * b_)))
    flip_rate = flips / (pairs + EPS)
    return {"kendall_tau_mean": float(np.nanmean(taus)),
            "spearman_rho_mean": float(np.nanmean(rhos)),
            "flip_rate": float(flip_rate)}

def jackknife_leave_one_out(values: np.ndarray, agg_method: str = "median_mad") -> Dict[str, float]:
    """
    Influence via leave-one-out aggregates. Returns mean and std of LOO aggregates.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 3:
        return {"loo_mean": np.nan, "loo_std": np.nan}
    agg_vals = []
    for i in range(n):
        sub = np.delete(v, i)
        agg_vals.append(aggregate(sub, method=agg_method)["agg"])
    agg_vals = np.array(agg_vals, dtype=float)
    return {"loo_mean": float(np.nanmean(agg_vals)), "loo_std": float(np.nanstd(agg_vals, ddof=1))}

__all__ = ["aggregate", "median_mad", "trimmed_mean", "huber_mean", "hodges_lehmann", "rank_stability", "jackknife_leave_one_out"]
