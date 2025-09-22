# alpha_discovery/eval/metrics/regime.py
"""
Regime detection & regime-aware metrics (no placeholders).
Implements:
  - Gaussian HMM (multivariate) via EM; Viterbi decoding
  - Regime-specific metric wrapper
  - Worst-regime performance extractor
"""
from __future__ import annotations
from typing import Dict, Tuple, Callable
import numpy as np
import pandas as pd

from .distribution import calibration_mae, crps, wasserstein1
from .info_theory import info_gain

EPS = 1e-12

def _gaussian_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    cov = np.atleast_2d(cov)
    # regularize
    cov = cov + np.eye(d) * 1e-8
    L = np.linalg.cholesky(cov)
    inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(d)))
    xc = X - mean
    quad = np.einsum("...i,ij,...j->...", xc, inv, xc)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

def _forward_backward(log_pi, log_A, log_pdf):
    T, K = log_pdf.shape
    # Forward
    alpha = np.zeros((T, K))
    alpha[0] = log_pi + log_pdf[0]
    for t in range(1, T):
        alpha[t] = log_pdf[t] + np.logaddexp.reduce(alpha[t-1][:, None] + log_A, axis=0)
    ll = np.logaddexp.reduce(alpha[-1])
    # Backward
    beta = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        beta[t] = np.logaddexp.reduce(log_A + log_pdf[t+1] + beta[t+1], axis=1)
    # Posteriors
    gamma = alpha + beta - ll
    gamma = np.exp(gamma)
    # Xi
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        m = (alpha[t][:, None] + log_A + log_pdf[t+1][None, :] + beta[t+1][None, :])
        m = m - np.logaddexp.reduce(m.ravel())
        xi[t] = np.exp(m)
    return ll, gamma, xi

def _kmeans_init(X: np.ndarray, K: int, random_state: int = 123) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    # k-means++ like init
    N = X.shape[0]
    means = np.empty((K, X.shape[1]))
    means[0] = X[rng.integers(0, N)]
    D2 = np.sum((X - means[0])**2, axis=1) + 1e-9
    for k in range(1, K):
        probs = D2 / D2.sum()
        idx = rng.choice(N, p=probs)
        means[k] = X[idx]
        D2 = np.minimum(D2, np.sum((X - means[k])**2, axis=1) + 1e-9)
    covs = np.array([np.cov(X.T) + np.eye(X.shape[1])*1e-6 for _ in range(K)])
    return means, covs

def fit_hmm_gaussian(features: np.ndarray, K: int = 3, n_iter: int = 50, tol: float = 1e-4, random_state: int = 123) -> Dict:
    """
    EM for Gaussian HMM with full covariances.
    features: shape (T, D)
    Returns dict with pi, A, means, covs, gamma (state posteriors), states (Viterbi).
    """
    X = np.asarray(features, dtype=float)
    T, D = X.shape
    means, covs = _kmeans_init(X, K, random_state)
    A = np.full((K, K), 1.0 / K)
    pi = np.full(K, 1.0 / K)

    prev_ll = -np.inf
    for _ in range(n_iter):
        log_pdf = np.column_stack([_gaussian_logpdf(X, means[k], covs[k]) for k in range(K)])
        log_A = np.log(A + EPS)
        log_pi = np.log(pi + EPS)
        ll, gamma, xi = _forward_backward(log_pi, log_A, log_pdf)

        # M-step
        Nk = gamma.sum(axis=0) + EPS
        pi = gamma[0] / gamma[0].sum()
        A = xi.sum(axis=0)
        A = A / A.sum(axis=1, keepdims=True)
        means = (gamma.T @ X) / Nk[:, None]
        covs = np.zeros((K, D, D))
        for k in range(K):
            xc = X - means[k]
            covs[k] = (gamma[:, k][:, None] * xc).T @ xc / Nk[k]
            covs[k].flat[::D+1] += 1e-8  # jitter

        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # Viterbi decode
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi + EPS) + np.column_stack([_gaussian_logpdf(X, means[k], covs[k]) for k in range(K)])
    log_A = np.log(A + EPS)
    for t in range(1, T):
        m = delta[t-1][:, None] + log_A
        psi[t] = np.argmax(m, axis=0)
        delta[t] = np.max(m, axis=0) + np.column_stack([_gaussian_logpdf(X, means[k], covs[k]) for k in range(K)])
    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T-2, -1, -1):
        states[t] = int(psi[t+1, states[t+1]])

    return {"pi": pi, "A": A, "means": means, "covs": covs, "gamma": gamma, "states": states, "loglik": prev_ll}

def detect_regimes(returns: np.ndarray, k_states: int = 3, vol_window: int = 20) -> Dict:
    """
    Build 2D features [r_t, rolling_vol_t] and fit HMM.
    """
    r = np.asarray(returns, dtype=float).ravel()
    vol = pd.Series(r).rolling(vol_window, min_periods=max(2, vol_window//2)).std().values
    vol = np.where(np.isfinite(vol), vol, np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826)
    X = np.column_stack([r, vol])
    return fit_hmm_gaussian(X, K=k_states)

def regime_metrics(states: np.ndarray,
                   observed_values: np.ndarray,
                   forecast_probs: np.ndarray,
                   band_edges: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Compute per-regime metrics using distribution/info_theory modules.
    """
    y = np.asarray(observed_values, dtype=float).ravel()
    s = np.asarray(states, dtype=int).ravel()
    out = {}
    for k in np.unique(s):
        mask = s == k
        if mask.sum() < 5:
            out[int(k)] = {"n": int(mask.sum()), "cal_mae": np.nan, "crps": np.nan, "w1": np.nan, "ig": np.nan}
            continue
        yk = y[mask]
        out[int(k)] = {
            "n": int(mask.sum()),
            "cal_mae": float(calibration_mae(forecast_probs, yk, band_edges)),
            "crps": float(crps(forecast_probs, yk, band_edges)),
            "w1": float(wasserstein1(forecast_probs, yk, band_edges)),
            "ig": float(info_gain(forecast_probs, yk, band_edges)),
        }
    return out

def worst_regime(per_regime: Dict[int, Dict[str, float]], key: str = "crps") -> Dict[str, float]:
    """
    Identify worst regime by a chosen metric (higher worse for CRPS/W1/IG, lower worse for negatives).
    """
    # assume higher is worse for all listed metrics
    worst_k = None
    worst_val = -np.inf
    for k, d in per_regime.items():
        v = d.get(key, np.nan)
        if np.isfinite(v) and v > worst_val:
            worst_val = v
            worst_k = k
    if worst_k is None:
        return {"regime": np.nan, "value": np.nan}
    # gap-to-next
    vals = [d.get(key, np.nan) for d in per_regime.values() if np.isfinite(d.get(key, np.nan))]
    vals_sorted = sorted(vals, reverse=True)
    gap = vals_sorted[0] - (vals_sorted[1] if len(vals_sorted) > 1 else vals_sorted[0])
    return {"regime": int(worst_k), "value": float(worst_val), "gap_to_next": float(gap)}

__all__ = ["fit_hmm_gaussian", "detect_regimes", "regime_metrics", "worst_regime"]
