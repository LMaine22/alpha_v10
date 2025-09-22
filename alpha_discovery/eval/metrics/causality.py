# alpha_discovery/eval/metrics/causality.py
"""
Causality metrics (no placeholders).
Implements:
  - Granger causality with lag selection (AIC/BIC) and F-test
  - Convergent Cross Mapping (CCM) with library-size convergence
  - Transfer Entropy causality with block-permutation significance
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from scipy.stats import f as f_dist, pearsonr
except Exception:  # scipy optional
    f_dist = None
    pearsonr = None

EPS = 1e-12

def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()

def _lagmat(y: np.ndarray, p: int) -> np.ndarray:
    if p <= 0:
        return np.empty((y.size, 0))
    n = y.size - p
    if n <= 0:
        return np.empty((0, p))
    return np.column_stack([y[p - i - 1 : y.size - i - 1] for i in range(p)])

def _ols(y: np.ndarray, X: np.ndarray) -> Tuple[float, float, int]:
    # returns RSS, sigma2, dof
    if X.size == 0:
        X = np.ones((y.size, 1), dtype=float)
    else:
        X = np.column_stack([np.ones(X.shape[0]), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    dof = y.size - X.shape[1]
    sigma2 = rss / max(dof, 1)
    return rss, sigma2, dof

def _ic(y: np.ndarray, X: np.ndarray, crit: str = "bic") -> float:
    rss, _, dof = _ols(y, X)
    n = y.size
    k = X.shape[1] + 1  # intercept already added in _ols, but for IC count parameters carefully
    if crit.lower() == "aic":
        return float(n * np.log(rss / n + EPS) + 2 * k)
    return float(n * np.log(rss / n + EPS) + k * np.log(n + EPS))  # BIC

def granger_causality(y_values, x_values, max_lag: int = 8, criterion: str = "bic") -> Dict[str, float]:
    """
    Tests X -> Y. Returns best_lag, F, df_num, df_den, p_value (if scipy available).
    """
    y = _as_1d(y_values)
    x = _as_1d(x_values)
    T = min(y.size, x.size)
    if T < max_lag + 5:
        return {"best_lag": np.nan, "F": np.nan, "df_num": 0, "df_den": 0, "p_value": np.nan}

    best_ic = np.inf
    best_p = 1
    for p in range(1, max_lag + 1):
        y_t = y[p:]
        Xr = _lagmat(y, p)  # restricted model: Y lags only
        Xu = np.column_stack([_lagmat(y, p), _lagmat(x, p)])
        n = y_t.size
        Xr = Xr[:n]; Xu = Xu[:n]
        ic = _ic(y_t, Xu, criterion)
        if ic < best_ic:
            best_ic = ic
            best_p = p

    # Build models at best_p
    p = best_p
    y_t = y[p:]
    Xr = _lagmat(y, p)
    Xu = np.column_stack([_lagmat(y, p), _lagmat(x, p)])
    n = y_t.size
    Xr = Xr[:n]; Xu = Xu[:n]
    rss_r, _, dof_r = _ols(y_t, Xr)
    rss_u, _, dof_u = _ols(y_t, Xu)
    df_num = p  # # of X lag params added
    df_den = max(n - (2 * p + 1), 1)  # intercept + 2p regressors
    F = ((rss_r - rss_u) / max(df_num, 1)) / (rss_u / max(df_den, 1))
    if f_dist is not None:
        pval = float(1.0 - f_dist.cdf(F, df_num, df_den))
    else:
        pval = np.nan
    return {"best_lag": float(p), "F": float(F), "df_num": int(df_num), "df_den": int(df_den), "p_value": pval}

def _knn_predict(target: np.ndarray, lib: np.ndarray, lib_resp: np.ndarray, k: int) -> np.ndarray:
    # predict target from lib using kNN weights ~ 1/d
    d = np.linalg.norm(lib[:, None, :] - target[None, :, :], axis=2)
    # avoid zero
    d = np.where(d <= 1e-12, 1e-12, d)
    idx = np.argpartition(d, kth=np.minimum(k, d.shape[0]-1), axis=0)[:k]
    sel_d = np.take_along_axis(d, idx, axis=0)
    w = 1.0 / (sel_d + EPS)
    w /= np.sum(w, axis=0, keepdims=True)
    sel_y = lib_resp[idx, np.arange(target.shape[0])]
    return np.sum(w * sel_y, axis=0)

def ccm(x_values, y_values, E: int = 3, tau: int = 1,
        lib_sizes: Optional[List[int]] = None, k: Optional[int] = None,
        n_shuffle: int = 0, random_state: int = 123) -> Dict[str, np.ndarray]:
    """
    Convergent Cross Mapping (Sugihara et al.). Measures X->Y skill and Y->X skill curves.
    Returns dict with library sizes and Pearson r curves (if scipy available, else fallback to corrcoef).
    """
    rng = np.random.default_rng(random_state)
    x = _as_1d(x_values); y = _as_1d(y_values)
    # Build embeddings
    def embed(v):
        return _lagmat(v, E * tau)[:, ::tau]  # shape (N - E*tau, E)
    X = embed(x); Y = embed(y)
    N = min(X.shape[0], Y.shape[0])
    if N <= 5:
        return {"L": np.array([]), "skill_x_to_y": np.array([]), "skill_y_to_x": np.array([])}
    X = X[-N:]; Y = Y[-N:]
    x_resp = x[-N + E * tau:]  # align to embedding end
    y_resp = y[-N + E * tau:]

    if k is None:
        k = min(E + 1, N - 1)
    if lib_sizes is None:
        lib_sizes = np.unique(np.linspace(max(20, k + 2), N - 1, 10).astype(int)).tolist()

    skills_xy = []
    skills_yx = []
    for L in lib_sizes:
        # choose library indices
        lib_idx = np.arange(L)
        tgt_idx = np.arange(L, N)
        # X->Y: use manifold of Y to infer X
        y_lib = Y[lib_idx]; y_tgt = Y[tgt_idx]
        x_lib_resp = x_resp[lib_idx]; x_tgt_true = x_resp[tgt_idx]
        x_pred = _knn_predict(y_tgt, y_lib, x_lib_resp, k=k)
        # Y->X: symmetric
        x_lib = X[lib_idx]; x_tgt = X[tgt_idx]
        y_lib_resp = y_resp[lib_idx]; y_tgt_true = y_resp[tgt_idx]
        y_pred = _knn_predict(x_tgt, x_lib, y_lib_resp, k=k)

        # skill
        def _r(a, b):
            if pearsonr is not None:
                return pearsonr(a, b)[0]
            a = a - a.mean(); b = b - b.mean()
            denom = (np.std(a) + EPS) * (np.std(b) + EPS)
            return float(np.dot(a, b) / denom)
        skills_xy.append(_r(x_pred, x_tgt_true))
        skills_yx.append(_r(y_pred, y_tgt_true))

    out = {"L": np.array(lib_sizes, dtype=int),
           "skill_x_to_y": np.array(skills_xy, dtype=float),
           "skill_y_to_x": np.array(skills_yx, dtype=float)}

    # Optional shuffle significance (time-shuffled Y to break causality)
    if n_shuffle > 0:
        sh = []
        for _ in range(n_shuffle):
            sh_idx = np.arange(N)
            rng.shuffle(sh_idx)
            Y_sh = Y[sh_idx]
            y_lib = Y_sh[:lib_sizes[-1]]
            x_pred = _knn_predict(Y[lib_sizes[-1]:], y_lib, x_resp[:lib_sizes[-1]], k=k)
            sh.append(np.corrcoef(x_pred, x_resp[lib_sizes[-1]:])[0,1])
        out["shuffle_null_mean"] = float(np.nanmean(sh))
        out["shuffle_null_std"] = float(np.nanstd(sh))
    return out

def transfer_entropy_causality(x_values, y_values, lag: int = 1, bins: int = 10,
                               n_perm: int = 0, block: int = 10, random_state: int = 123) -> Dict[str, float]:
    """
    TE_{X→Y} with optional block-permutation significance.
    """
    from .info_theory import transfer_entropy
    rng = np.random.default_rng(random_state)
    x = _as_1d(x_values); y = _as_1d(y_values)
    te = float(transfer_entropy(x, y, lag=lag, bins=bins))
    out = {"te_x_to_y": te, "p_value": np.nan}
    if n_perm > 0:
        N = x.size
        idx = np.arange(N)
        vals = []
        for _ in range(n_perm):
            # block-permute x
            blocks = [idx[i:i+block] for i in range(0, N, block)]
            rng.shuffle(blocks)
            pb = np.concatenate(blocks)[:N]
            vals.append(transfer_entropy(x[pb], y, lag=lag, bins=bins))
        vals = np.array(vals)
        p = (np.sum(vals >= te) + 1) / (n_perm + 1)
        out["p_value"] = float(p)
        out["null_mean"] = float(vals.mean())
        out["null_std"] = float(vals.std())
    return out

__all__ = ["granger_causality", "ccm", "transfer_entropy_causality"]
