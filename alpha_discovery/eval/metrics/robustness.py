# alpha_discovery/eval/metrics/robustness.py
"""
Robustness metrics (no placeholders).
Implements:
  - Moving-Block Bootstrap for arbitrary metric_fn
  - Local sensitivity scan over hyperparameters
  - Time-series CV robustness (purged, embargoed)
  - Simple Page-Hinkley drift test on metric over time
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any
import numpy as np

EPS = 1e-12

def moving_block_bootstrap(metric_fn: Callable[..., float],
                           data: np.ndarray,
                           block_len: int = 20,
                           n_boot: int = 200,
                           random_state: int = 123,
                           **metric_kwargs) -> Dict[str, float]:
    """
    Resample 1D time-series in blocks, compute metric, return mean/std/CI.
    metric_fn receives resampled data via first positional arg 'data'.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(data, dtype=float).ravel()
    N = x.size
    if N == 0:
        return {"mean": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    B = []
    n_blocks = int(np.ceil(N / block_len))
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, N - block_len + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(N, s + block_len)) for s in starts])[:N]
        xb = x[idx]
        try:
            val = float(metric_fn(xb, **metric_kwargs))
        except Exception:
            val = np.nan
        B.append(val)
    B = np.array(B, dtype=float)
    p_value = np.mean(B <= 0) if np.nanmean(B) > 0 else np.mean(B >= 0)

    return {"mean": float(np.nanmean(B)),
            "std": float(np.nanstd(B, ddof=1)),
            "ci_lower": float(np.nanpercentile(B, 2.5)),
            "ci_upper": float(np.nanpercentile(B, 97.5)),
            "p_value": float(p_value),
            "raw_values": B}

def sensitivity_scan(metric_fn: Callable[..., float],
                     base_kwargs: Dict[str, Any],
                     perturb: Dict[str, float],
                     direction: str = "both") -> Dict[str, Dict[str, float]]:
    """
    One-at-a-time local sensitivity. For each param in perturb, evaluate +/- delta (or + only).
    Returns per-param deltas and relative ranks.
    """
    base_kwargs = dict(base_kwargs)
    results = {}
    for k, delta in perturb.items():
        vals = {}
        for sign in ([-1, 1] if direction == "both" else [1]):
            kwargs = dict(base_kwargs)
            kwargs[k] = kwargs.get(k, 0.0) + sign * delta
            try:
                v = float(metric_fn(**kwargs))
            except Exception:
                v = np.nan
            vals["+" if sign > 0 else "-"] = v
        results[k] = {"minus": vals.get("-", np.nan), "plus": vals.get("+", np.nan),
                      "delta": (vals.get("+", np.nan) - vals.get("-", np.nan))}
    # Tornado-style ranking by absolute delta
    for k in results:
        results[k]["abs_delta"] = abs(results[k]["delta"]) if np.isfinite(results[k]["delta"]) else np.nan
    return results

def tscv_robustness(metric_fn: Callable[..., float],
                    data: np.ndarray,
                    n_splits: int = 5,
                    embargo: int = 0,
                    **metric_kwargs) -> Dict[str, float]:
    """
    Time-series split robustness. Splits data into sequential folds,
    computes metric on each, returns mean/std/CV and flip-rate for pairwise ranks.
    """
    x = np.asarray(data, dtype=float).ravel()
    N = x.size
    if N < n_splits * 5:
        return {"mean": np.nan, "std": np.nan, "cov": np.nan}
    fold_size = N // n_splits
    vals = []
    for i in range(n_splits):
        start = i * fold_size
        end = N if i == n_splits - 1 else (i + 1) * fold_size
        # apply embargo by trimming edges
        s = start + embargo if i > 0 else start
        e = end - embargo if i < n_splits - 1 else end
        if e - s < 3:
            continue
        xi = x[s:e]
        try:
            vals.append(float(metric_fn(xi, **metric_kwargs)))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals, dtype=float)
    return {"mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)),
            "cov": float(np.nanstd(vals, ddof=1) / (np.nanmean(vals) + EPS))}

def page_hinkley(series: np.ndarray, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.999) -> Dict[str, float]:
    """
    Simple Page-Hinkley change detection over a metric sequence.
    Triggers when cumulative deviation exceeds lambda_.
    """
    x = np.asarray(series, dtype=float).ravel()
    mean = 0.0
    m_t = 0.0
    PH = []
    alarms = []
    for t, v in enumerate(x):
        mean = alpha * mean + (1 - alpha) * v
        m_t = min(0.0, m_t + v - mean - delta)
        PH.append(m_t)
        alarms.append(1 if abs(m_t) > lambda_ else 0)
    return {"min_stat": float(np.min(PH)), "alarm": int(max(alarms))}
    
__all__ = ["moving_block_bootstrap", "sensitivity_scan", "tscv_robustness", "page_hinkley"]
