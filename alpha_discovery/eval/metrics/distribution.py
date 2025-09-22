# alpha_discovery/eval/metrics/distribution.py
"""
Distribution-based metrics for forecast evaluation (no placeholders).
Implements:
  - CRPS for banded piecewise-uniform forecasts
  - Pinball (quantile) loss via CDF inversion inside bands
  - Calibration MAE between forecasted band masses and empirical histogram
  - 1-Wasserstein distance between forecast and empirical distributions
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

EPS = 1e-12

def _as_1d(a) -> np.ndarray:
    x = np.asarray(a, dtype=float).ravel()
    return x

def _normalize_probs(p: np.ndarray) -> np.ndarray:
    s = np.nansum(p)
    if s <= 0 or not np.isfinite(s):
        return np.full_like(p, np.nan)
    return np.clip(p / s, 0.0, 1.0)

def _validate_bands(forecast_probs: np.ndarray, band_edges: np.ndarray) -> None:
    if band_edges.ndim != 1 or np.any(~np.isfinite(band_edges)):
        raise ValueError("band_edges must be 1D finite.")
    if np.any(np.diff(band_edges) <= 0):
        raise ValueError("band_edges must be strictly increasing.")
    if forecast_probs.shape[0] != band_edges.shape[0] - 1:
        raise ValueError("len(forecast_probs) must equal len(band_edges)-1.")

def _cdf_params(forecast_probs: np.ndarray, band_edges: np.ndarray):
    """Return per-bin cumulative masses and slopes to express F(z)=alpha+beta*z on each bin."""
    L = np.diff(band_edges)
    p = forecast_probs
    cumsum_before = np.concatenate(([0.0], np.cumsum(p)[:-1]))
    beta = np.where(L>0, p / L, 0.0)
    alpha = cumsum_before - beta * band_edges[:-1]
    return alpha, beta

def _integrate_quad(alpha: float, beta: float, z0: float, z1: float, target: float) -> float:
    """∫_{z0}^{z1} (alpha + beta*z - target)^2 dz."""
    a = alpha - target
    b = beta
    return (a*a)*(z1 - z0) + (a*b)*(z1**2 - z0**2) + (b*b/3.0)*(z1**3 - z0**3)

def crps(forecast_probs: np.ndarray,
         observed_values: np.ndarray,
         band_edges: np.ndarray) -> float:
    """Continuous Ranked Probability Score averaged over observations.

    We model the forecast distribution as piecewise-uniform over each band.
    For a single observation y, CRPS(F,y)=∫ (F(z)-1{z≥y})^2 dz, evaluated exactly
    by summing polynomial integrals on each band and handling tails analytically.
    """
    fp = _as_1d(forecast_probs)
    ys = _as_1d(observed_values)
    be = _as_1d(band_edges)
    if fp.size == 0 or ys.size == 0:
        return np.nan
    _validate_bands(fp, be)
    fp = _normalize_probs(fp)
    if np.any(~np.isfinite(fp)):
        return np.nan

    alpha, beta = _cdf_params(fp, be)
    a0, bN = be[0], be[-1]
    total = 0.0
    count = 0

    for y in ys:
        if not np.isfinite(y):
            continue

        # Tails where F is 0 or 1
        tail = 0.0
        if y < a0:
            tail += (a0 - y)  # z in [y, a0): (0-1)^2 = 1
        if y > bN:
            tail += (y - bN)  # z in [bN, y): (1-0)^2 = 1

        # Integrate within bands
        acc = tail
        for i in range(len(fp)):
            z_left = be[i]
            z_right = be[i+1]
            al = alpha[i]; bt = beta[i]

            # Left part: z in [z_left, min(z_right, y)) with target=0
            zl = z_left
            zr = min(z_right, y)
            if zr > zl:
                acc += _integrate_quad(al, bt, zl, zr, target=0.0)

            # Right part: z in [max(z_left, y), z_right) with target=1
            zl = max(z_left, y)
            zr = z_right
            if zr > zl:
                acc += _integrate_quad(al, bt, zl, zr, target=1.0)

        total += acc
        count += 1

    return total / count if count > 0 else np.nan

def _quantile_from_bands(forecast_probs: np.ndarray,
                         band_edges: np.ndarray,
                         q: float) -> float:
    """Invert the piecewise-linear CDF at probability q∈[0,1]."""
    q = float(np.clip(q, 0.0, 1.0))
    p = _normalize_probs(_as_1d(forecast_probs))
    be = _as_1d(band_edges)
    _validate_bands(p, be)
    cum = np.concatenate(([0.0], np.cumsum(p)))
    # Find bin
    idx = np.searchsorted(cum, q, side='right') - 1
    idx = int(np.clip(idx, 0, p.size - 1))
    a = be[idx]; b = be[idx+1]; L = b - a
    if p[idx] <= EPS or L <= 0:
        # Degenerate: fall back to bin midpoint
        return 0.5*(a+b)
    # Linear interpolation within bin
    local = (q - cum[idx]) / (p[idx] + EPS)
    return a + L * np.clip(local, 0.0, 1.0)

def pinball_loss(forecast_probs: np.ndarray,
                 observed_values: np.ndarray,
                 band_edges: np.ndarray,
                 quantile: float = 0.5) -> float:
    """Average quantile (pinball) loss using proper CDF inversion within bands."""
    fp = _as_1d(forecast_probs)
    ys = _as_1d(observed_values)
    be = _as_1d(band_edges)
    if fp.size == 0 or ys.size == 0:
        return np.nan
    qz = _quantile_from_bands(fp, be, quantile)
    losses = []
    for y in ys:
        if not np.isfinite(y): 
            continue
        if y >= qz:
            losses.append((1.0 - quantile) * (y - qz))
        else:
            losses.append(quantile * (qz - y))
    return np.mean(losses) if losses else np.nan

def calibration_mae(forecast_probs: np.ndarray,
                    observed_values: np.ndarray,
                    band_edges: np.ndarray) -> float:
    """Expected calibration error (MAE) between forecasted and empirical band masses."""
    fp = _normalize_probs(_as_1d(forecast_probs))
    ys = _as_1d(observed_values)
    be = _as_1d(band_edges)
    if fp.size == 0 or ys.size == 0:
        return np.nan
    _validate_bands(fp, be)
    # Empirical histogram over same bands
    counts, _ = np.histogram(ys[np.isfinite(ys)], bins=be)
    emp = counts.astype(float)
    if emp.sum() > 0:
        emp /= emp.sum()
    else:
        return np.nan
    # Align lengths (defensive)
    m = min(fp.size, emp.size)
    return float(np.mean(np.abs(fp[:m] - emp[:m])))

def wasserstein1(forecast_probs: np.ndarray,
                 observed_values: np.ndarray,
                 band_edges: np.ndarray) -> float:
    """W1 ≡ ∫ |F(z) - G(z)| dz, approximated on the band partition."""
    fp = _normalize_probs(_as_1d(forecast_probs))
    ys = _as_1d(observed_values)
    be = _as_1d(band_edges)
    if fp.size == 0 or ys.size == 0:
        return np.nan
    _validate_bands(fp, be)

    # Empirical mass per band
    counts, _ = np.histogram(ys[np.isfinite(ys)], bins=be)
    emp = counts.astype(float)
    if emp.sum() == 0:
        return np.nan
    emp /= emp.sum()

    # CDFs at band right-edges; integrate per-bin using width
    F = np.cumsum(fp)
    G = np.cumsum(emp)
    widths = np.diff(be)
    m = min(F.size, G.size, widths.size)
    return float(np.sum(np.abs(F[:m] - G[:m]) * widths[:m]))

__all__ = ["crps", "pinball_loss", "calibration_mae", "wasserstein1"]
