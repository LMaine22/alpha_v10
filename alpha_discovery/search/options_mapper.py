from __future__ import annotations
import numpy as np
from typing import Sequence, Optional


def _sum_between(lo_edges: np.ndarray, hi_edges: np.ndarray, probs: np.ndarray, a: float, b: float) -> float:
    mask = (np.isclose(lo_edges, a) & np.isclose(hi_edges, b))
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_ge(lo_edges: np.ndarray, probs: np.ndarray, a: float) -> float:
    mask = (lo_edges >= a)
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_le(hi_edges: np.ndarray, probs: np.ndarray, b: float) -> float:
    mask = (hi_edges <= b)
    return float(probs[mask].sum()) if mask.any() else 0.0


def suggest_option_structure(
    band_edges: Sequence[float] | np.ndarray,
    band_probs: Sequence[float] | np.ndarray,
) -> str:
    """
    Downstream mapper (no coupling to options backtester). Supports bull/flat/bear.
    Loosened thresholds + rule reordering to prefer skew monetization when right-tail mass is extreme.
    """
    edges = np.asarray(band_edges, dtype=float)
    p = np.asarray(band_probs, dtype=float)
    if edges.size < 2 or p.size != edges.size - 1:
        return "no-structure / wait (insufficient edge)"

    lo = edges[:-1]
    hi = edges[1:]

    flat     = _sum_between(lo, hi, p, -0.01, 0.01)
    up_mid   = _sum_between(lo, hi, p, 0.03, 0.05)
    up_big   = _sum_ge(lo, p, 0.05)
    down_mid = _sum_between(lo, hi, p, -0.05, -0.03)
    down_big = _sum_le(hi, p, -0.05)  # <= -5%

    # --- Reordered/loosened gates ---
    # Prefer call ratio first when right-tail mass is very large
    if up_big >= 0.50 and down_big < 0.15:
        return "call_ratio or OTM calls (skew monetization)"

    # Bullish spread when mid+big band mass is solid
    if (up_mid + up_big) > 0.45 and down_big < 0.12:
        return "debit_call_spread (target 3–5% band, width ≈ 2–3%)"

    # Bearish structures
    if (down_mid + down_big) > 0.40 and up_big < 0.12:
        return "debit_put_spread (express −3% to −5%)"
    if down_big > 0.18 and up_big < 0.12:
        return "OTM puts or put ratio (tail hedge)"

    # Neutral / mean-revert
    if flat > 0.40 and up_big < 0.15 and down_big < 0.15:
        return "calendar/iron_condor (mean-revert/low vol)"

    return "no-structure / wait (insufficient edge)"
