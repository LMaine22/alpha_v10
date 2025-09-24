from __future__ import annotations
import numpy as np
from typing import Sequence, Optional, Dict, Tuple


def _sum_between(lo_edges: np.ndarray, hi_edges: np.ndarray, probs: np.ndarray, a: float, b: float) -> float:
    """Sum probabilities where lo_edge >= a and hi_edge <= b"""
    mask = (lo_edges >= a) & (hi_edges <= b)
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_ge(lo_edges: np.ndarray, probs: np.ndarray, a: float) -> float:
    """Sum probabilities where lo_edge >= a"""
    mask = (lo_edges >= a)
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_le(hi_edges: np.ndarray, probs: np.ndarray, b: float) -> float:
    """Sum probabilities where hi_edge <= b"""
    mask = (hi_edges <= b)
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_z_between(z_lo_edges: np.ndarray, z_hi_edges: np.ndarray, probs: np.ndarray, a: float, b: float) -> float:
    """Sum probabilities where z_lo_edge >= a and z_hi_edge <= b (z-score version)"""
    mask = (z_lo_edges >= a) & (z_hi_edges <= b)
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_z_ge(z_lo_edges: np.ndarray, probs: np.ndarray, a: float) -> float:
    """Sum probabilities where z_lo_edge >= a (z-score version)"""
    mask = (z_lo_edges >= a)
    return float(probs[mask].sum()) if mask.any() else 0.0


def _sum_z_le(z_hi_edges: np.ndarray, probs: np.ndarray, b: float) -> float:
    """Sum probabilities where z_hi_edge <= b (z-score version)"""
    mask = (z_hi_edges <= b)
    return float(probs[mask].sum()) if mask.any() else 0.0


def suggest_option_structure(
    band_edges: Sequence[float] | np.ndarray,
    band_probs: Sequence[float] | np.ndarray,
    historical_std: Optional[float] = None,
) -> str:
    """
    Downstream mapper that suggests option structures based on probability distribution.
    Uses z-scores to adapt to different volatility profiles across tickers.
    Implements a scoring system with confidence levels and lower rejection thresholds.
    
    Args:
        band_edges: Sequence of return band edges
        band_probs: Probabilities for each band (must sum to 1)
        historical_std: Historical standard deviation of returns (if None, will estimate from bands)
    
    Returns:
        String with option structure suggestion and confidence level
    """
    edges = np.asarray(band_edges, dtype=float)
    p = np.asarray(band_probs, dtype=float)
    
    if edges.size < 2 or p.size != edges.size - 1:
        return "no-structure / wait (invalid edge format)"
    
    # Check for NaN values that would cause calculation failures
    if np.isnan(edges).any() or np.isnan(p).any():
        return "no-structure / wait (contains NaN values)"
    
    # Set default volatility - always use a reasonable value
    if historical_std is None:
        historical_std = 0.02  # Default 2% daily std
    
    # Safety check - ensure volatility is sensible (between 0.5% and 10%)
    historical_std = min(max(historical_std, 0.005), 0.10)

    lo = edges[:-1]
    hi = edges[1:]
    
    # Convert edges to z-scores for probability calculations
    # We use normalized values but keep original edges for strategy descriptions
    z_edges = edges / historical_std
    z_lo = z_edges[:-1]
    z_hi = z_edges[1:]
    
    # Calculate probability masses based on z-scores
    flat     = _sum_z_between(z_lo, z_hi, p, -0.5, 0.5)  # Within 0.5 std (central tendency)
    up_small = _sum_z_between(z_lo, z_hi, p, 0.5, 1.0)   # 0.5 to 1.0 std up
    up_mid   = _sum_z_between(z_lo, z_hi, p, 1.0, 1.5)   # 1.0 to 1.5 std up
    up_big   = _sum_z_ge(z_lo, p, 1.5)                   # > 1.5 std up
    down_small = _sum_z_between(z_lo, z_hi, p, -1.0, -0.5)  # -1.0 to -0.5 std down
    down_mid = _sum_z_between(z_lo, z_hi, p, -1.5, -1.0) # -1.5 to -1.0 std down
    down_big = _sum_z_le(z_hi, p, -1.5)                  # < -1.5 std down
    
    # Calculate expected return and basic distribution properties
    mid_points = (lo + hi) / 2
    expected_return = sum(mid_points * p)
    
    # Calculate strategy scores with more nuanced weights
    scores: Dict[str, float] = {
        # Bullish strategies
        "call_ratio": (up_big * 2.0) + (up_mid * 0.8) - (down_big * 1.2),
        "debit_call_spread": (up_mid * 1.5) + (up_small * 0.8) + (up_big * 0.3) - (down_big * 1.0),
        "long_calls": (up_big * 1.0) + (up_mid * 1.2) + (up_small * 0.5) - (down_big * 0.8),
        
        # Bearish strategies
        "put_ratio": (down_big * 2.0) + (down_mid * 0.8) - (up_big * 1.2),
        "debit_put_spread": (down_mid * 1.5) + (down_small * 0.8) + (down_big * 0.3) - (up_big * 1.0),
        "long_puts": (down_big * 1.0) + (down_mid * 1.2) + (down_small * 0.5) - (up_big * 0.8),
        
        # Neutral strategies
        "iron_condor": (flat * 1.8) - ((up_big + down_big) * 1.0),
        "calendar": (flat * 1.5) - ((up_big + down_big) * 0.8),
        
        # Wait (null strategy)
        "wait": 0.0  # Baseline score
    }
    
    # Add expected return bias
    if expected_return > 0:
        # Positive expected return: boost bullish strategies
        scores["call_ratio"] += expected_return * 3.0
        scores["debit_call_spread"] += expected_return * 4.0
        scores["long_calls"] += expected_return * 3.5
    elif expected_return < 0:
        # Negative expected return: boost bearish strategies
        scores["put_ratio"] += -expected_return * 3.0
        scores["debit_put_spread"] += -expected_return * 4.0
        scores["long_puts"] += -expected_return * 3.5
    
    # Find best strategy
    best_strategy, best_score = max(scores.items(), key=lambda x: x[1])
    
    # Set threshold for "wait" recommendation - now even lower
    min_threshold = 0.08  # Further lowered from 0.10
    
    if best_score < min_threshold:
        return "no-structure / wait (insufficient edge)"
    
    # Map strategies to human-readable descriptions with confidence levels
    confidence = "high" if best_score > 0.22 else "moderate"
    if best_score < 0.12:
        confidence = "lower"  # Lowered threshold for 'lower' tier
    
    # Calculate target move ranges for option strategies
    # We'll use standard deviation multiples for clear ranges
    small_move = historical_std * 1.0  # 1 std move
    mid_move = historical_std * 1.5    # 1.5 std move
    big_move = historical_std * 2.0    # 2 std move
    
    # Format as percentages with 1 decimal place
    small_move_pct = f"{small_move:.1%}"
    mid_move_pct = f"{mid_move:.1%}"
    big_move_pct = f"{big_move:.1%}"
        
    strategy_descriptions = {
        "call_ratio": f"call_ratio or OTM calls ({big_move_pct}+ upside play, {confidence} confidence)",
        "debit_call_spread": f"debit_call_spread (target {small_move_pct}–{mid_move_pct} upside, {confidence} confidence)",
        "long_calls": f"long_calls (bullish directional, {confidence} confidence)",
        "put_ratio": f"put_ratio or OTM puts ({big_move_pct}+ downside hedge, {confidence} confidence)",
        "debit_put_spread": f"debit_put_spread (target {small_move_pct}–{mid_move_pct} downside, {confidence} confidence)",
        "long_puts": f"long_puts (bearish directional, {confidence} confidence)",
        "iron_condor": f"iron_condor (range-bound within ±{small_move_pct}, {confidence} confidence)",
        "calendar": f"calendar spread (low volatility play, {confidence} confidence)",
        "wait": f"no-structure / wait (insufficient edge)"
    }
    
    return strategy_descriptions[best_strategy]
