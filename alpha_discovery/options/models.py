# alpha_discovery/options/models.py
from __future__ import annotations
import math
from typing import Optional, Literal

# ---- Shared numeric constants (unchanged semantics) ----
EPS_SIGMA = 1e-9          # floor for volatility
EPS_T = 1e-6              # floor for time to expiry (years)
CLAMP_SIGMA_MIN = 0.0002  # 2% annualized lower clamp
CLAMP_SIGMA_MAX = 5.0     # 500% annualized upper clamp
TRADING_DAYS_PER_YEAR = 252.0
REF_3M_BD = 63            # business days representing 3M
REF_1M_BD = 21            # business days representing 1M/30D

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ---- Black–Scholes primitives (unchanged) ----
def bs_price_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    S = float(S); K = float(K)
    T = max(float(T), EPS_T)
    sigma = max(float(sigma), EPS_SIGMA)
    r = float(r); q = float(q)
    if S <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_price_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    S = float(S); K = float(K)
    T = max(float(T), EPS_T)
    sigma = max(float(sigma), EPS_SIGMA)
    r = float(r); q = float(q)
    if S <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

# ---- Black-Scholes Delta calculations ----
def bs_delta_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate call delta using Black-Scholes formula."""
    S = float(S); K = float(K)
    T = max(float(T), EPS_T)
    sigma = max(float(sigma), EPS_SIGMA)
    r = float(r); q = float(q)
    if S <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return math.exp(-q * T) * _norm_cdf(d1)

def bs_delta_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate put delta using Black-Scholes formula."""
    S = float(S); K = float(K)
    T = max(float(T), EPS_T)
    sigma = max(float(sigma), EPS_SIGMA)
    r = float(r); q = float(q)
    if S <= 0.0 or K <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: Literal["call", "put"], q: float = 0.0) -> float:
    """Calculate delta for call or put option."""
    if option_type == "call":
        return bs_delta_call(S, K, T, r, sigma, q)
    else:
        return bs_delta_put(S, K, T, r, sigma, q)

# ---- Strike solving by target delta ----
def solve_strike_for_delta(
    S: float, 
    target_delta: float, 
    T: float, 
    r: float, 
    sigma: float, 
    option_type: Literal["call", "put"],
    q: float = 0.0,
    tolerance: float = 0.05,  # Increased tolerance for robustness
    max_iterations: int = 100
) -> Optional[float]:
    """
    Solve for strike K such that Black-Scholes delta matches target_delta.
    Uses simple bisection method for robustness.
    
    Returns None if solver fails to converge.
    """
    if S <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return None
    
    # More aggressive bounds for strike search to handle extreme deltas
    K_min = S * 0.2   # Very deep ITM
    K_max = S * 5.0   # Very deep OTM
    
    # Check if target is achievable within bounds
    try:
        delta_min = bs_delta(S, K_max, T, r, sigma, option_type, q)
        delta_max = bs_delta(S, K_min, T, r, sigma, option_type, q)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    
    # For puts, delta_max is the most negative (at deep ITM), delta_min is least negative (at deep OTM)
    if option_type == "call":
        if target_delta < delta_min or target_delta > delta_max:
            # Try to find achievable delta close to target
            if target_delta < delta_min:
                target_delta = delta_min + 0.01
            elif target_delta > delta_max:
                target_delta = delta_max - 0.01
    else:  # put
        if target_delta > delta_min or target_delta < delta_max:
            # Try to find achievable delta close to target  
            if target_delta > delta_min:
                target_delta = delta_min - 0.01
            elif target_delta < delta_max:
                target_delta = delta_max + 0.01
    
    # Bisection search with improved convergence
    for iteration in range(max_iterations):
        K_mid = (K_min + K_max) / 2.0
        
        try:
            delta_mid = bs_delta(S, K_mid, T, r, sigma, option_type, q)
        except (ValueError, ZeroDivisionError, OverflowError):
            # If calculation fails, adjust bounds and continue
            if option_type == "call":
                K_max = K_mid
            else:
                K_min = K_mid
            continue
        
        if abs(delta_mid - target_delta) < tolerance:
            return K_mid
        
        # Improved bisection logic
        if option_type == "call":
            if delta_mid > target_delta:  # Strike too low, need higher strike
                K_min = K_mid
            else:  # Strike too high, need lower strike
                K_max = K_mid
        else:  # put
            if delta_mid < target_delta:  # Delta too negative, need higher strike  
                K_min = K_mid
            else:  # Delta not negative enough, need lower strike
                K_max = K_mid
        
        # Prevent infinite loops with very small intervals
        if abs(K_max - K_min) < S * 1e-6:
            return K_mid
    
    # If we reach here, return the last mid-point as best estimate
    return (K_min + K_max) / 2.0
