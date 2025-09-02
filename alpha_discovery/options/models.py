# alpha_discovery/options/models.py
from __future__ import annotations
import math

# ---- Shared numeric constants (unchanged semantics) ----
EPS_SIGMA = 1e-9          # floor for volatility
EPS_T = 1e-6              # floor for time to expiry (years)
CLAMP_SIGMA_MIN = 0.0002  # 2% annualized lower clamp
CLAMP_SIGMA_MAX = 5.0     # 500% annualized upper clamp
TRADING_DAYS_PER_YEAR = 252.0
REF_3M_BD = 63            # business days representing 3M

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
