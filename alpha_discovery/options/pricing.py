# alpha_discovery/options/pricing.py
"""
Facade module for options pricing utilities.

Public API is unchanged for callers that do:
    from ..options import pricing
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd

from ..config import settings
from .models import (
    EPS_SIGMA, EPS_T, CLAMP_SIGMA_MIN, CLAMP_SIGMA_MAX, TRADING_DAYS_PER_YEAR,
    bs_price_call, bs_price_put
)
from .market import (
    get_underlying_price, get_price_on_exit,
    get_entry_iv_3m, get_exit_iv_3m, map_sigma_from_3m_to_T, has_required_iv_series,
    get_risk_free_rate
)

# ---------------- Entry/Exit pricing wrapper (same signature/logic) ----------------

@dataclass
class PricedLeg:
    entry_price: float
    exit_price: float
    T_exit: float
    entry_iv: float
    exit_iv: float
    option_type: Literal["call", "put"]

def price_entry_exit(
    S0: float,
    S1: float,
    K: float,
    T0: float,
    h_days: int,
    r: float,
    direction: Literal["long", "short"],
    entry_sigma: Optional[float],
    exit_sigma: Optional[float],
    q: float = 0.0
) -> Optional[PricedLeg]:
    if any(x is None for x in [S0, S1, K, T0, entry_sigma, exit_sigma]):
        return None

    T0 = max(float(T0), EPS_T)
    T1 = max(T0 - float(h_days) / TRADING_DAYS_PER_YEAR, EPS_T)

    entry_sigma = float(min(max(float(entry_sigma), CLAMP_SIGMA_MIN), CLAMP_SIGMA_MAX))
    exit_sigma = float(min(max(float(exit_sigma), CLAMP_SIGMA_MIN), CLAMP_SIGMA_MAX))

    option_type: Literal["call", "put"] = "call" if direction == "long" else "put"

    if option_type == "call":
        p0 = bs_price_call(S0, K, T0, r, entry_sigma, q=0.0)
        p1 = bs_price_call(S1, K, T1, r, exit_sigma, q=0.0)
    else:
        p0 = bs_price_put(S0, K, T0, r, entry_sigma, q=0.0)
        p1 = bs_price_put(S1, K, T1, r, exit_sigma, q=0.0)

    p0 = max(0.0, float(p0))
    p1 = max(0.0, float(p1))

    return PricedLeg(
        entry_price=p0,
        exit_price=p1,
        T_exit=T1,
        entry_iv=float(entry_sigma),
        exit_iv=float(exit_sigma),
        option_type=option_type,
    )

# ---------------- Backtester compatibility aliases (unchanged) ----------------

def get_iv_3m_for_entry(ticker: str, date: pd.Timestamp, direction: Literal["long", "short"], df: pd.DataFrame) -> Optional[float]:
    return get_entry_iv_3m(ticker, date, direction, df)

def get_underlier_on_entry(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, date, df)

def get_underlying_eod(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, date, df)

def get_price_on_entry(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, date, df)

def build_option_strike(S0: float, direction: Literal["long", "short"]) -> float:
    return float(S0)

__all__ = [
    # constants (if anyone imports through the facade)
    "EPS_SIGMA", "EPS_T", "CLAMP_SIGMA_MIN", "CLAMP_SIGMA_MAX", "TRADING_DAYS_PER_YEAR",
    # primitives
    "bs_price_call", "bs_price_put",
    # IV + tenor mapping
    "get_entry_iv_3m", "get_exit_iv_3m", "map_sigma_from_3m_to_T", "has_required_iv_series",
    # rates
    "get_risk_free_rate",
    # underlying access
    "get_underlying_price", "get_price_on_exit",
    # wrapper + dataclass
    "PricedLeg", "price_entry_exit",
    # backtester aliases (unchanged)
    "get_iv_3m_for_entry", "get_underlier_on_entry", "get_underlying_eod", "get_price_on_entry", "build_option_strike",
]
