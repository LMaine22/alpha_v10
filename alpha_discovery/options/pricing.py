# alpha_discovery/options/pricing.py
"""
Facade module for options pricing utilities.

Public API is unchanged for callers that do:
    from ..options import pricing
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import math

import numpy as np
import pandas as pd

from ..config import settings
from .models import (
    EPS_SIGMA, EPS_T, CLAMP_SIGMA_MIN, CLAMP_SIGMA_MAX, TRADING_DAYS_PER_YEAR, REF_3M_BD, REF_1M_BD,
    bs_price_call, bs_price_put, bs_delta, solve_strike_for_delta
)
from .market import (
    get_underlying_price, get_price_on_exit,
    get_entry_iv_3m, get_exit_iv_3m, map_sigma_from_3m_to_T, has_required_iv_series,
    get_risk_free_rate, get_30d_atm_iv, get_1m_smile_iv, interpolate_smile_iv,
    check_new_iv_columns_available, _normalize_iv
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

    # New fields for enhanced IV tracking
    iv_anchor: Optional[str] = None
    delta_bucket: Optional[str] = None
    iv_ref_days: Optional[int] = None
    sigma_anchor: Optional[float] = None
    sigma_entry: Optional[float] = None
    sigma_exit: Optional[float] = None
    delta_target: Optional[float] = None
    delta_achieved: Optional[float] = None
    K_over_S: Optional[float] = None
    fallback_to_3M: bool = False


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

def get_iv_3m_for_entry(ticker: str, date: pd.Timestamp, direction: Literal["long", "short"], df: pd.DataFrame) -> \
Optional[float]:
    return get_entry_iv_3m(ticker, date, direction, df)


def get_underlier_on_entry(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, date, df)


def get_underlying_eod(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, date, df)


def get_price_on_entry(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, date, df)


def build_option_strike(S0: float, direction: Literal["long", "short"]) -> float:
    return float(S0)


# ---- NEW: Realistic Strike Generation ----

def get_realistic_strike_increment(S: float, ticker: str = None) -> float:
    """
    Get realistic strike increment based on underlying price and ticker.
    This mimics real options market conventions.
    """
    # Common ETF/Index rules
    if ticker and any(x in ticker.upper() for x in ['SPY', 'QQQ', 'IWM', 'VIX']):
        if S <= 25:
            return 0.50
        elif S <= 50:
            return 1.00
        elif S <= 200:
            return 1.00
        else:
            return 5.00

    # Individual stock rules (approximate)
    if S <= 25:
        return 2.50
    elif S <= 50:
        return 2.50
    elif S <= 100:
        return 2.50
    elif S <= 200:
        return 5.00
    elif S <= 500:
        return 5.00
    else:
        return 10.00


def round_to_realistic_strike(K: float, S: float, ticker: str = None) -> float:
    """
    Round strike to realistic increment and ensure it's reasonable vs spot.
    """
    increment = get_realistic_strike_increment(S, ticker)

    # Round to nearest increment
    rounded_K = round(K / increment) * increment

    # Ensure strike is reasonable (within ±50% of spot as sanity check)
    min_strike = S * 0.5
    max_strike = S * 1.5

    if rounded_K < min_strike:
        rounded_K = math.ceil(min_strike / increment) * increment
    elif rounded_K > max_strike:
        rounded_K = math.floor(max_strike / increment) * increment

    return rounded_K


def get_expiration_days_from_horizon(trade_horizon_days: int, entry_date: pd.Timestamp = None) -> int:
    """
    Convert trade horizon to option expiration days.

    Args:
        trade_horizon_days: How long you plan to hold the trade
        entry_date: When the trade starts (unused for now, but kept for future use)

    Returns:
        Days to option expiration (should be >= trade_horizon_days + buffer)
    """
    # Add buffer days to avoid early expiration
    # Common practice: add 5-10 business days buffer
    buffer_days = max(5, trade_horizon_days // 4)  # At least 5 days, or 25% buffer

    # Total days to expiration
    expiration_days = trade_horizon_days + buffer_days

    # Round to common expiration cycles (weekly/monthly)
    # Most liquid options expire on Fridays (weekly) or 3rd Friday (monthly)
    if expiration_days <= 10:
        # Use weekly options (round to nearest week)
        return ((expiration_days + 3) // 7) * 7  # Round up to nearest Friday
    elif expiration_days <= 45:
        # Use monthly options (round to nearest month ~21-22 trading days)
        return min(21, 42, 63, key=lambda x: abs(x - expiration_days))
    else:
        # Use quarterly options
        return min(63, 126, 189, key=lambda x: abs(x - expiration_days))


# ---- Enhanced IV Pricing System ----

def get_anchor_iv(
        ticker: str,
        date: pd.Timestamp,
        df: pd.DataFrame,
        iv_anchor: str,
        delta_bucket: str,
        direction: Literal["long", "short"],
        strict_new_iv: bool = True
) -> tuple[Optional[float], int, bool]:
    """
    Get anchor IV based on configuration.
    Returns (iv, ref_days, fallback_used).
    """
    fallback_used = False

    # Map pricing regime to specific settings
    if hasattr(settings.options, 'pricing_regime'):
        regime = settings.options.pricing_regime
        if regime == 'LEGACY_3M':
            iv_anchor = '3M'
            delta_bucket = 'ATM'
        elif regime == 'ATM_30D':
            iv_anchor = '30D'
            delta_bucket = 'ATM'
        elif regime == 'SMILE_1M':
            iv_anchor = '1M'
            # Keep existing delta_bucket setting

    # Check if new columns are available
    if iv_anchor in ['1M', '30D']:
        available, missing = check_new_iv_columns_available(ticker, date, df)
        if not available:
            if strict_new_iv:
                raise ValueError(f"Missing required IV columns for {ticker} on {date}: {missing}")
            else:
                print(f"Fallback to 3M IV for {ticker} on {date}: missing {missing}")
                iv_anchor = '3M'
                delta_bucket = 'ATM'
                fallback_used = True

    # Get IV based on anchor
    if iv_anchor == '3M':
        iv = get_entry_iv_3m(ticker, date, direction, df)
        ref_days = REF_3M_BD
    elif iv_anchor == '30D':
        iv = get_30d_atm_iv(ticker, date, df, direction)
        ref_days = REF_1M_BD
    elif iv_anchor == '1M':
        if delta_bucket == 'ATM':
            iv = get_30d_atm_iv(ticker, date, df, direction)
        else:
            # Map AUTO_BY_DIRECTION to specific bucket
            if delta_bucket == 'AUTO_BY_DIRECTION':
                bucket = 'CALL_40D' if direction == 'long' else 'PUT_40D'
            else:
                bucket = delta_bucket
            iv = get_1m_smile_iv(ticker, date, df, bucket)
        ref_days = REF_1M_BD
    else:
        raise ValueError(f"Unknown iv_anchor: {iv_anchor}")

    return iv, ref_days, fallback_used


def map_anchor_to_tenor(
        anchor_iv: float,
        ref_days: int,
        target_days: int,
        ticker: str,
        date: pd.Timestamp,
        df: pd.DataFrame
) -> float:
    """
    Map anchor IV to target tenor using power law and realized vol blend.
    """
    alpha = getattr(settings.options, "iv_map_alpha", 0.7)
    beta = getattr(settings.options, "power_law_beta", -0.15)

    # Power law scaling
    scaling_factor = (target_days / ref_days) ** beta
    scaled_iv = anchor_iv * scaling_factor

    # Blend with realized volatility (as-of date, no look-ahead)
    rv_series = None
    try:
        px_series = df.get(f"{ticker}_PX_LAST")
        if px_series is not None:
            # Get 20-day realized vol as-of date
            prices_to_date = px_series[px_series.index <= date].dropna()
            if len(prices_to_date) >= 21:  # Need 21 days for 20-day returns
                returns = prices_to_date.pct_change().dropna()
                if len(returns) >= 20:
                    rv_20d = returns.iloc[-20:].std() * np.sqrt(252)
                    if rv_20d > 0:
                        rv_series = rv_20d
    except Exception:
        pass

    if rv_series is not None:
        # Blend scaled IV with realized vol
        final_iv = alpha * scaled_iv + (1 - alpha) * rv_series
    else:
        final_iv = scaled_iv

    # Clamp to reasonable bounds
    return max(CLAMP_SIGMA_MIN, min(CLAMP_SIGMA_MAX, final_iv))


def determine_strike_and_delta_realistic(
        S: float,
        anchor_iv: float,
        ref_days: int,
        target_days: int,
        delta_bucket: str,
        direction: Literal["long", "short"],
        r: float,
        ticker: str,
        date: pd.Timestamp,
        df: pd.DataFrame,
        q: float = 0.0,
        force_atm: bool = False
) -> tuple[float, Optional[float], Optional[float]]:
    """
    Enhanced version with realistic strike rounding and ATM forcing.
    Returns (K, target_delta, achieved_delta).
    """
    option_type: Literal["call", "put"] = "call" if direction == "long" else "put"
    T_years = target_days / TRADING_DAYS_PER_YEAR

    # Map sigma for target tenor
    sigma = map_anchor_to_tenor(anchor_iv, ref_days, target_days, ticker, date, df)

    # FORCE ATM MODE - this is probably what you want for most strategies
    if force_atm or delta_bucket == 'ATM':
        K = round_to_realistic_strike(S, S, ticker)  # Round spot to nearest strike
        target_delta = None
        achieved_delta = bs_delta(S, K, T_years, r, sigma, option_type, q)
        return K, target_delta, achieved_delta

    # Delta-target mode (only if explicitly requested)
    delta_mapping = {
        'CALL_40D': 0.40, 'CALL_25D': 0.25, 'CALL_10D': 0.10,
        'PUT_40D': -0.40, 'PUT_25D': -0.25, 'PUT_10D': -0.10
    }

    if delta_bucket == 'AUTO_BY_DIRECTION':
        target_delta = 0.40 if direction == "long" else -0.40
    elif delta_bucket in delta_mapping:
        target_delta = delta_mapping[delta_bucket]
    else:
        # Unknown bucket - default to ATM
        K = round_to_realistic_strike(S, S, ticker)
        target_delta = None
        achieved_delta = bs_delta(S, K, T_years, r, sigma, option_type, q)
        return K, target_delta, achieved_delta

    # Solve for strike with realistic rounding
    try:
        K_solved = solve_strike_for_delta(S, target_delta, T_years, r, sigma, option_type, q)
        if K_solved is not None:
            K = round_to_realistic_strike(K_solved, S, ticker)
        else:
            # Fallback to ATM
            K = round_to_realistic_strike(S, S, ticker)
    except Exception:
        # Fallback to ATM
        K = round_to_realistic_strike(S, S, ticker)

    achieved_delta = bs_delta(S, K, T_years, r, sigma, option_type, q)

    return K, target_delta, achieved_delta


# LEGACY VERSION (keep for compatibility)
def determine_strike_and_delta(
        S: float,
        anchor_iv: float,
        ref_days: int,
        target_days: int,
        delta_bucket: str,
        direction: Literal["long", "short"],
        r: float,
        ticker: str,
        date: pd.Timestamp,
        df: pd.DataFrame,
        q: float = 0.0
) -> tuple[float, Optional[float], Optional[float]]:
    """
    Legacy version - now just calls the realistic version with force_atm=True
    """
    return determine_strike_and_delta_realistic(
        S, anchor_iv, ref_days, target_days, delta_bucket, direction, r,
        ticker, date, df, q, force_atm=True
    )


def price_entry_exit_enhanced(
        S0: float,
        S1: float,
        T0_days: int,  # Business days to expiry at entry
        h_days: int,  # Holding period days
        r: float,
        direction: Literal["long", "short"],
        ticker: str,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        df: pd.DataFrame,
        q: float = 0.0,
        K_override: Optional[float] = None  # For consistency with existing strike
) -> Optional[PricedLeg]:
    """
    Enhanced pricing using new IV anchor system.
    """
    # Get configuration
    iv_anchor = getattr(settings.options, "iv_anchor", "1M")
    delta_bucket = getattr(settings.options, "delta_bucket", "AUTO_BY_DIRECTION")
    strict_new_iv = getattr(settings.options, "strict_new_iv", True)

    # Get anchor IV for entry
    try:
        entry_anchor_iv, ref_days, fallback_used = get_anchor_iv(
            ticker, entry_date, df, iv_anchor, delta_bucket, direction, strict_new_iv
        )
        if entry_anchor_iv is None:
            return None
    except Exception as e:
        if strict_new_iv:
            raise e
        else:
            #print(f"Error getting anchor IV, falling back to 3M: {e}")
            entry_anchor_iv = get_entry_iv_3m(ticker, entry_date, direction, df)
            if entry_anchor_iv is None:
                return None
            ref_days = REF_3M_BD
            fallback_used = True

    # Determine strike (use override if provided for exit consistency)
    if K_override is not None:
        K = K_override
        # Calculate deltas with existing strike
        T0_years = T0_days / TRADING_DAYS_PER_YEAR
        entry_sigma = map_anchor_to_tenor(entry_anchor_iv, ref_days, T0_days, ticker, entry_date, df)
        option_type: Literal["call", "put"] = "call" if direction == "long" else "put"
        target_delta = None
        achieved_delta = bs_delta(S0, K, T0_years, r, entry_sigma, option_type, q)
    else:
        # Solve for new strike with error handling
        try:
            K, target_delta, achieved_delta = determine_strike_and_delta(
                S0, entry_anchor_iv, ref_days, T0_days, delta_bucket, direction, r,
                ticker, entry_date, df, q
            )
        except Exception as e:
            # Fallback to ATM if strike solving fails
            K = S0
            target_delta = None
            T0_years = T0_days / TRADING_DAYS_PER_YEAR
            entry_sigma = map_anchor_to_tenor(entry_anchor_iv, ref_days, T0_days, ticker, entry_date, df)
            option_type: Literal["call", "put"] = "call" if direction == "long" else "put"
            achieved_delta = bs_delta(S0, K, T0_years, r, entry_sigma, option_type, q)

    # Calculate entry and exit sigmas
    entry_sigma = map_anchor_to_tenor(entry_anchor_iv, ref_days, T0_days, ticker, entry_date, df)

    # Get exit anchor IV (may be different from entry due to time change)
    try:
        exit_anchor_iv, _, _ = get_anchor_iv(
            ticker, exit_date, df, iv_anchor, delta_bucket, direction, strict_new_iv
        )
        if exit_anchor_iv is None:
            exit_anchor_iv = entry_anchor_iv  # Fallback to entry IV
    except Exception:
        exit_anchor_iv = entry_anchor_iv

    T1_days = max(1, T0_days - h_days)
    exit_sigma = map_anchor_to_tenor(exit_anchor_iv, ref_days, T1_days, ticker, exit_date, df)

    # Calculate option prices
    T0_years = T0_days / TRADING_DAYS_PER_YEAR
    T1_years = T1_days / TRADING_DAYS_PER_YEAR
    option_type: Literal["call", "put"] = "call" if direction == "long" else "put"

    if option_type == "call":
        entry_price = bs_price_call(S0, K, T0_years, r, entry_sigma, q)
        exit_price = bs_price_call(S1, K, T1_years, r, exit_sigma, q)
    else:
        entry_price = bs_price_put(S0, K, T0_years, r, entry_sigma, q)
        exit_price = bs_price_put(S1, K, T1_years, r, exit_sigma, q)

    entry_price = max(0.0, float(entry_price))
    exit_price = max(0.0, float(exit_price))

    return PricedLeg(
        entry_price=entry_price,
        exit_price=exit_price,
        T_exit=T1_years,
        entry_iv=entry_sigma,
        exit_iv=exit_sigma,
        option_type=option_type,
        iv_anchor=iv_anchor,
        delta_bucket=delta_bucket,
        iv_ref_days=ref_days,
        sigma_anchor=entry_anchor_iv,
        sigma_entry=entry_sigma,
        sigma_exit=exit_sigma,
        delta_target=target_delta,
        delta_achieved=achieved_delta,
        K_over_S=K / S0,
        fallback_to_3M=fallback_used
    )


def price_entry_exit_enhanced_fixed(
        S0: float,
        S1: float,
        trade_horizon_days: int,  # CHANGED: This is your actual trade horizon
        r: float,
        direction: Literal["long", "short"],
        ticker: str,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        df: pd.DataFrame,
        q: float = 0.0,
        K_override: Optional[float] = None,
        force_atm: bool = True  # NEW: Default to ATM mode
) -> Optional[PricedLeg]:
    """
    Fixed pricing with realistic strikes and proper expiration handling.
    This should be your main pricing function going forward.
    """
    # Calculate option expiration days (should be > trade horizon)
    T0_days = get_expiration_days_from_horizon(trade_horizon_days, entry_date)

    # Get configuration - but override with force_atm if needed
    iv_anchor = getattr(settings.options, "iv_anchor", "30D")  # Changed default to 30D ATM
    delta_bucket = "ATM" if force_atm else getattr(settings.options, "delta_bucket", "ATM")
    strict_new_iv = getattr(settings.options, "strict_new_iv", False)  # More forgiving

    # Get anchor IV for entry
    try:
        entry_anchor_iv, ref_days, fallback_used = get_anchor_iv(
            ticker, entry_date, df, iv_anchor, delta_bucket, direction, strict_new_iv
        )
        if entry_anchor_iv is None:
            return None
    except Exception as e:
        #print(f"Error getting anchor IV, falling back to 3M: {e}")
        entry_anchor_iv = get_entry_iv_3m(ticker, entry_date, direction, df)
        if entry_anchor_iv is None:
            return None
        ref_days = REF_3M_BD
        fallback_used = True

    # Determine strike with realistic rounding
    if K_override is not None:
        K = round_to_realistic_strike(K_override, S0, ticker)
        T0_years = T0_days / TRADING_DAYS_PER_YEAR
        entry_sigma = map_anchor_to_tenor(entry_anchor_iv, ref_days, T0_days, ticker, entry_date, df)
        option_type: Literal["call", "put"] = "call" if direction == "long" else "put"
        target_delta = None
        achieved_delta = bs_delta(S0, K, T0_years, r, entry_sigma, option_type, q)
    else:
        K, target_delta, achieved_delta = determine_strike_and_delta_realistic(
            S0, entry_anchor_iv, ref_days, T0_days, delta_bucket, direction, r,
            ticker, entry_date, df, q, force_atm
        )

    # Rest of pricing logic...
    entry_sigma = map_anchor_to_tenor(entry_anchor_iv, ref_days, T0_days, ticker, entry_date, df)

    # Get exit IV
    try:
        exit_anchor_iv, _, _ = get_anchor_iv(
            ticker, exit_date, df, iv_anchor, delta_bucket, direction, strict_new_iv
        )
        if exit_anchor_iv is None:
            exit_anchor_iv = entry_anchor_iv
    except Exception:
        exit_anchor_iv = entry_anchor_iv

    # Calculate remaining time at exit
    T1_days = max(1, T0_days - trade_horizon_days)  # Should be buffer days remaining
    exit_sigma = map_anchor_to_tenor(exit_anchor_iv, ref_days, T1_days, ticker, exit_date, df)

    # Calculate option prices
    T0_years = T0_days / TRADING_DAYS_PER_YEAR
    T1_years = T1_days / TRADING_DAYS_PER_YEAR
    option_type: Literal["call", "put"] = "call" if direction == "long" else "put"

    if option_type == "call":
        entry_price = bs_price_call(S0, K, T0_years, r, entry_sigma, q)
        exit_price = bs_price_call(S1, K, T1_years, r, exit_sigma, q)
    else:
        entry_price = bs_price_put(S0, K, T0_years, r, entry_sigma, q)
        exit_price = bs_price_put(S1, K, T1_years, r, exit_sigma, q)

    entry_price = max(0.0, float(entry_price))
    exit_price = max(0.0, float(exit_price))

    return PricedLeg(
        entry_price=entry_price,
        exit_price=exit_price,
        T_exit=T1_years,
        entry_iv=entry_sigma,
        exit_iv=exit_sigma,
        option_type=option_type,
        iv_anchor=iv_anchor,
        delta_bucket=delta_bucket,
        iv_ref_days=ref_days,
        sigma_anchor=entry_anchor_iv,
        sigma_entry=entry_sigma,
        sigma_exit=exit_sigma,
        delta_target=target_delta,
        delta_achieved=achieved_delta,
        K_over_S=K / S0,
        fallback_to_3M=fallback_used
    )


__all__ = [
    # constants (if anyone imports through the facade)
    "EPS_SIGMA", "EPS_T", "CLAMP_SIGMA_MIN", "CLAMP_SIGMA_MAX", "TRADING_DAYS_PER_YEAR",
    # primitives
    "bs_price_call", "bs_price_put", "bs_delta", "solve_strike_for_delta",
    # IV + tenor mapping
    "get_entry_iv_3m", "get_exit_iv_3m", "map_sigma_from_3m_to_T", "has_required_iv_series",
    # NEW: Enhanced IV system
    "get_anchor_iv", "map_anchor_to_tenor", "determine_strike_and_delta", "determine_strike_and_delta_realistic",
    "price_entry_exit_enhanced", "price_entry_exit_enhanced_fixed",
    "get_30d_atm_iv", "get_1m_smile_iv", "interpolate_smile_iv", "check_new_iv_columns_available",
    # NEW: Realistic strike functions
    "get_realistic_strike_increment", "round_to_realistic_strike", "get_expiration_days_from_horizon",
    # rates
    "get_risk_free_rate",
    # underlying access
    "get_underlying_price", "get_price_on_exit",
    # wrapper + dataclass
    "PricedLeg", "price_entry_exit",
    # backtester aliases (unchanged)
    "get_iv_3m_for_entry", "get_underlier_on_entry", "get_underlying_eod", "get_price_on_entry", "build_option_strike",
]