# alpha_discovery/engine/bt_common.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, cast
import math
import hashlib
import numpy as np
import pandas as pd

from ..config import settings
from ..options import pricing  # user module; we only *call if exists*

# ---------------- Constants & tiny utils ----------------

TRADE_HORIZONS_DAYS = [6]  # Optimized for regime-aware exits


def _add_bdays(start_date: pd.Timestamp, bd: int) -> pd.Timestamp:
    """Inclusive business-day add (entry day = 0)."""
    return pd.bdate_range(start_date, periods=bd + 1, freq="C")[-1]


def _get_eod_value(df: pd.DataFrame, col: str, d: pd.Timestamp) -> Optional[float]:
    """As-of lookup (no future peeking): last known value <= d."""
    if not col or col not in df.columns:
        return None
    s = df[col]
    try:
        if d in s.index and pd.notna(s.loc[d]):
            return float(s.loc[d])
        s2 = s.loc[:d]
        if not s2.empty:
            val = s2.iloc[-1]
            return float(val) if pd.notna(val) else None
    except Exception:
        pass
    return None


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_df_token(df: pd.DataFrame) -> str:
    tok = df.attrs.get("_cache_token", None)
    if tok is None:
        tok = f"{id(df)}-{df.shape[0]}x{df.shape[1]}"
        df.attrs["_cache_token"] = tok
    return tok


# ---------------- Simplified Exit Policy (Regime-Aware Only) ----------------


@dataclass(frozen=True)
class ExitPolicy:
    """
    Simplified exit policy that only supports regime-aware exits.
    All traditional exit policies (PT, TS, SL, time caps) are removed.
    """
    enabled: bool = True
    regime_aware: bool = True  # Always true for this system

    def is_noop(self) -> bool:
        """Returns true if exit policies are disabled."""
        return not self.enabled


def _policy_id(policy: ExitPolicy, horizon_days: int) -> str:
    """Generate a simple policy ID for regime-aware exits."""
    if not policy.enabled:
        return "disabled"
    return f"regime_aware_h{horizon_days}"


def _decide_exit_from_path(price_path: pd.Series,
                           entry_exec_price: float,
                           horizon_days: int,
                           policy: ExitPolicy,
                           direction: str = "long",
                           entry_date: pd.Timestamp = None,
                           tenor_bd: int = None) -> Optional[Tuple[int, str, int]]:
    """
    SIMPLIFIED: Only regime-aware exits are supported.

    Returns:
        None: if no exit condition is met (position remains open)
        (exit_idx, exit_reason, holding_days_actual): if an exit condition is triggered
    """
    n = len(price_path)
    if n == 0:
        return None

    # If exit policies are disabled, let it run to natural expiration
    if not policy.enabled or not policy.regime_aware:
        # Check if we've reached actual option expiration
        if entry_date and tenor_bd:
            option_expiration = _add_bdays(entry_date, tenor_bd)
            current_date = price_path.index[-1]  # Last date in our data
            if current_date >= option_expiration:
                return len(price_path) - 1, "option_expired", len(price_path) - 1
        return None  # Position remains open

    # Run regime-aware exit logic
    regime_exit_idx, regime_reason = _check_regime_aware_exit_options(
        price_path, entry_exec_price, horizon_days, policy, direction,
        entry_date, tenor_bd
    )

    if regime_exit_idx is not None:
        return int(regime_exit_idx), regime_reason, int(regime_exit_idx)

    # Check if option has actually expired
    if entry_date and tenor_bd:
        option_expiration = _add_bdays(entry_date, tenor_bd)
        current_date = price_path.index[-1]
        if current_date >= option_expiration:
            return len(price_path) - 1, "option_expired", len(price_path) - 1

    # No exit condition met and option hasn't expired - position remains open
    return None


def _check_regime_aware_exit_options(
        price_path: pd.Series,
        entry_exec_price: float,
        horizon_days: int,
        policy: ExitPolicy,
        direction: str = "long",
        entry_date: pd.Timestamp = None,
        tenor_bd: int = None
) -> Tuple[Optional[int], str]:
    """
    Enhanced regime-aware exit conditions for options trading.

    This implements the six core regime-aware exit conditions:
    1. Profit Target Hit - Leg A (conservative profit taking)
    2. Volatility Spike Profit (capitalize on vol expansion)
    3. Time Decay Protection (protect against theta burn)
    4. Stop Loss (risk management)
    5. ATR-based Trailing Stop (let winners run while protecting gains)
    6. Option Expiration (natural expiration of the contract)

    Returns:
        exit_idx (Optional[int]): Index where to exit, or None if no exit
        exit_reason (str): Reason for exit
    """
    p = np.asarray(price_path, dtype=float)
    n = len(p)
    if n == 0:
        return None, "no_data"

    entry = float(entry_exec_price)

    # Get regime-aware config
    from alpha_discovery.config import settings
    regime_config = settings.regime_aware

    # Calculate option expiration date if we have the information
    option_expiration = None
    if entry_date and tenor_bd:
        option_expiration = _add_bdays(entry_date, tenor_bd)

    # Process each day in the price path
    for i in range(n):
        current_date = price_path.index[i]
        option_price = p[i]
        days_held = i
        profit_pct = (option_price - entry) / entry if entry > 0 else 0

        # === REGIME-AWARE EXIT CONDITIONS ===

        # 1. Profit Target Hit - Leg A (conservative profit taking at 100%)
        if profit_pct >= 1.5:  # 100% profit target
            return i, "pt_hit_legA"

        # 2. Volatility Spike Profit (quick exit on volatility expansion)
        if i >= 2:  # Need at least 3 days of data for volatility calculation
            recent_prices = p[max(0, i - 2):i + 1]
            if len(recent_prices) >= 3:
                price_changes = np.diff(recent_prices) / recent_prices[:-1]
                vol_spike = np.std(price_changes) if len(price_changes) > 0 else 0

                # Exit if we have decent profit AND high volatility
                if profit_pct >= 0.8 and vol_spike > 0.15:  # 50% profit + 10% daily volatility
                    return i, "volatility_spike_profit"

        # 3. Time Decay Protection (theta burn protection)
        # This protects against time decay in the final portion of the option's life
        if option_expiration and tenor_bd:
            days_to_expiration = (option_expiration - current_date).days
            time_remaining_pct = days_to_expiration / tenor_bd if tenor_bd > 0 else 0

            # Exit in final 20% of option life if we have some profit
            if time_remaining_pct <= 0.1 and profit_pct >= 0.4:  # Final 20% + 30% profit
                return i, "time_decay_protection"

        # 4. Stop Loss (risk management - prevents catastrophic losses)
        if profit_pct <= -0.6:  # 50% stop loss
            return i, "stop_loss"

        # 5. ATR-based Trailing Stop (let winners run while protecting gains)
        if i >= 5 and profit_pct >= 0.2:  # Need 5 days of data and 20% profit to start trailing
            recent_prices = p[max(0, i - 4):i + 1]
            if len(recent_prices) >= 5:
                # Calculate ATR over last 5 days
                high_low_range = np.max(recent_prices) - np.min(recent_prices)
                atr = high_low_range / len(recent_prices)

                # Trail at 2x ATR distance from recent high
                recent_high = np.max(recent_prices)
                trail_level = recent_high - (atr * 2.0)

                if option_price <= trail_level:
                    return i, "atr_trail_hit"

        # 6. Option Expiration Check (natural expiration)
        if option_expiration and current_date >= option_expiration:
            return i, "option_expired"

    # No regime-aware exit triggered and option hasn't expired
    return None, "no_exit"


# ---------------- Pricing bridge (delegates to options.pricing if present) ----------------

def _call_if_exists(func_name: str, *args, **kwargs):
    fn = getattr(pricing, func_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _get_risk_free_rate(date: pd.Timestamp, df: Optional[pd.DataFrame]) -> float:
    r = cast(Optional[float], _call_if_exists("get_risk_free_rate", date, df=df))
    if r is not None:
        return float(r)
    return float(getattr(settings.options, "constant_r", 0.0))


def _has_iv_for_ticker(ticker: str, df: pd.DataFrame) -> bool:
    if getattr(settings.options, "allow_nonoptionable", False):
        return True
    ok = cast(Optional[bool], _call_if_exists("has_required_iv_series", ticker, df))
    if ok is not None:
        return bool(ok)
    iv_candidates = [f"{ticker}_IVOL_3M", f"{ticker}_IV_3M", f"{ticker}_IV3M", f"{ticker}_IVOL"]
    return _find_col(df, iv_candidates) is not None


def _get_underlier_eod_raw(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    for fn in ["get_price_on_entry", "get_underlier_on_entry", "get_underlying_eod", "get_underlying_price"]:
        v = cast(Optional[float], _call_if_exists(fn, ticker, date, df))
        if v is not None:
            return float(v)
    price_col = _find_col(df, [f"{ticker}_PX_LAST", f"{ticker}_Close", f"{ticker}_AdjClose", f"{ticker}_PX_LAST_USD"])
    return _get_eod_value(df, price_col, date) if price_col else None


def _get_iv3m_eod_raw(ticker: str, date: pd.Timestamp, direction: str, df: pd.DataFrame,
                      fallback_sigma: Optional[float] = None) -> Optional[float]:
    v = cast(Optional[float], _call_if_exists("get_entry_iv_3m", ticker, date, direction, df))
    if v is not None:
        return float(v)
    v = cast(Optional[float],
             _call_if_exists("get_exit_iv_3m", ticker, date, direction, df, fallback_sigma=fallback_sigma))
    if v is not None:
        return float(v)
    iv_col = _find_col(df, [f"{ticker}_IVOL_3M", f"{ticker}_IV_3M", f"{ticker}_IV3M", f"{ticker}_IVOL"])
    return _get_eod_value(df, iv_col, date) if iv_col else fallback_sigma


def _build_option_strike(s0: float, direction: str) -> float:
    k = cast(Optional[float], _call_if_exists("build_option_strike", s0, direction))
    return float(k) if k is not None else float(s0)


def _map_sigma_from_3m_to_T_raw(sigma_3m: float, tenor_bd: int, ticker: str,
                                date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    sig = cast(Optional[float], _call_if_exists("map_sigma_from_3m_to_T", sigma_3m, tenor_bd, ticker, date, df))
    if sig is not None:
        return float(sig)
    if sigma_3m is None or not np.isfinite(sigma_3m) or sigma_3m <= 0:
        return None
    beta = float(getattr(settings.options, "power_law_beta", -0.15))
    base_days = 63.0
    t_days = max(1.0, float(tenor_bd))
    mapped = float(sigma_3m) * (t_days / base_days) ** beta
    return max(1e-4, min(5.0, mapped))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(s: float, k: float, t: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    s = float(s)
    k = float(k)
    t = max(1e-6, float(t))
    r = float(r)
    sigma = max(1e-6, float(sigma))
    q = float(q)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type.lower().startswith("c"):
        return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    else:
        return k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)


def _price_entry_exit_fallback(s0: float, s1: float, k: float, t0_years: float, h_days: int, r: float,
                               option_type: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Dict[
    str, float]:
    t1_years = max(1e-6, (max(0, int(round(t0_years * 252))) - h_days) / 252.0)
    entry_price = _bs_price(s0, k, t0_years, r, entry_sigma, option_type, q)
    exit_price = _bs_price(s1, k, t1_years, r, exit_sigma, option_type, q)
    return {
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "entry_iv": float(entry_sigma),
        "exit_iv": float(exit_sigma),
        "option_type": "C" if option_type.lower().startswith("c") else "P",
    }


def _price_entry_exit(s0: float, s1: float, k: float, t0_years: float, h_days: int, r: float,
                      direction: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Optional[
    Dict[str, float]]:
    """Try user pricing.price_entry_exit; else Black–Scholes fallback."""
    try:
        h_days_num = int(h_days)
    except Exception:
        try:
            h_days_num = int(float(h_days))
        except Exception:
            h_days_num = 0

    result = cast(Optional[Dict], _call_if_exists(
        "price_entry_exit",
        S0=s0, S1=s1, K=k, T0=t0_years, h_days=h_days_num, r=r,
        direction=direction, entry_sigma=entry_sigma, exit_sigma=exit_sigma, q=q
    ))
    if result is not None:
        try:
            return {
                "entry_price": float(getattr(result, "entry_price", result["entry_price"])),
                "exit_price": float(getattr(result, "exit_price", result["exit_price"])),
                "entry_iv": float(getattr(result, "entry_iv", result["entry_iv"])),
                "exit_iv": float(getattr(result, "exit_iv", result["exit_iv"])),
                "option_type": str(getattr(result, "option_type", result["option_type"])),
            }
        except Exception:
            pass

    opt_type = "call" if direction == "long" else "put"
    return _price_entry_exit_fallback(s0, s1, k, t0_years, h_days_num, r, opt_type, entry_sigma, exit_sigma, q)


# ---- NEW: Enhanced pricing with IV anchor system ----

def _price_entry_exit_enhanced(
        s0: float, s1: float, t0_days: int, h_days: int, r: float,
        direction: str, ticker: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp,
        master_df: pd.DataFrame, q: float = 0.0, K_override: Optional[float] = None
) -> Optional[Dict[str, float]]:
    """
    Enhanced pricing using the new IV anchor system with realistic strikes.
    Returns dict with original fields plus new tracking fields.
    """
    try:
        # Use the NEW pricing function with realistic strikes
        result = cast(Optional[object], _call_if_exists(
            "price_entry_exit_enhanced_fixed",  # ← CHANGED: Use the new function
            S0=s0, S1=s1,
            trade_horizon_days=h_days,  # ← CHANGED: Use correct parameter name
            r=r, direction=direction, ticker=ticker,
            entry_date=entry_date, exit_date=exit_date,
            df=master_df, q=q, K_override=K_override,
            force_atm=True  # ← ADDED: This fixes your strike issues!
        ))

        if result is not None:
            # Extract all fields from PricedLeg dataclass
            return {
                "entry_price": float(getattr(result, "entry_price", 0.0)),
                "exit_price": float(getattr(result, "exit_price", 0.0)),
                "entry_iv": float(getattr(result, "entry_iv", 0.0)),
                "exit_iv": float(getattr(result, "exit_iv", 0.0)),
                "option_type": str(getattr(result, "option_type", "call")),
                # New enhanced fields
                "iv_anchor": str(getattr(result, "iv_anchor", "")),
                "delta_bucket": str(getattr(result, "delta_bucket", "")),
                "iv_ref_days": int(getattr(result, "iv_ref_days", 0)),
                "sigma_anchor": float(getattr(result, "sigma_anchor", 0.0)),
                "sigma_entry": float(getattr(result, "sigma_entry", 0.0)),
                "sigma_exit": float(getattr(result, "sigma_exit", 0.0)),
                "delta_target": float(getattr(result, "delta_target", 0.0)) if getattr(result, "delta_target",
                                                                                       None) is not None else None,
                "delta_achieved": float(getattr(result, "delta_achieved", 0.0)) if getattr(result, "delta_achieved",
                                                                                           None) is not None else None,
                "K_over_S": float(getattr(result, "K_over_S", 1.0)),
                "fallback_to_3M": bool(getattr(result, "fallback_to_3M", False)),
            }
    except Exception as e:
        # Log error but don't fail completely
        print(f"Enhanced pricing failed for {ticker} on {entry_date}: {e}")

    # Fallback to traditional pricing if enhanced fails
    return None