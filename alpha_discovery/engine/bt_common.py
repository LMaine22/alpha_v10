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

TRADE_HORIZONS_DAYS = [12]  # Optimized for regime-aware exits


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

# ---------------- Exit policy ----------------


@dataclass(frozen=True)
class ExitPolicy:
    enabled: bool = True
    pt_multiple: float | None = None
    trail_frac: float | None = None
    sl_multiple: float | None = None
    time_cap_days: int | None = None  # if None, defaults to horizon

    # NEW behavior knobs (read from settings or GA policy dict)
    pt_behavior: str = "exit"         # 'exit' | 'arm_trail' | 'scale_out' | 'regime_aware'
    armed_trail_frac: float | None = None
    scale_out_frac: float = 0.50
    # regime-aware settings
    regime_aware: bool = False

    def is_noop(self) -> bool:
        return (self.pt_multiple is None and
                self.trail_frac is None and
                self.sl_multiple is None and
                self.time_cap_days is None)


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    idxs = np.nonzero(mask)[0]
    return int(idxs[0]) if idxs.size > 0 else None


def _policy_id(policy: ExitPolicy, horizon_days: int) -> str:
    active = {
        "pt": policy.pt_multiple,
        "ts": policy.trail_frac,
        "sl": policy.sl_multiple,
        "tc": policy.time_cap_days,
        "hz": horizon_days,
        # NEW:
        "pb": policy.pt_behavior,
        "at": policy.armed_trail_frac,
        "sf": policy.scale_out_frac,
    }
    s = "|".join(f"{k}={v}" for k, v in active.items() if v is not None)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10] if s else ""


def _decide_exit_with_scale_out(price_path: pd.Series,
                                entry_exec_price: float,
                                horizon_days: int,
                                policy: ExitPolicy) -> tuple[bool, dict]:
    # DISABLED - Only regime-aware exits should be used for extreme runners
    return False, {"exit_idx": len(price_path) - 1, "exit_reason": "horizon", "holding_days": len(price_path) - 1}
    """
    Returns (did_scale, info dict).
    If PT isn't hit or behavior is incompatible, returns (False, {}).

    When scale-out triggers:
      info = {
        "pt_idx": int,                 # day index of PT hit (0-based)
        "pt_reason": "profit_target_partial",
        "final_idx": int,              # day index of final exit
        "final_reason": str,           # e.g., 'trailing_stop' | 'stop_loss' | 'horizon' | 'time_stop'
      }
    """
    n = len(price_path)
    if n == 0:
        return False, {}

    # Respect time cap just like the regular path
    time_cap = min(horizon_days, policy.time_cap_days if policy.time_cap_days is not None else horizon_days)
    time_cap = max(0, time_cap)
    last_idx = min(time_cap, n - 1)

    # Need a real PT and the 'scale_out' behavior
    if (not policy.enabled) or policy.pt_multiple is None or policy.pt_multiple <= 0:
        return False, {}
    if str(policy.pt_behavior).lower() != "scale_out":
        return False, {}

    p = price_path.iloc[: last_idx + 1].astype(float).values
    pt_thresh = float(entry_exec_price) * float(policy.pt_multiple)
    pt_mask = p >= pt_thresh
    pt_idxs = np.nonzero(pt_mask)[0]
    if pt_idxs.size == 0:
        return False, {}  # PT never hit

    pt_idx = int(pt_idxs[0])

    # After PT, tighten trailing (if configured), keep SL, same time cap remainder
    armed_frac = policy.armed_trail_frac if (policy.armed_trail_frac is not None) else policy.trail_frac
    p2 = p[pt_idx: last_idx + 1]  # includes PT day
    peak2 = np.maximum.accumulate(p2)

    ts2_mask = np.zeros_like(p2, dtype=bool)
    if armed_frac is not None and 0 < float(armed_frac) < 1:
        ts2_thresh = peak2 * float(armed_frac)
        ts2_mask = p2 <= ts2_thresh

    sl2_mask = np.zeros_like(p2, dtype=bool)
    if policy.sl_multiple is not None and policy.sl_multiple > 0:
        sl_thresh = float(entry_exec_price) * float(policy.sl_multiple)
        sl2_mask = p2 <= sl_thresh

    # earliest of (trailing, stop) after PT; otherwise time/horizon
    day_ts2 = _first_true_index(ts2_mask)
    day_sl2 = _first_true_index(sl2_mask)

    candidates = []
    if day_ts2 is not None: candidates.append((day_ts2, "trailing_stop"))
    if day_sl2 is not None: candidates.append((day_sl2, "stop_loss"))

    if candidates:
        earliest_rel, reason = min(candidates, key=lambda t: t[0])
        final_idx = pt_idx + int(earliest_rel)
        return True, {"pt_idx": pt_idx, "pt_reason": "profit_target_partial",
                      "final_idx": final_idx, "final_reason": reason}

    # No TS/SL after PT -> end at time cap / horizon
    final_idx = last_idx
    final_reason = "horizon" if time_cap == horizon_days else "time_stop"
    return True, {"pt_idx": pt_idx, "pt_reason": "profit_target_partial",
                  "final_idx": final_idx, "final_reason": final_reason}


def _decide_exit_from_path(price_path: pd.Series,
                           entry_exec_price: float,
                           horizon_days: int,
                           policy: ExitPolicy,
                           direction: str = "long") -> Tuple[int, str, int]:
    """
    Single-leg exit decision on an option price path with:
      - Profit Target (PT)
      - Trailing Stop (TS)
      - Stop-Loss (SL)
      - Time cap / Horizon fallback
      - Regime-aware exits (if enabled)

    Returns:
        exit_idx (int): 0-based index into price_path where we exit
        exit_reason (str): 'profit_target' | 'trailing_stop' | 'stop_loss' | 'time_stop' | 'horizon' | regime-aware reasons
        holding_days_actual (int): equals exit_idx (0...len(path)-1)
    """
    n = len(price_path)
    if n == 0:
        return 0, "horizon", 0

    # Respect time cap (defaults to horizon if None)
    time_cap = min(int(horizon_days),
                   int(policy.time_cap_days) if policy.time_cap_days is not None else int(horizon_days))
    time_cap = max(0, time_cap)
    last_idx = min(time_cap, n - 1)

    # Slice path up to cap
    p = price_path.iloc[: last_idx + 1].astype(float).values

    # Regime-aware exits (if enabled) - check FIRST
    if policy.regime_aware or policy.pt_behavior == "regime_aware":
        regime_exit_idx, regime_reason = _check_regime_aware_exit_options(
            price_path, entry_exec_price, horizon_days, policy, direction
        )
        if regime_exit_idx is not None:
            return int(regime_exit_idx), regime_reason, int(regime_exit_idx)
        # If regime-aware exits are enabled but none triggered, fall through to traditional logic

    # TRADITIONAL EXIT POLICIES DISABLED - ONLY REGIME-AWARE EXITS FOR BIG RUNNERS
    # All traditional profit targets, trailing stops, and stop losses are bypassed
    # to allow regime-aware system to catch 300-1000%+ runners

    # No tripwire -> time/horizon
    final_idx = last_idx
    final_reason = "horizon" if time_cap == int(horizon_days) else "time_stop"
    return int(final_idx), final_reason, int(final_idx)


def _check_regime_aware_exit_options(
    price_path: pd.Series,
    entry_exec_price: float,
    horizon_days: int,
    policy: ExitPolicy,
    direction: str = "long"
) -> Tuple[Optional[int], str]:
    """
    Check for regime-aware exit conditions on option prices.
    
    This function implements regime-aware exit logic specifically for options trading,
    considering volatility regimes, time decay, and market conditions.
    
    Returns:
        exit_idx (Optional[int]): Index where to exit, or None if no exit
        exit_reason (str): Reason for exit
    """
    p = np.asarray(price_path, dtype=float)
    n = len(p)
    if n == 0:
        return None, "horizon"
    
    entry = float(entry_exec_price)
    
    # Get regime-aware config
    from alpha_discovery.config import settings
    regime_config = settings.regime_aware
    
    # Process each day in the price path
    for i in range(n):
        current_date = price_path.index[i]
        option_price = p[i]
        days_held = i
        profit_pct = (option_price - entry) / entry if entry > 0 else 0
        
        # === REGIME-AWARE EXIT CONDITIONS ===
        
        # 1. Profit Target Hit - Leg A (conservative profit taking)
        if profit_pct >= 1.0:  # 100% profit target
            return i, "pt_hit_legA"
        
        # 2. Volatility Spike Profit (quick exit on volatility expansion)
        if i >= 2:  # Need at least 3 days of data
            # Calculate recent volatility
            recent_prices = p[max(0, i-2):i+1]
            if len(recent_prices) >= 3:
                price_changes = np.diff(recent_prices) / recent_prices[:-1]
                vol_spike = np.std(price_changes) if len(price_changes) > 0 else 0
                
                # Exit if we have profit AND high volatility (volatility spike profit)
                if profit_pct >= 0.5 and vol_spike > 0.1:  # 50% profit + 10% daily volatility
                    return i, "volatility_spike_profit"
        
        # 3. Time Decay Protection (theta burn)
        if days_held >= horizon_days * 0.8:  # Exit in final 20% of horizon
            if profit_pct >= 0.3:  # Only if we have some profit
                return i, "time_decay_protection"
        
        # 4. Stop Loss (risk management)
        if profit_pct <= -0.5:  # 50% stop loss
            return i, "stop_loss"
        
        # 5. ATR-based Trailing Stop (regime-aware)
        if i >= 5 and profit_pct >= 0.2:  # Need 5 days of data and 20% profit
            # Calculate ATR over last 5 days
            recent_prices = p[max(0, i-4):i+1]
            if len(recent_prices) >= 5:
                high_low_range = np.max(recent_prices) - np.min(recent_prices)
                atr = high_low_range / len(recent_prices)
                
                # Trail at ATR distance from recent high
                recent_high = np.max(recent_prices)
                trail_level = recent_high - (atr * 2.0)  # 2x ATR trail
                
                if option_price <= trail_level:
                    return i, "atr_trail_hit"
        
        # 6. Horizon Protection (final safety net)
        if days_held >= horizon_days:
            return i, "horizon_protection"
    
    # No regime-aware exit triggered
    return None, "horizon"


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
    v = cast(Optional[float], _call_if_exists("get_exit_iv_3m", ticker, date, direction, df, fallback_sigma=fallback_sigma))
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
                               option_type: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Dict[str, float]:
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
                      direction: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Optional[Dict[str, float]]:
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
    Enhanced pricing using the new IV anchor system.
    Returns dict with original fields plus new tracking fields.
    """
    try:
        # Try enhanced pricing from pricing module
        result = cast(Optional[object], _call_if_exists(
            "price_entry_exit_enhanced",
            S0=s0, S1=s1, T0_days=t0_days, h_days=h_days, r=r,
            direction=direction, ticker=ticker, entry_date=entry_date, 
            exit_date=exit_date, df=master_df, q=q, K_override=K_override
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
                "delta_target": float(getattr(result, "delta_target", 0.0)) if getattr(result, "delta_target", None) is not None else None,
                "delta_achieved": float(getattr(result, "delta_achieved", 0.0)) if getattr(result, "delta_achieved", None) is not None else None,
                "K_over_S": float(getattr(result, "K_over_S", 1.0)),
                "fallback_to_3M": bool(getattr(result, "fallback_to_3M", False)),
            }
    except Exception as e:
        # Log error but don't fail completely
        print(f"Enhanced pricing failed for {ticker} on {entry_date}: {e}")
    
    # Fallback to traditional pricing
    return None