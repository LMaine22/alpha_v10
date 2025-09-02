# alpha_discovery/engine/bt_common.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import math
import numpy as np
import pandas as pd

from ..config import settings
from ..options import pricing  # user module; we only *call if exists*

# ---------------- Constants & tiny utils ----------------

TRADE_HORIZONS_DAYS = [10]

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
    pt_behavior: str = "exit"         # 'exit' | 'arm_trail' | 'scale_out'
    armed_trail_frac: float | None = None
    scale_out_frac: float = 0.50

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
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10] if s else ""

def _decide_exit_with_scale_out(price_path: pd.Series,
                                entry_exec_price: float,
                                horizon_days: int,
                                policy: ExitPolicy) -> tuple[bool, dict]:
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
    def _first_true_index(mask: np.ndarray) -> Optional[int]:
        idxs = np.nonzero(mask)[0]
        return int(idxs[0]) if idxs.size > 0 else None

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
                           policy: ExitPolicy) -> Tuple[int, str, int]:
    """
    Single-leg exit decision on an option price path with:
      - Profit Target (PT)
      - Trailing Stop (TS)
      - Stop-Loss (SL)
      - Time cap / Horizon fallback

    Returns:
        exit_idx (int): 0-based index into price_path where we exit
        exit_reason (str): 'profit_target' | 'trailing_stop' | 'stop_loss' | 'time_stop' | 'horizon'
        holding_days_actual (int): equals exit_idx (0..len(path)-1)
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

    # Tripwires (None => disabled)
    # Profit Target
    pt_idx = None
    if policy.enabled and policy.pt_multiple is not None and policy.pt_multiple > 0:
        pt_thresh = float(entry_exec_price) * float(policy.pt_multiple)
        pt_hits = np.nonzero(p >= pt_thresh)[0]
        pt_idx = int(pt_hits[0]) if pt_hits.size > 0 else None

    # Trailing Stop (uses running peak from entry)
    ts_idx = None
    if policy.enabled and policy.trail_frac is not None and 0.0 < float(policy.trail_frac) < 1.0:
        peak = np.maximum.accumulate(p)
        ts_thresh = peak * float(policy.trail_frac)
        ts_hits = np.nonzero(p <= ts_thresh)[0]
        ts_idx = int(ts_hits[0]) if ts_hits.size > 0 else None

    # Hard Stop-Loss (vs entry)
    sl_idx = None
    if policy.enabled and policy.sl_multiple is not None and policy.sl_multiple > 0:
        sl_thresh = float(entry_exec_price) * float(policy.sl_multiple)
        sl_hits = np.nonzero(p <= sl_thresh)[0]
        sl_idx = int(sl_hits[0]) if sl_hits.size > 0 else None

    # Choose earliest event; on same-day ties, PT beats TS, TS beats SL
    candidates = []
    if pt_idx is not None: candidates.append((pt_idx, "profit_target", 0))
    if ts_idx is not None: candidates.append((ts_idx, "trailing_stop", 1))
    if sl_idx is not None: candidates.append((sl_idx, "stop_loss", 2))

    if candidates:
        day, reason, _prio = sorted(candidates, key=lambda t: (t[0], t[2]))[0]
        return int(day), reason, int(day)

    # No tripwire -> time/horizon
    final_idx = last_idx
    final_reason = "horizon" if time_cap == int(horizon_days) else "time_stop"
    return int(final_idx), final_reason, int(final_idx)



# ---------------- Pricing bridge (delegates to options.pricing if present) ----------------

def _call_if_exists(func_name: str, *args, **kwargs):
    fn = getattr(pricing, func_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None

def _get_risk_free_rate(date: pd.Timestamp, df: Optional[pd.DataFrame]) -> float:
    r = _call_if_exists("get_risk_free_rate", date, df=df)
    if r is not None:
        return float(r)
    return float(getattr(settings.options, "constant_r", 0.0))

def _has_iv_for_ticker(ticker: str, df: pd.DataFrame) -> bool:
    if getattr(settings.options, "allow_nonoptionable", False):
        return True
    ok = _call_if_exists("has_required_iv_series", ticker, df)
    if ok is not None:
        return bool(ok)
    iv_candidates = [f"{ticker}_IVOL_3M", f"{ticker}_IV_3M", f"{ticker}_IV3M", f"{ticker}_IVOL"]
    return _find_col(df, iv_candidates) is not None

def _get_underlier_eod_raw(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    for fn in ["get_price_on_entry", "get_underlier_on_entry", "get_underlying_eod", "get_underlying_price"]:
        v = _call_if_exists(fn, ticker, date, df)
        if v is not None:
            return float(v)
    price_col = _find_col(df, [f"{ticker}_PX_LAST", f"{ticker}_Close", f"{ticker}_AdjClose", f"{ticker}_PX_LAST_USD"])
    return _get_eod_value(df, price_col, date) if price_col else None

def _get_iv3m_eod_raw(ticker: str, date: pd.Timestamp, direction: str, df: pd.DataFrame,
                      fallback_sigma: Optional[float] = None) -> Optional[float]:
    v = _call_if_exists("get_entry_iv_3m", ticker, date, direction, df)
    if v is not None:
        return float(v)
    v = _call_if_exists("get_exit_iv_3m", ticker, date, direction, df, fallback_sigma=fallback_sigma)
    if v is not None:
        return float(v)
    iv_col = _find_col(df, [f"{ticker}_IVOL_3M", f"{ticker}_IV_3M", f"{ticker}_IV3M", f"{ticker}_IVOL"])
    return _get_eod_value(df, iv_col, date) if iv_col else fallback_sigma

def _build_option_strike(S0: float, direction: str) -> float:
    K = _call_if_exists("build_option_strike", S0, direction)
    return float(K) if K is not None else float(S0)

def _map_sigma_from_3m_to_T_raw(sigma_3m: float, tenor_bd: int, ticker: str,
                                date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    sig = _call_if_exists("map_sigma_from_3m_to_T", sigma_3m, tenor_bd, ticker, date, df)
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

def _bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    S = float(S); K = float(K); T = max(1e-6, float(T)); r = float(r); sigma = max(1e-6, float(sigma)); q = float(q)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type.lower().startswith("c"):
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)

def _price_entry_exit_fallback(S0: float, S1: float, K: float, T0_years: float, h_days: int, r: float,
                               option_type: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Dict[str, float]:
    T1_years = max(1e-6, (max(0, int(round(T0_years * 252))) - h_days) / 252.0)
    entry_price = _bs_price(S0, K, T0_years, r, entry_sigma, option_type, q)
    exit_price = _bs_price(S1, K, T1_years, r, exit_sigma, option_type, q)
    return {
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "entry_iv": float(entry_sigma),
        "exit_iv": float(exit_sigma),
        "option_type": "C" if option_type.lower().startswith("c") else "P",
    }

def _price_entry_exit(S0: float, S1: float, K: float, T0_years: float, h_days: int, r: float,
                      direction: str, entry_sigma: float, exit_sigma: float, q: float = 0.0) -> Optional[Dict[str, float]]:
    """Try user pricing.price_entry_exit; else Black–Scholes fallback."""
    try:
        h_days_num = int(h_days)
    except Exception:
        try:
            h_days_num = int(float(h_days))
        except Exception:
            h_days_num = 0

    result = _call_if_exists(
        "price_entry_exit",
        S0=S0, S1=S1, K=K, T0=T0_years, h_days=h_days_num, r=r,
        direction=direction, entry_sigma=entry_sigma, exit_sigma=exit_sigma, q=q
    )
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
    return _price_entry_exit_fallback(S0, S1, K, T0_years, h_days_num, r, opt_type, entry_sigma, exit_sigma, q)
