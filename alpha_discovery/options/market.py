# alpha_discovery/options/market.py
from __future__ import annotations
from typing import Optional, Literal
import math
import numpy as np
import pandas as pd

from ..config import settings
from .models import (
    CLAMP_SIGMA_MIN, CLAMP_SIGMA_MAX, TRADING_DAYS_PER_YEAR, REF_3M_BD, REF_1M_BD
)

# ---------- Generic column helpers ----------
def _series(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    return df[name] if name in df.columns else None

def _col(df: pd.DataFrame, ticker: str, field: str) -> Optional[pd.Series]:
    return _series(df, f"{ticker}_{field}")

def _value_at_or_pad(series: Optional[pd.Series], when: pd.Timestamp) -> Optional[float]:
    if series is None or series.empty:
        return None
    try:
        if when in series.index:
            v = series.loc[when]
            return None if pd.isna(v) else float(v)
        s = series.dropna()
        if s.empty:
            return None
        try:
            v = s.asof(when)  # type: ignore[attr-defined]
        except Exception:
            s = s.loc[s.index <= when]
            if s.empty:
                return None
            v = s.iloc[-1]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None

# ---------- Underlying price reads ----------
def get_underlying_price(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    s = _col(df, ticker, "PX_LAST")
    return _value_at_or_pad(s, date)

def get_price_on_exit(ticker: str, exit_date: pd.Timestamp, df: pd.DataFrame) -> Optional[float]:
    return get_underlying_price(ticker, exit_date, df)

# ---------- Implied vol (3M) ----------
def _iv_col_suffix(direction: Literal["long", "short"]) -> str:
    return "3MO_CALL_IMP_VOL" if direction == "long" else "3MO_PUT_IMP_VOL"

def _normalize_iv_unit(iv: Optional[float]) -> Optional[float]:
    if iv is None or not np.isfinite(iv) or iv <= 0.0:
        return None
    x = float(iv)
    # Use same threshold as _normalize_iv for consistency
    if x > 2.0:  # e.g., 25 -> 0.25
        x = x / 100.0
    return float(min(max(x, CLAMP_SIGMA_MIN), CLAMP_SIGMA_MAX))

def get_entry_iv_3m(ticker: str, date: pd.Timestamp, direction: Literal["long", "short"], df: pd.DataFrame) -> Optional[float]:
    suffix = _iv_col_suffix(direction)
    series = _col(df, ticker, suffix)
    iv = _value_at_or_pad(series, date)
    return _normalize_iv_unit(iv)

def get_exit_iv_3m(
    ticker: str,
    exit_date: pd.Timestamp,
    direction: Literal["long", "short"],
    df: pd.DataFrame,
    fallback_sigma: Optional[float]
) -> Optional[float]:
    suffix = _iv_col_suffix(direction)
    series = _col(df, ticker, suffix)
    iv = _value_at_or_pad(series, exit_date)
    if iv is None:
        iv = fallback_sigma
    return _normalize_iv_unit(iv)

def has_required_iv_series(ticker: str, df: pd.DataFrame) -> bool:
    call_iv = f"{ticker}_3MO_CALL_IMP_VOL"
    put_iv = f"{ticker}_3MO_PUT_IMP_VOL"
    return (call_iv in df.columns) and (put_iv in df.columns)

# ---------- Map 3M IV to tenor T ----------
def _realized_vol_lookback(ticker: str, date: pd.Timestamp, df: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    px = _col(df, ticker, "PX_LAST")
    if px is None or px.empty:
        return None
    s = px.dropna()
    s = s.loc[s.index <= date]
    if len(s) < lookback + 2:
        return None
    rets = np.log(s).diff().dropna().tail(lookback)
    if rets.empty or rets.std() <= 0.0:
        return None
    return float(rets.std() * math.sqrt(TRADING_DAYS_PER_YEAR))

def map_sigma_from_3m_to_T(
    sigma_3m: Optional[float],
    T_bd: int,
    ticker: str,
    date: pd.Timestamp,
    df: pd.DataFrame
) -> Optional[float]:
    """sigma_T = alpha * (sigma_3m * (T/REF_3M_BD)^beta) + (1 - alpha) * realized_vol_20d"""
    if sigma_3m is None or not np.isfinite(sigma_3m) or sigma_3m <= 0.0:
        return None

    alpha = float(getattr(settings.options, "iv_map_alpha", 0.7))
    beta = float(getattr(settings.options, "power_law_beta", -0.15))
    T = max(int(T_bd), 1)

    term_scaled = float(sigma_3m) * (T / REF_3M_BD) ** beta
    rv = _realized_vol_lookback(ticker, date, df, lookback=20)
    if rv is None:
        rv = term_scaled

    sigma_T = alpha * term_scaled + (1.0 - alpha) * rv
    return float(min(max(sigma_T, CLAMP_SIGMA_MIN), CLAMP_SIGMA_MAX))

# ---------- Risk-free ----------
def get_risk_free_rate(date: pd.Timestamp, df: Optional[pd.DataFrame] = None) -> float:
    mode = getattr(settings.options, "risk_free_rate_mode", "constant")
    if mode == "constant" or df is None:
        return float(getattr(settings.options, "constant_r", 0.0))

    candidates = ["USGG3M Index", "USGG6M Index", "USGG1YR Index", "USGG2YR Index"]
    for tk in candidates:
        s = _col(df, tk, "PX_LAST") or _series(df, tk)  # tolerate raw columns
        if s is None:
            continue
        val = _value_at_or_pad(s, date)
        if val is None:
            continue
        r = float(val)
        if r > 1.0:
            r = r / 100.0
        return r

    return float(getattr(settings.options, "constant_r", 0.0))

# ---------- NEW: 30D ATM and 1M Delta Smile IV Readers ----------

def _normalize_iv(value: Optional[float]) -> Optional[float]:
    """Normalize IV to decimal form and clamp to reasonable bounds."""
    if value is None or not math.isfinite(value):
        return None
    
    # Convert percentage to decimal if needed (e.g., 25 -> 0.25)
    # Use threshold of 2.0 since IV rarely exceeds 200% in decimal form
    if value > 2.0:  # Assume it's in percentage form
        value = value / 100.0
    
    # Clamp to reasonable bounds only for truly extreme values
    if value < CLAMP_SIGMA_MIN:
        return CLAMP_SIGMA_MIN
    elif value > CLAMP_SIGMA_MAX:
        return CLAMP_SIGMA_MAX
    
    return float(value)

def get_30d_atm_iv(ticker: str, date: pd.Timestamp, df: pd.DataFrame, direction: Literal["long", "short"]) -> Optional[float]:
    """Get 30D ATM implied volatility."""
    suffix = "CALL_IMP_VOL_30D" if direction == "long" else "PUT_IMP_VOL_30D"
    s = _col(df, ticker, suffix)
    raw_value = _value_at_or_pad(s, date)
    return _normalize_iv(raw_value)

def get_1m_smile_iv(
    ticker: str, 
    date: pd.Timestamp, 
    df: pd.DataFrame, 
    delta_bucket: str
) -> Optional[float]:
    """Get 1M delta smile implied volatility for specific bucket."""
    # Map delta bucket to column suffix
    bucket_mapping = {
        'CALL_40D': '1M_CALL_IMP_VOL_40DELTA_DFLT',
        'CALL_25D': '1M_CALL_IMP_VOL_25DELTA_DFLT', 
        'CALL_10D': '1M_CALL_IMP_VOL_10DELTA_DFLT',
        'PUT_40D': '1M_PUT_IMP_VOL_40DELTA_DFLT',
        'PUT_25D': '1M_PUT_IMP_VOL_25DELTA_DFLT',
        'PUT_10D': '1M_PUT_IMP_VOL_10DELTA_DFLT'
    }
    
    if delta_bucket not in bucket_mapping:
        return None
    
    suffix = bucket_mapping[delta_bucket]
    s = _col(df, ticker, suffix)
    raw_value = _value_at_or_pad(s, date)
    return _normalize_iv(raw_value)

def interpolate_smile_iv(
    ticker: str,
    date: pd.Timestamp, 
    df: pd.DataFrame,
    target_delta: float,
    option_type: Literal["call", "put"]
) -> Optional[float]:
    """
    Interpolate 1M smile IV for a target delta between available buckets.
    Works with whatever columns are available (need at least 2).
    """
    if option_type == "call":
        # Check what call deltas are available
        all_call_deltas = [0.10, 0.25, 0.40]
        all_call_buckets = ['CALL_10D', 'CALL_25D', 'CALL_40D']
    else:
        # Check what put deltas are available
        all_put_deltas = [-0.40, -0.25, -0.10]
        all_put_buckets = ['PUT_40D', 'PUT_25D', 'PUT_10D']
    
    # Build list of actually available delta/IV pairs
    available_pairs = []
    
    if option_type == "call":
        for delta, bucket in zip(all_call_deltas, all_call_buckets):
            iv = get_1m_smile_iv(ticker, date, df, bucket)
            if iv is not None:
                available_pairs.append((delta, iv))
    else:
        for delta, bucket in zip(all_put_deltas, all_put_buckets):
            iv = get_1m_smile_iv(ticker, date, df, bucket)
            if iv is not None:
                available_pairs.append((delta, iv))
    
    # Need at least 2 points for interpolation
    if len(available_pairs) < 2:
        return None
    
    # Sort by delta for proper interpolation
    available_pairs.sort(key=lambda x: x[0])
    deltas = [pair[0] for pair in available_pairs]
    ivs = [pair[1] for pair in available_pairs]
    
    # Handle extrapolation (use nearest point)
    if option_type == "call":
        if target_delta <= deltas[0]:
            return ivs[0]
        elif target_delta >= deltas[-1]:
            return ivs[-1]
    else:  # put
        if target_delta >= deltas[0]:  # Less negative than smallest
            return ivs[0]
        elif target_delta <= deltas[-1]:  # More negative than largest
            return ivs[-1]
    
    # Linear interpolation between available points
    for i in range(len(deltas) - 1):
        d1, d2 = deltas[i], deltas[i + 1]
        if option_type == "call":
            if d1 <= target_delta <= d2:
                weight = (target_delta - d1) / (d2 - d1)
                return ivs[i] * (1 - weight) + ivs[i + 1] * weight
        else:  # put
            if d2 <= target_delta <= d1:
                weight = (target_delta - d2) / (d1 - d2)
                return ivs[i + 1] * (1 - weight) + ivs[i] * weight
    
    return None

def check_new_iv_columns_available(ticker: str, date: pd.Timestamp, df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check if new IV columns are available for this ticker/date.
    Returns (sufficient_available, missing_columns).
    Now requires only 30D ATM + at least 2 delta smile columns for each side.
    """
    # Core required columns (30D ATM)
    core_required = ['CALL_IMP_VOL_30D', 'PUT_IMP_VOL_30D']
    
    # Smile columns (need at least 2 for each side to enable interpolation)
    call_smile_options = ['1M_CALL_IMP_VOL_10DELTA_DFLT', '1M_CALL_IMP_VOL_25DELTA_DFLT', '1M_CALL_IMP_VOL_40DELTA_DFLT']
    put_smile_options = ['1M_PUT_IMP_VOL_40DELTA_DFLT', '1M_PUT_IMP_VOL_25DELTA_DFLT', '1M_PUT_IMP_VOL_10DELTA_DFLT']
    
    missing = []
    
    # Check core 30D ATM columns
    for col in core_required:
        full_col = f"{ticker}_{col}"
        if full_col not in df.columns:
            missing.append(full_col)
        else:
            s = df[full_col]
            value = _value_at_or_pad(s, date)
            if value is None:
                missing.append(f"{full_col} (no data for {date})")
    
    # Check smile columns - need at least 2 available for each side
    call_smile_available = []
    put_smile_available = []
    
    for col in call_smile_options:
        full_col = f"{ticker}_{col}"
        if full_col in df.columns:
            s = df[full_col]
            value = _value_at_or_pad(s, date)
            if value is not None:
                call_smile_available.append(col)
    
    for col in put_smile_options:
        full_col = f"{ticker}_{col}"
        if full_col in df.columns:
            s = df[full_col]
            value = _value_at_or_pad(s, date)
            if value is not None:
                put_smile_available.append(col)
    
    # Add missing smile info
    if len(call_smile_available) < 2:
        missing.append(f"Insufficient call smile columns (need 2, have {len(call_smile_available)})")
    if len(put_smile_available) < 2:
        missing.append(f"Insufficient put smile columns (need 2, have {len(put_smile_available)})")
    
    # Success if we have 30D ATM + at least 2 smile columns per side
    sufficient = len(missing) == 0
    
    return sufficient, missing
