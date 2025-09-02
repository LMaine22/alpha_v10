# alpha_discovery/options/market.py
from __future__ import annotations
from typing import Optional, Literal
import math
import numpy as np
import pandas as pd

from ..config import settings
from .models import (
    CLAMP_SIGMA_MIN, CLAMP_SIGMA_MAX, TRADING_DAYS_PER_YEAR, REF_3M_BD
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
