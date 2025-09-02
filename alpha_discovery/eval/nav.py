
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Optional, List

# Accept both your artifact names and generic names
_PNL_CANDIDATES: Sequence[str] = (
    "pnl_dollars", "pnl$", "pnl", "realized_pnl", "pnl_net", "profit", "profit_dollars"
)
_EXIT_DATE_CANDIDATES: Sequence[str] = ("exit_time", "exit_date", "exit_dt", "exit")
_ENTRY_DATE_CANDIDATES: Sequence[str] = ("entry_time", "trigger_date", "entry_date", "entry_dt", "entry")

def _coerce_dt(x: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(x):
        return x.dt.normalize()
    try:
        return pd.to_datetime(x, errors="coerce").dt.normalize()
    except Exception:
        return pd.to_datetime(x.astype(str), errors="coerce").dt.normalize()

def _find_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        lo = name.lower()
        if lo in cols_lower:
            return cols_lower[lo]
    return None

def get_entry_series(ledger: pd.DataFrame) -> pd.Series:
    if ledger is None or len(ledger) == 0:
        return pd.Series(dtype="datetime64[ns]")
    df = ledger.copy()
    col = _find_col(df, _ENTRY_DATE_CANDIDATES)
    if col is None:
        return pd.Series(dtype="datetime64[ns]")
    return _coerce_dt(df[col])

def get_exit_series(ledger: pd.DataFrame) -> pd.Series:
    if ledger is None or len(ledger) == 0:
        return pd.Series(dtype="datetime64[ns]")
    df = ledger.copy()
    col = _find_col(df, _EXIT_DATE_CANDIDATES)
    if col is None:
        return pd.Series(dtype="datetime64[ns]")
    return _coerce_dt(df[col])

def _pick_pnl_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in _PNL_CANDIDATES:
        if c in df.columns:
            return c
        lo = c.lower()
        if lo in cols_lower:
            return cols_lower[lo]
    raise KeyError(f"No PnL column found. Tried {_PNL_CANDIDATES}. Ledger has: {list(df.columns)}")

def nav_daily_returns_from_ledger(ledger: pd.DataFrame, base_capital: float) -> pd.Series:
    """Build daily returns using *realized* P&L on the exit date.
    Accepts columns: exit_time/exit_date/... and pnl_dollars/pnl/...
    """
    if ledger is None or len(ledger) == 0:
        return pd.Series(dtype="float64")
    df = ledger.copy()
    exit_col = _find_col(df, _EXIT_DATE_CANDIDATES)
    if exit_col is None:
        raise KeyError("Ledger must contain an exit date column (one of: "
                       f"{list(_EXIT_DATE_CANDIDATES)}). Got: {list(df.columns)}")
    pnl_col = _pick_pnl_col(df)
    df[exit_col] = _coerce_dt(df[exit_col])
    daily_pnl = df.groupby(exit_col, dropna=True)[pnl_col].sum().sort_index()
    if len(daily_pnl) == 0:
        return pd.Series(dtype="float64")
    idx = pd.bdate_range(start=daily_pnl.index.min(), end=daily_pnl.index.max(), freq="B")
    daily_pnl = daily_pnl.reindex(idx, fill_value=0.0)
    daily_ret = daily_pnl.astype("float64") / float(base_capital)
    daily_ret.name = "ret"
    return daily_ret

def median_holding_days(ledger: pd.DataFrame) -> float:
    if ledger is None or len(ledger) == 0:
        return 1.0
    ent = get_entry_series(ledger)
    exi = get_exit_series(ledger)
    if len(ent) == 0 or len(exi) == 0:
        return 1.0
    # Bus-day difference inclusive of both endpoints
    days = np.maximum(1, np.busday_count(ent.values.astype("datetime64[D]"),
                                         exi.values.astype("datetime64[D]")) + 1)
    med = float(np.median(days)) if len(days) else 1.0
    return max(1.0, med)

def sharpe(ret: pd.Series) -> float:
    if ret is None or len(ret) < 2:
        return 0.0
    m = float(ret.mean())
    s = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
    if s == 0.0:
        return 0.0
    return (m / s) * np.sqrt(252.0)

def sortino_like_ewma_scalar(ret: pd.Series, halflife_days: float) -> float:
    if ret is None or len(ret) < 2:
        return 0.0
    halflife_days = max(2.0, float(halflife_days))
    neg = ret.copy()
    neg[neg > 0] = 0.0
    mu = ret.ewm(halflife=halflife_days, adjust=False).mean().iloc[-1]
    neg_std = neg.ewm(halflife=halflife_days, adjust=False).std(bias=False).iloc[-1]
    if neg_std is None or np.isnan(neg_std) or neg_std == 0.0:
        return 0.0
    return float(mu / (abs(neg_std) + 1e-12) * np.sqrt(252.0))

def max_drawdown_pct(ret: pd.Series) -> float:
    if ret is None or len(ret) == 0:
        return 0.0
    nav = (1.0 + ret).cumprod()
    peak = nav.cummax()
    dd = (nav / peak) - 1.0
    return float(dd.min())

def short_window_activity_lambda(ledger: pd.DataFrame, window_days: int) -> float:
    if ledger is None or len(ledger) == 0:
        return 0.0
    ent = get_entry_series(ledger).sort_values()
    if len(ent) == 0:
        return 0.0
    end = ent.max()
    start = pd.bdate_range(end=end, periods=max(1, int(window_days)), freq="B")[0]
    n_trig = int(((ent >= start) & (ent <= end)).sum())
    days = max(1, np.busday_count(start.date(), end.date()) + 1)
    return float(n_trig) / float(days)
