# alpha_discovery/eval/selection_core.py
"""
Core selection logic (split out from selection.py).
Re-exported by alpha_discovery/eval/selection.py so imports remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from ..config import settings
from . import metrics as M
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The behavior of array concatenation with empty entries is deprecated"
)

# =========================
# Cached daily series per (ticker, horizon) FOR A GIVEN LEDGER
# =========================
_DAILY_RETURNS_MI: Optional[pd.Series] = None
_DAILY_RETURNS_TOKEN: Optional[int] = None  # id() of the ledger used to build the cache


def _ensure_daily_cache(ledger: pd.DataFrame) -> None:
    """Build the per-(ticker, horizon) daily mean pnl_pct cache once per ledger object."""
    global _DAILY_RETURNS_MI, _DAILY_RETURNS_TOKEN
    if ledger is None or ledger.empty:
        _DAILY_RETURNS_MI = None
        _DAILY_RETURNS_TOKEN = None
        return

    token = id(ledger)
    if _DAILY_RETURNS_MI is not None and _DAILY_RETURNS_TOKEN == token:
        return  # cache is valid for this ledger object

    df = ledger.copy()

    # Ensure expected columns exist
    needed = ["ticker", "horizon_days", "trigger_date", "pnl_pct"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Ledger is missing required columns for selection cache: {missing}")

    # Normalize dtypes for a consistent MultiIndex
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce").astype("Int64")
    df["trigger_date"] = pd.to_datetime(df["trigger_date"]).dt.normalize()

    # Aggregate once for the entire ledger (mean by trigger_date)
    grouped = (
        df.groupby(["ticker", "horizon_days", "trigger_date"])["pnl_pct"]
          .mean()
          .sort_index()
    )

    # Force float dtype, drop NaNs
    _DAILY_RETURNS_MI = pd.to_numeric(grouped, errors="coerce").dropna()
    _DAILY_RETURNS_TOKEN = token


def _daily_returns_for_pair(ledger: pd.DataFrame, ticker: str, horizon: int) -> pd.Series:
    """
    Build a daily return series (by trigger_date) for a (ticker, horizon) pair
    using the precomputed cache (identical aggregation to the original code).
    """
    if ledger is None or ledger.empty:
        return pd.Series(dtype=float)

    _ensure_daily_cache(ledger)
    if _DAILY_RETURNS_MI is None:
        return pd.Series(dtype=float)

    h = int(horizon)
    try:
        s = _DAILY_RETURNS_MI.loc[(ticker, h)]
    except KeyError:
        return pd.Series(dtype=float)

    # s is indexed by trigger_date already; ensure numeric/clean/sorted
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index().astype(float)
    return s


def _compute_metrics_from_returns(daily_returns: pd.Series) -> Dict[str, float]:
    """
    Compute metrics consistent with eval.metrics, but starting
    from a daily returns series rather than a trade_ledger.
    """
    if daily_returns is None or daily_returns.empty:
        return {}

    # Winsorize for robustness (using reporting.trimmed_alpha)
    alpha = float(getattr(settings.reporting, "trimmed_alpha", 0.05))
    alpha = min(max(alpha, 0.0), 0.2)  # clamp
    cleaned = M.winsorize(daily_returns, lower_q=alpha, upper_q=1.0 - alpha)
    if cleaned.empty:
        return {}

    # Support = number of daily observations (pre-winsorize count)
    support = float(len(daily_returns))

    sharpe_stats = M.block_bootstrap_sharpe(cleaned)  # returns dict with sharpe_median/lb/ub
    omega = M.calculate_omega_ratio(cleaned)
    max_dd = M.calculate_max_drawdown(cleaned)

    # Annualized return if sufficient length (robust)
    min_days_for_annualization = 126
    if len(cleaned) >= min_days_for_annualization:
        one_plus = 1.0 + cleaned
        if (one_plus > 0).all():
            # geometric mean via log returns (safe, no invalid power)
            log_mean = np.log1p(cleaned).mean() * 252.0
            # Clip to prevent overflow in exp operation
            log_mean_clipped = np.clip(log_mean, -20, 20)
            annualized_return = float(np.exp(log_mean_clipped) - 1.0)
        else:
            # fallback: arithmetic scaling if any (1+r_t) <= 0
            annualized_return = float(cleaned.mean() * 252.0)
    else:
        annualized_return = 0.0

    out = {
        "support": support,
        "annualized_return": float(annualized_return),
        "volatility": float(cleaned.std() * np.sqrt(252.0)),
        "max_drawdown": float(max_dd),
        "omega_ratio": float(omega),
        "mean_return": float(cleaned.mean()),
        "median_return": float(cleaned.median()),
    }
    out.update({k: float(v) for k, v in sharpe_stats.items()})  # sharpe_median, sharpe_lb, sharpe_ub

    # Replace NaNs/infs defensively
    return {k: (0.0 if (v is None or not np.isfinite(v)) else float(v)) for k, v in out.items()}


def _metric_key(metrics: Dict[str, float], name: str) -> float:
    """Safely pull a metric from dict; default to 0.0 if missing or non-finite."""
    v = metrics.get(name, 0.0)
    try:
        v = float(v)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v


def _compare_metric_tuple(metrics: Dict[str, float],
                          primary: str,
                          tiebreakers: List[str]) -> Tuple[float, ...]:
    """
    Build a sorting tuple (descending) for robust ranking:
      (primary, tie1, tie2, ...)
    """
    vals = [_metric_key(metrics, primary)]
    for tb in tiebreakers:
        vals.append(_metric_key(metrics, tb))
    return tuple(vals)


# =========================
# Public data structures
# =========================

@dataclass
class TickerBest:
    ticker: str
    horizon: int
    metrics: Dict[str, float]


# =========================
# Helpers for faster stepwise (logic-identical aggregation)
# =========================

def _aggregate_daily_from_series(raw_vals: pd.Series) -> pd.Series:
    """
    Given a Series indexed by trigger_date containing per-trade pnl_pct values,
    aggregate to a daily series using the same robust aggregator as portfolio_daily_returns.
    """
    if raw_vals is None or raw_vals.empty:
        return pd.Series(dtype=float)

    agg = str(getattr(settings.reporting, "robust_agg_metric", "median")).lower()
    by_day = raw_vals.groupby(level=0)
    out = (by_day.median() if agg == "median" else by_day.mean()).sort_index()
    return pd.to_numeric(out, errors="coerce").dropna()


def _precompute_candidate_series(full_ledger: pd.DataFrame,
                                 ranked_candidates: List[TickerBest]) -> Dict[Tuple[str, int], pd.Series]:
    """
    Precompute raw per-trade return series (indexed by trigger_date) for each candidate.
    This avoids repeatedly filtering DataFrames and reboxing columns in the stepwise loop.
    """
    if full_ledger is None or full_ledger.empty or not ranked_candidates:
        return {}

    df = full_ledger.copy()
    df["trigger_date"] = pd.to_datetime(df["trigger_date"]).dt.normalize()
    df["horizon_days"] = pd.to_numeric(df["horizon_days"], errors="coerce").astype("Int64")

    wanted = {(c.ticker, int(c.horizon)) for c in ranked_candidates}
    key = pd.MultiIndex.from_tuples(wanted, names=["ticker", "horizon_days"])
    mask = pd.MultiIndex.from_arrays([df["ticker"], df["horizon_days"]]).isin(key)
    df = df.loc[mask, ["ticker", "horizon_days", "trigger_date", "pnl_pct"]]

    out: Dict[Tuple[str, int], pd.Series] = {}
    for (tk, h), sub in df.groupby(["ticker", "horizon_days"], sort=False):
        s = pd.to_numeric(sub.set_index("trigger_date")["pnl_pct"], errors="coerce").dropna()
        out[(tk, int(h))] = s.sort_index()
    return out


# =========================
# Public API
# =========================

def score_ticker_horizon(ledger: pd.DataFrame, ticker: str, horizon: int) -> Dict[str, float]:
    """Score a single (ticker, horizon) pair."""
    returns = _daily_returns_for_pair(ledger, ticker, horizon)
    return _compute_metrics_from_returns(returns)


def select_best_horizon_per_ticker(
    ledger: pd.DataFrame,
    min_support_per_ticker: Optional[int] = None,
) -> List[TickerBest]:
    """Pick best horizon per ticker by primary metric with tie-breakers and gates."""
    if ledger is None or ledger.empty:
        return []

    sel_cfg = getattr(settings, "selection", None)
    metric_primary = getattr(sel_cfg, "metric_primary", "sharpe_lb")
    metric_tiebreakers = list(getattr(sel_cfg, "metric_tiebreakers", ["omega_ratio", "support"]))
    per_ticker_min_sharpe_lb = getattr(sel_cfg, "per_ticker_min_sharpe_lb", None)
    per_ticker_min_omega = getattr(sel_cfg, "per_ticker_min_omega", None)
    if min_support_per_ticker is None:
        min_support_per_ticker = getattr(sel_cfg, "min_support_per_ticker",
                                         getattr(settings.validation, "min_initial_support", 10))

    best_by_ticker: Dict[str, TickerBest] = {}
    tickers = sorted(ledger["ticker"].dropna().unique().tolist())
    horizons = sorted(ledger["horizon_days"].dropna().unique().astype(int).tolist())

    _ensure_daily_cache(ledger)

    for tk in tickers:
        best_tuple: Optional[Tuple[float, ...]] = None
        best_choice: Optional[TickerBest] = None

        for h in horizons:
            m = score_ticker_horizon(ledger, tk, h)
            if not m:
                continue
            if m.get("support", 0.0) < float(min_support_per_ticker):
                continue
            if per_ticker_min_sharpe_lb is not None and m.get("sharpe_lb", 0.0) < float(per_ticker_min_sharpe_lb):
                continue
            if per_ticker_min_omega is not None and m.get("omega_ratio", 0.0) < float(per_ticker_min_omega):
                continue

            tup = _compare_metric_tuple(m, metric_primary, metric_tiebreakers)
            if (best_tuple is None) or (tup > best_tuple):
                best_tuple = tup
                best_choice = TickerBest(ticker=tk, horizon=int(h), metrics=m)

        if best_choice is not None:
            best_by_ticker[tk] = best_choice

    candidates = list(best_by_ticker.values())
    candidates.sort(key=lambda x: _compare_metric_tuple(x.metrics, metric_primary, metric_tiebreakers), reverse=True)
    return candidates


def stepwise_select_portfolio(
    full_ledger: pd.DataFrame,
    ranked_candidates: List[TickerBest],
) -> List[TickerBest]:
    """Greedy stepwise assembly with identical aggregation semantics."""
    if not ranked_candidates:
        return []

    sel_cfg = getattr(settings, "selection", None)
    delta_sharpe_req = float(getattr(sel_cfg, "stepwise_min_delta_sharpe_lb", 0.0))
    delta_omega_req = float(getattr(sel_cfg, "stepwise_min_delta_omega", 0.0))
    max_names = getattr(sel_cfg, "max_tickers_in_portfolio", None)
    primary_name = getattr(sel_cfg, "metric_primary", "sharpe_lb")

    cand_series = _precompute_candidate_series(full_ledger, ranked_candidates)

    current_raw = pd.Series(dtype=float)
    final_selection: List[TickerBest] = []
    best_primary = -np.inf
    best_omega = -np.inf

    for cand in ranked_candidates:
        s_cand = cand_series.get((cand.ticker, int(cand.horizon)), pd.Series(dtype=float))

        parts = []
        if not current_raw.empty:
            parts.append(current_raw)
        if not s_cand.empty:
            parts.append(s_cand)
        trial_raw = (pd.concat(parts, ignore_index=False) if parts else pd.Series(dtype=float)).sort_index()

        trial_daily = _aggregate_daily_from_series(trial_raw)
        temp_metrics = _compute_metrics_from_returns(trial_daily)

        temp_primary = float(temp_metrics.get(primary_name, 0.0))
        temp_omega = float(temp_metrics.get("omega_ratio", 0.0))

        is_first = (len(final_selection) == 0)
        is_improvement = ((temp_primary - best_primary) >= delta_sharpe_req) and \
                         ((temp_omega - best_omega) >= delta_omega_req)

        if is_first or is_improvement:
            current_raw = trial_raw
            best_primary = temp_primary
            best_omega = temp_omega
            final_selection.append(cand)
            if max_names is not None and len(final_selection) >= int(max_names):
                break
        else:
            break

    return final_selection


def filter_ledger_to_selection(ledger: pd.DataFrame, selection: List[TickerBest]) -> pd.DataFrame:
    """Keep only rows for (ticker, horizon) pairs present in selection."""
    if ledger is None or ledger.empty or not selection:
        return pd.DataFrame(columns=ledger.columns)

    pairs = {(sel.ticker, int(sel.horizon)) for sel in selection}
    df = ledger.copy()
    df["horizon_days"] = df["horizon_days"].astype(int)

    key = pd.MultiIndex.from_tuples(pairs, names=["ticker", "horizon_days"])
    mask = pd.MultiIndex.from_arrays([df["ticker"], df["horizon_days"]]).isin(key)
    out = df[mask].copy()
    out.sort_values(by=["trigger_date", "ticker", "horizon_days"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def portfolio_daily_returns(filtered_ledger: pd.DataFrame) -> pd.Series:
    """Aggregate pnl_pct by trigger_date using configured robust aggregator."""
    if filtered_ledger is None or filtered_ledger.empty:
        return pd.Series(dtype=float)

    agg = str(getattr(settings.reporting, "robust_agg_metric", "median")).lower()
    by_day = filtered_ledger.groupby("trigger_date")["pnl_pct"]
    s = (by_day.median() if agg == "median" else by_day.mean()).sort_index()
    return pd.to_numeric(s, errors="coerce").dropna()


def portfolio_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """Compute metrics on the portfolio daily return series."""
    return _compute_metrics_from_returns(daily_returns)


def assemble_portfolio_stepwise(options_ledger: pd.DataFrame) -> Dict[str, object]:
    """One-shot API used by the GA layer."""
    best = select_best_horizon_per_ticker(options_ledger)
    chosen = stepwise_select_portfolio(options_ledger, best)
    filt = filter_ledger_to_selection(options_ledger, chosen)
    daily = portfolio_daily_returns(filt)
    met = portfolio_metrics(daily)
    return {
        "best_per_ticker": best,
        "final_selection": chosen,
        "portfolio_ledger": filt,
        "portfolio_daily": daily,
        "portfolio_metrics": met,
    }


def selection_summary(selection: List[TickerBest]) -> Dict[str, object]:
    """Produce compact metadata for reporting, including a specialty tag."""
    if not selection:
        return {"chosen_tickers": "", "chosen_horizons": {}, "specialist_type": "none"}

    chosen_tickers = [x.ticker for x in selection]
    chosen_horizons = {x.ticker: x.horizon for x in selection}

    if len(chosen_tickers) == 1:
        specialist_type = "solo_specialist"
    elif len(chosen_tickers) < len(getattr(settings.data, "tradable_tickers", chosen_tickers)):
        specialist_type = "cluster_specialist"
    else:
        specialist_type = "generalist"

    return {
        "chosen_tickers": ", ".join(chosen_tickers),
        "chosen_horizons": chosen_horizons,
        "specialist_type": specialist_type,
    }
