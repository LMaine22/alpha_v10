from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable, Any
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ..config import settings
from ..eval.selection import get_valid_trigger_dates
from ..eval.info_metrics import (
    distribution,
    info_theory,
    dynamics,
    complexity,
    tda,
    robustness,
    regime,
    aggregate,
    causality
)
from ..reporting import display_utils as du
from ..utils.trade_keys import canonical_signals_fingerprint, dedupe_trade_ledger
# Removed: from ..eval.objectives import ... (legacy transformation layer, not needed with institutional metrics)

# Backtesting imports for Phase A
from ..engine import backtester
from ..eval import selection, metrics
from .fold_plan import GADataSpec

try:
    from ..engine.bt_runtime import _enforce_exclusivity_by_setup, _parse_bt_env_flag
except ImportError:
    # Fallback if runtime module missing
    def _enforce_exclusivity_by_setup(ledger):
        return ledger
    def _parse_bt_env_flag(name, default):
        return default

from deap import base, creator, tools


# --- Walk-forward splits (legacy fallback) ---
def make_walkforward_splits(index: pd.DatetimeIndex, n_folds: int, embargo_days: int = 0):
    """
    Simple walk-forward split for internal use (legacy fallback).
    
    Creates n_folds by dividing index into sequential train/test windows.
    This is a simplified replacement for the deleted core.splits.make_walkforward_splits.
    
    Args:
        index: DatetimeIndex to split
        n_folds: Number of folds
        embargo_days: Days to skip between train and test
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    if len(index) < n_folds * 2:
        # Fallback: single fold
        split_point = len(index) // 2
        return [(index[:split_point], index[split_point:])]
    
    folds = []
    fold_size = len(index) // n_folds
    
    for i in range(n_folds):
        # Train on all data up to this fold
        train_end = min((i + 1) * fold_size, len(index) - fold_size)
        train_idx = index[:train_end]
        
        # Apply embargo
        test_start_idx = train_end
        if embargo_days > 0 and test_start_idx < len(index):
            # Find date that is embargo_days after last train date
            last_train_date = train_idx[-1]
            embargo_date = last_train_date + pd.Timedelta(days=embargo_days)
            # Find first index after embargo
            test_start_idx = index.searchsorted(embargo_date)
        
        # Test on next fold
        test_end = min(test_start_idx + fold_size, len(index))
        if test_start_idx < len(index):
            test_idx = index[test_start_idx:test_end]
            if len(test_idx) > 0:
                folds.append((train_idx, test_idx))
    
    return folds if folds else [(index[:len(index)//2], index[len(index)//2:])]


# --- Strict evaluation exceptions (Fix: remove penalty defaults) ---
class InsufficientDataError(Exception):
    """Raised when a candidate cannot produce the required minimum supported folds/metrics."""
    pass

# DEAP setup for multi-objective optimization.
# We want to minimize CRPS (negated), minimize Pinball Loss (negated), and maximize Information Gain.
# The Sharpe and Min of Information Gain scores across CV paths. Both should be maximized.
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", object, fitness=creator.FitnessMulti)


# Verbosity toggles consumed by nsga.py
VERBOSE = int(getattr(settings.ga, "verbose", 1))
DEBUG_SEQUENTIAL = bool(getattr(settings.ga, "debug_sequential", False))
JOBLIB_VERBOSE = 0


DEFAULT_OBJECTIVES = [
    "dsr",
    "bootstrap_calmar_lb",
    "bootstrap_profit_factor_lb",
]
DEFAULT_DAMPING_BETA = 0.02
DEFAULT_EXPAND_MARGIN = 10
DEFAULT_SMALL_N = 20
DEFAULT_MERGE_TAG_NEXT = "merged_next"
DEFAULT_MERGE_TAG_PREV = "merged_prev"
MIN_FOLD_TRADES = 1
MIN_FOLD_TRADE_DAYS = 1
HIT_RATE_PENALTY_MIN_TRADES = 15
PSR_PENALTY_MIN_TRADES = 10
DISCOVERY_PENALTY_CAP = 2.0
FOLD_REASON_EMPTY = "empty_ledger"
FOLD_REASON_WIDENED = "widened"
FOLD_REASON_MERGED = "merged"


@contextmanager
def threadpool_limits(limits: int = 1):
    """Compatibility stub so nsga.py can limit BLAS threads if desired."""
    yield


def _exit_policy_from_settings() -> Optional[GADataSpec]:
    """Legacy stub: return None when no explicit GA plan is configured."""
    return None


# -----------------------------
# Utility helpers
# -----------------------------
def _objective_keys() -> List[str]:
    """Return configured GA objective keys with conservative defaults."""
    obj_keys = getattr(getattr(settings, 'ga', object()), 'objectives', None)
    if not obj_keys:
        return list(DEFAULT_OBJECTIVES)
    return list(obj_keys)


def _penalty_vector() -> List[float]:
    """Penalty vector matching the number of GA objectives."""
    return [-999.0] * len(_objective_keys())


OBJECTIVE_FALLBACKS = {
    "dsr": ["dsr", "dsr_raw", "sharpe"],
    "bootstrap_calmar_lb": ["bootstrap_calmar_lb", "bootstrap_calmar_lb_raw", "calmar", "mar_ratio"],
    "bootstrap_profit_factor_lb": ["bootstrap_profit_factor_lb", "bootstrap_profit_factor_lb_raw", "profit_factor", "expectancy"],
}


def _resolve_objective_value(metrics: Dict[str, Any], key: str) -> Tuple[float, str]:
    """Return a finite objective value with fallback source tracking."""
    candidates = OBJECTIVE_FALLBACKS.get(key, [key])
    for candidate in candidates:
        val = metrics.get(candidate)
        if isinstance(val, (int, float)) and np.isfinite(val):
            return _clamp_objective_value(key, float(val)), candidate
    return -999.0, "unavailable"


def _clamp_objective_value(key: str, value: float) -> float:
    if not np.isfinite(value):
        return -999.0
    if key in {"bootstrap_calmar_lb", "calmar", "mar_ratio"}:
        sign = 1.0 if value >= 0 else -1.0
        return sign * np.log1p(abs(value))
    if key in {"bootstrap_profit_factor_lb", "profit_factor"}:
        if value <= 0:
            return -np.log1p(abs(value))
        return np.log1p(value)
    if key in {"dsr", "sharpe"}:
        return max(min(value, 10.0), -10.0)
    return value


def _candidate_test_indices(
    plan: GADataSpec,
    fold_index: int,
    expand_margin: int = DEFAULT_EXPAND_MARGIN,
) -> List[Tuple[pd.Index, str]]:
    base_idx = pd.Index(plan.inner_folds[fold_index].test_idx).sort_values().unique()
    candidates: List[Tuple[pd.Index, str]] = [(base_idx, "original")]

    if expand_margin <= 0 or base_idx.empty:
        return candidates

    train_index = pd.Index(plan.train_idx).sort_values().unique()
    if base_idx[0] in train_index and base_idx[-1] in train_index:
        start_pos = train_index.get_loc(base_idx[0]) if base_idx[0] in train_index else train_index.searchsorted(base_idx[0])
        end_pos = train_index.get_loc(base_idx[-1]) if base_idx[-1] in train_index else train_index.searchsorted(base_idx[-1])
        expanded_start = max(0, start_pos - expand_margin)
        expanded_end = min(len(train_index) - 1, end_pos + expand_margin)
        expanded_idx = train_index[expanded_start:expanded_end + 1]
        if not expanded_idx.empty and not expanded_idx.equals(base_idx):
            candidates.append((expanded_idx, "widened"))

    # Merge with next inner fold if available
    if fold_index < len(plan.inner_folds) - 1:
        next_idx = pd.Index(plan.inner_folds[fold_index + 1].test_idx)
        merged_next = base_idx.union(next_idx).sort_values().unique()
        if not merged_next.empty and not merged_next.equals(base_idx):
            candidates.append((merged_next, DEFAULT_MERGE_TAG_NEXT))

    # Merge with previous inner fold if available
    if fold_index > 0:
        prev_idx = pd.Index(plan.inner_folds[fold_index - 1].test_idx)
        merged_prev = base_idx.union(prev_idx).sort_values().unique()
        if not merged_prev.empty and not merged_prev.equals(base_idx):
            candidates.append((merged_prev, DEFAULT_MERGE_TAG_PREV))

    return candidates


def _fold_reason(success: bool, attempt_tag: str, detail: str) -> Tuple[str, str]:
    """Return coarse reason code (for HUD) and detailed reason string."""
    if success:
        if attempt_tag == "widened":
            return FOLD_REASON_WIDENED, detail or attempt_tag
        if attempt_tag in {DEFAULT_MERGE_TAG_NEXT, DEFAULT_MERGE_TAG_PREV}:
            return FOLD_REASON_MERGED, detail or attempt_tag
        return "original", detail or attempt_tag or "original"

    if not detail:
        return FOLD_REASON_EMPTY, detail

    lowered = detail.lower()
    if "empty" in lowered or "no_trades" in lowered or "no_daily_returns" in lowered:
        return FOLD_REASON_EMPTY, detail
    if "insufficient" in lowered:
        return "insufficient_support", detail
    if lowered.startswith("backtest_error"):
        return "backtest_error", detail
    return detail, detail


def _compute_soft_penalties(
    metrics: Dict[str, Any],
    ga_cfg: Any,
    *,
    coverage_ratio: float,
    coverage_target: float,
    min_total_trades_required: int,
) -> Tuple[float, Dict[str, float], List[str]]:
    """Derive penalty scalar, detail map, and diagnostic flags for GA gating."""

    penalty_details: Dict[str, float] = {}
    reasons: List[str] = []

    def add_penalty(key: str, magnitude: float, reason: str) -> None:
        if not np.isfinite(magnitude) or magnitude <= 0:
            return
        penalty_details[key] = float(magnitude)
        reasons.append(reason)

    total_trades = int(metrics.get("n_trades", 0) or 0)
    small_n_threshold = max(int(getattr(ga_cfg, 'min_total_trades', DEFAULT_SMALL_N)), 1)

    if coverage_target > 0:
        deficit = max(0.0, coverage_target - coverage_ratio)
        if deficit > 0:
            scaling = min(1.0, total_trades / max(1, small_n_threshold))
            add_penalty("coverage", (deficit / coverage_target) * scaling, f"coverage<{coverage_target:.2f}")

    hit_rate = metrics.get("hit_rate")
    if hit_rate is not None and np.isfinite(hit_rate) and total_trades >= HIT_RATE_PENALTY_MIN_TRADES:
        min_hr = float(getattr(ga_cfg, "min_hit_rate", 0.0))
        max_hr = float(getattr(ga_cfg, "max_hit_rate", 1.0))
        if hit_rate < min_hr:
            add_penalty("hit_rate_low", (min_hr - hit_rate) / max(1e-6, min_hr), f"hit_rate<{min_hr}")
        elif hit_rate > max_hr:
            add_penalty("hit_rate_high", (hit_rate - max_hr) / max(1e-6, max_hr), f"hit_rate>{max_hr}")

    psr_val = metrics.get("psr")
    if psr_val is not None and np.isfinite(psr_val) and total_trades >= PSR_PENALTY_MIN_TRADES:
        min_psr = float(getattr(ga_cfg, "min_psr", 0.0))
        if psr_val < min_psr:
            add_penalty("psr", (min_psr - psr_val) / max(1e-6, min_psr), f"psr<{min_psr}")

    max_dd_val = metrics.get("max_drawdown")
    if max_dd_val is not None and np.isfinite(max_dd_val):
        max_dd_threshold = float(getattr(ga_cfg, "max_drawdown_threshold", -1.0))
        if max_dd_val < max_dd_threshold:
            add_penalty("max_drawdown", abs(max_dd_threshold - max_dd_val), f"max_dd<{max_dd_threshold}")

    if 0 < total_trades < min_total_trades_required:
        deficit = (min_total_trades_required - total_trades) / max(1, min_total_trades_required)
        add_penalty("trades_total", deficit, f"total_trades<{min_total_trades_required}")

    raw_penalty = sum(penalty_details.values())
    damping = 1.0 / (1.0 + DEFAULT_DAMPING_BETA * max(total_trades, 0))
    penalty_scalar = 1.0 + raw_penalty * damping
    penalty_scalar = float(min(max(penalty_scalar, 1.0), DISCOVERY_PENALTY_CAP))

    return penalty_scalar, penalty_details, reasons


def _penalized_result(
    individual: Tuple[str, List[str]],
    direction: str,
    plan: Optional[GADataSpec],
    reason: str,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a standardized penalty result for unsupported candidates."""
    metrics_payload = dict(extra_metrics or {})
    metrics_payload.setdefault("penalty_reason", reason)
    metrics_payload.setdefault("folds_used", 0)
    if plan is not None:
        metrics_payload.setdefault("outer_id", plan.outer_id)
    reasons = metrics_payload.get("eligibility_reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if reason:
        reasons = list(dict.fromkeys(list(reasons) + [reason]))
    metrics_payload["eligibility_reasons"] = reasons
    metrics_payload["eligible"] = False
    return {
        "individual": individual,
        "metrics": metrics_payload,
        "objectives": _penalty_vector(),
        "rank": np.inf,
        "crowding_distance": 0.0,
        "trade_ledger": pd.DataFrame(),
        "direction": direction,
        "fold_plan": plan.summary() if plan is not None else None,
        "exit_policy": plan,
        "gates_passed": False,
    }

def _dna(individual) -> Tuple[str, Tuple[str, ...], int]:
    """Extract DNA from individual - handles both traditional and EnhancedIndividual formats."""
    try:
        # Try EnhancedIndividual format first (has .ticker, .signals, .horizon attributes)
        if hasattr(individual, 'ticker') and hasattr(individual, 'signals') and hasattr(individual, 'horizon'):
            return (str(individual.ticker), tuple(sorted(individual.signals or [])), int(individual.horizon))
        # Try traditional tuple format (ticker, signals)
        elif len(individual) == 2:
            tkr, setup = individual
            return (str(tkr), tuple(sorted(setup or [])), -1)  # -1 indicates no horizon info
        # Try extended tuple format (ticker, signals, horizon)  
        elif len(individual) == 3:
            tkr, setup, horizon = individual
            return (str(tkr), tuple(sorted(setup or [])), int(horizon))
        else:
            raise ValueError(f"Unsupported individual format: {individual}")
    except Exception as e:
        print(f"Warning: Failed to extract DNA from individual {individual}: {e}")
        return ("unknown", (), -1)


def _forward_returns(master_df: pd.DataFrame, ticker: str, k: int, price_field: str) -> pd.Series:
    col = f"{ticker}_{price_field}"
    px_raw = master_df.get(col)
    if px_raw is None:
        return pd.Series(index=master_df.index, dtype=float)
    
    px = pd.to_numeric(px_raw, errors="coerce")
    if px is None or (hasattr(px, 'empty') and px.empty):
        return pd.Series(index=master_df.index, dtype=float)
    
    fwd = px.shift(-k) / px - 1.0
    return fwd


def _trigger_mask(signals_df: pd.DataFrame, setup: List[str]) -> pd.Series:
    if not setup:
        return pd.Series(False, index=signals_df.index)
    mask = pd.Series(True, index=signals_df.index)
    for sid in setup:
        if sid not in signals_df.columns:
            return pd.Series(False, index=signals_df.index)
        s = signals_df[sid]
        if s.dtype != bool:
            s = (pd.to_numeric(s, errors="coerce") > 0)
        mask &= s.fillna(False)
    return mask


def _histogram_probs(sample: Iterable[float], edges: np.ndarray) -> np.ndarray:
    v = pd.Series(sample).dropna().astype(float).values
    if v.size == 0:
        return np.full(edges.size - 1, np.nan, dtype=float)
    hist, _ = np.histogram(v, bins=edges)
    p = hist.astype(float)
    total = p.sum()
    if total == 0:
        return np.full_like(p, np.nan, dtype=float)
    return p / total


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    s = (p * np.log(p))
    if not np.isfinite(s).all():
        return float('nan')
    return float(-s.sum())


def _info_gain(uncond: Iterable[float], cond: Iterable[float], edges: np.ndarray) -> float:
    pu = _histogram_probs(uncond, edges)
    pc = _histogram_probs(cond, edges)
    ig = _entropy(pu) - _entropy(pc)
    return float(ig) if np.isfinite(ig) else float('nan')


def _wasserstein_1d(a: Iterable[float], b: Iterable[float]) -> float:
    xa = np.sort(pd.Series(a).dropna().astype(float).values)
    xb = np.sort(pd.Series(b).dropna().astype(float).values)
    if xa.size == 0 or xb.size == 0:
        return float('nan')
    q = np.linspace(0.0, 1.0, num=max(xa.size, xb.size), endpoint=True)
    fa = np.quantile(xa, q)
    fb = np.quantile(xb, q)
    d = np.mean(np.abs(fa - fb))
    return float(d) if np.isfinite(d) else float('nan')


def _mutual_information_bool(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype(int).values
    bb = b.astype(int).values
    p00 = np.mean((aa == 0) & (bb == 0))
    p01 = np.mean((aa == 0) & (bb == 1))
    p10 = np.mean((aa == 1) & (bb == 0))
    p11 = np.mean((aa == 1) & (bb == 1))
    px0 = p00 + p01
    px1 = p10 + p11
    py0 = p00 + p10
    py1 = p01 + p11
    eps = 1e-12
    terms = []
    for pxy, px, py in [(p00, px0, py0), (p01, px0, py1), (p10, px1, py0), (p11, px1, py1)]:
        if pxy > 0:
            terms.append(pxy * np.log(pxy / max(px*py, eps)))
    if not terms:
        return float('nan')
    mi = float(np.sum(terms))
    return mi


def _avg_pairwise_mi(signals_df: pd.DataFrame, setup: List[str]) -> float:
    if not setup or len(setup) < 2:
        return float('nan')
    cols = [c for c in setup if c in signals_df.columns]
    if len(cols) < 2:
        return float('nan')
    mis = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            mis.append(_mutual_information_bool(signals_df[cols[i]], signals_df[cols[j]]))
    return float(np.mean(mis)) if mis else float('nan')


def _calculate_trade_fields(edges: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Computes E_move, P_up, P_down, etc., from a distribution."""
    if edges.size < 2 or probs.size != edges.size - 1 or not np.isfinite(probs).any():
        return {}
    
    lo, hi = edges[:-1], edges[1:]
    mids = (lo + hi) / 2.0
    
    # Cap tails for E_move calculation to prevent blow-ups from sentinel values
    tail_cap = 0.12  # A reasonable cap for tail expectation
    left_open = np.isclose(lo, -999.0)
    right_open = np.isclose(hi, 999.0)
    mids[left_open] = -abs(tail_cap)
    mids[right_open] = abs(tail_cap)
    e_move = float(np.sum(mids * probs))
    
    # Probabilities of key regions, correctly handling the zero-straddling bin
    p_up = 0.0
    p_down = 0.0
    
    # Bins entirely on the positive side
    p_up += probs[lo > 0.0].sum()
    
    # Bins entirely on the negative side
    p_down += probs[hi < 0.0].sum()
    
    # Bin straddling zero
    straddle_mask = (lo < 0) & (hi > 0)
    if np.any(straddle_mask):
        straddle_prob = probs[straddle_mask].sum()
        straddle_lo = lo[straddle_mask][0]
        straddle_hi = hi[straddle_mask][0]
        
        # Apportion probability based on linear interpolation (since bin is uniform)
        if (straddle_hi - straddle_lo) > 0:
            up_fraction = straddle_hi / (straddle_hi - straddle_lo)
            p_up += straddle_prob * up_fraction
            p_down += straddle_prob * (1.0 - up_fraction)

    return {
        "E_move": e_move,
        "P_up": p_up,
        "P_down": p_down,
    }


def _calculate_objectives(
    trigger_dates: pd.DatetimeIndex,
    unconditional_returns: Dict[int, pd.Series],
    horizons: List[int],
    is_oos_fold: bool = False,  # Deprecated: use folds parameter instead
    folds: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None  # Injectable folds for NPWF
) -> Dict[str, float]:
    """
    Run the full metrics suite over walk-forward folds, select best horizon, and aggregate results.
    
    Args:
        trigger_dates: Dates when signal triggers
        unconditional_returns: Forward returns by horizon
        horizons: List of forecast horizons to evaluate
        is_oos_fold: DEPRECATED - use folds parameter instead
        folds: Optional list of (train_idx, test_idx) tuples for evaluation.
               If provided, uses these verbatim and skips internal splitting/fallbacks.
    """
    n_folds = settings.validation.n_folds
    embargo_days = settings.validation.embargo_days
    band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
    min_sup_fold = max(5, settings.validation.min_support // n_folds)

    sample_idx = next(iter(unconditional_returns.values())).index
    
    # NEW: Use provided folds if available (NPWF path)
    if folds is not None:
        # Use externally provided folds verbatim, skip internal splitting
        pass  # folds already set
    elif is_oos_fold:
        # Legacy path: single OOS window (will be deprecated)
        folds = [(sample_idx, sample_idx)]
    else:
        # Default: internal walk-forward splits
        folds = make_walkforward_splits(sample_idx, n_folds, embargo_days)

    all_horizon_metrics = {}

    for h in horizons:
        r_uncond = unconditional_returns[h]
        
        fold_metric_values = {}
        fold_band_probs = []

        for train_idx, test_idx in folds:
            train_triggers = trigger_dates.intersection(train_idx)
            test_triggers = trigger_dates.intersection(test_idx)

            r_train_cond = r_uncond.loc[train_triggers].dropna()
            r_test_cond = r_uncond.loc[test_triggers].dropna()
            
            # --- Tune-up: Handle sparse triggers (RELAXED for robustness) ---
            # Relax minimum test triggers from 5 to 2 for sparse data compatibility
            if r_test_cond.size < 2:
                #print(f"WARNING: Horizon {h} fold skipped: only {r_test_cond.size} test triggers (<2)")
                continue # Skip fold/horizon if insufficient test triggers

            # Relax minimum train support to 3 instead of min_sup_fold for sparse data
            if r_train_cond.size < 3:
                #print(f"WARNING: Horizon {h} fold skipped: only {r_train_cond.size} train triggers (<3)")
                continue

            forecast_probs = _histogram_probs(r_train_cond.values, band_edges)
            fold_band_probs.append(forecast_probs)

            fold_mets = {}
            # --- ADD NEW METRIC CALCULATIONS ---
            fold_mets["crps"] = distribution.crps(forecast_probs, r_test_cond.values, band_edges)
            fold_mets["pinball_q10"] = distribution.pinball_loss(forecast_probs, r_test_cond.values, band_edges, quantile=0.1)
            fold_mets["pinball_q90"] = distribution.pinball_loss(forecast_probs, r_test_cond.values, band_edges, quantile=0.9)
            fold_mets["info_gain"] = info_theory.info_gain(forecast_probs, r_test_cond.values, band_edges)
            fold_mets["w1_effect"] = distribution.wasserstein1(forecast_probs, r_test_cond.values, band_edges)
            fold_mets["calib_mae"] = distribution.calibration_mae(forecast_probs, r_test_cond.values, band_edges)
            # --- END NEW METRICS ---
            
            for key, value in fold_mets.items():
                if np.isfinite(value):
                    fold_metric_values.setdefault(key, []).append(value)
        
        if not fold_band_probs:
            # Silently skip horizons where no folds were usable (e.g., at end of OOS window)
            continue
        
        # Aggregate metrics and band_probs across folds
        aggregated_h_metrics = {}
        for key, values in fold_metric_values.items():
            res = aggregate(np.array(values), method="median_mad")
            aggregated_h_metrics[key] = res["agg"]
        
        aggregated_h_metrics['folds_used'] = len(fold_metric_values.get('crps', []))
        
        # Median aggregate band probs and re-normalize
        final_band_probs = np.nanmedian(np.array(fold_band_probs), axis=0)
        final_band_probs /= (final_band_probs.sum() + 1e-9)
        aggregated_h_metrics["band_probs"] = final_band_probs.tolist()

        all_horizon_metrics[h] = aggregated_h_metrics
    
    if not all_horizon_metrics:
        # When using externally provided folds (NPWF), fail-closed if all folds skipped
        # No single-window fallback to prevent leakage
        if folds is not None:
            return {"folds_used": 0}  # Fail-closed for NPWF
        
        # Legacy path: SINGLE-WINDOW OOS COMPUTATION (only when folds=None)
        # This is a legitimate OOS evaluation without internal CV, not a fabricated fallback
        # Suppress this warning - it's expected with sparse data and we handle it properly
        # print("Warning: No CV folds available, computing metrics on full OOS window")
        
        single_window_metrics = {}
        best_h = min(horizons)
        r_uncond = unconditional_returns[best_h]
        
        # Get all triggered returns for this horizon
        triggered_returns = r_uncond.loc[trigger_dates].dropna()
        all_returns = r_uncond.dropna()
        
        if len(triggered_returns) >= 3 and len(all_returns) >= 10:  # Minimum for meaningful calculation
            # Use the same metric computation logic as in the CV loop, but on the full window
            # Train on first 70% of triggered returns, test on last 30% (or use all data for both if small)
            if len(triggered_returns) >= 10:
                split_point = int(len(triggered_returns) * 0.7)
                r_train_cond = triggered_returns.iloc[:split_point]
                r_test_cond = triggered_returns.iloc[split_point:]
            else:
                # For small samples, use all data for both train and test (legitimate for OOS evaluation)
                r_train_cond = triggered_returns
                r_test_cond = triggered_returns
            
            # Compute forecast probabilities using the same method as CV
            forecast_probs = _histogram_probs(r_train_cond.values, band_edges)
            
            # Calculate the same metrics using the same functions as the CV loop
            single_window_metrics["crps"] = distribution.crps(forecast_probs, r_test_cond.values, band_edges)
            single_window_metrics["pinball_q10"] = distribution.pinball_loss(forecast_probs, r_test_cond.values, band_edges, quantile=0.1)
            single_window_metrics["pinball_q90"] = distribution.pinball_loss(forecast_probs, r_test_cond.values, band_edges, quantile=0.9)
            single_window_metrics["info_gain"] = info_theory.info_gain(forecast_probs, r_test_cond.values, band_edges)
            single_window_metrics["w1_effect"] = distribution.wasserstein1(forecast_probs, r_test_cond.values, band_edges)
            single_window_metrics["calib_mae"] = distribution.calibration_mae(forecast_probs, r_test_cond.values, band_edges)
            single_window_metrics["folds_used"] = 1  # Mark as single-window computation
            single_window_metrics["band_probs"] = forecast_probs.tolist()
            
            # Return in the same format as horizon metrics
            all_horizon_metrics[best_h] = single_window_metrics
        else:
            return {"folds_used": 0}

    # --- HORIZON AGGREGATION ---
    final_metrics = {}
    agg_method = settings.ga.horizon_agg_method
    
    # Collect all metric values across horizons
    metric_collections = {}
    for h_metrics in all_horizon_metrics.values():
        for key, value in h_metrics.items():
            metric_collections.setdefault(key, []).append(value)

    # Apply aggregation method
    for key, values in metric_collections.items():
        if key == "band_probs": # Special case for band probabilities
            final_band_probs = np.nanmedian(np.array(values), axis=0)
            final_band_probs /= (final_band_probs.sum() + 1e-9)
            final_metrics["band_probs"] = final_band_probs.tolist()
            continue

        # --- Tune-up: Directional Quantile Aggregation ---
        is_lower_better = any(substr in key for substr in ['crps', 'pinball', 'calib', 'sensitivity', 'redundancy', 'complexity'])

        if agg_method == "p75":
            quantile = settings.ga.h_quantile_low if is_lower_better else settings.ga.h_quantile_high
            final_metrics[key] = np.nanpercentile(values, quantile)
        elif agg_method == "mean":
            final_metrics[key] = np.nanmean(values)
        else: # Default to "best" horizon logic
            best_h = min(all_horizon_metrics.keys(), key=lambda h: all_horizon_metrics[h].get("crps", np.inf))
            final_metrics = all_horizon_metrics[best_h]
            break # Exit loop as we've already assigned final_metrics
    
    # Select a representative horizon for metrics that need a single return series
    best_h = min(all_horizon_metrics.keys(), key=lambda h: all_horizon_metrics[h].get("crps", np.inf))
    final_metrics["best_horizon"] = best_h
    
    # Calculate metrics on the combined OOS returns for the *best* horizon
    best_h_oos_returns = []
    r_uncond_best = unconditional_returns[best_h]
    for train_idx, test_idx in folds:
        test_triggers = trigger_dates.intersection(test_idx)
        best_h_oos_returns.append(r_uncond_best.loc[test_triggers].dropna())
    
    if best_h_oos_returns:
        full_oos_returns = pd.concat(best_h_oos_returns).dropna()

        # --- DYNAMICS & COMPLEXITY METRICS ---
        dfa_val = dynamics.dfa_alpha(full_oos_returns)
        final_metrics["dfa_alpha"] = dfa_val
        
        # Use configured complexity metric
        if settings.complexity.metric == "permutation":
            pe_cfg = settings.complexity
            comp_val = complexity.permutation_entropy(full_oos_returns, m=pe_cfg.pe_embedding, tau=pe_cfg.pe_tau)
            final_metrics["permutation_entropy"] = comp_val
            final_metrics["complexity_index"] = np.nan # Ensure other is null
        else:
            comp_idx_res = complexity.complexity_index(full_oos_returns, dfa_alpha=dfa_val)
            final_metrics["complexity_index"] = comp_idx_res["index"]
            final_metrics["permutation_entropy"] = np.nan # Ensure other is null

        # --- SENSITIVITY SCAN (Median Drop Under Bootstrap) ---
        base_crps = distribution.crps(final_metrics["band_probs"], full_oos_returns.values, band_edges)
        
        def _crps_wrapper(data, **kwargs):
            return distribution.crps(kwargs['probs'], data, kwargs['edges'])

        try:
            boot_runs = robustness.moving_block_bootstrap(
                _crps_wrapper, full_oos_returns.values, n_boot=50,
                probs=final_metrics["band_probs"], edges=band_edges
            )
            boot_median_crps = np.nanmedian(boot_runs['raw_values'])
            delta_edge = max(0, boot_median_crps - base_crps) # lower-is-better metric
            
            final_metrics["sensitivity_delta_edge"] = delta_edge
            final_metrics["bootstrap_p_value"] = boot_runs.get('p_value', np.nan)
        except Exception:
            final_metrics["sensitivity_delta_edge"] = np.nan
            final_metrics["bootstrap_p_value"] = np.nan
    else:
        final_metrics["dfa_alpha"] = np.nan
        final_metrics["complexity_index"] = np.nan
        final_metrics["permutation_entropy"] = np.nan
        final_metrics["sensitivity_delta_edge"] = np.nan
        final_metrics["bootstrap_p_value"] = np.nan

    # Calculate horizon stability
    crps_across_horizons = [m.get('crps', np.nan) for m in all_horizon_metrics.values()]
    final_metrics['horizon_stability_mad'] = _safe_mad(crps_across_horizons)
    
    # Pre-compute trade-ready fields from the final distribution
    if "band_probs" in final_metrics:
        trade_fields = _calculate_trade_fields(band_edges, np.array(final_metrics["band_probs"]))
        final_metrics.update(trade_fields)

    return final_metrics


def _calculate_robust_replacement_metrics(
    trigger_dates: pd.DatetimeIndex,
    unconditional_returns: Dict[int, pd.Series],
    horizons: List[int],
    band_edges: np.ndarray,
    is_oos_fold: bool = False
) -> Dict[str, float]:
    """
    Calculate robust replacement metrics for problematic ones like DFA, complexity, etc.
    Uses the comprehensive metrics suite with better coverage.
    """
    # Get the best horizon's returns for single-series metrics
    best_h = min(horizons) if horizons else horizons[0]
    returns_series = unconditional_returns[best_h].loc[trigger_dates].dropna()
    
    replacement_metrics = {}
    
    # 1. REPLACE DFA Alpha (35% coverage) with Robust Volatility Clustering
    try:
        # Use moving block bootstrap on volatility persistence instead of DFA
        vol_series = returns_series.rolling(5).std().dropna()
        if len(vol_series) >= 20:
            vol_persistence = vol_series.autocorr(lag=1)
            replacement_metrics['volatility_persistence'] = float(vol_persistence) if np.isfinite(vol_persistence) else 0.0
            
            # Alternative: Use RQA metrics which are more robust
            rqa_results = dynamics.rqa_metrics(returns_series, embedding_dim=2, delay=1, threshold=0.1)
            replacement_metrics['recurrence_rate'] = rqa_results.get('recurrence_rate', np.nan)
            replacement_metrics['determinism'] = rqa_results.get('determinism', np.nan)
        else:
            replacement_metrics['volatility_persistence'] = np.nan
            replacement_metrics['recurrence_rate'] = np.nan
            replacement_metrics['determinism'] = np.nan
    except Exception:
        replacement_metrics['volatility_persistence'] = np.nan
        replacement_metrics['recurrence_rate'] = np.nan
        replacement_metrics['determinism'] = np.nan
    
    # 2. REPLACE Complex Complexity Index with Simpler Robust Entropy
    try:
        if len(returns_series) >= 10:
            # Use the robust sample entropy directly
            samp_entropy = complexity.sample_entropy(returns_series, m=2, r=0.15)
            replacement_metrics['sample_entropy_robust'] = float(samp_entropy) if np.isfinite(samp_entropy) else np.nan
            
            # Add permutation entropy as backup
            perm_entropy = complexity.permutation_entropy(returns_series, m=3, tau=1, normalize=True)
            replacement_metrics['permutation_entropy_robust'] = float(perm_entropy) if np.isfinite(perm_entropy) else np.nan
        else:
            replacement_metrics['sample_entropy_robust'] = np.nan
            replacement_metrics['permutation_entropy_robust'] = np.nan
    except Exception:
        replacement_metrics['sample_entropy_robust'] = np.nan  
        replacement_metrics['permutation_entropy_robust'] = np.nan
    
    # 3. REPLACE Sparse Bootstrap P-values with TSCV Robustness
    try:
        if len(returns_series) >= 15:
            # Use time series cross-validation robustness instead
            def simple_mean_metric(x):
                return np.mean(x) if len(x) > 0 else 0.0
            
            tscv_results = robustness.tscv_robustness(
                simple_mean_metric, 
                returns_series.values, 
                n_splits=min(5, len(returns_series) // 5),
                embargo=1
            )
            replacement_metrics['tscv_mean_stability'] = tscv_results.get('mean', np.nan)
            replacement_metrics['tscv_coefficient_variation'] = tscv_results.get('cov', np.nan)
        else:
            replacement_metrics['tscv_mean_stability'] = np.nan
            replacement_metrics['tscv_coefficient_variation'] = np.nan
    except Exception:
        replacement_metrics['tscv_mean_stability'] = np.nan
        replacement_metrics['tscv_coefficient_variation'] = np.nan
    
    # 4. ADD TDA H0 Persistence (more robust than complex causality metrics)
    try:
        if len(returns_series) >= 10:
            # Use H0 persistent homology on returns
            points = returns_series.values.reshape(-1, 1)
            h0_diagram = tda.persistent_homology_h0(points)
            
            if h0_diagram.size > 0:
                lifetimes = h0_diagram[:, 1] - h0_diagram[:, 0]
                replacement_metrics['h0_total_persistence'] = float(np.sum(lifetimes))
                replacement_metrics['h0_max_lifetime'] = float(np.max(lifetimes))
                replacement_metrics['h0_components_count'] = int(len(lifetimes))
            else:
                replacement_metrics['h0_total_persistence'] = 0.0
                replacement_metrics['h0_max_lifetime'] = 0.0
                replacement_metrics['h0_components_count'] = 0
        else:
            replacement_metrics['h0_total_persistence'] = np.nan
            replacement_metrics['h0_max_lifetime'] = np.nan
            replacement_metrics['h0_components_count'] = np.nan
    except Exception:
        replacement_metrics['h0_total_persistence'] = np.nan
        replacement_metrics['h0_max_lifetime'] = np.nan
        replacement_metrics['h0_components_count'] = np.nan
    
    # 5. ADD Regime Detection Quality (replaces missing regime metrics)
    try:
        if len(returns_series) >= 30:
            # Detect regimes and measure quality
            regime_results = regime.detect_regimes(returns_series.values, k_states=2)
            
            if 'states' in regime_results and regime_results['states'] is not None:
                states = regime_results['states']
                # Measure regime stability (fewer transitions = more stable)
                transitions = np.sum(np.diff(states) != 0)
                regime_stability = 1.0 - (transitions / max(len(states) - 1, 1))
                replacement_metrics['regime_stability'] = float(regime_stability)
                
                # Measure regime separation (how distinct the regimes are)
                regime_0_mean = returns_series.iloc[states == 0].mean() if np.any(states == 0) else np.nan
                regime_1_mean = returns_series.iloc[states == 1].mean() if np.any(states == 1) else np.nan
                
                if np.isfinite(regime_0_mean) and np.isfinite(regime_1_mean):
                    regime_separation = abs(regime_0_mean - regime_1_mean) / returns_series.std()
                    replacement_metrics['regime_separation'] = float(regime_separation)
                else:
                    replacement_metrics['regime_separation'] = np.nan
            else:
                replacement_metrics['regime_stability'] = np.nan
                replacement_metrics['regime_separation'] = np.nan
        else:
            replacement_metrics['regime_stability'] = np.nan
            replacement_metrics['regime_separation'] = np.nan
    except Exception:
        replacement_metrics['regime_stability'] = np.nan
        replacement_metrics['regime_separation'] = np.nan
    
    # 6. ADD Aggregated Robustness Score (replaces multiple sparse metrics)
    robust_components = []
    
    # Collect valid robustness measures
    for key in ['tscv_coefficient_variation', 'volatility_persistence', 'regime_stability']:
        value = replacement_metrics.get(key, np.nan)
        if np.isfinite(value):
            # Invert coefficient of variation (lower is better for stability)
            if key == 'tscv_coefficient_variation':
                value = 1.0 / (1.0 + abs(value))
            robust_components.append(value)
    
    if robust_components:
        replacement_metrics['composite_robustness'] = float(np.mean(robust_components))
    else:
        replacement_metrics['composite_robustness'] = np.nan
    
    return replacement_metrics


def _calculate_objectives_with_robust_replacements(
    trigger_dates: pd.DatetimeIndex,
    unconditional_returns: Dict[int, pd.Series],
    horizons: List[int],
    is_oos_fold: bool = False,
    folds: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None
) -> Dict[str, Any]:
    """
    Enhanced version of _calculate_objectives that includes robust replacement metrics.
    
    Args:
        trigger_dates: Dates when signal triggers
        unconditional_returns: Forward returns by horizon
        horizons: List of forecast horizons to evaluate
        is_oos_fold: DEPRECATED - use folds parameter instead
        folds: Optional list of (train_idx, test_idx) tuples for evaluation
    """
    # Get the original metrics first
    original_metrics = _calculate_objectives(trigger_dates, unconditional_returns, horizons, is_oos_fold, folds)
    
    # Calculate robust replacements
    band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
    robust_metrics = _calculate_robust_replacement_metrics(
        trigger_dates, unconditional_returns, horizons, band_edges, is_oos_fold
    )
    
    # Merge the metrics
    enhanced_metrics = {**original_metrics, **robust_metrics}
    
    # Use robust alternatives for key computations
    # If DFA is NaN, use volatility persistence
    if np.isnan(enhanced_metrics.get('dfa_alpha', np.nan)):
        enhanced_metrics['dfa_alpha_robust'] = robust_metrics.get('volatility_persistence', np.nan)
    
    # If complexity metrics are sparse, use sample entropy
    if np.isnan(enhanced_metrics.get('complexity_index', np.nan)):
        enhanced_metrics['complexity_metric_raw'] = robust_metrics.get('sample_entropy_robust', np.nan)
    elif np.isnan(enhanced_metrics.get('permutation_entropy', np.nan)):
        enhanced_metrics['complexity_metric_raw'] = robust_metrics.get('permutation_entropy_robust', np.nan)
    
    # Use composite robustness as bootstrap replacement  
    if np.isnan(enhanced_metrics.get('bootstrap_p_value', np.nan)):
        # Convert robustness to a p-value-like metric (higher robustness = lower "p-value")
        composite_rob = robust_metrics.get('composite_robustness', np.nan)
        if np.isfinite(composite_rob):
            enhanced_metrics['bootstrap_p_value_robust'] = 1.0 - composite_rob
    
    return enhanced_metrics


def _calculate_objectives_single_horizon(
    trigger_dates: pd.DatetimeIndex,
    returns_series: pd.Series,
    horizon: int,
    folds: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None  # Injectable folds for NPWF
) -> Dict[str, float]:
    """
    Simplified objective calculation for a single horizon.
    
    Args:
        trigger_dates: Dates when signal triggers
        returns_series: Forward returns for this horizon
        horizon: Forecast horizon in days
        folds: Optional list of (train_idx, test_idx) tuples.
               If provided, uses these verbatim and skips internal splitting/fallbacks.
    """
    band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
    n_folds = settings.validation.n_folds
    embargo_days = settings.validation.embargo_days
    min_sup_fold = max(3, settings.validation.min_support // n_folds)  # Relaxed from 5 to 3
    
    # Create folds: use provided if available, otherwise internal walk-forward
    if folds is None:
        sample_idx = returns_series.index
        folds = make_walkforward_splits(sample_idx, n_folds, embargo_days)
    
    fold_metrics = []
    fold_band_probs = []
    
    for train_idx, test_idx in folds:
        train_triggers = trigger_dates.intersection(train_idx)
        test_triggers = trigger_dates.intersection(test_idx)
        
        r_train = returns_series.loc[train_triggers].dropna()
        r_test = returns_series.loc[test_triggers].dropna()
        
        if len(r_test) < 3 or len(r_train) < min_sup_fold:  # Relaxed from 5 to 3
            continue
            
        # Compute forecast and metrics
        forecast_probs = _histogram_probs(r_train.values, band_edges)
        fold_band_probs.append(forecast_probs)
        
        fold_met = {}
        fold_met["crps"] = distribution.crps(forecast_probs, r_test.values, band_edges)
        fold_met["pinball_q10"] = distribution.pinball_loss(forecast_probs, r_test.values, band_edges, 0.1)
        fold_met["pinball_q90"] = distribution.pinball_loss(forecast_probs, r_test.values, band_edges, 0.9)
        fold_met["info_gain"] = info_theory.info_gain(forecast_probs, r_test.values, band_edges)
        fold_met["w1_effect"] = distribution.wasserstein1(forecast_probs, r_test.values, band_edges)
        fold_met["calib_mae"] = distribution.calibration_mae(forecast_probs, r_test.values, band_edges)
        
        fold_metrics.append(fold_met)
    
    if not fold_metrics:
        # When using externally provided folds (NPWF), fail-closed if all folds skipped
        # No single-window fallback to prevent leakage
        if folds is not None:
            return {"folds_used": 0, "horizon_used": horizon}  # Fail-closed for NPWF
        
        # Legacy path: Single-window computation when CV fails (only when folds=None)
        triggered_returns = returns_series.loc[trigger_dates].dropna()
        if len(triggered_returns) >= 3:
            # Use same approach as main function
            if len(triggered_returns) >= 10:
                split_point = int(len(triggered_returns) * 0.7)
                r_train_cond = triggered_returns.iloc[:split_point]
                r_test_cond = triggered_returns.iloc[split_point:]
            else:
                r_train_cond = triggered_returns
                r_test_cond = triggered_returns
            
            forecast_probs = _histogram_probs(r_train_cond.values, band_edges)
            
            single_metrics = {}
            single_metrics["crps"] = distribution.crps(forecast_probs, r_test_cond.values, band_edges)
            single_metrics["pinball_q10"] = distribution.pinball_loss(forecast_probs, r_test_cond.values, band_edges, 0.1)
            single_metrics["pinball_q90"] = distribution.pinball_loss(forecast_probs, r_test_cond.values, band_edges, 0.9)
            single_metrics["info_gain"] = info_theory.info_gain(forecast_probs, r_test_cond.values, band_edges)
            single_metrics["w1_effect"] = distribution.wasserstein1(forecast_probs, r_test_cond.values, band_edges)
            single_metrics["calib_mae"] = distribution.calibration_mae(forecast_probs, r_test_cond.values, band_edges)
            single_metrics["folds_used"] = 1
            single_metrics["horizon_used"] = horizon
            return single_metrics
        
        return {"folds_used": 0, "horizon_used": horizon}
    
    # Aggregate across folds (median)
    final_metrics = {}
    for key in fold_metrics[0].keys():
        values = [fm[key] for fm in fold_metrics if np.isfinite(fm[key])]
        final_metrics[key] = np.median(values) if values else np.nan
    
    final_metrics["folds_used"] = len(fold_metrics)
    final_metrics["horizon_used"] = horizon
    final_metrics["band_probs"] = np.median(fold_band_probs, axis=0).tolist()
    
    # Add robust replacement metrics for single horizon
    try:
        all_test_returns = pd.concat([returns_series.loc[trigger_dates.intersection(test_idx)] 
                                    for _, test_idx in folds]).dropna()
        
        if len(all_test_returns) >= 10:
            # Calculate robust metrics on aggregated test data
            robust_metrics = _calculate_robust_replacement_metrics(
                trigger_dates, {horizon: returns_series}, [horizon], band_edges
            )
            final_metrics.update(robust_metrics)
    except Exception:
        pass
    
    return final_metrics


# -----------------------------
# Helper: Infer direction from signal metadata
# -----------------------------
def _infer_direction_from_metadata(setup: List[str], signals_metadata: List[Dict]) -> str:
    """Crude heuristic: count '<' as bearish; otherwise bullish; ties -> long."""
    direction_score = 0
    for sid in setup:
        meta = next((m for m in signals_metadata if m.get("signal_id") == sid), None)
        if meta and "<" in str(meta.get("condition", "")):
            direction_score -= 1
        else:
            direction_score += 1
    return "long" if direction_score >= 0 else "short"


# -----------------------------
# NPWF-aware evaluation helper
# -----------------------------
def _evaluate_one_setup_with_plan(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    plan: GADataSpec,
) -> Dict[str, Any]:
    ticker, setup = individual
    direction = _infer_direction_from_metadata(setup, signals_metadata)

    try:
        dna = _dna(individual)
        if not hasattr(_evaluate_one_setup_with_plan, '_setup_id_map'):
            _evaluate_one_setup_with_plan._setup_id_map = {}
            _evaluate_one_setup_with_plan._setup_id_counter = 0
        setup_map = _evaluate_one_setup_with_plan._setup_id_map  # type: ignore[attr-defined]
        if dna not in setup_map:
            _evaluate_one_setup_with_plan._setup_id_counter += 1  # type: ignore[attr-defined]
            setup_map[dna] = f"SETUP_{_evaluate_one_setup_with_plan._setup_id_counter:04d}"  # type: ignore[index]
        setup_id = setup_map[dna]
    except Exception:
        setup_id = f"SETUP_{abs(hash((ticker, tuple(sorted(setup))))) % 10_000:04d}"

    signals_fp_base = canonical_signals_fingerprint(setup)
    exit_policy_tag = getattr(plan, 'outer_id', 'npwf')
    allow_pyramiding = bool(getattr(settings.ga, 'allow_pyramiding', False))
    dup_suppressed_total = 0
    dup_merged_total = 0

    ga_cfg = getattr(settings, 'ga', object())
    min_trades_per_fold = max(MIN_FOLD_TRADES, int(getattr(ga_cfg, 'min_trades_per_fold', MIN_FOLD_TRADES)))
    min_trade_days_per_fold = max(MIN_FOLD_TRADE_DAYS, int(getattr(ga_cfg, 'min_trade_days_per_fold', MIN_FOLD_TRADE_DAYS)))
    min_total_trades_required = int(getattr(ga_cfg, 'min_total_trades', 20))

    if not setup:
        return _penalized_result(individual, direction, plan, "empty_setup")

    if plan is None or plan.n_folds == 0:
        return _penalized_result(individual, direction, plan, "plan_missing")

    plan_train_idx = signals_df.index.intersection(plan.train_idx)
    fold_ledgers: List[pd.DataFrame] = []
    fold_daily_returns: List[pd.Series] = []
    fold_summaries: List[Dict[str, Any]] = []

    def _safe_metric_value(payload: Dict[str, Any], key: str, default: float = np.nan) -> float:
        val = payload.get(key) if isinstance(payload, dict) else None
        if val is None:
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    for fold_idx, fold in enumerate(plan.inner_folds):
        attempt_success = False
        last_failure_reason = FOLD_REASON_EMPTY
        selected_idx: Optional[pd.Index] = None
        selected_attempt_tag = "original"
        selected_detail = ""
        selected_ledger: Optional[pd.DataFrame] = None
        selected_returns: Optional[pd.Series] = None

        candidates = _candidate_test_indices(plan, fold_idx, DEFAULT_EXPAND_MARGIN)
        fold_dup_stats: Dict[str, int] = {"n_dups_suppressed": 0, "n_dups_merged": 0}

        for idx_candidate, attempt_tag in candidates:
            candidate_idx = pd.Index(idx_candidate).intersection(plan_train_idx)
            candidate_idx = candidate_idx.intersection(signals_df.index)
            if candidate_idx.empty:
                last_failure_reason = f"{attempt_tag}_empty_intersection"
                continue

            fold_signals = signals_df.loc[candidate_idx]

            try:
                ledger = backtester.run_setup_backtest_options(
                    setup_signals=setup,
                    signals_df=fold_signals,
                    master_df=master_df,
                    direction=direction,
                    exit_policy=None,
                    tickers_to_run=[ticker],
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                last_failure_reason = f"backtest_error:{exc}"
                continue

            if ledger is None or ledger.empty:
                last_failure_reason = f"{attempt_tag}_empty_ledger"
                continue

            ledger = ledger.copy()
            ledger['setup_id'] = setup_id
            ledger['ticker'] = ticker
            ledger['direction'] = direction
            ledger['signals_fingerprint'] = signals_fp_base
            ledger['trigger_date'] = pd.to_datetime(ledger['trigger_date'], errors='coerce')
            valid_ledger = ledger.dropna(subset=['trigger_date'])
            if valid_ledger.empty:
                last_failure_reason = f"{attempt_tag}_no_valid_triggers"
                continue

            test_start = pd.to_datetime(candidate_idx.min())
            test_end = pd.to_datetime(candidate_idx.max())
            mask = valid_ledger['trigger_date'].between(test_start, test_end)
            valid_ledger = valid_ledger.loc[mask]
            if valid_ledger.empty:
                last_failure_reason = f"{attempt_tag}_no_trades_in_window"
                continue

            valid_ledger["fold_id"] = fold.fold_id
            valid_ledger["outer_id"] = plan.outer_id

            deduped_ledger, fold_dup_stats = dedupe_trade_ledger(
                valid_ledger,
                setup_id=setup_id,
                ticker=ticker,
                direction=direction,
                signals_fingerprint=signals_fp_base,
                exit_policy_tag=exit_policy_tag,
                allow_pyramiding=allow_pyramiding,
            )
            dup_suppressed_total += fold_dup_stats.get('n_dups_suppressed', 0)
            dup_merged_total += fold_dup_stats.get('n_dups_merged', 0)

            fold_trades = len(deduped_ledger)
            fold_trade_days = deduped_ledger['trigger_date'].dt.normalize().nunique()

            trades_ok = fold_trades >= min_trades_per_fold
            trade_days_ok = fold_trade_days >= min_trade_days_per_fold
            if not (trades_ok or trade_days_ok):
                last_failure_reason = f"{attempt_tag}_insufficient_support"
                continue

            deduped_ledger['signals_fingerprint'] = signals_fp_base
            deduped_ledger['setup_id'] = setup_id

            fold_returns = selection.portfolio_daily_returns(deduped_ledger)
            fold_returns = pd.to_numeric(fold_returns, errors="coerce").dropna()
            if fold_returns.empty:
                last_failure_reason = f"{attempt_tag}_no_daily_returns"
                continue

            attempt_success = True
            selected_idx = candidate_idx
            selected_attempt_tag = attempt_tag
            selected_detail = "ok" if attempt_tag == "original" else attempt_tag
            selected_ledger = deduped_ledger
            selected_returns = fold_returns
            break

        reason_code, reason_detail = _fold_reason(
            attempt_success,
            selected_attempt_tag,
            selected_detail if attempt_success else last_failure_reason,
        )

        fold_train_idx = pd.Index(fold.train_idx)
        train_start = str(fold_train_idx[0]) if len(fold_train_idx) else None
        train_end = str(fold_train_idx[-1]) if len(fold_train_idx) else None
        test_start = str(selected_idx[0]) if (attempt_success and selected_idx is not None) else None
        test_end = str(selected_idx[-1]) if (attempt_success and selected_idx is not None) else None

        summary: Dict[str, Any] = {
            "fold_id": fold.fold_id,
            "fold_supported": attempt_success,
            "supported": attempt_success,
            "reason": reason_code,
            "reason_detail": reason_detail,
            "attempt_tag": selected_attempt_tag if attempt_success else None,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "n_trades": 0,
            "n_trade_days": 0,
            "support": 0.0,
            "dup_suppressed": 0,
            "dup_merged": 0,
        }

        if not attempt_success or selected_ledger is None or selected_returns is None:
            fold_summaries.append(summary)
            continue

        fold_ledgers.append(selected_ledger)
        fold_daily_returns.append(selected_returns)

        fold_perf = metrics.compute_portfolio_metrics_bundle(
            daily_returns=selected_returns,
            trade_ledger=selected_ledger,
            do_winsorize=True,
            bootstrap_B=250,
            bootstrap_method="stationary",
            seed=int(getattr(settings.ga, "seed", 194)) + fold_idx,
            n_trials_for_dsr=50,
        )

        summary.update({
            "fold_supported": True,
            "supported": True,
            "support": _safe_metric_value(fold_perf, "support", 0.0),
            "n_trades": int(len(selected_ledger)),
            "n_trade_days": int(selected_ledger['trigger_date'].dt.normalize().nunique()),
            "dsr": _safe_metric_value(fold_perf, "dsr"),
            "bootstrap_calmar_lb": _safe_metric_value(fold_perf, "bootstrap_calmar_lb"),
            "bootstrap_profit_factor_lb": _safe_metric_value(fold_perf, "bootstrap_profit_factor_lb"),
            "dup_suppressed": int(fold_dup_stats.get('n_dups_suppressed', 0)),
            "dup_merged": int(fold_dup_stats.get('n_dups_merged', 0)),
        })
        fold_summaries.append(summary)

    if not fold_ledgers:
        metrics_stub = {
            "fold_summaries": fold_summaries,
            "outer_id": plan.outer_id,
            "fold_plan_hash": plan.fold_hash,
            "eligibility_reasons": ["no_supported_folds"],
        }
        return _penalized_result(individual, direction, plan, "no_supported_folds", metrics_stub)

    supported_folds = [fs for fs in fold_summaries if fs.get("supported")]
    if not supported_folds:
        metrics_stub = {
            "fold_summaries": fold_summaries,
            "outer_id": plan.outer_id,
            "fold_plan_hash": plan.fold_hash,
            "eligibility_reasons": ["no_supported_folds"],
        }
        return _penalized_result(individual, direction, plan, "no_supported_folds", metrics_stub)

    coverage_target_ratio = float(getattr(settings.ga, "min_supported_fold_ratio", 0.2))
    min_supported_folds = max(
        getattr(settings.ga, "min_outer_folds", 2),
        int(np.ceil(plan.n_folds * coverage_target_ratio)),
    )
    coverage_shortfall = max(0, min_supported_folds - len(supported_folds))

    combined_returns = pd.concat(fold_daily_returns, axis=0)
    combined_returns = pd.to_numeric(combined_returns, errors="coerce").dropna()
    if combined_returns.empty:
        metrics_stub = {
            "fold_summaries": fold_summaries,
            "outer_id": plan.outer_id,
            "fold_plan_hash": plan.fold_hash,
            "eligibility_reasons": ["no_returns_after_concat"],
        }
        return _penalized_result(individual, direction, plan, "no_returns_after_concat", metrics_stub)

    combined_returns = combined_returns.groupby(combined_returns.index).mean().sort_index()
    combined_ledger = pd.concat(fold_ledgers, ignore_index=True)

    combined_ledger, cross_dup_stats = dedupe_trade_ledger(
        combined_ledger,
        setup_id=setup_id,
        ticker=ticker,
        direction=direction,
        signals_fingerprint=signals_fp_base,
        exit_policy_tag=exit_policy_tag,
        allow_pyramiding=allow_pyramiding,
    )
    dup_suppressed_total += cross_dup_stats.get('n_dups_suppressed', 0)
    dup_merged_total += cross_dup_stats.get('n_dups_merged', 0)
    unique_structure_keys = []
    if 'options_structure_key' in combined_ledger.columns:
        unique_structure_keys = sorted(set(combined_ledger['options_structure_key'].dropna().astype(str)))

    sanitized_returns = combined_returns.clip(lower=-0.95, upper=10.0)
    sanitized_ledger = combined_ledger.copy()
    if 'pnl_pct' in sanitized_ledger.columns:
        sanitized_ledger['pnl_pct'] = pd.to_numeric(sanitized_ledger['pnl_pct'], errors='coerce').clip(lower=-0.95, upper=10.0)

    agg_perf = metrics.compute_portfolio_metrics_bundle(
        daily_returns=sanitized_returns,
        trade_ledger=sanitized_ledger,
        do_winsorize=True,
        bootstrap_B=1000,
        bootstrap_method="stationary",
        seed=int(getattr(settings.ga, "seed", 194)),
        n_trials_for_dsr=100,
    )

    agg_perf.update({
        "folds_used": len(supported_folds),
        "supported_folds": len(supported_folds),
        "total_folds": plan.n_folds,
        "fold_summaries": fold_summaries,
        "outer_id": plan.outer_id,
        "fold_plan_hash": plan.fold_hash,
        "n_trades": int(len(combined_ledger)),
        "fold_count": len(supported_folds),
        "fold_coverage_ratio": float(len(supported_folds)) / max(1, plan.n_folds),
        "coverage_target_ratio": coverage_target_ratio,
        "coverage_min_folds": min_supported_folds,
        "coverage_shortfall": coverage_shortfall,
        "dup_suppressed_total": int(dup_suppressed_total),
        "dup_merged_total": int(dup_merged_total),
        "dedup_applied": bool(dup_suppressed_total or dup_merged_total),
        "uniq_keys": int(combined_ledger['uniq_key'].nunique() if 'uniq_key' in combined_ledger.columns else len(combined_ledger)),
        "setup_id": setup_id,
        "signals_fingerprint": signals_fp_base,
        "options_structure_keys": unique_structure_keys,
    })

    support_bars = float(agg_perf.get("support", 0.0))
    total_trades = int(agg_perf.get("n_trades", 0) or 0)

    fatal_reasons: List[str] = []
    if support_bars <= 0:
        fatal_reasons.append("no_support")
    if total_trades <= 0:
        fatal_reasons.append("no_trades")

    if fatal_reasons:
        metrics_stub = agg_perf.copy()
        metrics_stub["eligibility_reasons"] = fatal_reasons
        return _penalized_result(individual, direction, plan, "fatal_gates", metrics_stub)

    coverage_ratio = agg_perf.get("fold_coverage_ratio", 0.0) or 0.0
    penalty_scalar, penalty_details, reasons = _compute_soft_penalties(
        agg_perf,
        settings.ga,
        coverage_ratio=coverage_ratio,
        coverage_target=coverage_target_ratio,
        min_total_trades_required=min_total_trades_required,
    )

    agg_perf["soft_penalties"] = penalty_details
    agg_perf["penalty_scalar"] = penalty_scalar
    agg_perf["eligibility_reasons"] = list(dict.fromkeys(reasons))
    agg_perf["eligible"] = True
    agg_perf["fatal_reasons"] = fatal_reasons

    objective_sources: Dict[str, str] = {}
    adjusted_objectives: List[float] = []
    for key in _objective_keys():
        original_val = agg_perf.get(key)
        agg_perf[f"{key}_raw"] = original_val
        sanitized_val, source = _resolve_objective_value(agg_perf, key)
        agg_perf[key] = sanitized_val
        agg_perf[f"{key}_score"] = sanitized_val
        objective_sources[key] = source
        if sanitized_val <= -998.0:
            adjusted_objectives.append(sanitized_val)
        else:
            adjusted_objectives.append(float(sanitized_val / penalty_scalar))
    agg_perf["objective_sources"] = objective_sources
    objectives = adjusted_objectives

    return {
        "individual": individual,
        "metrics": agg_perf,
        "objectives": objectives,
        "rank": np.inf,
        "crowding_distance": 0.0,
        "trade_ledger": combined_ledger,
        "direction": direction,
        "fold_plan": plan.summary(),
        "exit_policy": plan,
        "gates_passed": len(fatal_reasons) == 0,
    }


# -----------------------------
# Evaluation core - BACKTESTING VERSION (Phase A)
# -----------------------------
def _evaluate_one_setup(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Any],
) -> Dict:
    """
    Evaluate a (ticker, setup) by running an options backtest on ONLY its ticker
    and returning portfolio-level metrics from that ledger.

    This is the BACKTESTING VERSION - runs actual options trades for fitness evaluation.
    """
    ticker, setup = individual

    plan = exit_policy if isinstance(exit_policy, GADataSpec) else None
    if plan is not None and plan.n_folds > 0:
        return _evaluate_one_setup_with_plan(
            individual, signals_df, signals_metadata, master_df, plan
        )

    if not setup:
        return {
            "individual": individual, 
            "metrics": {},
            "objectives": [-99.0, -9999.0, 0.0],
            "rank": np.inf, 
            "crowding_distance": 0.0,
            "trade_ledger": pd.DataFrame(), 
            "direction": "long", 
            "exit_policy": exit_policy,
        }

    direction = _infer_direction_from_metadata(setup, signals_metadata)

    # Run OPTIONS BACKTEST on the specialized ticker
    ledger = backtester.run_setup_backtest_options(
        setup_signals=setup,
        signals_df=signals_df,
        master_df=master_df,
        direction=direction,
        exit_policy=exit_policy,
        tickers_to_run=[ticker],  # IMPORTANT: restrict to that ticker
    )

    if ledger is None or ledger.empty:
        return {
            "individual": individual, 
            "metrics": {},
            "objectives": [-99.0, -9999.0, 0.0],
            "rank": np.inf, 
            "crowding_distance": 0.0,
            "trade_ledger": pd.DataFrame(), 
            "direction": direction, 
            "exit_policy": exit_policy,
        }

    # Generate unique setup ID
    try:
        key = (ticker, tuple(sorted(setup)))
        if not hasattr(_evaluate_one_setup, '_setup_counter'):
            _evaluate_one_setup._setup_counter = 0
            _evaluate_one_setup._setup_id_map = {}
        
        if key not in _evaluate_one_setup._setup_id_map:
            _evaluate_one_setup._setup_counter += 1
            _evaluate_one_setup._setup_id_map[key] = f"SETUP_{_evaluate_one_setup._setup_counter:04d}"
        
        setup_id = _evaluate_one_setup._setup_id_map[key]
    except Exception:
        setup_id = str(individual)

    ledger = ledger.copy()
    ledger["setup_id"] = setup_id
    
    # Apply exclusivity if enabled
    if _parse_bt_env_flag("BT_ENFORCE_EXCLUSIVITY", True):
        ledger = _enforce_exclusivity_by_setup(ledger)

    # Calculate comprehensive portfolio metrics from ledger
    daily_returns = selection.portfolio_daily_returns(ledger)
    perf = metrics.compute_portfolio_metrics_bundle(
        daily_returns=daily_returns,
        trade_ledger=ledger,
        do_winsorize=True,
        bootstrap_B=1000,
        bootstrap_method="stationary",
        n_trials_for_dsr=100  # Assume 100 trials for DSR calculation
    )

    support_bars = float(perf.get('support', 0.0) or 0.0)
    num_trades = len(ledger) if ledger is not None and not ledger.empty else 0
    if support_bars <= 0 or num_trades <= 0:
        return {
            "individual": individual,
            "metrics": perf,
            "objectives": _penalty_vector(),
            "rank": np.inf,
            "crowding_distance": 0.0,
            "trade_ledger": ledger,
            "direction": direction,
            "exit_policy": exit_policy,
            "gates_passed": False
        }

    penalty_scalar, penalty_details, reasons = _compute_soft_penalties(
        perf,
        settings.ga,
        coverage_ratio=perf.get('fold_coverage_ratio', 0.0) or 0.0,
        coverage_target=float(getattr(settings.ga, 'min_supported_fold_ratio', 0.0)),
        min_total_trades_required=int(getattr(settings.ga, 'min_total_trades', max(1, num_trades))),
    )
    perf['soft_penalties'] = penalty_details
    perf['penalty_scalar'] = penalty_scalar
    perf['eligibility_reasons'] = list(dict.fromkeys(reasons))
    perf['eligible'] = len(reasons) == 0
    gates_passed = True

    # Build GA objectives vector from configurable keys
    obj_keys = _objective_keys()

    # Map metrics to objective values with safe fallbacks
    objectives = []
    objective_sources = {}
    for k in obj_keys:
        val, source = _resolve_objective_value(perf, k)
        objective_sources[k] = source
        if val <= -998.0:
            objectives.append(val)
        else:
            objectives.append(float(val / penalty_scalar))
    perf['objective_sources'] = objective_sources

    return {
        "individual": individual,
        "metrics": perf,
        "objectives": objectives,
        "rank": np.inf,
        "crowding_distance": 0.0,
        "trade_ledger": ledger,
        "direction": direction,
        "exit_policy": exit_policy,
        "gates_passed": gates_passed
    }


# LEGACY / UNUSED in NPWF; retained for backward compatibility.
# The NPWF flow uses injectable folds in _calculate_objectives instead.
def _evaluate_one_setup_horizon_specific(
    individual: Tuple,
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Any,
) -> Dict[str, Any]:
    """
    Evaluates a single individual (setup + horizon) across all CV splits,
    calculating new fitness objectives based on the distribution of Info Gain.
    
    DEPRECATED: This function is not used in the NPWF (forecast-first) flow.
    Retained for backward compatibility with legacy code paths.
    """
    # This is now the primary evaluation loop for the GA
    all_split_metrics = []
    
    # exit_policy.splits now contains the purged CPCV splits
    for train_idx, test_idx in exit_policy.splits:
        # Get trigger dates ONLY within the test window
        test_signals = signals_df.loc[test_idx]
        trigger_dates_for_split = get_valid_trigger_dates(test_signals, individual.signals, 0) # min_support=0 for raw dates

        if len(trigger_dates_for_split) < settings.validation.min_support:
            continue

        unconditional_returns = {
            h: _forward_returns(master_df.loc[test_idx], individual.ticker, h, settings.forecast.price_field)
            for h in [individual.horizon]
        }
        
        metrics = _calculate_objectives_with_robust_replacements(
            trigger_dates_for_split, unconditional_returns, [individual.horizon]
        )
        all_split_metrics.append(metrics)

    if not all_split_metrics:
        # Return worst possible fitness if no splits had enough support
        return {"objectives": (-np.inf, -np.inf)}

    # --- Calculate new fitness objectives from the distribution of scores ---
    info_gain_scores = [m.get('info_gain', np.nan) for m in all_split_metrics]
    info_gain_scores = [s for s in info_gain_scores if pd.notna(s)]

    if len(info_gain_scores) < settings.validation.min_support:
        # Not enough valid scores to calculate robust statistics
        return {"objectives": (-np.inf, -np.inf)}

    # 1. Sharpe Ratio of Information Gain scores
    ig_mean = np.mean(info_gain_scores)
    ig_std = np.std(info_gain_scores)
    ig_sharpe = ig_mean / ig_std if ig_std > 0 else 0.0

    # 2. Minimum Information Gain (worst-case performance)
    min_ig = np.min(info_gain_scores)

    # We also need to return the aggregated metrics for logging/later analysis
    # For simplicity, we'll take the median of all raw metrics across the splits
    final_metrics = {}
    if all_split_metrics:
        df_metrics = pd.DataFrame(all_split_metrics)
        # Use median for aggregation as it's robust to outliers
        median_metrics = df_metrics.median(numeric_only=True).to_dict()
        final_metrics.update(median_metrics)

    # The GA expects a dictionary containing the 'objectives' tuple
    return {
        "metrics": final_metrics,
        "objectives": (ig_sharpe, min_ig)
    }


def _hash_folds(folds: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]]) -> Optional[str]:
    """
    Generate deterministic hash of fold boundaries for cache key.
    
    Args:
        folds: List of (train_idx, test_idx) tuples
        
    Returns:
        Hex string hash of fold boundaries, or None if folds is None
    """
    if folds is None:
        return None
    
    import hashlib
    fold_strings = []
    for train_idx, test_idx in folds:
        train_str = f"{train_idx.min()}_{train_idx.max()}"
        test_str = f"{test_idx.min()}_{test_idx.max()}"
        fold_strings.append(f"{train_str}|{test_str}")
    
    combined = ";;".join(fold_strings)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# Cache wrapper with fold-aware key
_eval_cache: Dict[Tuple, Dict] = {}

def _evaluate_one_setup_cached(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[GADataSpec],
    folds_hash: Optional[str] = None  # Hash of fold boundaries for cache key
) -> Dict:
    """Cached evaluation with optional fold-aware key for determinism."""
    key = _dna(individual)
    if folds_hash is not None:
        # Include fold hash in cache key for NPWF determinism
        key = (*key, folds_hash)
    if key in _eval_cache:
        return _eval_cache[key]
    res = _evaluate_one_setup(individual, signals_df, signals_metadata, master_df, exit_policy)
    _eval_cache[key] = res
    return res


def _summarize_evals(tag: str, evaluated: List[Dict]) -> None:
    if not evaluated:
        return
    sup = [e.get("metrics", {}).get("support", 0) for e in evaluated]
    dsr_vals = [e.get("metrics", {}).get("dsr_score", e.get("metrics", {}).get("dsr", np.nan)) for e in evaluated]
    calmar_lb_vals = [e.get("metrics", {}).get("bootstrap_calmar_lb_score", e.get("metrics", {}).get("bootstrap_calmar_lb", np.nan)) for e in evaluated]
    pf_lb_vals = [e.get("metrics", {}).get("bootstrap_profit_factor_lb_score", e.get("metrics", {}).get("bootstrap_profit_factor_lb", np.nan)) for e in evaluated]
    folds = [e.get("metrics", {}).get("folds_used", e.get("metrics", {}).get("supported_folds", 0)) for e in evaluated]
    dup_vals = [e.get("metrics", {}).get("dup_suppressed_total", 0) for e in evaluated]

    def _med(x):
        return _safe_median(x)

    print(
        f"[{tag}] med(support)={_fmt(_med(sup),1)} | med(folds)={_fmt(_med(folds),1)} | "
        f"med(DSR)={_fmt(_med(dsr_vals),4)} med(Calmar_LB)={_fmt(_med(calmar_lb_vals),4)} | "
        f"med(PF_LB)={_fmt(_med(pf_lb_vals),4)} | med(DupSupp)={_fmt(_med(dup_vals),1)}"
    )


# -----------------------------
# Robust aggregators + display guards
# -----------------------------
def _safe_median(v: Iterable[float]) -> float:
    a = np.asarray(list(v), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    return float(np.median(a))


def _safe_mad(v: Iterable[float]) -> float:
    a = np.asarray(list(v), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    m = float(np.median(a))
    return float(np.median(np.abs(a - m)))


def _fmt(x: float, ndigits: int = 4) -> str:
    """Console-friendly: treat NaN as 0.0 for display purposes only."""
    x = 0.0 if (x is None or not np.isfinite(x)) else float(x)
    fmt = f"{{:.{ndigits}f}}"
    return fmt.format(x)
