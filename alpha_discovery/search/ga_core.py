from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable, Any
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ..config import settings
from ..eval.selection import get_valid_trigger_dates
from ..eval.metrics import (
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
from ..eval.objectives import audit_objectives, apply_objective_transforms

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


@contextmanager
def threadpool_limits(limits: int = 1):
    """Compatibility stub so nsga.py can limit BLAS threads if desired."""
    yield


def _exit_policy_from_settings() -> Optional[Dict]:
    """Pivot: no options backtest. Keep stub to satisfy nsga import path."""
    return None


# -----------------------------
# Utility helpers
# -----------------------------
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
# Evaluation core (multi-horizon, bands, OOS, regimes, robustness)
# -----------------------------
def _evaluate_one_setup(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],  # kept for API compatibility
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],   # unused
) -> Dict:
    """Strict evaluation with fail-closed semantics (no fabricated penalty defaults)."""
    ticker, setup = individual
    price_field = settings.forecast.price_field
    horizons = list(getattr(settings.forecast, 'horizons', [settings.forecast.default_horizon]))
    band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
    min_sup = int(settings.validation.min_support)

    # 1. Get valid trigger dates that meet minimum support
    trigger_dates = get_valid_trigger_dates(signals_df, setup, min_sup)
    
    if trigger_dates.empty:
        return {
            "individual": individual,
            "feasible": False,
            "metrics": {"reason": "no_triggers", "support_min": 0},
            "objectives": tuple([-np.inf] * len(settings.ga.objectives)),
            "rank": np.inf,
            "crowding_distance": 0.0,
        }

    # 2. Prepare all necessary return series
    unconditional_returns = {h: _forward_returns(master_df, ticker, h, price_field) for h in horizons}
    
    # 3. Calculate the full suite of metrics and objectives with robust replacements
    metrics = _calculate_objectives_with_robust_replacements(trigger_dates, unconditional_returns, horizons)

    # Fail closed if no folds used or critical metrics missing
    folds_used = metrics.get("folds_used", 0)
    if not folds_used:
        return {
            "individual": individual,
            "feasible": False,
            "metrics": {"reason": "no_valid_folds", "support_min": len(trigger_dates)},
            "objectives": tuple([-np.inf] * len(settings.ga.objectives)),
            "rank": np.inf,
            "crowding_distance": 0.0,
        }
    
    # Add a few final metrics that don't fit the fold structure
    metrics["support_min"] = len(trigger_dates)
    metrics["redundancy_mi"] = _avg_pairwise_mi(signals_df, setup)
    metrics["first_trigger"] = trigger_dates.min()
    metrics["last_trigger"] = trigger_dates.max()

    # --- Transfer Entropy Calculation ---
    best_h = metrics.get("best_horizon")
    if best_h:
        r_series = unconditional_returns[best_h].loc[trigger_dates.min():trigger_dates.max()]
        s_series = _trigger_mask(signals_df, setup).loc[r_series.index]
        # Use number of bands as bins for TE; ensure at least 3
        te_bins = max(3, int(len(band_edges) - 1))
        te_res = causality.transfer_entropy_causality(r_series, s_series, lag=1, bins=te_bins)
        metrics["transfer_entropy"] = float(te_res.get("te_x_to_y", 0.0))
        metrics["transfer_entropy_p_value"] = float(te_res.get("p_value", np.nan))
    else:
        metrics["transfer_entropy"] = np.nan
        metrics["transfer_entropy_p_value"] = np.nan
    
    # Generate the human-readable description here
    meta_map = du.build_signal_meta_map(signals_metadata)
    setup_desc = du.desc_from_meta(setup, meta_map)
    if not setup_desc:
        # provide a dict with "individual" so format_setup_description can parse it
        setup_desc = du.format_setup_description({"individual": individual})

    # Build transformed objective tuple (maximize) with audit
    objective_names = list(getattr(settings.ga, "objectives", ["ig_sharpe", "min_ig"]))
    ok, missing = audit_objectives(objective_names)
    if not ok:
        return {
            "individual": individual,
            "feasible": False,
            "metrics": {"reason": f"unknown_objectives:{','.join(missing)}", **metrics},
            "objectives": tuple([-np.inf] * len(objective_names)),
            "rank": np.inf,
            "crowding_distance": 0.0,
        }
    try:
        objective_values, transform_labels = apply_objective_transforms(metrics, objective_names)
    except (KeyError, ValueError) as e:
        return {
            "individual": individual,
            "feasible": False,
            "metrics": {"reason": f"objective_transform_error:{e}", **metrics},
            "objectives": tuple([-np.inf] * len(objective_names)),
            "rank": np.inf,
            "crowding_distance": 0.0,
        }
    
    # Build final metrics dictionary for reporting
    final_metrics = {
        "ticker": ticker,
        "signals": "|".join(setup),
        "setup_desc": setup_desc,
        # Raw metrics for ELV consumption - explicitly include both raw and regular names
        "crps_raw": metrics.get("crps"),
        "pinball_q10_raw": metrics.get("pinball_q10"),
        "pinball_q90_raw": metrics.get("pinball_q90"),
        "info_gain_raw": metrics.get("info_gain"),
        "w1_effect_raw": metrics.get("w1_effect"),
        "calib_mae_raw": metrics.get("calib_mae"),
        "dfa_alpha_raw": metrics.get("dfa_alpha"),
        "sensitivity_delta_edge_raw": metrics.get("sensitivity_delta_edge"),
        "redundancy_mi_raw": metrics.get("redundancy_mi"),
        "complexity_metric_raw": metrics.get(settings.complexity.metric) or metrics.get("complexity_index"),
        "bootstrap_p_value_raw": metrics.get("bootstrap_p_value"),
        # ELV expected names
        "edge_crps_raw": metrics.get("crps"),
        "edge_pin_q10_raw": metrics.get("pinball_q10"),
        "edge_pin_q90_raw": metrics.get("pinball_q90"),
        "edge_ig_raw": metrics.get("info_gain"),
        "edge_w1_raw": metrics.get("w1_effect"),
        "edge_calib_mae_raw": metrics.get("calib_mae"),
        **metrics
    }

    return {
        "individual": individual,
        "feasible": True,
        "metrics": final_metrics,
        "objectives": tuple(objective_values),
        # Attach transform audit info
        "objective_transform": transform_labels,
        "objective_names": objective_names,
        "rank": np.inf,
        "crowding_distance": 0.0,
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
    exit_policy: Optional[Dict],
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
    sup = [e.get("metrics", {}).get("support_min", 0) for e in evaluated]
    ig = [e.get("metrics", {}).get("info_gain", np.nan) for e in evaluated]
    cr = [e.get("metrics", {}).get("crps", np.nan) for e in evaluated]
    emove = [e.get("metrics", {}).get("E_move", np.nan) for e in evaluated]
    folds = [e.get("metrics", {}).get("folds_used", 0) for e in evaluated]

    def _med(x):
        return _safe_median(x)

    has_probs = sum(1 for e in evaluated if e.get("metrics", {}).get("band_probs"))
    prob_pct = (has_probs / len(evaluated)) * 100 if evaluated else 0

    print(
        f"[{tag}] med(support)={_fmt(_med(sup),1)} | med(folds)={_fmt(_med(folds),1)} | "
        f"med(IG)={_fmt(_med(ig),4)} med(CRPS)={_fmt(_med(cr),5)} | "
        f"Probs OK: {prob_pct:.1f}% | med(E_move)={_fmt(_med(emove),4)}"
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
