from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Iterable
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ..config import settings
from ..core.splits import make_walkforward_splits
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
def _dna(individual: Tuple[str, List[str]]) -> Tuple[str, Tuple[str, ...]]:
    tkr, setup = individual
    return (str(tkr), tuple(sorted(setup or [])))


def _forward_returns(master_df: pd.DataFrame, ticker: str, k: int, price_field: str) -> pd.Series:
    col = f"{ticker}_{price_field}"
    px = pd.to_numeric(master_df.get(col), errors="coerce")
    if px is None or px.empty:
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
        return np.zeros(edges.size - 1, dtype=float)
    hist, _ = np.histogram(v, bins=edges)
    p = hist.astype(float)
    total = p.sum()
    if total == 0:
        return np.zeros_like(p)
    return p / total


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _info_gain(uncond: Iterable[float], cond: Iterable[float], edges: np.ndarray) -> float:
    pu = _histogram_probs(uncond, edges)
    pc = _histogram_probs(cond, edges)
    return max(0.0, _entropy(pu) - _entropy(pc))


def _wasserstein_1d(a: Iterable[float], b: Iterable[float]) -> float:
    xa = np.sort(pd.Series(a).dropna().astype(float).values)
    xb = np.sort(pd.Series(b).dropna().astype(float).values)
    if xa.size == 0 or xb.size == 0:
        return 0.0
    q = np.linspace(0.0, 1.0, num=max(xa.size, xb.size), endpoint=True)
    fa = np.quantile(xa, q)
    fb = np.quantile(xb, q)
    return float(np.mean(np.abs(fa - fb)))


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
    mi = float(np.sum(terms))
    return max(mi, 0.0)


def _avg_pairwise_mi(signals_df: pd.DataFrame, setup: List[str]) -> float:
    if not setup or len(setup) < 2:
        return 0.0
    cols = [c for c in setup if c in signals_df.columns]
    if len(cols) < 2:
        return 0.0
    mis = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            mis.append(_mutual_information_bool(signals_df[cols[i]], signals_df[cols[j]]))
    return float(np.mean(mis)) if mis else 0.0


def _calculate_trade_fields(edges: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Computes E_move, P_up, P_down, etc., from a distribution."""
    if edges.size < 2 or probs.size != edges.size - 1:
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
    is_oos_fold: bool = False # Add flag to control internal splitting
) -> Dict[str, float]:
    """
    Run the full metrics suite over walk-forward folds, select best horizon, and aggregate results.
    """
    n_folds = settings.validation.n_folds
    embargo_days = settings.validation.embargo_days
    band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
    min_sup_fold = max(5, settings.validation.min_support // n_folds)

    sample_idx = next(iter(unconditional_returns.values())).index
    # If it's a single OOS fold, don't re-split it. Evaluate on the whole window.
    if is_oos_fold:
        folds = [(sample_idx, sample_idx)]
    else:
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
            
            # --- Tune-up: Handle sparse triggers ---
            if r_test_cond.size < 5:
                #print(f"WARNING: Horizon {h} fold skipped: only {r_test_cond.size} test triggers (<5)")
                continue # Skip fold/horizon if insufficient test triggers

            if r_train_cond.size < min_sup_fold:
                #print(f"WARNING: Horizon {h} fold skipped: only {r_train_cond.size} train triggers (<{min_sup_fold})")
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
            if is_oos_fold:
                print(f"WARNING: No usable folds for horizon {h} in OOS window (all skipped)")
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
    """
    Overhauled evaluation function using new validation framework and modular metrics suite.
    """
    ticker, setup = individual
    price_field = settings.forecast.price_field
    horizons = list(getattr(settings.forecast, 'horizons', [settings.forecast.default_horizon]))
    band_edges = np.asarray(settings.forecast.band_edges, dtype=float)
    min_sup = int(settings.validation.min_support)

    # 1. Get valid trigger dates that meet minimum support
    trigger_dates = get_valid_trigger_dates(signals_df, setup, min_sup)
    
    if trigger_dates.empty:
        return {
            "individual": individual, "metrics": {"support_min": 0},
            "objectives": [0.0] * len(settings.ga.objectives), "rank": np.inf, "crowding_distance": 0.0,
        }

    # 2. Prepare all necessary return series
    unconditional_returns = {h: _forward_returns(master_df, ticker, h, price_field) for h in horizons}
    
    # 3. Calculate the full suite of metrics and objectives
    metrics = _calculate_objectives(trigger_dates, unconditional_returns, horizons)
    
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

    # Add negative/transformed versions for GA objectives
    metrics["crps_neg"] = -metrics.get("crps", 1e6)
    metrics["pinball_loss_neg_q10"] = -metrics.get("pinball_q10", 1e6)
    metrics["pinball_loss_neg_q90"] = -metrics.get("pinball_q90", 1e6)
    metrics["sensitivity_scan_neg"] = -metrics.get("sensitivity_delta_edge", 1e6)
    metrics["complexity_index_neg"] = -metrics.get("complexity_index", 1e6)
    target_dfa = settings.forecast.dfa_alpha_target
    metrics["dfa_alpha_closeness_neg"] = -abs(metrics.get("dfa_alpha", target_dfa) - target_dfa)
    metrics["redundancy_neg"] = -metrics.get("redundancy_mi", 1e6)
    metrics["transfer_entropy_neg"] = -metrics.get("transfer_entropy", 1e6)
    
    # Final objectives list for the GA, respecting the configurable complexity vs. redundancy choice
    ga_objectives = list(settings.ga.objectives)
    if settings.ga.complexity_objective not in ga_objectives:
        # Ensure the selected complexity/redundancy objective is in the list
        if "redundancy_neg" in ga_objectives:
            ga_objectives[ga_objectives.index("redundancy_neg")] = settings.ga.complexity_objective
        elif "complexity_index_neg" in ga_objectives:
            ga_objectives[ga_objectives.index("complexity_index_neg")] = settings.ga.complexity_objective
        else:
             ga_objectives.append(settings.ga.complexity_objective)

    objectives = [float(metrics.get(k, 0.0)) for k in ga_objectives]
    
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
        "metrics": final_metrics,
        "objectives": objectives,
        "rank": np.inf,
        "crowding_distance": 0.0,
    }


# Cache wrapper
_eval_cache: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}

def _evaluate_one_setup_cached(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
) -> Dict:
    key = _dna(individual)
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
