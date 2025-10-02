# alpha_discovery/reporting/forecast_slate.py
"""
Forecast slate generation for the new forecast-based evaluation system.
Creates trade-ready fields and enforces Top-N and max-per-ticker caps.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

from ..config import settings
from ..search.options_mapper import suggest_option_structure
from . import display_utils as du


def _ensure_dir(d: str) -> None:
    """Ensure directory exists."""
    os.makedirs(d, exist_ok=True)


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with default."""
    try:
        value = d.get(key, default)
        # Handle NaN values
        if isinstance(value, float) and np.isnan(value):
            return default
        return value
    except Exception:
        return default


def _fmt_date(value: Any) -> str:
    """Format a date-like value as YYYY-MM-DD or return empty string if NaT/invalid."""
    try:
        dt = pd.to_datetime(value, errors='coerce')
        if pd.isna(dt):
            return ""
        return dt.strftime('%Y-%m-%d')
    except Exception:
        return ""


def _trade_ready_fields(edges: np.ndarray, probs: List[float], tail_cap: float = 0.12) -> Dict[str, float]:
    """Compute E[move], P_up/down, and specific band masses from bands."""
    e = np.asarray(edges, dtype=float)
    p = np.asarray(probs, dtype=float)
    
    if e.size < 2 or p.size != e.size - 1:
        return {
            "E_move": np.nan, "P_up": np.nan, "P_down": np.nan,
            "P_3to5": np.nan, "P_gt5": np.nan, "P_m5tom3": np.nan, "P_lt5": np.nan
        }

    lo = e[:-1]
    hi = e[1:]
    mids = (lo + hi) / 2.0
    
    # Handle open-ended tails for E_move calculation
    left_open = np.isclose(lo, -999.0)
    right_open = np.isclose(hi, 999.0)
    mids[left_open] = -abs(tail_cap)
    mids[right_open] = abs(tail_cap)

    E_move = float(np.sum(mids * p))

    # Correctly calculate P_up and P_down, handling the zero-straddling bin
    p_up = 0.0
    p_down = 0.0
    
    # Bins entirely on the positive side (lo >= 0, but we use > to avoid zero in straddle)
    p_up += p[lo > 0].sum()
    
    # Bins entirely on the negative side
    p_down += p[hi < 0].sum()
    
    # Bin straddling zero
    straddle_mask = (lo < 0) & (hi > 0)
    if np.any(straddle_mask):
        straddle_prob = p[straddle_mask].sum()
        straddle_lo = lo[straddle_mask][0]
        straddle_hi = hi[straddle_mask][0]
        
        # Apportion probability based on linear interpolation (since bin is uniform)
        if (straddle_hi - straddle_lo) > 0:
            up_fraction = straddle_hi / (straddle_hi - straddle_lo)
            p_up += straddle_prob * up_fraction
            p_down += straddle_prob * (1.0 - up_fraction)

    # Helper function for probability calculations
    def _sum_mask(mask):
        return float(p[mask].sum()) if np.any(mask) else 0.0

    # Calculate probabilities of interest
    P_3to5 = _sum_mask((np.isclose(lo, 0.03) & np.isclose(hi, 0.05)))
    P_gt5 = _sum_mask(lo >= 0.05)
    P_m5tom3 = _sum_mask((np.isclose(lo, -0.05) & np.isclose(hi, -0.03)))
    P_lt5 = _sum_mask(hi <= -0.05)

    return {
        "E_move": E_move,
        "P_up": p_up,
        "P_down": p_down,
        "P_3to5": P_3to5,
        "P_gt5": P_gt5,
        "P_m5tom3": P_m5tom3,
        "P_lt5": P_lt5
    }


def _rank_score(metrics: Dict[str, Any]) -> float:
    """
    DEPRECATED: This function's logic was flawed and is being replaced.
    The HartIndex is now the primary ranking score for the forecast slate.
    This function now simply returns the HartIndex if present.
    """
    # The HartIndex is the new, primary composite rank score.
    return float(_safe_get(metrics, "hart_index", 0.0))


def _apply_ticker_caps(df: pd.DataFrame, max_per_ticker: int) -> pd.DataFrame:
    """Apply max-per-ticker caps to the dataframe."""
    if max_per_ticker <= 0:
        return df
    
    # Group by ticker and apply caps
    capped_rows = []
    for ticker, group in df.groupby("ticker"):
        # Sort by rank_score within each ticker group
        group_sorted = group.sort_values("rank_score", ascending=False)
        # Take only the top max_per_ticker
        capped_group = group_sorted.head(max_per_ticker)
        capped_rows.append(capped_group)
    
    if capped_rows:
        return pd.concat(capped_rows, ignore_index=True)
    else:
        return df


def write_forecast_slate(
    pareto_front: List[Dict], 
    signals_meta: Optional[List[Dict]] = None,
    run_dir: str = "runs"
) -> str:
    """
    Create a human-friendly forecast slate with trade-ready fields and option structure suggestions.
    
    Args:
        pareto_front: List of Pareto-optimal individuals with metrics
        signals_meta: Optional metadata for signal name mapping
        run_dir: Directory to write the CSV file
        
    Returns:
        Path to the written CSV file
    """
    _ensure_dir(run_dir)
    
    # Build signal ID to label mapping
    signal_meta_map = du.build_signal_meta_map(signals_meta)
    
    # Prepare rows for slate
    rows = []
    
    for individual in pareto_front:
        metrics = individual.get("metrics", {}) or {}
        # Unpack individual, ignoring horizon if present
        individual_tuple = individual.get("individual", (None, [], None))
        ticker, signal_ids, *_ = individual_tuple
        signal_ids = signal_ids or []
        
        # Create human-readable setup description, using pre-computed one if available
        setup_desc = metrics.get("setup_desc")
        if not setup_desc:
            # Fallback for older data or if description was missed
            setup_desc = du.desc_from_meta(signal_ids, signal_meta_map)
            if not setup_desc:
                setup_desc = du.format_setup_description(individual)
        
        # Extract band information with NaN handling
        band_edges = np.array(metrics.get("band_edges") or settings.forecast.band_edges, dtype=float)
        band_probs_raw = metrics.get("band_probs")
        
        # Handle missing or NaN band_probs: honor "no fallbacks" rule.
        if band_probs_raw is None or (isinstance(band_probs_raw, float) and np.isnan(band_probs_raw)):
            band_probs = []  # Empty list will result in NaNs from _trade_ready_fields
        else:
            band_probs = list(band_probs_raw)
        
        # Calculate trade-ready fields
        trade_fields = _trade_ready_fields(band_edges, band_probs, tail_cap=0.12)
        
        # Calculate rank score
        rank_score = _rank_score(metrics)
        
        # Get suggested option structure (reuse validated band arrays)
        suggested_structure = suggest_option_structure(band_edges, band_probs)
        
        row = {
            # Basic identification
            "ticker": ticker or "",
            "signals": "|".join(signal_ids),
            "setup_desc": setup_desc,
            "first_trigger": _fmt_date(_safe_get(metrics, "first_trigger")),
            "last_trigger": _fmt_date(_safe_get(metrics, "last_trigger")),
            "rank_score": rank_score,
            "suggested_structure": suggested_structure,
        }
        
        # Dynamically add all numerical metrics from the individual's metric dict
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
        
        # Add trade-ready fields (can overwrite if keys collide, which is fine)
        row.update({
            "E_move": trade_fields["E_move"],
            "P_up": trade_fields["P_up"],
            "P_down": trade_fields["P_down"],
            "P(3-5%)": trade_fields["P_3to5"],
            "P(>5%)": trade_fields["P_gt5"],
            "P(-5%,-3%)": trade_fields["P_m5tom3"],
            "P(<-5%)": trade_fields["P_lt5"],
        })
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("Warning: No individuals in Pareto front to write to slate")
        return ""
    
    # Apply max-per-ticker caps
    max_per_ticker = settings.reporting.slate_max_per_ticker
    df = _apply_ticker_caps(df, max_per_ticker)
    
    # Sort by rank score
    df = df.sort_values("rank_score", ascending=False)
    
    # Apply Top-N cap
    top_n = settings.reporting.slate_top_n
    if top_n > 0:
        df = df.head(top_n)

    # Remove the complexity_index column if it exists, as it's not used
    if 'complexity_index' in df.columns:
        df = df.drop(columns=['complexity_index'])
    
    # Format percentage columns
    pct_cols = ["P_up", "P_down", "P(3-5%)", "P(>5%)", "P(-5%,-3%)", "P(<-5%)"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    # Write forecast slate CSV
    slate_path = os.path.join(run_dir, "forecast_slate.csv")
    df.to_csv(slate_path, index=False, float_format="%.4f")
    
    print(f"Wrote forecast slate with {len(df)} individuals to: {slate_path}")
    
    # Write ticker coverage summary
    _write_ticker_coverage(df, run_dir)
    
    return slate_path


def _write_ticker_coverage(df: pd.DataFrame, run_dir: str) -> None:
    """Write ticker coverage summary CSV."""
    if df.empty:
        return
    
    # Build aggregation dictionary dynamically based on available columns
    agg_dict = {}
    
    # Always include row count
    agg_dict["ticker"] = "count"
    
    # Add support median if available
    if "n_trig_oos" in df.columns:
        agg_dict["n_trig_oos"] = "median"
    
    # Add rank median if available
    if "rank_score" in df.columns:
        agg_dict["rank_score"] = "median"
    
    # Add info gain column if available
    if "edge_ig_raw" in df.columns:
        agg_dict["edge_ig_raw"] = "median"
    elif "info_gain" in df.columns:
        agg_dict["info_gain"] = "median"
    
    # Add w1 effect column if available  
    if "edge_w1_raw" in df.columns:
        agg_dict["edge_w1_raw"] = "median"
    elif "w1_effect" in df.columns:
        agg_dict["w1_effect"] = "median"
    
    coverage = df.groupby("ticker").agg(agg_dict)
    
    # Rename columns to match expected output
    rename_dict = {}
    if "n_trig_oos" in coverage.columns:
        rename_dict["n_trig_oos"] = "med_support"
    if "rank_score" in coverage.columns:
        rename_dict["rank_score"] = "med_rank"
    if "edge_ig_raw" in coverage.columns:
        rename_dict["edge_ig_raw"] = "med_ig"
    elif "info_gain" in coverage.columns:
        rename_dict["info_gain"] = "med_ig"
    if "edge_w1_raw" in coverage.columns:
        rename_dict["edge_w1_raw"] = "med_w1"
    elif "w1_effect" in coverage.columns:
        rename_dict["w1_effect"] = "med_w1"
    
    coverage = coverage.rename(columns=rename_dict)
    coverage.rename(columns={"ticker": "rows"}, inplace=True)
    coverage = coverage.reset_index()
    coverage = coverage.sort_values("rows", ascending=False)
    
    coverage_path = os.path.join(run_dir, "ticker_coverage.csv")
    coverage.to_csv(coverage_path, index=False)
    
    print(f"Wrote ticker coverage summary to: {coverage_path}")


def write_forecast_slate_v2(
    pareto_front: List[Dict], 
    signals_meta: Optional[List[Dict]] = None,
    run_dir: str = "runs"
) -> str:
    """
    Alternative version of forecast slate with additional metrics and formatting.
    This version includes more detailed regime and robustness information.
    """
    _ensure_dir(run_dir)
    
    # Build signal ID to label mapping
    signal_meta_map = du.build_signal_meta_map(signals_meta)
    
    # Prepare rows for slate
    rows = []
    
    for individual in pareto_front:
        metrics = individual.get("metrics", {}) or {}
        # Unpack individual, ignoring horizon if present
        individual_tuple = individual.get("individual", (None, [], None))
        ticker, signal_ids, *_ = individual_tuple
        signal_ids = signal_ids or []
        
        # Create human-readable setup description, using pre-computed one if available
        setup_desc = metrics.get("setup_desc")
        if not setup_desc:
            # Fallback for older data or if description was missed
            setup_desc = du.desc_from_meta(signal_ids, signal_meta_map)
            if not setup_desc:
                setup_desc = du.format_setup_description(individual)
        
        # Extract band information with NaN handling
        band_edges = np.array(metrics.get("band_edges") or settings.forecast.band_edges, dtype=float)
        band_probs_raw = metrics.get("band_probs")
        
        # Handle missing or NaN band_probs by using a default uniform distribution
        if band_probs_raw is None or (isinstance(band_probs_raw, float) and np.isnan(band_probs_raw)):
            # Create a default uniform distribution based on the band edges
            if len(band_edges) > 1:
                band_probs = [1.0 / (len(band_edges) - 1)] * (len(band_edges) - 1)
                print(f"Warning: Using default uniform band_probs for {ticker} - {setup_desc}")
            else:
                # Really bad case, but let's provide a minimal default
                band_edges = np.array([-999.0, -0.10, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.10, 999.0])
                band_probs = [0.111] * 9  # Approximately uniform
                print(f"Warning: Using fallback band_probs for {ticker} - {setup_desc}")
        else:
            band_probs = list(band_probs_raw)
        
        # Calculate trade-ready fields
        trade_fields = _trade_ready_fields(band_edges, band_probs, tail_cap=0.12)
        
        # Calculate rank score
        rank_score = _rank_score(metrics)
        
        # Get suggested option structure
        band_edges = np.array(metrics.get("band_edges") or [], dtype=float)
        band_probs = list(metrics.get("band_probs") or [])
        suggested_structure = suggest_option_structure(band_edges, band_probs)
        
        row = {
            # Basic identification
            "ticker": ticker or "",
            "signals": "|".join(signal_ids),
            "setup_desc": setup_desc,
            "first_trigger": pd.to_datetime(_safe_get(metrics, "first_trigger")).strftime('%Y-%m-%d') if _safe_get(metrics, "first_trigger") else "",
            "last_trigger": pd.to_datetime(_safe_get(metrics, "last_trigger")).strftime('%Y-%m-%d') if _safe_get(metrics, "last_trigger") else "",
            "rank_score": rank_score,
            "suggested_structure": suggested_structure,
        }

        # Dynamically add all numerical metrics from the individual's metric dict
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value

        # Add trade-ready fields (can overwrite if keys collide, which is fine)
        row.update({
            "E_move": trade_fields["E_move"],
            "P_up": trade_fields["P_up"],
            "P_down": trade_fields["P_down"],
            "P(3-5%)": trade_fields["P_3to5"],
            "P(>5%)": trade_fields["P_gt5"],
            "P(-5%,-3%)": trade_fields["P_m5tom3"],
            "P(<-5%)": trade_fields["P_lt5"],
        })
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("Warning: No individuals in Pareto front to write to slate v2")
        return ""
    
    # Apply max-per-ticker caps
    max_per_ticker = settings.reporting.slate_max_per_ticker
    df = _apply_ticker_caps(df, max_per_ticker)
    
    # Sort by rank score
    df = df.sort_values("rank_score", ascending=False)
    
    # Apply Top-N cap
    top_n = settings.reporting.slate_top_n
    if top_n > 0:
        df = df.head(top_n)

    # Remove the complexity_index column if it exists, as it's not used
    if 'complexity_index' in df.columns:
        df = df.drop(columns=['complexity_index'])
    
    # Format percentage columns
    pct_cols = ["P_up", "P_down", "P(3-5%)", "P(>5%)", "P(-5%,-3%)", "P(<-5%)"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")

    # Write forecast slate CSV
    slate_path = os.path.join(run_dir, "forecast_slatev2.csv")
    df.to_csv(slate_path, index=False, float_format="%.4f")
    
    print(f"Wrote forecast slate v2 with {len(df)} individuals to: {slate_path}")
    
    # Write ticker coverage summary
    _write_ticker_coverage(df, run_dir)
    
    return slate_path