# alpha_discovery/reporting/pareto_csv.py
"""
Pareto front CSV generation for the new forecast-based evaluation system.
"""

from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from ..config import settings
from ..reporting import display_utils as du


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


def _format_band_probs(edges: np.ndarray, probs: List[float]) -> str:
    """Format band probabilities as a more readable string."""
    if edges.size < 2 or len(probs) != edges.size - 1:
        return ""
    
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        prob = probs[i] if i < len(probs) else 0.0
        
        lo_str = "-inf" if lo < -99 else f"{lo:.1%}"
        hi_str = "+inf" if hi > 99 else f"{hi:.1%}"
        
        labels.append(f"[{lo_str},{hi_str}):{prob:.1%}")
    
    return " | ".join(labels)


def _format_objectives_for_csv(objectives: Optional[List[float]]) -> str:
    """Format objectives list as a rounded string for CSV display."""
    if not objectives or not isinstance(objectives, list):
        return ""
    try:
        return ", ".join([f"{x:.3f}" for x in objectives])
    except (TypeError, ValueError):
        return str(objectives)  # Fallback to default string representation


def _format_list_as_json_string(data: Optional[List[float]]) -> str:
    """Format a list of numbers as a JSON string."""
    if data is None or not isinstance(data, (list, np.ndarray)):
        return ""
    try:
        # Use a compact JSON representation
        return json.dumps([round(x, 6) for x in data], separators=(',', ':'))
    except (TypeError, ValueError):
        return ""


def write_pareto_csv(results_df: pd.DataFrame, signals_metadata: List[Dict], run_dir: str) -> None:
    """
    Writes the main Pareto front CSV, now enriched with ELV scores and labels.
    """
    # This function can be simplified as most data is already in results_df
    # We just need to add the human-readable description.
    
    meta_map = du.build_signal_meta_map(signals_metadata)
    
    def get_desc(row):
        # individual is stored as a string, e.g., "('TICKER', ('SIG_001', 'SIG_002'))"
        try:
            _, signals = eval(row['individual'])
            return du.desc_from_meta(signals, meta_map)
        except:
            return "Invalid setup format"

    df = results_df.copy()
    df['description'] = df.apply(get_desc, axis=1)
    
    # Define column order for readability
    core_cols = ['elv', 'individual', 'description']
    elv_components = ['edge_oos', 'live_tr_prior', 'coverage_factor', 'penalty_adj']
    labels = ['dormant_flag', 'specialist_flag', 'pass_cv_gates']
    
    other_cols = [c for c in df.columns if c not in core_cols + elv_components + labels]
    
    final_cols = core_cols + elv_components + labels + sorted(other_cols)
    df = df[final_cols]

    # The main pareto file is now saved directly from artifacts.py
    # This function can be deprecated or kept for specific diagnostic purposes.
    # For now, we assume it's part of the main artifacts save.
    pass


def write_diagnostics(
    pareto_front: List[Dict], 
    run_dir: str = "runs"
) -> None:
    """
    Write diagnostic files for each individual in the Pareto front.
    
    Args:
        pareto_front: List of Pareto-optimal individuals with metrics
        run_dir: Directory to write diagnostic files
    """
    diag_dir = os.path.join(run_dir, "diagnostics")
    _ensure_dir(diag_dir)
    
    for i, individual in enumerate(pareto_front):
        metrics = individual.get("metrics", {}) or {}
        ticker, signal_ids = individual.get("individual", (None, []))
        signal_ids = signal_ids or []
        
        # Create calibration diagnostic
        band_edges = np.array(metrics.get("band_edges") or [], dtype=float)
        band_probs = list(metrics.get("band_probs") or [])
        
        if band_edges.size > 0 and len(band_probs) == band_edges.size - 1:
            # Create band labels
            band_labels = []
            for j in range(len(band_edges) - 1):
                lo, hi = band_edges[j], band_edges[j + 1]
                band_labels.append(f"[{lo:.2%},{hi:.2%})")
            
            # Write calibration diagnostic
            calib_df = pd.DataFrame({
                "band_label": band_labels,
                "p_hat_conditional_all": band_probs,
            })
            
            ticker_safe = (ticker or "UNK").replace(" ", "_")
            signal_hash = hash(tuple(sorted(signal_ids)))
            calib_path = os.path.join(diag_dir, f"calibration__{ticker_safe}__{signal_hash}.csv")
            calib_df.to_csv(calib_path, index=False)
