# alpha_discovery/reporting/artifacts.py

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import pandas as pd
from datetime import datetime

from ..config import Settings  # do not import global settings here
from . import display_utils as du
from ..core.splits import HybridSplits
from .diagnostics import write_split_audit
from .pareto_csv import write_pareto_csv
from .forecast_slate import write_forecast_slate
from ..eval.regime import RegimeModel


# ------------ run directory helpers ------------

def _find_latest_run_dir(base: str = "runs") -> str:
    """
    Return the most recently modified run directory under `base`, or "" if none.
    """
    if not os.path.isdir(base):
        return ""
    cand: List[Tuple[float, str]] = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            try:
                cand.append((os.path.getmtime(p), p))
            except OSError:
                pass
    if not cand:
        return ""
    cand.sort(reverse=True)
    return cand[0][1]


def _ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.
    """
    os.makedirs(path, exist_ok=True)


def _ensure_run_dir(base: str = "runs", specified_dir: Optional[str] = None) -> str:
    """
    Use the specified_dir if provided, else create a fresh timestamped one.
    """
    if specified_dir and os.path.isdir(specified_dir):
        return specified_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_seed{Settings().ga.seed}_{ts}"
    rd = os.path.join(base, folder_name)
    _ensure_dir(rd)
    return rd


def _load_reference_columns(run_dir: str, candidates: Sequence[str]) -> Optional[List[str]]:
    """
    Try to infer a reference CSV schema from known files/folders in `run_dir`.
    """
    # 1) Explicit file paths in `run_dir`
    for rel in candidates:
        path = os.path.join(run_dir, rel)
        if os.path.isfile(path):
            try:
                return pd.read_csv(path, nrows=0).columns.tolist()
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
    # 2) Any CSV inside candidate folders
    for rel in candidates:
        p = os.path.join(run_dir, rel)
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.lower().endswith(".csv"):
                    try:
                        return pd.read_csv(os.path.join(p, name), nrows=0).columns.tolist()
                    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                        continue
    return None


def _align_to_reference(df: pd.DataFrame, ref_cols: Optional[List[str]]) -> pd.DataFrame:
    """
    Add missing ref columns (filled with NA) and order columns to match `ref_cols`.
    If `ref_cols` is None, return df as-is.
    """
    if ref_cols is None:
        return df
    out = df.copy()
    for c in ref_cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[list(ref_cols)]


# ------------ presentation helpers ------------

# All presentation helpers moved to display_utils.py

# ------------ per-fold TRAIN artifact writers ------------

def write_fold_pareto_summary(pareto_df: pd.DataFrame, fold_num: int, base: str = "runs") -> str:
    run_dir: str = _ensure_run_dir(base)
    fold_dir: str = os.path.join(run_dir, "folds", f"fold_{fold_num:02d}")
    _ensure_dir(fold_dir)
    ref_cols = _load_reference_columns(run_dir, ["pareto_front_summary.csv"])
    out = _align_to_reference(pareto_df, ref_cols)
    out_path: str = os.path.join(fold_dir, "pareto_summary.csv")
    out.to_csv(out_path, index=False)
    return out_path


def write_fold_ledger(ledger_df: pd.DataFrame, fold_num: int, base: str = "runs") -> str:
    run_dir: str = _ensure_run_dir(base)
    fold_dir: str = os.path.join(run_dir, "folds", f"fold_{fold_num:02d}")
    _ensure_dir(fold_dir)
    ref_cols = _load_reference_columns(run_dir, ["pareto_front_trade_ledger.csv"])
    out = _align_to_reference(ledger_df, ref_cols)
    out_path: str = os.path.join(fold_dir, "pareto_ledger.csv")
    out.to_csv(out_path, index=False)
    return out_path


def materialize_per_fold_artifacts(base_dir: str) -> None:
    """
    Splits the global training CSVs into per-fold files.
    """
    sum_path = os.path.join(base_dir, "pareto", "pareto_front_summary.csv")
    ledg_path = os.path.join(base_dir, "pareto", "pareto_front_trade_ledger.csv")

    folds_dir = os.path.join(base_dir, "folds")
    _ensure_dir(folds_dir)

    if os.path.isfile(sum_path):
        try:
            df = pd.read_csv(sum_path)
            if "fold" in df.columns:
                for fnum, g in df.groupby("fold"):
                    fold_dir = os.path.join(folds_dir, f"fold_{int(fnum):02d}")
                    _ensure_dir(fold_dir)
                    g.to_csv(os.path.join(fold_dir, "pareto_summary.csv"), index=False)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    if os.path.isfile(ledg_path):
        try:
            df = pd.read_csv(ledg_path)
            if "fold" in df.columns:
                for fnum, g in df.groupby("fold"):
                    fold_dir = os.path.join(folds_dir, f"fold_{int(fnum):02d}")
                    _ensure_dir(fold_dir)
                    g.to_csv(os.path.join(fold_dir, "pareto_ledger.csv"), index=False)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            pass


# ------------ misc utils ------------

def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def _format_date(dt: Any) -> str:
    try:
        return pd.Timestamp(dt).strftime('%Y-%m-%d')
    except Exception:
        return ""


def _settings_to_json(settings_model: Settings) -> str:
    """Safely dump pydantic settings to JSON."""
    if hasattr(settings_model, "model_dump_json"):
        return settings_model.model_dump_json(indent=4)
    return settings_model.json(indent=4)


def _get_base_portfolio_capital(settings_model: Settings) -> float:
    try:
        reporting = getattr(settings_model, "reporting", None)
        if reporting is not None and hasattr(reporting, "base_capital_for_portfolio"):
            return float(reporting.base_capital_for_portfolio)
    except Exception:
        pass
    return 100_000.0


def _portfolio_daily_returns_from_ledger(filtered_ledger: pd.DataFrame, settings_model: Settings) -> pd.Series:
    """
    NAV-based daily portfolio returns using realized P&L on exit dates (TRAIN-only).
    """
    if not isinstance(filtered_ledger, pd.DataFrame) or filtered_ledger.empty:
        return pd.Series(dtype=float)

    df = filtered_ledger.copy()
    if 'trigger_date' not in df.columns or 'exit_date' not in df.columns or 'pnl_dollars' not in df.columns:
        return pd.Series(dtype=float)

    df["trigger_date"] = pd.to_datetime(df["trigger_date"], errors='coerce').dt.normalize()
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors='coerce').dt.normalize()
    df = df.dropna(subset=['trigger_date', 'exit_date'])

    if df.empty:
        return pd.Series(dtype=float)

    start = min(df["exit_date"].min(), df["trigger_date"].min())
    end = max(df["exit_date"].max(), df["trigger_date"].max())
    if pd.isna(start) or pd.isna(end):
        return pd.Series(dtype=float)

    idx = pd.date_range(start=start, end=end, freq="B")
    realized = (
        df.groupby("exit_date")["pnl_dollars"]
        .sum(min_count=1)
        .reindex(idx, fill_value=0.0)
        .astype(float)
    )
    base_nav = _get_base_portfolio_capital(settings_model)
    cum = realized.cumsum()
    nav_prev = (base_nav + cum.shift(1)).fillna(base_nav).astype(float)
    denom = nav_prev.replace(0.0, pd.NA)
    daily_ret = (realized / denom).fillna(0.0).astype(float)
    return daily_ret


def _compound_total_return(daily_returns: pd.Series) -> float:
    if daily_returns is None or daily_returns.empty:
        return 0.0
    try:
        return float((1.0 + daily_returns).prod() - 1.0)
    except Exception:
        return 0.0


# ------------ main saver (TRAIN-only artifacts) ------------

def _df_to_individual_list(df: pd.DataFrame) -> list[Dict[str, Any]]:
    rows = []
    for _, r in df.iterrows():
        try:
            individual = eval(r['individual'])
        except Exception:
            continue
        metrics = r.to_dict()
        rows.append({"individual": individual, "metrics": metrics})
    return rows

def save_results(
    final_results_df: pd.DataFrame,
    signals_metadata: List[Dict[str, Any]],
    run_dir: str,
    splits: HybridSplits,
    settings: Settings,
    regime_model: Optional[RegimeModel] = None,
    simulation_summary: Optional[pd.DataFrame] = None,
    simulation_ledger: Optional[pd.DataFrame] = None,
    correlation_report: Optional[str] = None
) -> str:
    """
    Main entry point for saving all run artifacts.
    This orchestrates calls to specialized writers for each artifact.
    
    Returns:
        The path to the generated forecast slate CSV.
    """
    print("\n--- Saving All Run Artifacts ---")
    _ensure_dir(run_dir)

    # 1. Save config and split audit
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(_settings_to_json(settings))
    print(f"Configuration saved to: {config_path}")
    
    write_split_audit(splits, run_dir)
    
    # Add human-readable description
    meta_map = du.build_signal_meta_map(signals_metadata)
    def get_desc(row):
        try:
            # Handle both tuple and string representations
            if isinstance(row['individual'], str):
                _, signals = eval(row['individual'])
            else:
                _, signals = row['individual']
            return du.desc_from_meta(signals, meta_map)
        except Exception as e:
            print(f"Error getting description: {e}")
            return "Invalid setup"
    final_results_df['setup_desc'] = final_results_df.apply(get_desc, axis=1)

    # Convert individual tuples to clean string representation for CSV
    # but keep the original tuple form for passing to write_forecast_slate
    df_for_csv = final_results_df.copy()
    
    def clean_individual_str(ind):
        if isinstance(ind, str):
            return ind
        ticker, signals = ind
        ticker_str = str(ticker) if hasattr(ticker, '__str__') else ticker
        signals_list = [str(s) if hasattr(s, '__str__') else s for s in signals]
        return str((ticker_str, signals_list))
    
    df_for_csv['individual'] = df_for_csv['individual'].apply(clean_individual_str)
    pareto_path = os.path.join(run_dir, "pareto_front_elv.csv")
    df_for_csv.to_csv(pareto_path, index=False, float_format='%.4f')
    print(f"ELV-scored Pareto front saved to: {pareto_path}")

    # Save the Forecast Slate
    # Convert DataFrame back to list of dicts for the writer function
    pf_list = []
    for _, row in final_results_df.iterrows():
        metrics = row.to_dict()
        # The 'individual' is already in the correct tuple format in final_results_df
        individual = metrics.pop('individual')
        pf_list.append({'individual': individual, 'metrics': metrics})

    forecast_slate_path = write_forecast_slate(pf_list, signals_metadata, run_dir)
    
    # Save Regime Diagnostics
    if regime_model:
        write_regime_artifacts(regime_model, run_dir)

    # Save simulation artifacts if they exist
    if simulation_summary is not None:
        write_simulation_summary(simulation_summary, run_dir)
    if simulation_ledger is not None:
        write_simulation_ledger(simulation_ledger, run_dir)
    if correlation_report is not None:
        write_correlation_report(correlation_report, run_dir)
        
    return forecast_slate_path


def write_simulation_summary(summary_df: pd.DataFrame, run_dir: str):
    """Saves the simulation summary to a CSV file."""
    path = os.path.join(run_dir, "simulation_summary.csv")
    summary_df.to_csv(path, index=False, float_format="%.4f")
    print(f"Simulation summary saved to: {path}")

def write_simulation_ledger(ledger_df: pd.DataFrame, run_dir: str):
    """Saves the full simulation trade ledger to a CSV file."""
    path = os.path.join(run_dir, "simulation_trade_ledger.csv")
    ledger_df.to_csv(path, index=False, float_format="%.4f")
    print(f"Simulation trade ledger saved to: {path}")

def write_correlation_report(report_str: str, run_dir: str):
    """Saves the HartIndex correlation report to a text file."""
    diag_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "hart_index_correlation.txt")
    with open(path, 'w') as f:
        f.write(report_str)
    print(f"HartIndex correlation report saved to: {path}")


def write_regime_artifacts(regime_model: RegimeModel, run_dir: str):
    """Saves the regime model parameters and summary to disk."""
    diag_dir = os.path.join(run_dir, "diagnostics", "regimes")
    os.makedirs(diag_dir, exist_ok=True)
    
    # Save model params to JSON
    meta = {
        "anchor_model_id": regime_model.anchor_model_id,
        "n_regimes": regime_model.n_regimes,
        "features_used": regime_model.features_used,
        "means": regime_model.original_means.tolist(),
        "covariances": regime_model.model.covars_.tolist(),
        "scaler_mean": regime_model.scaler.mean_.tolist(),
        "scaler_scale": regime_model.scaler.scale_.tolist()
    }
    with open(os.path.join(diag_dir, "regime_model.json"), 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"Regime model artifacts saved to: {diag_dir}")

