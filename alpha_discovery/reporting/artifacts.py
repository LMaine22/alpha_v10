# alpha_discovery/reporting/artifacts.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import pandas as pd
from datetime import datetime

from ..config import Settings  # do not import global settings here


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

def _format_setup_description(sol: Dict[str, Any]) -> str:
    """
    Fallback: derive a one-line description from `setup` (list/dicts) or `signals`.
    """
    desc = (sol.get("description") or "").strip()
    if desc:
        return desc

    # Handle new (ticker, [signals]) DNA
    setup_items = sol.get("individual", sol.get("setup"))
    if isinstance(setup_items, tuple) and len(setup_items) == 2:
        _, setup_signal_items = setup_items
    else:
        setup_signal_items = setup_items or sol.get("signals") or []

    parts: List[str] = []
    for item in setup_signal_items:
        if isinstance(item, dict):
            label = item.get("label") or item.get("name") or item.get("id") or str(item)
            op = item.get("op") or item.get("operator")
            parts.append(f"{label} {op}".strip() if op else str(label))
        else:
            parts.append(str(item))
    return " AND ".join(p for p in parts if p)


def _build_signal_meta_map(signals_metadata: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a metadata lookup map keyed by multiple possible identifiers for robustness.
    """
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not signals_metadata:
        return meta_map
    for m in signals_metadata:
        if not isinstance(m, dict):
            continue
        for key_field in ("signal_id", "id", "name"):
            v = m.get(key_field)
            if v is not None:
                meta_map[str(v)] = m
    return meta_map


def _extract_signal_id_token(item: Any) -> str:
    """
    Given a setup item (string or dict), return the best-guess identifier string.
    """
    if isinstance(item, dict):
        for k in ("signal_id", "id", "name", "label"):
            if item.get(k) is not None:
                return str(item[k])
        return str(item)
    return str(item)


def _desc_from_meta(setup_items: Any, meta_map: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a one-line description from signal metadata.
    """
    if isinstance(setup_items, (list, tuple)):
        it = setup_items
    elif setup_items is None:
        it = []
    else:
        it = [setup_items]

    parts: List[str] = []
    for s in it:
        key = _extract_signal_id_token(s)
        meta = meta_map.get(key, {})
        desc = (meta.get("description") or "").strip()
        if not desc:
            label = meta.get("label") or meta.get("name") or meta.get("id") or key
            op = None
            if isinstance(s, dict):
                op = s.get("operator") or s.get("op")
            if not op:
                op = meta.get("operator") or meta.get("op")
            desc = f"{label} {op}".strip() if op else str(label)
        parts.append(desc)
    return " AND ".join(p for p in parts if p)


def _signal_ids_str(setup_items: Any) -> str:
    """
    Canonical comma-separated list of signal identifiers for CSV.
    """
    if isinstance(setup_items, (list, tuple)):
        return ", ".join(_extract_signal_id_token(x) for x in setup_items)
    if setup_items is None:
        return ""
    return _extract_signal_id_token(setup_items)


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
    sum_path = os.path.join(base_dir, "pareto_front_summary.csv")
    ledg_path = os.path.join(base_dir, "pareto_front_trade_ledger.csv")

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

def save_results(
        all_fold_results: List[Dict[str, Any]],
        signals_metadata: List[Dict[str, Any]],
        settings: Settings,
        output_dir: Optional[str] = None,
) -> None:
    """
    Persist TRAIN results of a run, correctly handling the new specialized DNA.
    """
    if not all_fold_results:
        print("No solutions were found in any fold. Nothing to save.")
        return

    print("\n--- Saving In-Sample Training Results ---")

    if output_dir is None:
        output_dir = _ensure_run_dir(base="runs")
    else:
        _ensure_dir(output_dir)

    config_path: str = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(_settings_to_json(settings))
    print(f"Configuration saved to: {config_path}")

    signal_meta_map = _build_signal_meta_map(signals_metadata)

    summary_rows: List[Dict[str, Any]] = []
    all_trade_ledgers: List[pd.DataFrame] = []

    print("Generating final reports from winning setups across all folds (TRAIN-only)...")

    for i, solution in enumerate(all_fold_results):
        setup_id = f"SETUP_{i:04d}"

        # --- DEFINITIVE FIX for <null> values and key mismatches ---
        dna = solution.get('individual') or solution.get('setup', (None, []))
        if isinstance(dna, tuple) and len(dna) == 2:
            specialized_ticker, setup_signal_items = dna
        else:
            specialized_ticker = 'UNKNOWN'
            setup_signal_items = []

        direction = solution.get('direction', 'N/A')
        # --- END FIX ---

        fold_num = solution.get('fold', 0)
        rank = solution.get('rank', None)

        sig_str = _signal_ids_str(setup_signal_items)
        desc = _desc_from_meta(setup_signal_items, signal_meta_map)
        if not desc:
            desc = _format_setup_description(solution)

        perf_metrics = solution.get('metrics', {}) or {}

        ledger = solution.get('trade_ledger', pd.DataFrame())

        if isinstance(ledger, pd.DataFrame) and not ledger.empty:
            ledger_with_id = ledger.copy()
            ledger_with_id['fold'] = fold_num
            ledger_with_id['solution_rank'] = rank
            ledger_with_id['setup_id'] = setup_id

            trades_count = int(len(ledger_with_id))
            sum_capital_allocated = float(
                pd.to_numeric(ledger_with_id.get('capital_allocated', 0), errors="coerce").fillna(0).sum())
            sum_pnl_dollars = float(
                pd.to_numeric(ledger_with_id.get('pnl_dollars', 0), errors="coerce").fillna(0).sum())

            daily_returns = _portfolio_daily_returns_from_ledger(ledger_with_id, settings)
            nav_total_return_pct = _compound_total_return(daily_returns)

            first_dt = _format_date(ledger_with_id['trigger_date'].min())
            last_dt = _format_date(ledger_with_id['trigger_date'].max())

            # Using original 'best_performing_ticker' calculation logic from your file
            try:
                best_tkr_series = ledger_with_id.groupby('ticker')['pnl_pct'].mean()
                # Since this is specialized, there's only one ticker, but this is robust
                best_performing_ticker = best_tkr_series.idxmax() if not best_tkr_series.empty else specialized_ticker
            except Exception:
                best_performing_ticker = specialized_ticker

            flat_record = {
                'setup_id': setup_id,
                'fold': fold_num,
                'rank': rank,
                'specialized_ticker': specialized_ticker,
                'direction': direction,  # <<< CORRECTED KEY
                'best_performing_ticker': best_performing_ticker,
                'first_trigger_date': first_dt,
                'last_trigger_date': last_dt,
                'trades_count': trades_count,
                'sum_capital_allocated': sum_capital_allocated,
                'sum_pnl_dollars': sum_pnl_dollars,
                'nav_total_return_pct': nav_total_return_pct,
                'final_nav': float(_get_base_portfolio_capital(settings) * (1.0 + nav_total_return_pct)),
                'description': desc,
                'signal_ids': sig_str,
                **perf_metrics
            }
            summary_rows.append(flat_record)
            all_trade_ledgers.append(ledger_with_id)

        else:  # Handle case where there is no ledger
            flat_record = {
                'setup_id': setup_id,
                'fold': fold_num,
                'rank': rank,
                'specialized_ticker': specialized_ticker,
                'direction': direction,  # <<< CORRECTED KEY
                'best_performing_ticker': specialized_ticker,
                'first_trigger_date': '',
                'last_trigger_date': '',
                'trades_count': 0,
                'sum_capital_allocated': 0.0,
                'sum_pnl_dollars': 0.0,
                'nav_total_return_pct': 0.0,
                'final_nav': float(_get_base_portfolio_capital(settings)),
                'description': desc,
                'signal_ids': sig_str,
                **perf_metrics
            }
            summary_rows.append(flat_record)

    summary_df = pd.DataFrame(summary_rows)

    ordered_cols = [
        'setup_id', 'fold', 'rank', 'specialized_ticker', 'direction', 'description',
        'signal_ids', 'sortino_lb', 'expectancy', 'support', 'sharpe_lb', 'omega_ratio',
        'max_drawdown', 'trades_count', 'sum_pnl_dollars', 'nav_total_return_pct', 'final_nav',
        'first_trigger_date', 'last_trigger_date', 'best_performing_ticker'
    ]

    existing_cols = [c for c in ordered_cols if c in summary_df.columns]
    remaining_cols = sorted([c for c in summary_df.columns if c not in existing_cols])
    summary_df = summary_df[existing_cols + remaining_cols]

    if 'nav_total_return_pct' in summary_df.columns:
        summary_df['nav_total_return_pct'] = (
                pd.to_numeric(summary_df['nav_total_return_pct'], errors="coerce").fillna(0.0) * 100.0
        ).round(2).map(lambda x: f"{x:.2f} %")

    summary_path: str = os.path.join(output_dir, 'pareto_front_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"Enriched summary saved to: {summary_path}")

    if all_trade_ledgers:
        full_trade_ledger_df = pd.concat(all_trade_ledgers, ignore_index=True)

        ledger_order = [
            "setup_id", "fold", "solution_rank",
            "trigger_date", "exit_date",
            "ticker", "horizon_days", "direction", "exit_policy_id", "exit_reason",
            "strike", "entry_underlying", "exit_underlying",
            "entry_iv", "exit_iv",
            "entry_option_price", "exit_option_price",
            "contracts", "capital_allocated", "capital_allocated_used",
            "pnl_dollars", "pnl_pct",
        ]
        existing_ledger_cols = [c for c in ledger_order if c in full_trade_ledger_df.columns]
        remaining_cols = [c for c in full_trade_ledger_df.columns if c not in existing_ledger_cols]
        full_trade_ledger_df = full_trade_ledger_df.reindex(columns=existing_ledger_cols + remaining_cols)

        ledger_path: str = os.path.join(output_dir, 'pareto_front_trade_ledger.csv')
        full_trade_ledger_df.to_csv(ledger_path, index=False, float_format='%.6f')
        print(f"Full options trade ledger saved to: {ledger_path}")
    else:
        print("No trade ledgers found to save.")

