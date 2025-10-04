# alpha_discovery/reporting/artifacts.py

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import pandas as pd
import numpy as np
from datetime import datetime

from pandas.tseries.offsets import BDay

# ------------ priors helper ------------

OBJECTIVE_FALLBACKS = {
    "dsr": ["dsr", "dsr_raw", "sharpe"],
    "bootstrap_calmar_lb": ["bootstrap_calmar_lb", "bootstrap_calmar_lb_raw", "calmar", "mar_ratio"],
    "bootstrap_profit_factor_lb": ["bootstrap_profit_factor_lb", "bootstrap_profit_factor_lb_raw", "profit_factor", "expectancy"],
}


def _resolve_metric(metrics: Dict[str, Any], key: str) -> Tuple[float, str]:
    for candidate in OBJECTIVE_FALLBACKS.get(key, [key]):
        val = metrics.get(candidate)
        if isinstance(val, (int, float)) and np.isfinite(val):
            return float(val), candidate
    return 0.0, "unavailable"


def _sanitized_growth(ledger: Optional[pd.DataFrame]) -> float:
    if ledger is None or ledger.empty:
        return 0.0
    returns = pd.to_numeric(ledger.get('pnl_pct'), errors='coerce').dropna()
    if returns.empty:
        return 0.0
    clipped = returns.clip(lower=-0.95)
    return float(np.expm1(np.log1p(clipped).sum()))

def _build_prior_record(
    setup_id: str,
    ticker: str,
    signals: List[str],
    direction: str,
    metrics: Dict[str, Any],
    ledger: Optional[pd.DataFrame],
    data_end: Optional[pd.Timestamp],
    ga_settings: Any,
) -> Dict[str, Any]:
    as_of = pd.to_datetime(data_end) if data_end is not None else None
    if as_of is None or pd.isna(as_of):
        as_of = pd.Timestamp.utcnow().normalize()

    trades_total = 0
    trades_12m = 0
    trades_6m = 0
    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
        ledger_dates = pd.to_datetime(ledger.get('trigger_date'), errors='coerce')
        ledger_dates = ledger_dates.dropna()
        trades_total = int(len(ledger_dates))
        if trades_total:
            trades_12m = int((ledger_dates >= (as_of - BDay(252))).sum())
            trades_6m = int((ledger_dates >= (as_of - BDay(126))).sum())

    replacements: Dict[str, bool] = {}

    def _finite(value: Any, key: str) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            replacements[key] = True
            return 0.0
        if not np.isfinite(val):
            replacements[key] = True
            return 0.0
        return float(val)

    psr_val = _finite(metrics.get('psr'), 'psr')
    max_dd_val = _finite(metrics.get('max_drawdown'), 'max_drawdown')
    max_dd_threshold = float(getattr(ga_settings, 'max_drawdown_threshold', -0.6))
    min_psr = float(getattr(ga_settings, 'min_psr', 0.60))

    last_trigger = metrics.get('last_trigger')
    first_trigger = metrics.get('first_trigger')

    coverage_ratio = _finite(metrics.get('fold_coverage_ratio'), 'fold_coverage_ratio')
    coverage_target = _finite(metrics.get('coverage_target_ratio'), 'coverage_target_ratio')

    flags = {
        'psr_ok': ('psr' not in replacements) and psr_val >= min_psr,
        'dd_ok': ('max_drawdown' not in replacements) and max_dd_val >= max_dd_threshold,
        'coverage_pct': coverage_ratio,
        'coverage_ok': coverage_ratio >= coverage_target if coverage_target > 0 else True,
        'n_outer_with_support': int(metrics.get('fold_count') or 0),
        'eligible': bool(metrics.get('eligible', False)),
        'dedup_applied': bool(metrics.get('dedup_applied', False)),
    }

    record = {
        'setup_id': setup_id,
        'ticker': ticker,
        'signals': '|'.join(signals),
        'signals_list': signals,
        'signals_fingerprint': metrics.get('signals_fingerprint'),
        'options_structure_keys': metrics.get('options_structure_keys', []),
        'individual': [ticker, signals],
        'direction': direction,
        'horizon': metrics.get('label_horizon'),
        'regime': metrics.get('regime', 'ALL'),
        'dsr': _finite(metrics.get('dsr'), 'dsr'),
        'bootstrap_calmar_lb': _finite(metrics.get('bootstrap_calmar_lb'), 'bootstrap_calmar_lb'),
        'bootstrap_profit_factor_lb': _finite(metrics.get('bootstrap_profit_factor_lb'), 'bootstrap_profit_factor_lb'),
        'expectancy': _finite(metrics.get('expectancy'), 'expectancy'),
        'trades_total': trades_total,
        'trades_12m': trades_12m,
        'trades_6m': trades_6m,
        'first_trigger': first_trigger,
        'last_trigger': last_trigger,
        'fold_count': int(metrics.get('fold_count') or 0),
        'fold_coverage_ratio': coverage_ratio,
        'psr': psr_val,
        'max_drawdown': max_dd_val,
        'penalty_scalar': _finite(metrics.get('penalty_scalar'), 'penalty_scalar'), 
        'soft_penalties': metrics.get('soft_penalties', {}),
        'eligible': bool(metrics.get('eligible', False)),
        'eligibility_reasons': metrics.get('eligibility_reasons', []),
        'objective_sources': metrics.get('objective_sources', {}),
        'flags': flags,
        'dup_suppressed_total': int(metrics.get('dup_suppressed_total', 0) or 0),
        'dup_merged_total': int(metrics.get('dup_merged_total', 0) or 0),
        'dedup_applied': bool(metrics.get('dedup_applied', False)),
    }

    if replacements:
        record['flags'] = dict(flags, metric_replacements=sorted(k for k, v in replacements.items() if v))

    score_values: Dict[str, float] = {}
    score_sources: Dict[str, str] = {}
    for key in ('dsr', 'bootstrap_calmar_lb', 'bootstrap_profit_factor_lb'):
        val, source = _resolve_metric(metrics, key)
        score_values[f'{key}_score'] = val
        score_sources[f'{key}_source'] = source
    record.update(score_values)
    record.update(score_sources)

    recency_days = None
    if last_trigger:
        try:
            recency_days = int((as_of - pd.to_datetime(last_trigger)).days)
        except Exception:
            recency_days = None
    record['recency_days'] = recency_days
    if isinstance(recency_days, int) and recency_days >= 0:
        half_life = max(1, int(getattr(getattr(settings, 'dts', object()), 'dormancy_half_life_days', 180)))
        record['recency_weight'] = float(max(0.05, 0.5 ** (recency_days / half_life)))
    else:
        record['recency_weight'] = 1.0

    record['penalty_scalar'] = float(max(1.0, record['penalty_scalar'] or 1.0))

    return record

from ..config import Settings, settings
from ..utils.trade_keys import canonical_signals_fingerprint, dedupe_trade_ledger
from . import display_utils as du
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


# ------------ main saver (PAWF/NPWF artifacts) ------------

def save_results(
    final_results_df: pd.DataFrame,
    signals_metadata: List[Dict[str, Any]],
    run_dir: str,
    splits: Any,
    settings: Settings,
    data_end: Optional[pd.Timestamp] = None,
    regime_model: Optional[RegimeModel] = None,
    simulation_summary: Optional[pd.DataFrame] = None,
    simulation_ledger: Optional[pd.DataFrame] = None,
    correlation_report: Optional[str] = None
) -> str:
    """Persist PAWF/NPWF discovery outputs and DTS-ready artifacts."""
    print("\n--- Saving All Run Artifacts ---")
    _ensure_dir(run_dir)

    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(_settings_to_json(settings))
    print(f"Configuration saved to: {config_path}")

    meta_map = du.build_signal_meta_map(signals_metadata)

    summary_records: List[Dict[str, Any]] = []
    ledger_frames: List[pd.DataFrame] = []
    priors_records: List[Dict[str, Any]] = []

    data_end_override = pd.to_datetime(data_end) if data_end is not None else None
    data_end_computed: Optional[pd.Timestamp] = None
    if isinstance(splits, dict):
        split_candidates = []
        for spec in splits.get('splits', []) or []:
            candidate = spec.get('test_end') or spec.get('train_end')
            if candidate:
                try:
                    split_candidates.append(pd.to_datetime(candidate))
                except Exception:
                    continue
        if split_candidates:
            data_end_computed = max(split_candidates)

    data_end_final = data_end_override or data_end_computed

    if data_end_final is None:
        try:
            data_end_final = pd.to_datetime(getattr(settings.data, 'end_date', None))
        except Exception:
            data_end_final = None

    def _normalize_individual(ind: Any) -> Tuple[str, List[str]]:
        if isinstance(ind, str):
            ind = eval(ind)
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            ticker = str(ind[0])
            signals = [str(s) for s in ind[1]] if isinstance(ind[1], (list, tuple)) else []
            return ticker, signals
        return str(ind), []

    for idx, row in final_results_df.iterrows():
        metrics = dict(row.get('metrics', {}) or {})
        direction = row.get('direction') or metrics.get('direction') or 'long'
        individual = row.get('individual')
        ticker, signals = _normalize_individual(individual)
        setup_id = metrics.get('setup_id') or f"SETUP_{idx + 1:04d}"

        signals_fp = metrics.get('signals_fingerprint') or canonical_signals_fingerprint(signals)
        exit_policy_tag = metrics.get('outer_id') or metrics.get('exit_policy_tag') or 'npwf'
        allow_pyramiding = bool(getattr(settings.ga, 'allow_pyramiding', False))

        desc = metrics.get('setup_desc') or du.desc_from_meta(signals, meta_map)
        if not desc:
            try:
                desc = du.format_setup_description({'individual': individual})
            except Exception:
                desc = ""

        objectives = row.get('objectives') or []
        eligible_flag = bool(metrics.get('eligible', False))
        support_val = metrics.get('support', 0.0) or 0.0
        n_trades_val = metrics.get('n_trades', 0) or 0

        ledger = row.get('trade_ledger')
        ledger_for_growth: Optional[pd.DataFrame]
        if isinstance(ledger, pd.DataFrame) and not ledger.empty:
            ledger_with_id = ledger.copy()
            ledger_with_id['setup_id'] = setup_id
            ledger_with_id['ticker'] = ticker
            ledger_with_id['direction'] = direction
            ledger_with_id['signals_fingerprint'] = signals_fp
            deduped_ledger, ledger_dup_stats = dedupe_trade_ledger(
                ledger_with_id,
                setup_id=setup_id,
                ticker=ticker,
                direction=direction,
                signals_fingerprint=signals_fp,
                exit_policy_tag=exit_policy_tag,
                allow_pyramiding=allow_pyramiding,
            )
            ledger_frames.append(deduped_ledger)
            metrics.setdefault('dup_suppressed_total', 0)
            metrics.setdefault('dup_merged_total', 0)
            metrics['dup_suppressed_total'] = int(metrics.get('dup_suppressed_total', 0)) + int(ledger_dup_stats.get('n_dups_suppressed', 0))
            metrics['dup_merged_total'] = int(metrics.get('dup_merged_total', 0)) + int(ledger_dup_stats.get('n_dups_merged', 0))
            metrics['dedup_applied'] = bool(metrics.get('dedup_applied', False) or ledger_dup_stats.get('n_dups_suppressed', 0) or ledger_dup_stats.get('n_dups_merged', 0))
            ledger_for_growth = deduped_ledger
        else:
            ledger_for_growth = None

        geom_cagr = _sanitized_growth(ledger_for_growth)
        display_cagr = geom_cagr
        display_calmar = metrics.get('bootstrap_calmar_lb')
        display_pf = metrics.get('bootstrap_profit_factor_lb')

        summary_records.append({
            'setup_id': setup_id,
            'ticker': ticker,
            'signals': '|'.join(signals),
            'direction': direction,
            'description': desc,
            'dsr': metrics.get('dsr'),
            'bootstrap_calmar_lb': display_calmar,
            'bootstrap_profit_factor_lb': display_pf,
            'expectancy': metrics.get('expectancy'),
            'support': support_val,
            'n_trades': n_trades_val,
            'fold_count': metrics.get('fold_count'),
            'fold_coverage_ratio': metrics.get('fold_coverage_ratio'),
            'outer_split_ids': '|'.join(metrics.get('outer_split_ids', [])),
            'first_trigger': metrics.get('first_trigger'),
            'last_trigger': metrics.get('last_trigger'),
            'max_drawdown': metrics.get('max_drawdown'),
            'cagr': display_cagr,
            'geom_cagr': geom_cagr,
            'eligible': eligible_flag,
            'eligibility_reasons': ';'.join(metrics.get('eligibility_reasons', [])),
            'fatal_reasons': ';'.join(metrics.get('fatal_reasons', [])),
            'penalty_scalar': metrics.get('penalty_scalar'),
            'soft_penalties': metrics.get('soft_penalties'),
            'dup_suppressed_total': metrics.get('dup_suppressed_total', 0),
            'dup_merged_total': metrics.get('dup_merged_total', 0),
            'dedup_applied': metrics.get('dedup_applied', False),
            'objectives': objectives,
        })

        priors_records.append(
            _build_prior_record(
                setup_id=setup_id,
                ticker=ticker,
                signals=signals,
                direction=direction,
                metrics=metrics,
                ledger=ledger_for_growth,
                data_end=data_end_final,
                ga_settings=settings.ga,
            )
        )

    summary_df = pd.DataFrame(summary_records)
    summary_cols = [
        'setup_id', 'ticker', 'signals', 'direction', 'description',
        'dsr', 'bootstrap_calmar_lb', 'bootstrap_profit_factor_lb',
        'expectancy', 'support', 'n_trades', 'fold_count', 'fold_coverage_ratio',
        'outer_split_ids', 'first_trigger', 'last_trigger', 'max_drawdown', 'cagr',
        'geom_cagr', 'eligible', 'eligibility_reasons', 'fatal_reasons', 'penalty_scalar',
        'soft_penalties', 'dup_suppressed_total', 'dup_merged_total', 'dedup_applied',
        'objective_sources', 'objectives'
    ]
    summary_df = summary_df.reindex(columns=summary_cols)
    for col in ('soft_penalties', 'objective_sources'):
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    summary_path = os.path.join(run_dir, 'options_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Options summary saved to: {summary_path}")

    if ledger_frames:
        ledger_df = pd.concat(ledger_frames, ignore_index=True)
        if 'uniq_key' in ledger_df.columns:
            before_rows = len(ledger_df)
            ledger_df = ledger_df.loc[~ledger_df.duplicated('uniq_key', keep='first')].copy()
            dup_rows = before_rows - len(ledger_df)
            if dup_rows > 0:
                print(f"Deduped {dup_rows} duplicate trade rows at ledger export.")
        ledger_columns = [
            'setup_id', 'fold_id', 'outer_id', 'trigger_date', 'exit_date',
            'ticker', 'direction', 'horizon_days', 'exit_policy_id', 'exit_reason',
            'strike', 'entry_underlying', 'exit_underlying', 'entry_iv', 'exit_iv',
            'entry_option_price', 'exit_option_price', 'contracts',
            'capital_allocated', 'capital_allocated_used', 'pnl_dollars', 'pnl_pct'
        ]
        existing = [c for c in ledger_columns if c in ledger_df.columns]
        ledger_df = ledger_df.reindex(columns=existing + [c for c in ledger_df.columns if c not in existing])
        ledger_path = os.path.join(run_dir, 'options_trade_ledger.csv')
        ledger_df.to_csv(ledger_path, index=False)
        print(f"Options trade ledger saved to: {ledger_path}")
    else:
        ledger_path = ""

    split_rows: List[Dict[str, Any]] = []
    split_list = splits.get('splits', []) if isinstance(splits, dict) else []
    for spec in split_list:
        split_id = spec.get('split_id') or spec.get('id')
        if not split_id:
            continue
        candidate_count = sum(split_id in (summary_df.at[i, 'outer_split_ids'] or '') for i in summary_df.index)
        row = dict(spec)
        row['candidates_with_support'] = candidate_count
        split_rows.append(row)

    if split_rows:
        split_df = pd.DataFrame(split_rows)
        split_path = os.path.join(run_dir, 'split_summary.csv')
        split_df.to_csv(split_path, index=False)
        print(f"Split summary saved to: {split_path}")
    else:
        split_path = ""

    priors_df = pd.DataFrame(priors_records)
    if not priors_df.empty:
        for col in ('flags', 'soft_penalties', 'objective_sources'):
            if col in priors_df.columns:
                priors_df[col] = priors_df[col].apply(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
    priors_csv_path = os.path.join(run_dir, 'options_priors.csv')
    priors_df.to_csv(priors_csv_path, index=False)
    print(f"Options priors saved to: {priors_csv_path}")

    priors_json_path = os.path.join(run_dir, 'options_priors.json')
    with open(priors_json_path, 'w') as f:
        json.dump(priors_records, f, indent=2, default=str)

    if regime_model:
        write_regime_artifacts(regime_model, run_dir)

    if simulation_summary is not None:
        write_simulation_summary(simulation_summary, run_dir)
    if simulation_ledger is not None:
        write_simulation_ledger(simulation_ledger, run_dir)

    if correlation_report is not None:
        print(f"Correlation report available at: {correlation_report}")

    run_summary_payload: Dict[str, Any] = {
        "outer_split_count": splits.get('outer_split_count') if isinstance(splits, dict) else None,
        "tail_aligned": splits.get('tail_aligned') if isinstance(splits, dict) else None,
        "cutoff_respected": splits.get('cutoff_respected') if isinstance(splits, dict) else None,
        "rsd_summary": splits.get('rsd_summary') if isinstance(splits, dict) else None,
        "frozen_library_stats": splits.get('frozen_library_stats') if isinstance(splits, dict) else None,
        "dts_summary": splits.get('dts_summary') if isinstance(splits, dict) else None,
        "dts_hud_line": splits.get('dts_hud_line') if isinstance(splits, dict) else None,
        "dts_near_miss": splits.get('dts_near_miss') if isinstance(splits, dict) else None,
        "files": {
            "options_summary_csv": summary_path,
            "options_trade_ledger_csv": ledger_path,
            "split_summary_csv": split_path,
            "options_priors_csv": priors_csv_path,
            "options_priors_json": priors_json_path,
        },
    }
    if isinstance(splits, dict):
        run_summary_payload["plans"] = splits.get('plans')
        run_summary_payload["signal_duplicate_suppressed"] = splits.get('signal_duplicate_suppressed')

    run_summary_path = os.path.join(run_dir, 'run_summary.json')
    with open(run_summary_path, 'w') as f:
        json.dump(run_summary_payload, f, indent=2, default=str)
    print(f"Run summary saved to: {run_summary_path}")

    return {
        "summary_path": summary_path,
        "ledger_path": ledger_path,
        "split_path": split_path,
        "priors_csv": priors_csv_path,
        "priors_json": priors_json_path,
        "priors_records": priors_records,
        "run_summary": run_summary_path,
    }
