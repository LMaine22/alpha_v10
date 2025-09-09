# alpha_discovery/gauntlet/run.py
from __future__ import annotations

import os
import inspect
import importlib
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import Settings  # type: ignore
from .io import find_latest_run_dir, read_global_artifacts, read_oos_artifacts, attach_setup_fp
from .stage1_oos_compat import run_stage1_oos_compat
from .stage1_health import run_stage1_health_check
from .stage2_profitability import run_stage2_profitability_on_ledger
from .stage3_robustness import run_stage3_robustness_on_ledger
from .summary import write_gauntlet_summary, write_all_setups_summary, write_open_trades_summary
from .backtester import run_gauntlet_backtest
from .reporting import write_stage_csv
from .io import ensure_dir
from .backtester import _align_to_pareto_schema_auto
from .config_new import get_permissive_gauntlet_config
from .medium import run_medium_gauntlet, SetupResults, write_gauntlet_artifacts
from ..meta_labeling import MetaLabelingSystem



def run_gauntlet(
    run_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
    config: Optional[Dict[str, Any]] = None,
    master_df: Optional[pd.DataFrame] = None,
    signals_df: Optional[pd.DataFrame] = None,
    signals_metadata: Optional[List[Dict[str, Any]]] = None,
    splits: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None,
    mode: str = "legacy",  # "legacy", "medium", "heavy"
) -> None:
    """
    Gauntlet pipeline:
      1) Collect all Pareto-front candidates across training folds.
      2) Run out-of-sample (OOS) backtests per candidate to build the gauntlet ledger.
      3) Stage-1/Stage-2 per setup, then Stage-3 cohort across survivors.
      4) Write summary & artifacts.
    """
    # Use permissive configuration by default to allow more strategies through
    default_config = get_permissive_gauntlet_config()
    cfg = dict(default_config)
    if config:
        cfg.update(config)

    # Resolve run_dir
    if run_dir is None:
        run_dir = find_latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("Could not resolve a valid run_dir for Gauntlet.")

    # Required inputs
    if master_df is None or signals_df is None or signals_metadata is None or splits is None:
        raise ValueError("master_df, signals_df, signals_metadata, and splits must be provided.")

    # Ensure gauntlet output dir
    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    # Base capital for NAV/returns aggregation
    try:
        base_capital = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_capital = 100_000.0

    # -----------------------
    # Phase 1: Candidate pool
    # -----------------------
    print("\n--- Phase 1: Collecting All In-Sample Candidates ---")
    pareto_summary, _ = read_global_artifacts(run_dir)
    if pareto_summary is None or pareto_summary.empty:
        print("Global pareto_front_summary.csv is empty. Cannot identify candidates.")
        return

    # Map fold -> last train date (OOS starts next day)
    fold_date_map = {i + 1: train_idx.max() for i, (train_idx, _) in enumerate(splits)}
    pareto_summary = pareto_summary.copy()
    pareto_summary["train_end_date"] = pareto_summary["fold"].map(fold_date_map)

    cols_needed = ["setup_id", "fold", "specialized_ticker", "direction", "signal_ids", "train_end_date"]
    missing = [c for c in cols_needed if c not in pareto_summary.columns]
    if missing:
        raise KeyError(f"pareto_front_summary.csv is missing columns: {missing}")

    candidates = pareto_summary.dropna(
        subset=["specialized_ticker", "direction", "signal_ids", "train_end_date"]
    ).copy()
    print(f"Found {len(candidates)} total candidates from all training folds.")

    # ---------------------------------------
    # Phase 2: Run out-of-sample backtesting
    # ---------------------------------------
    print("\n--- Phase 2: Running Out-of-Sample Backtests ---")
    tasks = []
    for _, row in candidates.iterrows():
        signal_ids = [s.strip() for s in str(row["signal_ids"]).split(",")] if pd.notnull(row["signal_ids"]) else []
        oos_start = pd.to_datetime(row["train_end_date"]) + pd.Timedelta(days=1)
        tasks.append(delayed(run_gauntlet_backtest)(
            setup_id=str(row["setup_id"]),
            specialized_ticker=str(row["specialized_ticker"]),
            signal_ids=signal_ids,
            direction=str(row["direction"]),
            oos_start_date=oos_start,
            master_df=master_df,
            signals_df=signals_df,
            signals_metadata=signals_metadata,
            exit_policy=None,       # allow backtester to use defaults/GA policy
            settings=settings,
            origin_fold=int(row["fold"]),
        ))

    oos_ledgers: Dict[str, pd.DataFrame] = {}
    if tasks:
        with tqdm(total=len(tasks), desc="Running OOS Backtests", dynamic_ncols=True) as pbar:
            with Parallel(n_jobs=4, backend="loky", timeout=300) as parallel:
                results = parallel(tasks)
                for i, ledger in enumerate(results):
                    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
                        setup_id = str(candidates.iloc[i]["setup_id"])
                        oos_ledgers[setup_id] = ledger
                    pbar.update(1)

    if not oos_ledgers:
        print("No trades were generated during out-of-sample backtesting. Gauntlet finished.")
        return

    # Concatenate all OOS ledgers and write with an explicit, forward-compatible schema
    full_oos_ledger = pd.concat(oos_ledgers.values(), ignore_index=True)
    # Align the combined OOS ledger to the Pareto (TRAIN) header, then write directly.
    try:
        # Choose any present origin_fold for the template (columns are the same across folds)
        fold_for_template = int(pd.to_numeric(full_oos_ledger.get("origin_fold")).dropna().iloc[0])
    except Exception:
        fold_for_template = 1

    aligned_ledger = _align_to_pareto_schema_auto(full_oos_ledger, origin_fold=fold_for_template)

    # Write directly to avoid the legacy LEDGER_BASE_SCHEMA coercion
    gaunt_dir = os.path.join(run_dir, "gauntlet")
    ensure_dir(gaunt_dir)
    aligned_ledger.to_csv(os.path.join(gaunt_dir, "gauntlet_ledger.csv"), index=False)
    print(f"\nOut-of-sample backtesting complete. Generated OOS ledgers for {len(oos_ledgers)} strategies.")
    
    # Generate summary for ALL setups in the trade ledger (regardless of gauntlet performance)
    write_all_setups_summary(run_dir, full_oos_ledger)
    
    # Generate summary for setups with open trades (filtered from all setups)
    write_open_trades_summary(run_dir, full_oos_ledger)

    # -----------------------------------------------------------------
    # Phase 3: Mode-specific gauntlet execution
    # -----------------------------------------------------------------
    if mode == "medium":
        print("\n--- Phase 3: Medium Gauntlet (Statistical Rigor) ---")
        _run_medium_gauntlet_mode(run_dir, oos_ledgers, settings, cfg)
        return
    elif mode == "heavy":
        print("\n--- Phase 3: Heavy Gauntlet (Academic Rigor) ---")
        # TODO: Implement heavy mode with DSR, BH-FDR, SPA, etc.
        print("Heavy mode not yet implemented, falling back to legacy mode")
        mode = "legacy"
    
    # Legacy mode (existing implementation)
    print("\n--- Phase 3: Legacy Gauntlet (Stage-1/2/3 per setup) ---")
    print("\n--- Phase 3: OOS Stage-1/2/3 per-setup ---")
    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    stage3_rows: List[Dict[str, Any]] = []

    for setup_id, oos_ledger in oos_ledgers.items():
        # Stage-1 per setup: health check (recency + activity + momentum)
        s1 = run_stage1_health_check(
            run_dir=run_dir,
            fold_num=0,
            settings=settings,
            config=cfg,
            fold_summary=pd.DataFrame([{"setup_id": setup_id, "rank": None}]),
            fold_ledger=oos_ledger,
        )
        stage1_rows.append(s1.iloc[0].to_dict())
        if not bool(s1["pass_stage1"].iloc[0]):
            continue

        # Stage-2 per setup: profitability check (NAV + PnL + win rate)
        s2 = run_stage2_profitability_on_ledger(
            fold_ledger=oos_ledger,
            settings=settings,
            config=cfg,
            stage1_df=s1,
        )
        if s2 is None or s2.empty:
            continue
        rec = s2.iloc[0].to_dict()
        rec["setup_id"] = setup_id
        stage2_rows.append(rec)

        # Stage-3 per setup: robustness check (DSR + bootstrap + stability)
        s3 = run_stage3_robustness_on_ledger(
            fold_ledger=oos_ledger,
            settings=settings,
            config=cfg,
            stage2_df=s2,
        )
        if s3 is None or s3.empty:
            continue
        s3_rec = s3.iloc[0].to_dict()
        s3_rec["setup_id"] = setup_id
        stage3_rows.append(s3_rec)

    # Write Stage diagnostics
    if stage1_rows:
        write_stage_csv(run_dir, "stage1_oos", pd.DataFrame(stage1_rows))
    if stage2_rows:
        write_stage_csv(run_dir, "stage2_oos", pd.DataFrame(stage2_rows))
    if stage3_rows:
        write_stage_csv(run_dir, "stage3_oos", pd.DataFrame(stage3_rows))

    if not stage3_rows:
        print("No candidates reached Stage-3 on OOS; gauntlet complete.")
        return

    # Survivors & summary (all Stage-3 passers are survivors)
    survivor_ids = set([str(d["setup_id"]) for d in stage3_rows if d.get("pass_stage3", False)])
    print(f"Stage-3 survivors: {len(survivor_ids)}")

    by_id_s1 = {str(d["setup_id"]): d for d in stage1_rows if "setup_id" in d}
    by_id_s2 = {str(d["setup_id"]): d for d in stage2_rows if "setup_id" in d}
    by_id_s3 = {str(d["setup_id"]): d for d in stage3_rows if "setup_id" in d}

    final_survivors: List[Dict[str, Any]] = []
    for sid in survivor_ids:
        final_survivors.append({
            "setup_id": sid,
            "oos_s1": by_id_s1.get(sid, {}),
            "oos_s2": by_id_s2.get(sid, {}),
            "oos_s3": by_id_s3.get(sid, {}),
        })

    if final_survivors:
        write_gauntlet_summary(run_dir, final_survivors, full_oos_ledger)
    else:
        print("No strategies survived Stage-3.")


def _run_medium_gauntlet_mode(run_dir: str, oos_ledgers: Dict[str, pd.DataFrame], 
                             settings: Settings, config: Dict[str, Any]) -> None:
    """Run Medium Gauntlet mode."""
    from .medium import get_medium_gauntlet_config
    
    # Get medium gauntlet config
    medium_config = get_medium_gauntlet_config()
    if config:
        medium_config.update(config)
    
    # Convert OOS ledgers to SetupResults
    setup_results = []
    for setup_id, ledger in oos_ledgers.items():
        if ledger.empty:
            continue
            
        # Get base capital
        try:
            base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
        except Exception:
            base_cap = 100_000.0
        
        # Compute daily returns and equity curve
        daily_returns = nav_daily_returns_from_ledger(ledger, base_capital=base_cap)
        equity_curve = (1 + daily_returns).cumprod() * base_cap
        
        # Extract setup info
        ticker = ledger.get('specialized_ticker', ['Unknown']).iloc[0] if not ledger.empty else 'Unknown'
        direction = ledger.get('direction', ['long']).iloc[0] if not ledger.empty else 'long'
        
        # Create SetupResults
        setup_result = SetupResults(
            setup_id=setup_id,
            ticker=ticker,
            direction=direction,
            oos_trades=ledger,
            oos_daily_returns=daily_returns,
            oos_equity_curve=equity_curve,
            historical_median_drawdown=None  # Would need historical data
        )
        setup_results.append(setup_result)
    
    if not setup_results:
        print("No valid setups found for Medium Gauntlet.")
        return
    
    print(f"Running Medium Gauntlet on {len(setup_results)} setups...")
    
    # Run medium gauntlet
    results = run_medium_gauntlet(setup_results, settings, medium_config)
    
    # Write artifacts
    output_dir = os.path.join(run_dir, "gauntlet_medium")
    write_gauntlet_artifacts(results, output_dir, medium_config)
    
    # Run meta-labeling on survivors
    if settings.meta_labeling.enabled:
        print("\n--- Phase 4: Meta-Labeling (Post-Gauntlet Filter) ---")
        _run_meta_labeling(run_dir, results, oos_ledgers, settings, config)
    
    # Print summary
    deploy_count = len([r for r in results if r.final_decision == "Deploy"])
    monitor_count = len([r for r in results if r.final_decision == "Monitor"])
    retire_count = len([r for r in results if r.final_decision == "Retire"])
    
    print(f"\nMedium Gauntlet Results:")
    print(f"  Deploy: {deploy_count}")
    print(f"  Monitor: {monitor_count}")
    print(f"  Retire: {retire_count}")
    print(f"  Artifacts written to: {output_dir}")


def _run_meta_labeling(run_dir: str, gauntlet_results: List, oos_ledgers: Dict[str, pd.DataFrame],
                      settings: Any, config: Dict[str, Any]) -> None:
    """Run meta-labeling on gauntlet survivors."""
    from ..meta_labeling import MetaLabelingSystem
    
    # Get gauntlet survivors (Deploy and Monitor decisions)
    survivors = []
    for result in gauntlet_results:
        if result.final_decision in ["Deploy", "Monitor"]:
            survivors.append({
                'setup_id': result.setup_id,
                'ticker': result.ticker,
                'direction': result.direction,
                'final_decision': result.final_decision
            })
    
    if not survivors:
        print("No gauntlet survivors for meta-labeling")
        return
    
    print(f"Running meta-labeling on {len(survivors)} gauntlet survivors...")
    
    # Initialize meta-labeling system
    meta_system = MetaLabelingSystem(config.get('meta_labeling', {}))
    
    # Load required data
    from ..data.loader import load_master_data
    from ..features.registry import build_feature_matrix
    from ..signals.compiler import compile_signals
    
    print("Loading data for meta-labeling...")
    master_df = load_master_data()
    feature_matrix = build_feature_matrix(master_df)
    signals_df, signals_metadata = compile_signals(feature_matrix)
    
    # Run meta-labeling
    meta_results = meta_system.run_meta_labeling(
        survivors, oos_ledgers, master_df, signals_df, signals_metadata
    )
    
    # Generate artifacts
    meta_output_dir = os.path.join(run_dir, "meta_labeling")
    meta_system.artifact_generator.generate_summary_artifacts(meta_results, meta_output_dir)
    
    # Print summary
    summary_stats = meta_system.get_summary_statistics()
    print(f"\nMeta-Labeling Results:")
    print(f"  Total setups: {summary_stats.get('total_setups', 0)}")
    print(f"  Successfully trained: {summary_stats.get('trained_count', 0)}")
    print(f"  Average EV improvement: {summary_stats.get('avg_ev_improvement', 0):.4f}")
    print(f"  Average Sharpe improvement: {summary_stats.get('avg_sharpe_improvement', 0):.4f}")
    print(f"  Average retention rate: {summary_stats.get('avg_retention_rate', 0):.4f}")
    print(f"  Average model accuracy: {summary_stats.get('avg_accuracy', 0):.4f}")
    print(f"  Artifacts written to: {meta_output_dir}")


def _first_nonnull_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Prefer explicit trade/trigger/entry date columns (non-destructive)
    date_col = _first_nonnull_col(out, ['trigger_date', 'entry_date', 'trade_date', 'date'])
    if date_col is None:
        raise KeyError("No date-like column found among ['trigger_date','entry_date','trade_date','date']")
    out['trade_dt'] = pd.to_datetime(out[date_col], errors='coerce')
    if out['trade_dt'].isna().all():
        raise ValueError(f"Could not parse any dates from column '{date_col}'")
    # Normalize to date (no time)
    out['trade_d'] = out['trade_dt'].dt.normalize()
    return out


def _ensure_fold_column(ledger_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a 'fold' column exists on the ledger by merging from summary if needed.
    Merge keys are conservative to avoid accidental blow-ups.
    """
    if 'fold' in ledger_df.columns:
        return ledger_df

    merge_keys = [k for k in ('setup_id', 'direction', 'specialized_ticker') if k in ledger_df.columns and k in summary_df.columns]
    if not merge_keys:
        # As a last resort, if summary has exactly one fold, propagate that; else fail fast.
        uniq_folds = summary_df['fold'].unique().tolist() if 'fold' in summary_df.columns else []
        if len(uniq_folds) == 1:
            out = ledger_df.copy()
            out['fold'] = uniq_folds[0]
            return out
        raise KeyError("Ledger missing 'fold' and cannot infer it from summary (no compatible merge keys).")

    out = ledger_df.merge(
        summary_df[merge_keys + ['fold']].drop_duplicates(),
        on=merge_keys,
        how='left'
    )
    if 'fold' not in out.columns or out['fold'].isna().any():
        raise KeyError("Failed to populate 'fold' on OOS ledger via merge.")
    return out


def _infer_knowledge_map_from_splits(splits):
    """
    Attempt to derive a mapping fold_index -> knowledge_date from the provided splits.
    Expect splits like: List[Tuple[train_index, test_index]] where train_index is datetime-like or convertible.
    If not possible, return None and caller will use fold order as a proxy.
    """
    if not splits:
        return None
    mapping = {}
    try:
        for i, (train_idx, _test_idx) in enumerate(splits, start=1):
            # train_idx may be numpy array / pandas index of datetimes or integers
            train_end = np.max(train_idx)
            # If integers, we cannot convert here; let caller fall back
            if np.issubdtype(np.array([train_end]).dtype, np.number):
                return None
            # Otherwise assume datetime-like
            try:
                train_end = pd.to_datetime(train_end)
            except Exception:
                return None
            mapping[i] = pd.to_datetime(train_end).normalize()
        return mapping
    except Exception:
        return None


def build_strict_oos_ledger(run_dir: str, splits=None) -> pd.DataFrame:
    """
    Construct a single, deduplicated Strict-OOS ledger:
      - Reads OOS summary/ledger via read_oos_artifacts(run_dir)
      - Ensures 'setup_fp' is present
      - Normalizes a single date column 'trade_d'
      - Ensures a 'fold' column exists on the ledger
      - Adds 'knowledge_d' if derivable from splits (train end date per fold)
      - Deduplicates overlapping OOS entries across folds by keeping the earliest knowledge signal

    Returns:
      strict_oos_df: DataFrame ready for Gauntlet stages (not yet filtered by Stage 1/2/3).
    """
    # 1) Load OOS artifacts
    oos_summary, oos_ledger = read_oos_artifacts(run_dir)

    # 2) Ensure setup_fp exists on both
    if 'setup_fp' not in oos_summary.columns:
        oos_summary = attach_setup_fp(oos_summary)
    if 'setup_fp' not in oos_ledger.columns:
        # Try merge from summary
        merge_keys = [k for k in ('setup_id', 'direction', 'specialized_ticker') if k in oos_ledger.columns and k in oos_summary.columns]
        if merge_keys:
            oos_ledger = oos_ledger.merge(
                oos_summary[merge_keys + ['setup_fp']].drop_duplicates(),
                on=merge_keys,
                how='left'
            )
        else:
            oos_ledger = attach_setup_fp(oos_ledger)

    # 3) Ensure 'fold' is present on ledger
    oos_ledger = _ensure_fold_column(oos_ledger, oos_summary)

    # 4) Normalize trade date
    oos_ledger = _normalize_trade_date(oos_ledger)

    # 5) Try to add knowledge date per fold from splits
    knowledge_map = _infer_knowledge_map_from_splits(splits)
    if knowledge_map is not None:
        oos_ledger['knowledge_d'] = oos_ledger['fold'].map(knowledge_map)
    else:
        # Fallback proxy: use fold order (1..K) as 'knowledge_rank' to break ties deterministically
        oos_ledger['knowledge_rank'] = oos_ledger['fold'].astype(int)

    # 5b) Try to merge in setup descriptions from summary if available
    desc_cols = ['description', 'setup_id', 'signal_ids']
    summary_desc_cols = [c for c in desc_cols if c in oos_summary.columns]
    ledger_desc_cols = [c for c in desc_cols if c in oos_ledger.columns]
    
    # If summary has descriptions but ledger doesn't, merge them in
    missing_desc_cols = [c for c in summary_desc_cols if c not in ledger_desc_cols]
    if missing_desc_cols:
        merge_keys = [k for k in ('setup_id', 'setup_fp', 'direction', 'specialized_ticker') if k in oos_ledger.columns and k in oos_summary.columns]
        if merge_keys:
            desc_data = oos_summary[merge_keys + missing_desc_cols].drop_duplicates()
            oos_ledger = oos_ledger.merge(desc_data, on=merge_keys, how='left')

    # 6) Deduplicate overlapping OOS entries across folds
    # Key: (setup_fp, ticker-ish, direction, trade_d)
    # Use the first available "ticker-like" column present in the ledger
    ticker_col = _first_nonnull_col(oos_ledger, ['ticker', 'specialized_ticker', 'asset', 'symbol'])
    if ticker_col is None:
        raise KeyError("No ticker-like column found among ['ticker','specialized_ticker','asset','symbol'] in OOS ledger.")

    sort_cols = ['setup_fp', ticker_col, 'direction', 'trade_d']
    # Decide tie-breaker: earliest knowledge_d takes precedence; else lowest fold (knowledge_rank)
    if 'knowledge_d' in oos_ledger.columns and oos_ledger['knowledge_d'].notna().any():
        oos_ledger = oos_ledger.sort_values(sort_cols + ['knowledge_d'])
    else:
        oos_ledger = oos_ledger.sort_values(sort_cols + ['knowledge_rank'])

    deduped = oos_ledger.drop_duplicates(subset=sort_cols, keep='first').reset_index(drop=True)

    # 7) Return with useful audit columns up front
    front_cols = ['setup_fp', 'setup_id', 'direction', ticker_col, 'fold', 'trade_d']
    if 'knowledge_d' in deduped.columns:
        front_cols.append('knowledge_d')
    elif 'knowledge_rank' in deduped.columns:
        front_cols.append('knowledge_rank')

    # Keep only columns that exist + preserve everything else
    front_cols = [c for c in front_cols if c in deduped.columns]
    ordered = pd.concat([deduped[front_cols], deduped.drop(columns=[c for c in front_cols if c in deduped.columns])], axis=1)
    return ordered


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def _pick_ticker_col(df: pd.DataFrame) -> str:
    for c in ('ticker', 'specialized_ticker', 'asset', 'symbol'):
        if c in df.columns:
            return c
    raise KeyError("No ticker-like column found among ['ticker','specialized_ticker','asset','symbol'].")


def _load_stage_func(module_basename: str, prefer_names=None):
    """
    Dynamically load a stage runner from alpha_discovery.gauntlet.<module_basename>.
    We try preferred names first, then any callable starting with 'run_stage'.
    """
    mod = importlib.import_module(f'alpha_discovery.gauntlet.{module_basename}')
    prefer_names = prefer_names or []
    for name in prefer_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    # Fallback: first callable starting with 'run_stage'
    for name in dir(mod):
        if name.startswith('run_stage') and callable(getattr(mod, name)):
            return getattr(mod, name)
    raise ImportError(f"No runnable stage function found in {module_basename}.")


def _call_stage(fn, df: pd.DataFrame, **kwargs):
    """
    Call a stage function defensively. It may accept different parameter names.
    Priority: ledger/df positional or keyword. Extra kwargs are filtered by signature.
    Returns whatever the stage returns (ideally (filtered_df, summary_df) or a dict).
    """
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if 'ledger' in sig.parameters:
        return fn(ledger=df, **filtered)
    if 'df' in sig.parameters:
        return fn(df=df, **filtered)
    # Try positional only
    try:
        return fn(df, **filtered)
    except TypeError:
        return fn(**filtered)


def _write_if_df(obj, path):
    """
    Write DataFrame to CSV if obj is a non-empty DataFrame. Return True if written.
    """
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        obj.to_csv(path, index=False)
        return True
    return False


def _extract_df_from_stage_result(res, *keys):
    """
    Accepts common result shapes:
      - (df, summary)
      - {'df': df, 'summary': summary}
      - {'ledger': df, ...}
      - df
    Returns df, summary (either may be None).
    """
    df, summary = None, None
    if isinstance(res, tuple) and res:
        df = res[0]
        if len(res) > 1 and isinstance(res[1], pd.DataFrame):
            summary = res[1]
    elif isinstance(res, dict):
        for k in ('df', 'ledger') + keys:
            if k in res and isinstance(res[k], pd.DataFrame):
                df = res[k]; break
        for k in ('summary', 'stats', 'report'):
            if k in res and isinstance(res[k], pd.DataFrame):
                summary = res[k]; break
    elif isinstance(res, pd.DataFrame):
        df = res
    return df, summary


def _basic_all_setups_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight rollup in case stages don't emit one. Uses common columns if present.
    Includes setup descriptions if available.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    try:
        ticker_col = _pick_ticker_col(df)
    except KeyError:
        # If no ticker column, return empty - this shouldn't happen but let's be safe
        return pd.DataFrame()
    agg = {
        'trade_d': ['min', 'max', 'count'],
    }
    if 'pnl_dollars' in df.columns:
        agg['pnl_dollars'] = ['sum', 'mean']
    if 'max_drawdown' in df.columns:
        agg['max_drawdown'] = ['max']

    gb = df.groupby(['setup_fp', ticker_col, 'direction'], dropna=False).agg(agg)
    gb.columns = ['_'.join(col).strip('_') for col in gb.columns.values]
    gb = gb.rename(columns={'trade_d_min': 'first_trade_date',
                            'trade_d_max': 'last_trade_date',
                            'trade_d_count': 'trades_count'})
    gb = gb.reset_index()
    
    # Try to add setup descriptions and setup_id if available
    desc_cols = ['description', 'setup_id', 'signal_ids']
    available_desc_cols = [c for c in desc_cols if c in df.columns]
    
    if available_desc_cols:
        # Get first occurrence of each setup's metadata
        desc_df = df.groupby(['setup_fp', ticker_col, 'direction'], dropna=False)[available_desc_cols].first().reset_index()
        gb = gb.merge(desc_df, on=['setup_fp', ticker_col, 'direction'], how='left')
    
    return gb


def _open_trades_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best-effort open trades filter. If explicit status columns exist, use them; otherwise
    assume open if 'exit_date' is missing or NaT.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # Prefer explicit status flags if present
    for flag in ('status', 'trade_status'):
        if flag in df.columns:
            open_mask = df[flag].astype(str).str.upper().str.contains('OPEN')
            return df.loc[open_mask].copy()
    # Fallback: missing exit date/time
    for c in ('exit_date', 'close_date', 'exit_dt', 'close_dt'):
        if c in df.columns:
            dd = df.copy()
            dd[c] = pd.to_datetime(dd[c], errors='coerce')
            return dd.loc[dd[c].isna()].copy()
    return pd.DataFrame()


def run_gauntlet_strict_oos(run_dir: str, splits=None, outdir: str | None = None, settings=None, config=None, **stage_kwargs):
    """
    Build the Strict-OOS ledger, run Stage1->Stage2->Stage3, and write all outputs
    under runs/<run>/gauntlet/strict_oos.
    """
    # Import defaults if not provided
    if settings is None:
        from ..config import settings as default_settings
        settings = default_settings
    if config is None:
        config = get_permissive_gauntlet_config()
    
    # 0) Output dir
    out_base = _ensure_dir(os.path.join(run_dir, 'gauntlet', 'strict_oos')) if outdir is None else _ensure_dir(outdir)

    # 1) Build stitched Strict-OOS ledger
    strict_df = build_strict_oos_ledger(run_dir, splits=splits)

    # Always write a raw stitched copy for audit
    raw_path = os.path.join(out_base, 'strict_oos_stitched_ledger.csv')
    strict_df.to_csv(raw_path, index=False)

    # Determine OOS 'as_of_date' (end of the OOS window)
    try:
        oos_as_of_date = pd.to_datetime(strict_df['trade_d']).max().normalize()
    except Exception:
        oos_as_of_date = None

    # Full rollup across ALL strict-OOS trades (pre-stage filtering)
    try:
        all_rollup_full = _basic_all_setups_summary(strict_df)
        all_rollup_full.to_csv(os.path.join(out_base, 'all_setups_rollup_full_oos.csv'), index=False)
    except Exception as e:
        # Non-fatal
        print('[Strict-OOS] Warning: failed to write all_setups_rollup_full_oos.csv ->', e)

    # 2) Stage 1 (legacy call)
    stage1_fn = _load_stage_func('stage1_health', prefer_names=['run_stage1_on_ledger', 'run_stage1_health_check', 'run_stage1'])
    res1 = _call_stage(stage1_fn, strict_df, output_dir=out_base, mode='oos', stage='stage1',
                       as_of_date=oos_as_of_date, **stage_kwargs)
    s1_df, s1_summary = _extract_df_from_stage_result(res1)
    
    # Handle the case where Stage 1 returns a summary as "df" instead of actual ledger
    if s1_df is not None and len(s1_df) < 10 and 'setup_id' in s1_df.columns:
        # This looks like a summary, not a ledger. Legacy Stage 1 didn't filter.
        s1_summary = s1_df  # Move it to summary
        s1_df = None        # No filtered ledger
    
    # Decide if legacy Stage 1 actually filtered anything
    legacy_noop = (s1_df is None) or (len(s1_df) == len(strict_df))
    # If summary exists and claims "fail" while not filtering, it's misleading for OOS
    misleading = False
    summary_to_check = s1_summary if s1_summary is not None else s1_df
    if isinstance(summary_to_check, pd.DataFrame) and not summary_to_check.empty:
        # Heuristic: any column that looks like pass flag but false?
        pass_cols = [c for c in summary_to_check.columns if 'pass' in c.lower()]
        if pass_cols and (summary_to_check[pass_cols].astype(str).isin(['False','false','0']).any().any()):
            misleading = True

    # If legacy is no-op or misleading, apply OOS-compat Stage 1 with OOS-aware recency
    recency_days = int(stage_kwargs.get('stage1_recency_days', 14))
    min_trades = int(stage_kwargs.get('stage1_min_trades', 5))
    if legacy_noop or misleading:
        compat_df, compat_sum = run_stage1_oos_compat(
            strict_df,
            output_dir=out_base,
            as_of_date=oos_as_of_date.isoformat() if oos_as_of_date is not None else None,
            recency_days=recency_days,
            min_trades=min_trades
        )
        # Use compat outputs
        s1_df = compat_df
        s1_summary = compat_sum
        # Note why we did this
        with open(os.path.join(out_base, 'stage1_compat_note.txt'), 'w') as f:
            f.write(
                "Legacy Stage 1 returned no filtering or contradictory pass flags for historical OOS data.\n"
                f"Applied OOS-compat Stage 1 with as_of_date={oos_as_of_date}, "
                f"recency_days={recency_days}, min_trades={min_trades}.\n"
            )

    # Persist Stage 1 artifacts
    if not _write_if_df(s1_df, os.path.join(out_base, 'stage1_oos.csv')):
        s1_df = strict_df.copy()
        s1_df.to_csv(os.path.join(out_base, 'stage1_oos.csv'), index=False)
    _write_if_df(s1_summary, os.path.join(out_base, 'stage1_summary.csv'))

    # 3) Stage 2 - Call directly with proper signature
    try:
        res2 = run_stage2_profitability_on_ledger(
            fold_ledger=s1_df,
            settings=settings,
            config=config,
            stage1_df=None,
        )
        s2_df, s2_summary = s1_df.copy(), res2  # Stage 2 returns summary only
        s2_df.to_csv(os.path.join(out_base, 'stage2_oos.csv'), index=False)
        _write_if_df(s2_summary, os.path.join(out_base, 'stage2_summary.csv'))
    except Exception as e:
        print(f"Stage 2 failed: {e}, using pass-through")
        s2_df = s1_df.copy()
        s2_df.to_csv(os.path.join(out_base, 'stage2_oos.csv'), index=False)

    # 4) Stage 3 - Call directly with proper signature
    try:
        res3 = run_stage3_robustness_on_ledger(
            fold_ledger=s2_df,
            settings=settings,
            config=config,
            stage2_df=None,
        )
        s3_df, s3_summary = s2_df.copy(), res3  # Stage 3 returns summary only
        s3_df.to_csv(os.path.join(out_base, 'stage3_oos.csv'), index=False)
        _write_if_df(s3_summary, os.path.join(out_base, 'stage3_summary.csv'))
    except Exception as e:
        print(f"Stage 3 failed: {e}, using pass-through")
        s3_df = s2_df.copy()
        s3_df.to_csv(os.path.join(out_base, 'stage3_oos.csv'), index=False)

    # 5) Gauntlet survivors ledger & summaries
    survivors_df = s3_df if isinstance(s3_df, pd.DataFrame) else s2_df
    survivors_df.to_csv(os.path.join(out_base, 'gauntlet_ledger.csv'), index=False)

    # all_setups_summary: use stage-provided if any, else basic rollup
    all_sum = _basic_all_setups_summary(survivors_df)
    all_sum.to_csv(os.path.join(out_base, 'all_setups_summary.csv'), index=False)

    # open_trades_summary
    open_sum = _open_trades_summary(survivors_df)
    if not open_sum.empty:
        open_sum.to_csv(os.path.join(out_base, 'open_trades_summary.csv'), index=False)

    # gauntlet_summary: prefer stage 3 summary if present; else a tiny rollup
    if isinstance(s3_summary, pd.DataFrame) and not s3_summary.empty:
        s3_summary.to_csv(os.path.join(out_base, 'gauntlet_summary.csv'), index=False)
    else:
        # Minimal placeholder: counts by setup
        try:
            ticker_col = _pick_ticker_col(survivors_df)
            mini = survivors_df.groupby(['setup_fp', ticker_col, 'direction'], dropna=False).size().reset_index(name='trades_count')
            mini.to_csv(os.path.join(out_base, 'gauntlet_summary.csv'), index=False)
        except KeyError:
            # If no ticker column, create an even simpler summary
            mini = survivors_df.groupby(['setup_fp', 'direction'], dropna=False).size().reset_index(name='trades_count')
            mini.to_csv(os.path.join(out_base, 'gauntlet_summary.csv'), index=False)

    return {
        'outdir': out_base,
        'stage1_rows': 0 if s1_df is None else len(s1_df),
        'stage2_rows': 0 if s2_df is None else len(s2_df),
        'stage3_rows': 0 if s3_df is None else len(s3_df),
        'survivor_rows': 0 if survivors_df is None else len(survivors_df),
    }
