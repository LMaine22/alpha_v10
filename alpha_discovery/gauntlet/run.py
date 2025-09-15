# alpha_discovery/gauntlet/run.py
from __future__ import annotations

import os
import inspect
import importlib
import shutil
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import Settings  # type: ignore
from ..eval.nav import nav_daily_returns_from_ledger  # use your real nav util
from .io import find_latest_run_dir, read_global_artifacts, read_oos_artifacts, attach_setup_fp, ensure_dir
from .stage1_oos_compat import run_stage1_oos_compat
from .stage1_health import run_stage1_health_check
from .stage2_profitability import run_stage2_profitability_on_ledger
from .stage3_robustness import run_stage3_robustness_on_ledger
from .summary import write_gauntlet_summary, write_all_setups_summary, write_open_trades_summary
from .backtester import run_gauntlet_backtest, _align_to_pareto_schema_auto, _mark_open_positions_at_eod
from .config_new import get_permissive_gauntlet_config
from .medium import run_medium_gauntlet, SetupResults, write_gauntlet_artifacts
from .reporting import write_stage_csv


# -------------------------
# LEGACY GAUNTLET PIPELINE
# -------------------------
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
    Legacy pipeline:
      1) Collect Pareto candidates across training folds.
      2) OOS backtest candidates to end-of-data (fresh ledger).
      3) Stage-1/2/3 per setup.
      4) Write canonical summaries + open trades.
    """
    default_config = get_permissive_gauntlet_config()
    cfg = dict(default_config)
    if config:
        cfg.update(config)

    if run_dir is None:
        run_dir = find_latest_run_dir()
    if not run_dir or not os.path.isdir(run_dir):
        raise FileNotFoundError("Could not resolve a valid run_dir for Gauntlet.")

    if master_df is None or signals_df is None or signals_metadata is None or splits is None:
        raise ValueError("master_df, signals_df, signals_metadata, and splits must be provided.")

    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    try:
        base_capital = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
    except Exception:
        base_capital = 100_000.0

    # Phase 1: Candidates
    pareto_summary, _ = read_global_artifacts(run_dir)
    if pareto_summary is None or pareto_summary.empty:
        print("Global pareto_front_summary.csv is empty. Cannot identify candidates.")
        return

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

    # Phase 2: Backtests
    print("\n--- Running OOS Backtests ---")
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
            exit_policy=None,
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

    full_oos_ledger = pd.concat(oos_ledgers.values(), ignore_index=True)
    try:
        fold_for_template = int(pd.to_numeric(full_oos_ledger.get("origin_fold")).dropna().iloc[0])
    except Exception:
        fold_for_template = 1

    aligned_ledger = _align_to_pareto_schema_auto(full_oos_ledger, origin_fold=fold_for_template)
    ensure_dir(gaunt_dir)
    aligned_ledger.to_csv(os.path.join(gaunt_dir, "gauntlet_ledger.csv"), index=False)

    # Canonical outputs (legacy writers)
    write_all_setups_summary(run_dir, full_oos_ledger)
    write_open_trades_summary(run_dir, full_oos_ledger)

    # Optional: legacy Stages (unchanged)
    print("\n--- Stage-1/2/3 per setup (legacy mode) ---")
    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    stage3_rows: List[Dict[str, Any]] = []

    for setup_id, oos_ledger in oos_ledgers.items():
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

    if stage1_rows:
        write_stage_csv(run_dir, "stage1_oos", pd.DataFrame(stage1_rows))
    if stage2_rows:
        write_stage_csv(run_dir, "stage2_oos", pd.DataFrame(stage2_rows))
    if stage3_rows:
        write_stage_csv(run_dir, "stage3_oos", pd.DataFrame(stage3_rows))

    if stage3_rows:
        survivor_ids = set([str(d["setup_id"]) for d in stage3_rows if d.get("pass_stage3", False)])
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


# ----------------------
# STRICT OOS CONSTRUCTOR
# ----------------------
def _first_nonnull_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = _first_nonnull_col(out, ['trigger_date', 'entry_date', 'trade_date', 'date'])
    if date_col is None:
        raise KeyError("No date-like column found among ['trigger_date','entry_date','trade_date','date']")
    out['trade_dt'] = pd.to_datetime(out[date_col], errors='coerce')
    if out['trade_dt'].isna().all():
        raise ValueError(f"Could not parse any dates from column '{date_col}'")
    out['trade_d'] = out['trade_dt'].dt.normalize()
    return out


def _ensure_fold_column(ledger_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if 'fold' in ledger_df.columns:
        return ledger_df
    merge_keys = [k for k in ('setup_id', 'direction', 'specialized_ticker') if
                  k in ledger_df.columns and k in summary_df.columns]
    if not merge_keys:
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
    if not splits:
        return None
    mapping = {}
    try:
        for i, (train_idx, _test_idx) in enumerate(splits, start=1):
            train_end = np.max(train_idx)
            if np.issubdtype(np.array([train_end]).dtype, np.number):
                return None
            try:
                train_end = pd.to_datetime(train_end)
            except Exception:
                return None
            mapping[i] = pd.to_datetime(train_end).normalize()
        return mapping
    except Exception:
        return None


def build_strict_oos_ledger(run_dir: str, splits=None) -> pd.DataFrame:
    oos_summary, oos_ledger = read_oos_artifacts(run_dir)

    if 'setup_fp' not in oos_summary.columns:
        oos_summary = attach_setup_fp(oos_summary)
    if 'setup_fp' not in oos_ledger.columns:
        merge_keys = [k for k in ('setup_id', 'direction', 'specialized_ticker') if
                      k in oos_ledger.columns and k in oos_summary.columns]
        if merge_keys:
            oos_ledger = oos_ledger.merge(
                oos_summary[merge_keys + ['setup_fp']].drop_duplicates(),
                on=merge_keys,
                how='left'
            )
        else:
            oos_ledger = attach_setup_fp(oos_ledger)

    oos_ledger = _ensure_fold_column(oos_ledger, oos_summary)
    oos_ledger = _normalize_trade_date(oos_ledger)

    knowledge_map = _infer_knowledge_map_from_splits(splits)
    if knowledge_map is not None:
        oos_ledger['knowledge_d'] = oos_ledger['fold'].map(knowledge_map)
    else:
        oos_ledger['knowledge_rank'] = oos_ledger['fold'].astype(int)

    ticker_col = _first_nonnull_col(oos_ledger, ['ticker', 'specialized_ticker', 'asset', 'symbol'])
    if ticker_col is None:
        raise KeyError("No ticker-like column found among ['ticker','specialized_ticker','asset','symbol'].")

    sort_cols = ['setup_fp', ticker_col, 'direction', 'trade_d']
    if 'knowledge_d' in oos_ledger.columns and oos_ledger['knowledge_d'].notna().any():
        oos_ledger = oos_ledger.sort_values(sort_cols + ['knowledge_d'])
    else:
        oos_ledger = oos_ledger.sort_values(sort_cols + ['knowledge_rank'])

    deduped = oos_ledger.drop_duplicates(subset=sort_cols, keep='first').reset_index(drop=True)

    front_cols = ['setup_fp', 'setup_id', 'direction', ticker_col, 'fold', 'trade_d']
    if 'knowledge_d' in deduped.columns:
        front_cols.append('knowledge_d')
    elif 'knowledge_rank' in deduped.columns:
        front_cols.append('knowledge_rank')

    front_cols = [c for c in front_cols if c in deduped.columns]
    ordered = pd.concat([deduped[front_cols], deduped.drop(columns=[c for c in deduped.columns if c in front_cols])],
                        axis=1)
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
    mod = importlib.import_module(f'alpha_discovery.gauntlet.{module_basename}')
    prefer_names = prefer_names or []
    for name in prefer_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    for name in dir(mod):
        if name.startswith('run_stage') and callable(getattr(mod, name)):
            return getattr(mod, name)
    raise ImportError(f"No runnable stage function found in {module_basename}.")


def _call_stage(fn, df: pd.DataFrame, **kwargs):
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if 'ledger' in sig.parameters:
        return fn(ledger=df, **filtered)
    if 'df' in sig.parameters:
        return fn(df=df, **filtered)
    try:
        return fn(df, **filtered)
    except TypeError:
        return fn(**filtered)


def _write_if_df(obj, path):
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        obj.to_csv(path, index=False)
        return True
    return False


def _extract_df_from_stage_result(res, *keys):
    df, summary = None, None
    if isinstance(res, tuple) and res:
        df = res[0]
        if len(res) > 1 and isinstance(res[1], pd.DataFrame):
            summary = res[1]
    elif isinstance(res, dict):
        for k in ('df', 'ledger') + keys:
            if k in res and isinstance(res[k], pd.DataFrame):
                df = res[k];
                break
        for k in ('summary', 'stats', 'report'):
            if k in res and isinstance(res[k], pd.DataFrame):
                summary = res[k];
                break
    elif isinstance(res, pd.DataFrame):
        df = res
    return df, summary


# ---------------------------------
# STRICT OOS GAUNTLET (ACTIONABLE)
# ---------------------------------
def run_gauntlet_strict_oos(
        run_dir: str,
        splits=None,
        outdir: str | None = None,
        settings: Optional[Settings] = None,
        config: Optional[Dict[str, Any]] = None,
        master_df: Optional[pd.DataFrame] = None,
        signals_df: Optional[pd.DataFrame] = None,
        signals_metadata: Optional[List[Dict[str, Any]]] = None,
        **stage_kwargs
):
    """
    Strict-OOS selection using test-only evidence; then rebuild survivors to EOD.
    Final CSVs are written via your canonical writers to guarantee identical schema,
    and mirrored into gauntlet/strict_oos/.
    """
    if settings is None:
        from ..config import settings as default_settings
        settings = default_settings
    if config is None:
        config = get_permissive_gauntlet_config()

    out_base = _ensure_dir(os.path.join(run_dir, 'gauntlet', 'strict_oos')) if outdir is None else _ensure_dir(outdir)

    # 1) Build Strict-OOS stitched ledger (diagnostic only)
    strict_df = build_strict_oos_ledger(run_dir, splits=splits)
    strict_df.to_csv(os.path.join(out_base, 'strict_oos_stitched_ledger.csv'), index=False)

    # 2) Stage 1
    stage1_fn = _load_stage_func('stage1_health',
                                 prefer_names=['run_stage1_on_ledger', 'run_stage1_health_check', 'run_stage1'])
    oos_as_of_date = pd.to_datetime(strict_df['trade_d']).max().normalize() if 'trade_d' in strict_df.columns else None
    res1 = _call_stage(stage1_fn, strict_df, output_dir=out_base, mode='oos', stage='stage1',
                       as_of_date=oos_as_of_date, **stage_kwargs)
    s1_df, s1_summary = _extract_df_from_stage_result(res1)

    if s1_df is not None and len(s1_df) < 10 and 'setup_id' in s1_df.columns:
        s1_summary = s1_df
        s1_df = None

    legacy_noop = (s1_df is None) or (len(s1_df) == len(strict_df))
    misleading = False
    summary_to_check = s1_summary if s1_summary is not None else s1_df
    if isinstance(summary_to_check, pd.DataFrame) and not summary_to_check.empty:
        pass_cols = [c for c in summary_to_check.columns if 'pass' in c.lower()]
        if pass_cols and (summary_to_check[pass_cols].astype(str).isin(['False', 'false', '0']).any().any()):
            misleading = True

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
        s1_df = compat_df
        s1_summary = compat_sum

    _write_if_df(s1_summary, os.path.join(out_base, 'stage1_summary.csv'))

    # 3) Stage 2 (profitability summary; keep df stream in parallel)
    try:
        res2 = run_stage2_profitability_on_ledger(
            fold_ledger=s1_df,
            settings=settings,
            config=config,
            stage1_df=None,
        )
        s2_df, s2_summary = s1_df.copy(), res2
        s2_df.to_csv(os.path.join(out_base, 'stage2_oos.csv'), index=False)
        _write_if_df(s2_summary, os.path.join(out_base, 'stage2_summary.csv'))
    except Exception as e:
        print(f"Stage 2 failed: {e}, using pass-through")
        s2_df = s1_df.copy()
        s2_df.to_csv(os.path.join(out_base, 'stage2_oos.csv'), index=False)

    # 4) Stage 3 (apply real filtering based on summary pass column)
    try:
        res3 = run_stage3_robustness_on_ledger(
            fold_ledger=s2_df,
            settings=settings,
            config=config,
            stage2_df=None,
        )
        s3_df = s2_df.copy()
        s3_summary = res3 if isinstance(res3, pd.DataFrame) else None
        if isinstance(s3_summary, pd.DataFrame) and not s3_summary.empty:
            # Find pass column
            pass_col = None
            for c in s3_summary.columns:
                cl = c.lower()
                if 'pass' in cl and ('stage3' in cl or cl == 'pass' or cl.endswith('_pass')):
                    pass_col = c
                    break
            if pass_col is None:
                cand = [c for c in s3_summary.columns if 'pass' in c.lower()]
                if cand:
                    pass_col = cand[0]
            if pass_col is not None:
                survivor_key_cols = [c for c in ["setup_fp", "setup_id", "specialized_ticker", "direction"] if
                                     c in s3_df.columns and c in s3_summary.columns]
                survivors = s3_summary.loc[s3_summary[pass_col].astype(bool)]
                if survivor_key_cols and not survivors.empty:
                    survivors = survivors[survivor_key_cols].drop_duplicates()
                    s3_df = s3_df.merge(survivors, on=survivor_key_cols, how="inner")
        s3_df.to_csv(os.path.join(out_base, 'stage3_oos.csv'), index=False)
        _write_if_df(s3_summary, os.path.join(out_base, 'stage3_summary.csv'))
    except Exception as e:
        print(f"Stage 3 failed: {e}, using pass-through")
        s3_df = s2_df.copy()
        s3_df.to_csv(os.path.join(out_base, 'stage3_oos.csv'), index=False)

    # 5) Survivors ledger (post Stage-3)
    survivors_df = s3_df if isinstance(s3_df, pd.DataFrame) else s2_df
    survivors_df.to_csv(os.path.join(out_base, 'gauntlet_ledger.csv'), index=False)

    # 6) Rebuild survivors to EOD (preferred) OR mark opens as-of last OOS
    ledger_for_outputs = None
    if master_df is not None and signals_df is not None and signals_metadata is not None and not survivors_df.empty:
        # Build unique survivor setups and re-backtest each
        try:
            oos_summary, _ = read_oos_artifacts(run_dir)
        except Exception:
            oos_summary = pd.DataFrame()

        uniq_cols = [c for c in ["setup_id", "setup_fp", "specialized_ticker", "direction", "signal_ids", "fold"] if
                     c in survivors_df.columns]
        uniq = survivors_df[uniq_cols].drop_duplicates().copy()

        if "signal_ids" not in uniq.columns or uniq["signal_ids"].isna().any():
            if not oos_summary.empty and "setup_id" in uniq.columns and "setup_id" in oos_summary.columns:
                uniq = uniq.merge(
                    oos_summary[["setup_id", "signal_ids"]].drop_duplicates(),
                    on="setup_id",
                    how="left",
                    suffixes=("", "_oos")
                )
                if "signal_ids" in uniq.columns and "signal_ids_oos" in uniq.columns:
                    uniq["signal_ids"] = uniq["signal_ids"].fillna(uniq["signal_ids_oos"])
                    uniq = uniq.drop(columns=["signal_ids_oos"])

        backtest_tasks = []
        for _, row in uniq.iterrows():
            setup_id = str(row.get("setup_id", ""))
            ticker = str(row.get("specialized_ticker", ""))
            direction = str(row.get("direction", ""))
            signal_ids_raw = row.get("signal_ids")
            signal_ids = [s.strip() for s in str(signal_ids_raw).split(",")] if pd.notnull(signal_ids_raw) else []
            try:
                oos_start = pd.to_datetime(
                    survivors_df.loc[
                        (survivors_df.get("setup_id") == row.get("setup_id")) &
                        (survivors_df.get("specialized_ticker") == row.get("specialized_ticker")) &
                        (survivors_df.get("direction") == row.get("direction")),
                        "trade_d"
                    ].min()
                ) - pd.Timedelta(days=1)
                if pd.isna(oos_start):
                    oos_start = None
            except Exception:
                oos_start = None

            backtest_tasks.append(delayed(run_gauntlet_backtest)(
                setup_id=setup_id,
                specialized_ticker=ticker,
                signal_ids=signal_ids,
                direction=direction,
                oos_start_date=oos_start,
                master_df=master_df,
                signals_df=signals_df,
                signals_metadata=signals_metadata,
                exit_policy=None,
                settings=settings,
                origin_fold=int(row["fold"]) if "fold" in row and pd.notna(row["fold"]) else 0,
            ))

        if backtest_tasks:
            with tqdm(total=len(backtest_tasks), desc="Rebuilding deployment ledger for survivors",
                      dynamic_ncols=True) as pbar:
                with Parallel(n_jobs=4, backend="loky", timeout=300) as parallel:
                    dep_results = parallel(backtest_tasks)
                    pbar.update(len(backtest_tasks))
            dep_results = [d for d in dep_results if isinstance(d, pd.DataFrame) and not d.empty]
            if dep_results:
                ledger_for_outputs = pd.concat(dep_results, ignore_index=True)
                ledger_for_outputs.to_csv(os.path.join(out_base, "deploy_ledger.csv"), index=False)

    if ledger_for_outputs is None:
        # Best-effort: mark opens as of last OOS date
        # Use the actual data end date, not max trade_d which might be earlier
        # This ensures we properly mark trades as open if they extend beyond data availability
        data_end_date = pd.Timestamp("2025-09-15").normalize()  # Your actual data end date
        
        # Clean up any fictional future exit dates before marking open positions
        survivors_clean = survivors_df.copy()
        if "exit_date" in survivors_clean.columns:
            survivors_clean["exit_date"] = pd.to_datetime(survivors_clean["exit_date"], errors="coerce")
            # Set any exit dates after data end to NaT (these are fictional)
            fictional_exits = survivors_clean["exit_date"] > data_end_date
            if fictional_exits.any():
                print(f"[Strict-OOS] Found {fictional_exits.sum()} trades with fictional future exit dates, marking as open")
                survivors_clean.loc[fictional_exits, "exit_date"] = pd.NaT
        
        ledger_for_outputs = _mark_open_positions_at_eod(
            ledger=survivors_clean,
            oos_end=data_end_date,
            last_mark=None
        )

    # 7) *** Write Strict OOS results directly to strict_oos directory ***
    # Use the full summary functions but redirect their output to strict_oos directory
    
    # Create a custom function to write summaries to strict_oos instead of main gauntlet
    def write_to_strict_oos(summary_func, ledger):
        """Write summary to strict_oos directory by temporarily modifying the gauntlet path"""
        import tempfile
        import shutil
        
        # Create a temporary run directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_run_dir = os.path.join(temp_dir, "temp_run")
            temp_gauntlet_dir = os.path.join(temp_run_dir, "gauntlet")
            os.makedirs(temp_gauntlet_dir, exist_ok=True)
            
            # Generate the summary in the temp directory
            result_path = summary_func(temp_run_dir, ledger)
            
            # Copy the result to the strict_oos directory
            if os.path.exists(result_path):
                filename = os.path.basename(result_path)
                dest_path = os.path.join(out_base, filename)
                shutil.copy2(result_path, dest_path)
                print(f"Strict OOS {filename} generated at: {dest_path}")
                return dest_path
            else:
                print(f"Warning: {summary_func.__name__} did not generate expected file")
                return None
    
    # Generate full summaries for strict_oos directory
    if ledger_for_outputs is not None and not ledger_for_outputs.empty:
        try:
            # Write full open trades summary to strict_oos
            write_to_strict_oos(write_open_trades_summary, ledger_for_outputs)
            
            # Write full all setups summary to strict_oos  
            write_to_strict_oos(write_all_setups_summary, ledger_for_outputs)
            
            # Write full gauntlet summary to strict_oos (this was missing!)
            # Note: write_gauntlet_summary needs survivors list, so we'll create a minimal one
            if survivors_df is not None and not survivors_df.empty:
                # Create a minimal survivors list for the gauntlet summary
                survivors_list = []
                for _, row in survivors_df.iterrows():
                    survivors_list.append({
                        'setup_id': row.get('setup_id', 'unknown'),
                        'rank': row.get('rank', 0),
                        'fold': row.get('fold', 0)
                    })
                
                # Write gauntlet summary using the survivors
                def write_gauntlet_to_strict_oos(temp_run_dir, ledger):
                    return write_gauntlet_summary(temp_run_dir, survivors_list, ledger)
                
                write_to_strict_oos(write_gauntlet_to_strict_oos, ledger_for_outputs)
            
        except Exception as e:
            print(f"Error generating strict OOS summaries: {e}")
            # Create minimal fallback files
            empty_df = pd.DataFrame(columns=[
                "setup_id", "specialized_ticker", "direction", "description", "signal_ids",
                "oos_first_trigger", "oos_last_trigger", "oos_days_since_last_trigger",
                "oos_total_trades", "oos_open_trades", "oos_sum_pnl_dollars", 
                "oos_final_nav", "oos_nav_total_return_pct", "expectancy"
            ])
            from .summary import _write_csv
            _write_csv(os.path.join(out_base, "open_trades_summary.csv"), empty_df)
    else:
        print("No ledger data for strict OOS summaries")

    # Note: We now write directly to strict_oos directory above, no mirroring needed

    return {
        'outdir': out_base,
        'stage1_rows': 0 if s1_df is None else len(s1_df),
        'stage2_rows': 0 if s2_df is None else len(s2_df),
        'stage3_rows': 0 if s3_df is None else len(s3_df),
        'survivor_rows': 0 if survivors_df is None else len(survivors_df)
    }


# -------------------------
# MEDIUM GAUNTLET WRAPPER
# -------------------------
def _run_medium_gauntlet_mode(run_dir: str, oos_ledgers: Dict[str, pd.DataFrame],
                              settings: Settings, config: Dict[str, Any]) -> None:
    """Run Medium Gauntlet mode."""
    from .medium import get_medium_gauntlet_config

    medium_config = get_medium_gauntlet_config()
    if config:
        medium_config.update(config)

    setup_results = []
    for setup_id, ledger in oos_ledgers.items():
        if ledger.empty:
            continue
        try:
            base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
        except Exception:
            base_cap = 100_000.0

        daily_returns = nav_daily_returns_from_ledger(ledger, base_capital=base_cap)
        equity_curve = (1 + daily_returns).cumprod() * base_cap

        ticker = ledger.get('specialized_ticker', ['Unknown']).iloc[0] if not ledger.empty else 'Unknown'
        direction = ledger.get('direction', ['long']).iloc[0] if not ledger.empty else 'long'

        setup_result = SetupResults(
            setup_id=setup_id,
            ticker=ticker,
            direction=direction,
            oos_trades=ledger,
            oos_daily_returns=daily_returns,
            oos_equity_curve=equity_curve,
            historical_median_drawdown=None
        )
        setup_results.append(setup_result)

    if not setup_results:
        print("No valid setups found for Medium Gauntlet.")
        return

    print(f"Running Medium Gauntlet on {len(setup_results)} setups...")

    results = run_medium_gauntlet(setup_results, settings, medium_config)

    output_dir = os.path.join(run_dir, "gauntlet_medium")
    write_gauntlet_artifacts(results, output_dir, medium_config)

    deploy_count = len([r for r in results if r.final_decision == "Deploy"])
    monitor_count = len([r for r in results if r.final_decision == "Monitor"])
    retire_count = len([r for r in results if r.final_decision == "Retire"])

    print(f"\nMedium Gauntlet Results:")
    print(f"  Deploy: {deploy_count}")
    print(f"  Monitor: {monitor_count}")
    print(f"  Retire: {retire_count}")
    print(f"  Artifacts written to: {output_dir}")
