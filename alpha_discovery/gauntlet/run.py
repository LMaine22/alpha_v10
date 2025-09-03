# alpha_discovery/gauntlet/run.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import Settings  # type: ignore
from .io import find_latest_run_dir, read_global_artifacts
from .stage1_recency import run_stage1_recency_liveness
from .stage2_mbb import run_stage2_mbb_on_ledger
from .stage3_fdr_dsr import stage3_cohort_oos
from .summary import write_gauntlet_summary
from .backtester import run_gauntlet_backtest
from .reporting import write_stage_csv, LEDGER_BASE_SCHEMA


def run_gauntlet(
    run_dir: Optional[str] = None,
    settings: Optional[Settings] = None,
    config: Optional[Dict[str, Any]] = None,
    master_df: Optional[pd.DataFrame] = None,
    signals_df: Optional[pd.DataFrame] = None,
    signals_metadata: Optional[List[Dict[str, Any]]] = None,
    splits: Optional[List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]] = None,
) -> None:
    """
    Gauntlet pipeline:
      1) Collect all Pareto-front candidates across training folds.
      2) Run out-of-sample (OOS) backtests per candidate to build the gauntlet ledger.
      3) Stage-1/Stage-2 per setup, then Stage-3 cohort across survivors.
      4) Write summary & artifacts.
    """
    cfg = dict(config or {})

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
            with Parallel(n_jobs=-1, backend="loky") as parallel:
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
    write_stage_csv(run_dir, "gauntlet_ledger", full_oos_ledger, base_schema=LEDGER_BASE_SCHEMA)
    print(f"\nOut-of-sample backtesting complete. Generated OOS ledgers for {len(oos_ledgers)} strategies.")

    # -----------------------------------------------------------------
    # Phase 3: Stage-1/2 per setup, then cohort-wide Stage-3 on OOS set
    # -----------------------------------------------------------------
    print("\n--- Phase 3: OOS Stage-1/2 per-setup, then cohort-wide Stage-3 ---")
    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    ret_map: Dict[str, pd.Series] = {}

    for setup_id, oos_ledger in oos_ledgers.items():
        # Stage-1 per setup: recency/liveness gates
        s1 = run_stage1_recency_liveness(
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

        # Stage-2 per setup: block bootstrap p-values
        s2 = run_stage2_mbb_on_ledger(
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

        # Daily returns for Stage-3 cohort calc
        from ..eval.nav import nav_daily_returns_from_ledger
        base_cap = float(getattr(getattr(settings, "reporting", object()), "base_capital_for_portfolio", 100_000.0))
        ret_map[setup_id] = nav_daily_returns_from_ledger(oos_ledger, base_capital=base_cap)

    # Write Stage-1/2 diagnostics
    if stage1_rows:
        write_stage_csv(run_dir, "stage1_oos", pd.DataFrame(stage1_rows))
    if stage2_rows:
        write_stage_csv(run_dir, "stage2_oos", pd.DataFrame(stage2_rows))

    if not stage2_rows:
        print("No candidates reached Stage-2 on OOS; gauntlet complete.")
        return

    # Cohort Stage-3
    oos_s2_all = pd.DataFrame(stage2_rows)
    cohort_s3 = stage3_cohort_oos(oos_s2_all, ret_map, base_capital, cfg)
    write_stage_csv(run_dir, "stage3_cohort_oos", cohort_s3)

    # Survivors & summary
    survivor_ids = set(cohort_s3.loc[cohort_s3["fdr_pass"], "setup_id"].astype(str))
    print(f"Cohort Stage-3 survivors: {len(survivor_ids)}")

    by_id_s1 = {str(d["setup_id"]): d for d in stage1_rows if "setup_id" in d}
    by_id_s2 = {str(d["setup_id"]): d for d in stage2_rows if "setup_id" in d}
    by_id_s3 = cohort_s3.set_index("setup_id").to_dict(orient="index")

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
        print("No strategies survived cohort-wide Stage-3.")
