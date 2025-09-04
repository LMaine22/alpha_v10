# alpha_discovery/gauntlet/summary.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
import pandas as pd

from ..config import settings
from .io import read_global_artifacts

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce") if s is not None else s

def _write_csv(path: str, df: pd.DataFrame) -> str:
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    return path

def _flatten_survivors(survivors: List[Dict[str, Any]]) -> pd.DataFrame:
    if not survivors:
        return pd.DataFrame(columns=["setup_id"])
    rows = []
    for rec in survivors:
        sid = str(rec.get("setup_id"))
        # Try to carry fold through if present in survivor records
        fld = rec.get("fold") or rec.get("origin_fold") or rec.get("train_fold")
        s1 = {f"s1_{k}": v for k, v in (rec.get("oos_s1") or {}).items()}
        s2 = {f"s2_{k}": v for k, v in (rec.get("oos_s2") or {}).items()}
        s3 = {f"s3_{k}": v for k, v in (rec.get("oos_s3") or {}).items()}
        row = {"setup_id": sid}
        if fld is not None:
            try:
                row["fold"] = int(fld)
            except Exception:
                row["fold"] = fld
        row.update(s1)
        row.update(s2)
        row.update(s3)
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(subset=["setup_id"], keep="last")
    for src, dst in [
        ("s2_pvalue_sharpe_gt0", "oos_pvalue"),
        ("s3_dsr", "oos_dsr"),
        ("s3_N_eff", "oos_N_eff"),
    ]:
        if src in df.columns:
            df[dst] = df[src]
    return df


def write_gauntlet_summary(
    run_dir: str,
    survivors: List[Dict[str, Any]],
    full_oos_ledger: pd.DataFrame,
) -> str:
    """
    Final OOS gauntlet summary. Recency is computed from trigger/entry (NOT exit).
    """
    if survivors is None or len(survivors) == 0:
        raise ValueError("No survivors passed to write_gauntlet_summary.")

    gaunt_dir = os.path.join(run_dir, "gauntlet")
    os.makedirs(gaunt_dir, exist_ok=True)

    # Survivors
    df_s = _flatten_survivors(survivors)

    # Ledger normalization
    led = full_oos_ledger.copy()
    for dc in ["trigger_date", "entry_date", "exit_date"]:
        if dc in led.columns:
            led[dc] = pd.to_datetime(led[dc], errors="coerce")

    # Recency key
    recency_col = "trigger_date" if ("trigger_date" in led.columns and led["trigger_date"].notna().any()) else "entry_date"

    # Setup key
    setup_col = None
    for c in ["setup_id", "Setup", "strategy_id", "StrategyID"]:
        if c in led.columns:
            setup_col = c
            break
    if setup_col is None:
        led["_tmp_setup_id"] = "unknown"
        setup_col = "_tmp_setup_id"

    # Per-setup recency
    recency = (
        led.groupby(setup_col)[recency_col]
        .agg(oos_first_trigger="min", oos_last_trigger="max")
        .reset_index()
        .rename(columns={setup_col: "setup_id"})
    )

    # Open/closed counts
    led["is_open"] = led["exit_date"].isna() if "exit_date" in led.columns else False
    counts = (
        led.groupby(setup_col)["is_open"].agg(oos_open_trades="sum", oos_total_trades="count")
        .reset_index().rename(columns={setup_col: "setup_id"})
    )
    counts["oos_closed_trades"] = counts["oos_total_trades"] - counts["oos_open_trades"]

    # Data end (for days-since-last-trigger)
    data_end_candidates = [led[dc].max() for dc in ["trigger_date", "entry_date", "exit_date"] if dc in led.columns]
    data_end = max([d for d in data_end_candidates if pd.notnull(d)]) if data_end_candidates else pd.NaT

    # OOS performance aggregates per setup (include unrealized PnL for OPEN)
    realized = pd.to_numeric(led.get("pnl_dollars", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    unreal = pd.to_numeric(led.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    is_open = led["is_open"].astype(bool)
    led["_oos_total_pnl"] = realized + (unreal.where(is_open, 0.0))

    agg = led.groupby(setup_col).agg(
        oos_sum_pnl_dollars=("_oos_total_pnl", "sum"),
        origin_fold=("fold", "first") if "fold" in led.columns else (setup_col, "size"),
    ).reset_index().rename(columns={setup_col: "setup_id"})

    # NAV base capital — use REPORTING config (was mistakenly taken from selection before)
    oos_initial_nav = float(getattr(settings.reporting, "base_capital_for_portfolio", 100000.0))
    agg["oos_final_nav"] = oos_initial_nav + agg["oos_sum_pnl_dollars"].astype(float)
    agg["oos_nav_total_return_pct"] = (agg["oos_final_nav"] / oos_initial_nav - 1.0) * 100.0
    if "origin_fold" in agg.columns:
        agg = agg.rename(columns={"origin_fold": "fold"})
        try:
            agg["fold"] = agg["fold"].astype(int)
        except Exception:
            pass

    # Merge TRAIN artifacts
    pareto_df, _ = read_global_artifacts(run_dir)
    if not pareto_df.empty:
        pareto_df["setup_id"] = pareto_df["setup_id"].astype(str)
        # Only join on 'fold' if BOTH sides have it
        if ("fold" in df_s.columns) and ("fold" in pareto_df.columns):
            df_s = df_s.merge(pareto_df, on=["setup_id", "fold"], how="left", suffixes=("", ""))
        else:
            df_s = df_s.merge(pareto_df, on="setup_id", how="left", suffixes=("", ""))
            # After this merge, df_s will inherit 'fold' from pareto_df if it exists there

    # Final join
    final_df = (
        df_s.merge(recency, on="setup_id", how="left")
            .merge(counts, on="setup_id", how="left")
            .merge(agg, on="setup_id", how="left")
    )

    # Days since last trigger
    if "oos_last_trigger" in final_df.columns and pd.notnull(data_end):
        final_df["oos_days_since_last_trigger"] = (
            pd.to_datetime(data_end) - pd.to_datetime(final_df["oos_last_trigger"]).dt.tz_localize(None)
        ).dt.days

    # Column order
    col_order = [
        # IDs / meta
        "setup_id", "fold", "rank", "specialized_ticker", "direction", "description", "signal_ids",
        # TRAIN perf snapshot
        "support", "trades_count", "sum_pnl_dollars", "nav_total_return_pct", "final_nav",
        "first_trigger_date", "last_trigger_date", "best_performing_ticker",
        "sharpe_lb", "sharpe_median", "sharpe_ub", "sortino_lb", "sortino_median", "sortino_ub",
        "omega_ratio", "max_drawdown", "sum_capital_allocated",
        # Gauntlet stages
        "s1_pass_stage1", "s1_rank", "s1_days_since_last_trigger", "s1_short_window_days",
        "s1_trades_in_short_window", "s1_max_dd_short", "s1_recency_max_days", "s1_min_trades_short",
        "s1_max_dd_short_cap", "s1_reason",
        "oos_pvalue", "s2_T", "s2_mbb_block_len", "s2_sr_train", "s2_pvalue_sharpe_gt0",
        "oos_dsr", "oos_N_eff", "s3_fdr_pass",
        # OOS activity + performance
        "oos_first_trigger", "oos_last_trigger", "oos_days_since_last_trigger",
        "oos_total_trades", "oos_open_trades", "oos_closed_trades",
        "oos_sum_pnl_dollars", "oos_final_nav", "oos_nav_total_return_pct",
    ]
    existing_cols = [c for c in col_order if c in final_df.columns]
    final_df = final_df[existing_cols + [c for c in final_df.columns if c not in existing_cols]]

    # Sort — prioritize DSR, then pvalue, then recency
    sort_cols = []
    if "oos_dsr" in final_df.columns:
        sort_cols.append(("oos_dsr", False))
    if "oos_pvalue" in final_df.columns:
        sort_cols.append(("oos_pvalue", True))
    if "oos_days_since_last_trigger" in final_df.columns:
        sort_cols.append(("oos_days_since_last_trigger", True))
    if sort_cols:
        final_df = final_df.sort_values(by=[c for c, _ in sort_cols], ascending=[a for _, a in sort_cols])

    out_path = os.path.join(gaunt_dir, "gauntlet_summary.csv")
    _write_csv(out_path, final_df)
    print(f"Definitive gauntlet summary successfully generated at: {out_path}")
    return out_path
