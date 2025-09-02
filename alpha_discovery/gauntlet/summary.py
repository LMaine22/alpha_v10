# alpha_discovery/gauntlet/summary.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
import pandas as pd


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce") if s is not None else s


def _write_csv(path: str, df: pd.DataFrame) -> str:
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f")
    return path


def _flatten_survivors(survivors: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten [{'setup_id', 'oos_s1': {...}, 'oos_s2': {...}, 'oos_s3': {...}}, ...]
    into a single DataFrame keyed by setup_id.
    """
    if not survivors:
        return pd.DataFrame(columns=["setup_id"])
    rows = []
    for rec in survivors:
        sid = str(rec.get("setup_id"))
        s1 = {f"s1_{k}": v for k, v in (rec.get("oos_s1") or {}).items()}
        s2 = {f"s2_{k}": v for k, v in (rec.get("oos_s2") or {}).items()}
        s3 = {f"s3_{k}": v for k, v in (rec.get("oos_s3") or {}).items()}
        row = {"setup_id": sid}
        row.update(s1)
        row.update(s2)
        row.update(s3)
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(subset=["setup_id"], keep="last")
    # Keep some friendly aliases if present
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

    # --- Survivors (flatten Stage-1/2/3) ---
    df_s = _flatten_survivors(survivors)

    # --- Ledger-level recency + open/closed counts ---
    led = full_oos_ledger.copy()
    # Coerce common date columns
    for dc in ["trigger_date", "entry_date", "exit_date"]:
        if dc in led.columns:
            led[dc] = pd.to_datetime(led[dc], errors="coerce")

    # Choose the event used for "recency"
    # Prefer trigger_date; if missing, fall back to entry_date.
    if "trigger_date" in led.columns and led["trigger_date"].notna().any():
        recency_col = "trigger_date"
    else:
        recency_col = "entry_date"

    # Map per-setup recency and first/last triggers
    # We infer setup_id column; if absent, try 'Setup' or use a stable key.
    setup_col = None
    for c in ["setup_id", "Setup", "strategy_id", "StrategyID"]:
        if c in led.columns:
            setup_col = c
            break
    if setup_col is None:
        # If the ledger doesn't carry setup_id, use the survivors list as the index
        led["_tmp_setup_id"] = "unknown"
        setup_col = "_tmp_setup_id"

    recency = (
        led.groupby(setup_col)[recency_col]
        .agg(oos_first_trigger="min", oos_last_trigger="max")
        .reset_index()
        .rename(columns={setup_col: "setup_id"})
    )

    # Open/closed counts (exit_date NaT means OPEN)
    if "exit_date" in led.columns:
        led["is_open"] = led["exit_date"].isna()
    else:
        led["is_open"] = False
    counts = (
        led.groupby(setup_col)["is_open"]
        .agg(oos_open_trades="sum", oos_total_trades="count")
        .reset_index()
        .rename(columns={setup_col: "setup_id"})
    )
    counts["oos_closed_trades"] = counts["oos_total_trades"] - counts["oos_open_trades"]

    # Data end (for days-since-last-trigger)
    data_end_candidates = []
    for dc in ["trigger_date", "entry_date", "exit_date"]:
        if dc in led.columns:
            data_end_candidates.append(led[dc].max())
    data_end = max([d for d in data_end_candidates if pd.notnull(d)]) if data_end_candidates else pd.NaT

    # Merge
    final_df = df_s.merge(recency, on="setup_id", how="left").merge(counts, on="setup_id", how="left")

    # Days since last trigger (if we have a finite data_end)
    final_df["oos_days_since_last_trigger"] = (
        (pd.to_datetime(data_end, errors="coerce").normalize() - pd.to_datetime(final_df["oos_last_trigger"]).dt.normalize())
        .dt.days
    )

    # Friendly ordering if present
    col_order = [
        "setup_id",
        # Stage 1 highlights (if available)
        "s1_pass_stage1",
        # Stage 2
        "oos_pvalue", "s2_T", "s2_mbb_block_len",
        # Stage 3
        "oos_dsr", "oos_N_eff", "s3_fdr_pass",
        # Activity
        "oos_first_trigger", "oos_last_trigger", "oos_days_since_last_trigger",
        # Trade counts
        "oos_total_trades", "oos_open_trades", "oos_closed_trades",
    ]
    existing_cols = [c for c in col_order if c in final_df.columns]
    final_df = final_df[existing_cols + [c for c in final_df.columns if c not in existing_cols]]

    # Sort — prioritize DSR if present, else recent activity
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
