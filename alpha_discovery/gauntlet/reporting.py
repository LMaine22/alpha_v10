# alpha_discovery/gauntlet/reporting.py

from __future__ import annotations
import os
import json
from typing import Optional, List

import numpy as np
import pandas as pd

from .io import ensure_dir


# Columns we want the gauntlet ledger to carry (in order),
# including the new partial-exit / scale-out fields.
LEDGER_BASE_SCHEMA: List[str] = [
    # Identifiers / provenance
    "setup_id", "origin_fold", "specialized_ticker",

    # Timing
    "trigger_date", "entry_date", "exit_date",

    # Instrument details
    "ticker", "horizon_days", "direction", "option_type", "strike",

    # Underlier / vol at entry/exit
    "entry_underlying", "exit_underlying", "entry_iv", "exit_iv",

    # Option execution prices
    "entry_option_price", "exit_option_price",

    # Sizing / capital
    "contracts", "contracts_partial", "contracts_remaining",
    "capital_allocated", "capital_allocated_used",

    # PnL
    "pnl_dollars", "pnl_pct",

    # Exit attribution & policy
    "first_exit_reason",          # e.g. profit_target_partial
    "exit_reason",                # final exit reason (trail/SL/time/horizon)
    "exit_policy_id", "holding_days_actual",

    # Partial-exit details
    "partial_exit_date", "partial_exit_price", "partial_exit_frac",

    # (Legacy/reporting extras sometimes added downstream; keep slots here
    # so they show up first if present.)
    "status", "exit_price", "unrealized_pnl", "realized_pnl",
    "signal_ids", "holding_days_open",
]


def _gauntlet_dir(run_dir: str) -> str:
    d = os.path.join(run_dir, "gauntlet")
    ensure_dir(d)
    return d


def _coerce_schema(df: pd.DataFrame, base_schema: Optional[List[str]]) -> pd.DataFrame:
    """
    Make the DataFrame forward-compatible by:
      - adding any missing columns as NaN
      - ordering columns with base_schema first (in order), then any extras
    If base_schema is None, pass through unchanged.
    """
    if df is None or len(df.columns) == 0:
        return pd.DataFrame(columns=base_schema or [])
    if not base_schema:
        return df

    out = df.copy()

    # Add any missing base columns as NaN
    for col in base_schema:
        if col not in out.columns:
            out[col] = np.nan

    # Place base columns first (preserve order), then all remaining columns
    ordered = [c for c in base_schema if c in out.columns]
    tail = [c for c in out.columns if c not in ordered]
    out = out.loc[:, ordered + tail]

    return out


def write_stage_csv(
    run_dir: str,
    name: str,
    df: pd.DataFrame,
    base_schema: Optional[List[str]] = None,
) -> str:
    """
    Generic CSV writer used by the gauntlet stages. Ensures directories exist,
    optionally coerces to a forward-compatible schema, and writes a CSV.

    IMPORTANT: If `name == "gauntlet_ledger"` and `base_schema` is not provided,
    this function will automatically apply LEDGER_BASE_SCHEMA so your new
    scale-out columns always show up (even when some are missing/NaN).
    """
    d = _gauntlet_dir(run_dir)
    path = os.path.join(d, f"{name}.csv")

    # Auto-apply gauntlet ledger schema if caller didn't pass one.
    effective_schema = base_schema
    if base_schema is None and name == "gauntlet_ledger":
        effective_schema = LEDGER_BASE_SCHEMA

    # Coerce and write with a safe fallback
    try:
        out = _coerce_schema(df, effective_schema)
        out.to_csv(path, index=False)
    except Exception:
        out = _coerce_schema(df.reset_index(drop=True), effective_schema)
        out.to_csv(path, index=False)

    return path


def write_json(run_dir: str, name: str, payload: dict) -> str:
    """
    Convenience: write a small JSON artifact under runs/<ts>/gauntlet/.
    """
    d = _gauntlet_dir(run_dir)
    path = os.path.join(d, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(payload or {}, fh, indent=2, default=str)
    return path


def write_readme(run_dir: str, extra: Optional[dict] = None) -> str:
    """
    Write a short README describing the gauntlet outputs and (optionally) the
    configuration used.
    """
    d = _gauntlet_dir(run_dir)
    readme = os.path.join(d, "README.txt")
    lines: List[str] = [
        "Gauntlet outputs",
        "=================",
        "",
        "Files:",
        "  - stage1_recency_liveness.csv : Multi-lookback EWMA Sharpe + liveness/risk gates per fold.",
        "  - stage2_mbb_pvalues.csv      : Moving Block Bootstrap p-values and block lengths.",
        "  - stage3_fdr_dsr.csv          : Benjamini–Hochberg FDR and Deflated Sharpe across the cohort.",
        "  - gauntlet_ledger.csv         : Full OOS trade ledger (forward-compatible schema).",
        "",
        "Notes:",
        "  * Stage-1 uses short/medium/long EWMAs and a live-trigger gate.",
        "  * Stage-2 uses block bootstrap (overlapping) to respect autocorrelation.",
        "  * Stage-3 applies Benjamini–Hochberg FDR to Stage-2 p-values, then computes DSR.",
        "",
    ]
    if extra:
        lines.append("Config:")
        lines.append(json.dumps(extra, indent=2))
        lines.append("")
    with open(readme, "w") as fh:
        fh.write("\n".join(lines))
    return readme
