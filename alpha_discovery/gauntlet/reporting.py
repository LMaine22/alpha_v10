# alpha_discovery/gauntlet/reporting.py

from __future__ import annotations
import os
import json
import pandas as pd
import numpy as np                 # <-- ADD
from typing import Optional, List   # <-- ADD
from .io import ensure_dir

def _gauntlet_dir(run_dir: str) -> str:
    d = os.path.join(run_dir, "gauntlet")
    ensure_dir(d)
    return d

def _coerce_schema(df: pd.DataFrame, base_schema: Optional[List[str]]) -> pd.DataFrame:
    """
    Make the DataFrame forward-compatible by:
      - adding any missing columns as NaN
      - ordering columns with base_schema first (in order), then any extras
    """
    if not base_schema:
        return df

    out = df.copy()
    for col in base_schema:
        if col not in out.columns:
            out[col] = np.nan

    # place base_schema columns first, then any remaining
    ordered = [c for c in base_schema if c in out.columns]
    tail = [c for c in out.columns if c not in ordered]
    return out[ordered + tail]

def write_stage_csv(run_dir: str, name: str, df: pd.DataFrame, base_schema: Optional[List[str]] = None) -> str:
    d = _gauntlet_dir(run_dir)
    path = os.path.join(d, f"{name}.csv")
    try:
        df = _coerce_schema(df, base_schema)
        df.to_csv(path, index=False)
    except Exception:
        df = _coerce_schema(df.reset_index(drop=True), base_schema)
        df.to_csv(path, index=False)
    return path

def write_readme(run_dir: str, extra: Optional[dict] = None) -> str:
    ...


def write_readme(run_dir: str, extra: Optional[dict] = None) -> str:
    d = _gauntlet_dir(run_dir)
    readme = os.path.join(d, "README.txt")
    lines = [
        "Gauntlet outputs",
        "=================",
        "",
        "Files:",
        "  - stage1_recency_liveness.csv : Multi-lookback EWMA Sharpe + liveness/risk gates per fold.",
        "  - stage2_mbb_pvalues.csv      : Moving Block Bootstrap p-values and block lengths.",
        "  - stage3_fdr_dsr.csv          : FDR filtering and Deflated Sharpe Ratio (per-fold survivors).",
        "  - final_cohort.csv            : Combined survivors across folds with DSR and N_eff.",
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
