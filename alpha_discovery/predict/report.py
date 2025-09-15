from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Optional

from .config import PredictConfig

def _dir_label(p: float, hi: float = 0.60, lo: float = 0.40) -> str:
    if p >= hi:
        return "Bullish"
    if p <= lo:
        return "Bearish"
    return "Neutral"

def _pct(x: float) -> str:
    return f"{x*100:.1f}%"

def _to_pct_range(med: float, q10: float, q90: float) -> str:
    return f"{med*100:.1f}% [{q10*100:.1f}%, {q90*100:.1f}%]"

def build_outlook_table(forecasts: pd.DataFrame,
                        cfg: PredictConfig,
                        top_n: int = 25,
                        prob_min: float = 0.55) -> pd.DataFrame:
    """
    Converts the raw per-horizon forecasts into a readable daily outlook table.
    """
    df = forecasts.copy()

    # Narrative columns
    df["stance"] = df["dir_prob"].apply(lambda p: _dir_label(p))
    df["median_move"] = df.apply(lambda r: _to_pct_range(r["ret_q50"], r["ret_q10"], r["ret_q90"]), axis=1)
    df["confidence"] = df["dir_prob"].round(3)

    # Filter to actionable horizon-views
    df = df[df["dir_prob"].between(prob_min, 1 - (prob_min - 0.5), inclusive="both")]
    # Sort by strongest conviction within horizon
    df = df.sort_values(["horizon_days", "dir_prob"], ascending=[True, False])

    # Take top-N per horizon
    top_frames = []
    for H in sorted(df["horizon_days"].unique()):
        sub = df[df["horizon_days"] == H].head(top_n).copy()
        top_frames.append(sub)
    out = pd.concat(top_frames, ignore_index=True)

    # Final presentation columns
    cols = [
        cfg.pred_date_col, cfg.ticker_col, "horizon_days",
        "stance", "confidence", "median_move",
        "ret_q10", "ret_q50", "ret_q90", "dir_prob"
    ]
    return out[cols]

def save_outlook_csv(table: pd.DataFrame, cfg: PredictConfig, fname: Optional[str] = None) -> str:
    out_path = os.path.join(cfg.outdir, "forecasts")
    os.makedirs(out_path, exist_ok=True)
    if fname is None:
        fname = "latest_outlook.csv"
    full = os.path.join(out_path, fname)
    table.to_csv(full, index=False)
    return full
