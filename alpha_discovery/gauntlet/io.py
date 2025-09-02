# alpha_discovery/gauntlet/io.py
from __future__ import annotations
import os
import re
import pandas as pd
from typing import Tuple, List

RUN_SUMMARY = "pareto_front_summary.csv"
RUN_LEDGER  = "pareto_front_trade_ledger.csv"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def find_latest_run_dir(base: str = "runs") -> str:
    if not os.path.isdir(base):
        return ""
    candidates = []
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            try:
                candidates.append((os.path.getmtime(p), p))
            except Exception:
                pass
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][1]

def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    # normalize common date columns if present
    for c in ("trigger_date", "exit_date", "first_trigger_date", "last_trigger_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def read_global_artifacts(run_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, ledger_df) from the run root."""
    summary = _read_csv_safe(os.path.join(run_dir, RUN_SUMMARY))
    ledger  = _read_csv_safe(os.path.join(run_dir, RUN_LEDGER))
    return summary, ledger

def list_folds(run_dir: str) -> List[int]:
    """Return fold numbers present under runs/<ts>/folds/fold_XX/ sorted ascending."""
    folds_dir = os.path.join(run_dir, "folds")
    if not os.path.isdir(folds_dir):
        return []
    out: List[int] = []
    pat = re.compile(r"^fold_(\d+)$")
    for name in os.listdir(folds_dir):
        m = pat.match(name)
        if not m:
            continue
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue
    out.sort()
    return out

def read_fold_artifacts(run_dir: str, fold_num: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read per-fold summary/ledger; if missing, fall back to slicing global artifacts by fold column.
    """
    fold_dir = os.path.join(run_dir, "folds", f"fold_{int(fold_num):02d}")
    sum_path = os.path.join(fold_dir, "pareto_summary.csv")
    led_path = os.path.join(fold_dir, "pareto_ledger.csv")
    sum_f = _read_csv_safe(sum_path)
    led_f = _read_csv_safe(led_path)
    if not sum_f.empty and not led_f.empty:
        return sum_f, led_f

    # Fallback to global slice
    gsum, gled = read_global_artifacts(run_dir)
    if not gsum.empty:
        sum_f = gsum[gsum.get("fold").astype(str) == str(fold_num)]
    if not gled.empty:
        led_f = gled[gled.get("fold").astype(str) == str(fold_num)]
    return sum_f.copy(), led_f.copy()
