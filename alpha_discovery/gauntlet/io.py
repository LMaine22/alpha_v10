# alpha_discovery/gauntlet/io.py
from __future__ import annotations
import os
import re
import glob
import hashlib
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
    """Return (summary_df, ledger_df) from the pareto subdirectory."""
    pareto_dir = os.path.join(run_dir, "pareto")
    summary = _read_csv_safe(os.path.join(pareto_dir, RUN_SUMMARY))
    ledger  = _read_csv_safe(os.path.join(pareto_dir, RUN_LEDGER))
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


# ---------- OOS artifact readers + setup fingerprint (non-breaking additions) ----------

def _normalize_signal_ids(signal_ids_value):
    """Return a sorted list of signal ids from either a list or a delimited string."""
    if pd.isna(signal_ids_value):
        return []
    if isinstance(signal_ids_value, (list, tuple)):
        sigs = list(signal_ids_value)
    else:
        s = str(signal_ids_value)
        sep = '|' if '|' in s else ','
        sigs = [t.strip() for t in s.split(sep) if t.strip()]
    return sorted(set(sigs))


def _row_setup_signature(row: pd.Series) -> str:
    """
    Build a stable, human-readable signature for a setup using available columns.
    Priority: signal_ids + direction + specialized_ticker + params/description.
    Falls back to setup_id if needed (hashed later).
    """
    parts = []

    # 1) signals
    if 'signal_ids' in row and pd.notna(row['signal_ids']):
        sigs = _normalize_signal_ids(row['signal_ids'])
        if sigs:
            parts.append('signals=' + ';'.join(sigs))

    # 2) direction
    if 'direction' in row and pd.notna(row['direction']):
        parts.append(f"dir={row['direction']}")

    # 3) scope / specialization / ticker hint
    for scope_col in ('specialized_ticker', 'ticker_scope', 'ticker', 'best_performing_ticker'):
        if scope_col in row and pd.notna(row[scope_col]):
            parts.append(f"scope={row[scope_col]}")
            break

    # 4) params / description
    if 'param_json' in row and pd.notna(row['param_json']):
        parts.append(f"params={row['param_json']}")
    elif 'description' in row and pd.notna(row['description']):
        parts.append(f"desc={row['description']}")

    # Fallback: leave empty; caller may substitute setup_id
    return '|'.join(parts)


def _hash_setup_signature(sig: str) -> str:
    return hashlib.sha1(sig.encode('utf-8')).hexdigest() if sig else ''


def attach_setup_fp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-destructive: returns a copy with 'setup_fp' added.
    If a rich signature can't be built, falls back to hashing setup_id.
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    sig_series = out.apply(_row_setup_signature, axis=1)

    # If signature is empty, fall back to setup_id if available
    if 'setup_id' in out.columns:
        need_fallback = (sig_series.isna()) | (sig_series == '')
        sig_series = sig_series.mask(need_fallback, out['setup_id'].astype(str))

    out['setup_fp'] = sig_series.astype(str).map(_hash_setup_signature)
    return out


def read_oos_fold_ledgers(run_dir: str) -> List[pd.DataFrame]:
    """Return list of OOS fold ledgers for walk-forward stage."""
    folds_dir = os.path.join(run_dir, "oos_folds")
    if not os.path.isdir(folds_dir):
        return []
    pattern = os.path.join(folds_dir, "fold_*_oos", "oos_pareto_front_trade_ledger.csv")
    ledgers: List[pd.DataFrame] = []
    for path in sorted(glob.glob(pattern)):
        df = _read_csv_safe(path)
        if not df.empty:
            ledgers.append(df)
    return ledgers


def read_oos_artifacts(run_dir: str):
    """
    Load per-run OOS artifacts.

    Prefers combined files at the run root:
      - oos_pareto_front_summary_combined.csv
      - oos_pareto_front_trade_ledger_combined.csv

    If missing, falls back to concatenating any per-fold:
      - oos_folds/fold_*_oos/oos_pareto_front_summary.csv
      - oos_folds/fold_*_oos/oos_pareto_front_trade_ledger.csv

    Returns:
        (oos_summary_df, oos_ledger_df) with setup_fp attached to summary (and ledger if merge keys available).
    """
    combined_summary = os.path.join(run_dir, "oos_pareto_front_summary_combined.csv")
    combined_ledger  = os.path.join(run_dir, "oos_pareto_front_trade_ledger_combined.csv")

    summary_paths = []
    ledger_paths = []

    if os.path.exists(combined_summary):
        summary_paths.append(combined_summary)
    if os.path.exists(combined_ledger):
        ledger_paths.append(combined_ledger)

    # Per-fold fallback
    if not summary_paths:
        summary_paths = sorted(glob.glob(os.path.join(run_dir, "oos_folds", "fold_*_oos", "oos_pareto_front_summary.csv")))
    if not ledger_paths:
        ledger_paths = sorted(glob.glob(os.path.join(run_dir, "oos_folds", "fold_*_oos", "oos_pareto_front_trade_ledger.csv")))

    if not summary_paths or not ledger_paths:
        raise FileNotFoundError(f"[read_oos_artifacts] Could not find OOS files in: {run_dir}")

    read_kwargs = dict(low_memory=False)
    oos_summary_df = pd.concat((pd.read_csv(p, **read_kwargs) for p in summary_paths), ignore_index=True)
    oos_ledger_df  = pd.concat((pd.read_csv(p, **read_kwargs) for p in ledger_paths),  ignore_index=True)

    # Ensure setup_fp on summary
    oos_summary_df = attach_setup_fp(oos_summary_df)

    # Ensure setup_fp on ledger: try merge on shared keys; else compute row-wise
    if 'setup_fp' not in oos_ledger_df.columns:
        merge_keys = [c for c in ('setup_id', 'direction', 'specialized_ticker')
                      if c in oos_ledger_df.columns and c in oos_summary_df.columns]
        if merge_keys:
            oos_ledger_df = oos_ledger_df.merge(
                oos_summary_df[merge_keys + ['setup_fp']].drop_duplicates(),
                on=merge_keys,
                how='left'
            )
        else:
            oos_ledger_df = attach_setup_fp(oos_ledger_df)

    return oos_summary_df, oos_ledger_df


def read_is_artifacts(run_dir: str):
    """
    Load per-run IN-SAMPLE artifacts for diagnostic replay ONLY.

    Prefers combined files at the run root:
      - pareto_front_summary.csv
      - pareto_front_trade_ledger.csv

    If missing, falls back to per-fold:
      - fold_*_is/pareto_front_summary_*.csv
      - fold_*_is/pareto_front_trade_ledger_*.csv

    Returns:
        (is_summary_df, is_ledger_df) or (None, None) if nothing found.
    """
    combined_summary = os.path.join(run_dir, "pareto", "pareto_front_summary.csv")
    combined_ledger  = os.path.join(run_dir, "pareto", "pareto_front_trade_ledger.csv")

    summary_paths = []
    ledger_paths = []

    if os.path.exists(combined_summary):
        summary_paths.append(combined_summary)
    if os.path.exists(combined_ledger):
        ledger_paths.append(combined_ledger)

    # Per-fold fallback (common naming variants)
    if not summary_paths:
        summary_paths = sorted(glob.glob(os.path.join(run_dir, "fold_*_is", "pareto_front_summary_*.csv")))
        if not summary_paths:
            summary_paths = sorted(glob.glob(os.path.join(run_dir, "fold_*", "pareto_front_summary_*.csv")))
    if not ledger_paths:
        ledger_paths = sorted(glob.glob(os.path.join(run_dir, "fold_*_is", "pareto_front_trade_ledger_*.csv")))
        if not ledger_paths:
            ledger_paths = sorted(glob.glob(os.path.join(run_dir, "fold_*", "pareto_front_trade_ledger_*.csv")))

    if not summary_paths and not ledger_paths:
        return None, None

    read_kwargs = dict(low_memory=False)
    is_summary_df = pd.concat((pd.read_csv(p, **read_kwargs) for p in summary_paths), ignore_index=True) if summary_paths else None
    is_ledger_df  = pd.concat((pd.read_csv(p, **read_kwargs) for p in ledger_paths),  ignore_index=True) if ledger_paths  else None

    return is_summary_df, is_ledger_df


__all__ = ['attach_setup_fp', 'read_oos_fold_ledgers', 'read_oos_artifacts', 'read_is_artifacts']
