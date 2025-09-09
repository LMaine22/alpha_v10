import os
import pandas as pd
import numpy as np

from .io import read_oos_artifacts, read_is_artifacts, attach_setup_fp
from .run import _first_nonnull_col, _normalize_trade_date, _ensure_fold_column, _infer_knowledge_map_from_splits, _ensure_dir, _pick_ticker_col

def _load_strict_oos_survivors(run_dir: str) -> pd.DataFrame | None:
    """
    Load survivors from the Strict-OOS Gauntlet. Prefer gauntlet_ledger.csv;
    fall back to stage3_oos.csv; finally accept stage2_oos.csv.
    Returns a DataFrame with at least ['setup_fp','direction',ticker_col] if available.
    """
    base = os.path.join(run_dir, 'gauntlet', 'strict_oos')
    for name in ('gauntlet_ledger.csv', 'stage3_oos.csv', 'stage2_oos.csv'):
        p = os.path.join(base, name)
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, low_memory=False)
                if not df.empty:
                    return df
            except Exception:
                pass
    return None


def _prep_with_fp_and_dates(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure setup_fp, fold, and normalized trade dates exist on df.
    """
    out = df.copy()
    if 'setup_fp' not in out.columns:
        merge_keys = [k for k in ('setup_id', 'direction', 'specialized_ticker') if k in out.columns and k in (summary.columns if summary is not None else [])]
        if merge_keys and summary is not None:
            summary_fp = attach_setup_fp(summary)
            out = out.merge(
                summary_fp[merge_keys + ['setup_fp']].drop_duplicates(),
                on=merge_keys,
                how='left'
            )
        else:
            out = attach_setup_fp(out)
    if summary is not None:
        out = _ensure_fold_column(out, summary)
    out = _normalize_trade_date(out)
    return out


def build_diagnostic_replay(run_dir: str, splits=None, survivors_only: bool = True):
    """
    Construct a non-gating diagnostic replay by combining IS + OOS ledgers
    and tagging trades that occurred before the setup was 'known' (knowledge date).

    Steps:
      1) Load OOS (summary+ledger) and compute knowledge date per fold.
      2) Load IS (summary+ledger) if available.
      3) Attach setup_fp, fold, and normalized trade dates to both.
      4) Compute earliest knowledge per setup_fp from OOS:
           - prefer knowledge_d from splits;
           - else use fold order as proxy.
      5) Concatenate IS + OOS, dedupe on (setup_fp, ticker, direction, trade_d).
      6) Mark uses_pre_knowledge if trade_d < earliest knowledge_d for that setup_fp.
      7) Optionally restrict to survivors from Strict-OOS.
    """
    # 1) OOS
    oos_summary, oos_ledger = read_oos_artifacts(run_dir)
    oos_ledger = _prep_with_fp_and_dates(oos_ledger, oos_summary)

    # Try to map fold -> knowledge date
    knowledge_map = _infer_knowledge_map_from_splits(splits)
    if knowledge_map is not None:
        oos_ledger['knowledge_d'] = oos_ledger['fold'].map(knowledge_map)
    else:
        # If we cannot map real dates, create an artificial monotone ordering
        # per fold (1..K) and convert to a pseudo-date rank so we can choose earliest.
        oos_ledger['knowledge_rank'] = oos_ledger['fold'].astype(int)

    # Earliest knowledge per setup_fp
    if 'knowledge_d' in oos_ledger.columns and oos_ledger['knowledge_d'].notna().any():
        knowledge_by_setup = oos_ledger.groupby('setup_fp', dropna=False)['knowledge_d'].min()
    else:
        knowledge_by_setup = oos_ledger.groupby('setup_fp', dropna=False)['knowledge_rank'].min()

    # 2) IS (optional)
    is_summary, is_ledger = read_is_artifacts(run_dir)
    if is_ledger is not None and not is_ledger.empty:
        is_ledger = _prep_with_fp_and_dates(is_ledger, is_summary)
        frames = [is_ledger, oos_ledger]
    else:
        frames = [oos_ledger]

    full = pd.concat(frames, ignore_index=True)

    # 3) Dedupe across (setup_fp, ticker, direction, trade_d)
    ticker_col = _pick_ticker_col(full)
    key_cols = ['setup_fp', ticker_col, 'direction', 'trade_d']
    full = full.sort_values(key_cols).drop_duplicates(subset=key_cols, keep='first').reset_index(drop=True)

    # 4) Attach earliest knowledge marker and flag pre-knowledge trades
    if 'knowledge_d' in oos_ledger.columns and oos_ledger['knowledge_d'].notna().any():
        full['earliest_knowledge_d'] = full['setup_fp'].map(knowledge_by_setup)
        full['uses_pre_knowledge'] = (full['trade_d'] < full['earliest_knowledge_d'])
    else:
        full['earliest_knowledge_rank'] = full['setup_fp'].map(knowledge_by_setup)
        # Map fold to rank if missing (IS may not have folds; assume max rank so it flags True deterministically)
        if 'fold' in full.columns:
            full['fold_rank'] = full['fold'].astype('Int64')
        else:
            full['fold_rank'] = pd.Series([np.iinfo('int64').max] * len(full), index=full.index, dtype='Int64')
        full['uses_pre_knowledge'] = full['fold_rank'].astype('Int64') < full['earliest_knowledge_rank'].astype('Int64')

    # 5) Optionally restrict to survivors
    if survivors_only:
        surv = _load_strict_oos_survivors(run_dir)
        if surv is not None and not surv.empty:
            # Build survivor key on (setup_fp, ticker, direction)
            surv = surv.copy()
            if 'setup_fp' not in surv.columns:
                surv = _prep_with_fp_and_dates(surv, None)
            tcol_s = _pick_ticker_col(surv)
            surv_key = surv[['setup_fp', tcol_s, 'direction']].drop_duplicates()
            full_key = full[['setup_fp', ticker_col, 'direction']].merge(
                surv_key,
                left_on=['setup_fp', ticker_col, 'direction'],
                right_on=['setup_fp', tcol_s, 'direction'],
                how='inner'
            )
            key_tuples = set(map(tuple, full_key[['setup_fp', ticker_col, 'direction']].to_records(index=False)))
            full = full[[tuple(r) in key_tuples for r in full[['setup_fp', ticker_col, 'direction']].to_records(index=False)]].copy()

    return full


def summarize_diagnostic_replay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal non-gating summary: trades_count, pnl sums if present, and
    counts split by uses_pre_knowledge flag.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tcol = _pick_ticker_col(df)
    agg = {
        'trade_d': 'count',
        'uses_pre_knowledge': 'sum'
    }
    if 'pnl_dollars' in df.columns:
        agg['pnl_dollars'] = ['sum', 'mean']

    gb = df.groupby(['setup_fp', tcol, 'direction'], dropna=False).agg(agg)
    # flatten
    gb.columns = ['_'.join(c) if isinstance(c, tuple) else c for c in gb.columns]
    gb = gb.rename(columns={'trade_d_count': 'trades_count', 'uses_pre_knowledge_sum': 'pre_knowledge_trades'})
    gb = gb.reset_index()
    return gb
