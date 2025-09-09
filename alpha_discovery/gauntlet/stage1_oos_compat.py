import os
import pandas as pd
from datetime import timedelta

def _pick_ticker_col(df: pd.DataFrame) -> str:
    for c in ('ticker', 'specialized_ticker', 'asset', 'symbol'):
        if c in df.columns:
            return c
    raise KeyError("No ticker-like column found among ['ticker','specialized_ticker','asset','symbol'].")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def run_stage1_oos_compat(
    ledger: pd.DataFrame,
    output_dir: str | None = None,
    as_of_date: str | None = None,
    recency_days: int = 14,
    min_trades: int = 5,
    **kwargs
):
    """
    OOS-aware Stage 1:
      - Groups by (setup_fp, ticker, direction)
      - Requires last trade date within 'recency_days' of as_of_date
      - Requires at least 'min_trades' total trades in the OOS ledger
    Returns: (filtered_ledger, summary_df)
    """
    if ledger is None or ledger.empty:
        return ledger, pd.DataFrame()

    df = ledger.copy()
    if 'trade_d' not in df.columns:
        # Shouldn't happen because Strict-OOS has trade_d, but fail safe:
        if 'trigger_date' in df.columns:
            df['trade_d'] = pd.to_datetime(df['trigger_date'], errors='coerce').dt.normalize()
        else:
            raise KeyError("Stage1 OOS compat requires a 'trade_d' or 'trigger_date' column.")

    df['trade_d'] = pd.to_datetime(df['trade_d'], errors='coerce').dt.normalize()
    asof = pd.to_datetime(as_of_date, errors='coerce').normalize() if as_of_date else df['trade_d'].max()

    tcol = _pick_ticker_col(df)

    gb = df.groupby(['setup_fp', tcol, 'direction'], dropna=False)['trade_d']
    last_trade = gb.max().rename('last_trade_d')
    count = gb.count().rename('trades_count')

    rollup = pd.concat([last_trade, count], axis=1).reset_index()

    # Apply gates
    recency_cut = asof - timedelta(days=int(recency_days)) if recency_days and recency_days > 0 else None
    if recency_cut is not None:
        rollup['pass_recency'] = rollup['last_trade_d'] >= recency_cut
    else:
        rollup['pass_recency'] = True

    rollup['pass_min_trades'] = rollup['trades_count'] >= int(min_trades)
    rollup['pass_stage1'] = rollup['pass_recency'] & rollup['pass_min_trades']

    # Survivors key
    survivors = rollup.loc[rollup['pass_stage1'], ['setup_fp', tcol, 'direction']]
    if not survivors.empty:
        filtered = df.merge(survivors, on=['setup_fp', tcol, 'direction'], how='inner')
    else:
        filtered = df.iloc[0:0].copy()

    # Save optional summary
    if output_dir:
        _ensure_dir(output_dir)
        rollup.to_csv(os.path.join(output_dir, 'stage1_oos_compat_summary.csv'), index=False)

    return filtered, rollup
