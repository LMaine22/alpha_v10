import os
import pandas as pd
import numpy as np
from datetime import timedelta

from .run import _first_nonnull_col, _pick_ticker_col, _ensure_dir

# ---- Helpers ----

def _pick_entry_col(df: pd.DataFrame):
    return _first_nonnull_col(df, ['entry_date', 'trigger_date', 'trade_date', 'date', 'trade_d'])

def _pick_exit_col(df: pd.DataFrame):
    return _first_nonnull_col(df, ['exit_date', 'close_date', 'exit_dt', 'close_dt'])

def _ensure_dates(df: pd.DataFrame):
    out = df.copy()
    ent = _pick_entry_col(out)
    if ent is None:
        raise KeyError("No entry-like column found among ['entry_date','trigger_date','trade_date','date','trade_d']")
    out['entry_dt'] = pd.to_datetime(out[ent], errors='coerce')
    if out['entry_dt'].isna().all():
        raise ValueError(f"Could not parse any entry dates from column '{ent}'")

    ex = _pick_exit_col(out)
    if ex:
        out['exit_dt'] = pd.to_datetime(out[ex], errors='coerce')
    else:
        # Single-day holding if no exit column exists
        out['exit_dt'] = out['entry_dt']

    # Normalize to date for concurrency counting
    out['entry_d'] = out['entry_dt'].dt.normalize()
    out['exit_d']  = out['exit_dt'].dt.normalize()

    # If exit < entry due to data oddities, clamp to entry
    bad = out['exit_d'] < out['entry_d']
    if bad.any():
        out.loc[bad, 'exit_d'] = out.loc[bad, 'entry_d']
        out.loc[bad, 'exit_dt'] = out.loc[bad, 'entry_dt']

    return out

def _require_pnl_column(df: pd.DataFrame):
    # Prefer explicit dollars; otherwise try common return columns
    if 'pnl_dollars' in df.columns:
        return 'pnl_dollars', 'dollars'
    for c in ('pnl', 'pnl_$', 'pnl_usd'):
        if c in df.columns:
            return c, 'dollars'
    for c in ('ret', 'return_pct', 'nav_total_return_pct'):
        if c in df.columns:
            return c, 'return_pct'
    # If none found, we proceed with zeros (still useful for concurrency stats)
    return None, None

# ---- Core simulation ----

def simulate_portfolio(
    df: pd.DataFrame,
    out_base: str,
    starting_capital: float = 100_000.0,
    position_size: float = 1_000.0,
    max_concurrent: int = 5,
    since_knowledge: bool = False
):
    """
    Non-gating portfolio diagnostic on the Diagnostic Replay ledger.
    - Concurrency cap: accept a trade only if current open positions < max_concurrent
    - Sizing: fixed nominal per-trade size (only used when we need to convert returns -> PnL)
    - PnL: if 'pnl_dollars' is available, we realize it on the exit date; else, if a return column exists,
           we compute pnl_dollars = position_size * return_per_trade; else pnl=0 with concurrency stats only.
    - Equity curve: jumps at exit dates (no mark-to-market path available)
    - since_knowledge: if True, drop rows where uses_pre_knowledge == True

    Outputs inside out_base:
      - portfolio/portfolio_positions.csv
      - portfolio/portfolio_daily_equity.csv
      - portfolio/portfolio_summary.csv
    """
    _ensure_dir(out_base)
    out_dir = _ensure_dir(os.path.join(out_base, 'portfolio'))

    if df is None or df.empty:
        # Write empty shells and return
        pd.DataFrame().to_csv(os.path.join(out_dir, 'portfolio_positions.csv'), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, 'portfolio_daily_equity.csv'), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, 'portfolio_summary.csv'), index=False)
        return {
            'positions_rows': 0,
            'equity_rows': 0,
            'skipped_due_to_cap': 0,
            'accepted': 0,
            'since_knowledge_applied': since_knowledge
        }

    work = df.copy()

    # live-only filter
    if since_knowledge and 'uses_pre_knowledge' in work.columns:
        work = work.loc[~work['uses_pre_knowledge'].astype(bool)].copy()

    # Ensure time columns
    work = _ensure_dates(work)

    # Sort by entry, stable within day
    sort_cols = ['entry_d', 'entry_dt']
    # Keep deterministic tie-breakers if present
    for c in ('setup_fp', _pick_ticker_col(work), 'direction'):
        if c in work.columns and c not in sort_cols:
            sort_cols.append(c)
    work = work.sort_values(sort_cols).reset_index(drop=True)

    # Choose pnl source
    pnl_col, pnl_mode = _require_pnl_column(work)

    # Walk forward day by day, enforcing concurrency cap by counting open intervals [entry_d, exit_d]
    start = work['entry_d'].min()
    end   = work['exit_d'].max()
    days = pd.date_range(start, end, freq='D')

    # Track open positions for overlap logic
    open_flags = np.zeros(len(work), dtype=bool)
    accepted = np.zeros(len(work), dtype=bool)
    skipped_due_to_cap = 0

    # Index trades by entry day
    by_entry = work.groupby('entry_d').indices  # dict: date -> index array

    # Maintain a set of currently open trade indices
    open_idx = set()

    for day in days:
        # Close positions whose exit < today (i.e., they were open until end of previous day)
        to_close = [i for i in list(open_idx) if work.loc[i, 'exit_d'] < day]
        for i in to_close:
            open_idx.remove(i)
            open_flags[i] = False

        # Consider new entries today (entry_d == day)
        if day in by_entry:
            cand_idx = list(by_entry[day])
            for i in cand_idx:
                # Before accepting, purge any positions that exit today but early in the day:
                # We treat exit at end-of-day; so they still count for concurrency on this day.
                if len(open_idx) < max_concurrent:
                    accepted[i] = True
                    open_idx.add(i)
                    open_flags[i] = True
                else:
                    skipped_due_to_cap += 1
                    accepted[i] = False

        # No need to mark still-open; we keep open until exit < current day

    work['accepted'] = accepted

    # Build daily equity: realize PnL on exit day for accepted trades
    equity = pd.DataFrame({'date': days})
    equity['realized_pnl'] = 0.0
    equity['open_positions'] = 0

    # Count open positions per day (inclusive of both ends)
    # A trade is considered open for a date 'd' if entry_d <= d <= exit_d
    for i, row in work.iterrows():
        if not row['accepted']:
            continue
        d0, d1 = row['entry_d'], row['exit_d']
        mask = (equity['date'] >= d0) & (equity['date'] <= d1)
        equity.loc[mask, 'open_positions'] += 1

    # Realize PnL on exit day
    if pnl_col is not None:
        if pnl_mode == 'dollars':
            # Sum PnL on exit date for accepted trades
            acc = work.loc[work['accepted']].groupby('exit_d')[pnl_col].sum()
            equity = equity.set_index('date')
            equity['realized_pnl'] = equity.index.map(acc).fillna(0.0)
            equity = equity.reset_index()
        else:
            # return_pct mode
            # Convert return to dollars using fixed position_size
            work['_pnl_dol_from_ret'] = 0.0
            mask = work['accepted'] & work[pnl_col].notna()
            work.loc[mask, '_pnl_dol_from_ret'] = position_size * work.loc[mask, pnl_col].astype(float)
            acc = work.loc[work['accepted']].groupby('exit_d')['_pnl_dol_from_ret'].sum()
            equity = equity.set_index('date')
            equity['realized_pnl'] = equity.index.map(acc).fillna(0.0)
            equity = equity.reset_index()
    # else: leave zeros

    # Build equity curve from realized PnL
    equity['cum_pnl'] = equity['realized_pnl'].cumsum()
    equity['equity']  = starting_capital + equity['cum_pnl']
    equity['daily_ret'] = equity['equity'].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Summary stats
    def _max_drawdown(series: pd.Series):
        peak = series.cummax()
        dd = (series - peak) / peak.replace(0, np.nan)
        return dd.min()

    total_accepted = int(work['accepted'].sum())
    total_skipped  = int(skipped_due_to_cap)
    wins = int((work.loc[work['accepted'], pnl_col] > 0).sum()) if pnl_col else 0
    losses = int((work.loc[work['accepted'], pnl_col] < 0).sum()) if pnl_col else 0
    win_rate = wins / max(1, wins + losses)

    mean_daily = equity['daily_ret'].mean()
    std_daily  = equity['daily_ret'].std(ddof=1)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily and std_daily > 0 else 0.0
    mdd = _max_drawdown(equity['equity'])

    summary = pd.DataFrame([{
        'starting_capital': starting_capital,
        'ending_capital': float(equity['equity'].iloc[-1]) if len(equity) else starting_capital,
        'total_realized_pnl': float(equity['cum_pnl'].iloc[-1]) if len(equity) else 0.0,
        'trades_total': int(len(work)),
        'trades_accepted': total_accepted,
        'trades_skipped_due_to_cap': total_skipped,
        'max_concurrent': max_concurrent,
        'position_size': position_size,
        'since_knowledge': since_knowledge,
        'win_rate': float(win_rate),
        'daily_sharpe': float(sharpe),
        'max_drawdown': float(mdd) if pd.notna(mdd) else np.nan,
        'dates_start': equity['date'].min() if len(equity) else pd.NaT,
        'dates_end': equity['date'].max() if len(equity) else pd.NaT,
    }])

    # Persist outputs
    positions_out = work.copy()
    positions_out.to_csv(os.path.join(out_dir, 'portfolio_positions.csv'), index=False)
    equity.to_csv(os.path.join(out_dir, 'portfolio_daily_equity.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'portfolio_summary.csv'), index=False)

    return {
        'positions_rows': int(len(positions_out)),
        'equity_rows': int(len(equity)),
        'skipped_due_to_cap': total_skipped,
        'accepted': total_accepted,
        'ending_capital': float(summary['ending_capital'].iloc[0]),
        'since_knowledge_applied': since_knowledge
    }
