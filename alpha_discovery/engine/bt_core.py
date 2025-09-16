# alpha_discovery/engine/bt_core.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import math
import numpy as np
import pandas as pd

from ..config import settings
from .bt_common import (
    TRADE_HORIZONS_DAYS,
    _add_bdays,
    _get_risk_free_rate,
    _has_iv_for_ticker,
    _build_option_strike,
    ExitPolicy,
    _decide_exit_from_path,
    _policy_id,
    _price_entry_exit_enhanced,
)

from .bt_runtime import (
    _maybe_reset_caches,
    _cached_underlier,
    _cached_iv3m,
    _cached_sigma_map,
    _cached_price_entry_exit,
    _get_or_build_price_path,
)


def _choose_tenor_bd(horizon_days: int) -> int:
    """Pick a tenor from options.tenor_grid_bd that is >= k * horizon_days; else longest."""
    grid = sorted(getattr(settings.options, "tenor_grid_bd", [63]))
    k = float(getattr(settings.options, "tenor_buffer_k", 1.25))
    need = int(max(1, round(horizon_days * k)))
    for t in grid:
        if t >= need:
            return int(t)
    return int(grid[-1])


def _slippage_for_tenor(tenor_bd: int) -> float:
    tiers = getattr(settings.options, "slippage_tiers", {"days_15": 0.0, "days_30": 0.0, "days_any": 0.0})
    if tenor_bd <= 15:
        return float(tiers.get("days_15", 0.0))
    if tenor_bd <= 30:
        return float(tiers.get("days_30", 0.0))
    return float(tiers.get("days_any", 0.0))


def run_setup_backtest_options(
        setup_signals: List[str],
        signals_df: pd.DataFrame,
        master_df: pd.DataFrame,
        direction: str,  # "long" or "short"
        exit_policy: dict | None = None,
        tickers_to_run: Optional[List[str]] = None,  # allow narrowing to a subset
) -> pd.DataFrame:
    """
    SIMPLIFIED OPTIONS BACKTESTING: Only regime-aware exits supported.

    This backtester is designed to handle open positions properly by:
    1. Only using regime-aware exit policies (all traditional exits removed)
    2. Setting exit_date to NaT when no exit condition is met
    3. Distinguishing between data end and actual option expiration
    4. Preventing phantom same-day trades through better validation
    """
    if not setup_signals:
        return pd.DataFrame()

    _maybe_reset_caches(master_df)

    # Rising-edge trigger from all selected signals with reset mechanism
    try:
        trigger_df = signals_df[setup_signals].astype("boolean").fillna(False)
        trigger_mask = trigger_df.all(axis=1)
    except KeyError as e:
        print(f" Error: Missing signals in signals_df: {e}")
        return pd.DataFrame()

    # Enhanced signal reset mechanism: force reset after signal has been true for too long
    signal_reset_days = getattr(settings.regime_aware, "signal_reset_days", 5)
    if signal_reset_days > 0:
        # Create a mask that resets signals after they've been true for too long
        signal_streak = trigger_mask.groupby((~trigger_mask).cumsum()).cumsum()
        reset_mask = signal_streak > signal_reset_days
        trigger_mask = trigger_mask & ~reset_mask

    prev = trigger_mask.shift(1, fill_value=False)
    edge = trigger_mask & ~prev
    trigger_dates = edge[edge].index

    if len(trigger_dates) < settings.validation.min_initial_support:
        return pd.DataFrame()

    min_premium = float(getattr(settings.options, "min_premium", 0.0))
    max_contracts = int(getattr(settings.options, "max_contracts", 10 ** 9))
    r_mode = getattr(settings.options, "risk_free_rate_mode", "constant")

    ticker_list = tickers_to_run if tickers_to_run is not None else settings.data.tradable_tickers

    rows: List[Dict[str, Any]] = []

    for trigger_date in trigger_dates:
        r = _get_risk_free_rate(trigger_date, df=master_df if r_mode == "macro" else None)

        for ticker in ticker_list:
            if not _has_iv_for_ticker(ticker, master_df):
                continue

            # Regular multi-day horizons
            for h in TRADE_HORIZONS_DAYS:
                tenor_bd = _choose_tenor_bd(h)

                S0 = _cached_underlier(master_df, ticker, trigger_date)
                if S0 is None:
                    continue

                # Calculate the actual option expiration date
                option_expiration_date = _add_bdays(trigger_date, tenor_bd)

                # For exit pricing, we need to check how much data we have available
                # Use the minimum of (horizon_date, last_available_data_date) for backtesting
                horizon_date = _add_bdays(trigger_date, h)
                last_data_date = master_df.index.max() if hasattr(master_df.index, 'max') else pd.Timestamp.now()

                # Use the earlier of horizon_date or last available data
                analysis_end_date = min(horizon_date, last_data_date)

                S_analysis_end = _cached_underlier(master_df, ticker, analysis_end_date)
                if S_analysis_end is None:
                    continue

                slip = _slippage_for_tenor(tenor_bd)

                # Use enhanced pricing system
                priced = _price_entry_exit_enhanced(
                    s0=S0, s1=S_analysis_end, t0_days=tenor_bd, h_days=h, r=r,
                    direction=direction, ticker=ticker,
                    entry_date=trigger_date, exit_date=analysis_end_date,
                    master_df=master_df, q=0.0
                )

                # Set variables needed for price path regardless of pricing method
                if priced is None:
                    # Fallback to traditional pricing
                    entry_iv3m = _cached_iv3m(master_df, ticker, trigger_date, direction, fallback=None)
                    if entry_iv3m is None:
                        continue
                    exit_iv3m = _cached_iv3m(master_df, ticker, analysis_end_date, direction, fallback=entry_iv3m)
                    if exit_iv3m is None:
                        continue

                    K = _build_option_strike(S0, direction)
                    T0_years = max(1, tenor_bd) / 252.0
                    entry_sigma_T0 = _cached_sigma_map(master_df, ticker, trigger_date, tenor_bd, entry_iv3m)
                    T1_bd = max(1, tenor_bd - h)
                    exit_sigma_T1 = _cached_sigma_map(master_df, ticker, analysis_end_date, T1_bd, exit_iv3m)
                    if entry_sigma_T0 is None or exit_sigma_T1 is None:
                        continue

                    priced = _cached_price_entry_exit(
                        S0=S0, S1=S_analysis_end, K=K, T0_years=T0_years, h_days=h, r=r,
                        direction="long" if direction == "long" else "short",
                        entry_sigma=entry_sigma_T0, exit_sigma=exit_sigma_T1, q=0.0
                    )
                    if priced is None:
                        continue
                else:
                    # Enhanced pricing succeeded - extract values
                    K = priced.get("K_over_S", 1.0) * S0
                    T0_years = max(1, tenor_bd) / 252.0

                    # For price path compatibility, get 3M IV values as fallback
                    entry_iv3m = _cached_iv3m(master_df, ticker, trigger_date, direction, fallback=None)
                    if entry_iv3m is None:
                        # If 3M IV not available, use enhanced pricing IV
                        entry_iv3m = priced.get("sigma_entry", priced["entry_iv"])

                    entry_sigma_T0 = priced.get("sigma_entry", priced["entry_iv"])

                entry_exec = float(priced["entry_price"]) * (1.0 + slip)
                if entry_exec < min_premium:
                    continue

                # Build price path for regime-aware exit analysis
                price_path = _get_or_build_price_path(
                    df=master_df,
                    ticker=ticker,
                    trigger_date=trigger_date,
                    horizon_days=int(h),
                    tenor_bd=int(tenor_bd),
                    direction=direction,
                    r=float(r),
                    S0=float(S0),
                    entry_iv3m=float(entry_iv3m),
                    entry_sigma_T0=float(entry_sigma_T0),
                    T0_years=float(T0_years),
                )

                # SIMPLIFIED: Only regime-aware exit policies
                pol = ExitPolicy(
                    enabled=getattr(settings.options, "exit_policies_enabled", True),
                    regime_aware=True  # Always use regime-aware exits
                )

                policy_str = _policy_id(pol, int(h))

                # ===================================
                # REGIME-AWARE EXIT DECISION (ONLY)
                # ===================================

                exit_result = _decide_exit_from_path(
                    price_path=price_path,
                    entry_exec_price=float(entry_exec),
                    horizon_days=int(h),
                    policy=pol,
                    direction=direction,
                    entry_date=trigger_date,
                    tenor_bd=tenor_bd
                )

                # Handle the exit result
                if exit_result is None:
                    # No exit condition met - position remains open
                    chosen_exit_date = pd.NaT  # This is the key fix for open positions!
                    exit_reason = "OPEN"
                    holding_days_actual = (analysis_end_date - trigger_date).days

                    # For open positions, use current market price for unrealized P&L calculation
                    exit_exec = float(price_path.iloc[-1]) * (1.0 - slip) if len(price_path) > 0 else entry_exec

                else:
                    # Exit condition was triggered
                    exit_idx, exit_reason, holding_days_actual = exit_result
                    chosen_exit_date = price_path.index[min(exit_idx, len(price_path) - 1)]
                    chosen_exit_mid = float(price_path.iloc[min(exit_idx, len(price_path) - 1)])
                    exit_exec = chosen_exit_mid * (1.0 - slip)

                # Position sizing and P&L calculation
                capital = float(getattr(settings.options, "capital_per_trade", 10000.0))
                contract_mult = int(getattr(settings.options, "contract_multiplier", 100))
                cost_per_contract = float(priced["entry_price"]) * contract_mult
                contracts = int(min(max_contracts, max(0, math.floor(capital / max(cost_per_contract, 1e-12)))))
                if contracts <= 0:
                    continue

                capital_used = float(contracts) * cost_per_contract
                pnl_dlr = (exit_exec - float(entry_exec)) * contracts * contract_mult
                pnl_pc = pnl_dlr / capital if capital > 0 else 0.0

                # Data quality check: prevent same-day entry/exit with identical underlying prices
                if (not pd.isna(chosen_exit_date) and
                        chosen_exit_date.normalize() == trigger_date.normalize() and
                        abs(S0 - _cached_underlier(master_df, ticker, chosen_exit_date)) < 1e-6):
                    # Skip this phantom trade
                    continue

                # Build the trade record
                trade_record = {
                    "trigger_date": pd.Timestamp(trigger_date),
                    "exit_date": chosen_exit_date,  # Will be NaT for open positions
                    "ticker": ticker,
                    "horizon_days": int(h),
                    "direction": direction,
                    "option_type": priced.get("option_type", "C" if direction == "long" else "P"),
                    "strike": float(K),
                    "entry_underlying": float(S0),
                    "contracts": int(contracts),
                    "capital_allocated": float(capital),
                    "capital_allocated_used": float(capital_used),
                    "exit_reason": exit_reason,
                    "exit_policy_id": policy_str,
                    "holding_days_actual": int(holding_days_actual),

                    # Enhanced IV tracking fields
                    "iv_anchor": str(priced.get("iv_anchor", "")),
                    "delta_bucket": str(priced.get("delta_bucket", "")),
                    "iv_ref_days": int(priced.get("iv_ref_days", 0)),
                    "sigma_anchor": float(priced.get("sigma_anchor", 0.0)),
                    "sigma_entry": float(priced.get("sigma_entry", 0.0)),
                    "delta_target": priced.get("delta_target"),  # Can be None
                    "delta_achieved": priced.get("delta_achieved"),  # Can be None
                    "K_over_S": float(priced.get("K_over_S", K / S0)),
                    "fallback_to_3M": bool(priced.get("fallback_to_3M", False)),
                }

                # Handle entry metrics (always present)
                trade_record.update({
                    "entry_iv": float(priced["entry_iv"]),
                    "entry_option_price": float(entry_exec),
                })

                # Handle exit metrics (only for closed positions)
                if pd.isna(chosen_exit_date):
                    # Open position - no exit metrics
                    trade_record.update({
                        "exit_underlying": float(S0),  # Use entry underlying for open positions
                        "exit_iv": np.nan,
                        "exit_option_price": np.nan,
                        "sigma_exit": np.nan,
                        "pnl_dollars": float(pnl_dlr),  # This is unrealized P&L
                        "pnl_pct": float(pnl_pc),
                    })
                else:
                    # Closed position - include exit metrics
                    trade_record.update({
                        "exit_underlying": float(_cached_underlier(master_df, ticker, chosen_exit_date)),
                        "exit_iv": float(priced.get("exit_iv", priced["entry_iv"])),
                        "exit_option_price": float(exit_exec),
                        "sigma_exit": float(priced.get("sigma_exit", 0.0)),
                        "pnl_dollars": float(pnl_dlr),  # This is realized P&L
                        "pnl_pct": float(pnl_pc),
                    })

                # Add P&L categorization
                if pd.isna(chosen_exit_date):
                    trade_record["unrealized_pnl"] = float(pnl_dlr)
                    trade_record["realized_pnl"] = 0.0
                else:
                    trade_record["unrealized_pnl"] = 0.0
                    trade_record["realized_pnl"] = float(pnl_dlr)

                rows.append(trade_record)

    if not rows:
        return pd.DataFrame()

    # Build the final ledger with data quality checks
    ledger = pd.DataFrame(rows)

    # Data quality: Remove exact duplicates based on key fields
    key_cols = ["trigger_date", "ticker", "direction", "horizon_days", "strike"]
    ledger = ledger.drop_duplicates(subset=key_cols, keep="first")

    # Sort by trigger date and ticker for consistent output
    ledger = ledger.sort_values(by=["trigger_date", "ticker", "horizon_days"]).reset_index(drop=True)

    #print(f"Generated {len(ledger)} trades, {ledger['exit_date'].isna().sum()} are open positions")

    return ledger