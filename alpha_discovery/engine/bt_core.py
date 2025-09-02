# alpha_discovery/engine/bt_core.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import math
import numpy as np
import pandas as pd

from ..config import settings
from .bt_common import (
    TRADE_HORIZONS_DAYS, _add_bdays,
    _get_risk_free_rate, _has_iv_for_ticker, _build_option_strike,
    ExitPolicy, _decide_exit_from_path, _policy_id
)
from .bt_runtime import (
    _maybe_reset_caches, _cached_underlier, _cached_iv3m,
    _cached_sigma_map, _cached_price_entry_exit, _get_or_build_price_path
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
    tickers_to_run: Optional[List[str]] = None # <<< NEW ARGUMENT
) -> pd.DataFrame:
    """
    Options ledger with optional dynamic exits.
    Can now run on all tradable tickers or a specified subset.
    """
    if not setup_signals:
        return pd.DataFrame()

    _maybe_reset_caches(master_df)

    try:
        trigger_mask = signals_df[setup_signals].all(axis=1)
    except KeyError as e:
        print(f" Error: Missing signals in signals_df: {e}")
        return pd.DataFrame()
    trigger_dates = trigger_mask[trigger_mask].index
    if len(trigger_dates) < settings.validation.min_initial_support:
        return pd.DataFrame()

    min_premium = float(getattr(settings.options, "min_premium", 0.0))
    max_contracts = int(getattr(settings.options, "max_contracts", 10**9))
    r_mode = getattr(settings.options, "risk_free_rate_mode", "constant")

    # --- MODIFIED PART ---
    # Use the provided ticker list, or default to all tradable tickers from settings
    ticker_list = tickers_to_run if tickers_to_run is not None else settings.data.tradable_tickers
    # --- END MODIFIED PART ---

    rows: List[Dict[str, Any]] = []

    for trigger_date in trigger_dates:
        r = _get_risk_free_rate(trigger_date, df=master_df if r_mode == "macro" else None)

        for ticker in ticker_list: # Loop over the specified ticker list
            if not _has_iv_for_ticker(ticker, master_df):
                continue

            for h in TRADE_HORIZONS_DAYS:
                tenor_bd = _choose_tenor_bd(h)

                S0 = _cached_underlier(master_df, ticker, trigger_date)
                if S0 is None:
                    continue
                entry_iv3m = _cached_iv3m(master_df, ticker, trigger_date, direction, fallback=None)
                if entry_iv3m is None:
                    continue

                K = _build_option_strike(S0, direction)
                T0_years = max(1, tenor_bd) / 252.0
                slip = _slippage_for_tenor(tenor_bd)

                horizon_date = _add_bdays(trigger_date, h)

                S_hz = _cached_underlier(master_df, ticker, horizon_date)
                if S_hz is None:
                    continue
                exit_iv3m_hz = _cached_iv3m(master_df, ticker, horizon_date, direction, fallback=entry_iv3m)
                if exit_iv3m_hz is None:
                    continue

                entry_sigma_T0 = _cached_sigma_map(master_df, ticker, trigger_date, tenor_bd, entry_iv3m)
                T1_bd_hz = max(1, tenor_bd - h)
                exit_sigma_T1_hz = _cached_sigma_map(master_df, ticker, horizon_date, T1_bd_hz, exit_iv3m_hz)
                if entry_sigma_T0 is None or exit_sigma_T1_hz is None:
                    continue

                priced = _cached_price_entry_exit(
                    S0=S0, S1=S_hz, K=K, T0_years=T0_years, h_days=h, r=r,
                    direction="long" if direction == "long" else "short",
                    entry_sigma=entry_sigma_T0, exit_sigma=exit_sigma_T1_hz, q=0.0
                )
                if priced is None:
                    continue

                entry_exec = float(priced["entry_price"]) * (1.0 + slip)
                if entry_exec < min_premium:
                    continue

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

                pol = ExitPolicy(
                    enabled=getattr(settings.options, "exit_policies_enabled", True),
                    pt_multiple=(exit_policy or {}).get("pt_multiple", getattr(settings.options, "exit_pt_multiple", None)) if isinstance(exit_policy, dict) else getattr(settings.options, "exit_pt_multiple", None),
                    trail_frac=(exit_policy or {}).get("trail_frac", getattr(settings.options, "exit_trail_frac", None)) if isinstance(exit_policy, dict) else getattr(settings.options, "exit_trail_frac", None),
                    sl_multiple=(exit_policy or {}).get("sl_multiple", getattr(settings.options, "exit_sl_multiple", None)) if isinstance(exit_policy, dict) else getattr(settings.options, "exit_sl_multiple", None),
                    time_cap_days=(exit_policy or {}).get("time_cap_days", getattr(settings.options, "exit_time_cap_days", None)) if isinstance(exit_policy, dict) else getattr(settings.options, "exit_time_cap_days", None),
                )

                exit_idx, exit_reason, holding_days_actual = _decide_exit_from_path(
                    price_path=price_path,
                    entry_exec_price=float(entry_exec),
                    horizon_days=int(h),
                    policy=pol
                )
                chosen_exit_date = price_path.index[min(exit_idx, len(price_path) - 1)]
                chosen_exit_mid = float(price_path.iloc[min(exit_idx, len(price_path) - 1)])

                exit_exec = chosen_exit_mid * (1.0 - slip)
                policy_str = _policy_id(pol, int(h))

                capital = float(getattr(settings.options, "capital_per_trade", 10000.0))
                contract_mult = int(getattr(settings.options, "contract_multiplier", 100))
                cost_per_contract = float(priced["entry_price"]) * contract_mult
                contracts = int(min(max_contracts, max(0, math.floor(capital / max(cost_per_contract, 1e-12)))))
                if contracts <= 0:
                    continue

                capital_used = float(contracts) * cost_per_contract
                pnl_dlr = (exit_exec - entry_exec) * contracts * contract_mult
                pnl_pc = pnl_dlr / capital if capital > 0 else 0.0

                rows.append({
                    "trigger_date": pd.Timestamp(trigger_date),
                    "exit_date": pd.Timestamp(chosen_exit_date),
                    "ticker": ticker,
                    "horizon_days": int(h),
                    "direction": direction,
                    "option_type": priced.get("option_type", "C" if direction == "long" else "P"),
                    "strike": float(K),
                    "entry_underlying": float(S0),
                    "exit_underlying": float(_cached_underlier(master_df, ticker, chosen_exit_date) or S0),
                    "entry_iv": float(priced["entry_iv"]),
                    "exit_iv": float(exit_sigma_T1_hz),
                    "entry_option_price": float(entry_exec),
                    "exit_option_price": float(exit_exec),
                    "contracts": int(contracts),
                    "capital_allocated": float(capital),
                    "capital_allocated_used": float(capital_used),
                    "pnl_dollars": float(pnl_dlr),
                    "pnl_pct": float(pnl_pc),
                    "exit_reason": exit_reason,
                    "exit_policy_id": policy_str,
                    "holding_days_actual": int(holding_days_actual),
                })

    if not rows:
        return pd.DataFrame()

    ledger = pd.DataFrame(rows).sort_values(by=["trigger_date", "ticker", "horizon_days"]).reset_index(drop=True)
    return ledger
