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
)
from .bt_common import _decide_exit_with_scale_out

from .bt_runtime import (
    _maybe_reset_caches,
    _cached_underlier,
    _cached_iv3m,
    _cached_sigma_map,
    _cached_price_entry_exit,
    _get_or_build_price_path,
)

# --- NEW: Exit policy A (Timebox → Breakeven → Trailing) ---
def _decide_exit_timebox_be_trail(
    price_path: pd.Series,
    entry_exec_price: float,
    horizon_days: int,
    *,
    time_cap_days: int | None,
    sl_multiple: float | None,
    be_trigger_multiple: float = 1.25,   # +25% arms breakeven
    trail_arm_multiple: float = 1.40,    # +40% arms trailing
    trail_frac: float = 0.80             # 20% giveback once trailing is armed
) -> tuple[int, str, int]:
    """
    Returns (exit_idx, exit_reason, holding_days_actual).
    Assumes long option price series (direction & PnL sign handled elsewhere).
    """
    p = np.asarray(price_path, dtype=float)
    n = len(p)
    if n == 0:
        return 0, "horizon", 0

    # Cap the walk to time_cap_days if provided; otherwise use full horizon
    time_cap = min(horizon_days, time_cap_days if time_cap_days is not None else horizon_days)
    time_cap = max(0, int(time_cap))
    last_idx = min(time_cap, n - 1)

    entry = float(entry_exec_price)
    be_armed = False
    trail_armed = False
    peak = p[0] if n else entry

    has_trail = (trail_frac is not None) and (0.0 < float(trail_frac) < 1.0)

    for i in range(0, last_idx + 1):
        px = p[i]

        # 1) Hard stop vs entry (e.g., 0.65 => -35%)
        if sl_multiple is not None and sl_multiple > 0 and px <= entry * float(sl_multiple):
            return i, "stop_loss", i

        # 2) Arm breakeven once up +X%
        if (not be_armed) and px >= entry * float(be_trigger_multiple):
            be_armed = True

        # 3) If breakeven armed, protect entry
        if be_armed and px <= entry:
            return i, "breakeven", i

        # 4) Arm trailing once up +Y%
        if (not trail_armed) and px >= entry * float(trail_arm_multiple):
            trail_armed = True
            peak = px

        # 5) Trailing stop (keep ≥ trail_frac of max since arm)
        if trail_armed and has_trail:
            peak = max(peak, px)
            thresh = peak * float(trail_frac)
            if px <= thresh:
                return i, "trailing_stop", i

    # 6) Time stop or horizon
    if time_cap < horizon_days:
        return last_idx, "time_stop", last_idx
    return last_idx, "horizon", last_idx


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
    direction: str,                         # "long" or "short"
    exit_policy: dict | None = None,
    tickers_to_run: Optional[List[str]] = None,  # allow narrowing to a subset
) -> pd.DataFrame:
    """
    Options ledger with optional dynamic exits.
    Can run on all tradable tickers or a specified subset.
    """
    if not setup_signals:
        return pd.DataFrame()

    _maybe_reset_caches(master_df)

    # Rising-edge trigger from all selected signals
    try:
        trigger_df = signals_df[setup_signals].astype("boolean").fillna(False)
        trigger_mask = trigger_df.all(axis=1)
    except KeyError as e:
        print(f" Error: Missing signals in signals_df: {e}")
        return pd.DataFrame()

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
                    pt_multiple=(exit_policy or {}).get("pt_multiple", getattr(settings.options, "exit_pt_multiple", None))
                        if isinstance(exit_policy, dict) else getattr(settings.options, "exit_pt_multiple", None),
                    trail_frac=(exit_policy or {}).get("trail_frac", getattr(settings.options, "exit_trail_frac", None))
                        if isinstance(exit_policy, dict) else getattr(settings.options, "exit_trail_frac", None),
                    sl_multiple=(exit_policy or {}).get("sl_multiple", getattr(settings.options, "exit_sl_multiple", None))
                        if isinstance(exit_policy, dict) else getattr(settings.options, "exit_sl_multiple", None),
                    time_cap_days=(exit_policy or {}).get("time_cap_days", getattr(settings.options, "exit_time_cap_days", None))
                        if isinstance(exit_policy, dict) else getattr(settings.options, "exit_time_cap_days", None),
                    # behavior knobs
                    pt_behavior=(exit_policy or {}).get("pt_behavior", "regime_aware") if exit_policy is None else (exit_policy or {}).get("pt_behavior", getattr(settings.options, "pt_behavior", "regime_aware")),
                    armed_trail_frac=(exit_policy or {}).get("armed_trail_frac", getattr(settings.options, "armed_trail_frac", None)),
                    scale_out_frac=float((exit_policy or {}).get("scale_out_frac", getattr(settings.options, "scale_out_frac", 0.50))),
                    # regime-aware settings
                    regime_aware=(exit_policy or {}).get("regime_aware", True) if isinstance(exit_policy, dict) else True,
                )
                

                # --- Build a stable policy id string up front
                policy_str = _policy_id(pol, int(h))

                # =========================
                # SCALE-OUT path (existing)
                # =========================
                did_scale = False
                if getattr(pol, "pt_behavior", "exit") == "scale_out" and pol.pt_multiple:
                    # Clip by time cap / horizon
                    time_cap = min(int(h), int(pol.time_cap_days) if pol.time_cap_days is not None else int(h))
                    last_idx = min(time_cap, len(price_path) - 1)

                    # Prices up to cap
                    p = price_path.iloc[: last_idx + 1].astype(float).values

                    # PT threshold on option exec price path
                    pt_thresh = float(entry_exec) * float(pol.pt_multiple)
                    pt_hits = np.nonzero(p >= pt_thresh)[0]

                    if pt_hits.size > 0:
                        did_scale = True
                        pt_idx = int(pt_hits[0])

                        # Exec prices (apply slippage)
                        pt_mid = float(price_path.iloc[pt_idx])
                        exit_exec_pt = pt_mid * (1.0 - slip)

                        # After PT, tighten trailing (armed) and keep SL; same time cap remainder
                        armed_frac = pol.armed_trail_frac if (pol.armed_trail_frac is not None) else pol.trail_frac
                        p2 = p[pt_idx: last_idx + 1]  # from PT day to end
                        peak2 = np.maximum.accumulate(p2)

                        ts2_mask = np.zeros_like(p2, dtype=bool)
                        if armed_frac is not None and 0 < float(armed_frac) < 1:
                            ts2_thresh = peak2 * float(armed_frac)
                            ts2_mask = p2 <= ts2_thresh

                        sl2_mask = np.zeros_like(p2, dtype=bool)
                        if pol.sl_multiple is not None and pol.sl_multiple > 0:
                            sl_thresh = float(entry_exec) * float(pol.sl_multiple)
                            sl2_mask = p2 <= sl_thresh

                        def _first_true(mask: np.ndarray):
                            idxs = np.nonzero(mask)[0]
                            return int(idxs[0]) if idxs.size > 0 else None

                        day_ts2 = _first_true(ts2_mask)
                        day_sl2 = _first_true(sl2_mask)

                        candidates = []
                        if day_ts2 is not None:
                            candidates.append((day_ts2, "trailing_stop"))
                        if day_sl2 is not None:
                            candidates.append((day_sl2, "stop_loss"))

                        if candidates:
                            rel_idx, final_reason = min(candidates, key=lambda t: t[0])
                            final_idx = pt_idx + int(rel_idx)
                        else:
                            final_idx = last_idx
                            final_reason = "horizon" if time_cap == int(h) else "time_stop"

                        # Final exec at end
                        final_mid = float(price_path.iloc[final_idx])
                        exit_exec_final = final_mid * (1.0 - slip)

                        # Position sizing / contracts
                        capital = float(getattr(settings.options, "capital_per_trade", 10000.0))
                        contract_mult = int(getattr(settings.options, "contract_multiplier", 100))
                        cost_per_contract = float(priced["entry_price"]) * contract_mult
                        contracts = int(min(max_contracts, max(0, math.floor(capital / max(cost_per_contract, 1e-12)))))
                        if contracts > 0:
                            capital_used = float(contracts) * cost_per_contract

                            so_frac = max(0.0, min(1.0, float(getattr(pol, "scale_out_frac", 0.50))))
                            contracts_partial = int(round(contracts * so_frac))
                            contracts_remaining = int(max(0, contracts - contracts_partial))
                            if contracts_partial > 0 and contracts_remaining == 0:
                                contracts_remaining = 1
                                contracts_partial = max(0, contracts - 1)

                            pnl_dlr = (
                                (exit_exec_pt - float(entry_exec)) * contracts_partial * contract_mult +
                                (exit_exec_final - float(entry_exec)) * contracts_remaining * contract_mult
                            )
                            pnl_pc = pnl_dlr / capital if capital > 0 else 0.0

                            rows.append({
                                "trigger_date": pd.Timestamp(trigger_date),
                                "exit_date": pd.Timestamp(price_path.index[final_idx]),
                                "ticker": ticker,
                                "horizon_days": int(h),
                                "direction": direction,
                                "option_type": priced.get("option_type", "C" if direction == "long" else "P"),
                                "strike": float(K),
                                "entry_underlying": float(S0),
                                "exit_underlying": float(_cached_underlier(master_df, ticker, price_path.index[final_idx]) or S0),
                                "entry_iv": float(priced["entry_iv"]),
                                "exit_iv": float(exit_sigma_T1_hz),

                                # Prices (exec)
                                "entry_option_price": float(entry_exec),
                                "exit_option_price": float(exit_exec_final),

                                # Partial exit info
                                "partial_exit_date": pd.Timestamp(price_path.index[pt_idx]),
                                "partial_exit_price": float(exit_exec_pt),
                                "partial_exit_frac": float(so_frac),

                                # Position sizing
                                "contracts": int(contracts),
                                "contracts_partial": int(contracts_partial),
                                "contracts_remaining": int(contracts_remaining),
                                "capital_allocated": float(capital),
                                "capital_allocated_used": float(capital_used),

                                # PnL rollup
                                "pnl_dollars": float(pnl_dlr),
                                "pnl_pct": float(pnl_pc),

                                # Reasons & policy
                                "first_exit_reason": "profit_target_partial",
                                "exit_reason": final_reason,
                                "exit_policy_id": policy_str,
                                "holding_days_actual": int(final_idx),
                            })

                # ==============================
                # SINGLE-LEG path (default/NEW)
                # ==============================
                if not did_scale:
                    # --- NEW policy gate: Timebox → BE → Trail ---
                    policy_mode = (exit_policy or {}).get("pt_behavior", "regime_aware") if exit_policy is None else (exit_policy or {}).get("pt_behavior", getattr(settings.options, "pt_behavior", "exit"))
                    if str(policy_mode).lower() == "timebox_be_trail":
                        # Build safe parameter defaults for the new policy
                        be_mult = float((exit_policy or {}).get(
                            "be_trigger_multiple", getattr(settings.options, "be_trigger_multiple", 1.25)
                        ))
                        arm_mult = float((exit_policy or {}).get(
                            "trail_arm_multiple", getattr(settings.options, "trail_arm_multiple", 1.40)
                        ))
                        trail_frac_cfg = (exit_policy or {}).get("trail_frac",
                                                                 getattr(settings.options, "exit_trail_frac", None))
                        if trail_frac_cfg is None:
                            trail_frac_cfg = 0.80  # default 20% giveback once trailing is armed

                        exit_idx, exit_reason, holding_days_actual = _decide_exit_timebox_be_trail(
                            price_path=price_path,
                            entry_exec_price=float(entry_exec),
                            horizon_days=int(h),
                            time_cap_days=pol.time_cap_days,
                            sl_multiple=pol.sl_multiple,
                            be_trigger_multiple=be_mult,
                            trail_arm_multiple=arm_mult,
                            trail_frac=float(trail_frac_cfg),
                        )

                    else:
                        exit_idx, exit_reason, holding_days_actual = _decide_exit_from_path(
                            price_path=price_path,
                            entry_exec_price=float(entry_exec),
                            horizon_days=int(h),
                            policy=pol,
                            direction=direction
                        )

                    chosen_exit_date = price_path.index[min(exit_idx, len(price_path) - 1)]
                    chosen_exit_mid = float(price_path.iloc[min(exit_idx, len(price_path) - 1)])
                    exit_exec = chosen_exit_mid * (1.0 - slip)

                    capital = float(getattr(settings.options, "capital_per_trade", 10000.0))
                    contract_mult = int(getattr(settings.options, "contract_multiplier", 100))
                    cost_per_contract = float(priced["entry_price"]) * contract_mult
                    contracts = int(min(max_contracts, max(0, math.floor(capital / max(cost_per_contract, 1e-12)))))
                    if contracts <= 0:
                        continue

                    capital_used = float(contracts) * cost_per_contract
                    pnl_dlr = (exit_exec - float(entry_exec)) * contracts * contract_mult
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

            # Intraday patterns (if enabled)
            if getattr(settings.options, "enable_intraday_patterns", False):
                use_regular_horizons = getattr(settings.options, "intraday_use_regular_horizons", False)
                
                if use_regular_horizons:
                    # Test intraday patterns with all regular horizons
                    for pattern in getattr(settings.options, "intraday_patterns", []):
                        for h in TRADE_HORIZONS_DAYS:
                            _add_intraday_trade(rows, master_df, ticker, trigger_date, direction, pattern, h)
                else:
                    # Test intraday patterns with only 1-day horizon
                    for pattern in getattr(settings.options, "intraday_patterns", []):
                        _add_intraday_trade(rows, master_df, ticker, trigger_date, direction, pattern, 1)

    if not rows:
        return pd.DataFrame()

    ledger = pd.DataFrame(rows).sort_values(by=["trigger_date", "ticker", "horizon_days"]).reset_index(drop=True)
    return ledger


def _add_intraday_trade(rows: list, master_df: pd.DataFrame, ticker: str, trigger_date: pd.Timestamp, 
                       direction: str, pattern: str, horizon_days: int = 1) -> None:
    """Add intraday trading patterns to the ledger."""
    
    if pattern == 'overnight':
        # EOD entry → Next day open exit
        entry_date = trigger_date
        exit_date = _add_bdays(trigger_date, horizon_days)
        entry_time = 'close'
        exit_time = 'open'
        horizon_desc = 'overnight'
        
    elif pattern == 'intraday':
        # Open entry → Next day open exit  
        entry_date = trigger_date
        exit_date = _add_bdays(trigger_date, horizon_days)
        entry_time = 'open'
        exit_time = 'open'
        horizon_desc = 'intraday'
        
    else:
        return  # Unknown pattern
    
    # Get entry and exit underlying prices
    S0 = _cached_underlier(master_df, ticker, entry_date)
    S1 = _cached_underlier(master_df, ticker, exit_date)
    
    if S0 is None or S1 is None:
        return
        
    # Get IV data
    entry_iv3m = _cached_iv3m(master_df, ticker, entry_date, direction, fallback=None)
    exit_iv3m = _cached_iv3m(master_df, ticker, exit_date, direction, fallback=entry_iv3m)
    
    if entry_iv3m is None or exit_iv3m is None:
        return
    
    # Use 1-day tenor for intraday trades
    tenor_bd = 1
    K = _build_option_strike(S0, direction)
    T0_years = tenor_bd / 252.0
    slip = _slippage_for_tenor(tenor_bd)
    
    # Map sigma from 3M to 1-day
    entry_sigma_T0 = _cached_sigma_map(master_df, ticker, entry_date, tenor_bd, entry_iv3m)
    exit_sigma_T1 = _cached_sigma_map(master_df, ticker, exit_date, tenor_bd, exit_iv3m)
    
    if entry_sigma_T0 is None or exit_sigma_T1 is None:
        return
    
    # Price the option
    priced = _cached_price_entry_exit(
        S0=S0, S1=S1, K=K, T0_years=T0_years, h_days=1, r=0.0,
        direction="long" if direction == "long" else "short",
        entry_sigma=entry_sigma_T0, exit_sigma=exit_sigma_T1, q=0.0
    )
    
    if priced is None:
        return
    
    # Calculate costs and P&L
    entry_exec = float(priced["entry_price"]) * (1.0 + slip)
    exit_exec = float(priced["exit_price"]) * (1.0 - slip)
    
    if entry_exec < getattr(settings.options, "min_premium", 0.30):
        return
    
    # Simple position sizing
    capital = getattr(settings.options, "capital_per_trade", 10000.0)
    contract_mult = getattr(settings.options, "contract_multiplier", 100)
    contracts = int(capital / (entry_exec * contract_mult))
    
    if contracts <= 0:
        return
    
    cost_per_contract = entry_exec * contract_mult
    capital_used = float(contracts) * cost_per_contract
    pnl_dlr = (exit_exec - float(entry_exec)) * contracts * contract_mult
    pnl_pc = pnl_dlr / capital if capital > 0 else 0.0
    
    # Add to ledger
    rows.append({
        "trigger_date": pd.Timestamp(trigger_date),
        "exit_date": pd.Timestamp(exit_date),
        "ticker": ticker,
        "horizon_days": horizon_days,  # Use the specified horizon
        "direction": direction,
        "option_type": priced.get("option_type", "C" if direction == "long" else "P"),
        "strike": float(K),
        "entry_underlying": float(S0),
        "exit_underlying": float(S1),
        "entry_iv": float(priced["entry_iv"]),
        "exit_iv": float(priced["exit_iv"]),
        "entry_option_price": float(priced["entry_price"]),
        "exit_option_price": float(priced["exit_price"]),
        "contracts": contracts,
        "capital_allocated": capital,
        "capital_allocated_used": capital_used,
        "pnl_dollars": pnl_dlr,
        "pnl_pct": pnl_pc,
        "exit_reason": f"{pattern}_pattern",  # Special exit reason for intraday patterns
        "exit_policy_id": f"intraday_{pattern}",
        "holding_days_actual": 1,
    })
