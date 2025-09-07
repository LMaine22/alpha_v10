# alpha_discovery/search/ga_core.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from collections import OrderedDict, Counter
import copy as _copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import settings
from ..engine import backtester
from ..eval import selection, metrics
from ..engine.bt_runtime import _enforce_exclusivity_by_setup, _parse_bt_env_flag

try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import nullcontext
    def threadpool_limits(*_args, **_kwargs):
        return nullcontext()

VERBOSE: int = int(getattr(settings.ga, "verbose", 1))
DEBUG_SEQUENTIAL: bool = bool(getattr(settings.ga, "debug_sequential", False))
JOBLIB_VERBOSE: int = 0

# ---------------------------------------------------------------------
# Exit policy: read directly from config (NO GA grids / mutation here).
# ---------------------------------------------------------------------
def _exit_policy_from_settings() -> Optional[Dict]:
    """Build exit policy with regime-aware options enabled."""
    if not getattr(settings.options, "exit_policies_enabled", True):
        return None

    # Enable regime-aware exits
    pol: Dict = {
        "pt_behavior": "regime_aware",
        "regime_aware": True,
        "enabled": True,
    }

    return pol

# --- DNA & cache --------------------------------------------------------------

def _dna(individual: Tuple[str, List[str]]) -> Tuple[str, Tuple[str, ...]]:
    """Canonical identity for (ticker, setup)."""
    ticker, setup = individual
    return (ticker, tuple(sorted(setup or [])))

_EVAL_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_EVAL_CACHE_MAX = 4096

def _canon_exit_policy_for_cache(pol: Optional[Dict]) -> Optional[Tuple[Tuple[str, object], ...]]:
    if pol is None:
        return None
    try:
        return tuple(sorted(pol.items()))
    except Exception:
        return tuple()

def _evaluate_one_setup_cached(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
):
    """Cache wrapper for (ticker, setup) + data + exit_policy."""
    key = (
        _dna(individual),
        id(signals_df),
        id(signals_metadata),
        id(master_df),
        _canon_exit_policy_for_cache(exit_policy),
        id(settings),
    )
    if key in _EVAL_CACHE:
        return _copy.deepcopy(_EVAL_CACHE[key])
    res = _evaluate_one_setup(individual, signals_df, signals_metadata, master_df, exit_policy)
    _EVAL_CACHE[key] = _copy.deepcopy(res)
    if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
        _EVAL_CACHE.popitem(last=False)
    return res

# --- Helpers ------------------------------------------------------------------

def _infer_direction_from_metadata(setup: List[str], signals_metadata: List[Dict]) -> str:
    """Crude heuristic: count '<' as bearish; otherwise bullish; ties -> long."""
    direction_score = 0
    for sid in setup:
        meta = next((m for m in signals_metadata if m.get("signal_id") == sid), None)
        if meta and "<" in str(meta.get("condition", "")):
            direction_score -= 1
        else:
            direction_score += 1
    return "long" if direction_score >= 0 else "short"

# --- Evaluation ---------------------------------------------------------------

def _evaluate_one_setup(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
) -> Dict:
    """
    Evaluate a (ticker, setup) by running an options backtest on ONLY its ticker
    and returning portfolio-level metrics from that ledger.
    """
    ticker, setup = individual
    if not setup:
        return {
            "individual": individual, "metrics": {},
            "objectives": [-99.0, -9999.0, 0.0],
            "rank": np.inf, "crowding_distance": 0.0,
            "trade_ledger": pd.DataFrame(), "direction": "long", "exit_policy": exit_policy,
        }

    direction = _infer_direction_from_metadata(setup, signals_metadata)

    # Backtest only the specialized ticker
    ledger = backtester.run_setup_backtest_options(
        setup_signals=setup,
        signals_df=signals_df,
        master_df=master_df,
        direction=direction,
        exit_policy=exit_policy,
        tickers_to_run=[ticker],  # IMPORTANT: restrict to that ticker
    )

    if ledger is None or ledger.empty:
        return {
            "individual": individual, "metrics": {},
            "objectives": [-99.0, -9999.0, 0.0],
            "rank": np.inf, "crowding_distance": 0.0,
            "trade_ledger": pd.DataFrame(), "direction": direction, "exit_policy": exit_policy,
        }

    # TRAIN-side exclusivity per setup
    try:
        setup_id = f"{ticker}__" + "|".join(sorted(setup))
    except Exception:
        setup_id = str(individual)

    ledger = ledger.copy()
    ledger["setup_id"] = setup_id
    if _parse_bt_env_flag("BT_ENFORCE_EXCLUSIVITY", True):
        ledger = _enforce_exclusivity_by_setup(ledger)

    # Portfolio metrics
    daily_returns = selection.portfolio_daily_returns(ledger)
    perf = metrics.calculate_portfolio_metrics(
        daily_returns=daily_returns,
        portfolio_ledger=ledger
    )

    return {
        "individual": individual,
        "metrics": perf,
        "objectives": [
            perf.get("sortino_lb", -99.0),
            perf.get("expectancy", -9999.0),
            perf.get("support", 0.0),
        ],
        "rank": np.inf,
        "crowding_distance": 0.0,
        "trade_ledger": ledger,
        "direction": direction,
        "exit_policy": exit_policy,
    }

# --- Logging helper -----------------------------------------------------------

def _summarize_evals(tag: str, evaluated: List[Dict]) -> Dict[str, float]:
    total_ledgers = 0
    total_trades = 0
    support_vals = []
    exit_counter = Counter()
    first_exit_counter = Counter()

    for ind in evaluated:
        df = ind.get("trade_ledger")
        if isinstance(df, pd.DataFrame) and not df.empty:
            total_ledgers += 1
            total_trades += len(df)

            if "exit_reason" in df.columns:
                exit_counter.update(df["exit_reason"].value_counts().to_dict())
            if "first_exit_reason" in df.columns:
                first_exit_counter.update(df["first_exit_reason"].value_counts().to_dict())

            if sup := ind.get("metrics", {}).get("support"):
                try:
                    support_vals.append(float(sup))
                except Exception:
                    pass

    avg_support = float(np.mean(support_vals)) if support_vals else 0.0

    if VERBOSE >= 2:
        from tqdm.auto import tqdm
        top_exit = ", ".join(f"{k}:{v}" for k, v in exit_counter.most_common(3))
        tqdm.write(f"[{tag}] Ledgers:{total_ledgers} Trades:{total_trades} "
                   f"AvgSupport:{avg_support:.1f} ExitReasons[{top_exit}]")
        pt_partials = first_exit_counter.get("profit_target_partial", 0)
        if pt_partials:
            tqdm.write(f"[{tag}] ScaleOuts[profit_target_partial:{pt_partials}]")

    return {"total_trades": float(total_trades), "avg_support": float(avg_support)}
