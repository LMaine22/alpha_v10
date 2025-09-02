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

try:
    from threadpoolctl import threadpool_limits
except Exception:
    from contextlib import nullcontext
    def threadpool_limits(*_args, **_kwargs):
        return nullcontext()

VERBOSE: int = int(getattr(settings.ga, "verbose", 1))
DEBUG_SEQUENTIAL: bool = bool(getattr(settings.ga, "debug_sequential", False))
JOBLIB_VERBOSE: int = 0

EXIT_PT_GRID = [None, 1.5, 2.0, 3.0]
EXIT_TS_GRID = [0.5, 0.33]
EXIT_SL_GRID = [0.5, 0.6]
EXIT_TC_TAGS = [None, "horizon"]

# --- MODIFIED: DNA is now (ticker, [signals]) ---
def _dna(individual: Tuple[str, List[str]]) -> Tuple[str, Tuple[str, ...]]:
    ticker, setup = individual
    return (ticker, tuple(sorted(setup)))

_EVAL_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_EVAL_CACHE_MAX = 2048

def _canon_exit_policy_for_cache(pol):
    if pol is None: return None
    try: return tuple(sorted(pol.items()))
    except Exception: return tuple()

def _evaluate_one_setup_cached(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
):
    """Cache wrapper for the new (ticker, setup) individual structure."""
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
    _EVAL_CACHE[key] = res
    if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
        _EVAL_CACHE.popitem(last=False)
    return res

# --- MODIFIED: Policy ops now handle the new individual structure ---
def _sample_exit_policy(rng: np.random.Generator) -> Dict:
    return {
        "pt_multiple": rng.choice(EXIT_PT_GRID),
        "trail_frac": rng.choice(EXIT_TS_GRID),
        "sl_multiple": rng.choice(EXIT_SL_GRID),
        "time_cap_days": None,
    }

def _mutate_exit_policy(pol: Dict, rng: np.random.Generator) -> Dict:
    if not pol: return _sample_exit_policy(rng)
    k = rng.choice(["pt_multiple", "trail_frac", "sl_multiple", "time_cap_days"])
    if k == "pt_multiple": pol[k] = rng.choice(EXIT_PT_GRID)
    elif k == "trail_frac": pol[k] = rng.choice(EXIT_TS_GRID)
    elif k == "sl_multiple": pol[k] = rng.choice(EXIT_SL_GRID)
    else: pol[k] = None
    return pol

def _crossover_exit_policy(a: Dict, b: Dict, rng: np.random.Generator) -> Dict:
    if not a: return dict(b) if b else {}
    if not b: return dict(a)
    child = {}
    for k in ["pt_multiple", "trail_frac", "sl_multiple", "time_cap_days"]:
        child[k] = a.get(k) if rng.random() < 0.5 else b.get(k)
    return child

# --- MODIFIED: One-setup evaluation is now specialized ---
def _infer_direction_from_metadata(setup: List[str], signals_metadata: List[Dict]) -> str:
    direction_score = 0
    for sid in setup:
        meta = next((m for m in signals_metadata if m.get("signal_id") == sid), None)
        if meta and "<" in str(meta.get("condition", "")):
            direction_score -= 1
        else:
            direction_score += 1
    return "long" if direction_score >= 0 else "short"

def _evaluate_one_setup(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
) -> Dict:
    """
    Evaluates a single (ticker, setup) individual.
    Backtest is run ONLY on the specialized ticker.
    Portfolio selection logic is bypassed.
    """
    ticker, setup = individual
    direction = _infer_direction_from_metadata(setup, signals_metadata)

    # Instruct the backtester to run on ONLY the specialized ticker
    ledger = backtester.run_setup_backtest_options(
        setup_signals=setup,
        signals_df=signals_df,
        master_df=master_df,
        direction=direction,
        exit_policy=exit_policy if settings.options.exit_policies_enabled else None,
        tickers_to_run=[ticker] # <<< THIS IS THE KEY CHANGE
    )

    # --- SIMPLIFIED EVALUATION ---
    # Since the ledger is for a single ticker, it IS the "portfolio ledger"
    # We no longer need the complex selection.assemble_portfolio_stepwise
    if ledger is None or ledger.empty:
        return {
            "individual": individual, "metrics": {},
            "objectives": [-99.0, -9999.0, 0.0],
            "rank": np.inf, "crowding_distance": 0.0,
            "trade_ledger": pd.DataFrame(), "direction": direction,
            "exit_policy": exit_policy,
        }

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

def _summarize_evals(tag: str, evaluated: List[Dict]) -> Dict[str, float]:
    total_ledgers = 0; total_trades = 0; support_vals = []
    exit_counter = Counter()
    for ind in evaluated:
        df = ind.get("trade_ledger")
        if isinstance(df, pd.DataFrame) and not df.empty:
            total_ledgers += 1
            total_trades += len(df)
            if "exit_reason" in df.columns:
                exit_counter.update(df["exit_reason"].value_counts().to_dict())
            if sup := ind.get("metrics", {}).get("support"):
                try: support_vals.append(float(sup))
                except Exception: pass
    avg_support = float(np.mean(support_vals)) if support_vals else 0.0
    if VERBOSE >= 2:
        from tqdm.auto import tqdm
        top_exit = ", ".join(f"{k}:{v}" for k, v in exit_counter.most_common(3))
        tqdm.write(f"[{tag}] Ledgers:{total_ledgers} Trades:{total_trades} "
                   f"AvgSupport:{avg_support:.1f} ExitReasons[{top_exit}]")
    return {"total_trades": float(total_trades), "avg_support": float(avg_support)}

