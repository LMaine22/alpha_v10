# alpha_discovery/search/ga_core.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from collections import OrderedDict, Counter
import copy as _copy
import math
import zlib

import numpy as np
import pandas as pd

from ..config import settings
from ..engine import backtester
from ..engine import bt_common as _btc
from ..eval import selection, metrics

try:
    from threadpoolctl import threadpool_limits  # noqa: F401
except Exception:  # pragma: no cover
    from contextlib import nullcontext
    def threadpool_limits(*_args, **_kwargs):  # type: ignore
        return nullcontext()

VERBOSE: int = int(getattr(settings.ga, "verbose", 1))
DEBUG_SEQUENTIAL: bool = bool(getattr(settings.ga, "debug_sequential", False))
JOBLIB_VERBOSE: int = 0

# ──────────────────────────────────────────────────────────────────────────────
# Exit policy search space
# ──────────────────────────────────────────────────────────────────────────────
EXIT_PT_GRID = [1.6, 1.8, 2.0, 2.2]
EXIT_TS_GRID = [None, 0.90]
EXIT_SL_GRID = [None, 0.50, 0.60]
EXIT_PT_BEHAVIOR = ["arm_trail", "scale_out", "exit"]
EXIT_ARMED_TRAIL_GRID = [0.97, 0.98, 0.99]
EXIT_SCALE_OUT_FRAC_GRID = [0.33, 0.50, 0.67]

def _sample_exit_policy(rng: np.random.Generator) -> Dict:
    return {
        "pt_multiple":      float(rng.choice(EXIT_PT_GRID)),
        "trail_frac":       rng.choice(EXIT_TS_GRID),
        "sl_multiple":      rng.choice(EXIT_SL_GRID),
        "time_cap_days":    None,
        "pt_behavior":      rng.choice(EXIT_PT_BEHAVIOR),
        "armed_trail_frac": float(rng.choice(EXIT_ARMED_TRAIL_GRID)),
        "scale_out_frac":   float(rng.choice(EXIT_SCALE_OUT_FRAC_GRID)),
    }

def _mutate_exit_policy(pol: Dict, rng: np.random.Generator) -> Dict:
    if not pol: return _sample_exit_policy(rng)
    k = rng.choice([
        "pt_multiple", "armed_trail_frac",
        "trail_frac", "sl_multiple",
        "pt_behavior", "scale_out_frac",
        "time_cap_days",
    ])
    if k == "pt_multiple":
        pol[k] = float(rng.choice(EXIT_PT_GRID))
    elif k == "armed_trail_frac":
        pol[k] = float(rng.choice(EXIT_ARMED_TRAIL_GRID))
    elif k == "trail_frac":
        pol[k] = rng.choice(EXIT_TS_GRID)
    elif k == "sl_multiple":
        pol[k] = rng.choice(EXIT_SL_GRID)
    elif k == "pt_behavior":
        pol[k] = rng.choice(EXIT_PT_BEHAVIOR)
    elif k == "scale_out_frac":
        pol[k] = float(rng.choice(EXIT_SCALE_OUT_FRAC_GRID))
    else:
        pol[k] = None
    return pol

def _crossover_exit_policy(a: Dict, b: Dict, rng: np.random.Generator) -> Dict:
    if not a: return dict(b) if b else {}
    if not b: return dict(a)
    child = {}
    for k in [
        "pt_multiple", "trail_frac", "sl_multiple", "time_cap_days",
        "pt_behavior", "armed_trail_frac", "scale_out_frac",
    ]:
        child[k] = a.get(k) if rng.random() < 0.5 else b.get(k)
    return child

# ──────────────────────────────────────────────────────────────────────────────
# DNA / cache
# ──────────────────────────────────────────────────────────────────────────────
def _dna(individual: Tuple[str, List[str]]) -> Tuple[str, Tuple[str, ...]]:
    ticker, setup = individual
    return (ticker, tuple(sorted(setup)))

_EVAL_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_EVAL_CACHE_MAX = int(getattr(settings.ga, "eval_cache_max", 8192))

def _canon_exit_policy_for_cache(pol: Optional[Dict]):
    if pol is None: return None
    try: return tuple(sorted(pol.items()))
    except Exception: return tuple()

# ──────────────────────────────────────────────────────────────────────────────
# Regime labeling
# ──────────────────────────────────────────────────────────────────────────────
def _label_vol_regimes(daily_returns: pd.Series) -> pd.Series:
    if daily_returns is None or len(daily_returns) == 0:
        return pd.Series(dtype="object")
    dr = pd.Series(daily_returns).astype(float)
    roll_full = dr.rolling(window=21, min_periods=21).std()
    roll_nonnull = roll_full.dropna()
    if roll_nonnull.empty:
        return pd.Series("vol_mid", index=dr.index, dtype="object")
    q1, q2 = np.quantile(roll_nonnull.values, [0.33, 0.66])
    mask_low  = (roll_full <= q1).fillna(False)
    mask_high = (roll_full >  q2).fillna(False)
    reg = pd.Series("vol_mid", index=dr.index, dtype="object")
    reg.loc[mask_low]  = "vol_low"
    reg.loc[mask_high] = "vol_high"
    return reg

def _regime_sortino_map(daily_returns: pd.Series, ledger: pd.DataFrame) -> Dict[str, float]:
    if daily_returns is None or len(daily_returns) == 0:
        return {}
    reg_labels = _label_vol_regimes(daily_returns)
    out: Dict[str, float] = {}
    for name in ("vol_low", "vol_mid", "vol_high"):
        idx = reg_labels.index[reg_labels == name]
        if len(idx) == 0:
            out[name] = float("-inf"); continue
        sub_dr = pd.Series(daily_returns).reindex(idx).dropna()
        if sub_dr.empty:
            out[name] = float("-inf"); continue
        perf = metrics.calculate_portfolio_metrics(daily_returns=sub_dr, portfolio_ledger=ledger)
        out[name] = float(perf.get("sortino_lb", -99.0))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Multi-fidelity slicing
# ──────────────────────────────────────────────────────────────────────────────
def _mf_apply(signals_df: pd.DataFrame, master_df: pd.DataFrame, mf_mode: Tuple[bool,int,float]):
    enabled, stride, span_frac = mf_mode
    if not enabled: return signals_df, master_df
    n = len(master_df.index)
    cut = max(1, int(math.ceil(n * float(span_frac))))
    idx = master_df.index[:cut]
    sig_sub = signals_df.reindex(idx)
    m_sub = master_df.reindex(idx)
    if stride > 1:
        idx2 = idx[::stride]
        sig_sub = sig_sub.reindex(idx2)
        m_sub = m_sub.reindex(idx2)
    return sig_sub, m_sub

# ──────────────────────────────────────────────────────────────────────────────
# Shaping terms
# ──────────────────────────────────────────────────────────────────────────────
def _family_entropy(setup: List[str], signals_metadata: List[Dict]) -> float:
    fams = []
    md = {m.get("signal_id"): (m.get("family") or m.get("group") or "unknown") for m in signals_metadata}
    for sid in setup:
        fams.append(md.get(sid, "unknown"))
    if not fams: return 0.0
    vals = np.array(list({f: fams.count(f) for f in set(fams)}.values()), dtype=float)
    p = vals / vals.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = math.log(len(p)) if len(p) > 0 else 1.0
    return float(H / (Hmax + 1e-12))

def _complexity_penalty(setup: List[str]) -> float:
    k0 = int(getattr(getattr(settings.ga, "complexity", object()), "preferred", 6))
    w  = float(getattr(getattr(settings.ga, "complexity", object()), "lambda_", 0.03))
    over = max(0, len(setup) - k0)
    return w * float(over)

# ──────────────────────────────────────────────────────────────────────────────
# Safe horizon override
# ──────────────────────────────────────────────────────────────────────────────
class _HorizonPatch:
    def __init__(self, horizons: List[int]):
        self._new = list(horizons)
        self._old = None
    def __enter__(self):
        self._old = getattr(_btc, "TRADE_HORIZONS_DAYS", None)
        setattr(_btc, "TRADE_HORIZONS_DAYS", list(self._new))
    def __exit__(self, exc_type, exc, tb):
        if self._old is not None:
            setattr(_btc, "TRADE_HORIZONS_DAYS", self._old)

def _run_one_ledger_for_horizon(
    ticker: str, setup: List[str], signals_df: pd.DataFrame, master_df: pd.DataFrame,
    direction: str, exit_policy: Optional[Dict], horizon_days: int
) -> pd.DataFrame:
    with _HorizonPatch([int(horizon_days)]):
        return backtester.run_setup_backtest_options(
            setup_signals=setup,
            signals_df=signals_df,
            master_df=master_df,
            direction=direction,
            exit_policy=exit_policy if settings.options.exit_policies_enabled else None,
            tickers_to_run=[ticker],
        )

# ──────────────────────────────────────────────────────────────────────────────
# Permissive early-reject with coarse-gen bypass
# ──────────────────────────────────────────────────────────────────────────────
def _probe_bypass(payload_bytes: bytes, mf_mode: Tuple[bool,int,float]) -> bool:
    """
    Deterministic bypass that is more permissive when multi-fidelity is active.
    When coarse (span_frac<0.75 or stride>1): ≈ 50% pass-through.
    Otherwise: ≈ 6% pass-through.
    """
    h = zlib.adler32(payload_bytes)
    enabled, stride, span_frac = mf_mode
    if enabled and (span_frac < 0.75 or stride > 1):
        # use 1/2 pass rate
        return (h & 0x01) == 0
    # else ~1/16
    return (h & 0x0F) == 0

def _early_reject_threshold(len_frame: int, setup_len: int, mf_mode: Tuple[bool,int,float]) -> int:
    rf = getattr(settings, "risk_floors", None)
    min_abs = int(getattr(rf, "support_floor", 5))
    frac    = float(getattr(rf, "support_floor_fraction_of_test", 0.01))
    min_rel = int(math.ceil(frac * max(1, len_frame)))
    threshold = max(min_abs, min_rel)
    if setup_len > 1:
        threshold = max(2, int(round(threshold / (1.0 + 0.8 * (setup_len - 1)))))
    enabled, stride, span_frac = mf_mode
    if enabled and (span_frac < 0.75 or stride > 1):
        threshold = max(1, int(math.floor(threshold * 0.5)))
    return threshold

# ──────────────────────────────────────────────────────────────────────────────
# Single-setup evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _evaluate_one_setup(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
    mf_mode: Tuple[bool,int,float],
) -> Dict:
    ticker, setup = individual

    # 1) Apply multi-fidelity slicing FIRST
    s_df, m_df = _mf_apply(signals_df, master_df, mf_mode)

    # 2) Permissive early-reject on sliced frame
    try:
        # AND across selected signals
        trigger_cnt_and = int(s_df[setup].all(axis=1).sum())
    except Exception:
        trigger_cnt_and = 0

    # Also compute a quorum proxy (at least ceil(k*0.6) signals true)
    k = max(1, len(setup))
    quorum = max(1, int(math.ceil(0.6 * k)))
    try:
        trigger_cnt_quorum = int((s_df[setup].sum(axis=1) >= quorum).sum())
    except Exception:
        trigger_cnt_quorum = 0

    threshold = _early_reject_threshold(len(s_df), len(setup), mf_mode)

    payload = (f"{ticker}|" + "|".join(sorted(setup))).encode("utf-8", "ignore")
    bypass = _probe_bypass(payload, mf_mode)

    # If strict AND fails but quorum passes AND we are coarse, allow it through
    enabled, stride, span_frac = mf_mode
    allow_quorum = enabled and (span_frac < 0.75 or stride > 1)

    if not bypass:
        if trigger_cnt_and < threshold:
            if not (allow_quorum and trigger_cnt_quorum >= threshold):
                return {
                    "individual": individual, "metrics": {},
                    "objectives": [-99.0, -9999.0, 0.0, -99.0],
                    "rank": np.inf, "crowding_distance": 0.0,
                    "trade_ledger": pd.DataFrame(), "direction": "long",
                    "exit_policy": exit_policy, "regime_objectives": {},
                }

    # 3) crude direction heuristic
    direction_score = 0
    for sid in setup:
        meta = next((m for m in signals_metadata if m.get("signal_id") == sid), None)
        if meta and "<" in str(meta.get("condition", "")):
            direction_score -= 1
        else:
            direction_score += 1
    direction = "long" if direction_score >= 0 else "short"

    # 4) Horizon cases
    H_SET = list(getattr(settings.options, "trade_horizons_days", [5])) or [5]
    H_primary = 5 if 5 in H_SET else H_SET[int(len(H_SET)//2)]

    regime_case_scores: Dict[str, float] = {}
    robust_scores: List[float] = []

    ledger_primary = None
    for H in H_SET:
        ledger_H = _run_one_ledger_for_horizon(ticker, setup, s_df, m_df, direction, exit_policy, H)
        if ledger_H is None or getattr(ledger_H, "empty", True):
            for rn in ("vol_low", "vol_mid", "vol_high"):
                regime_case_scores[f"{rn}|H{H}"] = float("-inf")
            robust_scores.append(-99.0)
            if H == H_primary:
                ledger_primary = pd.DataFrame()
            continue

        dr_H = selection.portfolio_daily_returns(ledger_H)
        reg_map = _regime_sortino_map(dr_H, ledger_H)
        for rn, val in reg_map.items():
            regime_case_scores[f"{rn}|H{H}"] = float(val)
        robust_scores.append(float(min(reg_map.values()) if reg_map else -99.0))
        if H == H_primary:
            ledger_primary = ledger_H

    if ledger_primary is None:
        ledger_primary = pd.DataFrame()

    if ledger_primary.empty:
        perf = {}
        sortino_lb = -99.0
        expectancy = -9999.0
        support    = 0.0
    else:
        dr = selection.portfolio_daily_returns(ledger_primary)
        perf = metrics.calculate_portfolio_metrics(daily_returns=dr, portfolio_ledger=ledger_primary)
        sortino_lb = float(perf.get("sortino_lb", -99.0))
        expectancy = float(perf.get("expectancy", -9999.0))
        support    = float(perf.get("support", 0.0))

    sortino_adj = sortino_lb - _complexity_penalty(setup)
    w_ent = float(getattr(getattr(settings.ga, "family_entropy", object()), "weight", 0.0))
    if w_ent != 0.0:
        sortino_adj += w_ent * _family_entropy(setup, signals_metadata)

    robust_sortino = float(min(robust_scores)) if robust_scores else -99.0

    return {
        "individual": individual,
        "metrics": perf,
        "objectives": [sortino_adj, expectancy, support, robust_sortino],
        "rank": np.inf,
        "crowding_distance": 0.0,
        "trade_ledger": ledger_primary,
        "direction": direction,
        "exit_policy": exit_policy,
        "regime_objectives": regime_case_scores,
    }

def _evaluate_one_setup_cached(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
    mf_mode: Tuple[bool,int,float],
):
    key = (
        _dna(individual),
        id(signals_df), id(master_df),
        _canon_exit_policy_for_cache(exit_policy),
        ("mf",) + tuple(mf_mode),
        id(settings),
    )
    if key in _EVAL_CACHE:
        return _copy.deepcopy(_EVAL_CACHE[key])
    res = _evaluate_one_setup(individual, signals_df, signals_metadata, master_df, exit_policy, mf_mode)
    _EVAL_CACHE[key] = res
    if len(_EVAL_CACHE) > _EVAL_CACHE_MAX:
        _EVAL_CACHE.popitem(last=False)
    return res

def memetic_tune_exit_policy(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    base_policy: Optional[Dict],
    rng: np.random.Generator,
    mf_mode: Tuple[bool,int,float],
) -> Dict:
    enabled = bool(getattr(getattr(settings.ga, "memetic", object()), "enabled", True))
    if not enabled or not settings.options.exit_policies_enabled:
        return base_policy or _sample_exit_policy(rng)

    max_evals = int(getattr(getattr(settings.ga, "memetic", object()), "max_evals", 8))
    tol = float(getattr(getattr(settings.ga, "memetic", object()), "improve_tol", 0.004))

    def score(pol: Dict) -> float:
        out = _evaluate_one_setup_cached(individual, signals_df, signals_metadata, master_df, pol, mf_mode)
        return float(out.get("objectives", [-99.0])[0])

    best = dict(base_policy) if base_policy else _sample_exit_policy(rng)
    best_s = score(best)
    evals = 1

    def _adjacent(val, grid):
        if val not in grid: return grid
        i = grid.index(val); out = [val]
        if i > 0: out.append(grid[i-1])
        if i < len(grid)-1: out.append(grid[i+1])
        return list(dict.fromkeys(out))

    for pt in _adjacent(best.get("pt_multiple", EXIT_PT_GRID[0]), EXIT_PT_GRID):
        for tr in _adjacent(best.get("armed_trail_frac", EXIT_ARMED_TRAIL_GRID[0]), EXIT_ARMED_TRAIL_GRID):
            if evals >= max_evals: break
            cand = dict(best); cand["pt_multiple"] = float(pt); cand["armed_trail_frac"] = float(tr)
            s = score(cand); evals += 1
            if (best_s <= 0 and s > best_s) or (best_s > 0 and s >= best_s * (1.0 + tol)):
                best, best_s = cand, s

    return best

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
            sup = ind.get("metrics", {}).get("support")
            if sup is not None:
                try: support_vals.append(float(sup))
                except Exception: pass

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
