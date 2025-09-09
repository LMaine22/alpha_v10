#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ENHANCED PREDICTION EVALUATION PIPELINE (single-file, with aggregator)
- Loads your data, builds features & primitive signals (imports from alpha_discovery/*)
- Creates walk-forward splits with an embargo
- Evolves predictive setups per fold using a simple NSGA-II style algorithm
- Objectives: maximize OOS directional accuracy, maximize OOS t-stat/IR,
              maximize IS support (stability), and maximize population diversity
- Writes per-fold CSVs AND aggregated CSVs at the end.

This file does NOT modify your existing backtesting/gauntlet modules.
"""

from __future__ import annotations

import os
import sys
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Limit BLAS threads to avoid CPU thrash
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.getenv("TMPDIR", "/tmp"))

# ---- Import from your repo (read-only) ----
from alpha_discovery.config import settings as core_settings
from alpha_discovery.data.loader import load_data_from_parquet
from alpha_discovery.features.registry import build_feature_matrix
from alpha_discovery.signals.compiler import compile_signals
from alpha_discovery.search import population as pop  # reuse ticker/signal gene ops


# =========================
# Config & CLI
# =========================

@dataclass
class PredictRunConfig:
    population_size: int = 150
    generations: int = 15
    seed: int = 42
    n_jobs: int = -1  # -1 => all cores; 1 => single-threaded

    # Genetic knobs
    elitism_rate: float = 0.15
    crossover_rate: float = 0.9
    base_mutation_rate: float = 0.35
    max_mutation_rate: float = 0.65
    stagnation_kick_after: int = 5   # gens w/o improvement before boosting mutation
    hard_kick_after: int = 9         # gens w/o improvement before extra mutation pass
    horizon_mutation_prob: float = 0.25

    # Predictive horizons to consider (trading days)
    horizons: List[int] = (1, 2, 3, 5, 10)

    # Setup size (keep small for support)
    min_signals_per_setup: int = 1
    max_signals_per_setup: int = 2

    # Support thresholds (avoid tiny-sample illusions)
    min_is_support: int = 15
    min_oos_support: int = 10

    # Walk-forward split knobs
    train_years: int = 3
    test_years: float = 1.0
    split_step_months: int = 6  # new split every 6 months
    use_back_aligned_final: bool = True

    # Output
    out_dir: str = "runs_predict_onefile"

    # Misc
    debug_sequential: bool = False  # if True, no parallel for easier debugging


def get_cfg_from_cli() -> PredictRunConfig:
    p = argparse.ArgumentParser(description="Enhanced Prediction Evaluation Pipeline (single-file).")
    p.add_argument("--pop", type=int, default=150, help="Population size")
    p.add_argument("--gens", type=int, default=15, help="Number of generations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n_jobs", type=int, default=-1, help="-1=all cores, 1=single-threaded")
    p.add_argument("--out", type=str, default="runs_predict_onefile", help="Output directory")
    p.add_argument("--min_is", type=int, default=15, help="Minimum IS support required")
    p.add_argument("--min_oos", type=int, default=10, help="Minimum OOS support required")
    args = p.parse_args()

    cfg = PredictRunConfig(
        population_size=args.pop,
        generations=args.gens,
        seed=args.seed,
        n_jobs=args.n_jobs,
        out_dir=args.out,
        min_is_support=args.min_is,
        min_oos_support=args.min_oos,
    )
    return cfg


# =========================
# Utils
# =========================

def _safe_n_jobs(n):
    """Joblib doesn't accept 0; coerce 0/None -> 1."""
    try:
        n = int(n)
    except Exception:
        return 1
    return 1 if n == 0 else n


def _pick(rng: np.random.Generator, seq):
    """Randomly pick one element from any sequence via integer indexing (avoids ragged rng.choice issues)."""
    return seq[int(rng.integers(0, len(seq)))]


def forward_log_return(price: pd.Series, horizon: int) -> pd.Series:
    p = pd.to_numeric(price, errors="coerce")
    return np.log(p.shift(-horizon) / p)


def tstat(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.size < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0:
        return 0.0
    return float(np.sqrt(x.size) * mu / sd)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================
# Diversity, NSGA helpers
# =========================

def individual_signature(ind: Dict) -> Tuple[str, Tuple[str, ...], int]:
    """Signature for diversity & uniqueness: (ticker, signals tuple, horizon)."""
    t = ind["ticker"]
    sigs = tuple(sorted(map(str, ind["setup_signals"])))
    h = int(ind["horizon"])
    return (t, sigs, h)


def jaccard(a: Tuple[str, Tuple[str, ...], int], b: Tuple[str, Tuple[str, ...], int]) -> float:
    _, sigs_a, h_a = a
    _, sigs_b, h_b = b
    set_a, set_b = set(sigs_a), set(sigs_b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b) if set_a or set_b else 1
    sig_div = 1.0 - inter / union
    h_div = 0.0 if h_a == h_b else 1.0
    return 0.5 * sig_div + 0.5 * h_div


def calculate_population_diversity(population: List[Dict]) -> None:
    """Assign a diversity score to each individual's objectives[3]."""
    if not population:
        return
    sigs = [individual_signature(ind) for ind in population]
    n = len(sigs)
    if n == 1:
        population[0]["objectives"][3] = 0.1
        return
    k = min(5, n - 1)
    dmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard(sigs[i], sigs[j])
            dmat[i, j] = d
            dmat[j, i] = d
    for i in range(n):
        nn = np.sort(dmat[i, :])[1: k + 1]
        div = float(nn.mean()) if nn.size > 0 else 0.1
        population[i]["objectives"][3] = max(0.1, min(1.0, div))


def dominates(a_obj: List[float], b_obj: List[float]) -> bool:
    return all(x >= y for x, y in zip(a_obj, b_obj)) and any(x > y for x, y in zip(a_obj, b_obj))


def non_dominated_sort(population: List[Dict]) -> List[List[Dict]]:
    if not population:
        return []
    for ind in population:
        ind["rank"] = None

    S = {i: [] for i in range(len(population))}
    n = [0] * len(population)
    fronts: List[List[int]] = [[]]

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if dominates(p["objectives"], q["objectives"]):
                S[i].append(j)
            elif dominates(q["objectives"], p["objectives"]):
                n[i] += 1
        if n[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front: List[int] = []
        for i in fronts[k]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    object_fronts: List[List[Dict]] = []
    for r, front in enumerate(fronts):
        if not front:
            continue
        obj_front = []
        for idx in front:
            population[idx]["rank"] = r + 1
            obj_front.append(population[idx])
        object_fronts.append(obj_front)
    return object_fronts


def calculate_crowding_distance(front: List[Dict]) -> None:
    if not front:
        return
    m = len(front[0]["objectives"])
    for ind in front:
        ind["crowding_distance"] = 0.0
    for k in range(m):
        front.sort(key=lambda ind: ind["objectives"][k])
        front[0]["crowding_distance"] = front[-1]["crowding_distance"] = float("inf")
        vals = [ind["objectives"][k] for ind in front]
        vmin, vmax = vals[0], vals[-1]
        if vmax == vmin:
            continue
        for i in range(1, len(front) - 1):
            prev_v = front[i - 1]["objectives"][k]
            next_v = front[i + 1]["objectives"][k]
            front[i]["crowding_distance"] += (next_v - prev_v) / (vmax - vmin)


def select_next_generation(population: List[Dict], pop_size: int) -> List[Dict]:
    """Assumes 'rank' is present (hence must be called AFTER non_dominated_sort on same pool)."""
    fronts = non_dominated_sort(population)
    next_pop: List[Dict] = []
    for front in fronts:
        calculate_crowding_distance(front)
        if len(next_pop) + len(front) <= pop_size:
            next_pop.extend(front)
        else:
            front.sort(key=lambda ind: ind["crowding_distance"], reverse=True)
            next_pop.extend(front[: (pop_size - len(next_pop))])
            break
    return next_pop


# =========================
# Population encoding
# =========================

def _random_setup(rng: np.random.Generator, all_signal_ids: List[str], k_min: int, k_max: int) -> List[str]:
    k = int(rng.integers(k_min, k_max + 1))
    return list(map(str, rng.choice(all_signal_ids, size=k, replace=False)))


def init_population(
    rng: np.random.Generator,
    all_signal_ids: List[str],
    cfg: PredictRunConfig
) -> List[Tuple[str, List[str], int]]:
    """Seed population with 1–2 signal setups to ensure support."""
    base = pop.initialize_population(rng, all_signal_ids)  # list of (ticker, setup)
    if not isinstance(base, list):
        base = list(base)
    if len(base) == 0:
        raise RuntimeError("alpha_discovery.search.population.initialize_population returned empty base.")

    horizons = np.array(cfg.horizons, dtype=int)
    out: List[Tuple[str, List[str], int]] = []

    for (t, _old) in base[: cfg.population_size]:
        s = _random_setup(rng, all_signal_ids, cfg.min_signals_per_setup, cfg.max_signals_per_setup)
        h = int(horizons[int(rng.integers(0, len(horizons)))])
        out.append((t, s, h))

    while len(out) < cfg.population_size:
        t0, s0 = _pick(rng, base)
        t_mut, s_mut = pop.mutate((t0, s0), all_signal_ids, rng)
        s = _random_setup(rng, all_signal_ids, cfg.min_signals_per_setup, cfg.max_signals_per_setup)
        h = int(horizons[int(rng.integers(0, len(horizons)))])
        out.append((t_mut, s, h))

    return out


def mutate(
    rng: np.random.Generator,
    ind: Tuple[str, List[str], int],
    all_signal_ids: List[str],
    cfg: PredictRunConfig,
    rate: float
) -> Tuple[str, List[str], int]:
    t, s, h = ind

    if rng.random() < rate:
        t2, s2 = pop.mutate((t, s), all_signal_ids, rng)
        t, s = t2, list(map(str, s2))

    if rng.random() < rate:
        if len(s) > cfg.min_signals_per_setup and rng.random() < 0.5:
            idx = int(rng.integers(0, len(s)))
            s = s[:idx] + s[idx + 1:]
        elif len(s) < cfg.max_signals_per_setup:
            choices = [x for x in all_signal_ids if x not in s]
            if choices:
                s.append(str(rng.choice(choices)))

    if rng.random() < cfg.horizon_mutation_prob:
        h = int(rng.choice(np.array(cfg.horizons, dtype=int)))

    return (t, s, h)


def crossover(
    rng: np.random.Generator,
    p1: Tuple[str, List[str], int],
    p2: Tuple[str, List[str], int]
) -> Tuple[str, List[str], int]:
    ts = pop.crossover((p1[0], p1[1]), (p2[0], p2[1]), rng)
    h = p1[2] if rng.random() < 0.5 else p2[2]
    s = list(dict.fromkeys(map(str, ts[1])))
    return (ts[0], s[:2], int(h))


# =========================
# Evaluation
# =========================

@dataclass
class EvalMetrics:
    ticker: str
    setup_signals: List[str]
    horizon: int
    support_is: int
    support_oos: int
    mu_is: float
    p_long_is: float
    mu_oos: float
    acc_oos: float
    tstat_oos: float


def evaluate_predictive_power(
    ind: Tuple[str, List[str], int],
    *,
    signals_df: pd.DataFrame,
    master_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    cfg: PredictRunConfig
) -> Optional[Dict]:
    ticker, setup, H = ind
    setup = list(map(str, setup))

    try:
        trig = signals_df[setup].all(axis=1)
    except KeyError:
        return None

    px_col = f"{ticker}_PX_LAST"
    if px_col not in master_df.columns:
        return None

    fwd = forward_log_return(master_df[px_col], H)

    idx = signals_df.index.intersection(master_df.index).intersection(fwd.index)
    idx = pd.DatetimeIndex(idx).sort_values().unique()
    trig = trig.reindex(idx, fill_value=False)
    fwd = fwd.reindex(idx)

    train_idx = pd.DatetimeIndex(train_idx)
    test_idx = pd.DatetimeIndex(test_idx)

    is_mask = (trig.reindex(train_idx, fill_value=False)) & (fwd.reindex(train_idx).notna())
    oos_mask = (trig.reindex(test_idx, fill_value=False)) & (fwd.reindex(test_idx).notna())

    if is_mask.sum() < cfg.min_is_support or oos_mask.sum() < cfg.min_oos_support:
        return None

    y_is = fwd.loc[is_mask.index[is_mask]]
    y_oos = fwd.loc[oos_mask.index[oos_mask]]

    mu_is = float(y_is.mean())
    p_long_is = float((y_is > 0).mean())

    if mu_is == 0.0:
        return None
    predicted_sign = 1.0 if mu_is > 0 else -1.0
    acc = float(((np.sign(y_oos.values) == predicted_sign).astype(float)).mean())
    ts = tstat(y_oos)

    metrics = EvalMetrics(
        ticker=ticker,
        setup_signals=setup,
        horizon=int(H),
        support_is=int(y_is.size),
        support_oos=int(y_oos.size),
        mu_is=mu_is,
        p_long_is=p_long_is,
        mu_oos=float(y_oos.mean()),
        acc_oos=acc,
        tstat_oos=ts,
    )

    return {
        "ticker": ticker,
        "setup_signals": setup,
        "horizon": int(H),
        "objectives": [acc, ts, float(y_is.size), 0.5],
        "metrics": metrics,
    }


# =========================
# Walk-forward splits
# =========================

def create_walk_forward_splits(
    df_index: pd.DatetimeIndex,
    cfg: PredictRunConfig
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    embargo = int(getattr(core_settings.validation, "embargo_days", 7))
    df_index = pd.DatetimeIndex(df_index).sort_values().unique()
    if df_index.empty:
        return []

    first_day = df_index[0]
    last_day = df_index[-1]

    def add_years(ts: pd.Timestamp, years: float) -> pd.Timestamp:
        return ts + pd.DateOffset(years=years)

    def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
        return ts + pd.DateOffset(months=months)

    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(year=first_day.year, month=1 if first_day.month <= 6 else 7, day=1)
    while cursor < last_day:
        train_end = cursor + pd.DateOffset(years=cfg.train_years) - pd.DateOffset(days=1)
        test_start = train_end + pd.Timedelta(days=embargo + 1)
        test_end = add_years(test_start, cfg.test_years) - pd.DateOffset(days=1)

        if test_start > last_day:
            break
        train_start = train_end - pd.DateOffset(years=cfg.train_years) + pd.DateOffset(days=1)
        train_start = max(train_start, first_day)
        train_end = min(train_end, last_day)
        test_end = min(test_end, last_day)

        if train_start < train_end and test_start < test_end:
            splits.append((train_start, train_end, test_start, test_end))

        cursor = add_months(cursor, cfg.split_step_months)

    if cfg.use_back_aligned_final and len(df_index) > 10:
        train_end = last_day - pd.Timedelta(days=embargo + 1)
        train_start = train_end - pd.DateOffset(years=cfg.train_years) + pd.DateOffset(days=1)
        test_start = last_day - pd.DateOffset(years=cfg.test_years) + pd.DateOffset(days=1)
        test_end = last_day
        if train_start < train_end and test_start < test_end:
            if not splits or splits[-1] != (train_start, train_end, test_start, test_end):
                splits.append((train_start, train_end, test_start, test_end))

    return splits


# =========================
# Evolution loop (per fold)
# =========================

def evolve_prediction_strategies(
    *,
    signals_df: pd.DataFrame,
    master_df: pd.DataFrame,
    train_range: Tuple[pd.Timestamp, pd.Timestamp],
    test_range: Tuple[pd.Timestamp, pd.Timestamp],
    cfg: PredictRunConfig
) -> List[Dict]:
    rng = np.random.default_rng(cfg.seed)
    all_signal_ids = list(map(str, signals_df.columns))

    population = init_population(rng, all_signal_ids, cfg)

    def eval_one(ind):
        return evaluate_predictive_power(
            ind,
            signals_df=signals_df,
            master_df=master_df,
            train_idx=pd.date_range(train_range[0], train_range[1], freq="B"),
            test_idx=pd.date_range(test_range[0], test_range[1], freq="B"),
            cfg=cfg
        )

    if cfg.debug_sequential:
        evaluated = list(filter(None, (eval_one(ind) for ind in population)))
    else:
        evaluated = list(filter(
            None,
            Parallel(n_jobs=_safe_n_jobs(cfg.n_jobs))(
                delayed(eval_one)(ind) for ind in population
            )
        ))

    calculate_population_diversity(evaluated)
    _ = non_dominated_sort(evaluated)

    best_acc = max((ind["objectives"][0] for ind in evaluated), default=0.0)
    best_ts = max((ind["objectives"][1] for ind in evaluated), default=0.0)

    generations_without_improvement = 0
    current_mutation_rate = cfg.base_mutation_rate

    for gen in range(1, cfg.generations + 1):
        print(f"\nGeneration {gen}/{cfg.generations}")

        elite_count = max(1, int(cfg.elitism_rate * cfg.population_size))
        elites = select_next_generation(evaluated, elite_count)

        children: List[Tuple[str, List[str], int]] = []
        while len(children) < cfg.population_size * 2:
            p1 = _pick(rng, evaluated)
            p2 = _pick(rng, evaluated)
            c = crossover(rng, (p1["ticker"], p1["setup_signals"], p1["horizon"]),
                               (p2["ticker"], p2["setup_signals"], p2["horizon"]))
            c = mutate(rng, c, all_signal_ids, cfg, rate=current_mutation_rate)
            if generations_without_improvement >= cfg.hard_kick_after and rng.random() < 0.40:
                c = mutate(rng, c, all_signal_ids, cfg, rate=min(1.0, current_mutation_rate * 1.25))
            children.append(c)

        if cfg.debug_sequential:
            off_evals = list(filter(None, (eval_one(ind) for ind in children)))
        else:
            off_evals = list(filter(
                None,
                Parallel(n_jobs=_safe_n_jobs(cfg.n_jobs))(
                    delayed(eval_one)(ind) for ind in children
                )
            ))

        combined = elites + off_evals + evaluated
        calculate_population_diversity(combined)
        fronts_combined = non_dominated_sort(combined)
        evaluated = select_next_generation(combined, cfg.population_size)

        best_front = fronts_combined[0] if fronts_combined else []
        if best_front:
            gen_best_acc = max(ind["objectives"][0] for ind in best_front)
            gen_best_ts  = max(ind["objectives"][1] for ind in best_front)
            avg_div      = float(np.mean([ind["objectives"][3] for ind in best_front]))

            improved = (gen_best_acc > best_acc + 1e-9) or (gen_best_ts > best_ts + 1e-9)
            if improved:
                best_acc = gen_best_acc
                best_ts = gen_best_ts
                generations_without_improvement = 0
                current_mutation_rate = cfg.base_mutation_rate
            else:
                generations_without_improvement += 1
                if generations_without_improvement >= cfg.stagnation_kick_after:
                    current_mutation_rate = min(cfg.max_mutation_rate, current_mutation_rate + 0.05)

            print(f"  Best accuracy: {gen_best_acc:.2%}, Best IR: {gen_best_ts:.2f}")
            print(f"  Diversity: {avg_div:.3f}, Mutation rate: {current_mutation_rate:.2f}")
            if generations_without_improvement > 0:
                print(f"  Generations without improvement: {generations_without_improvement}")

    final_front = non_dominated_sort(evaluated)[0] if evaluated else []
    final_front = sorted(final_front, key=lambda x: x.get("crowding_distance", 0.0), reverse=True)
    return final_front


# =========================
# Aggregation helpers
# =========================

def _write_fold_front_csv(front: List[Dict], fold_dir: str) -> None:
    ensure_dir(fold_dir)
    if not front:
        # Write an empty file as a marker
        with open(os.path.join(fold_dir, "pareto_front.csv"), "w") as f:
            f.write("")
        return

    rows = []
    for ind in front:
        m: EvalMetrics = ind["metrics"]
        row = {
            "rank": ind.get("rank", 1),
            "crowding_distance": ind.get("crowding_distance", np.inf),
            "ticker": m.ticker,
            "setup_signals": "|".join(map(str, m.setup_signals)),
            "horizon": m.horizon,
            "support_is": m.support_is,
            "support_oos": m.support_oos,
            "mu_is": m.mu_is,
            "p_long_is": m.p_long_is,
            "mu_oos": m.mu_oos,
            "acc_oos": m.acc_oos,
            "tstat_oos": m.tstat_oos,
            "obj_acc": ind["objectives"][0],
            "obj_tstat": ind["objectives"][1],
            "obj_support_is": ind["objectives"][2],
            "obj_diversity": ind["objectives"][3],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(fold_dir, "pareto_front.csv"), index=False)


def _aggregate_all_folds(out_dir: str,
                         wf_splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Read fold_XX/pareto_front.csv, concatenate, add metadata, derive direction/edge, and save two CSVs."""
    combined_list: List[pd.DataFrame] = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(wf_splits, start=1):
        fpath = os.path.join(out_dir, f"fold_{i:02d}", "pareto_front.csv")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        df["fold"] = i
        df["train_start"] = pd.to_datetime(tr_s).date()
        df["train_end"] = pd.to_datetime(tr_e).date()
        df["test_start"] = pd.to_datetime(te_s).date()
        df["test_end"] = pd.to_datetime(te_e).date()
        # Derive direction from IS mean; derive expected edge from OOS mean return
        df["direction"] = np.where(df["mu_is"] > 0, "long", "short")
        # Clip extreme values to prevent overflow in exp operation
        mu_oos_clipped = np.clip(df["mu_oos"], -20, 20)  # Clip to reasonable range
        df["expected_edge_pct"] = np.exp(mu_oos_clipped) - 1.0
        combined_list.append(df)

    if not combined_list:
        return None, None

    combined = pd.concat(combined_list, ignore_index=True)

    # Rank a unified “best” per (ticker, direction, horizon) by acc -> tstat -> support_oos
    sort_cols = ["acc_oos", "tstat_oos", "support_oos"]
    top = (combined.sort_values(sort_cols, ascending=[False, False, False])
                  .groupby(["ticker", "direction", "horizon"], as_index=False)
                  .head(1)
                  .reset_index(drop=True))

    # Save
    combined_path = os.path.join(out_dir, "combined_pareto_front.csv")
    top_path = os.path.join(out_dir, "tradeable_signals_top.csv")
    combined.to_csv(combined_path, index=False)
    top.to_csv(top_path, index=False)
    return combined, top


# =========================
# Main
# =========================

def main():
    cfg = get_cfg_from_cli()

    print("=" * 59)
    print("ENHANCED PREDICTION EVALUATION PIPELINE")
    print("=" * 59)

    # --- Loading data ---
    print("\n--- Loading Data ---")
    parquet_path = getattr(core_settings.data, "parquet_file_path", "data_store/processed/bb_data.parquet")
    print(f"Loading data from '{parquet_path}'...")
    master_df = load_data_from_parquet()
    if master_df is None or master_df.empty:
        raise SystemExit("Master dataframe is empty or missing.")
    print(f"Data loaded successfully. Shape: {master_df.shape}")

    # --- Building features ---
    print("\n--- Building Features ---")
    print("Starting feature matrix construction...")
    feature_matrix = build_feature_matrix(master_df)
    print(f" Feature matrix construction complete. Shape: {feature_matrix.shape}")

    # --- Compiling signals ---
    print("\n--- Compiling Signals ---")
    print("Compiling primitive signals from feature matrix...")
    signals_df, signals_metadata = compile_signals(feature_matrix)
    print(f" Signal compilation complete. Generated {signals_df.shape[1]} primitive signals.")

    # --- Walk-forward splits ---
    print("\n--- Creating Walk-Forward Splits ---")
    base_idx = pd.DatetimeIndex(master_df.index).sort_values().unique()
    wf_splits = create_walk_forward_splits(base_idx, cfg)
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(wf_splits, start=1):
        label = f"Created Split {i}: Train ({tr_s.date()} to {tr_e.date()}), Test ({te_s.date()} to {te_e.date()})"
        if i == len(wf_splits) and cfg.use_back_aligned_final:
            label = "Created Back-Aligned Final Split:  " + f"Train ({tr_s.date()} to {tr_e.date()}), Test ({te_s.date()} to {te_e.date()})"
        print(label)
    print(f"Generated {len(wf_splits)} walk-forward splits.")
    print(f"Total folds: {len(wf_splits)}")

    # --- Per fold evolution ---
    ensure_dir(cfg.out_dir)

    for fold, (tr_s, tr_e, te_s, te_e) in enumerate(wf_splits, start=1):
        print(f"\n==================== FOLD {fold}/{len(wf_splits)} ====================")
        print(f"Train: {tr_s.date()} to {tr_e.date()}")
        print(f"Test:  {te_s.date()} to {te_e.date()}")
        print(f"Initializing specialized population of size {cfg.population_size}...")
        print("\n--- Starting Enhanced Prediction Evolution ---")

        front = evolve_prediction_strategies(
            signals_df=signals_df,
            master_df=master_df,
            train_range=(tr_s, tr_e),
            test_range=(te_s, te_e),
            cfg=cfg
        )

        fold_dir = os.path.join(cfg.out_dir, f"fold_{fold:02d}")
        _write_fold_front_csv(front, fold_dir)

    # --- Aggregate all folds into a single CSV + top tradeables ---
    print("\nAll folds complete.")
    print("Aggregating per-fold results into a single file...")
    combined_df, top_df = _aggregate_all_folds(cfg.out_dir, wf_splits)
    if combined_df is None:
        print("No per-fold results found to aggregate (combined file not written).")
    else:
        print(f"Wrote combined file: {os.path.join(cfg.out_dir, 'combined_pareto_front.csv')}")
        print(f"Wrote top tradeables: {os.path.join(cfg.out_dir, 'tradeable_signals_top.csv')}")

    print(f"Outputs written under: {cfg.out_dir}")


if __name__ == "__main__":
    main()
