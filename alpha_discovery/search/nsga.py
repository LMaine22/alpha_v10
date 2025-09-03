# alpha_discovery/search/nsga.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from . import population as pop

from joblib import Parallel, delayed
from tqdm.auto import tqdm
import zlib

from ..config import settings
from . import population as pop
from .ga_core import (
    threadpool_limits,
    VERBOSE, DEBUG_SEQUENTIAL, JOBLIB_VERBOSE,
    _dna,
    _sample_exit_policy, _mutate_exit_policy, _crossover_exit_policy,
    _evaluate_one_setup_cached, _summarize_evals,
    memetic_tune_exit_policy,
)

def _seed_for_individual(individual: Tuple[str, List[str]], base_seed: int) -> int:
    try:
        ticker, setup = _dna(individual)
        payload = (f"{ticker}|" + "|".join(setup)).encode("utf-8", "ignore")
        s = zlib.adler32(payload)
        return int((s ^ (int(base_seed) & 0xFFFFFFFF)) & 0xFFFFFFFF)
    except Exception:
        return int(base_seed) & 0xFFFFFFFF

def _deterministic_eval(
    individual: Tuple[str, List[str]],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    exit_policy: Optional[Dict],
    base_seed: int,
    mf_mode: Tuple[bool,int,float],
) -> Dict:
    s = _seed_for_individual(individual, base_seed)
    try:
        import random as _rnd; _rnd.seed(int(s))
    except Exception:  # pragma: no cover
        pass
    try:
        import numpy as _np; _np.random.seed(int(s) & 0xFFFFFFFF)
    except Exception:  # pragma: no cover
        pass
    return _evaluate_one_setup_cached(individual, signals_df, signals_metadata, master_df, exit_policy, mf_mode)

# ──────────────────────────────────────────────────────────────────────────────
# NSGA-II
# ──────────────────────────────────────────────────────────────────────────────
def _non_dominated_sort(population: List[Dict]) -> List[List[Dict]]:
    fronts: List[List[Dict]] = []
    for ind1 in population:
        ind1["domination_count"] = 0
        ind1["dominated_solutions"] = []
        for ind2 in population:
            if ind1 is ind2: continue
            better_or_equal = all(a >= b for a, b in zip(ind1["objectives"], ind2["objectives"]))
            strictly_better = any(a > b for a, b in zip(ind1["objectives"], ind2["objectives"]))
            if better_or_equal and strictly_better:
                ind1["dominated_solutions"].append(ind2)
            elif all(b >= a for a, b in zip(ind1["objectives"], ind2["objectives"])) and any(b > a for a, b in zip(ind1["objectives"], ind2["objectives"])):
                ind1["domination_count"] += 1
        if ind1["domination_count"] == 0:
            if not fronts: fronts.append([])
            fronts[0].append(ind1)
    i = 0
    while i < len(fronts):
        next_front: List[Dict] = []
        for p in fronts[i]:
            for q in p["dominated_solutions"]:
                q["domination_count"] -= 1
                if q["domination_count"] == 0:
                    next_front.append(q)
        if next_front: fronts.append(next_front)
        i += 1
    return fronts

def _calculate_crowding_distance(front: List[Dict]) -> None:
    if not front: return
    num_obj = len(front[0]["objectives"])
    for ind in front: ind["crowding_distance"] = 0.0
    for m in range(num_obj):
        front.sort(key=lambda c: c["objectives"][m])
        front[0]["crowding_distance"] = float("inf")
        front[-1]["crowding_distance"] = float("inf")
        vals = [c["objectives"][m] for c in front]
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min: continue
        for i in range(1, len(front) - 1):
            front[i]["crowding_distance"] += (front[i + 1]["objectives"][m] - front[i - 1]["objectives"][m]) / (v_max - v_min)

# ε-lexicase over regime×horizon cases
def _select_parent_lexicase(evaluated: List[Dict], rng: np.random.Generator) -> Dict:
    case_names = set()
    for ind in evaluated:
        reg = ind.get("regime_objectives") or {}
        for k in reg.keys(): case_names.add(k)
    if not case_names:
        return min(evaluated, key=lambda x: (x["rank"], -x.get("crowding_distance", 0.0)))
    cases = list(case_names); rng.shuffle(cases)
    eps_q = float(getattr(getattr(settings.ga, "lexicase", object()), "epsilon_quantile", 0.20))
    pool = evaluated[:]
    for c in cases:
        vals = np.array([float(ind.get("regime_objectives", {}).get(c, -1e9)) for ind in pool], dtype=float)
        if vals.size == 0: continue
        best = float(np.max(vals))
        if not np.isfinite(best): continue
        thr = float(np.quantile(vals, 1.0 - eps_q))
        next_pool = [ind for (ind, v) in zip(pool, vals) if v >= thr]
        pool = next_pool if next_pool else pool
        if len(pool) == 1: return pool[0]
    return min(pool, key=lambda x: (x["rank"], -x.get("crowding_distance", 0.0)))

def _select_parent_tournament(evaluated: List[Dict], rng: np.random.Generator, k: int = 2) -> Dict:
    cand = rng.choice(evaluated, k, replace=False)
    return min(cand, key=lambda x: (x["rank"], -x.get("crowding_distance", 0.0)))

def _pick_parent(evaluated: List[Dict], rng: np.random.Generator) -> Dict:
    if str(getattr(getattr(settings.ga, "selector", object()), "name", "")).lower() == "lexicase":
        return _select_parent_lexicase(evaluated, rng)
    return _select_parent_tournament(evaluated, rng, k=2)

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    inter = len(sa & sb); union = len(sa | sb)
    return 0.0 if union == 0 else inter / union

def _u32(x: int) -> int:
    return int(x) & 0xFFFFFFFF

# ──────────────────────────────────────────────────────────────────────────────
# One generation step
# ──────────────────────────────────────────────────────────────────────────────
def _one_generation_step(
    rng: np.random.Generator,
    parent_population: List[Tuple[str, List[str]]],
    policy_map: Dict[Tuple[str, Tuple[str, ...]], Dict],
    signals_df: pd.DataFrame,
    signals_metadata: List[Dict],
    master_df: pd.DataFrame,
    gen_idx: int,
) -> Tuple[List[Tuple[str, List[str]]], Dict, List[List[Dict]]]:
    p = len(parent_population)

    # multi-fidelity
    mf_cfg = getattr(settings.ga, "multifidelity", None)
    if mf_cfg and bool(getattr(mf_cfg, "enabled", True)):
        full_after = int(getattr(mf_cfg, "full_after_gen", 10))
        if gen_idx <= full_after:
            stride = int(getattr(mf_cfg, "stride", 2))
            span_frac = float(getattr(mf_cfg, "coarse_span_frac", 0.5))
            mf_mode = (True, stride, span_frac)
        else:
            mf_mode = (False, 1, 1.0)
    else:
        mf_mode = (False, 1, 1.0)

    # skip memetic in very early gens
    use_memetic = bool(getattr(settings.ga, "memetic", None) and settings.ga.memetic.enabled and gen_idx > 2)

    # 1) Evaluate parents
    evaluated_parents: List[Dict] = []
    if DEBUG_SEQUENTIAL:
        evaluated_parents = [
            _deterministic_eval(
                ind, signals_df, signals_metadata, master_df,
                policy_map.get(_dna(ind)) if settings.options.exit_policies_enabled else None,
                base_seed=int(settings.ga.seed), mf_mode=mf_mode,
            ) for ind in parent_population
        ]
    else:
        with threadpool_limits(limits=1):
            evaluated_parents = Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSE)(
                delayed(_deterministic_eval)(
                    ind, signals_df, signals_metadata, master_df,
                    policy_map.get(_dna(ind)) if settings.options.exit_policies_enabled else None,
                    int(settings.ga.seed), mf_mode,
                ) for ind in parent_population
            )
    if VERBOSE >= 2:
        _summarize_evals("Parents", evaluated_parents)

    # 2) Create children
    children_population: List[Tuple[str, List[str]]] = []
    children_policies: List[Dict] = []
    all_signal_ids = list(signals_df.columns)
    NICHE_MAX = float(getattr(settings.ga, "niche_jaccard_max", 0.85))

    while len(children_population) < p:
        p1 = _pick_parent(evaluated_parents, rng)
        p2 = _pick_parent(evaluated_parents, rng)
        child_ind = pop.crossover(p1["individual"], p2["individual"], rng)
        child_ind = pop.mutate(child_ind, all_signal_ids, rng)
        if not child_ind[1]:
            continue
        too_similar = any(
            (child_ind[0] == exist[0]) and (_jaccard(child_ind[1], exist[1]) >= NICHE_MAX)
            for exist in children_population
        )
        if too_similar:
            child_ind = pop.mutate(child_ind, all_signal_ids, rng)
            if not child_ind[1]:
                continue

        if settings.options.exit_policies_enabled:
            pol1 = policy_map.get(_dna(p1["individual"]), _sample_exit_policy(rng))
            pol2 = policy_map.get(_dna(p2["individual"]), _sample_exit_policy(rng))
            child_pol = _crossover_exit_policy(pol1, pol2, rng)
            child_pol = _mutate_exit_policy(child_pol, rng)
            if use_memetic:
                child_pol = memetic_tune_exit_policy(child_ind, signals_df, signals_metadata, master_df, child_pol, rng, mf_mode)
            children_policies.append(child_pol)

        children_population.append(child_ind)

    # 3) Evaluate children
    evaluated_children: List[Dict] = []
    if DEBUG_SEQUENTIAL:
        evaluated_children = [
            _deterministic_eval(
                ind, signals_df, signals_metadata, master_df,
                children_policies[i] if settings.options.exit_policies_enabled and i < len(children_policies) else None,
                base_seed=int(settings.ga.seed), mf_mode=mf_mode,
            ) for i, ind in enumerate(children_population)
        ]
    else:
        with threadpool_limits(limits=1):
            evaluated_children = Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSE)(
                delayed(_deterministic_eval)(
                    ind, signals_df, signals_metadata, master_df,
                    children_policies[i] if settings.options.exit_policies_enabled and i < len(children_policies) else None,
                    int(settings.ga.seed), mf_mode,
                ) for i, ind in enumerate(children_population)
            )
    if VERBOSE >= 2:
        _summarize_evals("Children", evaluated_children)

    # 4) NSGA-II survival
    combined = evaluated_parents + evaluated_children
    fronts = _non_dominated_sort(combined)

    next_gen_parents: List[Dict] = []
    for front in fronts:
        _calculate_crowding_distance(front)
        if len(next_gen_parents) + len(front) <= p:
            next_gen_parents.extend(front)
        else:
            front.sort(key=lambda x: x["crowding_distance"], reverse=True)
            need = p - len(next_gen_parents)
            next_gen_parents.extend(front[:need])
            break

    parent_population = [ind["individual"] for ind in next_gen_parents]
    if settings.options.exit_policies_enabled:
        new_policy_map: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}
        for ind in next_gen_parents:
            dna = _dna(ind["individual"])
            new_policy_map[dna] = ind.get("exit_policy", _sample_exit_policy(rng))
        policy_map = new_policy_map

    return parent_population, policy_map, fronts

# ──────────────────────────────────────────────────────────────────────────────
# Coverage-biased bootstrap (ensures gen-1 viability)
# ──────────────────────────────────────────────────────────────────────────────
def _bootstrap_population_with_coverage(
    rng: np.random.Generator,
    signals_df: pd.DataFrame,
    base_pop: List[Tuple[str, List[str]]],
    target_size: int,
) -> List[Tuple[str, List[str]]]:
    """Add a few high-coverage singletons and pairs to help early ledgers."""
    if target_size <= 0: return base_pop
    # Estimate per-signal coverage (mean True)
    cov = signals_df.mean(axis=0).sort_values(ascending=False)
    if cov.empty: return base_pop
    # Choose top signals but avoid ones with absurdly high coverage (> 0.9)
    top = cov[(cov > 0.02) & (cov < 0.9)].index.tolist()[: min(200, len(cov))]
    if not top: top = cov.index.tolist()[: min(200, len(cov))]

    # Randomly pick tickers from columns if encoded in signal_id (optional)
    # Otherwise, we just let the downstream code pick ticker during crossover/mutation
    seeds: List[Tuple[str, List[str]]] = []
    for _ in range(max(4, target_size // 4)):
        # singleton
        s1 = rng.choice(top)
        ticker = "GEN"  # ticker gets overridden by your existing operators
        seeds.append((ticker, [str(s1)]))
        # pair
        s2 = rng.choice(top)
        if s2 != s1:
            seeds.append((ticker, [str(s1), str(s2)]))

        if len(seeds) >= target_size:
            break

    # Replace the tail of the base population with seeds (keeps size same)
    out = list(base_pop)
    replace = min(len(seeds), max(0, len(out)))
    if replace > 0:
        out[-replace:] = seeds[:replace]
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Public evolve(): islands + migration
# ──────────────────────────────────────────────────────────────────────────────
def evolve(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    g = int(settings.ga.generations)
    p = int(settings.ga.population_size)
    I = int(getattr(settings.ga, "islands", 1))
    MIG = int(getattr(settings.ga, "migration_period", 3))

    # ── Single-population path ────────────────────────────────────────────────
    if I <= 1:
        rng = np.random.default_rng(settings.ga.seed)

        # ✅ pass size=p here
        parents: List[Tuple[str, List[str]]] = pop.initialize_population(
            rng, list(signals_df.columns), size=p
        )

        policy_map: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}
        if settings.options.exit_policies_enabled:
            for individual in parents:
                policy_map[_dna(individual)] = _sample_exit_policy(rng)

        last_fronts: List[List[Dict]] = []
        with tqdm(range(1, g + 1), desc="Evolving Generations", dynamic_ncols=True, leave=True) as pbar:
            for gen in pbar:
                parents, policy_map, fronts = _one_generation_step(
                    rng, parents, policy_map, signals_df, signals_metadata, master_df, gen
                )
                best_front = fronts[0] if fronts else []
                if best_front:
                    best_sortino = max((ind["objectives"][0] for ind in best_front), default=0.0)
                    best_expectancy = max((ind["objectives"][1] for ind in best_front), default=0.0)
                    pbar.set_postfix({
                        "Sortino LB": f"{best_sortino:.2f}",
                        "Expectancy": f"${best_expectancy:.2f}",
                        "Front Size": len(best_front)
                    }, refresh=True)
                last_fronts = fronts

        return last_fronts[0] if last_fronts else []

    # ── Island model path ────────────────────────────────────────────────────
    per_island = max(2, p // I)
    islands = []
    for k in range(I):
        seed_k = (int(settings.ga.seed) & 0xFFFFFFFF) ^ (int(0x9E3779B1 * (k + 1)) & 0xFFFFFFFF)
        rng_k = np.random.default_rng(seed_k)

        # ✅ pass size=per_island here (AFTER rng_k is defined)
        parents_k = pop.initialize_population(
            rng_k, list(signals_df.columns), size=per_island
        )

        policy_k: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}
        if settings.options.exit_policies_enabled:
            for ind in parents_k:
                policy_k[_dna(ind)] = _sample_exit_policy(rng_k)

        islands.append({
            "rng": rng_k,
            "parents": parents_k,
            "policy": policy_k,
            "last_fronts": []
        })

    with tqdm(range(1, g + 1), desc=f"Islands x{I}", dynamic_ncols=True, leave=True) as pbar:
        for gen in pbar:
            # advance each island
            for isl in islands:
                parents, pol, fronts = _one_generation_step(
                    isl["rng"], isl["parents"], isl["policy"],
                    signals_df, signals_metadata, master_df, gen
                )
                isl["parents"], isl["policy"], isl["last_fronts"] = parents, pol, fronts

            # migrate top-1 + one random
            if gen % MIG == 0:
                migrants = []
                for isl in islands:
                    front = isl["last_fronts"][0] if isl["last_fronts"] else []
                    if front:
                        elites = sorted(front, key=lambda x: (x["rank"], -x.get("crowding_distance", 0.0)))[:1]
                        pick_rand = isl["rng"].choice(front) if front else None
                        mig = [e["individual"] for e in elites]
                        if pick_rand: mig.append(pick_rand["individual"])
                    else:
                        mig = isl["rng"].choice(isl["parents"], size=min(2, len(isl["parents"])), replace=False).tolist()
                    migrants.append(mig)

                # ring
                for k in range(I):
                    nxt = (k + 1) % I
                    if not migrants[k]: continue
                    for m in migrants[k]:
                        islands[nxt]["parents"][-1:] = [m]
                        islands[nxt]["policy"][_dna(m)] = islands[k]["policy"].get(_dna(m), _sample_exit_policy(islands[nxt]["rng"]))

            # progress bar
            best_sortino = 0.0; best_expect = 0.0; best_front_size = 0
            for isl in islands:
                bf = isl["last_fronts"][0] if isl["last_fronts"] else []
                if bf:
                    best_sortino = max(best_sortino, max((ind["objectives"][0] for ind in bf), default=0.0))
                    best_expect  = max(best_expect,  max((ind["objectives"][1] for ind in bf), default=0.0))
                    best_front_size = max(best_front_size, len(bf))
            pbar.set_postfix({"Sortino LB": f"{best_sortino:.2f}", "Expectancy": f"${best_expect:.2f}", "Front Size": best_front_size}, refresh=True)

    # best across islands
    best_front = []
    best_score = -1e9
    for isl in islands:
        bf = isl["last_fronts"][0] if isl["last_fronts"] else []
        if not bf: continue
        score = max((ind["objectives"][0] for ind in bf), default=-1e9)
        if score > best_score:
            best_score = score;
            best_front = bf
    return best_front
