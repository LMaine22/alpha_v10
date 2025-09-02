# alpha_discovery/search/nsga.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import settings
from . import population as pop

from .ga_core import (
    threadpool_limits,
    VERBOSE, DEBUG_SEQUENTIAL, JOBLIB_VERBOSE,
    _dna,
    _sample_exit_policy, _mutate_exit_policy, _crossover_exit_policy,
    _evaluate_one_setup_cached, _summarize_evals,
)

import zlib

def _seed_for_individual(individual: Tuple[str, List[str]], base_seed: int) -> int:
    """Deterministic 32-bit seed derived from the (ticker, setup) DNA."""
    try:
        dna = _dna(individual)
        payload = (f"{dna[0]}|" + "|".join(map(str, dna[1]))).encode("utf-8", "ignore")
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
) -> Dict:
    """Wrapper to seed evaluation for the new individual structure."""
    s = _seed_for_individual(individual, base_seed)
    try:
        import random as _rnd
        _rnd.seed(int(s))
    except Exception: pass
    try:
        import numpy as _np
        _np.random.seed(int(s) & 0xFFFFFFFF)
    except Exception: pass

    return _evaluate_one_setup_cached(
        individual, signals_df, signals_metadata, master_df, exit_policy
    )

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
            elif all(b >= a for a, b in zip(ind1["objectives"], ind2["objectives"])) and any(
                b > a for a, b in zip(ind1["objectives"], ind2["objectives"])
            ):
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

def evolve(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    """NSGA-II evolution for specialized (ticker, setup) individuals."""
    rng = np.random.default_rng(settings.ga.seed)
    all_signal_ids = list(signals_df.columns)

    parent_population: List[Tuple[str, List[str]]] = pop.initialize_population(rng, all_signal_ids)

    policy_map: Dict[Tuple[str, Tuple[str, ...]], Dict] = {}
    if settings.options.exit_policies_enabled:
        for individual in parent_population:
            policy_map[_dna(individual)] = _sample_exit_policy(rng)

    g = int(settings.ga.generations)
    p = int(settings.ga.population_size)
    tqdm.write("\n--- Starting Genetic Algorithm Evolution (Ticker Specialization Mode) ---")
    final_fronts: List[List[Dict]] = []

    with tqdm(range(1, g + 1), desc="Evolving Generations", dynamic_ncols=True, leave=True) as pbar:
        for gen in pbar:
            tqdm.write(f"Gen {gen}/{g} | Evaluating {p} parents...")
            if DEBUG_SEQUENTIAL:
                evaluated_parents = [
                    _deterministic_eval(
                        ind, signals_df, signals_metadata, master_df,
                        policy_map.get(_dna(ind)) if settings.options.exit_policies_enabled else None,
                        base_seed=int(settings.ga.seed),
                    ) for ind in parent_population
                ]
            else:
                with threadpool_limits(limits=1):
                    evaluated_parents = Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSE)(
                        delayed(_deterministic_eval)(
                            ind, signals_df, signals_metadata, master_df,
                            policy_map.get(_dna(ind)) if settings.options.exit_policies_enabled else None,
                            int(settings.ga.seed),
                        ) for ind in parent_population
                    )
            if VERBOSE >= 2: _summarize_evals("Parents", evaluated_parents)

            children_population: List[Tuple[str, List[str]]] = []
            children_policies: List[Dict] = []
            while len(children_population) < p:
                p1 = min(rng.choice(evaluated_parents, 2, replace=False), key=lambda x: (x["rank"], -x["crowding_distance"]))
                p2 = min(rng.choice(evaluated_parents, 2, replace=False), key=lambda x: (x["rank"], -x["crowding_distance"]))
                child_ind = pop.crossover(p1["individual"], p2["individual"], rng)
                child_ind = pop.mutate(child_ind, all_signal_ids, rng)
                if not child_ind[1]: continue
                children_population.append(child_ind)
                if settings.options.exit_policies_enabled:
                    pol1 = policy_map.get(_dna(p1["individual"]), _sample_exit_policy(rng))
                    pol2 = policy_map.get(_dna(p2["individual"]), _sample_exit_policy(rng))
                    child_pol = _crossover_exit_policy(pol1, pol2, rng)
                    child_pol = _mutate_exit_policy(child_pol, rng)
                    children_policies.append(child_pol)

            tqdm.write(f"Gen {gen}/{g} | Evaluating {p} children...")
            if DEBUG_SEQUENTIAL:
                evaluated_children = [
                    _deterministic_eval(
                        ind, signals_df, signals_metadata, master_df,
                        children_policies[i] if settings.options.exit_policies_enabled and i < len(children_policies) else None,
                        base_seed=int(settings.ga.seed),
                    ) for i, ind in enumerate(children_population)
                ]
            else:
                with threadpool_limits(limits=1):
                    evaluated_children = Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSE)(
                        delayed(_deterministic_eval)(
                            ind, signals_df, signals_metadata, master_df,
                            children_policies[i] if settings.options.exit_policies_enabled and i < len(children_policies) else None,
                            int(settings.ga.seed),
                        ) for i, ind in enumerate(children_population)
                    )
            if VERBOSE >= 2: _summarize_evals("Children", evaluated_children)

            tqdm.write(f"Gen {gen}/{g} | Selecting survivors...")
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
                    new_policy_map[dna] = ind.get("exit_policy", policy_map.get(dna, _sample_exit_policy(rng)))
                policy_map = new_policy_map

            best_front = fronts[0] if fronts else []
            if best_front:
                best_sortino = max((ind["objectives"][0] for ind in best_front), default=0.0)
                best_expectancy = max((ind["objectives"][1] for ind in best_front), default=0.0)
                pbar.set_postfix({
                    "Sortino LB": f"{best_sortino:.2f}",
                    "Expectancy": f"${best_expectancy:.2f}",
                    "Front Size": len(best_front)
                }, refresh=True)
            else:
                pbar.set_postfix({"Sortino LB": "0.00", "Expectancy": "$0.00", "Front Size": 0}, refresh=True)
            final_fronts = fronts

    tqdm.write("Evolution Complete.")
    return final_fronts[0] if final_fronts else []

