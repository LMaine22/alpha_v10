# alpha_discovery/search/nsga.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ..config import settings
from . import population as pop

from .ga_core import (
    threadpool_limits,
    VERBOSE, DEBUG_SEQUENTIAL, JOBLIB_VERBOSE,
    _dna, _exit_policy_from_settings,
    _evaluate_one_setup_cached, _summarize_evals,
)

import zlib

# ---------- Deterministic seeding per individual ----------
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
    except Exception:
        pass
    try:
        import numpy as _np
        _np.random.seed(int(s) & 0xFFFFFFFF)
    except Exception:
        pass

    return _evaluate_one_setup_cached(
        individual, signals_df, signals_metadata, master_df, exit_policy
    )

# ---------- NSGA machinery ----------
def _non_dominated_sort(population: List[Dict]) -> List[List[Dict]]:
    fronts: List[List[Dict]] = []
    for ind1 in population:
        ind1["domination_count"] = 0
        ind1["dominated_solutions"] = []
        for ind2 in population:
            if ind1 is ind2:
                continue
            better_or_equal = all(a >= b for a, b in zip(ind1["objectives"], ind2["objectives"]))
            strictly_better = any(a > b for a, b in zip(ind1["objectives"], ind2["objectives"]))
            if better_or_equal and strictly_better:
                ind1["dominated_solutions"].append(ind2)
            elif all(b >= a for a, b in zip(ind1["objectives"], ind2["objectives"])) and any(
                b > a for a, b in zip(ind1["objectives"], ind2["objectives"])
            ):
                ind1["domination_count"] += 1
        if ind1["domination_count"] == 0:
            if not fronts:
                fronts.append([])
            fronts[0].append(ind1)
    i = 0
    while i < len(fronts):
        next_front: List[Dict] = []
        for p in fronts[i]:
            for q in p["dominated_solutions"]:
                q["domination_count"] -= 1
                if q["domination_count"] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1
    return fronts

def _calculate_crowding_distance(front: List[Dict]) -> None:
    if not front:
        return
    num_obj = len(front[0]["objectives"])
    for ind in front:
        ind["crowding_distance"] = 0.0
    for m in range(num_obj):
        front.sort(key=lambda c: c["objectives"][m])
        front[0]["crowding_distance"] = float("inf")
        front[-1]["crowding_distance"] = float("inf")
        vals = [c["objectives"][m] for c in front]
        v_min, v_max = min(vals), max(vals)
        if v_max == v_min:
            continue
        for i in range(1, len(front) - 1):
            front[i]["crowding_distance"] += (
                (front[i + 1]["objectives"][m] - front[i - 1]["objectives"][m]) / (v_max - v_min)
            )

# ---------- Population de-duplication ----------
def _dedup_individuals(seq: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """Keep first occurrence of each DNA; preserve order."""
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()
    out: List[Tuple[str, List[str]]] = []
    for ind in seq:
        key = _dna(ind)
        if key in seen:
            continue
        seen.add(key)
        out.append(ind)
    return out

# ---------- Evolution loop ----------
def evolve(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    """NSGA-II evolution for specialized (ticker, setup) individuals."""
    
    # Check if island model is enabled
    if settings.ga.islands and settings.ga.islands.enabled:
        return _evolve_with_islands(signals_df, signals_metadata, master_df)
    else:
        return _evolve_single_population(signals_df, signals_metadata, master_df)


def _evolve_with_islands(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    """Evolve using island model."""
    from .island_model import IslandManager
    
    tqdm.write("\n--- Starting Island Model Evolution ---")
    
    # Create island manager
    island_manager = IslandManager(signals_df, signals_metadata, master_df)
    
    # Run evolution
    final_population = island_manager.evolve()
    
    # Log migration summary
    migration_summary = island_manager.get_migration_summary()
    diversity_metrics = island_manager.get_island_diversity_metrics()
    
    tqdm.write(f"\nIsland Model Evolution Complete:")
    tqdm.write(f"  Total Migrations: {migration_summary['total_migrations']}")
    tqdm.write(f"  Final Diversity: {diversity_metrics['final_diversity']:.3f}")
    tqdm.write(f"  Final Population Size: {len(final_population)}")
    
    return final_population


def _evolve_single_population(signals_df: pd.DataFrame, signals_metadata: List[Dict], master_df: pd.DataFrame) -> List[Dict]:
    """Original single population evolution."""
    rng = np.random.default_rng(settings.ga.seed)
    all_signal_ids = list(signals_df.columns)

    # Initialize parents and deduplicate
    parent_population: List[Tuple[str, List[str]]] = pop.initialize_population(rng, all_signal_ids)
    parent_population = _dedup_individuals(parent_population)

    # Build a single exit policy from config (used for ALL evals)
    exit_policy = _exit_policy_from_settings()

    g = int(settings.ga.generations)
    p = int(settings.ga.population_size)
    tqdm.write("\n--- Starting Genetic Algorithm Evolution (Ticker Specialization Mode) ---")
    final_fronts: List[List[Dict]] = []

    with tqdm(range(1, g + 1), desc="Evolving Generations", dynamic_ncols=True, leave=True) as pbar:
        for gen in pbar:
            tqdm.write(f"Gen {gen}/{g} | Evaluating {len(parent_population)} parents...")
            if DEBUG_SEQUENTIAL:
                evaluated_parents = [
                    _deterministic_eval(
                        ind, signals_df, signals_metadata, master_df,
                        exit_policy,
                        base_seed=int(settings.ga.seed),
                    ) for ind in parent_population
                ]
            else:
                with threadpool_limits(limits=1):
                    evaluated_parents = Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSE)(
                        delayed(_deterministic_eval)(
                            ind, signals_df, signals_metadata, master_df,
                            exit_policy,
                            int(settings.ga.seed),
                        ) for ind in parent_population
                    )
            if VERBOSE >= 2:
                _summarize_evals("Parents", evaluated_parents)

            # Children: build, dedup vs parents+children
            existing_keys = { _dna(ind) for ind in parent_population }
            children_population: List[Tuple[str, List[str]]] = []

            while len(children_population) < p:
                # Tournament selection on evaluated parents
                p1 = min(rng.choice(evaluated_parents, 2, replace=False), key=lambda x: (x["rank"], -x["crowding_distance"]))
                p2 = min(rng.choice(evaluated_parents, 2, replace=False), key=lambda x: (x["rank"], -x["crowding_distance"]))

                child_ind = pop.crossover(p1["individual"], p2["individual"], rng)
                child_ind = pop.mutate(child_ind, all_signal_ids, rng)
                if not child_ind[1]:
                    continue

                if _dna(child_ind) in existing_keys:
                    continue  # skip clones

                children_population.append(child_ind)
                existing_keys.add(_dna(child_ind))

                # Safety: if mutation space gets stuck, allow reseed
                if len(children_population) < p and len(existing_keys) > 4 * p:
                    seedling = pop.mutate(pop.crossover(p1["individual"], p2["individual"], rng),
                                         all_signal_ids, rng)
                    if seedling[1] and _dna(seedling) not in existing_keys:
                        children_population.append(seedling)
                        existing_keys.add(_dna(seedling))
                    else:
                        break

            tqdm.write(f"Gen {gen}/{g} | Evaluating {len(children_population)} children...")
            if DEBUG_SEQUENTIAL:
                evaluated_children = [
                    _deterministic_eval(
                        ind, signals_df, signals_metadata, master_df,
                        exit_policy,
                        base_seed=int(settings.ga.seed),
                    ) for ind in children_population
                ]
            else:
                with threadpool_limits(limits=1):
                    evaluated_children = Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSE)(
                        delayed(_deterministic_eval)(
                            ind, signals_df, signals_metadata, master_df,
                            exit_policy,
                            int(settings.ga.seed),
                        ) for ind in children_population
                    )
            if VERBOSE >= 2:
                _summarize_evals("Children", evaluated_children)

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

            # De-duplicate survivors by DNA (keep first by crowding order)
            uniq = []
            seen: Set[Tuple[str, Tuple[str, ...]]] = set()
            for ind in next_gen_parents:
                key = _dna(ind["individual"])
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(ind)
            next_gen_parents = uniq

            # If de-dup shrank below p, top up with mutated variants of best_front
            if len(next_gen_parents) < p:
                best_front = fronts[0] if fronts else []
                seeds = [bf["individual"] for bf in best_front] or parent_population
                existing = { _dna(ind["individual"]) for ind in next_gen_parents }
                rng_local = np.random.default_rng(settings.ga.seed + gen)
                while len(next_gen_parents) < p and seeds:
                    base = seeds[int(len(next_gen_parents)) % len(seeds)]
                    child = pop.mutate(base, all_signal_ids, rng_local)
                    if child[1] and _dna(child) not in existing:
                        next_gen_parents.append({
                            "individual": child,
                            "objectives": [0.0, 0.0, 0.0],
                            "rank": np.inf,
                            "crowding_distance": 0.0,
                            "trade_ledger": pd.DataFrame(),
                            "direction": "long",
                            "exit_policy": exit_policy,
                        })
                        existing.add(_dna(child))
                    else:
                        break

            parent_population = [ind["individual"] for ind in next_gen_parents]

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
