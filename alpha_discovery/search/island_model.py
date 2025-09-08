# alpha_discovery/search/island_model.py
"""
Island Model Implementation for NSGA-II Genetic Algorithm

This module implements an island model where the population is divided into
multiple semi-independent sub-populations that evolve in parallel, with
occasional migration between them to maintain diversity and prevent
premature convergence.
"""

from __future__ import annotations

import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..config import settings
from . import nsga
from . import population as pop
from .ga_core import _evaluate_one_setup_cached, _exit_policy_from_settings


@dataclass
class IslandMetrics:
    """Metrics for tracking island performance."""
    island_id: int
    generation: int
    population_size: int
    best_fitness: List[float]
    avg_fitness: List[float]
    diversity_metric: float
    migration_sent: int
    migration_received: int
    evaluation_time: float


@dataclass
class MigrationEvent:
    """Represents a migration event between islands."""
    from_island: int
    to_island: int
    individuals: List[Dict]
    generation: int


class Island:
    """Represents a single island in the island model."""
    
    def __init__(self, island_id: int, population_size: int, 
                 all_signal_ids: List[str], base_seed: int):
        self.island_id = island_id
        self.population_size = population_size
        self.all_signal_ids = all_signal_ids
        self.base_seed = base_seed
        
        # Island-specific RNG for reproducibility
        self.rng = np.random.default_rng(base_seed + island_id)
        
        # Population state
        self.parent_population: List[Tuple[str, List[str]]] = []
        self.evaluated_parents: List[Dict] = []
        self.generation = 0
        
        # Migration tracking
        self.migration_sent = 0
        self.migration_received = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the island's population."""
        self.parent_population = pop.initialize_population(
            self.rng, self.all_signal_ids, self.population_size
        )
        # Deduplicate
        self.parent_population = nsga._dedup_individuals(self.parent_population)
    
    def evolve_generation(self, signals_df: pd.DataFrame, signals_metadata: List[Dict],
                         master_df: pd.DataFrame, exit_policy: Optional[Dict]) -> IslandMetrics:
        """Evolve one generation on this island."""
        start_time = time.time()
        
        # Evaluate parents
        self.evaluated_parents = self._evaluate_population(
            self.parent_population, signals_df, signals_metadata, master_df, exit_policy
        )
        
        # Generate children through crossover and mutation
        children_population = self._generate_children()
        
        # Evaluate children
        evaluated_children = self._evaluate_population(
            children_population, signals_df, signals_metadata, master_df, exit_policy
        )
        
        # Select survivors using NSGA-II
        self.parent_population = self._select_survivors(
            self.evaluated_parents + evaluated_children
        )
        
        self.generation += 1
        
        # Calculate metrics
        evaluation_time = time.time() - start_time
        metrics = self._calculate_metrics(evaluation_time)
        
        return metrics
    
    def _evaluate_population(self, population: List[Tuple[str, List[str]]],
                           signals_df: pd.DataFrame, signals_metadata: List[Dict],
                           master_df: pd.DataFrame, exit_policy: Optional[Dict]) -> List[Dict]:
        """Evaluate a population of individuals."""
        if not population:
            return []
        
        # Use cached evaluation for reproducibility
        evaluated = []
        for individual in population:
            eval_result = _evaluate_one_setup_cached(
                individual, signals_df, signals_metadata, master_df, exit_policy
            )
            # Add island tracking
            eval_result['island_id'] = self.island_id
            evaluated.append(eval_result)
        
        return evaluated
    
    def _generate_children(self) -> List[Tuple[str, List[str]]]:
        """Generate children through crossover and mutation."""
        children = []
        n_children = self.population_size
        
        for _ in range(n_children):
            # Tournament selection for parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            child = pop.crossover(parent1, parent2, self.rng)
            
            # Mutation
            if self.rng.random() < settings.ga.mutation_rate:
                child = pop.mutate(child, self.all_signal_ids, self.rng)
            
            children.append(child)
        
        # Deduplicate children
        return nsga._dedup_individuals(children)
    
    def _tournament_selection(self) -> Tuple[str, List[str]]:
        """Tournament selection for parent selection."""
        if not self.evaluated_parents:
            return self.parent_population[0] if self.parent_population else None
        
        tournament_size = min(3, len(self.evaluated_parents))
        candidates = self.rng.choice(self.evaluated_parents, size=tournament_size, replace=False)
        
        # Select best based on objectives (assuming maximization)
        best = max(candidates, key=lambda x: sum(x.get('objectives', [0])))
        return best['individual']
    
    def _select_survivors(self, combined_population: List[Dict]) -> List[Tuple[str, List[str]]]:
        """Select survivors using NSGA-II non-dominated sorting and crowding distance."""
        if not combined_population:
            return []
        
        # Non-dominated sorting
        fronts = nsga._non_dominated_sort(combined_population)
        
        # Select survivors up to population size
        survivors = []
        for front in fronts:
            nsga._calculate_crowding_distance(front)
            if len(survivors) + len(front) <= self.population_size:
                survivors.extend(front)
            else:
                # Sort by crowding distance and take what we need
                front.sort(key=lambda x: x.get('crowding_distance', 0), reverse=True)
                need = self.population_size - len(survivors)
                survivors.extend(front[:need])
                break
        
        # Deduplicate survivors
        uniq = []
        seen = set()
        for ind in survivors:
            key = nsga._dna(ind['individual'])
            if key not in seen:
                seen.add(key)
                uniq.append(ind)
        
        # Convert back to individual format
        return [ind['individual'] for ind in uniq]
    
    def _calculate_metrics(self, evaluation_time: float) -> IslandMetrics:
        """Calculate island performance metrics."""
        if not self.evaluated_parents:
            return IslandMetrics(
                island_id=self.island_id,
                generation=self.generation,
                population_size=len(self.parent_population),
                best_fitness=[0.0],
                avg_fitness=[0.0],
                diversity_metric=0.0,
                migration_sent=self.migration_sent,
                migration_received=self.migration_received,
                evaluation_time=evaluation_time
            )
        
        # Calculate fitness metrics
        objectives = [ind.get('objectives', [0.0]) for ind in self.evaluated_parents]
        if objectives:
            best_fitness = [max(obj[i] for obj in objectives) for i in range(len(objectives[0]))]
            avg_fitness = [np.mean([obj[i] for obj in objectives]) for i in range(len(objectives[0]))]
        else:
            best_fitness = [0.0]
            avg_fitness = [0.0]
        
        # Calculate diversity metric (unique DNA count)
        unique_dna = set(nsga._dna(ind['individual']) for ind in self.evaluated_parents)
        diversity_metric = len(unique_dna) / max(1, len(self.evaluated_parents))
        
        return IslandMetrics(
            island_id=self.island_id,
            generation=self.generation,
            population_size=len(self.parent_population),
            best_fitness=best_fitness,
            avg_fitness=avg_fitness,
            diversity_metric=diversity_metric,
            migration_sent=self.migration_sent,
            migration_received=self.migration_received,
            evaluation_time=evaluation_time
        )
    
    def get_migration_candidates(self, migration_size: int) -> List[Dict]:
        """Get the best individuals for migration."""
        if not self.evaluated_parents:
            return []
        
        # Sort by fitness (sum of objectives)
        sorted_individuals = sorted(
            self.evaluated_parents,
            key=lambda x: sum(x.get('objectives', [0])),
            reverse=True
        )
        
        return sorted_individuals[:migration_size]
    
    def receive_migrants(self, migrants: List[Dict], replace_strategy: str = "worst"):
        """Receive migrants from other islands."""
        if not migrants:
            return
        
        self.migration_received += len(migrants)
        
        if replace_strategy == "worst":
            # Replace worst individuals
            if not self.evaluated_parents:
                return
            
            # Sort by fitness (worst first)
            sorted_individuals = sorted(
                self.evaluated_parents,
                key=lambda x: sum(x.get('objectives', [0]))
            )
            
            # Replace worst individuals
            n_replace = min(len(migrants), len(sorted_individuals))
            for i in range(n_replace):
                # Find the individual in parent_population and replace it
                worst_ind = sorted_individuals[i]
                worst_individual = worst_ind['individual']
                
                if worst_individual in self.parent_population:
                    idx = self.parent_population.index(worst_individual)
                    migrant_individual = migrants[i]['individual']
                    self.parent_population[idx] = migrant_individual
        
        elif replace_strategy == "random":
            # Replace random individuals
            n_replace = min(len(migrants), len(self.parent_population))
            replace_indices = self.rng.choice(
                len(self.parent_population), size=n_replace, replace=False
            )
            
            for i, idx in enumerate(replace_indices):
                migrant_individual = migrants[i]['individual']
                self.parent_population[idx] = migrant_individual


class IslandManager:
    """Manages multiple islands and coordinates migration."""
    
    def __init__(self, signals_df: pd.DataFrame, signals_metadata: List[Dict], 
                 master_df: pd.DataFrame):
        self.signals_df = signals_df
        self.signals_metadata = signals_metadata
        self.master_df = master_df
        
        # Configuration
        self.config = settings.ga.islands
        self.n_islands = self.config.n_islands
        self.migration_interval = self.config.migration_interval
        self.migration_size = self.config.migration_size
        self.replace_strategy = self.config.replace_strategy
        
        # Calculate island population size
        if self.config.island_population_size is not None:
            self.island_pop_size = self.config.island_population_size
        else:
            self.island_pop_size = settings.ga.population_size // self.n_islands
        
        # Initialize islands
        self.islands: List[Island] = []
        self.all_signal_ids = list(signals_df.columns)
        self.exit_policy = _exit_policy_from_settings()
        
        self._initialize_islands()
        
        # Migration tracking
        self.migration_events: List[MigrationEvent] = []
        self.island_metrics_history: List[IslandMetrics] = []
    
    def _initialize_islands(self):
        """Initialize all islands."""
        for i in range(self.n_islands):
            island = Island(
                island_id=i,
                population_size=self.island_pop_size,
                all_signal_ids=self.all_signal_ids,
                base_seed=settings.ga.seed
            )
            self.islands.append(island)
    
    def evolve(self) -> List[Dict]:
        """Run the complete island model evolution."""
        total_generations = settings.ga.generations
        
        logging.info(f"Starting Island Model Evolution:")
        logging.info(f"  Islands: {self.n_islands}")
        logging.info(f"  Island Population Size: {self.island_pop_size}")
        logging.info(f"  Total Generations: {total_generations}")
        logging.info(f"  Migration Interval: {self.migration_interval}")
        logging.info(f"  Migration Size: {self.migration_size}")
        
        # Main evolution loop
        with tqdm(range(1, total_generations + 1), desc="Island Evolution", dynamic_ncols=True) as pbar:
            for generation in pbar:
                # Evolve all islands
                generation_metrics = []
                for island in self.islands:
                    metrics = island.evolve_generation(
                        self.signals_df, self.signals_metadata, 
                        self.master_df, self.exit_policy
                    )
                    generation_metrics.append(metrics)
                
                # Store metrics
                self.island_metrics_history.extend(generation_metrics)
                
                # Migration step
                if generation % self.migration_interval == 0:
                    self._perform_migration(generation)
                
                # Update progress bar
                if self.config.log_island_metrics:
                    self._log_generation_metrics(generation, generation_metrics)
                
                pbar.set_postfix({
                    'islands': f"{self.n_islands}",
                    'migrations': len(self.migration_events)
                })
        
        # Final synchronization
        if self.config.sync_final:
            return self._synchronize_final_population()
        else:
            return self._get_all_evaluated_individuals()
    
    def _perform_migration(self, generation: int):
        """Perform migration between islands."""
        if self.config.migration_topology == "ring":
            self._ring_migration(generation)
        elif self.config.migration_topology == "random":
            self._random_migration(generation)
        elif self.config.migration_topology == "all_to_all":
            self._all_to_all_migration(generation)
    
    def _ring_migration(self, generation: int):
        """Ring topology migration (island i -> island i+1)."""
        migration_size = max(1, int(self.island_pop_size * self.migration_size))
        
        for i in range(self.n_islands):
            source_island = self.islands[i]
            target_island = self.islands[(i + 1) % self.n_islands]
            
            # Get migrants from source
            migrants = source_island.get_migration_candidates(migration_size)
            if not migrants:
                continue
            
            # Send to target
            target_island.receive_migrants(migrants, self.replace_strategy)
            
            # Track migration
            migration_event = MigrationEvent(
                from_island=i,
                to_island=(i + 1) % self.n_islands,
                individuals=migrants,
                generation=generation
            )
            self.migration_events.append(migration_event)
            
            source_island.migration_sent += len(migrants)
    
    def _random_migration(self, generation: int):
        """Random topology migration."""
        migration_size = max(1, int(self.island_pop_size * self.migration_size))
        
        for source_island in self.islands:
            migrants = source_island.get_migration_candidates(migration_size)
            if not migrants:
                continue
            
            # Randomly select target island (excluding self)
            available_targets = [i for i in range(self.n_islands) if i != source_island.island_id]
            if not available_targets:
                continue
            
            target_idx = np.random.choice(available_targets)
            target_island = self.islands[target_idx]
            
            # Send migrants
            target_island.receive_migrants(migrants, self.replace_strategy)
            
            # Track migration
            migration_event = MigrationEvent(
                from_island=source_island.island_id,
                to_island=target_idx,
                individuals=migrants,
                generation=generation
            )
            self.migration_events.append(migration_event)
            
            source_island.migration_sent += len(migrants)
    
    def _all_to_all_migration(self, generation: int):
        """All-to-all topology migration."""
        migration_size = max(1, int(self.island_pop_size * self.migration_size))
        
        # Collect migrants from all islands
        all_migrants = []
        for island in self.islands:
            migrants = island.get_migration_candidates(migration_size)
            all_migrants.extend(migrants)
            island.migration_sent += len(migrants)
        
        # Distribute migrants to all islands
        migrants_per_island = len(all_migrants) // self.n_islands
        for i, island in enumerate(self.islands):
            start_idx = i * migrants_per_island
            end_idx = start_idx + migrants_per_island
            island_migrants = all_migrants[start_idx:end_idx]
            
            island.receive_migrants(island_migrants, self.replace_strategy)
            
            # Track migration
            migration_event = MigrationEvent(
                from_island=-1,  # From all islands
                to_island=i,
                individuals=island_migrants,
                generation=generation
            )
            self.migration_events.append(migration_event)
    
    def _synchronize_final_population(self) -> List[Dict]:
        """Synchronize all islands into a final population."""
        # Collect all evaluated individuals from all islands
        all_individuals = self._get_all_evaluated_individuals()
        
        # Apply final NSGA-II selection
        fronts = nsga._non_dominated_sort(all_individuals)
        
        # Select final survivors
        final_population = []
        for front in fronts:
            nsga._calculate_crowding_distance(front)
            if len(final_population) + len(front) <= settings.ga.population_size:
                final_population.extend(front)
            else:
                front.sort(key=lambda x: x.get('crowding_distance', 0), reverse=True)
                need = settings.ga.population_size - len(final_population)
                final_population.extend(front[:need])
                break
        
        return final_population
    
    def _get_all_evaluated_individuals(self) -> List[Dict]:
        """Get all evaluated individuals from all islands."""
        all_individuals = []
        for island in self.islands:
            all_individuals.extend(island.evaluated_parents)
        return all_individuals
    
    def _log_generation_metrics(self, generation: int, metrics: List[IslandMetrics]):
        """Log metrics for the current generation."""
        if not self.config.log_island_metrics:
            return
        
        logging.info(f"Generation {generation} - Island Metrics:")
        for metrics in metrics:
            logging.info(f"  Island {metrics.island_id}: "
                        f"pop={metrics.population_size}, "
                        f"best_fitness={metrics.best_fitness}, "
                        f"diversity={metrics.diversity_metric:.3f}, "
                        f"migrations_sent={metrics.migration_sent}, "
                        f"migrations_received={metrics.migration_received}")
    
    def get_migration_summary(self) -> Dict[str, Any]:
        """Get summary of migration events."""
        if not self.migration_events:
            return {"total_migrations": 0, "migration_pattern": "none"}
        
        migration_counts = defaultdict(int)
        for event in self.migration_events:
            key = f"{event.from_island}->{event.to_island}"
            migration_counts[key] += 1
        
        return {
            "total_migrations": len(self.migration_events),
            "migration_pattern": dict(migration_counts),
            "migration_interval": self.migration_interval,
            "migration_size": self.migration_size
        }
    
    def get_island_diversity_metrics(self) -> Dict[str, Any]:
        """Get diversity metrics across all islands."""
        if not self.island_metrics_history:
            return {"diversity_metrics": "no_data"}
        
        # Calculate diversity over time
        diversity_by_generation = defaultdict(list)
        for metrics in self.island_metrics_history:
            diversity_by_generation[metrics.generation].append(metrics.diversity_metric)
        
        avg_diversity = {
            gen: np.mean(divs) for gen, divs in diversity_by_generation.items()
        }
        
        return {
            "diversity_metrics": avg_diversity,
            "final_diversity": avg_diversity.get(max(avg_diversity.keys()), 0.0),
            "island_count": self.n_islands
        }
