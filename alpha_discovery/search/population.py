# alpha_discovery/search/population.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, NamedTuple, Any
from tqdm import tqdm

from ..config import settings
from ..eval.selection import get_valid_trigger_dates


class EnhancedIndividual(NamedTuple):
    """Individual with horizon discovery - includes ticker, signals, and horizon."""
    ticker: str
    signals: List[str]
    horizon: int

# --- New Configuration for Ticker Specialization ---
# Of all mutations, what percentage should change the ticker vs. a signal?
TICKER_MUTATION_PROB = 0.15 # 15% chance to mutate the ticker, 85% to mutate a signal


def _meets_support_requirements(
    individual: Tuple[str, List[str]], 
    signals_df: pd.DataFrame, 
    min_support: Optional[int] = None
) -> bool:
    """Check if an individual meets minimum support requirements."""
    if min_support is None:
        min_support = settings.validation.min_support
    
    ticker, setup = individual
    if not setup:
        return False
    
    try:
        valid_dates = get_valid_trigger_dates(signals_df, setup, min_support)
        return len(valid_dates) >= min_support
    except Exception:
        return False


def _meets_support_requirements_enhanced(
    individual: EnhancedIndividual, 
    signals_df: pd.DataFrame, 
    min_support: Optional[int] = None
) -> bool:
    """Check if an enhanced individual meets minimum support requirements."""
    return _meets_support_requirements((individual.ticker, individual.signals), signals_df, min_support)


def initialize_population(
        rng: np.random.Generator,
        all_signal_ids: List[str],
        signals_df: Optional[pd.DataFrame] = None,
        population_size: Optional[int] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Creates the initial random population of specialized setups.
    Each individual is now a tuple: (ticker, [signal_ids]).
    """
    pop_size = population_size or settings.ga.population_size
    print(f"Initializing specialized population of size {pop_size}...")
    population: List[Tuple[str, List[str]]] = []
    
    # Use effective tradable tickers (respects single ticker mode)
    tradable_tickers = settings.data.effective_tradable_tickers
    if settings.data.single_ticker_mode:
        print(f"Single ticker mode enabled: only creating setups for {settings.data.single_ticker_mode}")
    
    if not tradable_tickers:
        raise ValueError("Cannot initialize population: no tradable tickers available.")

    # Use a set to ensure we don't create duplicate setups in the first generation
    seen_dna = set()
    max_attempts = pop_size * 10  # Prevent infinite loops
    attempts = 0

    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1
        
        # 1. Randomly choose a ticker for this individual
        ticker = rng.choice(tradable_tickers)

        # 2. Randomly choose a length for this setup (e.g., 2 or 3 signals)
        length = rng.choice(settings.ga.setup_lengths_to_explore)

        # 3. Randomly choose signal IDs without replacement
        setup = list(rng.choice(all_signal_ids, size=length, replace=False))

        # 4. Create a canonical representation (DNA) including ticker
        dna = (str(ticker), tuple(sorted(setup)))

        if dna not in seen_dna:
            seen_dna.add(dna)
            
            # 5. Check support requirements if signals_df is provided
            if signals_df is not None:
                if _meets_support_requirements((ticker, setup), signals_df):
                    population.append((ticker, setup))
            else:
                # If no signals_df provided, skip support filtering
                population.append((ticker, setup))

    if len(population) < pop_size:
        print(f"Warning: Only generated {len(population)}/{pop_size} individuals meeting support requirements")
    
    return population


def initialize_population_with_horizons(
    rng: np.random.Generator,
    all_signal_ids: List[str],
    signals_df: Optional[pd.DataFrame] = None,
    population_size: Optional[int] = None,
) -> List[EnhancedIndividual]:
    """Initialize population with horizon discovery."""
    pop_size = population_size or settings.ga.population_size
    print(f"Initializing horizon-discovery population of size {pop_size}...")
    
    population = []
    tradable_tickers = settings.data.effective_tradable_tickers
    available_horizons = settings.forecast.horizons  # [4, 10, 21] etc.
    
    if not tradable_tickers:
        raise ValueError("Cannot initialize population: no tradable tickers available.")
    
    if not available_horizons:
        raise ValueError("Cannot initialize population: no horizons available.")
    
    # Use a set to ensure we don't create duplicate DNA in the first generation
    seen_dna = set()
    max_attempts = pop_size * 15
    attempts = 0

    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1
        
        # Choose ticker, signals, AND horizon
        ticker = rng.choice(tradable_tickers)
        length = rng.choice(settings.ga.setup_lengths_to_explore)
        signals = list(rng.choice(all_signal_ids, size=length, replace=False))
        horizon = rng.choice(available_horizons)  # NEW: horizon discovery
        
        # DNA now includes horizon
        dna = (str(ticker), tuple(sorted(signals)), int(horizon))
        
        if dna not in seen_dna:
            seen_dna.add(dna)
            individual = EnhancedIndividual(ticker, signals, horizon)
            
            # Check support requirements
            if signals_df is not None:
                if _meets_support_requirements_enhanced(individual, signals_df):
                    population.append(individual)
            else:
                population.append(individual)

    if len(population) < pop_size:
        print(f"Warning: Only generated {len(population)}/{pop_size} horizon-discovery individuals meeting support requirements")
    
    print(f"Generated {len(population)} horizon-discovery individuals")
    return population


def crossover(
        parent1: Tuple[str, List[str]],
        parent2: Tuple[str, List[str]],
        rng: np.random.Generator,
        signals_df: Optional[pd.DataFrame] = None
) -> Tuple[str, List[str]]:
    """
    Combines two parent setups to create a new child setup.
    The child inherits its ticker from one parent and its signals from a pool of both.
    """
    ticker1, signals1 = parent1
    ticker2, signals2 = parent2

    # --- Crossover Ticker ---
    # In single ticker mode, child must use the single ticker
    if settings.data.single_ticker_mode:
        child_ticker = settings.data.single_ticker_mode
    else:
        # The child inherits the ticker from one of the parents randomly.
        child_ticker = ticker1 if rng.random() < 0.5 else ticker2

    # --- Crossover Signals ---
    # Pool all unique signals from both parents
    combined_pool = list(set(signals1) | set(signals2))

    # Choose a length for the child, defaulting to the length of the first parent
    child_length = rng.choice([len(signals1), len(signals2)])

    # Ensure we don't try to sample more signals than are in the pool
    if len(combined_pool) < child_length:
        child_length = len(combined_pool)

    if child_length == 0:
        return (child_ticker, [])

    # Create the child by sampling from the combined pool
    child_signals = list(rng.choice(combined_pool, size=child_length, replace=False))

    # Check support requirements if signals_df is provided
    if signals_df is not None:
        if not _meets_support_requirements((child_ticker, child_signals), signals_df):
            # If child doesn't meet support, try to create a valid variant
            max_attempts = 5
            for _ in range(max_attempts):
                # Try different signal combinations
                if len(combined_pool) > child_length:
                    child_signals = list(rng.choice(combined_pool, size=child_length, replace=False))
                else:
                    # Try different ticker
                    child_ticker = ticker2 if child_ticker == ticker1 else ticker1
                
                if _meets_support_requirements((child_ticker, child_signals), signals_df):
                    break
            else:
                # If still no valid child, return one of the parents
                return parent1 if rng.random() < 0.5 else parent2

    return (child_ticker, child_signals)


def mutate(
        individual: Tuple[str, List[str]],
        all_signal_ids: List[str],
        rng: np.random.Generator,
        signals_df: Optional[pd.DataFrame] = None
) -> Tuple[str, List[str]]:
    """
    Applies a random mutation to a specialized setup.

    With a probability defined by the mutation rate, this function will either:
    1. Mutate the Ticker: Change the individual to specialize on a different ticker.
    2. Mutate a Signal: Replace one signal in the setup with a new, random signal.
    """
    if rng.random() >= settings.ga.mutation_rate:
        return individual # No mutation occurs

    ticker, setup = individual
    
    # Use effective tradable tickers (respects single ticker mode)
    tradable_tickers = settings.data.effective_tradable_tickers

    # Decide whether to mutate the ticker or a signal
    if rng.random() < TICKER_MUTATION_PROB and len(tradable_tickers) > 1:
        # --- Mutate Ticker ---
        other_tickers = [t for t in tradable_tickers if t != ticker]
        if other_tickers:
            new_ticker = rng.choice(other_tickers)
            mutated_individual = (new_ticker, setup)
            
            # Check support requirements if signals_df is provided
            if signals_df is not None:
                if _meets_support_requirements(mutated_individual, signals_df):
                    return mutated_individual
                else:
                    # Try other tickers
                    for other_ticker in other_tickers[1:]:  # Skip the first one we tried
                        mutated_individual = (other_ticker, setup)
                        if _meets_support_requirements(mutated_individual, signals_df):
                            return mutated_individual
            else:
                return mutated_individual
    else:
        # --- Mutate Signal ---
        if len(setup) > 0:
            index_to_mutate = rng.integers(0, len(setup))
            current_signals = set(setup)
            potential_new_signals = [sig for sig in all_signal_ids if sig not in current_signals]

            if potential_new_signals:
                new_signal = rng.choice(potential_new_signals)
                mutated_setup = setup.copy()
                mutated_setup[index_to_mutate] = new_signal
                mutated_individual = (ticker, mutated_setup)
                
                # Check support requirements if signals_df is provided
                if signals_df is not None:
                    if _meets_support_requirements(mutated_individual, signals_df):
                        return mutated_individual
                    else:
                        # Try other signals
                        max_attempts = min(5, len(potential_new_signals) - 1)
                        for _ in range(max_attempts):
                            new_signal = rng.choice(potential_new_signals)
                            mutated_setup = setup.copy()
                            mutated_setup[index_to_mutate] = new_signal
                            mutated_individual = (ticker, mutated_setup)
                            if _meets_support_requirements(mutated_individual, signals_df):
                                return mutated_individual
                else:
                    return mutated_individual

    # If mutation was not possible or didn't meet support requirements, return original
    return individual


def crossover_with_horizons(
    parent1: EnhancedIndividual,
    parent2: EnhancedIndividual,
    rng: np.random.Generator,
    signals_df: Optional[pd.DataFrame] = None
) -> EnhancedIndividual:
    """Crossover that can mix horizons between parents."""
    
    # Ticker crossover (same logic)
    if settings.data.single_ticker_mode:
        child_ticker = settings.data.single_ticker_mode
    else:
        child_ticker = parent1.ticker if rng.random() < 0.5 else parent2.ticker
    
    # Signal crossover (same logic)
    combined_signals = list(set(parent1.signals) | set(parent2.signals))
    child_length = rng.choice([len(parent1.signals), len(parent2.signals)])
    child_length = min(child_length, len(combined_signals))
    
    if child_length == 0:
        child_signals = []
    else:
        child_signals = list(rng.choice(combined_signals, size=child_length, replace=False))
    
    # NEW: Horizon crossover (can inherit from either parent or be random)
    horizon_choice = rng.random()
    if horizon_choice < 0.4:
        child_horizon = parent1.horizon
    elif horizon_choice < 0.8:
        child_horizon = parent2.horizon  
    else:
        # 20% chance of random horizon (exploration)
        child_horizon = rng.choice(settings.forecast.horizons)
    
    child = EnhancedIndividual(child_ticker, child_signals, child_horizon)
    
    # Check support requirements if signals_df is provided
    if signals_df is not None:
        if not _meets_support_requirements_enhanced(child, signals_df):
            # If child doesn't meet support, try to create a valid variant
            max_attempts = 5
            for _ in range(max_attempts):
                # Try different signal combinations
                if len(combined_signals) > child_length:
                    child_signals = list(rng.choice(combined_signals, size=child_length, replace=False))
                    child = EnhancedIndividual(child_ticker, child_signals, child_horizon)
                else:
                    # Try different ticker
                    child_ticker = parent2.ticker if child_ticker == parent1.ticker else parent1.ticker
                    child = EnhancedIndividual(child_ticker, child_signals, child_horizon)
                
                if _meets_support_requirements_enhanced(child, signals_df):
                    break
            else:
                # If still no valid child, return one of the parents
                return parent1 if rng.random() < 0.5 else parent2

    return child


def mutate_with_horizons(
    individual: EnhancedIndividual,
    all_signal_ids: List[str],
    rng: np.random.Generator,
    signals_df: Optional[pd.DataFrame] = None
) -> EnhancedIndividual:
    """Mutation that can also mutate horizon."""
    if rng.random() >= settings.ga.mutation_rate:
        return individual
    
    # Choose what to mutate: ticker (15%), signals (70%), or horizon (15%)
    mutation_type = rng.choice(['ticker', 'signals', 'horizon'], p=[0.15, 0.70, 0.15])
    
    if mutation_type == 'horizon':
        # Mutate horizon
        other_horizons = [h for h in settings.forecast.horizons if h != individual.horizon]
        if other_horizons:
            new_horizon = rng.choice(other_horizons)
            mutated = EnhancedIndividual(individual.ticker, individual.signals, new_horizon)
            
            # Check support requirements (horizon change shouldn't affect support, but let's be safe)
            if signals_df is not None:
                if _meets_support_requirements_enhanced(mutated, signals_df):
                    return mutated
            else:
                return mutated
    
    elif mutation_type == 'ticker':
        # Existing ticker mutation logic
        tradable_tickers = settings.data.effective_tradable_tickers
        other_tickers = [t for t in tradable_tickers if t != individual.ticker]
        if other_tickers:
            new_ticker = rng.choice(other_tickers)
            mutated = EnhancedIndividual(new_ticker, individual.signals, individual.horizon)
            
            # Check support requirements
            if signals_df is not None:
                if _meets_support_requirements_enhanced(mutated, signals_df):
                    return mutated
                else:
                    # Try other tickers
                    for other_ticker in other_tickers[1:]:  # Skip the first one we tried
                        mutated = EnhancedIndividual(other_ticker, individual.signals, individual.horizon)
                        if _meets_support_requirements_enhanced(mutated, signals_df):
                            return mutated
            else:
                return mutated
    
    else:  # signals mutation
        # Existing signals mutation logic
        if len(individual.signals) > 0:
            idx = rng.integers(0, len(individual.signals))
            current_signals = set(individual.signals)
            potential_new = [s for s in all_signal_ids if s not in current_signals]
            
            if potential_new:
                new_signal = rng.choice(potential_new)
                new_signals = individual.signals.copy()
                new_signals[idx] = new_signal
                mutated = EnhancedIndividual(individual.ticker, new_signals, individual.horizon)
                
                # Check support requirements
                if signals_df is not None:
                    if _meets_support_requirements_enhanced(mutated, signals_df):
                        return mutated
                    else:
                        # Try other signals
                        max_attempts = min(5, len(potential_new) - 1)
                        for _ in range(max_attempts):
                            new_signal = rng.choice(potential_new)
                            new_signals = individual.signals.copy()
                            new_signals[idx] = new_signal
                            mutated = EnhancedIndividual(individual.ticker, new_signals, individual.horizon)
                            if _meets_support_requirements_enhanced(mutated, signals_df):
                                return mutated
                else:
                    return mutated
    
    return individual  # No mutation applied


def create_horizon_discovery_population(
    signals_df: pd.DataFrame,
    n_individuals: int,
    rng: np.random.Generator,
    exit_policy: Any 
) -> List[EnhancedIndividual]:
    """
    Creates a population of EnhancedIndividuals for horizon discovery mode using a more robust
    "generate-then-test" approach suitable for CPCV.
    """
    population = []
    
    # 1. Generate a large pool of potential candidates without immediate validation
    pool_size = n_individuals * 20  # Oversample to increase chances of finding valid individuals
    candidate_pool = []
    all_tickers = list(signals_df.columns.get_level_values(0).unique())
    all_signals = list(signals_df.columns.get_level_values(1).unique())

    for _ in range(pool_size):
        individual = _create_random_individual_no_validation(
            all_tickers, all_signals, settings.forecast.horizons, rng
        )
        if individual:
            candidate_pool.append(individual)

    # 2. Batch-validate the pool against the CV splits
    valid_individuals = []
    for individual in tqdm(candidate_pool, desc="Validating initial population", leave=False):
        total_triggers_in_cv = 0
        # Sum triggers across all test splits in the discovery CV
        for _, test_idx in exit_policy.splits:
            test_signals = signals_df.loc[test_idx]
            trigger_dates = get_valid_trigger_dates(test_signals, individual.signals, 0)
            total_triggers_in_cv += len(trigger_dates)

        if total_triggers_in_cv >= settings.validation.min_support:
            valid_individuals.append(individual)
            if len(valid_individuals) >= n_individuals:
                break # We have enough

    if len(valid_individuals) < n_individuals:
        print(f"Warning: Only generated {len(valid_individuals)}/{n_individuals} horizon-discovery individuals meeting support requirements")
    
    return valid_individuals[:n_individuals]


def _create_random_individual_no_validation(
    tickers: List[str],
    signals: List[str],
    horizons: List[int],
    rng: np.random.Generator
) -> Optional[EnhancedIndividual]:
    """Creates a single random individual without performing validation."""
    try:
        ticker = rng.choice(tickers)
        
        # Ensure signals are valid for the chosen ticker (conceptually)
        # This basic version just samples globally. A more advanced version might filter signals.
        n_signals = rng.integers(settings.ga.min_signals, settings.ga.max_signals + 1)
        setup_signals = tuple(rng.choice(signals, n_signals, replace=False))
        
        horizon = int(rng.choice(horizons))
        
        return EnhancedIndividual(ticker=ticker, signals=setup_signals, horizon=horizon)
    except Exception as e:
        print(f"Error creating random individual: {e}")
        return None
