# alpha_discovery/search/population.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from ..config import settings
from ..eval.selection import get_valid_trigger_dates

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
