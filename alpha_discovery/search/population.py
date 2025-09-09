# alpha_discovery/search/population.py

import numpy as np
from typing import List, Tuple, Optional

from ..config import settings

# --- New Configuration for Ticker Specialization ---
# Of all mutations, what percentage should change the ticker vs. a signal?
TICKER_MUTATION_PROB = 0.15 # 15% chance to mutate the ticker, 85% to mutate a signal


def initialize_population(
        rng: np.random.Generator,
        all_signal_ids: List[str],
        population_size: Optional[int] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Creates the initial random population of specialized setups.
    Each individual is now a tuple: (ticker, [signal_ids]).
    """
    pop_size = population_size or settings.ga.population_size
    print(f"Initializing specialized population of size {pop_size}...")
    population: List[Tuple[str, List[str]]] = []
    tradable_tickers = settings.data.tradable_tickers
    if not tradable_tickers:
        raise ValueError("Cannot initialize population: settings.data.tradable_tickers is empty.")

    # Use a set to ensure we don't create duplicate setups in the first generation
    seen_dna = set()

    while len(population) < pop_size:
        # 1. Randomly choose a ticker for this individual
        ticker = rng.choice(tradable_tickers)

        # 2. Randomly choose a length for this setup (e.g., 2 or 3 signals)
        length = rng.choice(settings.ga.setup_lengths_to_explore)

        # 3. Randomly choose signal IDs without replacement
        setup = list(rng.choice(all_signal_ids, size=length, replace=False))

        # 4. Create a canonical representation (DNA) - signals must be globally unique
        dna = tuple(sorted(setup))

        if dna not in seen_dna:
            seen_dna.add(dna)
            population.append((ticker, setup))

    return population


def crossover(
        parent1: Tuple[str, List[str]],
        parent2: Tuple[str, List[str]],
        rng: np.random.Generator
) -> Tuple[str, List[str]]:
    """
    Combines two parent setups to create a new child setup.
    The child inherits its ticker from one parent and its signals from a pool of both.
    """
    ticker1, signals1 = parent1
    ticker2, signals2 = parent2

    # --- Crossover Ticker ---
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

    return (child_ticker, child_signals)


def mutate(
        individual: Tuple[str, List[str]],
        all_signal_ids: List[str],
        rng: np.random.Generator
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
    tradable_tickers = settings.data.tradable_tickers

    # Decide whether to mutate the ticker or a signal
    if rng.random() < TICKER_MUTATION_PROB and len(tradable_tickers) > 1:
        # --- Mutate Ticker ---
        other_tickers = [t for t in tradable_tickers if t != ticker]
        if other_tickers:
            new_ticker = rng.choice(other_tickers)
            return (new_ticker, setup)
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
                return (ticker, mutated_setup)

    # If mutation was not possible (e.g., no other tickers/signals available), return original
    return individual
