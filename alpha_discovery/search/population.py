# alpha_discovery/search/population.py
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import re
import math
import numpy as np

from ..config import settings

Individual = Tuple[str, List[str]]  # (ticker, setup_signals)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to map signals → ticker “keys”
# ──────────────────────────────────────────────────────────────────────────────
def _ticker_key(t: str) -> str:
    """
    Produce a compact key for matching a ticker inside signal ids.
    Example: "TSLA US Equity" -> "TSLA"
    """
    if not t:
        return ""
    # First alnum token is a safe default
    m = re.match(r"([A-Za-z0-9]+)", t)
    return m.group(1) if m else t.split()[0]


def _build_signal_index(signal_ids: List[str]) -> Dict[str, List[str]]:
    """
    Build {ticker_key -> [signal_ids]} plus a special '__GLOBAL__' bucket
    for signals that don't match any known ticker key (macro/cross-asset).
    """
    tickers = list(getattr(getattr(settings, "data", object()), "tradable_tickers", []))
    keys = [_ticker_key(t) for t in tickers if t]
    keyset = {k for k in keys if k}

    per_key: Dict[str, List[str]] = {k: [] for k in keyset}
    global_bucket: List[str] = []

    # Precompile patterns for speed & flexibility
    pat_for_key: Dict[str, re.Pattern] = {}
    for k in keyset:
        # Match common embeddings of the ticker key in a signal id
        # e.g., "TSLA_", "|TSLA|", "/TSLA/", "TSLA " or start/end
        pat = re.compile(
            rf"(^|[^\w]){re.escape(k)}([^\w]|$)|{re.escape(k)}[_:/|]",
            re.IGNORECASE
        )
        pat_for_key[k] = pat

    for sid in signal_ids:
        matched = False
        for k, pat in pat_for_key.items():
            if pat.search(sid):
                per_key[k].append(sid)
                matched = True
        if not matched:
            global_bucket.append(sid)

    per_key["__GLOBAL__"] = global_bucket
    return per_key


# ──────────────────────────────────────────────────────────────────────────────
# Individual constructors
# ──────────────────────────────────────────────────────────────────────────────
def _sample_setup_for_ticker(
    rng: np.random.Generator,
    per_key: Dict[str, List[str]],
    tk_key: str,
    k: int,
    p_macro: float = 0.20,
) -> List[str]:
    """Sample a setup of length k for a given ticker key, with optional macro mix."""
    # Candidate pools
    local_pool = per_key.get(tk_key, []) or []
    macro_pool = per_key.get("__GLOBAL__", []) or []

    # Fallbacks: if local pool is empty, pull from the global pool; if both empty, raise
    if not local_pool and not macro_pool:
        raise RuntimeError("No signals available to sample from.")
    if not local_pool:
        local_pool = macro_pool

    # Decide how many macro signals to mix in
    m = 0
    for _ in range(k):
        if rng.random() < p_macro:
            m += 1
    m = min(m, k)
    l = k - m  # local count

    # Deduplicate picks; if pools too small, allow replacement
    replace_local = len(local_pool) < l
    replace_macro = len(macro_pool) < m

    chosen_local = rng.choice(local_pool, size=l, replace=replace_local).tolist() if l > 0 else []
    chosen_macro = rng.choice(macro_pool, size=m, replace=replace_macro).tolist() if m > 0 else []

    setup = list(dict.fromkeys(chosen_local + chosen_macro))  # unique order-preserving
    # If uniqueness shrank us below k, top up from local pool
    while len(setup) < k and local_pool:
        cand = rng.choice(local_pool)
        if cand not in setup:
            setup.append(str(cand))
    return [str(s) for s in setup[:k]]


def _choose_ticker(rng: np.random.Generator) -> str:
    tickers = list(getattr(getattr(settings, "data", object()), "tradable_tickers", []))
    if not tickers:
        # Safe default
        return "SPY US Equity"
    return str(rng.choice(tickers))


def _choose_setup_len(rng: np.random.Generator) -> int:
    L = list(getattr(getattr(settings, "ga", object()), "setup_lengths_to_explore", [1, 2]))
    if not L:
        L = [1, 2]
    # Bias slightly toward longer end to help early coverage
    w = np.array([1.0 + 0.15 * i for i, _ in enumerate(L)], dtype=float)
    w = w / w.sum()
    return int(rng.choice(L, p=w))


# ──────────────────────────────────────────────────────────────────────────────
# Public API used by nsga.py
# ──────────────────────────────────────────────────────────────────────────────
def initialize_population(
    rng: np.random.Generator,
    signal_ids: List[str],
    size: Optional[int] = None,
) -> List[Individual]:
    """
    Build an initial population of (ticker, setup) individuals.
    - If `size` is provided (recommended), returns exactly that many.
    - Otherwise, uses settings.ga.population_size.
    Signals are chosen to match the selected ticker (heuristic), with a small
    chance of mixing 'global/macro' signals.
    """
    target = int(size) if size is not None else int(getattr(getattr(settings, "ga", object()), "population_size", 64))
    target = max(2, target)

    per_key = _build_signal_index([str(s) for s in signal_ids])
    pop_list: List[Individual] = []

    for _ in range(target):
        ticker = _choose_ticker(rng)
        k = _choose_setup_len(rng)
        tk_key = _ticker_key(ticker)

        try:
            setup = _sample_setup_for_ticker(rng, per_key, tk_key, k, p_macro=0.20)
        except Exception:
            # Extremely defensive fallback: sample any signals
            pool = per_key.get(tk_key, []) or per_key.get("__GLOBAL__", []) or list(signal_ids)
            replace = len(pool) < k
            setup = [str(s) for s in rng.choice(pool, size=k, replace=replace).tolist()]

        pop_list.append((ticker, setup))

    if int(getattr(getattr(settings, "ga", object()), "verbose", 1)) >= 2:
        print(f"Initializing specialized population of size {len(pop_list)}...")

    return pop_list


# ──────────────────────────────────────────────────────────────────────────────
# Genetic operators
# ──────────────────────────────────────────────────────────────────────────────
def _bounded_len_after_mutation(
    rng: np.random.Generator,
    current_len: int,
    min_len: int = 1,
    max_len: Optional[int] = None,
) -> int:
    """
    Decide the new target length after add/removal mutation.
    Bias toward staying near settings.ga.complexity.preferred.
    """
    pref = int(getattr(getattr(getattr(settings, "ga", object()), "complexity", object()), "preferred", 6))
    if max_len is None:
        max_len = max(pref + 4, 8)

    # Choose -1 / 0 / +1 with a bias toward 0 if current_len ~ pref
    delta_choices = np.array([-1, 0, +1], dtype=int)
    # gaussian-ish preference centered at pref
    d = abs(current_len - pref)
    p0 = 0.55 if d <= 1 else 0.40
    p = np.array([(1 - p0) / 2, p0, (1 - p0) / 2], dtype=float)
    new_len = int(np.clip(current_len + int(rng.choice(delta_choices, p=p)), min_len, max_len))
    return new_len


def mutate(
    individual: Individual,
    all_signal_ids: List[str],
    rng: np.random.Generator,
    p_ticker_switch: float = 0.05,
    p_replace: float = 0.30,
    p_add_or_remove: float = 0.65,
) -> Individual:
    """
    Mutations:
      - With small prob, switch ticker and re-sample a compatible setup of same length.
      - Otherwise, replace one signal; or add/remove to move toward preferred length.
      - Small chance to introduce a macro/global signal.
    """
    ticker, setup = individual
    setup = list(dict.fromkeys(setup))  # ensure unique

    per_key = _build_signal_index([str(s) for s in all_signal_ids])
    tk_key = _ticker_key(ticker)

    # 1) Occasionally switch ticker
    if rng.random() < p_ticker_switch:
        new_ticker = _choose_ticker(rng)
        new_len = len(setup)
        new_key = _ticker_key(new_ticker)
        try:
            new_setup = _sample_setup_for_ticker(rng, per_key, new_key, new_len, p_macro=0.25)
        except Exception:
            pool = per_key.get(new_key, []) or per_key.get("__GLOBAL__", []) or list(all_signal_ids)
            replace = len(pool) < new_len
            new_setup = [str(s) for s in rng.choice(pool, size=new_len, replace=replace).tolist()]
        return (new_ticker, new_setup)

    # 2) Within-ticker edits
    if rng.random() < p_replace and len(setup) > 0:
        # replace one element
        idx = int(rng.integers(0, len(setup)))
        pool = per_key.get(tk_key, []) or per_key.get("__GLOBAL__", []) or list(all_signal_ids)
        cand = str(rng.choice(pool))
        if cand not in setup:
            setup[idx] = cand
        return (ticker, setup)

    if rng.random() < p_add_or_remove:
        # Nudge length toward preferred
        new_len = _bounded_len_after_mutation(rng, len(setup))
        if new_len > len(setup):
            # add one (prefer local; allow macro)
            pool_local = per_key.get(tk_key, []) or []
            pool_macro = per_key.get("__GLOBAL__", []) or []
            pool = pool_local + pool_macro if (pool_local and pool_macro) else (pool_local or pool_macro or list(all_signal_ids))
            # sample until unique or give up after a few tries
            for _ in range(5):
                cand = str(rng.choice(pool))
                if cand not in setup:
                    setup.append(cand)
                    break
        elif new_len < len(setup) and len(setup) > 1:
            # remove one
            idx = int(rng.integers(0, len(setup)))
            del setup[idx]
        return (ticker, setup)

    # otherwise: no-op slight shuffle
    if len(setup) > 1:
        i, j = int(rng.integers(0, len(setup))), int(rng.integers(0, len(setup)))
        if i != j:
            setup[i], setup[j] = setup[j], setup[i]
    return (ticker, setup)


def crossover(p1: Individual, p2: Individual, rng: np.random.Generator) -> Individual:
    """
    Uniform set crossover on signals; ticker picked from one parent.
    Small chance to keep both parents' tickers by sampling the one whose
    signals dominate the child.
    """
    t1, s1 = p1
    t2, s2 = p2

    # Signal pool = union; sample child length near parents' average
    pool = list(dict.fromkeys(list(s1) + list(s2)))
    if not pool:
        # degenerate fallback
        return p1 if rng.random() < 0.5 else p2

    # Choose child len with small bias toward parent avg
    avg_len = max(1, int(round((len(s1) + len(s2)) / 2)))
    child_len = int(np.clip(avg_len + int(rng.integers(-1, 2)), 1, max(8, avg_len + 2)))

    replace = len(pool) < child_len
    child_setup = [str(s) for s in rng.choice(pool, size=child_len, replace=replace).tolist()]
    child_setup = list(dict.fromkeys(child_setup))  # unique
    if len(child_setup) == 0:
        child_setup = [str(rng.choice(pool))]

    # Ticker decision: pick parent whose signals contribute more to the child
    overlap1 = len(set(set(child_setup)) & set(s1))
    overlap2 = len(set(set(child_setup)) & set(s2))
    if overlap1 > overlap2:
        child_ticker = t1
    elif overlap2 > overlap1:
        child_ticker = t2
    else:
        child_ticker = t1 if rng.random() < 0.5 else t2

    return (child_ticker, child_setup)
