"""Bootstrap methods for robustness testing."""

from __future__ import annotations
from typing import Callable, Tuple, Optional
import numpy as np
import pandas as pd


def stationary_bootstrap(
    data: pd.Series,
    block_size: int,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> pd.Series:
    """
    Stationary bootstrap (Politis & Romano 1994).
    
    Generates a bootstrap sample using geometric block lengths.
    Preserves time-series dependence while allowing varying block sizes.
    
    Args:
        data: Time series to bootstrap
        block_size: Average block length (controls dependence)
        n_samples: Number of samples to draw (defaults to len(data))
        rng: Random number generator (for reproducibility)
        
    Returns:
        Bootstrapped series with same length as input
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if n_samples is None:
        n_samples = len(data)
    
    n = len(data)
    p = 1.0 / block_size  # Probability of starting new block
    
    bootstrap_indices = []
    while len(bootstrap_indices) < n_samples:
        # Start a new block
        start_idx = rng.integers(0, n)
        bootstrap_indices.append(start_idx)
        
        # Continue block with probability (1-p)
        current_idx = start_idx
        while len(bootstrap_indices) < n_samples and rng.random() > p:
            current_idx = (current_idx + 1) % n
            bootstrap_indices.append(current_idx)
    
    bootstrap_indices = bootstrap_indices[:n_samples]
    return data.iloc[bootstrap_indices].reset_index(drop=True)


def heavy_block_bootstrap(
    data: pd.Series,
    block_size: int,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> pd.Series:
    """
    Block bootstrap for heavy-tailed distributions.
    
    Uses fixed-length overlapping blocks. More conservative than
    stationary bootstrap for capturing tail behavior.
    
    Args:
        data: Time series to bootstrap
        block_size: Fixed block length
        n_samples: Number of samples to draw (defaults to len(data))
        rng: Random number generator
        
    Returns:
        Bootstrapped series with requested length
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if n_samples is None:
        n_samples = len(data)
    
    n = len(data)
    if block_size >= n:
        # If block size >= data, just resample entire series
        return data.sample(n=n_samples, replace=True, random_state=rng.bit_generator.state["state"]["state"])
    
    # Calculate number of blocks needed
    n_blocks = int(np.ceil(n_samples / block_size))
    
    bootstrap_data = []
    for _ in range(n_blocks):
        # Sample a random starting point
        start = rng.integers(0, n - block_size + 1)
        block = data.iloc[start:start + block_size].values
        bootstrap_data.extend(block)
    
    # Trim to requested length
    bootstrap_data = bootstrap_data[:n_samples]
    return pd.Series(bootstrap_data)


def bootstrap_skill_delta(
    train_skill: pd.Series,
    test_skill: pd.Series,
    metric_fn: Callable[[pd.Series], float],
    block_size: int = 20,
    n_bootstrap: int = 1000,
    bootstrap_type: str = "stationary",
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    Bootstrap confidence interval for skill delta (test - train).
    
    Tests if test skill is significantly different from training skill
    using block bootstrap to preserve time-series structure.
    
    Args:
        train_skill: Skill metric series from training (e.g., log losses)
        test_skill: Skill metric series from test
        metric_fn: Function to aggregate series → scalar (e.g., np.mean)
        block_size: Bootstrap block size
        n_bootstrap: Number of bootstrap iterations
        bootstrap_type: "stationary" or "heavy" block bootstrap
        alpha: Significance level for CI (default 0.05 for 95% CI)
        seed: Random seed
        
    Returns:
        Tuple of (observed_delta, ci_lower, ci_upper, p_value)
        
    Example:
        >>> delta, ci_low, ci_high, p = bootstrap_skill_delta(
        ...     train_losses, test_losses, np.mean, block_size=21
        ... )
        >>> if ci_low > 0:
        ...     print("Test significantly worse than train (possible overfit)")
    """
    rng = np.random.default_rng(seed)
    
    # Observed delta
    train_metric = metric_fn(train_skill)
    test_metric = metric_fn(test_skill)
    observed_delta = test_metric - train_metric
    
    # Bootstrap function selector
    if bootstrap_type == "heavy":
        bootstrap_fn = heavy_block_bootstrap
    else:
        bootstrap_fn = stationary_bootstrap
    
    # Bootstrap both distributions
    boot_deltas = []
    for _ in range(n_bootstrap):
        boot_train = bootstrap_fn(train_skill, block_size, rng=rng)
        boot_test = bootstrap_fn(test_skill, block_size, rng=rng)
        
        boot_train_metric = metric_fn(boot_train)
        boot_test_metric = metric_fn(boot_test)
        
        boot_deltas.append(boot_test_metric - boot_train_metric)
    
    boot_deltas = np.array(boot_deltas)
    
    # Confidence interval (percentile method)
    ci_lower = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
    
    # Two-tailed p-value: proportion of bootstrap deltas as extreme as observed
    p_value = float(np.mean(np.abs(boot_deltas) >= np.abs(observed_delta)))
    
    return observed_delta, ci_lower, ci_upper, p_value
