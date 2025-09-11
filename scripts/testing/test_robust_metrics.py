#!/usr/bin/env python3
"""
Test script for the new robust metrics system
"""

import pandas as pd
import numpy as np
from alpha_discovery.eval.robust_metrics import compute_all
from alpha_discovery.ga.fitness_loader import compute_fitness_with_profile, get_fitness_profile

def create_test_ledger():
    """Create a test ledger with realistic options trading data"""
    np.random.seed(42)
    
    # Simulate 20 trades with varying PnL and capital allocation
    n_trades = 20
    
    # Generate realistic PnL distribution (some wins, some losses)
    pnl_dollars = np.random.normal(2000, 5000, n_trades)  # Mean $2k, std $5k
    pnl_dollars = np.clip(pnl_dollars, -8000, 15000)  # Cap extreme values
    
    # Generate capital allocation (varies by trade)
    capital_allocated = np.random.uniform(8000, 15000, n_trades)
    
    # Create ledger DataFrame
    ledger = pd.DataFrame({
        'pnl_dollars': pnl_dollars,
        'capital_allocated': capital_allocated,
        'setup_id': 'TEST_SETUP_001',
        'ticker': 'SPY US Equity',
        'direction': 'long',
        'entry_date': pd.date_range('2024-01-01', periods=n_trades, freq='D'),
        'exit_date': pd.date_range('2024-01-02', periods=n_trades, freq='D'),
    })
    
    return ledger

def test_robust_metrics():
    """Test the robust metrics calculation"""
    print("=== Testing Robust Metrics System ===\n")
    
    # Create test data
    ledger = create_test_ledger()
    print(f"Test ledger created with {len(ledger)} trades")
    print(f"Total PnL: ${ledger['pnl_dollars'].sum():,.2f}")
    print(f"Win rate: {(ledger['pnl_dollars'] > 0).mean():.1%}")
    print(f"Average win: ${ledger[ledger['pnl_dollars'] > 0]['pnl_dollars'].mean():,.2f}")
    print(f"Average loss: ${ledger[ledger['pnl_dollars'] < 0]['pnl_dollars'].mean():,.2f}")
    print()
    
    # Test robust metrics
    print("=== Robust Metrics ===")
    metrics = compute_all(ledger)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:12.4f}")
        else:
            print(f"{key:20s}: {value}")
    
    print()
    
    # Test fitness profiles
    print("=== Fitness Profiles ===")
    profiles = ["safe", "balanced", "aggressive", "quality", "legacy"]
    
    for profile in profiles:
        objectives = get_fitness_profile(profile)
        fitness = compute_fitness_with_profile(ledger, profile)
        print(f"{profile:12s}: {objectives}")
        print(f"{'':12s}  Values: {[f'{v:.4f}' for v in fitness]}")
        print()
    
    # Test individual metrics
    print("=== Individual Metric Tests ===")
    
    # Test with all positive returns (should handle gracefully)
    positive_ledger = ledger.copy()
    positive_ledger['pnl_dollars'] = np.abs(positive_ledger['pnl_dollars'])  # Make all positive
    positive_metrics = compute_all(positive_ledger)
    print(f"All positive returns - GPR: {positive_metrics['GPR']:.4f}")
    print(f"All positive returns - RET_over_CVaR5: {positive_metrics['RET_over_CVaR5']:.4f}")
    
    # Test with all negative returns
    negative_ledger = ledger.copy()
    negative_ledger['pnl_dollars'] = -np.abs(negative_ledger['pnl_dollars'])  # Make all negative
    negative_metrics = compute_all(negative_ledger)
    print(f"All negative returns - GPR: {negative_metrics['GPR']:.4f}")
    print(f"All negative returns - RET_over_CVaR5: {negative_metrics['RET_over_CVaR5']:.4f}")
    
    # Test clamps and fallbacks
    print("\n=== Clamps and Fallbacks Test ===")
    from alpha_discovery.ga.fitness_loader import compute_metric_dict
    
    # Test with extreme values
    extreme_ledger = ledger.copy()
    extreme_ledger['pnl_dollars'] = [1000000] * len(extreme_ledger)  # All huge wins
    extreme_metrics = compute_metric_dict(extreme_ledger)
    print(f"Extreme wins - GPR (should be clamped): {extreme_metrics['GPR']:.4f}")
    print(f"Extreme wins - All metrics finite: {all(np.isfinite(list(extreme_metrics.values())))}")
    
    # Test one-liner format
    print("\n=== One-Liner Format Test ===")
    from alpha_discovery.reporting.one_liner import format_one_liner, format_compact_summary
    
    test_line = format_one_liner(metrics)
    print(f"One-liner: {test_line}")
    
    compact_line = format_compact_summary(metrics, "SETUP_0001")
    print(f"Compact: {compact_line}")

if __name__ == "__main__":
    test_robust_metrics()
