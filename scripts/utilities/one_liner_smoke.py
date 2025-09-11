#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke test for one-liner reporting functionality
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from alpha_discovery.ga.fitness_loader import compute_metric_dict
from alpha_discovery.reporting.one_liner import format_one_liner, format_compact_summary

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

def test_one_liner_formats():
    """Test different one-liner formats"""
    print("=== One-Liner Smoke Test ===\n")
    
    # Create test data
    ledger = create_test_ledger()
    print(f"Test ledger created with {len(ledger)} trades")
    print(f"Total PnL: ${ledger['pnl_dollars'].sum():,.2f}")
    print(f"Win rate: {(ledger['pnl_dollars'] > 0).mean():.1%}")
    print()
    
    # Compute metrics
    metrics = compute_metric_dict(ledger)
    
    # Test basic one-liner
    print("=== Basic One-Liner ===")
    one_liner = format_one_liner(metrics)
    print(one_liner)
    print()
    
    # Test compact summary with setup ID
    print("=== Compact Summary with Setup ID ===")
    compact_summary = format_compact_summary(metrics, "SETUP_0001")
    print(compact_summary)
    print()
    
    # Test edge cases
    print("=== Edge Case Testing ===")
    
    # All positive returns
    positive_ledger = ledger.copy()
    positive_ledger['pnl_dollars'] = np.abs(positive_ledger['pnl_dollars'])
    positive_metrics = compute_metric_dict(positive_ledger)
    positive_line = format_one_liner(positive_metrics)
    print(f"All positive: {positive_line}")
    
    # All negative returns
    negative_ledger = ledger.copy()
    negative_ledger['pnl_dollars'] = -np.abs(negative_ledger['pnl_dollars'])
    negative_metrics = compute_metric_dict(negative_ledger)
    negative_line = format_one_liner(negative_metrics)
    print(f"All negative: {negative_line}")
    
    # Very small sample
    small_ledger = ledger.head(3)
    small_metrics = compute_metric_dict(small_ledger)
    small_line = format_one_liner(small_metrics)
    print(f"Small sample: {small_line}")
    
    print()
    print("=== Test Complete ===")
    print("If you see formatted lines above, the one-liner system is working!")

if __name__ == "__main__":
    test_one_liner_formats()
