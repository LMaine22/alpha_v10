"""
Unit tests for metrics module.
"""
import pytest
import pandas as pd
import numpy as np
from alpha_discovery.eval.metrics import calculate_portfolio_metrics


class TestMetrics:
    """Test cases for portfolio metrics calculation."""
    
    def test_calculate_portfolio_metrics_basic(self):
        """Test basic metrics calculation with simple data."""
        # Create sample trade ledger
        ledger = pd.DataFrame({
            'pnl_dollars': [100, -50, 200, -25, 150],
            'capital_allocated': [1000, 1000, 1000, 1000, 1000],
            'trade_date': pd.date_range('2020-01-01', periods=5)
        })
        
        metrics = calculate_portfolio_metrics(ledger)
        
        # Basic assertions
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['total_return'] > 0  # Should be positive with this data
    
    def test_calculate_portfolio_metrics_empty(self):
        """Test metrics calculation with empty ledger."""
        ledger = pd.DataFrame(columns=['pnl_dollars', 'capital_allocated'])
        
        metrics = calculate_portfolio_metrics(ledger)
        
        # Should handle empty data gracefully
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    def test_calculate_portfolio_metrics_all_losses(self):
        """Test metrics calculation with all losing trades."""
        ledger = pd.DataFrame({
            'pnl_dollars': [-100, -50, -200, -25, -150],
            'capital_allocated': [1000, 1000, 1000, 1000, 1000],
            'trade_date': pd.date_range('2020-01-01', periods=5)
        })
        
        metrics = calculate_portfolio_metrics(ledger)
        
        # Should handle all losses
        assert metrics['total_return'] < 0
        assert 'max_drawdown' in metrics
