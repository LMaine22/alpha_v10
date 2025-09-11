"""
Pytest configuration and shared fixtures for Alpha Discovery v10 tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path

# Project root for test data access
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def sample_bb_data():
    """Sample Bloomberg data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = ['AAPL US Equity', 'TSLA US Equity', 'MSFT US Equity']
    
    data = {}
    for ticker in tickers:
        data[f'{ticker}_PX_LAST'] = np.random.randn(100).cumsum() + 100
        data[f'{ticker}_VOLATILITY_90D'] = np.random.uniform(0.1, 0.5, 100)
        data[f'{ticker}_TURNOVER'] = np.random.uniform(1000000, 10000000, 100)
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_economic_data():
    """Sample economic releases data for testing."""
    return pd.DataFrame({
        'release_datetime': pd.date_range('2020-01-01', periods=50, freq='W'),
        'event_type': ['CPI', 'NFP', 'GDP'] * 17 + ['CPI'],
        'country': ['US'] * 50,
        'survey': np.random.uniform(0.1, 0.3, 50),
        'actual': np.random.uniform(0.1, 0.3, 50),
        'prior': np.random.uniform(0.1, 0.3, 50),
        'relevance': np.random.uniform(50, 100, 50),
        'bb_ticker': ['USCPI Index', 'USNFP Index', 'USGDP Index'] * 17 + ['USCPI Index']
    })


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'ga': {
            'population_size': 10,
            'generations': 2,
            'seed': 42
        },
        'data': {
            'start_date': '2020-01-01',
            'end_date': '2020-12-31'
        }
    }


@pytest.fixture
def temp_run_dir(tmp_path):
    """Temporary run directory for testing."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    return str(run_dir)
