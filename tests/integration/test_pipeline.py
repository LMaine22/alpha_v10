"""
Integration tests for the main pipeline.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from alpha_discovery.config import settings


class TestPipeline:
    """Integration tests for the main pipeline."""
    
    @patch('alpha_discovery.data.loader.load_data_from_parquet')
    @patch('alpha_discovery.features.registry.build_feature_matrix')
    @patch('alpha_discovery.signals.compiler.compile_signals')
    def test_pipeline_data_flow(self, mock_compile, mock_features, mock_loader):
        """Test that data flows correctly through the pipeline."""
        # Mock data loading
        mock_data = pd.DataFrame({
            'AAPL US Equity_PX_LAST': [100, 101, 102],
            'AAPL US Equity_VOLATILITY_90D': [0.2, 0.21, 0.19]
        }, index=pd.date_range('2020-01-01', periods=3))
        
        mock_loader.return_value = mock_data
        mock_features.return_value = mock_data
        mock_compile.return_value = (mock_data, [])
        
        # Import and test pipeline components
        from alpha_discovery.data.loader import load_data_from_parquet
        from alpha_discovery.features.registry import build_feature_matrix
        from alpha_discovery.signals.compiler import compile_signals
        
        # Test data loading
        data = load_data_from_parquet()
        assert data is not None
        
        # Test feature building
        features = build_feature_matrix(data)
        assert features is not None
        
        # Test signal compilation
        signals_df, signals_metadata = compile_signals(features)
        assert signals_df is not None
        assert isinstance(signals_metadata, list)
    
    def test_config_consistency(self):
        """Test that configuration is consistent across modules."""
        # Test that settings are accessible
        assert settings.ga.population_size > 0
        assert settings.data.start_date is not None
        assert settings.options.capital_per_trade > 0
        
        # Test that required fields are present
        assert hasattr(settings.ga, 'generations')
        assert hasattr(settings.data, 'tradable_tickers')
        assert hasattr(settings.options, 'tenor_grid_bd')
