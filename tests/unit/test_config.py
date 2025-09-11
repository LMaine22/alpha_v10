"""
Unit tests for configuration module.
"""
import pytest
from alpha_discovery.config import settings, GaConfig, DataConfig


class TestConfig:
    """Test cases for configuration system."""
    
    def test_settings_instantiation(self):
        """Test that settings can be instantiated."""
        assert settings is not None
        assert hasattr(settings, 'ga')
        assert hasattr(settings, 'data')
        assert hasattr(settings, 'options')
    
    def test_ga_config_defaults(self):
        """Test GA configuration defaults."""
        ga_config = GaConfig()
        
        assert ga_config.population_size > 0
        assert ga_config.generations > 0
        assert ga_config.mutation_rate >= 0
        assert ga_config.mutation_rate <= 1
        assert ga_config.elitism_rate >= 0
        assert ga_config.elitism_rate <= 1
    
    def test_data_config_defaults(self):
        """Test data configuration defaults."""
        data_config = DataConfig()
        
        assert data_config.start_date is not None
        assert data_config.end_date is not None
        assert data_config.start_date < data_config.end_date
        assert len(data_config.tradable_tickers) > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid population size
        with pytest.raises(ValueError):
            GaConfig(population_size=-1)
        
        # Test invalid mutation rate
        with pytest.raises(ValueError):
            GaConfig(mutation_rate=1.5)
