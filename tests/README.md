# Tests Directory

This directory contains comprehensive tests for the Alpha Discovery v10 project.

## 📁 Directory Structure

### Unit Tests (`unit/`)
Tests for individual modules and functions:
- `test_metrics.py` - Tests for metrics calculation
- `test_config.py` - Tests for configuration system
- `test_ga_core.py` - Tests for GA core functionality
- `test_backtester.py` - Tests for backtesting engine

### Integration Tests (`integration/`)
Tests for component interactions and data flow:
- `test_pipeline.py` - Tests for main pipeline execution
- `test_gauntlet.py` - Tests for gauntlet validation
- `test_backtesting.py` - Tests for end-to-end backtesting

### Fixtures (`fixtures/`)
Test data and mock configurations:
- `sample_data/` - Sample datasets for testing
- `mock_configs/` - Mock configuration files

## 🚀 Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_metrics.py
```

### Run with Coverage
```bash
pytest --cov=alpha_discovery tests/
```

## 📝 Test Guidelines

### Writing Tests
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern
- Use fixtures for common test data
- Mock external dependencies appropriately

### Test Data
- Use `sample_bb_data` fixture for Bloomberg data
- Use `sample_economic_data` fixture for economic releases
- Use `sample_config` fixture for configuration testing
- Use `temp_run_dir` fixture for temporary directories

### Coverage Goals
- Aim for >80% code coverage
- Focus on critical business logic
- Test edge cases and error conditions
- Validate data integrity and transformations

## 🔧 Test Configuration

Tests use pytest with the following configuration:
- Fixtures defined in `conftest.py`
- Mock data in `fixtures/` directory
- Coverage reporting enabled
- Parallel execution where possible
