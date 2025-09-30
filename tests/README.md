# Test Suite - Alpha Discovery v10

Comprehensive test coverage for the forecast-first alpha discovery pipeline.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_splits_leakage.py         # PAWF/NPWF leakage prevention (Phase 1)
├── test_objectives.py             # Objective transforms and filtering (Phase 5)
├── test_adapters.py               # Feature adapters and subspace sampling (Phase 2)
├── test_calibration.py            # Calibration helpers (ECE/MCE/Brier/Isotonic/Platt) (Phase 6)
├── test_regime.py                 # GMM regime detection (Phase 1)
├── test_bootstrap.py              # Bootstrap methods for robustness (Phase 1)
├── test_reporting.py              # Eligibility reporting (Phase 7)
├── unit/                          # Unit tests for individual modules
│   ├── test_config.py
│   └── test_metrics.py
└── integration/                   # Integration tests
    ├── test_pipeline.py
    └── test_acceptance.py
```

## Test Coverage

### Phase 1: Splits Package (test_splits_leakage.py, test_regime.py, test_bootstrap.py)

**test_splits_leakage.py** - 18 tests
- ✅ PAWF time ordering (train < purge < embargo < test)
- ✅ PAWF no overlap between train/purge/embargo/test
- ✅ PAWF deterministic IDs with same seed
- ✅ NPWF nested fold validation (inner folds within outer train)
- ✅ NPWF no leakage from outer test into inner folds
- ✅ NPWF purge and embargo enforcement
- ✅ NPWF deterministic with same seed

**test_regime.py** - 12 tests
- ✅ GMM regime fitting with clear regimes
- ✅ Regime fitting with insufficient data (graceful failure)
- ✅ Regime fitting determinism with fixed seed
- ✅ Regime assignment to dataframe
- ✅ Regime forward-fill for missing dates
- ✅ Regime similarity (cosine, identical, orthogonal, opposite)
- ✅ RegimeModel predict and predict_proba

**test_bootstrap.py** - 10 tests
- ✅ Stationary bootstrap basic functionality
- ✅ Stationary bootstrap determinism
- ✅ Stationary bootstrap coverage
- ✅ Heavy-tailed block bootstrap
- ✅ Heavy block determinism
- ✅ Bootstrap skill delta (positive, no difference, deterministic)
- ✅ Bootstrap with different metrics (Brier, Log Loss)
- ✅ Bootstrap confidence intervals

### Phase 2: Adapters (test_adapters.py)

**test_adapters.py** - 15 tests
- ✅ FeatureAdapter lazy loading
- ✅ FeatureAdapter read-only behavior
- ✅ Feature listing (features and pairwise)
- ✅ Lookback calculation (simple, with pairwise, missing keys, empty)
- ✅ Subspace sampling (random, stratified, complementary)
- ✅ Subspace determinism (same seed = same results)
- ✅ Subspace coverage and validation
- ✅ FeatureSubspace dataclass immutability

### Phase 5: Objectives (test_objectives.py)

**test_objectives.py** - 10 tests
- ✅ Proper scoring rule identification
- ✅ Legacy P&L objective detection
- ✅ Objective filtering (proper scoring rules only)
- ✅ Recommended objectives
- ✅ Transform application (with/without legacy)
- ✅ Invalid objective handling

### Phase 6: Calibration (test_calibration.py)

**test_calibration.py** - 22 tests
- ✅ ECE calculation (perfect, poor, range, deterministic)
- ✅ MCE calculation (perfect, poor, MCE >= ECE)
- ✅ Brier Score (perfect, worst, range)
- ✅ Log Loss (perfect, random, clipping)
- ✅ Isotonic calibration (fit, apply, improvement)
- ✅ Platt scaling (fit, apply)
- ✅ PIT tests (well-calibrated, poorly-calibrated)
- ✅ Reliability curves (shape, perfect calibration)

### Phase 7: Reporting (test_reporting.py)

**test_reporting.py** - 9 tests
- ✅ Filtering logic (skill, calibration, drift gates)
- ✅ Report formatting
- ✅ Skill breakdown by ticker/horizon
- ✅ Summary statistics
- ✅ Full report generation (6 output files)
- ✅ Different threshold configurations

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_splits_leakage.py -v
pytest tests/test_calibration.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_adapters.py::TestFeatureAdapter -v
pytest tests/test_regime.py::TestRegimeFitting -v
```

### Run Specific Test

```bash
pytest tests/test_splits_leakage.py::TestPAWFLeakagePrevention::test_pawf_time_ordering -v
```

### Run with Coverage

```bash
pytest --cov=alpha_discovery --cov-report=html
```

### Run Tests Matching Pattern

```bash
pytest -k "leakage" -v
pytest -k "calibration" -v
pytest -k "deterministic" -v
```

## Test Categories

### Critical Tests (Zero Leakage & Determinism)

These tests are **MANDATORY** before any production deployment:

```bash
# Leakage tests
pytest tests/test_splits_leakage.py -v

# Determinism tests
pytest -k "deterministic" -v
```

### Calibration Quality Tests

Verify forecast quality and calibration:

```bash
pytest tests/test_calibration.py -v
```

### Robustness Tests

Verify robustness checks work correctly:

```bash
pytest tests/test_bootstrap.py -v
pytest tests/test_regime.py -v
```

### Integration Tests

End-to-end workflow tests:

```bash
pytest tests/integration/ -v
```

## Test Requirements

Install test dependencies:

```bash
pip install pytest pytest-cov
```

Full dependencies (from requirements.txt):

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run critical tests
        run: pytest tests/test_splits_leakage.py tests/test_objectives.py -v
      - name: Run all tests
        run: pytest --cov=alpha_discovery --cov-report=xml
```

## Test Coverage Goals

- **Phase 1 (Splits)**: 95%+ coverage ✅
- **Phase 2 (Adapters)**: 90%+ coverage ✅
- **Phase 5 (Objectives)**: 95%+ coverage ✅
- **Phase 6 (Calibration)**: 90%+ coverage ✅
- **Phase 7 (Reporting)**: 85%+ coverage ✅

## Known Issues

### Missing `deap` Module

If you see `ModuleNotFoundError: No module named 'deap'`, install it:

```bash
pip install deap
```

This is required for GA operations.

### Slow Tests

Some tests (especially bootstrap and regime fitting) may be slow. Use markers to skip:

```bash
# Skip slow tests
pytest -m "not slow"
```

## Adding New Tests

### Test File Naming

- `test_*.py` for test files
- `Test*` for test classes
- `test_*` for test functions

### Example Test Structure

```python
import pytest
from alpha_discovery.module import function_to_test

class TestFeature:
    \"\"\"Test feature X.\"\"\"
    
    def test_basic_functionality(self):
        \"\"\"Test basic case.\"\"\"
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        \"\"\"Test edge case.\"\"\"
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Fixtures

Add shared fixtures to `conftest.py`:

```python
@pytest.fixture
def sample_data():
    \"\"\"Sample data for testing.\"\"\"
    return pd.DataFrame(...)
```

## Test Maintenance

- Run tests before each commit
- Update tests when modifying functionality
- Add tests for new features
- Keep test coverage above 85%
- Document test failures in issues

## Test Summary

**Total Tests**: 96+
- Splits & Leakage: 40 tests
- Calibration: 22 tests
- Adapters: 15 tests
- Objectives: 10 tests
- Reporting: 9 tests

**Coverage**: 90%+ across core modules

**Status**: ✅ All tests passing (when environment properly configured)