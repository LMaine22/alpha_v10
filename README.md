# Alpha Discovery Engine v10 - Forecast-First

> A rigorous, leakage-resistant quantitative alpha discovery framework built on proper scoring rules, nested cross-validation, and comprehensive calibration.

---

## 🎯 Quick Start

```bash
# Run forecast-first discovery
python main.py

# Output: runs/forecast_first_seed{SEED}_{TIMESTAMP}/
#   ├── eligibility_matrix.json       # Full validation results
#   └── reports/                       # Comprehensive metrics
#       ├── eligible_setups.csv
#       ├── skill_breakdown.csv
#       ├── calibration_summary.csv
#       └── drift_analysis.csv
```

## 📊 What This Does

The Alpha Discovery Engine discovers **forecast-based trading signals** using:

1. **Zero-Leakage Validation** (PAWF + NPWF with purge and embargo)
2. **Proper Scoring Rules** (CRPS, Brier, Log Loss, Pinball)
3. **Skill vs. Baselines** (Uniform/Marginal/Persistence)
4. **Calibration Checks** (ECE/MCE + Isotonic/Platt scaling)
5. **Drift Detection** (Adversarial AUC)
6. **Robustness Tests** (Bootstrap, regime stratification)

**Output**: EligibilityMatrix with comprehensive forecast quality metrics.

---

## 🏗️ System Architecture

```
Data Loading → Feature Registry → Signal Compilation
                                         ↓
                              ForecastOrchestrator
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    ↓                                         ↓
            PAWF Outer Splits (4 folds)              GMM Regime Detection
                    ↓                                         ↓
    ┌───────────────┴───────────────┐               Regime Assignment
    ↓                               ↓                         ↓
NPWF Inner Folds (3)          Outer Test Fold         Similarity Calc
    ↓                               ↓                         ↓
GA with Proper                 Evaluate Setup          Drift Detection
Scoring Rules                  + Calibrators           + Bootstrap
    ↓                               ↓                         ↓
Select Best                    Skill vs Baselines      Robustness
Candidates                     + ECE/MCE               Metrics
    └───────────────┬───────────────┘                         ↓
                    ↓                                  ┌───────┴────────┐
            Eligibility Matrix                        │                │
                    ↓                            Regime Stats    Drift Stats
            Report Generation
                    ↓
        6 Comprehensive CSVs + JSON Summary
```

---

## 📁 Project Structure

```
alpha_discovery/
├── splits/                  # PAWF/NPWF validation splits
│   ├── pawf.py             # Purged Anchored Walk-Forward
│   ├── npwf.py             # Nested Purged Walk-Forward
│   ├── regime.py           # GMM regime detection
│   ├── bootstrap.py        # Robustness tests
│   └── adversarial.py      # Drift detection
│
├── adapters/               # Read-only feature access
│   ├── features.py         # FeatureAdapter
│   └── subspace.py         # Feature subsampling
│
├── eval/                   # Forecast evaluation
│   ├── orchestrator.py     # ForecastOrchestrator (main)
│   ├── objectives.py       # Proper scoring rules
│   └── calibration.py      # ECE/MCE/Isotonic/Platt
│
├── reporting/              # Output generation
│   └── eligibility_report.py  # Generate 6 report files
│
├── search/                 # Genetic algorithm
│   ├── ga_core.py          # Setup evaluation
│   ├── nsga.py             # NSGA-II multi-objective
│   └── island_model.py     # Island-based evolution
│
├── features/               # Feature registry
├── signals/                # Signal compiler
└── config.py              # Configuration

main.py                     # Entry point
tests/                      # Comprehensive test suite
docs/                       # Documentation
```

---

## 🔬 Validation Methodology

### PAWF (Purged Anchored Walk-Forward)

**Outer validation** with 4 time-anchored folds:

```
Split 1: Train[2020-2021] → Purge → Embargo → Test[2022-H1]
Split 2: Train[2020-2022] → Purge → Embargo → Test[2022-H2]
Split 3: Train[2020-2023Q1] → Purge → Embargo → Test[2023-Q2]
Split 4: Train[2020-2023-H1] → Purge → Embargo → Test[2023-H2]
```

- **Purge**: 5-day gap between train and test
- **Embargo**: 10-day buffer (feature lookback tail)
- **Anchored**: Train always starts from same date

### NPWF (Nested Purged Walk-Forward)

**Inner folds for GA hyperparameter selection:**

```
For each PAWF outer fold:
  Split outer_train into 3 NPWF inner folds
  GA optimizes on these inner folds
  Final evaluation on outer_test (unseen)
```

- **Zero leakage**: Inner folds never see outer test data
- **Proper scoring only**: CRPS, Brier, Log Loss, Pinball
- **Deterministic**: Same seed = same folds

---

## 📈 Metrics & Objectives

### Proper Scoring Rules (GA Optimization)

- **CRPS** (Continuous Ranked Probability Score) - distributional accuracy
- **Brier Score** - probabilistic calibration
- **Log Loss** - probabilistic sharpness
- **Pinball Loss** (q10, q25, q50, q75, q90) - quantile forecasts

### Skill Metrics (vs Baselines)

- **Uniform** - naive 50/50 forecast
- **Marginal** - historical frequency
- **Persistence** - last observed value

### Calibration

- **ECE** (Expected Calibration Error)
- **MCE** (Maximum Calibration Error)
- **Isotonic Regression** - monotonic calibration
- **Platt Scaling** - logistic calibration

### Robustness

- **Bootstrap Stability** - skill delta confidence intervals
- **Drift Detection** - adversarial AUC (train vs test)
- **Regime Consistency** - GMM-based stratification

---

## 🚀 Usage Examples

### Basic Discovery

```bash
python main.py
```

### With Custom Seed

```python
# In config.py
settings.ga.seed = 999
```

Or programmatically:

```python
from alpha_discovery.eval.orchestrator import ForecastOrchestrator

orchestrator = ForecastOrchestrator(...)
eligibility = orchestrator.run_validation(
    n_outer_splits=4,
    test_size_months=6,
    purge_days=5,
    n_inner_folds=3,
    n_regimes=5
)
```

### Generate Reports

```python
from alpha_discovery.reporting.eligibility_report import (
    generate_eligibility_report,
    print_eligibility_summary
)

outputs = generate_eligibility_report(
    eligibility_matrix_path="runs/.../eligibility_matrix.json",
    output_dir="runs/.../reports",
    min_skill_vs_marginal=0.01,
    max_calibration_mae=0.15,
    drift_gate=True,
    top_n=50
)

print_eligibility_summary(outputs['summary'])
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_splits_leakage.py -v          # Leakage prevention
pytest tests/test_calibration.py -v             # Calibration quality
pytest tests/test_adapters.py -v                # Feature adapters
pytest tests/test_regime.py -v                  # GMM regime detection
pytest tests/test_bootstrap.py -v               # Bootstrap robustness

# Run with coverage
pytest --cov=alpha_discovery --cov-report=html
```

**Test Coverage**: 96+ tests, 90%+ coverage across core modules.

---

## 📚 Documentation

- **[CLI Usage](docs/CLI_USAGE.md)** - Command-line reference
- **[Test README](tests/README.md)** - Comprehensive test guide
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Transitioning from legacy system
- **[Metrics Reference](docs/METRICS_REFERENCE.md)** - All available metrics
- **[GA README](docs/GA_README.md)** - Genetic algorithm details

---

## 🔧 Configuration

Key settings in `alpha_discovery/config.py`:

```python
class GaConfig:
    population_size: int = 50
    generations: int = 5
    objectives: List[str] = [
        "crps_neg",
        "pinball_loss_neg_q10",
        "pinball_loss_neg_q90",
        "info_gain",
        "w1_effect",
        ...
    ]

class ForecastConfig:
    horizons: List[int] = [1, 3, 5, 21]

class ValidationConfig:
    n_outer_splits: int = 4
    test_size_months: int = 6
    purge_days: int = 5
    n_inner_folds: int = 3
```

---

## 🎯 Output Files

After running `python main.py`:

```
runs/forecast_first_seed194_20250930_120000/
├── eligibility_matrix.json                    # Complete validation results
│
└── reports/
    ├── report_summary.json                    # High-level statistics
    ├── eligible_setups.csv                    # Top eligible setups ranked by skill
    ├── skill_breakdown.csv                    # Skill by ticker/horizon
    ├── calibration_summary.csv                # Calibration diagnostics
    ├── drift_analysis.csv                     # Drift detection results
    └── regime_stratification.csv              # Performance by regime
```

### EligibilityMatrix Schema

```json
{
  "metadata": {
    "timestamp": "2025-09-30T12:00:00",
    "n_outer_splits": 4,
    "n_inner_folds": 3,
    "seed": 194
  },
  "results": [
    {
      "split_id": "PAWF_v1|OUTER:202401|H:5|E:normal|P:5|EMB:10|REG:R1",
      "ticker": "AAPL",
      "setup": ["sig_momentum_20", "sig_vol_surge"],
      "horizon": 5,
      "crps": 0.234,
      "brier_score": 0.189,
      "log_loss": 0.567,
      "skill_vs_marginal": 0.023,
      "calibration_mae": 0.087,
      "calibration_ece": 0.045,
      "drift_auc": 0.512,
      "drift_passed": true,
      "regime_similarity": 0.78,
      ...
    }
  ]
}
```

---

## 🏆 Key Features

✅ **Zero Leakage**: PAWF + NPWF with purge and embargo  
✅ **Proper Scoring**: CRPS, Brier, Log Loss only for optimization  
✅ **Skill Baselines**: Beat uniform/marginal/persistence  
✅ **Calibration**: ECE/MCE + Isotonic/Platt  
✅ **Drift Detection**: Adversarial AUC  
✅ **Regime-Aware**: GMM clustering  
✅ **Bootstrap Tests**: Stationary & heavy-tailed  
✅ **Comprehensive Reports**: 6 CSV/JSON outputs  
✅ **Deterministic**: Same seed = same results  
✅ **Well-Tested**: 96+ tests, 90%+ coverage  

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- pandas, numpy, scipy
- scikit-learn (GaussianMixture, calibration)
- joblib (caching)
- tqdm (progress bars)

---

## 🤝 Contributing

1. Run tests: `pytest`
2. Check coverage: `pytest --cov`
3. Lint: `flake8 alpha_discovery/`
4. Format: `black alpha_discovery/`

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

Built with rigorous validation methodology inspired by:
- Advances in Financial Machine Learning (López de Prado)
- Probabilistic Forecasting (Gneiting & Katzfuss)
- Multi-Objective Optimization (Deb, NSGA-II)

---

**Version**: 10.0 (Forecast-First)  
**Status**: Production Ready  
**Last Updated**: September 30, 2025
