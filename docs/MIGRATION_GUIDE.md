# Migration Guide: Legacy → Forecast-First

This guide explains the transition from the legacy P&L-based system to the new forecast-first validation framework.

---

## 🔴 What Was Removed (Legacy System)

The following components were **deleted** in Phase 10:

### Files Deleted
- `alpha_discovery/core/splits.py` - CPCV (Combinatorial Purged Cross-Validation)
- `alpha_discovery/eval/validation.py` - Legacy validation orchestrator  
- `alpha_discovery/eval/elv.py` - ELV scoring
- `alpha_discovery/eval/hart_index.py` - Hart Index calculation

### Concepts Removed
- **CPCV** (Combinatorial Purged Cross-Validation)
- **ELV** (Expected Lifetime Value) scoring
- **Hart Index** (0-100 trust scores)
- **P&L-based objectives** (ig_sharpe, min_ig, sharpe_ratio, etc.)
- **`--mode legacy`** CLI option

---

## 🟢 What Replaced It (Forecast-First)

### New Files
- **`alpha_discovery/splits/`** - Complete splitting package
  - `pawf.py` - Purged Anchored Walk-Forward (replaces CPCV)
  - `npwf.py` - Nested Purged Walk-Forward (new inner CV)
  - `regime.py` - GMM regime detection
  - `bootstrap.py` - Robustness tests
  - `adversarial.py` - Drift detection

- **`alpha_discovery/adapters/`** - Feature adapters
  - `features.py` - Read-only FeatureAdapter
  - `subspace.py` - Feature subsampling

- **`alpha_discovery/eval/orchestrator.py`** - New validation orchestrator (replaces validation.py)
- **`alpha_discovery/eval/objectives.py`** - Proper scoring rules
- **`alpha_discovery/eval/calibration.py`** - Calibration helpers
- **`alpha_discovery/reporting/eligibility_report.py`** - New reporting

### New Concepts
- **PAWF** (Purged Anchored Walk-Forward) - outer validation
- **NPWF** (Nested Purged Walk-Forward) - inner GA selection
- **EligibilityMatrix** - structured validation results (replaces ELV/Hart)
- **Proper Scoring Rules** - CRPS, Brier, Log Loss, Pinball
- **Skill vs Baselines** - Uniform/Marginal/Persistence
- **Calibration** - ECE/MCE + Isotonic/Platt

---

## 📊 Side-by-Side Comparison

| Aspect | Legacy (Removed) | Forecast-First (New) |
|--------|------------------|----------------------|
| **Validation** | CPCV | PAWF + NPWF |
| **Objectives** | P&L (Sharpe, ig_sharpe) | Proper scoring (CRPS, Brier) |
| **Scoring** | ELV + Hart Index | EligibilityMatrix |
| **Output** | Pareto CSV | eligibility_matrix.json + 6 reports |
| **Calibration** | None | ECE/MCE + Isotonic/Platt |
| **Baselines** | None | Uniform/Marginal/Persistence |
| **Drift** | None | Adversarial AUC |
| **Regimes** | HMM-based | GMM-based |
| **Inner CV** | None | NPWF (3 folds) |
| **Entry Point** | `python main.py --mode legacy` | `python main.py` |

---

## 🔄 Code Migration Examples

### Before (Legacy)

```python
# main.py (legacy)
from alpha_discovery.core.splits import HybridSplits
from alpha_discovery.eval.validation import run_full_pipeline
from alpha_discovery.eval.elv import calculate_elv_and_labels
from alpha_discovery.eval.hart_index import calculate_hart_index

# Create CPCV splits
splits = HybridSplits(data_index=signals_df.index)

# Run GA with CPCV
unique_candidates, _, _ = discovery_phase(splits, ...)

# Validate with CPCV
pre_elv_df = run_full_pipeline(
    unique_candidates, all_discovery_results, splits, ...
)

# Calculate ELV scores
final_results_df = calculate_elv_and_labels(pre_elv_df)

# Calculate Hart Index
final_results_df = calculate_hart_index(final_results_df)

# Output: Pareto CSV with ELV/Hart columns
```

### After (Forecast-First)

```python
# main.py (forecast-first)
from alpha_discovery.eval.orchestrator import ForecastOrchestrator
from alpha_discovery.reporting.eligibility_report import generate_eligibility_report

# Create orchestrator
orchestrator = ForecastOrchestrator(
    master_df=master_df,
    signals_df=signals_df,
    signals_meta=signals_meta,
    output_dir=Path(run_dir),
    feature_lookback_tail=lookback_tail,
    seed=settings.ga.seed
)

# Run validation with PAWF + NPWF
eligibility_matrix = orchestrator.run_validation(
    n_outer_splits=4,
    test_size_months=6,
    purge_days=5,
    n_inner_folds=3,
    n_regimes=5,
    run_ga=True
)

# Generate reports
outputs = generate_eligibility_report(
    eligibility_matrix_path=run_dir / "eligibility_matrix.json",
    output_dir=run_dir / "reports",
    min_skill_vs_marginal=0.01,
    max_calibration_mae=0.15,
    drift_gate=True,
    top_n=50
)

# Output: eligibility_matrix.json + 6 comprehensive CSVs
```

---

## 🎯 Objective Mapping

### Legacy Objectives → Forecast-First Equivalents

| Legacy (Removed) | Forecast-First (New) | Notes |
|------------------|----------------------|-------|
| `ig_sharpe` | `crps_neg` | CRPS is the primary forecast quality metric |
| `min_ig` | `skill_vs_marginal` | Skill delta vs marginal baseline |
| `sharpe_ratio` | N/A | Not used (P&L-based) |
| `sortino_ratio` | N/A | Not used (P&L-based) |
| `omega_ratio` | N/A | Not used (P&L-based) |
| `max_drawdown` | N/A | Not used (P&L-based) |
| N/A | `brier_score` | **NEW**: Probabilistic calibration |
| N/A | `log_loss` | **NEW**: Probabilistic sharpness |
| N/A | `pinball_q10/q90` | **NEW**: Quantile forecasts |
| N/A | `calibration_mae` | **NEW**: Calibration error |
| N/A | `drift_auc` | **NEW**: Drift detection |

### Config Update

**Before:**
```python
class GaConfig:
    objectives: List[str] = [
        "ig_sharpe",
        "min_ig",
        "sharpe_ratio",
        "sortino_ratio",
        ...
    ]
```

**After:**
```python
class GaConfig:
    objectives: List[str] = [
        "crps_neg",                     # Primary forecast quality
        "pinball_loss_neg_q10",         # Lower tail
        "pinball_loss_neg_q90",         # Upper tail
        "info_gain",                    # Information content
        "w1_effect",                    # Distributional distance
        ...
    ]
```

---

## 📁 Output Structure Changes

### Legacy Output

```
runs/pivot_forecast_seed194_TIMESTAMP/
├── pareto_front.csv                  # All candidates
├── forecast_slate.csv                # Top candidates
├── elv_scores.csv                    # ELV scores
├── hart_index.csv                    # Hart Index scores
└── diagnostics/
    └── hart_index_usage.txt
```

**Columns**: ticker, horizon, signals, ig_sharpe, min_ig, elv_score, hart_index, ...

### Forecast-First Output

```
runs/forecast_first_seed194_TIMESTAMP/
├── eligibility_matrix.json           # Complete validation results
│
└── reports/
    ├── report_summary.json           # High-level statistics
    ├── eligible_setups.csv           # Top eligible setups
    ├── skill_breakdown.csv           # Skill by ticker/horizon
    ├── calibration_summary.csv       # Calibration diagnostics
    ├── drift_analysis.csv            # Drift detection results
    └── regime_stratification.csv     # Performance by regime
```

**Columns**: ticker, horizon, setup, skill_vs_marginal, crps, brier_score, calibration_mae, drift_passed, ...

---

## 🔍 Key Differences Explained

### 1. Validation Strategy

**Legacy (CPCV)**:
- Combinatorial selection of train/test groups
- No inner cross-validation for GA
- Single-pass evaluation

**Forecast-First (PAWF + NPWF)**:
- Anchored walk-forward for realistic time progression
- Nested CV: inner folds for GA, outer for evaluation
- Strict purge (5 days) and embargo (10 days)

### 2. Optimization Objectives

**Legacy**:
- P&L-based: Sharpe ratio, ig_sharpe, min_ig
- Implicitly assumes trades are executed
- Can overfit to backtest artifacts

**Forecast-First**:
- Proper scoring rules: CRPS, Brier, Log Loss
- Evaluates forecast quality, not P&L
- Theoretically sound (incentive-compatible)

### 3. Scoring System

**Legacy (ELV + Hart Index)**:
- ELV: Expected lifetime value (proprietary formula)
- Hart Index: 0-100 trust score (composite metric)
- Both combine multiple signals into single score

**Forecast-First (EligibilityMatrix)**:
- Structured results: separate metrics for skill, calibration, drift
- No single "score" - multiple orthogonal dimensions
- Transparent: all metrics preserved

### 4. Calibration

**Legacy**:
- No calibration checks
- Probabilities assumed to be well-calibrated

**Forecast-First**:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Isotonic Regression + Platt Scaling
- Reliability curves exported

---

## 🚨 Breaking Changes

### 1. CLI

**Before**: `python main.py --mode legacy`  
**After**: `python main.py` (no mode flag)

### 2. Imports

**Before**:
```python
from alpha_discovery.core.splits import HybridSplits
from alpha_discovery.eval.validation import run_full_pipeline
from alpha_discovery.eval.elv import calculate_elv_and_labels
from alpha_discovery.eval.hart_index import calculate_hart_index
```

**After**:
```python
from alpha_discovery.eval.orchestrator import ForecastOrchestrator
from alpha_discovery.reporting.eligibility_report import generate_eligibility_report
```

### 3. Output Format

**Before**: CSV with ELV/Hart columns  
**After**: JSON eligibility matrix + 6 CSV reports

### 4. Metrics

**Before**: P&L-based (Sharpe, ig_sharpe)  
**After**: Forecast-based (CRPS, Brier, Log Loss)

---

## ✅ Migration Checklist

- [ ] Update `config.py` objectives to proper scoring rules
- [ ] Replace `HybridSplits` with `ForecastOrchestrator`
- [ ] Replace `run_full_pipeline` with `orchestrator.run_validation`
- [ ] Replace `calculate_elv_and_labels` with eligibility report generation
- [ ] Update downstream scripts to read `eligibility_matrix.json`
- [ ] Update CI/CD to run `python main.py` (no --mode flag)
- [ ] Archive old runs with ELV/Hart outputs
- [ ] Retrain any models that depended on legacy metrics
- [ ] Update documentation/runbooks

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'alpha_discovery.core.splits'"

**Cause**: Legacy code trying to import deleted CPCV module.

**Fix**: Replace with forecast-first imports:
```python
# OLD
from alpha_discovery.core.splits import HybridSplits

# NEW
from alpha_discovery.eval.orchestrator import ForecastOrchestrator
```

### "No such file or directory: 'runs/*/pareto_front.csv'"

**Cause**: Looking for legacy output format.

**Fix**: Load eligibility matrix instead:
```python
import json

with open('runs/.../eligibility_matrix.json', 'r') as f:
    eligibility = json.load(f)

results_df = pd.DataFrame(eligibility['results'])
```

### "Unknown objective: ig_sharpe"

**Cause**: Config still has legacy P&L objectives.

**Fix**: Update to proper scoring rules:
```python
objectives: List[str] = [
    "crps_neg",
    "pinball_loss_neg_q10",
    "pinball_loss_neg_q90",
    ...
]
```

---

## 📚 Additional Resources

- **[CLI Usage](CLI_USAGE.md)** - New command-line interface
- **[Metrics Reference](METRICS_REFERENCE.md)** - All available metrics
- **[Test README](../tests/README.md)** - Test suite documentation
- **[README](../README.md)** - Main project documentation

---

## ❓ FAQ

**Q: Can I still use the legacy system?**  
A: No, the legacy code was removed in Phase 10. The forecast-first system is the only option going forward.

**Q: How do I compare new results to old ELV/Hart scores?**  
A: You can't directly compare. ELV/Hart are P&L-based composites; forecast-first uses proper scoring rules. Archive old results and start fresh.

**Q: What if I need P&L metrics for reporting?**  
A: P&L metrics are still computed in `eval/objectives.py` but are **not used for GA optimization**. They're available for reporting only.

**Q: Is the forecast-first system slower?**  
A: NPWF adds inner folds, so discovery is ~3x slower. But validation is more rigorous with better generalization.

**Q: Can I run both systems?**  
A: No. Legacy code was deleted. Forecast-first is the single source of truth.

---

**Migration Complete!** You're now running a rigorous, forecast-first validation system. 🎉
