# CLI Usage Guide - Alpha Discovery Engine v10

## Overview

The Alpha Discovery Engine now supports three execution modes via command-line arguments:

- **`legacy`**: Original P&L-based discovery (default for backward compatibility)
- **`discover`**: Forecast-first discovery with PAWF validation
- **`select`**: Load eligibility matrix and generate forecast slate

## Modes

### 1. Legacy Mode (Default)

Runs the original P&L-based discovery workflow.

```bash
python main.py
# or explicitly:
python main.py --mode legacy
```

**Workflow:**
1. Load data and build features/signals
2. Run GA with CPCV splits
3. Validate on OOS folds
4. Calculate ELV and Hart Index
5. Generate forecast slate

### 2. Discover Mode (Forecast-First)

Runs the new forecast-first validation pipeline with proper scoring rules.

```bash
python main.py --mode discover
```

**Workflow:**
1. Load data and build features/signals
2. Create PAWF outer splits (4 folds, 6-month test windows)
3. Run GA with NPWF inner folds for hyperparameter selection
4. Validate all candidates on outer PAWF folds
5. Calculate skill vs. baselines (uniform/marginal/persistence)
6. Fit calibrators (Isotonic + Platt)
7. Run drift detection and bootstrap stability tests
8. Generate `EligibilityMatrix` with comprehensive metrics
9. Save eligibility matrix to `runs/validation_*/eligibility_matrix.json`
10. Generate eligibility reports (skill breakdown, calibration, drift analysis)

**Output Files:**
- `runs/validation_*/eligibility_matrix.json` - Full validation results
- `runs/validation_*/reports/eligible_setups.csv` - Top eligible setups
- `runs/validation_*/reports/skill_breakdown.csv` - Skill by ticker/horizon
- `runs/validation_*/reports/calibration_summary.csv` - Calibration diagnostics
- `runs/validation_*/reports/drift_analysis.csv` - Drift detection results
- `runs/validation_*/reports/regime_stratification.csv` - Performance by regime
- `runs/validation_*/reports/report_summary.json` - High-level statistics

**Key Features:**
- ✅ Zero leakage (PAWF + NPWF with purge and embargo)
- ✅ Proper scoring rules only (CRPS, Brier, Log Loss, Pinball)
- ✅ Skill delta vs. baselines
- ✅ Calibration checks (ECE, MCE, reliability curves)
- ✅ Drift detection (adversarial AUC)
- ✅ Regime stratification
- ✅ Bootstrap robustness tests

### 3. Select Mode (Portfolio Construction)

Load eligibility matrix and apply selection criteria to build final portfolio.

```bash
# Use most recent eligibility matrix
python main.py --mode select

# Use specific eligibility matrix
python main.py --mode select --eligibility runs/validation_001/eligibility_matrix.json
```

**Workflow:**
1. Load eligibility matrix
2. Apply filters:
   - Minimum skill vs. marginal: 0.01 (beat marginal by 1% CRPS)
   - Maximum calibration MAE: 0.15 (max 15% calibration error)
   - Drift gate: must pass adversarial drift test
3. Rank by `skill_vs_marginal`
4. Apply portfolio constraints:
   - Max 2 setups per ticker (diversification)
   - Top 20 overall (portfolio size)
5. Generate forecast slate: `runs/validation_*/selection/forecast_slate.csv`

**Output:**
- `runs/validation_*/selection/forecast_slate.csv` - Final actionable setups

## Command-Line Arguments

### `--mode {legacy,discover,select}`

Execution mode (default: `legacy`)

### `--eligibility PATH`

Path to `eligibility_matrix.json` file (only for `select` mode).

If not provided, uses the most recent eligibility matrix in `runs/`.

### `--seed SEED`

Override the GA seed from config.

```bash
python main.py --mode discover --seed 999
```

## Complete Workflow Example

### Step 1: Run Discovery

```bash
# Run forecast-first discovery
python main.py --mode discover --seed 194

# Output:
# ✓ Eligibility matrix saved to: runs/pivot_forecast_seed194_20250930_120000/eligibility_matrix.json
# ✓ Reports saved to: runs/pivot_forecast_seed194_20250930_120000/reports/
```

### Step 2: Review Reports

```bash
cd runs/pivot_forecast_seed194_20250930_120000/reports/

# Check summary statistics
cat report_summary.json

# Review eligible setups
head eligible_setups.csv

# Check calibration quality
head calibration_summary.csv
```

### Step 3: Run Selection

```bash
# Select from the discovered setups
python main.py --mode select --eligibility runs/pivot_forecast_seed194_20250930_120000/eligibility_matrix.json

# Output:
# ✓ Eligible setups: 45/120
# ✓ Final portfolio: 20 setups across 15 tickers
# ✓ Forecast slate saved to: runs/pivot_forecast_seed194_20250930_120000/selection/forecast_slate.csv
```

### Step 4: Deploy

```bash
# The forecast_slate.csv is now ready for production
cat runs/pivot_forecast_seed194_20250930_120000/selection/forecast_slate.csv
```

## Configuration

The CLI modes respect all configuration in `alpha_discovery/config.py`:

### Discovery Mode Config

```python
# In config.py

class ForecastConfig(BaseModel):
    horizons: List[int] = [1, 3, 5, 21]  # Forecast horizons

class ValidationConfig(BaseModel):
    n_outer_splits: int = 4               # PAWF outer splits
    test_size_months: int = 6             # Test window size
    purge_days: int = 5                   # Purge buffer
    n_inner_folds: int = 3                # NPWF inner folds for GA
    n_regimes: int = 5                    # GMM regimes
```

### Selection Mode Config

You can modify selection thresholds directly in `main_select()` function:

```python
MIN_SKILL = 0.01          # Minimum skill vs. marginal
MAX_CALIB_MAE = 0.15      # Maximum calibration error
DRIFT_GATE = True         # Enforce drift gate
MAX_PER_TICKER = 2        # Max setups per ticker
TOP_N = 20                # Portfolio size
```

## Help

```bash
python main.py --help
```

## Troubleshooting

### "No module named 'deap'"

Install DEAP for genetic algorithms:

```bash
pip install deap
```

### "No eligibility matrices found"

Run discovery mode first:

```bash
python main.py --mode discover
```

### "Eligible setups: 0/X"

Your selection criteria may be too strict. Try:
- Lowering `MIN_SKILL` (e.g., 0.005)
- Raising `MAX_CALIB_MAE` (e.g., 0.20)
- Disabling `DRIFT_GATE`

## Migration from Legacy Mode

To migrate from the original workflow:

1. **Review Phase**: Run both modes and compare
   ```bash
   python main.py --mode legacy --seed 194
   python main.py --mode discover --seed 194
   ```

2. **Transition Phase**: Use `discover` mode as primary, `legacy` as fallback
   ```bash
   python main.py --mode discover
   ```

3. **Full Adoption**: Default to `discover` mode
   - Update docs and runbooks
   - Deprecate legacy mode in future release

## Next Steps

After Phase 8 completion:
- **Phase 9**: Add comprehensive tests for all modes
- **Phase 10**: Migrate legacy code and update all documentation
