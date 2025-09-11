# Scripts Directory

This directory contains utility scripts organized by purpose.

## 📁 Directory Structure

### 🔧 Maintenance (`maintenance/`)
Scripts for maintaining and cleaning up the project:
- `cleanup_empty_runs.py` - Remove empty run directories
- `consolidate_open_trades.py` - Consolidate open trades summaries

### 📊 Analysis (`analysis/`)
Scripts for analyzing results and generating reports:
- `run_prediction.py` - Run prediction experiments

### 🧪 Testing (`testing/`)
Test scripts for validating system components:
- `test_8_feature_iv_system.py` - Test IV pricing system
- `test_robust_metrics.py` - Test robust metrics implementation
- `test_split_extension.py` - Test walk-forward split functionality

### 🛠️ Utilities (`utilities/`)
General utility scripts:
- `check_events.py` - Validate economic event data
- `example_fitness_config.py` - Example fitness configuration
- `one_liner_smoke.py` - Quick system validation

## 🚀 Usage

Run scripts from the project root:

```bash
# Maintenance
python scripts/maintenance/cleanup_empty_runs.py

# Analysis
python scripts/analysis/run_prediction.py

# Testing
python scripts/testing/test_robust_metrics.py

# Utilities
python scripts/utilities/check_events.py
```

## 📝 Notes

- All scripts are designed to be run from the project root directory
- Scripts maintain backward compatibility with existing workflows
- Test scripts can be run independently for validation
