# Runs Directory

This directory contains all experimental runs and results from the Alpha Discovery v10 system.

## 📁 Directory Structure

### `active/`
Contains the most recent and currently relevant runs (4 most recent):
- `run_seed100_20250910_182743/` - Most recent run
- `run_seed99_20250910_154937/` - Recent run  
- `run_seed97_20250910_131151/` - Recent run
- `run_seed97_20250910_123530/` - Recent run

### `archived/`
Contains all older runs and historical data:
- `1-30/`, `31-37/`, `38-50/`, `51-70/`, `71-91/` - Historical runs by seed ranges
- `run_seed85_*`, `run_seed92_*`, `run_seed93_*`, etc. - Individual older runs

### `summaries/`
Contains consolidated summary files:
- `master_open_trades_summary.csv` - Master summary of all open trades

### Numbered Directories
- `1-30/`, `31-37/`, `38-50/`, `51-70/`, `71-91/` - Historical runs organized by seed ranges

### Individual Run Directories
- `run_seedXXX_YYYYMMDD_HHMMSS/` - Individual experimental runs

## 🚀 Usage

### Finding Recent Runs
```bash
# List most recent runs
ls runs/active/

# List all runs sorted by date
ls -t runs/run_seed*
```

### Accessing Results
Each run directory contains:
- `config.json` - Configuration used for the run
- `pareto_front_summary.csv` - Pareto front results
- `pareto_front_trade_ledger.csv` - Detailed trade ledger
- `gauntlet/` - Gauntlet validation results
- `folds/` - Per-fold results

### Moving Runs to Archive
```bash
# Move old runs to archive
mv runs/run_seedXXX_YYYYMMDD_HHMMSS runs/archived/
```

## 📊 Run Naming Convention

Runs are named: `run_seed{SEED}_{YYYYMMDD}_{HHMMSS}`

- `SEED`: Random seed used for reproducibility
- `YYYYMMDD`: Date of the run
- `HHMMSS`: Time of the run

## 🔧 Maintenance

- **Active runs**: Keep 5-10 most recent runs in `active/`
- **Archived runs**: Move older runs to `archived/` by date
- **Cleanup**: Remove very old runs periodically to save space
- **Summaries**: Update master summaries after major runs
