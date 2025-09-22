# Alpha Discovery System

A comprehensive options-based backtesting and genetic algorithm system for discovering profitable trading strategies.

## System Architecture

### Core Entry Point
- **`main.py`** - Main pipeline orchestrator that coordinates the entire alpha discovery process

### Data Layer (`alpha_discovery/data/`)
- **`loader.py`** - Data ingestion from Excel/Parquet sources
- **`events.py`** - Economic event calendar integration and feature generation

### Feature Engineering (`alpha_discovery/features/`)
- **`core.py`** - Mathematical feature computation functions (z-scores, correlations, volatility, etc.)
- **`registry.py`** - Feature matrix builder that applies core functions to market data

### Signal Generation (`alpha_discovery/signals/`)
- **`compiler.py`** - Converts continuous features into binary trading signals using rule-based grammar

### Search & Optimization (`alpha_discovery/search/`)
- **`nsga.py`** - NSGA-II multi-objective genetic algorithm implementation
- **`population.py`** - Population initialization and genetic operations
- **`island_model.py`** - Parallel island model for genetic algorithm diversity
- **`ga_core.py`** - Core genetic algorithm evaluation and fitness functions

### Backtesting Engine (`alpha_discovery/engine/`)
- **`backtester.py`** - Main backtesting facade
- **`bt_core.py`** - Core backtesting logic and trade execution
- **`bt_common.py`** - Common backtesting utilities and trade horizons
- **`bt_runtime.py`** - Runtime backtesting execution

### Options Pricing (`alpha_discovery/options/`)
- **`pricing.py`** - Black-Scholes pricing and IV term structure mapping
- **`models.py`** - Options pricing models and mathematical functions
- **`market.py`** - Market data access for underlying prices and IV

### Evaluation (`alpha_discovery/eval/`)
- **`validation.py`** - Walk-forward validation and time series splits
- **`metrics.py`** - Portfolio performance metrics calculation
- **`selection.py`** - Portfolio selection and ranking algorithms
- **`nav.py`** - Net Asset Value calculations

### Reporting (`alpha_discovery/reporting/`)
- **`artifacts.py`** - Results persistence and CSV generation
- **`manifests.py`** - Run metadata and configuration tracking

### Configuration (`alpha_discovery/config.py`)
- Centralized configuration management using Pydantic models
- Settings for GA parameters, data sources, validation, options pricing, and reporting

## System Flow

### 1. Data Pipeline
```
main.py → data/loader.py → data/events.py
```
- Loads market data from Excel/Parquet files
- Integrates economic event calendar
- Creates unified master dataframe

### 2. Feature Engineering
```
main.py → features/registry.py → features/core.py
```
- Applies mathematical transformations to market data
- Generates technical indicators, volatility measures, sentiment features
- Creates cross-asset correlation and regime features
- Builds comprehensive feature matrix

### 3. Signal Compilation
```
main.py → signals/compiler.py
```
- Converts continuous features to binary signals
- Applies rule-based grammar (percentiles, z-scores, thresholds)
- Generates large pool of primitive trading signals

### 4. Genetic Algorithm Search
```
main.py → search/nsga.py → search/ga_core.py → search/population.py
```
- Initializes population of (ticker, signal_set) combinations
- Evolves solutions using NSGA-II multi-objective optimization
- Evaluates fitness using options backtesting engine
- Supports island model for parallel evolution

### 5. Options Backtesting
```
search/ga_core.py → engine/backtester.py → engine/bt_core.py → options/pricing.py
```
- Simulates options trades using Black-Scholes pricing
- Implements realistic strike selection and IV term structure mapping
- Tracks trade performance and portfolio metrics
- Supports multiple exit strategies and regime-aware exits

### 6. Validation & Evaluation
```
main.py → eval/validation.py → eval/metrics.py
```
- Creates walk-forward time series splits
- Validates strategies on out-of-sample data
- Calculates comprehensive performance metrics (Sharpe, Sortino, expectancy, etc.)

### 7. Results & Reporting
```
main.py → reporting/artifacts.py
```
- Persists training and OOS results
- Generates CSV reports with trade ledgers and performance summaries
- Creates per-fold artifacts for detailed analysis

## Key Design Patterns

### Specialized DNA Structure
- Each individual is a tuple: `(ticker, [signal_list])`
- Enables ticker-specific strategy optimization
- Supports cross-ticker signal combinations

### Multi-Objective Optimization
- Configurable fitness objectives (Sortino, expectancy, support)
- Pareto front ranking for trade-off analysis
- Crowding distance for diversity maintenance

### Options-First Architecture
- All strategies evaluated through options backtesting
- Realistic pricing with IV term structure mapping
- Support for multiple option types and strike selection methods

### Walk-Forward Validation
- Time series splits prevent look-ahead bias
- Embargo periods between train/test windows
- Out-of-sample testing on evolved strategies

### Event-Driven Features
- Economic calendar integration
- Event-based feature generation
- Regime-aware trading strategies

## Configuration-Driven
The system is highly configurable through `config.py`:
- GA parameters (population size, generations, mutation rates)
- Data sources and ticker universes
- Validation settings and walk-forward parameters
- Options pricing regimes and IV mapping
- Reporting and artifact generation

This architecture enables systematic discovery of profitable options trading strategies while maintaining rigorous validation standards and realistic market simulation.
