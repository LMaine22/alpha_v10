# Genetic Algorithm for Alpha Discovery

## Overview

The Alpha Discovery system uses a sophisticated **Genetic Algorithm (GA)** to automatically discover profitable options trading strategies. Think of it as an evolutionary process that breeds and improves trading strategies over multiple generations, just like how species evolve in nature.

## What It Does

The GA searches through thousands of possible combinations of:
- **Trading signals** (technical indicators, market features)
- **Specific stock tickers** (AAPL, MSFT, etc.)
- **Trading directions** (long/short)

To find the most profitable and robust options trading strategies.

## How It Works

### 1. **Individual Representation**
Each "individual" in the population is a trading strategy represented as:
```
(ticker, [signal1, signal2, signal3])
```
- **ticker**: The specific stock to trade (e.g., "AAPL")
- **signals**: A list of 1-3 technical indicators/features that trigger trades

### 2. **Fitness Evaluation**
Each strategy is evaluated using **three key metrics**:

1. **Sortino Ratio (Lower Bound)** - Risk-adjusted returns focusing on downside risk
2. **Expectancy** - Expected profit per trade
3. **Support** - Number of trades (more trades = more robust)

The GA tries to **maximize all three objectives simultaneously** using a multi-objective optimization approach.

### 3. **Evolution Process**

#### **Initialization**
- Creates 100 random trading strategies
- Each strategy gets a random ticker and 1-3 random signals
- Ensures no duplicate strategies in the first generation

#### **Island Model Architecture**
The population is divided into **4 separate "islands"** (sub-populations):
- Each island evolves independently with 25 individuals
- Islands occasionally exchange their best strategies (migration)
- This prevents premature convergence and maintains diversity

#### **Each Generation (8 total generations):**

1. **Parent Selection** - Uses tournament selection to pick the best strategies
2. **Crossover** - Combines two parent strategies to create offspring:
   - Child inherits ticker from one parent
   - Child inherits signals from both parents (mixed)
3. **Mutation** - Randomly modifies 20% of strategies:
   - 15% chance to change the ticker
   - 85% chance to change/add/remove signals
4. **Evaluation** - Backtests each new strategy on historical data
5. **Selection** - Keeps the best strategies for the next generation

### 4. **Multi-Objective Optimization (NSGA-II)**

Instead of optimizing just one metric, the GA uses **Pareto optimization**:
- Finds strategies that are "non-dominated" (can't improve one metric without hurting another)
- Creates a "Pareto front" of optimal trade-offs between risk, return, and robustness
- Uses crowding distance to maintain diversity among similar strategies

### 5. **Options Backtesting**

Each strategy is evaluated by:
- Running a full options backtest on historical data
- Simulating realistic options trades with proper pricing
- Calculating portfolio-level performance metrics
- Enforcing position limits and risk management rules

## Key Features

### **Ticker Specialization**
- Each strategy is specialized to trade only one specific stock
- This allows the GA to find ticker-specific patterns and behaviors
- Prevents overfitting to a single "magic" strategy

### **Signal Compilation**
- Starts with 6,000+ primitive signals (technical indicators, market features)
- GA combines these into meaningful trading strategies
- Signals include things like moving averages, volatility measures, economic indicators

### **Walk-Forward Validation**
- Tests strategies on 5 different time periods
- Each period has separate training and testing data
- Ensures strategies work across different market conditions

### **Out-of-Sample Testing**
- After training, strategies are tested on completely unseen data
- Only strategies that pass rigorous out-of-sample tests are kept
- Prevents overfitting and ensures real-world applicability

## Configuration

The GA can be tuned via `alpha_discovery/config.py`:

```python
class GaConfig:
    population_size: int = 100      # Number of strategies per generation
    generations: int = 8            # How many generations to evolve
    elitism_rate: float = 0.1      # Keep top 10% of strategies
    mutation_rate: float = 0.2     # Mutate 20% of strategies
    setup_lengths_to_explore: List[int] = [1]  # Number of signals per strategy
```

## Output

The GA produces:
- **Pareto Front**: A set of optimal strategies with different risk/return profiles
- **Trade Ledgers**: Detailed records of all trades for each strategy
- **Performance Metrics**: Comprehensive statistics for each strategy
- **Out-of-Sample Results**: Validation on unseen data

## Why This Works

1. **Exploration**: Tests thousands of different strategy combinations
2. **Exploitation**: Focuses on promising areas of the strategy space
3. **Diversity**: Maintains variety to avoid getting stuck in local optima
4. **Validation**: Rigorous testing ensures strategies work in real markets
5. **Multi-Objective**: Balances risk, return, and robustness simultaneously

The result is a robust, automated system that discovers profitable trading strategies without human bias or overfitting to historical data.
