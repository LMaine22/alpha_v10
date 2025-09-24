# Hart Index Guide

## Overview

The Hart Index is a comprehensive 0-100 trust score that evaluates how trustworthy a trading setup is based on all available metrics from the evaluation pipeline. It provides a single, easy-to-understand number that traders can use to quickly assess setup quality.

## Score Interpretation

- **85-100 (Exceptional)**: Highest confidence setups with strong performance across all metrics
- **70-84 (Strong)**: Very good setups that meet most quality criteria
- **55-69 (Moderate)**: Decent setups with acceptable performance but some weaknesses
- **40-54 (Marginal)**: Borderline setups that may have significant concerns
- **0-39 (Weak)**: Low confidence setups that should generally be avoided

## Components

The Hart Index is composed of four major categories:

### 1. Performance Components (40% weight)
- **Edge Performance (15%)**: CRPS, pinball loss, Wasserstein distance
- **Information Quality (10%)**: Information gain, entropy metrics
- **Prediction Accuracy (10%)**: Calibration, forecast accuracy
- **Risk/Reward (5%)**: Expected returns, directional probabilities

### 2. Robustness Components (30% weight)
- **Statistical Significance (10%)**: Bootstrap p-values, support counts
- **Stability & Consistency (10%)**: Cross-fold stability, regime consistency
- **Sensitivity Resilience (10%)**: Resilience to parameter perturbations

### 3. Complexity & Structure (15% weight)
- **Signal Quality (8%)**: Signal independence, redundancy measures
- **Complexity Balance (7%)**: Not too simple, not too complex

### 4. Live Trading Readiness (15% weight)
- **Trigger Reliability (8%)**: Trigger rates, dormancy analysis
- **Regime Coverage (7%)**: Performance across different market conditions

## Key Features

### Transparency
Each component of the Hart Index is tracked separately, allowing you to see exactly what contributes to the final score:
- `hart_performance_total`: Performance component subtotal
- `hart_robustness_total`: Robustness component subtotal
- `hart_complexity_total`: Complexity component subtotal
- `hart_readiness_total`: Readiness component subtotal

### Adjustments
The Hart Index includes smart adjustments:
- Penalties for setups that fail ELV quality gates
- Bonuses for setups with exceptionally high ELV scores
- Special handling for dormant and specialist setups

### Normalization
All metrics are normalized using:
- Percentile ranking for relative comparison
- Sigmoid transformations for smooth scaling
- Robust scaling to handle outliers

## Usage in Trading

### Primary Use Cases
1. **Quick Filtering**: Focus on setups with Hart Index > 70
2. **Risk Management**: Use lower scores (55-70) with smaller position sizes
3. **Comparison**: Compare multiple setups for the same ticker
4. **Monitoring**: Track how scores change over time

### Combining with Other Metrics
While the Hart Index is comprehensive, always consider:
- The specific option structure suggested
- Recent trigger history
- Current market regime
- Your personal risk tolerance

### Important Notes
- The Hart Index is relative within a run - scores may vary between different discovery runs
- A high Hart Index doesn't guarantee profits - it indicates higher statistical confidence
- Always use in conjunction with other risk management practices

## Technical Details

### Calculation Process
1. Each metric is normalized to 0-1 scale
2. Component scores are calculated by combining related metrics
3. Component weights are applied
4. Final adjustments based on quality gates
5. Scale to 0-100 and apply bounds

### Key Metrics Used
- **Performance**: CRPS, information gain, calibration MAE, Wasserstein distance
- **Robustness**: Bootstrap p-values, sensitivity analysis, stability metrics
- **Complexity**: Redundancy MI, complexity index, DFA alpha, transfer entropy
- **Readiness**: Trigger rates, coverage factors, regime breadth

### Edge Cases
- Missing metrics are handled with sensible defaults
- Extreme values are clipped to prevent single metrics from dominating
- Setups with insufficient data receive penalty adjustments
