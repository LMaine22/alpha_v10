# Alpha Discovery Metrics Reference Guide

This document provides a comprehensive reference for all metrics available in the `alpha_discovery/eval/metrics/` directory, organized by their use in the genetic algorithm (GA) optimization process and their mathematical properties.

## Table of Contents

1. [GA Objective Metrics](#ga-objective-metrics)
2. [Distribution Metrics](#distribution-metrics)
3. [Information Theory Metrics](#information-theory-metrics)
4. [Causality Metrics](#causality-metrics)
5. [Dynamics Metrics](#dynamics-metrics)
6. [Complexity Metrics](#complexity-metrics)
7. [Robustness Metrics](#robustness-metrics)
8. [Regime Metrics](#regime-metrics)
9. [Topological Data Analysis](#topological-data-analysis)
10. [Aggregation Methods](#aggregation-methods)

---

## GA Objective Metrics

These metrics are directly used in the genetic algorithm as optimization objectives (as defined in `alpha_discovery/config.py`).

### `crps_neg`
- **Description**: Continuous Ranked Probability Score (negated for maximization)
- **Purpose**: Evaluates the accuracy of a probabilistic forecast by comparing the entire predicted distribution against the observed outcome
- **Interpretation**: Lower values (higher when negated) indicate better forecast accuracy and calibration
- **File**: `distribution.py`

### `pinball_loss_neg_q10` & `pinball_loss_neg_q90`
- **Description**: Pinball loss at 10th and 90th percentiles (negated for maximization)
- **Purpose**: Evaluates the accuracy of specific quantiles of a probabilistic forecast
- **Interpretation**: Lower values (higher when negated) indicate better forecasts of downside and upside risks
- **File**: `distribution.py`

### `info_gain`
- **Description**: Information Gain (KL divergence between empirical and forecast distributions)
- **Purpose**: Measures how much additional information a forecast provides over a baseline
- **Interpretation**: Higher values indicate more informative forecasts
- **File**: `info_theory.py`

### `w1_effect`
- **Description**: 1-Wasserstein distance (Earth Mover's Distance)
- **Purpose**: Measures the minimal "work" needed to transform one distribution into another
- **Interpretation**: Higher values indicate the forecast distribution better matches the empirical distribution
- **File**: `distribution.py`

### `dfa_alpha_closeness_neg`
- **Description**: Detrended Fluctuation Analysis alpha exponent closeness to target (negated for maximization)
- **Purpose**: Measures how close a time series' persistence structure is to a desired target (0.65)
- **Interpretation**: Lower values (higher when negated) indicate time series with desirable autocorrelation properties
- **File**: `dynamics.py`

### `sensitivity_scan_neg`
- **Description**: Sensitivity to parameter changes (negated for maximization)
- **Purpose**: Evaluates how robust a model is to slight perturbations in parameters
- **Interpretation**: Lower values (higher when negated) indicate models that are less sensitive to small changes
- **File**: `robustness.py`

### `redundancy_neg`
- **Description**: Signal redundancy based on mutual information (negated for maximization)
- **Purpose**: Measures information overlap between signals in a model
- **Interpretation**: Lower values (higher when negated) indicate more independent, non-redundant signals
- **File**: `ga_core.py` (calculated directly in GA code using mutual_information)

### `transfer_entropy_neg`
- **Description**: Transfer entropy from market returns to signal (negated for maximization)
- **Purpose**: Measures directional information flow and causal influence
- **Interpretation**: Higher values (when negated) indicate the signal contains more predictive information about future returns
- **File**: `causality.py`

---

## Distribution Metrics

Metrics focused on evaluating probability distributions and their accuracy.

### `crps`
- **Description**: Continuous Ranked Probability Score
- **Purpose**: Proper scoring rule for evaluating probabilistic forecasts
- **Math**: Integral of squared differences between forecast and observation CDFs
- **File**: `distribution.py`

### `pinball_loss`
- **Description**: Quantile regression loss function
- **Purpose**: Evaluates accuracy of specific quantiles
- **Math**: Asymmetric linear penalty based on whether observation is above/below forecast quantile
- **File**: `distribution.py`

### `calibration_mae`
- **Description**: Mean Absolute Error of calibration
- **Purpose**: Measures reliability of probability forecasts
- **Math**: Mean absolute difference between forecast probability and observed frequency
- **File**: `distribution.py`

### `wasserstein1`
- **Description**: 1-Wasserstein distance (Earth Mover's Distance)
- **Purpose**: Computes minimum "work" to transform one distribution into another
- **Math**: Integral of absolute difference between CDFs
- **File**: `distribution.py`

---

## Information Theory Metrics

Metrics based on information theory principles.

### `entropy`
- **Description**: Shannon entropy
- **Purpose**: Measures uncertainty in a probability distribution
- **Math**: -∑ p(x) log p(x)
- **File**: `info_theory.py`

### `conditional_entropy`
- **Description**: Entropy of Y given X
- **Purpose**: Measures remaining uncertainty in Y after observing X
- **Math**: H(Y|X) = -∑ p(x,y) log p(y|x)
- **File**: `info_theory.py`

### `info_gain`
- **Description**: KL divergence between empirical and forecast distributions
- **Purpose**: Measures information added by a forecast
- **Math**: KL(P_emp || P_forecast)
- **File**: `info_theory.py`

### `mutual_information`
- **Description**: Shared information between two variables
- **Purpose**: Quantifies how much knowing one variable reduces uncertainty about another
- **Math**: I(X;Y) = H(X) - H(X|Y)
- **File**: `info_theory.py`

### `transfer_entropy`
- **Description**: Directional information flow from one time series to another
- **Purpose**: Measures predictive causality
- **Math**: TE(X→Y) = H(Yt|Yt-1) - H(Yt|Yt-1,Xt-1)
- **File**: `info_theory.py`

---

## Causality Metrics

Metrics focused on determining causal relationships between variables.

### `granger_causality`
- **Description**: Linear causality testing using F-tests
- **Purpose**: Tests if past values of X help predict future values of Y
- **Math**: Compares variance of residuals between restricted and unrestricted models
- **File**: `causality.py`

### `ccm` (Convergent Cross Mapping)
- **Description**: Nonlinear dynamic causality detection
- **Purpose**: Detects causality in complex nonlinear systems
- **Math**: Tests ability to predict states of one variable from the manifold reconstruction of another
- **File**: `causality.py`

### `transfer_entropy_causality`
- **Description**: Information-theoretic causal influence with significance testing
- **Purpose**: Measures directed information flow with statistical validation
- **Math**: TE with block-permutation significance testing
- **File**: `causality.py`

---

## Dynamics Metrics

Metrics that analyze the dynamical properties of time series data.

### `dfa_alpha`
- **Description**: Detrended Fluctuation Analysis scaling exponent
- **Purpose**: Characterizes long-range correlations in time series
- **Math**: Slope of log fluctuation vs. log time scale
- **Interpretation**: 0.5=random walk, >0.5=persistent, <0.5=anti-persistent
- **File**: `dynamics.py`

### `rqa_metrics`
- **Description**: Recurrence Quantification Analysis metrics
- **Purpose**: Analyzes recurrence patterns in phase space
- **Math**: Various statistics from recurrence plots
- **File**: `dynamics.py`

---

## Complexity Metrics

Metrics that measure the complexity and structure of time series data.

### `sample_entropy`
- **Description**: Regularity/predictability measure robust to noise
- **Purpose**: Quantifies the complexity of a time series
- **Math**: Negative logarithm of the conditional probability that subseries remain similar
- **File**: `complexity.py`

### `approximate_entropy`
- **Description**: Regularity measure (less robust than sample entropy)
- **Purpose**: Quantifies unpredictability of fluctuations
- **Math**: Relative frequency of similar patterns in the time series
- **File**: `complexity.py`

### `permutation_entropy`
- **Description**: Complexity based on ordinal patterns
- **Purpose**: Measures complexity of order relationships in time series
- **Math**: Shannon entropy of the distribution of ordinal patterns
- **File**: `complexity.py`

### `multiscale_entropy`
- **Description**: Sample entropy across multiple time scales
- **Purpose**: Distinguishes between complex and random processes
- **Math**: Sample entropy on coarse-grained time series
- **File**: `complexity.py`

### `complexity_index`
- **Description**: Composite measure of signal complexity
- **Purpose**: Single scalar index combining multiple complexity aspects
- **Math**: Combination of entropy and correlation properties
- **File**: `complexity.py`

---

## Robustness Metrics

Metrics that evaluate the stability and robustness of forecasts or models.

### `moving_block_bootstrap`
- **Description**: Time series resampling preserving temporal dependence
- **Purpose**: Estimates uncertainty in time series statistics
- **Math**: Resamples blocks of observations to preserve autocorrelation
- **File**: `robustness.py`

### `sensitivity_scan`
- **Description**: Evaluates model response to parameter perturbations
- **Purpose**: Tests robustness to small changes in model parameters
- **Math**: Summarizes performance variance under perturbations
- **File**: `robustness.py`

### `tscv_robustness`
- **Description**: Time-series cross-validation stability measure
- **Purpose**: Tests consistency of performance across validation windows
- **Math**: Variance of performance metrics across time-based CV folds
- **File**: `robustness.py`

### `page_hinkley`
- **Description**: Sequential change detection test
- **Purpose**: Detects abrupt changes in time series properties
- **Math**: Cumulative sum of deviations from a target value
- **File**: `robustness.py`

---

## Regime Metrics

Metrics related to market regime identification and analysis.

### `fit_hmm_gaussian`
- **Description**: Gaussian Hidden Markov Model fitting
- **Purpose**: Identifies distinct market regimes from data
- **Math**: Expectation-Maximization on multivariate Gaussian HMM
- **File**: `regime.py`

### `detect_regimes`
- **Description**: Assigns regime labels to time series
- **Purpose**: Classifies periods into distinct market conditions
- **Math**: Viterbi algorithm on fitted HMM
- **File**: `regime.py`

### `regime_metrics`
- **Description**: Characterizes properties of different regimes
- **Purpose**: Quantifies statistical properties of each regime
- **Math**: Conditional statistics within each regime
- **File**: `regime.py`

### `worst_regime`
- **Description**: Identifies the most challenging market regime
- **Purpose**: Risk analysis focused on worst-case performance
- **Math**: Performance statistics in lowest-performing regime
- **File**: `regime.py`

---

## Topological Data Analysis

Metrics based on persistent homology and topological features.

### `persistent_homology_h0`
- **Description**: 0-dimensional persistence diagram
- **Purpose**: Captures connected components' birth/death
- **Math**: Records when components appear and merge
- **File**: `tda.py`

### `h0_landscape_vector`
- **Description**: Vector representation of H0 persistence landscape
- **Purpose**: Creates feature vector from topological features
- **Math**: Transforms persistence diagram into functional summary
- **File**: `tda.py`

### `wasserstein1_h0`
- **Description**: 1-Wasserstein distance between persistence diagrams
- **Purpose**: Compares topological signatures of time series
- **Math**: Optimal transport distance between persistence diagrams
- **File**: `tda.py`

### `bottleneck_h0`
- **Description**: Bottleneck distance between persistence diagrams
- **Purpose**: Max-min distance between diagram points
- **Math**: Maximum distance in optimal matching between diagrams
- **File**: `tda.py`

---

## Aggregation Methods

Methods for combining metrics and handling multiple validation folds.

### `aggregate`
- **Description**: General-purpose robust aggregation
- **Purpose**: Combines metrics across folds/iterations
- **Math**: Configurable method (mean, median, etc.)
- **File**: `aggregate.py`

### `median_mad`
- **Description**: Median and Median Absolute Deviation
- **Purpose**: Robust central tendency and dispersion
- **Math**: median(x) and median(|x - median(x)|)
- **File**: `aggregate.py`

### `trimmed_mean`
- **Description**: Mean after removing extremes
- **Purpose**: Robust central tendency without outliers
- **Math**: Mean of middle percentiles
- **File**: `aggregate.py`

### `huber_mean`
- **Description**: Robust mean using Huber loss
- **Purpose**: Down-weights outliers in mean calculation
- **Math**: Minimizes Huber loss function
- **File**: `aggregate.py`

### `hodges_lehmann`
- **Description**: Median of pairwise means
- **Purpose**: Highly robust estimator of location
- **Math**: Median of (xi + xj)/2 for all i,j
- **File**: `aggregate.py`

### `rank_stability`
- **Description**: Stability of rankings across folds
- **Purpose**: Measures consistency of relative performance
- **Math**: Variance of ranks across validation sets
- **File**: `aggregate.py`

### `jackknife_leave_one_out`
- **Description**: Leave-one-out resampling
- **Purpose**: Estimates variance and bias of statistics
- **Math**: Recalculates statistic with each observation removed
- **File**: `aggregate.py`

---

## Guidelines for Adding New Metrics

When adding new metrics to the system:

1. **Placement**: Add the metric to the appropriate module based on its mathematical properties.
2. **Documentation**: Include detailed docstrings with mathematical descriptions and references.
3. **Robustness**: Ensure the metric handles edge cases (NaN, empty arrays, etc.) gracefully.
4. **Testing**: Write unit tests covering normal usage and edge cases.
5. **Registration**: Add exports to `__init__.py` for easy importing.
6. **GA Integration**: To use as a GA objective, add a negated version in `ga_core.py` if lower values are better.
7. **Configuration**: Add the metric name to `settings.ga.objectives` in `config.py` to use in optimization.

## Using Custom Metrics in the GA

To add a custom metric to the genetic algorithm:

1. Implement the metric function in the appropriate module.
2. Add a negated version (if needed) in the `_calculate_objectives` function in `ga_core.py`.
3. Add the metric name to the objectives list in `config.py`.
4. Test thoroughly to ensure proper integration.

Example:
```python
# In config.py
class GaConfig(BaseModel):
    objectives: List[str] = [
        "crps_neg",
        "your_new_metric_neg",  # Add your metric here
        ...
    ]
```