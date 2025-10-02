# Implementation Plan: Post-Evaluation Simulation & HartIndex Analysis

This document outlines the implementation plan for two major enhancements to the Alpha Discovery pipeline:
1.  **Part 1: Post-Evaluation Options Simulation**: A new step to run a full options backtest on the most promising candidates discovered by the GA.
2.  **Part 2: HartIndex Correlation Analysis**: A diagnostics step to measure the predictive power of the `HartIndex` by correlating it with the results of the options simulation.

---

## Part 1: Post-Evaluation Options Simulation

### Objective
To validate the top-performing setups from the forecast-based GA with a more realistic, computationally intensive options backtesting simulation. This provides a final layer of verification, bridging the gap between theoretical forecast quality and practical P&L performance.

### Workflow
1.  The existing pipeline runs to completion, generating a `final_results_df` that includes the `HartIndex` for all unique candidates.
2.  A new **Simulation Candidate Selection** step filters this DataFrame to select the "elite" setups to be simulated. This selection is based on a configurable `HartIndex` threshold and a `top_n` cap.
3.  The **Options Simulation Engine** iterates through each elite setup. For each one, it invokes the existing `run_setup_backtest_options` function to generate a detailed trade ledger based on realistic options pricing and regime-aware exit logic.
4.  The results from all individual simulations are collected and aggregated into two new summary artifacts:
    *   `simulation_summary.csv`: High-level performance metrics for each simulated setup (Sharpe, PnL, Max Drawdown, etc.).
    *   `simulation_trade_ledger.csv`: A combined ledger of all trades generated across all simulated setups.

### File Modifications & New Files

#### 1. `alpha_discovery/config.py` (Modified)
A new configuration class will be added to the `Settings` model to control the simulation process.

*   **Changes**:
    *   Add a `PostSimulationConfig` Pydantic model.
    *   Add a `simulation` attribute to the main `Settings` class.

*   **Code Snippet**:
    ```python
    class PostSimulationConfig(BaseModel):
        """Configuration for the post-discovery options simulation."""
        enabled: bool = True
        hart_index_threshold: int = 70
        top_n_candidates: int = 25

    class Settings(BaseModel):
        # ... existing settings ...
        simulation: PostSimulationConfig = PostSimulationConfig()
    ```

#### 2. `alpha_discovery/eval/post_simulation.py` (New File)
This new module will contain the core logic for running the simulation and analysis, keeping `main.py` clean.

*   **Contents**:
    *   `run_post_simulation(final_results_df, signals_df, master_df)`: The main orchestration function.
        *   Filters candidates based on the new config settings.
        *   Loops through candidates, calling the backtesting engine for each.
        *   Aggregates results and calculates summary statistics (Sharpe Ratio, Calmar, PnL, etc.).
        *   Returns a summary DataFrame and a combined ledger DataFrame.

#### 3. `main.py` (Modified)
The main pipeline will be updated to call the new post-simulation step.

*   **Changes**:
    *   Import `run_post_simulation` from the new module.
    *   After the HartIndex is calculated, add a new section: **Post-Simulation Phase**.
    *   This section will call `run_post_simulation` and pass its results to the `save_results` function.

#### 4. `alpha_discovery/reporting/artifacts.py` (Modified)
This file will be updated to handle the new simulation artifacts.

*   **Changes**:
    *   Add new functions: `write_simulation_summary(summary_df, run_dir)` and `write_simulation_ledger(ledger_df, run_dir)`.
    *   The main `save_results` function will be updated to accept the simulation results and call these new writer functions.

---

## Part 2: HartIndex Correlation Analysis

### Objective
To quantitatively assess the effectiveness of the `HartIndex` as a predictor of real-world trading performance. This creates a crucial feedback loop for validating and improving the `HartIndex` itself over time.

### Methodology
1.  The analysis will be performed after the Post-Evaluation Simulation is complete.
2.  The `simulation_summary.csv` (containing backtest metrics) will be merged with the `final_results_df` (containing the `HartIndex` and its components).
3.  **Spearman's rank correlation coefficient** will be calculated between the `hart_index` and key performance metrics from the simulation, such as:
    *   Sharpe Ratio
    *   Total PnL
    *   Win Rate
    *   Profit Factor
    *   Calmar Ratio
4.  The results will be saved in a human-readable text file.

### File Modifications & New Files

#### 1. `alpha_discovery/eval/post_simulation.py` (Modified)
The logic for the correlation analysis will be added to this new module.

*   **Changes**:
    *   Add a new function: `run_correlation_analysis(final_results_df, simulation_summary_df)`.
    *   This function will perform the merge and calculate the Spearman correlation for a predefined list of metrics.
    *   It will return a dictionary or a formatted string containing the correlation results.

#### 2. `main.py` (Modified)
The main orchestrator will call the analysis function.

*   **Changes**:
    *   After `run_post_simulation` returns, its results will be passed to `run_correlation_analysis`.
    *   The correlation results will then be passed to the `save_results` function.

#### 3. `alpha_discovery/reporting/artifacts.py` (Modified)
A new function will be added to save the analysis report.

*   **Changes**:
    *   Add a new function: `write_correlation_report(correlation_results, run_dir)`.
    *   This will save the formatted correlation data to a `.txt` file in the `diagnostics` subfolder of the run directory.
    *   The main `save_results` function will be updated to call this.

---

### Summary of New/Modified Files
*   **Modified**:
    *   `alpha_discovery/config.py`
    *   `main.py`
    *   `alpha_discovery/reporting/artifacts.py`
*   **New**:
    *   `alpha_discovery/eval/post_simulation.py`

### Summary of New Run Artifacts
*   `runs/{run_name}/simulation_summary.csv`
*   `runs/{run_name}/simulation_trade_ledger.csv`
*   `runs/{run_name}/diagnostics/hart_index_correlation.txt`
