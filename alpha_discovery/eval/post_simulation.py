# alpha_discovery/eval/post_simulation.py
"""
Handles the post-discovery options backtest simulation and HartIndex correlation analysis.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm
from scipy.stats import spearmanr

from ..config import settings
from ..engine.backtester import run_setup_backtest_options


def _calculate_summary_statistics(ledger: pd.DataFrame) -> Dict:
    """Calculates performance summary statistics from a trade ledger."""
    if ledger.empty:
        return {
            "pnl_total": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
            "calmar_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "n_trades": 0
        }

    # Use realized PnL for closed trades
    realized_pnl = ledger[ledger['exit_date'].notna()]['pnl_dollars']
    
    if realized_pnl.empty:
        return {
            "pnl_total": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
            "calmar_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "n_trades": len(ledger)
        }

    pnl_total = realized_pnl.sum()
    daily_pnl = ledger.groupby(pd.to_datetime(ledger['exit_date']).dt.date)['pnl_dollars'].sum()
    
    # Sharpe Ratio
    sharpe = 0.0
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)

    # Sortino Ratio
    downside_std = daily_pnl[daily_pnl < 0].std()
    sortino = 0.0
    if downside_std > 0:
        sortino = (daily_pnl.mean() / downside_std) * np.sqrt(252)

    # Max Drawdown & Calmar
    cumulative_pnl = realized_pnl.cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    calmar = 0.0
    if max_drawdown > 0:
        calmar = pnl_total / max_drawdown
        
    # Win Rate & Profit Factor
    wins = realized_pnl[realized_pnl > 0]
    losses = realized_pnl[realized_pnl <= 0]
    win_rate = len(wins) / len(realized_pnl) if len(realized_pnl) > 0 else 0.0
    
    total_gains = wins.sum()
    total_losses = abs(losses.sum())
    profit_factor = total_gains / total_losses if total_losses > 0 else np.inf

    return {
        "pnl_total": pnl_total,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(realized_pnl)
    }

def run_post_simulation(
    final_results_df: pd.DataFrame, 
    signals_df: pd.DataFrame, 
    master_df: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Runs a full options backtest simulation on the top candidates from the GA.
    """
    sim_cfg = settings.simulation
    if not sim_cfg.enabled:
        print("\n--- Post-evaluation simulation is disabled. Skipping. ---")
        return None, None

    print(f"\n--- Running Post-Evaluation Options Simulation ---")
    
    # 1. Select candidates
    candidates = final_results_df[
        final_results_df['hart_index'] >= sim_cfg.hart_index_threshold
    ].sort_values('hart_index', ascending=False).head(sim_cfg.top_n_candidates)

    if candidates.empty:
        print("No candidates met the HartIndex threshold for simulation.")
        return None, None
        
    print(f"Selected {len(candidates)} elite candidates for simulation (HartIndex >= {sim_cfg.hart_index_threshold}, Top {sim_cfg.top_n_candidates})")

    # 2. Run simulation
    all_summaries = []
    all_ledgers = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Simulating Setups"):
        try:
            # The 'individual' column is stored as a string, e.g., "('TICKER', ('SIG_001',))"
            individual_tuple = eval(row['individual'])
            ticker, setup_signals = individual_tuple
        except (TypeError, SyntaxError):
            print(f"Could not parse individual: {row['individual']}")
            continue

        # Run backtest for both long and short directions
        for direction in ["long", "short"]:
            ledger = run_setup_backtest_options(
                setup_signals=list(setup_signals),
                signals_df=signals_df,
                master_df=master_df,
                direction=direction,
                tickers_to_run=[ticker] # Simulate only for the specific ticker
            )

            if not ledger.empty:
                summary_stats = _calculate_summary_statistics(ledger)
                summary_stats['individual'] = str(individual_tuple)
                summary_stats['direction'] = direction
                all_summaries.append(summary_stats)

                ledger['individual'] = str(individual_tuple)
                ledger['direction'] = direction
                all_ledgers.append(ledger)

    if not all_summaries:
        print("Simulation did not produce any trades.")
        return None, None

    summary_df = pd.DataFrame(all_summaries)
    ledger_df = pd.concat(all_ledgers, ignore_index=True)

    return summary_df, ledger_df

def run_correlation_analysis(
    final_results_df: pd.DataFrame, 
    simulation_summary_df: Optional[pd.DataFrame]
) -> Optional[str]:
    """
    Analyzes the correlation between HartIndex and simulation performance.
    """
    if simulation_summary_df is None or simulation_summary_df.empty:
        return "No simulation data to run correlation analysis."

    # Merge dataframes
    merged_df = pd.merge(
        final_results_df,
        simulation_summary_df,
        on='individual',
        how='inner'
    )
    
    if merged_df.empty:
        return "Could not merge GA results with simulation results."

    report = ["HartIndex Correlation Analysis"]
    report.append("="*30)
    
    corr_metrics = [
        'sharpe_ratio', 'pnl_total', 'win_rate', 
        'profit_factor', 'calmar_ratio'
    ]

    for metric in corr_metrics:
        if metric in merged_df.columns:
            # Use spearman's rank correlation
            corr, p_value = spearmanr(merged_df['hart_index'], merged_df[metric], nan_policy='omit')
            report.append(f"HartIndex vs. {metric}:")
            report.append(f"  Spearman Correlation: {corr:.4f}")
            report.append(f"  P-value: {p_value:.4f}")
            report.append("")

    return "\n".join(report)
