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
from .selection import get_valid_trigger_dates
from .validation import _forward_returns
from ..eval.info_metrics import causality


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
    Fixed version that properly handles numpy string types in individual parsing.
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
        
    print(f"Selected {len(candidates)} elite candidates for simulation")

    # 2. Run simulation with fixed parsing
    all_summaries = []
    all_ledgers = []

    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Simulating Setups"):
        try:
            # FIXED: Handle both string and tuple representations
            individual_raw = row['individual']
            
            if isinstance(individual_raw, str):
                # Parse string representation: "('TICKER', ('SIG_001', 'SIG_002'))"
                individual_tuple = eval(individual_raw)
            elif isinstance(individual_raw, tuple):
                # Already a tuple
                individual_tuple = individual_raw
            else:
                print(f"Could not parse individual: {individual_raw} (type: {type(individual_raw)})")
                continue
            
            # FIXED: Convert numpy strings to Python strings recursively
            def _normalize_strings(obj):
                if isinstance(obj, np.str_):
                    return str(obj)
                elif isinstance(obj, tuple):
                    return tuple(_normalize_strings(x) for x in obj)
                elif isinstance(obj, list):
                    return [_normalize_strings(x) for x in obj]
                else:
                    return obj
            
            individual_tuple = _normalize_strings(individual_tuple)
            
            # Unpack, handling both 2 and 3 element tuples
            if len(individual_tuple) == 3:
                ticker, setup_signals, _ = individual_tuple  # Ignore horizon
            elif len(individual_tuple) == 2:
                ticker, setup_signals = individual_tuple
            else:
                print(f"Could not parse individual: {individual_tuple} - Unexpected length")
                continue
            
        except (TypeError, SyntaxError, ValueError) as e:
            print(f"Could not parse individual: {row['individual']} - Error: {e}")
            continue

        # Run backtest for both directions
        for direction in ["long", "short"]:
            ledger = run_setup_backtest_options(
                setup_signals=list(setup_signals),
                signals_df=signals_df,
                master_df=master_df,
                direction=direction,
                tickers_to_run=[ticker]
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


def _calculate_enhanced_causality_for_finalists(
    final_results_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    master_df: pd.DataFrame,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Calculate enhanced causality metrics (with significance testing) for top N candidates only.
    This addresses the sparse TE p-values and Granger issues by focusing computation.
    """
    print(f"\n--- Calculating Enhanced Causality for Top {top_n} Candidates ---")
    
    # Select top candidates by HartIndex
    top_candidates = final_results_df.nlargest(top_n, 'hart_index')
    enhanced_results = final_results_df.copy()
    
    # Initialize enhanced causality columns
    enhanced_results['transfer_entropy_p_value'] = np.nan
    enhanced_results['granger_p_value'] = np.nan
    enhanced_results['transfer_entropy_enhanced'] = np.nan
    
    for idx, row in tqdm(top_candidates.iterrows(), total=len(top_candidates), desc="Enhanced Causality"):
        try:
            # Parse individual
            if isinstance(row['individual'], str):
                individual_tuple = eval(row['individual'])
            else:
                individual_tuple = row['individual']
            
            # Unpack, handling both 2 and 3 element tuples
            if len(individual_tuple) == 3:
                ticker, setup_signals, _ = individual_tuple  # Ignore horizon
            elif len(individual_tuple) == 2:
                ticker, setup_signals = individual_tuple
            else:
                print(f"Could not parse individual for causality: {individual_tuple} - Unexpected length")
                continue

            # Get trigger dates
            trigger_dates = get_valid_trigger_dates(
                signals_df, list(setup_signals), settings.validation.min_support
            )
            
            if trigger_dates.empty:
                continue
            
            # Get returns for best horizon
            best_h = int(row.get('best_horizon', settings.forecast.default_horizon))
            price_field = settings.forecast.price_field
            
            returns = _forward_returns(master_df, ticker, best_h, price_field)
            
            # Create trigger series aligned with returns
            trigger_series = pd.Series(0.0, index=returns.index)
            trigger_series.loc[trigger_dates.intersection(returns.index)] = 1.0
            
            # Get overlapping valid data
            common_idx = returns.index.intersection(trigger_series.index)
            returns_clean = returns.loc[common_idx].replace([np.inf, -np.inf], np.nan).dropna()
            trigger_clean = trigger_series.loc[returns_clean.index]
            
            if len(returns_clean) < 60 or returns_clean.var() == 0 or trigger_clean.var() == 0:
                continue
            
            # Enhanced Transfer Entropy with significance testing
            try:
                te_result = causality.transfer_entropy_causality(
                    trigger_clean.values, returns_clean.values,
                    lag=1, bins=5, n_perm=200, block=10, random_state=42
                )
                enhanced_results.loc[idx, 'transfer_entropy_enhanced'] = te_result.get('te_x_to_y', np.nan)
                enhanced_results.loc[idx, 'transfer_entropy_p_value'] = te_result.get('p_value', np.nan)
            except Exception as e:
                print(f"TE calculation failed for {ticker}: {e}")
            
            # Enhanced Granger Causality with robust checks
            try:
                # Additional length and rank checks
                if len(returns_clean) >= 60 and np.linalg.matrix_rank(
                    np.column_stack([returns_clean.values[:-1], trigger_clean.values[:-1]])
                ) >= 2:
                    granger_result = causality.granger_causality(
                        y_values=returns_clean.values,
                        x_values=trigger_clean.values,
                        max_lag=min(8, len(returns_clean) // 10),
                        criterion="bic"
                    )
                    enhanced_results.loc[idx, 'granger_p_value'] = granger_result.get('p_value', np.nan)
            except Exception as e:
                print(f"Granger calculation failed for {ticker}: {e}")
                
        except Exception as e:
            print(f"Enhanced causality failed for row {idx}: {e}")
            continue
    
    # Update original columns for successfully computed values
    mask = enhanced_results['transfer_entropy_p_value'].notna()
    enhanced_results.loc[mask, 'transfer_entropy_p_value_raw'] = enhanced_results.loc[mask, 'transfer_entropy_p_value']
    
    mask = enhanced_results['granger_p_value'].notna()
    enhanced_results.loc[mask, 'granger_p_value_raw'] = enhanced_results.loc[mask, 'granger_p_value']
    
    n_enhanced = enhanced_results['transfer_entropy_p_value'].notna().sum()
    n_granger = enhanced_results['granger_p_value'].notna().sum()
    
    print(f"Enhanced causality computed: {n_enhanced} TE p-values, {n_granger} Granger p-values")
    
    return enhanced_results
