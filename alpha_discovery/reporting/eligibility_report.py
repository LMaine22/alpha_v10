"""
Eligibility reporting for forecast-first validation.

Generates production-ready reports from EligibilityMatrix artifacts:
- Top eligible setups ranked by skill
- Calibration diagnostics per setup
- Drift detection summaries
- Regime stratification analysis
- Skill decomposition tables
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import json


def generate_eligibility_report(
    eligibility_matrix_path: Path,
    output_dir: Path,
    min_skill_vs_marginal: float = 0.01,
    max_calibration_mae: float = 0.15,
    drift_gate: bool = True,
    top_n: int = 50
) -> Dict[str, Path]:
    """
    Generate comprehensive eligibility report from EligibilityMatrix.
    
    Creates multiple report files:
    - eligible_setups.csv: Top eligible setups with full metrics
    - skill_breakdown.csv: Skill metrics decomposed by ticker/horizon
    - calibration_summary.csv: Calibration diagnostics
    - drift_analysis.csv: Drift detection results
    - report_summary.json: High-level summary statistics
    
    Args:
        eligibility_matrix_path: Path to eligibility_matrix.json
        output_dir: Directory for report outputs
        min_skill_vs_marginal: Minimum skill threshold
        max_calibration_mae: Maximum calibration error threshold
        drift_gate: Whether to enforce drift gate
        top_n: Number of top setups to report
        
    Returns:
        Dict mapping report_name -> output_path
    """
    # Load eligibility matrix
    with open(eligibility_matrix_path, 'r') as f:
        data = json.load(f)
    
    results_df = pd.DataFrame(data['results'])
    metadata = data['metadata']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    
    # 1. Eligible setups ranked by skill
    eligible_df = _filter_eligible(
        results_df,
        min_skill_vs_marginal=min_skill_vs_marginal,
        max_calibration_mae=max_calibration_mae,
        drift_gate=drift_gate
    )
    
    eligible_sorted = eligible_df.nlargest(top_n, 'skill_vs_marginal')
    
    # Format for readability
    eligible_report = _format_eligible_report(eligible_sorted)
    
    path = output_dir / "eligible_setups.csv"
    eligible_report.to_csv(path, index=False)
    outputs['eligible_setups'] = path
    
    # 2. Skill breakdown by ticker and horizon
    skill_breakdown = _create_skill_breakdown(results_df)
    
    path = output_dir / "skill_breakdown.csv"
    skill_breakdown.to_csv(path, index=True)
    outputs['skill_breakdown'] = path
    
    # 3. Calibration summary
    calib_summary = _create_calibration_summary(results_df)
    
    path = output_dir / "calibration_summary.csv"
    calib_summary.to_csv(path, index=False)
    outputs['calibration_summary'] = path
    
    # 4. Drift analysis
    drift_analysis = _create_drift_analysis(results_df)
    
    path = output_dir / "drift_analysis.csv"
    drift_analysis.to_csv(path, index=False)
    outputs['drift_analysis'] = path
    
    # 5. Summary statistics
    summary = _create_summary_stats(
        results_df, eligible_df, metadata,
        min_skill_vs_marginal, max_calibration_mae, drift_gate
    )
    
    path = output_dir / "report_summary.json"
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    outputs['summary'] = path
    
    # 6. Regime stratification
    regime_analysis = _create_regime_analysis(results_df)
    
    path = output_dir / "regime_stratification.csv"
    regime_analysis.to_csv(path, index=True)
    outputs['regime_analysis'] = path
    
    return outputs


def _filter_eligible(
    df: pd.DataFrame,
    min_skill_vs_marginal: float,
    max_calibration_mae: float,
    drift_gate: bool
) -> pd.DataFrame:
    """Filter to eligible setups."""
    # Handle empty dataframe
    if df.empty:
        return df
    
    eligible = df.copy()
    
    # Skill gate
    eligible = eligible[eligible['skill_vs_marginal'] >= min_skill_vs_marginal]
    
    # Calibration gate
    eligible = eligible[eligible['calibration_mae'] <= max_calibration_mae]
    
    # Drift gate
    if drift_gate:
        eligible = eligible[eligible['drift_passed'] == True]
    
    return eligible


def _format_eligible_report(df: pd.DataFrame) -> pd.DataFrame:
    """Format eligible setups for reporting."""
    report = pd.DataFrame()
    
    report['rank'] = range(1, len(df) + 1)
    report['ticker'] = df['ticker'].values
    report['horizon'] = df['horizon'].values
    report['n_signals'] = df['setup'].apply(len).values
    
    # Skill metrics
    report['skill_vs_marginal'] = df['skill_vs_marginal'].round(4).values
    report['skill_vs_uniform'] = df['skill_vs_uniform'].round(4).values
    
    # Forecast quality
    report['crps'] = df['crps'].round(4).values
    report['brier_score'] = df['brier_score'].round(4).values
    report['log_loss'] = df['log_loss'].round(4).values
    
    # Calibration
    report['calib_mae'] = df['calibration_mae'].round(4).values
    report['calib_ece'] = df['calibration_ece'].round(4).values
    
    # Robustness
    report['drift_auc'] = df['drift_auc'].round(3).values
    report['drift_passed'] = df['drift_passed'].values
    report['regime_similarity'] = df['regime_similarity'].round(3).values
    
    # Support
    report['n_triggers_train'] = df['n_triggers_train'].values
    report['n_triggers_test'] = df['n_triggers_test'].values
    
    # Metadata
    report['split_id'] = df['split_id'].values
    report['regime_train'] = df['regime_train'].values
    report['regime_test'] = df['regime_test'].values
    
    return report


def _create_skill_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Create skill breakdown by ticker and horizon."""
    breakdown = df.groupby(['ticker', 'horizon']).agg({
        'skill_vs_marginal': ['mean', 'median', 'std', 'count'],
        'skill_vs_uniform': ['mean', 'median'],
        'crps': ['mean', 'median'],
        'calibration_mae': ['mean', 'median'],
        'drift_passed': 'mean'
    }).round(4)
    
    # Flatten column names
    breakdown.columns = ['_'.join(col).strip() for col in breakdown.columns.values]
    
    # Sort by mean skill
    breakdown = breakdown.sort_values('skill_vs_marginal_mean', ascending=False)
    
    return breakdown


def _create_calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create calibration summary across setups."""
    summary = pd.DataFrame()
    
    # Percentiles of calibration metrics
    summary['metric'] = ['calibration_mae', 'calibration_ece', 'crps', 'brier_score', 'log_loss']
    
    for metric in summary['metric']:
        if metric in df.columns:
            summary[f'{metric}_p10'] = [df[metric].quantile(0.10)]
            summary[f'{metric}_p50'] = [df[metric].quantile(0.50)]
            summary[f'{metric}_p90'] = [df[metric].quantile(0.90)]
    
    # Transpose for readability
    summary = summary.set_index('metric').T
    summary.index.name = 'percentile'
    
    return summary.round(4).reset_index()


def _create_drift_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create drift detection analysis."""
    analysis = pd.DataFrame()
    
    analysis['ticker'] = df['ticker'].unique()
    
    for ticker in analysis['ticker']:
        ticker_data = df[df['ticker'] == ticker]
        
        analysis.loc[analysis['ticker'] == ticker, 'mean_drift_auc'] = ticker_data['drift_auc'].mean()
        analysis.loc[analysis['ticker'] == ticker, 'drift_pass_rate'] = ticker_data['drift_passed'].mean()
        analysis.loc[analysis['ticker'] == ticker, 'n_validations'] = len(ticker_data)
    
    analysis = analysis.sort_values('mean_drift_auc', ascending=False)
    
    return analysis.round(4)


def _create_regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by regime."""
    regime_stats = df.groupby('regime_test').agg({
        'skill_vs_marginal': ['mean', 'std', 'count'],
        'crps': 'mean',
        'calibration_mae': 'mean',
        'drift_passed': 'mean'
    }).round(4)
    
    regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns.values]
    
    return regime_stats


def _create_summary_stats(
    all_df: pd.DataFrame,
    eligible_df: pd.DataFrame,
    metadata: Dict,
    min_skill: float,
    max_calib: float,
    drift_gate: bool
) -> Dict:
    """Create high-level summary statistics."""
    summary = {
        'metadata': metadata,
        'filters': {
            'min_skill_vs_marginal': min_skill,
            'max_calibration_mae': max_calib,
            'drift_gate_enabled': drift_gate
        },
        'counts': {
            'total_validations': len(all_df),
            'eligible_setups': len(eligible_df),
            'eligibility_rate': float(len(eligible_df) / len(all_df)) if len(all_df) > 0 else 0.0,
            'unique_tickers': int(all_df['ticker'].nunique()),
            'unique_horizons': int(all_df['horizon'].nunique())
        },
        'skill_metrics': {
            'mean_skill_vs_marginal': float(all_df['skill_vs_marginal'].mean()),
            'median_skill_vs_marginal': float(all_df['skill_vs_marginal'].median()),
            'eligible_mean_skill': float(eligible_df['skill_vs_marginal'].mean()) if len(eligible_df) > 0 else 0.0,
            'eligible_median_skill': float(eligible_df['skill_vs_marginal'].median()) if len(eligible_df) > 0 else 0.0
        },
        'calibration_metrics': {
            'mean_crps': float(all_df['crps'].mean()),
            'mean_calibration_mae': float(all_df['calibration_mae'].mean()),
            'mean_calibration_ece': float(all_df['calibration_ece'].mean())
        },
        'robustness_metrics': {
            'drift_pass_rate': float(all_df['drift_passed'].mean()),
            'mean_drift_auc': float(all_df['drift_auc'].mean()),
            'mean_regime_similarity': float(all_df['regime_similarity'].mean())
        },
        'top_tickers': all_df.groupby('ticker')['skill_vs_marginal'].mean().nlargest(10).to_dict(),
        'by_horizon': all_df.groupby('horizon')['skill_vs_marginal'].agg(['mean', 'count']).to_dict()
    }
    
    return summary


def print_eligibility_summary(summary_path: Path):
    """
    Print human-readable summary to console.
    
    Args:
        summary_path: Path to report_summary.json
    """
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("=" * 80)
    print("ELIGIBILITY REPORT SUMMARY")
    print("=" * 80)
    
    counts = summary['counts']
    print(f"\n📊 Validation Coverage:")
    print(f"  Total validations: {counts['total_validations']}")
    print(f"  Eligible setups: {counts['eligible_setups']}")
    print(f"  Eligibility rate: {counts['eligibility_rate']:.1%}")
    print(f"  Unique tickers: {counts['unique_tickers']}")
    print(f"  Unique horizons: {counts['unique_horizons']}")
    
    skill = summary['skill_metrics']
    print(f"\n🎯 Skill Metrics:")
    print(f"  All setups - Mean skill: {skill['mean_skill_vs_marginal']:.4f}")
    print(f"  All setups - Median skill: {skill['median_skill_vs_marginal']:.4f}")
    if counts['eligible_setups'] > 0:
        print(f"  Eligible - Mean skill: {skill['eligible_mean_skill']:.4f}")
        print(f"  Eligible - Median skill: {skill['eligible_median_skill']:.4f}")
    
    calib = summary['calibration_metrics']
    print(f"\n📏 Calibration:")
    print(f"  Mean CRPS: {calib['mean_crps']:.4f}")
    print(f"  Mean Calibration MAE: {calib['mean_calibration_mae']:.4f}")
    print(f"  Mean Calibration ECE: {calib['mean_calibration_ece']:.4f}")
    
    robust = summary['robustness_metrics']
    print(f"\n🛡️  Robustness:")
    print(f"  Drift pass rate: {robust['drift_pass_rate']:.1%}")
    print(f"  Mean drift AUC: {robust['mean_drift_auc']:.3f}")
    print(f"  Mean regime similarity: {robust['mean_regime_similarity']:.3f}")
    
    print(f"\n🏆 Top 5 Tickers by Skill:")
    for i, (ticker, skill_val) in enumerate(list(summary['top_tickers'].items())[:5], 1):
        print(f"  {i}. {ticker}: {skill_val:.4f}")
    
    print("\n" + "=" * 80)


# ============================================================================
# Calibration Visualization Helpers
# ============================================================================

def export_reliability_data(
    eligibility_matrix_path: Path,
    output_path: Path,
    top_n: int = 20
):
    """
    Export reliability curve data for top setups.
    
    Creates CSV with binned calibration data that can be used for
    reliability diagrams / calibration plots.
    
    Args:
        eligibility_matrix_path: Path to eligibility_matrix.json
        output_path: Output CSV path
        top_n: Number of top setups to include
    """
    # Load data
    with open(eligibility_matrix_path, 'r') as f:
        data = json.load(f)
    
    results_df = pd.DataFrame(data['results'])
    
    # Get top setups by skill
    top_setups = results_df.nlargest(top_n, 'skill_vs_marginal')
    
    # Extract band probabilities and edges for each setup
    reliability_data = []
    
    for idx, row in top_setups.iterrows():
        setup_id = f"{row['ticker']}_H{row['horizon']}_{row['split_id']}"
        
        # band_probs and band_edges should be lists
        band_probs = row.get('band_probs', [])
        band_edges = row.get('band_edges', [])
        
        if band_probs and band_edges:
            # Create bin centers
            bin_centers = [(band_edges[i] + band_edges[i+1]) / 2 
                          for i in range(len(band_edges) - 1)]
            
            for bin_idx, (center, prob) in enumerate(zip(bin_centers, band_probs)):
                reliability_data.append({
                    'setup_id': setup_id,
                    'ticker': row['ticker'],
                    'horizon': row['horizon'],
                    'bin_idx': bin_idx,
                    'bin_center': center,
                    'predicted_prob': prob,
                    'skill_vs_marginal': row['skill_vs_marginal'],
                    'calibration_mae': row['calibration_mae']
                })
    
    reliability_df = pd.DataFrame(reliability_data)
    reliability_df.to_csv(output_path, index=False)
    
    print(f"Exported reliability data for {len(top_setups)} setups to {output_path}")


__all__ = [
    'generate_eligibility_report',
    'print_eligibility_summary',
    'export_reliability_data',
]
