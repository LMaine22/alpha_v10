"""
Tests for eligibility reporting.

Coverage:
- Report generation from eligibility matrix
- Filtering logic
- Summary statistics
- Report file creation
"""

import pytest
import json
import pandas as pd
import numpy as np
from pathlib import Path
from alpha_discovery.reporting.eligibility_report import (
    generate_eligibility_report,
    _filter_eligible,
    _format_eligible_report,
    _create_skill_breakdown,
    _create_summary_stats
)


@pytest.fixture
def sample_eligibility_matrix(tmp_path):
    """Create a sample eligibility matrix for testing."""
    results = []
    
    tickers = ['AAPL', 'TSLA', 'MSFT']
    horizons = [1, 5, 21]
    
    np.random.seed(42)
    
    for ticker in tickers:
        for horizon in horizons:
            for split_idx in range(3):
                results.append({
                    'ticker': ticker,
                    'horizon': horizon,
                    'split_id': f'PAWF_v1|OUTER:202{split_idx+1}01|H:{horizon}|E:normal|P:5|EMB:10|REG:R1',
                    'setup': [f'sig_{ticker}_{i}' for i in range(2)],
                    'skill_vs_marginal': np.random.uniform(-0.05, 0.10),
                    'skill_vs_uniform': np.random.uniform(0.0, 0.15),
                    'crps': np.random.uniform(0.2, 0.5),
                    'brier_score': np.random.uniform(0.1, 0.3),
                    'log_loss': np.random.uniform(0.3, 0.7),
                    'calibration_mae': np.random.uniform(0.05, 0.25),
                    'calibration_ece': np.random.uniform(0.03, 0.20),
                    'drift_auc': np.random.uniform(0.45, 0.65),
                    'drift_passed': np.random.choice([True, False], p=[0.7, 0.3]),
                    'regime_similarity': np.random.uniform(0.5, 0.9),
                    'n_triggers_train': np.random.randint(10, 100),
                    'n_triggers_test': np.random.randint(5, 50),
                    'regime_train': f'R{np.random.randint(0, 5)}',
                    'regime_test': f'R{np.random.randint(0, 5)}',
                    'band_probs': list(np.random.dirichlet(np.ones(5))),
                    'band_edges': [-999, -0.05, 0.0, 0.05, 0.10, 999]
                })
    
    matrix = {
        'metadata': {
            'timestamp': '2025-09-30T12:00:00',
            'n_outer_splits': 3,
            'n_inner_folds': 3,
            'seed': 42
        },
        'results': results
    }
    
    # Save to temp file
    matrix_path = tmp_path / "eligibility_matrix.json"
    with open(matrix_path, 'w') as f:
        json.dump(matrix, f)
    
    return matrix_path, pd.DataFrame(results)


class TestFilterEligible:
    """Test filtering logic."""
    
    def test_filter_skill_gate(self):
        """Test filtering by skill threshold."""
        df = pd.DataFrame({
            'skill_vs_marginal': [0.02, 0.005, -0.01, 0.03],
            'calibration_mae': [0.10, 0.10, 0.10, 0.10],
            'drift_passed': [True, True, True, True]
        })
        
        eligible = _filter_eligible(
            df,
            min_skill_vs_marginal=0.01,
            max_calibration_mae=0.15,
            drift_gate=False
        )
        
        # Should keep only rows with skill >= 0.01
        assert len(eligible) == 2
        assert all(eligible['skill_vs_marginal'] >= 0.01)
    
    def test_filter_calibration_gate(self):
        """Test filtering by calibration threshold."""
        df = pd.DataFrame({
            'skill_vs_marginal': [0.02, 0.02, 0.02, 0.02],
            'calibration_mae': [0.10, 0.20, 0.05, 0.25],
            'drift_passed': [True, True, True, True]
        })
        
        eligible = _filter_eligible(
            df,
            min_skill_vs_marginal=0.01,
            max_calibration_mae=0.15,
            drift_gate=False
        )
        
        # Should keep only rows with calib_mae <= 0.15
        assert len(eligible) == 2
        assert all(eligible['calibration_mae'] <= 0.15)
    
    def test_filter_drift_gate(self):
        """Test filtering by drift gate."""
        df = pd.DataFrame({
            'skill_vs_marginal': [0.02, 0.02, 0.02, 0.02],
            'calibration_mae': [0.10, 0.10, 0.10, 0.10],
            'drift_passed': [True, False, True, False]
        })
        
        eligible = _filter_eligible(
            df,
            min_skill_vs_marginal=0.01,
            max_calibration_mae=0.15,
            drift_gate=True
        )
        
        # Should keep only rows that passed drift
        assert len(eligible) == 2
        assert all(eligible['drift_passed'] == True)
    
    def test_filter_combined(self):
        """Test combined filtering."""
        df = pd.DataFrame({
            'skill_vs_marginal': [0.02, 0.005, 0.03, 0.04],
            'calibration_mae': [0.10, 0.10, 0.20, 0.10],
            'drift_passed': [True, True, True, False]
        })
        
        eligible = _filter_eligible(
            df,
            min_skill_vs_marginal=0.01,
            max_calibration_mae=0.15,
            drift_gate=True
        )
        
        # Only first row passes all gates
        assert len(eligible) == 1
        assert eligible.iloc[0]['skill_vs_marginal'] == 0.02


class TestFormatEligibleReport:
    """Test report formatting."""
    
    def test_format_eligible_report(self):
        """Test formatting eligible setups for report."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'TSLA'],
            'horizon': [5, 21],
            'setup': [['sig1', 'sig2'], ['sig3']],
            'skill_vs_marginal': [0.02, 0.03],
            'skill_vs_uniform': [0.05, 0.06],
            'crps': [0.3, 0.25],
            'brier_score': [0.15, 0.12],
            'log_loss': [0.4, 0.35],
            'calibration_mae': [0.10, 0.08],
            'calibration_ece': [0.05, 0.04],
            'drift_auc': [0.52, 0.58],
            'drift_passed': [True, True],
            'regime_similarity': [0.7, 0.8],
            'n_triggers_train': [50, 60],
            'n_triggers_test': [25, 30],
            'split_id': ['split_1', 'split_2'],
            'regime_train': ['R0', 'R1'],
            'regime_test': ['R0', 'R2']
        })
        
        report = _format_eligible_report(df)
        
        # Check structure
        assert 'rank' in report.columns
        assert 'ticker' in report.columns
        assert 'horizon' in report.columns
        assert 'n_signals' in report.columns
        assert 'skill_vs_marginal' in report.columns
        
        # Check ranks
        assert list(report['rank']) == [1, 2]
        
        # Check n_signals
        assert list(report['n_signals']) == [2, 1]


class TestSkillBreakdown:
    """Test skill breakdown creation."""
    
    def test_create_skill_breakdown(self):
        """Test creating skill breakdown by ticker/horizon."""
        df = pd.DataFrame({
            'ticker': ['AAPL', 'AAPL', 'TSLA', 'TSLA'],
            'horizon': [5, 5, 21, 21],
            'skill_vs_marginal': [0.02, 0.03, 0.01, 0.02],
            'skill_vs_uniform': [0.05, 0.06, 0.03, 0.04],
            'crps': [0.3, 0.25, 0.35, 0.30],
            'calibration_mae': [0.10, 0.08, 0.12, 0.10],
            'drift_passed': [True, True, False, True]
        })
        
        breakdown = _create_skill_breakdown(df)
        
        # Should have 2 ticker-horizon combinations
        assert len(breakdown) == 2
        
        # Check aggregations
        assert 'skill_vs_marginal_mean' in breakdown.columns
        assert 'skill_vs_marginal_median' in breakdown.columns
        assert 'skill_vs_marginal_count' in breakdown.columns


class TestSummaryStats:
    """Test summary statistics creation."""
    
    def test_create_summary_stats(self):
        """Test creating summary statistics."""
        all_df = pd.DataFrame({
            'ticker': ['AAPL', 'TSLA', 'MSFT'],
            'horizon': [5, 21, 5],
            'skill_vs_marginal': [0.02, 0.01, 0.03],
            'crps': [0.3, 0.35, 0.25],
            'calibration_mae': [0.10, 0.12, 0.08],
            'calibration_ece': [0.05, 0.06, 0.04],
            'drift_passed': [True, False, True],
            'drift_auc': [0.52, 0.48, 0.58],
            'regime_similarity': [0.7, 0.6, 0.8]
        })
        
        eligible_df = all_df[all_df['skill_vs_marginal'] >= 0.015]
        
        metadata = {'timestamp': '2025-09-30', 'seed': 42}
        
        summary = _create_summary_stats(
            all_df, eligible_df, metadata,
            min_skill=0.015, max_calib=0.15, drift_gate=True
        )
        
        # Check structure
        assert 'metadata' in summary
        assert 'filters' in summary
        assert 'counts' in summary
        assert 'skill_metrics' in summary
        assert 'calibration_metrics' in summary
        assert 'robustness_metrics' in summary
        
        # Check counts
        assert summary['counts']['total_validations'] == 3
        assert summary['counts']['eligible_setups'] == 2  # skill >= 0.015
        
        # Check skill metrics
        assert 'mean_skill_vs_marginal' in summary['skill_metrics']
        assert 'eligible_mean_skill' in summary['skill_metrics']


class TestReportGeneration:
    """Test full report generation."""
    
    def test_generate_eligibility_report(self, sample_eligibility_matrix, tmp_path):
        """Test generating full eligibility report."""
        matrix_path, _ = sample_eligibility_matrix
        output_dir = tmp_path / "reports"
        
        outputs = generate_eligibility_report(
            eligibility_matrix_path=matrix_path,
            output_dir=output_dir,
            min_skill_vs_marginal=0.01,
            max_calibration_mae=0.15,
            drift_gate=True,
            top_n=10
        )
        
        # Check that all expected files were created
        assert 'eligible_setups' in outputs
        assert 'skill_breakdown' in outputs
        assert 'calibration_summary' in outputs
        assert 'drift_analysis' in outputs
        assert 'summary' in outputs
        assert 'regime_analysis' in outputs
        
        # Check that files exist
        for name, path in outputs.items():
            assert path.exists(), f"{name} file not created"
        
        # Check file formats
        eligible_df = pd.read_csv(outputs['eligible_setups'])
        assert 'rank' in eligible_df.columns
        assert 'ticker' in eligible_df.columns
        assert 'skill_vs_marginal' in eligible_df.columns
        
        with open(outputs['summary'], 'r') as f:
            summary = json.load(f)
        assert 'metadata' in summary
        assert 'counts' in summary
        assert 'skill_metrics' in summary
    
    def test_report_with_different_thresholds(self, sample_eligibility_matrix, tmp_path):
        """Test report generation with different thresholds."""
        matrix_path, _ = sample_eligibility_matrix
        
        # Strict thresholds
        output_dir1 = tmp_path / "reports_strict"
        outputs1 = generate_eligibility_report(
            matrix_path, output_dir1,
            min_skill_vs_marginal=0.05,  # High skill threshold
            max_calibration_mae=0.10,     # Strict calibration
            drift_gate=True,
            top_n=5
        )
        
        # Lenient thresholds
        output_dir2 = tmp_path / "reports_lenient"
        outputs2 = generate_eligibility_report(
            matrix_path, output_dir2,
            min_skill_vs_marginal=0.0,   # Low skill threshold
            max_calibration_mae=0.30,     # Lenient calibration
            drift_gate=False,             # No drift gate
            top_n=20
        )
        
        # Lenient should have more eligible setups
        eligible1 = pd.read_csv(outputs1['eligible_setups'])
        eligible2 = pd.read_csv(outputs2['eligible_setups'])
        
        assert len(eligible2) >= len(eligible1)
