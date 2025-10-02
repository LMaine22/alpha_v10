"""
Validation utilities to verify pipeline fixes are working correctly.
Can be imported and used within the main pipeline or standalone.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Any


def check_metric_coverage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Check coverage of key metrics that should be improved by the fixes.
    
    Returns:
        Dictionary mapping metric names to coverage percentages
    """
    key_metrics = [
        "edge_crps_raw", "edge_pin_q10_raw", "edge_pin_q90_raw", "edge_ig_raw", 
        "edge_w1_raw", "edge_calib_mae_raw", "sensitivity_delta_edge_raw",
        "bootstrap_p_value_raw", "redundancy_mi_raw", "complexity_metric_raw",
        "transfer_entropy", "tr_cv_reg", "tr_fg", "n_trig_oos"
    ]
    
    coverage_results = {}
    for metric in key_metrics:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            total = len(df)
            coverage_pct = (non_null / total) * 100
            coverage_results[metric] = coverage_pct
        else:
            coverage_results[metric] = 0.0
    
    return coverage_results


def check_robust_alternatives(df: pd.DataFrame) -> Dict[str, float]:
    """
    Check presence and coverage of robust metric alternatives.
    
    Returns:
        Dictionary mapping robust metric names to coverage percentages
    """
    robust_alternatives = [
        'volatility_persistence', 'sample_entropy_robust', 'permutation_entropy_robust',
        'tscv_mean_stability', 'h0_total_persistence', 'regime_stability', 
        'composite_robustness', 'dfa_alpha_robust', 'bootstrap_p_value_robust'
    ]
    
    robust_coverage = {}
    for alt_metric in robust_alternatives:
        if alt_metric in df.columns:
            non_null = df[alt_metric].notna().sum()
            total = len(df)
            coverage_pct = (non_null / total) * 100
            robust_coverage[alt_metric] = coverage_pct
        else:
            robust_coverage[alt_metric] = 0.0
    
    return robust_coverage


def validate_individual_parsing(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test that individual parsing is working correctly.
    
    Returns:
        Dictionary with parsing test results
    """
    if 'individual' not in df.columns:
        return {"status": "missing_column", "parsed_count": 0, "total_count": 0}
    
    sample_individuals = df['individual'].head(10)
    parsed_count = 0
    errors = []
    
    for i, ind in enumerate(sample_individuals):
        try:
            if isinstance(ind, str):
                parsed = eval(ind)
            elif isinstance(ind, tuple):
                parsed = ind
            else:
                parsed = None
                
            if parsed:
                # Test string normalization
                def _normalize_strings(obj):
                    if isinstance(obj, np.str_):
                        return str(obj)
                    elif isinstance(obj, tuple):
                        return tuple(_normalize_strings(x) for x in obj)
                    elif isinstance(obj, list):
                        return [_normalize_strings(x) for x in obj]
                    else:
                        return obj
                
                normalized = _normalize_strings(parsed)
                if len(normalized) == 2:  # Should have ticker and signals
                    parsed_count += 1
                    
        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")
    
    return {
        "status": "success" if parsed_count > 0 else "failed",
        "parsed_count": parsed_count,
        "total_count": len(sample_individuals),
        "success_rate": parsed_count / len(sample_individuals) if len(sample_individuals) > 0 else 0,
        "errors": errors
    }


def assess_fix_success(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Overall assessment of whether the fixes are working.
    
    Returns:
        Dictionary with assessment results
    """
    coverage = check_metric_coverage(df)
    robust_coverage = check_robust_alternatives(df)
    parsing_test = validate_individual_parsing(df)
    
    # Critical metrics that should be well-covered after fixes
    critical_metrics = ['edge_crps_raw', 'redundancy_mi_raw', 'tr_cv_reg', 'tr_fg']
    critical_coverage = [coverage.get(m, 0) for m in critical_metrics]
    avg_critical_coverage = np.mean(critical_coverage)
    
    # Expected improvements mapping
    expected_improvements = {
        'edge_crps_raw': 70,
        'redundancy_mi_raw': 90,
        'tr_cv_reg': 80,
        'tr_fg': 90,
        'complexity_metric_raw': 85
    }
    
    # Check if expectations are met
    improvements_met = {}
    for metric, expected in expected_improvements.items():
        actual = coverage.get(metric, 0)
        improvements_met[metric] = {
            'expected': expected,
            'actual': actual,
            'met': actual >= expected
        }
    
    # Overall status
    if avg_critical_coverage >= 75:
        overall_status = "excellent"
    elif avg_critical_coverage >= 50:
        overall_status = "good"
    else:
        overall_status = "needs_work"
    
    return {
        "overall_status": overall_status,
        "avg_critical_coverage": avg_critical_coverage,
        "total_candidates": len(df),
        "coverage": coverage,
        "robust_coverage": robust_coverage,
        "parsing_test": parsing_test,
        "improvements_met": improvements_met
    }


def print_validation_summary(assessment: Dict[str, Any]) -> None:
    """Print a formatted summary of the validation results."""
    
    print("\n=== PIPELINE FIXES VALIDATION SUMMARY ===")
    print(f"Total candidates: {assessment['total_candidates']}")
    print(f"Overall status: {assessment['overall_status'].upper()}")
    print(f"Average critical metric coverage: {assessment['avg_critical_coverage']:.1f}%")
    
    print("\nCRITICAL METRICS:")
    for metric, result in assessment['improvements_met'].items():
        status = "✓" if result['met'] else "✗"
        print(f"  {status} {metric}: {result['actual']:.1f}% (expected {result['expected']}%)")
    
    print("\nPARSING TEST:")
    parsing = assessment['parsing_test']
    if parsing['status'] == 'success':
        print(f"  ✓ Individual parsing: {parsing['success_rate']:.1%} success rate")
    else:
        print(f"  ✗ Individual parsing failed")
    
    # Show robust alternatives that are present
    robust_present = {k: v for k, v in assessment['robust_coverage'].items() if v > 0}
    if robust_present:
        print(f"\nROBUST ALTERNATIVES ACTIVE:")
        for metric, coverage in robust_present.items():
            print(f"  ✓ {metric}: {coverage:.1f}%")
    
    print()
