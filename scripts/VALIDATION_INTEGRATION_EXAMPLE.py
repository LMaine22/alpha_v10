"""
Example of how to integrate validation into your main pipeline.
This shows how to use the validation utilities to check that fixes are working.
"""

from alpha_discovery.utils.validation_utils import assess_fix_success, print_validation_summary


def main_pipeline_with_validation():
    """
    Example showing how to integrate validation into your main pipeline.
    Add this after your ELV calculation and Hart Index computation.
    """
    
    # ... your existing pipeline code ...
    # final_results_df = calculate_elv_and_labels(pre_elv_df)
    # final_results_df = calculate_hart_index(final_results_df)
    
    # Add validation check
    print("\n" + "="*60)
    print("VALIDATING PIPELINE FIXES")
    print("="*60)
    
    # Run comprehensive assessment
    assessment = assess_fix_success(final_results_df)
    
    # Print formatted summary
    print_validation_summary(assessment)
    
    # Log critical warnings
    if assessment['overall_status'] == 'needs_work':
        print("WARNING: Pipeline fixes may not be working correctly!")
        print("Consider running the full validation script for detailed analysis.")
    
    # Save assessment to file
    import json
    assessment_file = os.path.join(run_output_dir, 'pipeline_assessment.json')
    
    # Convert non-serializable types for JSON
    json_safe_assessment = {}
    for key, value in assessment.items():
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            json_safe_assessment[key] = value
        else:
            json_safe_assessment[key] = str(value)
    
    with open(assessment_file, 'w') as f:
        json.dump(json_safe_assessment, f, indent=2)
    
    print(f"Assessment saved to: {assessment_file}")
    
    return final_results_df


def quick_validation_check(results_df):
    """
    Quick validation that can be called anywhere in your pipeline.
    """
    from alpha_discovery.utils.validation_utils import check_metric_coverage
    
    coverage = check_metric_coverage(results_df)
    
    critical_metrics = ['redundancy_mi_raw', 'tr_cv_reg', 'tr_fg']
    critical_ok = all(coverage.get(m, 0) >= 75 for m in critical_metrics)
    
    if critical_ok:
        print("✓ Quick validation: Critical metrics look good")
    else:
        print("⚠ Quick validation: Some critical metrics may need attention")
        for metric in critical_metrics:
            cov = coverage.get(metric, 0)
            print(f"    {metric}: {cov:.1f}%")
    
    return critical_ok


if __name__ == "__main__":
    # Example of running standalone validation
    import pandas as pd
    import os
    
    # Find latest results file
    runs_dir = "runs"
    if os.path.exists(runs_dir):
        run_folders = [f for f in os.listdir(runs_dir) if f.startswith('pivot_forecast_')]
        if run_folders:
            latest_run = sorted(run_folders)[-1]
            results_file = os.path.join(runs_dir, latest_run, 'final_results.csv')
            
            if os.path.exists(results_file):
                print(f"Loading results from: {results_file}")
                df = pd.read_csv(results_file)
                
                assessment = assess_fix_success(df)
                print_validation_summary(assessment)
            else:
                print("No final_results.csv found in latest run")
        else:
            print("No forecast run folders found")
    else:
        print("No runs directory found")
