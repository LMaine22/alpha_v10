import pandas as pd
import os

# --- Instructions ---
# This script is now fully automatic. It will find your project's root directory
# and the latest results file on its own. You can run it from anywhere
# inside your project (like the 'notebooks' folder).

# --------------------------------------------------------------------

try:
    # --- Step 1: Automatically find the project's root directory ---
    # It does this by searching upwards from the current script's location
    # until it finds the 'runs' folder.
    current_dir = os.getcwd()
    project_root = current_dir
    while not os.path.isdir(os.path.join(project_root, 'runs')):
        parent_dir = os.path.dirname(project_root)
        if parent_dir == project_root:  # Reached the top of the filesystem
            raise FileNotFoundError("Could not find the 'runs' directory.")
        project_root = parent_dir

    print(f"Project root found at: {project_root}")

    # --- Step 2: Find and load the latest results file ---
    runs_dir = os.path.join(project_root, 'runs')
    latest_run_folder = sorted(os.listdir(runs_dir))[-1]
    summary_file_path = os.path.join(runs_dir, latest_run_folder, 'pareto_front_summary.csv')

    print(f"Automatically found latest results file: {summary_file_path}")

    df = pd.read_csv(summary_file_path)

    # --- Step 3: Analyze for consistency ---
    consistency_analysis = df.groupby('description').agg(
        consistency_count=('fold', 'nunique'),
        avg_sharpe_lb=('sharpe_lb', 'mean'),
        avg_omega=('omega_ratio', 'mean'),
        avg_support=('support', 'mean')
    ).reset_index()

    robust_strategies = consistency_analysis.sort_values(
        by=['consistency_count', 'avg_sharpe_lb'],
        ascending=[False, False]
    )

    print("\n--- Most Consistent & Robust Strategies Across All Folds ---")
    print(robust_strategies.head(20).to_string())

except (FileNotFoundError, IndexError) as e:
    print(f"\nERROR: {e}")
    print("Please ensure your 'runs' directory exists and contains results.")