#!/usr/bin/env python3
"""
Consolidate Open Trades Summaries Script

Finds all open_trades_summary.csv files across all run directories and consolidates
them into a single master file with clean 15 columns, sorted by OOS final NAV.
"""

import os
import pandas as pd
from pathlib import Path
import glob
from typing import List, Optional

# Define the 15 clean columns we want to keep
CLEAN_COLUMNS = [
    "setup_id", "specialized_ticker", "direction", "description", "signal_ids",
    "oos_first_trigger", "oos_last_trigger", "oos_days_since_last_trigger",
    "oos_total_trades", "oos_open_trades", "oos_sum_pnl_dollars", 
    "oos_final_nav", "oos_nav_total_return_pct", "oos_dsr", "expectancy"
]

def find_all_open_trades_files() -> List[str]:
    """Find all open_trades_summary.csv files in the runs directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("Error: 'runs' directory not found!")
        return []
    
    # Search patterns for different folder structures
    search_patterns = [
        "runs/*/gauntlet/open_trades_summary.csv",           # Direct runs
        "runs/*/open_trades_summary.csv",                    # Some may be directly in run folders
        "runs/1-30/*/gauntlet/open_trades_summary.csv",     # 1-30 subfolder
        "runs/31-37/*/gauntlet/open_trades_summary.csv",    # 31-37 subfolder  
        "runs/38-50/*/gauntlet/open_trades_summary.csv",    # 38-50 subfolder
        "runs/GA_FITNESS_META/*/gauntlet/open_trades_summary.csv",  # GA_FITNESS_META subfolder
    ]
    
    all_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
        if files:
            print(f"Found {len(files)} files matching pattern: {pattern}")
    
    # Remove duplicates and sort
    unique_files = list(set(all_files))
    unique_files.sort()
    
    print(f"\nTotal unique open_trades_summary.csv files found: {len(unique_files)}")
    return unique_files

def load_and_clean_file(file_path: str) -> Optional[pd.DataFrame]:
    """Load a single open trades file and clean it to the 15 columns."""
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            print(f"  Skipping {file_path}: empty file")
            return None
            
        # Add source column to track which run this came from
        run_name = Path(file_path).parent.parent.name
        df['source_run'] = run_name
        
        # Keep only the columns we want (plus source_run)
        available_columns = [col for col in CLEAN_COLUMNS if col in df.columns]
        missing_columns = [col for col in CLEAN_COLUMNS if col not in df.columns]
        
        if missing_columns:
            print(f"  {file_path}: Missing columns {missing_columns}")
        
        # Select available columns plus source
        final_columns = available_columns + ['source_run']
        df_clean = df[final_columns].copy()
        
        print(f"  Loaded {file_path}: {len(df)} rows, {len(available_columns)}/{len(CLEAN_COLUMNS)} columns")
        return df_clean
        
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def consolidate_open_trades():
    """Main function to consolidate all open trades summaries."""
    print("=" * 60)
    print("CONSOLIDATING OPEN TRADES SUMMARIES")
    print("=" * 60)
    
    # Find all files
    files = find_all_open_trades_files()
    if not files:
        print("No open trades summary files found!")
        return
    
    print(f"\nProcessing {len(files)} files...")
    
    # Load and clean all files
    all_dataframes = []
    successful_loads = 0
    
    for file_path in files:
        df = load_and_clean_file(file_path)
        if df is not None:
            all_dataframes.append(df)
            successful_loads += 1
    
    if not all_dataframes:
        print("No valid data loaded!")
        return
    
    print(f"\nSuccessfully loaded {successful_loads}/{len(files)} files")
    
    # Concatenate all dataframes
    print("\nConcatenating all data...")
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Master dataframe shape: {master_df.shape}")
    print(f"Columns: {list(master_df.columns)}")
    
    # Sort by OOS final NAV (descending - best first)
    if 'oos_final_nav' in master_df.columns:
        master_df = master_df.sort_values('oos_final_nav', ascending=False)
        print("\nSorted by OOS final NAV (descending)")
    else:
        print("\nWarning: 'oos_final_nav' column not found, keeping original order")
    
    # Save to master file
    output_file = "master_open_trades_summary.csv"
    master_df.to_csv(output_file, index=False)
    
    print(f"\nMaster file saved: {output_file}")
    print(f"Total rows: {len(master_df)}")
    print(f"Unique runs: {master_df['source_run'].nunique()}")
    
    # Show summary by run
    print(f"\nSummary by run:")
    run_summary = master_df.groupby('source_run').size().sort_values(ascending=False)
    for run, count in run_summary.items():
        print(f"  {run}: {count} open trades")
    
    # Show top 10 by OOS final NAV
    if 'oos_final_nav' in master_df.columns:
        print(f"\nTop 10 by OOS Final NAV:")
        top_10 = master_df.head(10)[['setup_id', 'specialized_ticker', 'oos_final_nav', 'source_run']]
        for _, row in top_10.iterrows():
            print(f"  {row['setup_id']} | {row['specialized_ticker']} | NAV: ${row['oos_final_nav']:,.0f} | {row['source_run']}")
    
    print(f"\nConsolidation complete! Master file: {output_file}")

if __name__ == "__main__":
    consolidate_open_trades()
