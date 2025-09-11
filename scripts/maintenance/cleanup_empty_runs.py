#!/usr/bin/env python3
"""
Clean up empty run folders

Finds and removes all empty directories in the runs folder and its subdirectories.
"""

import os
import shutil
from pathlib import Path

def is_empty_directory(path):
    """Check if a directory is completely empty (no files or subdirectories)"""
    try:
        return len(os.listdir(path)) == 0
    except (OSError, PermissionError):
        return False

def remove_empty_directories(base_path):
    """Recursively remove empty directories"""
    removed_count = 0
    
    # Walk through all directories, bottom-up so we can remove parent dirs if they become empty
    for root, dirs, files in os.walk(base_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if is_empty_directory(dir_path):
                try:
                    os.rmdir(dir_path)
                    print(f"Removed empty directory: {dir_path}")
                    removed_count += 1
                except (OSError, PermissionError) as e:
                    print(f"Could not remove {dir_path}: {e}")
    
    return removed_count

def main():
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("No 'runs' directory found!")
        return
    
    print("=" * 60)
    print("CLEANING UP EMPTY RUN DIRECTORIES")
    print("=" * 60)
    
    print(f"Scanning: {runs_dir.absolute()}")
    
    # First, let's see what we're dealing with
    all_dirs = []
    empty_dirs = []
    
    for root, dirs, files in os.walk(runs_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            all_dirs.append(dir_path)
            if is_empty_directory(dir_path):
                empty_dirs.append(dir_path)
    
    print(f"\nFound {len(all_dirs)} total directories")
    print(f"Found {len(empty_dirs)} empty directories")
    
    if empty_dirs:
        print(f"\nEmpty directories to remove:")
        for empty_dir in empty_dirs:
            print(f"  - {empty_dir}")
        
        # Remove them
        print(f"\nRemoving {len(empty_dirs)} empty directories...")
        removed_count = remove_empty_directories(runs_dir)
        
        print(f"\nCleanup complete!")
        print(f"Successfully removed {removed_count} empty directories")
    else:
        print("\nNo empty directories found!")
    
    # Final summary
    remaining_dirs = []
    for root, dirs, files in os.walk(runs_dir):
        for dir_name in dirs:
            remaining_dirs.append(os.path.join(root, dir_name))
    
    print(f"\nFinal state: {len(remaining_dirs)} directories remaining in runs/")

if __name__ == "__main__":
    main()
