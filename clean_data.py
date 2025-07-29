#!/usr/bin/env python3
"""
Script to remove all empty directories and invalid JSON files from the experiment data tree structure.
This script recursively traverses the directory tree and removes:
1. Directories that contain no files
2. JSON files that don't have length 119 or total_score 20
"""

import os
import sys
import json
from pathlib import Path


def is_dir_empty(dir_path):
    """
    Check if a directory is empty (contains no files or subdirectories).
    
    Args:
        dir_path (str): Path to the directory to check
        
    Returns:
        bool: True if directory is empty, False otherwise
    """
    try:
        # Check if directory exists and is actually a directory
        if not os.path.isdir(dir_path):
            return False
        
        # List all items in the directory
        items = os.listdir(dir_path)
        
        # If there are no items, the directory is empty
        return len(items) == 0
        
    except (OSError, PermissionError) as e:
        print(f"Error checking directory {dir_path}: {e}")
        return False


def is_valid_json_file(file_path):
    """
    Check if a JSON file meets the validation criteria:
    - total_timestamp array has length 119 OR total_score is 20
    (Keep file if EITHER condition is met)
    
    Args:
        file_path (str): Path to the JSON file to check
        
    Returns:
        bool: True if JSON file is valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data is a dictionary
        if not isinstance(data, dict):
            return False
        
        # Check if total_timestamp exists and has length 119
        timestamp_valid = False
        if 'total_timestamp' in data:
            total_timestamp = data['total_timestamp']
            if isinstance(total_timestamp, list) and len(total_timestamp) == 120:
                timestamp_valid = True
        
        # Check if total_score is 20
        score_valid = False
        if 'total_score' in data and data['total_score'] == 20:
            score_valid = True
        
        # Return True if EITHER condition is met (OR logic)
        return timestamp_valid or score_valid
        
    except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
        # If we can't parse the JSON or it doesn't have the expected structure, it's invalid
        return False


def remove_invalid_files_and_empty_dirs_recursive(root_path, dry_run=True, verbose=True):
    """
    Recursively remove invalid JSON files and empty directories starting from root_path.
    
    Args:
        root_path (str): Root directory to start from
        dry_run (bool): If True, only print what would be removed without actually removing
        verbose (bool): If True, print detailed information about what's being processed
        
    Returns:
        tuple: (removed_dirs, removed_files) - Lists of directories and files that were (or would be) removed
    """
    removed_dirs = []
    removed_files = []
    
    # Walk through the directory tree bottom-up (post-order traversal)
    for root, dirs, files in os.walk(root_path, topdown=False):
        # First, check and remove invalid JSON files
        for file_name in files:
            if file_name.endswith('.json'):
                file_path = os.path.join(root, file_name)
                
                if not is_valid_json_file(file_path):
                    if verbose:
                        print(f"Found invalid JSON file: {file_path}")
                    
                    if not dry_run:
                        try:
                            os.remove(file_path)
                            if verbose:
                                print(f"Removed: {file_path}")
                            removed_files.append(file_path)
                        except OSError as e:
                            print(f"Error removing {file_path}: {e}")
                    else:
                        removed_files.append(file_path)
        
        # Then, process directories in reverse order to handle nested empty dirs
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            
            if is_dir_empty(dir_path):
                if verbose:
                    print(f"Found empty directory: {dir_path}")
                
                if not dry_run:
                    try:
                        os.rmdir(dir_path)
                        if verbose:
                            print(f"Removed: {dir_path}")
                        removed_dirs.append(dir_path)
                    except OSError as e:
                        print(f"Error removing {dir_path}: {e}")
                else:
                    removed_dirs.append(dir_path)
    
    return removed_dirs, removed_files


def main():
    """Main function to handle command line arguments and execute the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove empty directories and invalid JSON files from experiment data tree structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - show what would be removed without actually removing
  python remove_empty_dirs.py /path/to/experiment_2025-07-09_temp_0.7
  
  # Actually remove empty directories and invalid JSON files
  python remove_empty_dirs.py /path/to/experiment_2025-07-09_temp_0.7 --execute
  
  # Quiet mode - minimal output
  python remove_empty_dirs.py /path/to/experiment_2025-07-09_temp_0.7 --execute --quiet
        """
    )
    
    parser.add_argument(
        "root_path",
        help="Root directory path to start searching for empty directories and invalid JSON files"
    )
    
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove empty directories and invalid JSON files (default is dry-run mode)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate the root path
    if not os.path.exists(args.root_path):
        print(f"Error: Path '{args.root_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.isdir(args.root_path):
        print(f"Error: Path '{args.root_path}' is not a directory.")
        sys.exit(1)
    
    # Convert to absolute path
    root_path = os.path.abspath(args.root_path)
    
    print(f"Scanning directory: {root_path}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("Criteria: JSON files must have total_timestamp array with length 119 OR total_score 20")
    print("-" * 50)
    
    # Remove invalid files and empty directories
    removed_dirs, removed_files = remove_invalid_files_and_empty_dirs_recursive(
        root_path, 
        dry_run=not args.execute, 
        verbose=not args.quiet
    )
    
    # Summary
    print("-" * 50)
    if args.execute:
        print(f"Successfully removed {len(removed_dirs)} empty directories and {len(removed_files)} invalid JSON files.")
    else:
        print(f"Found {len(removed_dirs)} empty directories and {len(removed_files)} invalid JSON files that would be removed.")
        if (removed_dirs or removed_files) and not args.quiet:
            if removed_dirs:
                print("\nDirectories that would be removed:")
                for dir_path in removed_dirs:
                    print(f"  {dir_path}")
            if removed_files:
                print("\nJSON files that would be removed:")
                for file_path in removed_files:
                    print(f"  {file_path}")
    
    if not args.execute and (removed_dirs or removed_files):
        print("\nTo actually remove these items, run with --execute flag.")


if __name__ == "__main__":
    main() 