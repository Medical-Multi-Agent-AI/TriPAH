#!/usr/bin/env python3
"""
Script to update paths in MIMIC-CXR CSV files.
This script removes the 'files/' prefix from paths in cxr-record-list.csv and cxr-study-list.csv
to match the actual directory structure.
"""

import pandas as pd
import os
from pathlib import Path

def update_csv_paths(base_dir=None):
    """Update paths in MIMIC-CXR CSV files"""

    from utils.image_path import mimic_root
    base_dir = Path(base_dir) if base_dir is not None else mimic_root

    # Files that need path updates
    files_to_update = {
        "cxr-record-list.csv": "path",
        "cxr-study-list.csv": "path"
    }

    for filename, path_column in files_to_update.items():
        file_path = base_dir / filename

        if not file_path.exists():
            print(f"Warning: {file_path} does not exist, skipping...")
            continue

        print(f"Processing {filename}...")

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the path column exists
        if path_column not in df.columns:
            print(f"Warning: Column '{path_column}' not found in {filename}, skipping...")
            continue

        # Update paths by removing 'files/' prefix
        original_count = len(df)
        files_prefix_count = df[path_column].str.startswith('files/').sum()

        print(f"  - Total rows: {original_count}")
        print(f"  - Rows with 'files/' prefix: {files_prefix_count}")

        # Remove 'files/' prefix from paths
        df[path_column] = df[path_column].str.replace('^files/', '', regex=True)

        # Create backup of original file
        backup_path = file_path.with_suffix('.csv.backup')
        if not backup_path.exists():
            print(f"  - Creating backup: {backup_path}")
            df_original = pd.read_csv(file_path)
            df_original.to_csv(backup_path, index=False)

        # Save updated file
        df.to_csv(file_path, index=False)
        print(f"  - Updated {filename} successfully")

        # Verify a few sample paths
        print(f"  - Sample updated paths:")
        for i, path in enumerate(df[path_column].head(3)):
            print(f"    {i+1}. {path}")

    print("\nPath update completed!")

    # Verify some paths exist
    print("\nVerifying updated paths...")
    verify_paths(base_dir)

def verify_paths(base_dir):
    """Verify that updated paths point to existing files"""

    # Check cxr-record-list.csv paths (images)
    record_file = base_dir / "cxr-record-list.csv"
    if record_file.exists():
        df_records = pd.read_csv(record_file)
        sample_paths = df_records['path'].head(5)

        print("Checking image paths:")
        for path in sample_paths:
            # Convert .dcm to .jpg and add mimic-cxr/mimic-cxr-images prefix
            jpg_path = path.replace('.dcm', '.jpg')
            full_path = base_dir / "mimic-cxr" / "mimic-cxr-images" / jpg_path
            exists = full_path.exists()
            print(f"  {'✓' if exists else '✗'} {jpg_path} -> {'EXISTS' if exists else 'NOT FOUND'}")

    # Check cxr-study-list.csv paths (reports)
    study_file = base_dir / "cxr-study-list.csv"
    if study_file.exists():
        df_studies = pd.read_csv(study_file)
        sample_paths = df_studies['path'].head(5)

        print("Checking report paths:")
        for path in sample_paths:
            # Add mimic-cxr/mimic-cxr-reports prefix
            full_path = base_dir / "mimic-cxr" / "mimic-cxr-reports" / path
            exists = full_path.exists()
            print(f"  {'✓' if exists else '✗'} {path} -> {'EXISTS' if exists else 'NOT FOUND'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Normalize MIMIC-CXR CSV paths; keeps .csv.backup files.")
    parser.add_argument("--root", default=None)
    update_csv_paths(parser.parse_args().root)