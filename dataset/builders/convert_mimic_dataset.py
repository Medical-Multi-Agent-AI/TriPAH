#!/usr/bin/env python3
"""
MIMIC-CXR to ODIR-style Dataset Converter (Optimized Version)
============================================================

This script converts MIMIC-CXR/MIMIC-CXR-JPG data into an ODIR-style dataset
suitable for image-text and image-image hash retrieval tasks.

Optimizations:
- Multi-processing for file I/O operations
- Vectorized pandas operations
- Memory-efficient data processing
- Optional GPU acceleration for text processing

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import scipy.io as sio
from collections import defaultdict
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional GPU acceleration imports
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available (cuDF/CuPy)")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU acceleration not available, using CPU")

def process_report_batch(report_paths_batch, reports_root):
    """Process a batch of reports in parallel."""
    results = []

    for report_path in report_paths_batch:
        if report_path and Path(reports_root / report_path).exists():
            try:
                with open(reports_root / report_path, 'r', encoding='utf-8') as f:
                    report_text = f.read().strip()

                # Extract findings and impression sections
                findings, impression = extract_report_sections_static(report_text)
                results.append({
                    'findings': findings,
                    'impression': impression,
                    'full_report': report_text
                })
            except Exception as e:
                results.append({
                    'findings': '',
                    'impression': '',
                    'full_report': ''
                })
        else:
            results.append({
                'findings': '',
                'impression': '',
                'full_report': ''
            })

    return results

def extract_report_sections_static(report_text):
    """Static version of extract_report_sections for multiprocessing."""
    # Clean the text
    text = re.sub(r'\s+', ' ', report_text).strip()

    # Extract FINDINGS section
    findings_match = re.search(r'FINDINGS:\s*(.*?)(?=\n\n|\n[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
    findings = findings_match.group(1).strip() if findings_match else ""

    # Extract IMPRESSION section
    impression_match = re.search(r'IMPRESSION:\s*(.*?)(?=\n\n|\n[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
    impression = impression_match.group(1).strip() if impression_match else ""

    return findings, impression

def check_batch_exists_static(batch):
    """Static function to check existence for a batch of files (for multiprocessing)."""
    return [os.path.exists(str(path)) if path else False for path in batch]

class MIMICToODIRConverterOptimized:
    """Optimized converter class for MIMIC-CXR to ODIR-style dataset transformation."""

    def __init__(self, mimic_root, output_root, n_workers=None, use_gpu=False):
        """
        Initialize the optimized converter.

        Args:
            mimic_root (str): Root directory of MIMIC-CXR-JPG data
            output_root (str): Output directory for processed dataset
            n_workers (int): Number of worker processes (default: CPU count)
            use_gpu (bool): Whether to use GPU acceleration when available
        """
        self.mimic_root = Path(mimic_root).expanduser().resolve()
        self.output_root = Path(output_root)
        self.n_workers = n_workers or mp.cpu_count()
        self.use_gpu = use_gpu and GPU_AVAILABLE

        # Define input file paths
        self.metadata_path = self.mimic_root / "mimic-cxr-2.0.0-metadata.csv"
        self.split_path = self.mimic_root / "mimic-cxr-2.0.0-split.csv"
        self.chexpert_path = self.mimic_root / "mimic-cxr-2.0.0-chexpert.csv"
        self.negbio_path = self.mimic_root / "mimic-cxr-2.0.0-negbio.csv"
        self.reports_root = self.mimic_root / "mimic-cxr" / "mimic-cxr-reports"
        self.images_root = self.mimic_root / "mimic-cxr" / "mimic-cxr-images"

        # CSV files with path information
        self.record_list_path = self.mimic_root / "cxr-record-list.csv"
        self.study_list_path = self.mimic_root / "cxr-study-list.csv"

        # Define 14 CheXpert labels
        self.label_names = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
            'Pleural Effusion', 'Pneumonia', 'Pneumothorax', 'Pleural Other',
            'Support Devices', 'No Finding'
        ]

        # View position priority (PA > AP > LATERAL)
        self.view_priority = {'PA': 1, 'AP': 2, 'LATERAL': 3}

        print(f"Initialized optimized converter:")
        print(f"  MIMIC root: {self.mimic_root}")
        print(f"  Output root: {self.output_root}")
        print(f"  Workers: {self.n_workers}")
        print(f"  GPU acceleration: {self.use_gpu}")

    def setup_output_directory(self):
        """Step 0: Setup output directory and dependencies."""
        print("\n=== Step 0: Setting up output directory ===")

        # Create output directory
        self.output_root.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {self.output_root}")

        # Verify input files exist
        required_files = [
            self.metadata_path, self.split_path,
            self.chexpert_path, self.negbio_path,
            self.record_list_path, self.study_list_path
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
            print(f"✓ Found: {file_path.name}")

        if not self.reports_root.exists():
            raise FileNotFoundError(f"Reports directory not found: {self.reports_root}")
        print(f"✓ Found reports directory: {self.reports_root}")

        if not self.images_root.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_root}")
        print(f"✓ Found images directory: {self.images_root}")

    def load_and_merge_metadata(self):
        """Step 1: Load split & metadata, construct image candidate table (optimized)."""
        print("\n=== Step 1: Loading and merging metadata (optimized) ===")

        # Load split data
        print("Loading split data...")
        if self.use_gpu:
            try:
                print("🚀 Using GPU-accelerated data loading...")
                split_df = cudf.read_csv(self.split_path)
                print(f"Split data shape: {split_df.shape} (GPU)")
                print("✓ GPU data loading successful")
            except Exception as e:
                print(f"❌ GPU loading failed, falling back to CPU: {e}")
                split_df = pd.read_csv(self.split_path)
                print(f"Split data shape: {split_df.shape}")
        else:
            split_df = pd.read_csv(self.split_path)
            print(f"Split data shape: {split_df.shape}")

        print(f"Split distribution: {split_df['split'].value_counts().to_dict()}")

        # Load metadata
        print("Loading metadata...")
        if self.use_gpu:
            try:
                metadata_df = cudf.read_csv(self.metadata_path)
                print(f"Metadata shape: {metadata_df.shape} (GPU)")
            except Exception as e:
                print(f"❌ GPU loading failed, falling back to CPU: {e}")
                metadata_df = pd.read_csv(self.metadata_path)
                print(f"Metadata shape: {metadata_df.shape}")
        else:
            metadata_df = pd.read_csv(self.metadata_path)
            print(f"Metadata shape: {metadata_df.shape}")

        # Merge on dicom_id (optimized)
        print("Merging split and metadata...")
        if self.use_gpu and hasattr(split_df, 'merge'):
            try:
                print("🚀 Using GPU-accelerated merge...")
                merged_df = split_df.merge(metadata_df, on='dicom_id', how='inner')
                print("✓ GPU merge successful")
            except Exception as e:
                print(f"❌ GPU merge failed, falling back to CPU: {e}")
                # Convert to pandas and merge
                if hasattr(split_df, 'to_pandas'):
                    split_df = split_df.to_pandas()
                if hasattr(metadata_df, 'to_pandas'):
                    metadata_df = metadata_df.to_pandas()
                merged_df = split_df.merge(metadata_df, on='dicom_id', how='inner')
        else:
            merged_df = split_df.merge(metadata_df, on='dicom_id', how='inner')
        print(f"Merged data shape: {merged_df.shape}")

        # Handle duplicate columns from merge (pandas adds _x, _y suffixes)
        if 'study_id_x' in merged_df.columns and 'study_id_y' in merged_df.columns:
            merged_df['study_id'] = merged_df['study_id_x']
            merged_df = merged_df.drop(['study_id_x', 'study_id_y'], axis=1)

        if 'subject_id_x' in merged_df.columns and 'subject_id_y' in merged_df.columns:
            merged_df['subject_id'] = merged_df['subject_id_x']
            merged_df = merged_df.drop(['subject_id_x', 'subject_id_y'], axis=1)

        # Keep only required columns
        required_cols = [
            'dicom_id', 'study_id', 'subject_id', 'split',
            'ViewPosition', 'StudyDate', 'StudyTime'
        ]

        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in merged_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            print(f"Available columns: {list(merged_df.columns)}")
            required_cols = [col for col in required_cols if col in merged_df.columns]
            print(f"Using available columns: {required_cols}")

        self.merged_df = merged_df[required_cols].copy()

        # Handle missing StudyTime
        if 'StudyTime' in self.merged_df.columns:
            self.merged_df['StudyTime'] = self.merged_df['StudyTime'].fillna('000000')

        # Convert back to pandas if using GPU
        if self.use_gpu and hasattr(self.merged_df, 'to_pandas'):
            print("✓ Converting to pandas for compatibility")
            self.merged_df = self.merged_df.to_pandas()

        print(f"Final merged data shape: {self.merged_df.shape}")
        print(f"ViewPosition distribution: {self.merged_df['ViewPosition'].value_counts().head()}")

        return self.merged_df

    def select_representative_images(self):
        """Step 2: Select one representative image per study (vectorized)."""
        print("\n=== Step 2: Selecting representative images (optimized) ===")

        df = self.merged_df.copy()

        # Add view priority score (vectorized)
        df['view_priority'] = df['ViewPosition'].map(self.view_priority).fillna(999)

        # Create sorting key for stable selection within same view (vectorized)
        if 'StudyTime' in df.columns:
            df['sort_key'] = df['StudyDate'].astype(str) + '_' + df['StudyTime'].astype(str)
        else:
            df['sort_key'] = df['dicom_id']

        # Optimized selection using groupby with idxmin
        print("Selecting best image per study (vectorized)...")
        df_sorted = df.sort_values(['study_id', 'view_priority', 'sort_key'])
        self.selected_df = df_sorted.groupby('study_id').first().reset_index()

        # Load cxr-record-list.csv to get actual image paths
        print("Loading image paths from cxr-record-list.csv...")
        record_df = pd.read_csv(self.record_list_path)

        # Check data integrity before merge
        print(f"Selected studies: {len(self.selected_df)}")
        print(f"Available records: {len(record_df)}")

        # Find missing dicom_ids
        missing_dicom_ids = set(self.selected_df['dicom_id']) - set(record_df['dicom_id'])
        if missing_dicom_ids:
            print(f"Warning: {len(missing_dicom_ids)} dicom_ids not found in record list")
            print(f"Sample missing IDs: {list(missing_dicom_ids)[:5]}")

        # Merge with record list to get image paths
        self.selected_df = pd.merge(
            self.selected_df,
            record_df[['dicom_id', 'path']],
            on='dicom_id',
            how='inner'  # Changed from 'left' to 'inner' to avoid None values
        )

        print(f"After merge with record list: {len(self.selected_df)} records")

        # Check for None values in path column
        none_paths = self.selected_df['path'].isna().sum()
        if none_paths > 0:
            print(f"Warning: {none_paths} records have None paths")
            self.selected_df = self.selected_df.dropna(subset=['path']).reset_index(drop=True)
            print(f"After removing None paths: {len(self.selected_df)} records")

        # Vectorized string operations - keep full MIMIC-CXR directory structure
        # Original path format: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm
        # Target format: mimic-cxr-images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        self.selected_df['image_rel_path'] = self.selected_df['path'].str.replace('.dcm', '.jpg', regex=False).str.replace('files/', '', regex=False)

        # Create absolute paths (vectorized)
        self.selected_df['image_abs_path'] = self.selected_df['image_rel_path'].apply(
            lambda x: (self.images_root / x.replace('images/', '')) if pd.notna(x) else None
        )

        # Remove rows without valid paths
        valid_paths = self.selected_df['image_abs_path'].notna()
        self.selected_df = self.selected_df[valid_paths].reset_index(drop=True)

        # Batch verify image files exist (ultra-fast optimized)
        print("Verifying image files exist (ultra-fast)...")
        image_paths = self.selected_df['image_abs_path'].tolist()

        # Use optimized batch processing for file existence checks
        existing_images = self._batch_check_files_exist(image_paths)

        existing_images = pd.Series(existing_images)
        print(f"Images found: {existing_images.sum()}/{len(existing_images)} ({existing_images.mean()*100:.1f}%)")

        # Keep only existing images
        self.selected_df = self.selected_df[existing_images].reset_index(drop=True)

        print(f"Final selected images: {len(self.selected_df)}")
        print(f"View distribution: {self.selected_df['ViewPosition'].value_counts().to_dict()}")

        return self.selected_df

    def load_reports_parallel(self):
        """Step 3: Load and process radiology reports (parallel processing)."""
        print("\n=== Step 3: Loading radiology reports (parallel) ===")

        # Load cxr-study-list.csv to get actual report paths
        print("Loading report paths from cxr-study-list.csv...")
        study_df = pd.read_csv(self.study_list_path)

        # Check data integrity before merge
        print(f"Selected studies: {len(self.selected_df)}")
        print(f"Available study records: {len(study_df)}")

        # Find missing study_ids
        missing_study_ids = set(self.selected_df['study_id']) - set(study_df['study_id'])
        if missing_study_ids:
            print(f"Warning: {len(missing_study_ids)} study_ids not found in study list")
            print(f"Sample missing IDs: {list(missing_study_ids)[:5]}")

        # Merge with selected data to get report paths
        self.selected_df = pd.merge(
            self.selected_df,
            study_df[['study_id', 'path']],
            on='study_id',
            how='inner',  # Changed from 'left' to 'inner' to avoid None values
            suffixes=('', '_report')
        )

        print(f"After merge with study list: {len(self.selected_df)} records")

        # Check for None values in path_report column
        none_report_paths = self.selected_df['path_report'].isna().sum()
        if none_report_paths > 0:
            print(f"Warning: {none_report_paths} records have None report paths")
            self.selected_df = self.selected_df.dropna(subset=['path_report']).reset_index(drop=True)
            print(f"After removing None report paths: {len(self.selected_df)} records")

        # Remove rows without valid report paths (additional safety check)
        valid_report_paths = self.selected_df['path_report'].notna()
        self.selected_df = self.selected_df[valid_report_paths].reset_index(drop=True)

        # Prepare report paths for parallel processing
        report_paths = self.selected_df['path_report'].tolist()

        # Split into batches for parallel processing
        batch_size = max(1, len(report_paths) // self.n_workers)
        batches = [report_paths[i:i + batch_size] for i in range(0, len(report_paths), batch_size)]

        print(f"Processing {len(report_paths)} reports in {len(batches)} batches using {self.n_workers} workers...")

        # Process reports in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            process_func = partial(process_report_batch, reports_root=self.reports_root)
            results = list(tqdm(
                executor.map(process_func, batches),
                total=len(batches),
                desc="Processing report batches"
            ))

        # Flatten results
        all_reports = []
        missing_reports = 0
        for batch_results in results:
            for report in batch_results:
                all_reports.append(report)
                if not report['full_report']:
                    missing_reports += 1

        # Add report data to dataframe
        report_df = pd.DataFrame(all_reports)
        self.selected_df = pd.concat([self.selected_df, report_df], axis=1)

        print(f"Reports loaded: {len(self.selected_df) - missing_reports}/{len(self.selected_df)}")
        print(f"Missing reports: {missing_reports}")

        # ----- Data cleaning: placeholder handling and length control -----
        print("Applying text cleaning (replace '___', normalize whitespace, truncate to 512 chars)...")
        self.selected_df['findings'] = self.selected_df['findings'].astype(str).apply(lambda t: self._clean_text(t))
        self.selected_df['impression'] = self.selected_df['impression'].astype(str).apply(lambda t: self._clean_text(t))

        # ----- Drop samples that would result in caption == "No findings reported." -----
        nf = "No findings reported."
        before_cnt = len(self.selected_df)
        imp_strip = self.selected_df['impression'].fillna('').str.strip()
        fin_strip = self.selected_df['findings'].fillna('').str.strip()
        drop_mask = (imp_strip.isin(['', nf])) & (fin_strip.isin(['', nf]))
        self.selected_df = self.selected_df[~drop_mask].reset_index(drop=True)
        dropped_cnt = before_cnt - len(self.selected_df)
        print(f"Dropped {dropped_cnt} samples where caption == '{nf}' or both sections empty.")

        return self.selected_df

    def extract_report_sections(self, report_text):
        """Extract findings and impression sections from report text."""
        return extract_report_sections_static(report_text)

    def _batch_check_files_exist(self, file_paths):
        """Ultra-fast batch file existence check with optimized parallel processing."""
        if not file_paths:
            return []

        # Use optimized batch size based on system capabilities
        batch_size = max(1000, len(file_paths) // (self.n_workers * 4))
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]

        # Use ProcessPoolExecutor for CPU-bound file I/O operations
        with ProcessPoolExecutor(max_workers=min(self.n_workers, len(batches))) as executor:
            results = list(tqdm(
                executor.map(check_batch_exists_static, batches),
                total=len(batches),
                desc="Checking image files",
                unit="batch"
            ))

        # Flatten results
        existing_files = []
        for batch_result in results:
            existing_files.extend(batch_result)

        return existing_files

    def _extract_impression_or_findings(self, report_text):
        """Extract Impression > Findings > last paragraph from report text."""
        # Clean the text
        text = re.sub(r'\s+', ' ', report_text).strip()

        # Try to extract IMPRESSION section
        impression_match = re.search(r'IMPRESSION:\s*(.*?)(?=\n\n|\n[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if impression_match:
            return impression_match.group(1).strip()

        # Try to extract FINDINGS section
        findings_match = re.search(r'FINDINGS:\s*(.*?)(?=\n\n|\n[A-Z]+:|$)', text, re.IGNORECASE | re.DOTALL)
        if findings_match:
            return findings_match.group(1).strip()

        # Fall back to last paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            return paragraphs[-1]

        return ""

    # ---------------- Text Cleaning Utilities ----------------
    def _clean_text(self, text, max_chars: int = 512, replace_placeholder: str = "[BLANK]") -> str:
        """Clean report text by handling placeholders and length.
        Steps:
        - Replace placeholder '___' with [BLANK]
        - Collapse excessive whitespace
        - Strip leading/trailing spaces
        - Truncate to at most `max_chars` characters
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        # Replace explicit placeholder
        cleaned = text.replace("___", replace_placeholder)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Truncate length
        if max_chars is not None and max_chars > 0 and len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
        return cleaned

    def merge_labels(self):
        """Step 4: Merge CheXpert and NegBio labels (vectorized)."""
        print("\n=== Step 4: Merging and processing labels (optimized) ===")

        # Load label data
        print("Loading CheXpert labels...")
        chexpert_df = pd.read_csv(self.chexpert_path)
        print(f"CheXpert shape: {chexpert_df.shape}")

        print("Loading NegBio labels...")
        negbio_df = pd.read_csv(self.negbio_path)
        print(f"NegBio shape: {negbio_df.shape}")

        # Merge with selected data
        selected_keys = self.selected_df[['subject_id', 'study_id']].copy()

        # Check data integrity before merge
        print(f"Selected studies for labels: {len(selected_keys)}")
        print(f"Available CheXpert records: {len(chexpert_df)}")
        print(f"Available NegBio records: {len(negbio_df)}")

        # Find missing study combinations
        selected_combinations = set(zip(selected_keys['subject_id'], selected_keys['study_id']))
        chexpert_combinations = set(zip(chexpert_df['subject_id'], chexpert_df['study_id']))
        negbio_combinations = set(zip(negbio_df['subject_id'], negbio_df['study_id']))

        missing_chexpert = selected_combinations - chexpert_combinations
        missing_negbio = selected_combinations - negbio_combinations

        if missing_chexpert:
            print(f"Warning: {len(missing_chexpert)} study combinations not found in CheXpert")
        if missing_negbio:
            print(f"Warning: {len(missing_negbio)} study combinations not found in NegBio")

        # Merge with CheXpert (primary) - use inner join to avoid None values
        labels_df = pd.merge(selected_keys, chexpert_df, on=['subject_id', 'study_id'], how='inner')
        print(f"After CheXpert merge: {len(labels_df)} records")

        # Merge with NegBio (for backfill) - use inner join
        negbio_merged = pd.merge(selected_keys, negbio_df, on=['subject_id', 'study_id'], how='inner')
        print(f"After NegBio merge: {len(negbio_merged)} records")

        # Update selected_df to only include records that have labels
        valid_label_keys = labels_df[['subject_id', 'study_id']].copy()
        self.selected_df = pd.merge(
            self.selected_df,
            valid_label_keys,
            on=['subject_id', 'study_id'],
            how='inner'
        )
        print(f"Final selected data after label filtering: {len(self.selected_df)} records")

        # Create label matrix (vectorized operations)
        label_matrix = np.zeros((len(self.selected_df), len(self.label_names)), dtype=np.float32)

        for i, label_name in enumerate(self.label_names):
            if label_name in labels_df.columns:
                chexpert_col = labels_df[label_name]
                negbio_col = negbio_merged[label_name] if label_name in negbio_merged.columns else pd.Series([np.nan] * len(labels_df))

                # Apply fusion logic: CheXpert first, then NegBio backfill
                final_col = chexpert_col.fillna(negbio_col)

                # Vectorized mapping: 1.0→1.0, 0.0→0.0, -1.0→0.5, NaN→0.0
                mapped_col = final_col.map({1.0: 1.0, 0.0: 0.0, -1.0: 0.5}).fillna(0.0)
                label_matrix[:, i] = mapped_col.values

        self.label_matrix = label_matrix

        # Print statistics
        print(f"Label matrix shape: {label_matrix.shape}")
        for i, label_name in enumerate(self.label_names):
            col = label_matrix[:, i]
            stats = {
                'positive': (col == 1.0).sum(),
                'negative': (col == 0.0).sum(),
                'uncertain': (col == 0.5).sum()
            }
            print(f"  {label_name}: pos={stats['positive']}, neg={stats['negative']}, unc={stats['uncertain']}")

        # Verify no None values in critical columns
        critical_columns = ['dicom_id', 'study_id', 'subject_id', 'split', 'ViewPosition', 'image_rel_path']
        for col in critical_columns:
            if col in self.selected_df.columns:
                none_count = self.selected_df[col].isna().sum()
                if none_count > 0:
                    print(f"Warning: {none_count} None values found in {col}")
                    # Fill with appropriate defaults
                    if col in ['dicom_id', 'study_id', 'subject_id']:
                        self.selected_df[col] = self.selected_df[col].fillna('unknown')
                    elif col == 'split':
                        self.selected_df[col] = self.selected_df[col].fillna('train')
                    elif col == 'ViewPosition':
                        self.selected_df[col] = self.selected_df[col].fillna('UNKNOWN')
                    elif col == 'image_rel_path':
                        self.selected_df[col] = self.selected_df[col].fillna('')
                    print(f"Filled None values in {col} with defaults")

        return label_matrix

    def create_data_splits(self):
        """Step 5: Validate data splits (optimized)."""
        print("\n=== Step 5: Validating data splits (optimized) ===")

        # Use existing train/validate/test splits
        split_counts = self.selected_df['split'].value_counts()
        print(f"Split distribution: {split_counts.to_dict()}")

        # Note: No need to create retrieval splits here
        # The load_data.py will handle dynamic splitting at training time
        print("✓ Data splits validated. Dynamic splitting will be handled by load_data.py")

        return None

    def write_output_files(self):
        """Step 6: Write all output files (optimized I/O)."""
        print("\n=== Step 6: Writing output files (optimized) ===")

        # Generate captions from impression or findings
        print("Generating captions...")
        self.captions = []
        nf = "No findings reported."
        for _, row in self.selected_df.iterrows():
            if row.get('impression', '').strip():
                caption = row['impression'].strip()
            elif row.get('findings', '').strip():
                caption = row['findings'].strip()
            else:
                caption = "No findings reported."
            self.captions.append(caption)

        # Write caption.txt (optimized)
        caption_path = self.output_root / "caption.txt"
        with open(caption_path, 'w', encoding='utf-8', buffering=8192) as f:
            f.write('\n'.join(self.captions))
        print(f"✓ Wrote {caption_path}")

        # Write index.mat (compatible with load_data.py expectations)
        # load_data.py expects keys: "index", "imgs", or "FAll"
        # We use "index" key with image paths array (same as ODIR format)
        image_paths = self.selected_df['image_rel_path'].fillna('').values.astype(str)

        # Create simple index array like ODIR format
        max_len = max(len(p) for p in image_paths) if len(image_paths) > 0 else 1
        index_array = np.array(image_paths, dtype=f"<U{max_len}")

        index_data = {
            'index': index_array,
            # Keep additional metadata for potential future use
            'study_id': self.selected_df['study_id'].values.astype(np.int64),
            'subject_id': self.selected_df['subject_id'].values.astype(np.int64),
            'dicom_id': self.selected_df['dicom_id'].fillna('').values.astype(str),
            'split': self.selected_df['split'].fillna('').values.astype(str),
            'view': self.selected_df['ViewPosition'].fillna('').values.astype(str)
        }

        index_path = self.output_root / "index.mat"
        sio.savemat(index_path, index_data, do_compression=True)
        print(f"✓ Wrote {index_path} with {len(image_paths)} image paths")

        # Write label.mat (compatible with load_data.py expectations)
        # load_data.py expects keys: "category", "LAll", "labels", or "label"
        # We use "labels" key (same as ODIR format)
        label_data = {
            'labels': self.label_matrix.astype(np.float32),
            'category': self.label_matrix.astype(np.float32),  # Alternative key for compatibility
            'label_names': np.array(self.label_names, dtype=object).reshape(-1, 1)
        }

        label_path = self.output_root / "label.mat"
        sio.savemat(label_path, label_data, do_compression=True)
        print(f"✓ Wrote {label_path} with shape {self.label_matrix.shape}")

    # Write prompt_caption.npz (compressed)
        # Save text embeddings (N, 77, 512) under key 'prompt_caption' so Dataset returns 6 items
        prompt_path = self.output_root / "prompt_caption.npz"
        try:
            import torch
            from model import open_clip  # use the same open_clip stack as training
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_arch = 'ViT-B-16-quickgelu'
            clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained='metaclip_fullcc')
            clip_model = clip_model.to(device).eval()
            tokenizer = open_clip.get_tokenizer(clip_arch)

            caps = [str(c)[:512] for c in self.captions]
            bs = 512
            feats = []
            with torch.no_grad():
                for i in range(0, len(caps), bs):
                    tok = tokenizer(caps[i:i+bs]).to(device)  # [B, 77]
                    # token embeddings + positional embedding => [B, 77, 512]
                    if hasattr(clip_model, 'text'):
                        x = clip_model.text.token_embedding(tok)
                        x = x + clip_model.text.positional_embedding
                    else:
                        x = clip_model.token_embedding(tok)
                        x = x + clip_model.positional_embedding
                    feats.append(x.detach().cpu().numpy().astype('float32'))
            prompt_caption = np.concatenate(feats, axis=0)  # [N, 77, 512]
            np.savez_compressed(prompt_path, prompt_caption=prompt_caption)
            print(f"✓ Wrote {prompt_path} with prompt_caption shape {prompt_caption.shape}")
        except Exception as e:
            print(f"⚠️ Fallback building text embeddings: {e}")
            N = len(self.captions)
            prompt_caption = np.zeros((N, 77, 512), dtype=np.float32)
            np.savez_compressed(prompt_path, prompt_caption=prompt_caption)
            print(f"✓ Wrote {prompt_path} with ZERO prompt_caption shape {prompt_caption.shape}")

        # Write dataset_info.json (standard format)
        dataset_info = {
            "dataset_name": "MIMIC-CXR",
            "description": "MIMIC-CXR Chest X-ray Dataset converted to ODIR-style format",
            "total_samples": len(self.selected_df),
            "split_strategy": "No split here. load_data.py will randomly split into query/train/retrieval at training time.",
            "disease_labels": dict(zip(range(len(self.label_names)), self.label_names)),
            "image_format": "DICOM converted to JPEG",
            "image_size": "Variable",
            "processing": "Representative image selection + Report-based labeling",
            "description_format": "Medical report impressions and findings",
            "created_by": "MIMIC-CXR Converter Script (Optimized)"
        }

        info_path = self.output_root / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        print(f"✓ Wrote {info_path}")

        # Create symbolic link to images
        images_link = self.output_root / "images"
        if not images_link.exists():
            try:
                if os.name == 'nt':  # Windows
                    import subprocess
                    subprocess.run(['mklink', '/D', str(images_link), str(self.images_root)],
                                 shell=True, check=True)
                else:  # Unix-like
                    images_link.symlink_to(self.images_root)
                print(f"✓ Created symbolic link: {images_link}")
            except Exception as e:
                print(f"Warning: Could not create symbolic link: {e}")
                print(f"Please manually link {images_link} to {self.images_root}")

        print(f"\n✅ Dataset conversion completed successfully!")
        print(f"Output directory: {self.output_root}")
        print(f"Total samples: {len(self.selected_df)}")

    def run_full_pipeline(self):
        """Run the complete conversion pipeline with optimizations."""
        print("🚀 Starting MIMIC-CXR to ODIR conversion pipeline (OPTIMIZED)")
        print("=" * 60)

        try:
            # Step 0: Setup
            self.setup_output_directory()

            # Step 1: Load and merge metadata
            self.load_and_merge_metadata()

            # Step 2: Select representative images
            self.select_representative_images()

            # Step 3: Load reports (parallel)
            self.load_reports_parallel()

            # Step 4: Merge labels
            self.merge_labels()

            # Step 5: Create data splits
            self.create_data_splits()

            # Step 6: Write output files
            self.write_output_files()

            print("\n🎉 Pipeline completed successfully!")

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function with optimized parameters."""
    import argparse
    from utils.image_path import mimic_root as default_root
    parser = argparse.ArgumentParser(description="Convert MIMIC-CXR for TriPAH.")
    parser.add_argument("--root", default=str(default_root))
    parser.add_argument("--output", default="dataset/mimic-cxr")
    args = parser.parse_args()
    mimic_root, output_root = args.root, args.output

    # Create optimized converter
    converter = MIMICToODIRConverterOptimized(
        mimic_root=mimic_root,
        output_root=output_root,
        n_workers=mp.cpu_count(),  # Use all available CPU cores
        use_gpu=GPU_AVAILABLE      # Use GPU if available
    )

    # Run pipeline
    converter.run_full_pipeline()

if __name__ == "__main__":
    main()
