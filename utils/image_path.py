"""Local raw-data locations. Set environment variables before running Python.

Converted medical data are loaded from dataset/<dataset>/ by load_data.py.
"""
import os
from pathlib import Path

data_root = Path(os.environ.get("TRIPAH_DATA_ROOT", "data")).expanduser()
odir_root = Path(os.environ.get("ODIR_ROOT", str(data_root / "ODIR" / "ODIR-5K"))).expanduser()
iuxray_root = Path(os.environ.get("IUXRAY_ROOT", str(data_root / "IU-Xray"))).expanduser()
mimic_root = Path(os.environ.get("MIMIC_CXR_ROOT", str(data_root / "mimic-cxr-jpg"))).expanduser()

flickr25k_img_path = os.environ.get("FLICKR25K_IMAGE_ROOT", str(data_root / "MIRFLICKR-25K" / "MIRFLICKR-25K-all-images" / "images"))
nuswide_img_path = os.environ.get("NUSWIDE_IMAGE_ROOT", str(data_root / "NUS-WIDE" / "NUS-WIDE-all-images" / "images"))
coco_img_path = os.environ.get("COCO_IMAGE_ROOT", str(data_root / "MS COCO" / "MSCOCO-all-images" / "coco_images"))
odir_train_img_path = os.environ.get("ODIR_TRAIN_IMAGES", str(odir_root / "Training_Images"))
odir_test_img_path = os.environ.get("ODIR_TEST_IMAGES", str(odir_root / "Testing_Images"))
mimic_cxr_img_path = os.environ.get("MIMIC_CXR_IMAGES", str(mimic_root / "mimic-cxr" / "mimic-cxr" / "mimic-cxr-images"))
mimic_cxr_report_path = os.environ.get("MIMIC_CXR_REPORTS", str(mimic_root / "mimic-cxr" / "mimic-cxr" / "mimic-cxr-reports"))
