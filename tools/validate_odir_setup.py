#!/usr/bin/env python3
"""Validate the local ODIR path configuration before conversion or training."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_path import odir_root as default_odir_root


def _count_images(path: Path) -> int:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    return sum(len(list(path.glob(ext))) for ext in exts)


def validate(args: argparse.Namespace) -> int:
    odir_root = Path(args.odir_root)
    train_default = odir_root / "Training_Images"
    if not train_default.exists():
        train_default = odir_root / "Training Images"
    test_default = odir_root / "Testing_Images"
    if not test_default.exists():
        test_default = odir_root / "Testing Images"
    train_dir = Path(args.train_dir) if args.train_dir else train_default
    test_dir = Path(args.test_dir) if args.test_dir else test_default
    data_file = odir_root / "data.xlsx"

    checks = [
        ("ODIR root", odir_root, odir_root.exists()),
        ("data.xlsx", data_file, data_file.exists()),
        ("training images", train_dir, train_dir.exists()),
    ]

    ok = True
    for name, path, passed in checks:
        print(f"{'OK' if passed else 'MISS'} {name}: {path}")
        ok = ok and passed

    if train_dir.exists():
        n_train = _count_images(train_dir)
        print(f"training image count: {n_train}")
        ok = ok and n_train > 0
        if args.verbose:
            for image_path in sorted(train_dir.iterdir())[:5]:
                print(f"  sample: {image_path.name}")

    if test_dir.exists():
        print(f"testing image count: {_count_images(test_dir)}")

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ODIR local paths.")
    parser.add_argument("--odir-root", default=str(default_odir_root))
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return validate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
