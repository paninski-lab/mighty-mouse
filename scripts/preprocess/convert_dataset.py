#!/usr/bin/env python3
"""
Convert a raw labeled dataset into standardized format for combined training.

Reads configs/datasets/<dataset>.yaml and applies session/keypoint exclusions,
lateralization, renaming to canonical names, and a visibility column. The
transform itself (and the visibility convention) lives in mighty_mouse/preprocess/convert.py.

Produces (in data_dir from paths.yaml):
  CollectedData_<dataset>_train.csv
  CollectedData_<dataset>_test.csv
  labeled-data/<dataset>/<session>/<frame>.png  (all images, both splits)

Usage:
  python scripts/preprocess/convert_dataset.py --dataset facemap
  python scripts/preprocess/convert_dataset.py --dataset ibl --raw_dir /other/raw
"""

import argparse
import sys
from pathlib import Path

from mighty_mouse.configs import (
    check_model_config,
    default_configs_dir,
    load_canonical_keypoints,
    load_dataset_config,
    load_model_config,
)
from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.convert import (
    copy_images,
    post_process,
    process_split,
    raw_dataset_dir,
    validate_dataset_config,
)

TRAIN_CSV = "CollectedData.csv"
TEST_CSV  = "CollectedData_test.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw labeled dataset to standardized format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (subdirectory under raw_dir, e.g. 'facemap')",
    )
    parser.add_argument("--raw_dir", type=Path, default=None,
                        help="Root of raw datasets (default: from paths.yaml)")
    parser.add_argument("--data_dir", type=Path, default=None,
                        help="Output directory (default: from paths.yaml)")
    parser.add_argument("--train_csv", default=TRAIN_CSV,
                        help=f"Train CSV filename (default: {TRAIN_CSV})")
    parser.add_argument("--test_csv", default=TEST_CSV,
                        help=f"Test CSV filename (default: {TEST_CSV})")
    parser.add_argument("--link_frames", action="store_true",
                        help="symlink frames to the raw files instead of copying "
                             "(frames never change between data versions)")
    args = parser.parse_args()

    if args.raw_dir is None or args.data_dir is None:
        paths = load_paths()
        args.raw_dir  = args.raw_dir  or Path(paths["raw_dir"])
        args.data_dir = args.data_dir or Path(paths["data_dir"])

    print("Loading configs...")
    configs_dir   = default_configs_dir()
    canonical_kps = load_canonical_keypoints(configs_dir)
    config        = load_dataset_config(configs_dir, args.dataset)
    raw_dir       = raw_dataset_dir(args.raw_dir, args.dataset, config)

    splits = {}
    for split_name, csv_name in (("train", args.train_csv), ("test", args.test_csv)):
        path = raw_dir / csv_name
        splits[split_name] = read_labels_csv(path) if path.exists() else None

    if all(df is None for df in splits.values()):
        print(f"ERROR: no CSVs found in {raw_dir}", file=sys.stderr)
        sys.exit(1)
    if splits["train"] is None:
        print(f"WARNING: {args.train_csv} not found — skipping train split")
    if splits["test"] is None:
        print(f"WARNING: {args.test_csv} not found — skipping test split")

    print("Validating config...")
    errors = check_model_config(load_model_config(configs_dir), canonical_kps)
    errors += validate_dataset_config(config, canonical_kps, splits.values())
    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print("  All checks passed.")

    args.data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        if df is None:
            continue
        print(f"\n── {split_name} ────────────────────────────────────────────")
        print(f"  {len(df)} frames in source CSV")

        processed = process_split(df, config, args.dataset, canonical_kps)
        processed = post_process(processed, config, args.dataset)
        n_excluded = len(df) - len(processed)
        print(f"  {n_excluded} frames excluded ({len(processed)} remaining)")

        out_csv = args.data_dir / f"CollectedData_{args.dataset}_{split_name}.csv"
        processed.to_csv(out_csv)
        print(f"  Saved {out_csv.name}")

        n_copied = copy_images(processed.index, raw_dir, args.data_dir, link=args.link_frames)
        n_exist  = len(processed) - n_copied
        print(f"  Images: {n_copied} copied, {n_exist} already present")

    print("\nDone.")


if __name__ == "__main__":
    main()
