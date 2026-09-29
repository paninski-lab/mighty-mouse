#!/usr/bin/env python3
"""
Build the combined training dataset by subsampling from pre-converted per-dataset CSVs.

Reads CollectedData_<dataset>_train.csv and CollectedData_<dataset>_test.csv
from data_dir (produced by convert_dataset.py), subsamples train frames, and
merges across datasets. Keypoints absent from a given dataset get visible=0.

Produces (in data_dir):
  CollectedData_<tag>_train.csv  merged train labels
  CollectedData_<tag>_test.csv   merged test labels (all frames, no subsampling)

--n_frames caps how many frames each dataset contributes (same cap for every
dataset in --datasets, so merges stay balanced — see docs/build_dataset.md).
Pass --n_frames -1 to instead take every available frame from every dataset,
unbalanced. By convention tags built this way drop the frame count from their
name (e.g. "face+cheese"), while balanced tags include it (e.g. "face+cheese-600").

--data_dir overrides data_dir from paths.yaml, for both input and output (e.g. to build
the next data version without repointing paths.yaml under a run that's training from it).
Sampling logic lives in mighty_mouse/build.py.

Usage:
  python scripts/build_dataset.py --tag face+ibl-600 --datasets facemap ibl --n_frames 600
  python scripts/build_dataset.py --tag face+ibl     --datasets facemap ibl --n_frames -1
  python scripts/build_dataset.py --tag face+ibl-600 --datasets facemap ibl --data_dir data_v3/
"""

import argparse
from pathlib import Path

import pandas as pd

from mighty_mouse.build import build_merged, dataset_rng, subsample
from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths


def build(data_dir: Path, datasets: list[str], n_frames: int, seed: int, tag: str) -> None:
    train_dfs: list[pd.DataFrame] = []
    test_dfs:  list[pd.DataFrame] = []

    for name in datasets:
        train_csv = data_dir / f"CollectedData_{name}_train.csv"
        test_csv  = data_dir / f"CollectedData_{name}_test.csv"

        if not train_csv.exists():
            print(f"WARNING: {train_csv.name} not found — skipping {name}")
            continue

        print(f"\n── {name} ──────────────────────────────────────")
        train_df = read_labels_csv(train_csv)
        print(f"  Train: {len(train_df)} frames available")

        if 0 <= n_frames and len(train_df) < n_frames:
            print(f"  WARNING: only {len(train_df)} frames available (requested {n_frames})")
        sample = subsample(train_df, n_frames, dataset_rng(seed, name))
        print(f"  Sampled {len(sample)} frames")
        train_dfs.append(sample)

        if test_csv.exists():
            test_df = read_labels_csv(test_csv)
            print(f"  Test:  {len(test_df)} frames")
            test_dfs.append(test_df)
        else:
            print(f"  WARNING: {test_csv.name} not found — skipping test split for {name}")

    if not train_dfs:
        print("ERROR: no datasets loaded.")
        return

    print("\n── merged train ────────────────────────────────────")
    merged_train = build_merged(train_dfs)
    merged_train_path = data_dir / f"CollectedData_{tag}_train.csv"
    merged_train.to_csv(merged_train_path)
    print(f"  {len(merged_train)} rows → {merged_train_path.name}")

    if test_dfs:
        print("\n── merged test ─────────────────────────────────────")
        merged_test = build_merged(test_dfs)
        merged_test_path = data_dir / f"CollectedData_{tag}_test.csv"
        merged_test.to_csv(merged_test_path)
        print(f"  {len(merged_test)} rows → {merged_test_path.name}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined dataset from pre-converted per-dataset CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tag", required=True,
        help='label for output CSVs, e.g. "all" → CollectedData_all_{train,test}.csv',
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help=f"Datasets to include (default: {ALL_DATASETS})",
    )
    parser.add_argument(
        "--n_frames", type=int, default=600,
        help="frames to sample per dataset (default: 600); "
             "-1 = use every available frame, no subsampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--data_dir", type=Path, default=None,
        help="directory to read converted CSVs from and write merged CSVs to "
             "(default: data_dir from paths.yaml)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir or Path(load_paths()["data_dir"])
    build(data_dir, args.datasets, args.n_frames, args.seed, args.tag)


if __name__ == "__main__":
    main()
