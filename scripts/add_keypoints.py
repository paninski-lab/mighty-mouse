#!/usr/bin/env python3
"""
Add new, empty keypoints to a raw dataset so they can be labeled in the LP app.

Appends empty x/y columns to _raw/<dataset>/CollectedData.csv and CollectedData_test.csv
and the names to project.yaml's keypoint_names. Existing bytes in the CSVs are left
untouched (see mighty_mouse.labels.append_keypoint_columns). Doesn't touch
configs/datasets/<dataset>.yaml, the changelog, or the version -- see
skills/add-keypoints-to-raw-dataset/SKILL.md for the rest of the workflow.

Usage:
  python scripts/add_keypoints.py hantman-mv --keypoints nose_top pad_top pad_side --dry-run
  python scripts/add_keypoints.py hantman-mv --keypoints nose_top pad_top pad_side
"""

import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mighty_mouse.labels import add_project_keypoints, append_keypoint_columns
from mighty_mouse.paths import load_paths

CSV_NAMES = ("CollectedData.csv", "CollectedData_test.csv")


def _read_csv_text(text: str) -> pd.DataFrame:
    return pd.read_csv(
        StringIO(text), header=[0, 1, 2], index_col=0, float_precision="round_trip",
    )


def _check_csv(old_text: str, new_text: str, keypoints: list[str]) -> None:
    """Sanity check: old columns unchanged, new columns present and empty, row count same."""
    old, new = _read_csv_text(old_text), _read_csv_text(new_text)
    pd.testing.assert_frame_equal(new.iloc[:, : old.shape[1]], old)
    assert list(new.columns.get_level_values(1)[old.shape[1]:]) == [
        kp for kp in keypoints for _ in range(2)
    ]
    assert np.isnan(new.iloc[:, old.shape[1]:].to_numpy(dtype=float)).all()
    # every line of the old file is a prefix of the new one: nothing existing was rewritten
    for a, b in zip(old_text.splitlines(), new_text.splitlines(), strict=True):
        assert b.startswith(a)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset", help="Dataset name (directory under raw_dir)")
    parser.add_argument("--keypoints", nargs="+", required=True,
                        help="New raw keypoint names, appended in this order")
    parser.add_argument("--raw_dir", type=Path, default=None,
                        help="Root of raw datasets (default: from paths.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report without writing anything")
    args = parser.parse_args()

    raw_root = args.raw_dir or Path(load_paths()["raw_dir"])
    dataset_dir = raw_root / args.dataset
    files = [dataset_dir / name for name in (*CSV_NAMES, "project.yaml")]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        sys.exit(f"Missing: {missing}")

    # build and check every new file before writing any of them
    new_contents: dict[Path, bytes] = {}
    try:
        for name in CSV_NAMES:
            path = dataset_dir / name
            old_text = path.read_bytes().decode()
            new_text = append_keypoint_columns(old_text, args.keypoints)
            _check_csv(old_text, new_text, args.keypoints)
            new_contents[path] = new_text.encode()

        project_path = dataset_dir / "project.yaml"
        project_cfg = yaml.safe_load(project_path.read_text())
        new_project = add_project_keypoints(project_cfg, args.keypoints)
        new_contents[project_path] = yaml.safe_dump(new_project).encode()
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    print(f"{'Would add' if args.dry_run else 'Adding'} {args.keypoints} to {dataset_dir}:")
    for path in new_contents:
        print(f"  {path.name}")
    if args.dry_run:
        return
    for path, content in new_contents.items():
        path.write_bytes(content)
    print("Done. Columns are empty -- label them in the LP app.")


if __name__ == "__main__":
    main()
