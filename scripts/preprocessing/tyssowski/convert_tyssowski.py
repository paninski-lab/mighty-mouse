#!/usr/bin/env python3
"""
Build _raw/tyssowski (single-view) from the multi-view DLC project at
_raw/_dlc/tyssowski.

Same shape as hantman-mv (see scripts/preprocessing/hantman-mv/): the source is already
standard DLC format, split across two per-view CSVs (CollectedData_side.csv,
CollectedData_front.csv) with identical schemas and view-encoding session names (e.g.
"M399_20240528_v004_front" / "..._side"), so "converting to single-view" is a row concat
plus a project.yaml with no view_names.

Unlike hantman-mv, this script also DROPS the source keypoints the corpus doesn't use
(KEEP_KEYPOINTS below) from the output CSVs and project.yaml, so the LP labeling app only
shows the relevant points. Source names are otherwise kept as-is (d2_3, ...); mapping to
canonical names is configs/datasets/tyssowski.yaml's job.

Only the images actually referenced by a CSV row are copied. Split is subject-level
(first '_'-delimited token of the session name) via mighty_mouse.subject_split.

Usage:
    python scripts/preprocessing/tyssowski/convert_tyssowski.py
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.subject_split import subject_of, subject_split

# Target ~10-15% of frames in test. The greedy split in subject_split() only ever
# overshoots (it stops as soon as the running total reaches the target), and subject
# sizes here are uneven enough that aiming for the range's upper edge (0.15) regularly
# overshoots past it -- aiming for the midpoint leaves room for that overshoot while
# still landing in range.
TEST_FRACTION = 0.125
# Source digits are d1 (thumb) and d{2-5}_{1,2,3} (_3 = tip). Only the d{2-5} tips, hand,
# nose and mouth are used; everything else (thumb, other joints, wrist, elbow, shoulder,
# pellet) is dropped from the output.
KEEP_KEYPOINTS = ["d2_3", "d3_3", "d4_3", "d5_3", "hand", "nose", "mouth"]
# project.yaml skeleton: wrist (source "hand") to each digit tip
SKELETON = [["hand", tip] for tip in ["d2_3", "d3_3", "d4_3", "d5_3"]]
VIEW_CSVS = ["CollectedData_side.csv", "CollectedData_front.csv"]


def copy_images(index: pd.Index, source_dir: Path, out_dir: Path) -> int:
    copied = 0
    for rel_path in index:
        src = source_dir / rel_path
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=1, help="subject-split seed (default: 1; 0 gives 15.7% test)")
    args = parser.parse_args()

    paths = load_paths()
    raw_dir = Path(paths["raw_dir"])
    source_dir = raw_dir / "_dlc" / "tyssowski"
    out_dir = raw_dir / "tyssowski"
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for csv_name in VIEW_CSVS:
        view_df = pd.read_csv(source_dir / csv_name, header=[0, 1, 2], index_col=0)
        print(f"  {csv_name}: {len(view_df)} frames")
        dfs.append(view_df)

    assert dfs[0].columns.equals(dfs[1].columns), "side/front keypoint schemas differ"
    combined = pd.concat(dfs)
    combined = combined.loc[:, combined.columns.get_level_values(1).isin(KEEP_KEYPOINTS)]
    print(f"\ncombined: {len(combined)} frames, {len(combined.columns) // 2} keypoints")

    subject_of_row = pd.Series(
        [subject_of(Path(p).parts[-2]) for p in combined.index], index=combined.index
    )
    counts = subject_of_row.value_counts().to_dict()
    train_subjects, test_subjects = subject_split(counts, args.seed, TEST_FRACTION)
    print(f"subjects: {len(train_subjects)} train, {len(test_subjects)} test")

    for split_name, split_subjects in (("train", train_subjects), ("test", test_subjects)):
        split_df = combined.loc[subject_of_row.isin(split_subjects)]
        csv_name = "CollectedData.csv" if split_name == "train" else "CollectedData_test.csv"
        split_df.to_csv(out_dir / csv_name)
        print(f"  {split_name}: {len(split_df)} frames -> {out_dir / csv_name}")

    n_copied = copy_images(combined.index, source_dir, out_dir)
    print(f"  images: {n_copied} copied, {len(combined) - n_copied} already present")

    with open(source_dir / "project.yaml") as f:
        project = yaml.safe_load(f)
    old_views = project.get("view_names")
    project["skeleton"] = SKELETON
    project["keypoint_names"] = [k for k in project["keypoint_names"] if k in KEEP_KEYPOINTS]
    project["view_names"] = []
    with open(out_dir / "project.yaml", "w") as f:
        yaml.safe_dump(project, f, default_flow_style=False, sort_keys=False)
    print(f"  project.yaml written with view_names: [] (was {old_views})")

    (out_dir / "videos").mkdir(exist_ok=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
