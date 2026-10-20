#!/usr/bin/env python3
"""
Build _raw/cavelli from the DLC project at _raw/_dlc/cavelli.

The source is already DLC layout (one CollectedData_cavelli.csv with an x/y/visible
triple per keypoint, left and right cameras as separate video directories), so this
only: drops the keypoints that have no labels at all (the source is a labeling
template for a 44-keypoint skeleton, of which a handful are filled), drops the
`visible` columns (raw datasets here are x/y only; empty = unlabeled), splits
train/test at the subject (animal) level, copies the referenced images, and writes a
project.yaml. Left and right cameras are merged into one project.

Usage:
    python scripts/preprocessing/cavelli/convert_cavelli.py
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.subject_split import subject_of, subject_split

SCORER = "cavelli"
# Target 10-20% test; seed chosen so the greedy subject split lands inside the range
# (it only overshoots) -- M3 + M6 = 47 of 265 rows.
TEST_FRACTION = 0.15
SEED = 3
SRC_PREFIX = "labeled-data/cavelli/"


def session_of(path: str) -> str:
    return path.split("/")[-2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--dst", type=Path, default=None)
    args = ap.parse_args()

    raw_dir = Path(load_paths()["raw_dir"])
    src = args.src or raw_dir / "_dlc" / "cavelli"
    dst = args.dst or raw_dir / "cavelli"

    df = pd.read_csv(src / "CollectedData_cavelli.csv", header=[0, 1, 2], index_col=0)
    scorer_in = df.columns[0][0]
    all_kps = list(dict.fromkeys(c[1] for c in df.columns))
    labeled = [kp for kp in all_kps if df[(scorer_in, kp, "x")].notna().any()]
    cols = [(SCORER, kp, c) for kp in labeled for c in ("x", "y")]
    out = pd.DataFrame({(SCORER, kp, c): df[(scorer_in, kp, c)] for _, kp, c in cols})
    out.columns = pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"])
    out.index = ["labeled-data/" + i.removeprefix(SRC_PREFIX) for i in out.index]

    sessions = pd.Series([session_of(i) for i in out.index], index=out.index)
    subjects = sessions.map(subject_of)
    counts = dict(sorted(subjects.value_counts().items()))
    _, test_subjects = subject_split(counts, SEED, TEST_FRACTION)
    is_test = subjects.isin(test_subjects)

    splits = {"CollectedData.csv": out[~is_test], "CollectedData_test.csv": out[is_test]}
    # _raw/cavelli holds hand-edited labels (version 1+); never clobber them.
    existing = [n for n in splits if (dst / n).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing} in {dst}; pass --dst elsewhere")
    dst.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(dst / name)
        print(f"{name}: {len(part)} rows")
    for rel_out, rel_src in zip(out.index, df.index, strict=True):
        (dst / rel_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel_src, dst / rel_out)
    (dst / "videos").mkdir(exist_ok=True)
    with open(dst / "project.yaml", "w") as f:
        project = {"keypoint_names": labeled, "schema_version": 1, "view_names": []}
        yaml.safe_dump(project, f, sort_keys=False)
    print(f"keypoints: {labeled}\ntest subjects: {sorted(test_subjects)}")


if __name__ == "__main__":
    main()
