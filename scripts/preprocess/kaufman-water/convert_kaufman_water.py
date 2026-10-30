#!/usr/bin/env python3
"""
Build _raw/kaufman-water from _raw/_dlc/kaufman-water.

The source is standard DLC format, but every labeled-data/<timestamp>-cam[12]/ dir has
its own CollectedData_Hank.csv (+ .h5) instead of one project-level CSV. This script
concatenates all 24 per-session CSVs into a top-level CollectedData.csv /
CollectedData_test.csv, copies only the images those rows reference (the per-session
CSV/.h5 files are not copied), writes project.yaml, and sorts the sample videos into
videos/ and videos_test/ to match the split.

kaufman-water is a calibrated multi-view source (cam1/cam2 cover the same trials) but is
built as a SINGLE-view dataset, same as kaufman -- cam1 and cam2 are independent rows.

Split is by subject (mouse), using the mapping at
_raw/_dlc/kaufman-water/labeled-sessions-mouse-date-time.txt ("<mouse>/<timestamp>" per
line), via mighty_mouse.preprocess.subject_split. Splitting by subject also keeps a session's cam1
and cam2 views -- the same trial, viewed twice -- on the same side. Target is the
midpoint of the requested 10-15% test range (the greedy split only overshoots); seeds
are tried in order until the result lands in range.

Usage:
    python scripts/preprocess/kaufman-water/convert_kaufman_water.py [--dry-run]
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.subject_split import subject_split

TEST_RANGE = (0.10, 0.15)
TEST_FRACTION = sum(TEST_RANGE) / 2

SESSION_RE = re.compile(r"^(?P<timestamp>\d{8}-\d{6})-cam[12]$")
VIDEO_RE = re.compile(r"^(?P<mouse>[^_]+)_(?P<timestamp>\d{8}-\d{6})_cam[12]_")


def load_subject_mapping(mapping_path: Path) -> dict[str, str]:
    """Parse "<mouse>/<timestamp>" lines into {timestamp: mouse}."""
    mapping = {}
    for line in mapping_path.read_text().splitlines():
        line = line.strip()
        if line:
            mouse, timestamp = line.split("/")
            mapping[timestamp] = mouse
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="report the split, write nothing")
    args = parser.parse_args()

    raw_dir = Path(load_paths()["raw_dir"])
    src_dir = raw_dir / "_dlc" / "kaufman-water"
    out_dir = raw_dir / "kaufman-water"

    dfs = []
    for session_dir in sorted(p for p in (src_dir / "labeled-data").iterdir() if p.is_dir()):
        csvs = list(session_dir.glob("CollectedData_*.csv"))
        assert len(csvs) == 1, (
            f"expected 1 CollectedData_*.csv in {session_dir}, found {len(csvs)}"
        )
        dfs.append(pd.read_csv(csvs[0], header=[0, 1, 2], index_col=0))
    for df in dfs[1:]:
        assert df.columns.equals(dfs[0].columns), "keypoint schemas differ across sessions"
    combined = pd.concat(dfs)
    keypoints = list(dict.fromkeys(combined.columns.get_level_values(1)))
    print(f"combined: {len(combined)} frames, {len(keypoints)} keypoints")

    session_of_row = pd.Series([Path(p).parts[-2] for p in combined.index], index=combined.index)
    timestamp_of_row = session_of_row.map(lambda s: SESSION_RE.match(s).group("timestamp"))
    subject_of_ts = load_subject_mapping(src_dir / "labeled-sessions-mouse-date-time.txt")
    missing = set(timestamp_of_row) - subject_of_ts.keys()
    assert not missing, f"no subject mapping for timestamps: {sorted(missing)}"
    subject_of_row = timestamp_of_row.map(subject_of_ts)

    counts = subject_of_row.value_counts().to_dict()
    total = len(combined)
    for seed in range(1000):
        train_subjects, test_subjects = subject_split(counts, seed, TEST_FRACTION)
        frac = sum(counts[s] for s in test_subjects) / total
        if TEST_RANGE[0] <= frac <= TEST_RANGE[1]:
            break
    else:
        raise RuntimeError(f"no seed landed test fraction in {TEST_RANGE}")
    print(f"seed {seed}: test subjects {sorted(test_subjects)} ({frac:.1%} of frames)")
    print(f"  subject frame counts: {dict(sorted(counts.items()))}")

    test_mask = subject_of_row.isin(test_subjects)
    splits = {
        "CollectedData.csv": combined[~test_mask],
        "CollectedData_test.csv": combined[test_mask],
    }
    for name, df in splits.items():
        print(f"  {name}: {len(df)} frames")
    if args.dry_run:
        return

    for df in combined.index:
        dst = out_dir / df
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / df, dst)
    for name, df in splits.items():
        df.to_csv(out_dir / name)

    (out_dir / "project.yaml").write_text(
        yaml.safe_dump({"keypoint_names": keypoints, "schema_version": 1, "view_names": []},
                       sort_keys=False)
    )

    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "videos_test").mkdir(exist_ok=True)
    for video in sorted((src_dir / "videos").glob("*.mp4")):
        m = VIDEO_RE.match(video.name)
        assert m, f"unrecognized video name: {video.name}"
        subject = subject_of_ts[m.group("timestamp")]
        assert subject == m.group("mouse"), f"{video.name}: mapping says {subject}"
        dst_dir = "videos_test" if subject in test_subjects else "videos"
        shutil.copy2(video, out_dir / dst_dir / video.name)
        print(f"  {video.name} -> {dst_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
