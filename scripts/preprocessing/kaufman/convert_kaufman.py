#!/usr/bin/env python3
"""
Build _raw/kaufman's top-level CollectedData.csv / CollectedData_test.csv from its
104 per-session CollectedData_Hank.csv files (labeled-data/<timestamp>-cam[12]/).

The source is already standard DLC format (project.yaml, labeled-data/, images all
present) -- each session dir just has its own CSV instead of one combined project-level
CSV, so this script's only job is to concatenate all 104 and split. No image copying is
needed (everything already lives at its final _raw/kaufman/ path).

kaufman is a calibrated multi-view source (shared calibration.toml, cam1/cam2 covering
the same trials) but is being built as a SINGLE-view dataset here per an explicit user
decision -- cam1 and cam2 sessions are kept as independent rows, not merged into one
multi-view sample.

Split is by subject, using the mouse-to-session mapping the user supplied at
_raw/_dlc/kaufman/labeled-sessions-mouse-date-time.txt (one "<mouse>/<timestamp>" per
line, e.g. "b8sSM5/20241208-125516"). By explicit user decision, b8sSM7 and b8sSM10 are
held out entirely for test and everything else (b8sSM5, b8sSM6, b8sSM8, b8sSM9) is
train -- not a random/percentage split. Splitting by subject (rather than by individual
labeled-data dir) also keeps a session's cam1 and cam2 views -- the same trial, viewed
twice -- on the same side of the split, since treating them as unrelated would leak
near-duplicate frames across train/test.

`videos/` and `videos_test/` were already reorganized by hand to match this subject
split (all b8sSM7/b8sSM10 sample videos in videos_test/, the rest in videos/) -- this
script only handles CollectedData.csv / CollectedData_test.csv, not video placement.

Usage:
    python scripts/preprocessing/kaufman/convert_kaufman.py
"""

import re
from pathlib import Path

import pandas as pd

from mighty_mouse.paths import load_paths

TEST_SUBJECTS = {"b8sSM7", "b8sSM10"}

SESSION_RE = re.compile(r"^(?P<timestamp>\d{8}-\d{6})-cam[12]$")


def timestamp_of(session_dir: str) -> str:
    m = SESSION_RE.match(session_dir)
    if not m:
        raise ValueError(f"unrecognized session dir name: {session_dir!r}")
    return m.group("timestamp")


def load_subject_mapping(mapping_path: Path) -> dict[str, str]:
    """Parse "<mouse>/<timestamp>" lines into {timestamp: mouse}."""
    mapping = {}
    for line in mapping_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        mouse, timestamp = line.split("/")
        mapping[timestamp] = mouse
    return mapping


def main() -> None:
    paths = load_paths()
    raw_dir = Path(paths["raw_dir"])
    out_dir = raw_dir / "kaufman"
    mapping_path = raw_dir / "_dlc" / "kaufman" / "labeled-sessions-mouse-date-time.txt"

    session_dirs = sorted(p for p in (out_dir / "labeled-data").iterdir() if p.is_dir())
    print(f"found {len(session_dirs)} session dirs")

    dfs = []
    for session_dir in session_dirs:
        csvs = list(session_dir.glob("CollectedData_*.csv"))
        assert len(csvs) == 1, (
            f"expected exactly 1 CollectedData_*.csv in {session_dir}, found {len(csvs)}"
        )
        df = pd.read_csv(csvs[0], header=[0, 1, 2], index_col=0)
        dfs.append(df)

    combined = pd.concat(dfs)
    for df in dfs[1:]:
        assert df.columns.equals(dfs[0].columns), "keypoint schemas differ across sessions"
    print(f"combined: {len(combined)} frames, {len(combined.columns) // 2} keypoints")

    session_of_row = pd.Series(
        [Path(p).parts[-2] for p in combined.index], index=combined.index
    )
    timestamp_of_row = session_of_row.map(timestamp_of)
    counts = timestamp_of_row.value_counts().to_dict()
    print(f"{len(counts)} distinct session timestamps")

    subject_of_ts = load_subject_mapping(mapping_path)
    missing = counts.keys() - subject_of_ts.keys()
    assert not missing, f"no subject mapping for timestamps: {sorted(missing)}"

    test_ts = {ts for ts in counts if subject_of_ts[ts] in TEST_SUBJECTS}
    train_ts = set(counts) - test_ts
    total = sum(counts.values())
    test_total = sum(counts[ts] for ts in test_ts)
    print(
        f"test subjects {sorted(TEST_SUBJECTS)}: {len(test_ts)} timestamps forced into "
        f"test ({test_total}/{total} frames, {test_total / total:.1%})"
    )
    print(f"timestamps: {len(train_ts)} train, {len(test_ts)} test")

    for split_name, split_timestamps in (("train", train_ts), ("test", test_ts)):
        split_df = combined.loc[timestamp_of_row.isin(split_timestamps)]
        csv_name = "CollectedData.csv" if split_name == "train" else "CollectedData_test.csv"
        split_df.to_csv(out_dir / csv_name)
        n_sessions = session_of_row[timestamp_of_row.isin(split_timestamps)].nunique()
        print(
            f"  {split_name}: {len(split_df)} frames, {n_sessions} session dirs "
            f"-> {out_dir / csv_name}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
