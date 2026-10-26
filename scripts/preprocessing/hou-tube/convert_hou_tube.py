#!/usr/bin/env python3
"""
Build _raw/hou-tube (standard DLC layout) from _raw/_dlc/hou-tube.

The source is a six-camera (L/R/TL/TR/BC/TC) recording rig, with one directory per
session-and-view (e.g. "20260612_Bri_Socialtubebreadcontrol_17-26-25_record_BC")
holding the labeled PNGs plus an `annotations.yaml` of the form

    <keypoint>:
      <img>.png:
      - - x
        - y

(`.nan` or `null` = unlabeled). There is no CollectedData CSV, no project.yaml and no train/test
split, so this script builds all three. Views are merged into one single-view project:
the source directory names already encode the view, so they're used verbatim as the
session names and nothing collides.

Only the images referenced by an annotations.yaml are copied (Thumbs.db etc. are not).

Split is session-level: every view of the held-out recording goes to test, so the same
animal/moment never appears in both splits from a different angle. A "recording" is
the source directory name minus its trailing "_record_<view>".

Usage:
    python scripts/preprocessing/hou-tube/convert_hou_tube.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mighty_mouse.paths import load_paths

# Recording held out for test (all 6 views). Chosen by the user; it's also the only
# session from a different experimenter than the other five.
TEST_RECORDINGS = ["20260612_Monica_Socialtubebreadcontrol_17-32-39"]
SCORER = "hou-tube"


def recording_of(session: str) -> str:
    return session.rsplit("_record_", 1)[0]


def _to_float(v) -> float:
    return np.nan if v is None else float(v)


def load_session(session_dir: Path) -> pd.DataFrame:
    """One session-view's annotations.yaml -> DataFrame (index = labeled-data/... path)."""
    with open(session_dir / "annotations.yaml") as f:
        ann = yaml.safe_load(f)
    frames = sorted(next(iter(ann.values())).keys())
    cols = {}
    for kp, per_frame in ann.items():
        assert sorted(per_frame) == frames, f"{session_dir.name}: {kp} frames differ"
        for i, coord in enumerate(("x", "y")):
            cols[(SCORER, kp, coord)] = [_to_float(per_frame[fr][0][i]) for fr in frames]
    df = pd.DataFrame(cols, index=[f"labeled-data/{session_dir.name}/{fr}" for fr in frames])
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["scorer", "bodyparts", "coords"])
    return df


def main() -> None:
    paths = load_paths()
    raw_dir = Path(paths["raw_dir"])
    source_dir = raw_dir / "_dlc" / "hou-tube"
    out_dir = raw_dir / "hou-tube"
    out_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(p for p in source_dir.iterdir() if (p / "annotations.yaml").exists())
    dfs = [load_session(p) for p in session_dirs]
    keypoints = list(dict.fromkeys(c[1] for c in dfs[0].columns))
    for p, df in zip(session_dirs, dfs, strict=True):
        assert set(c[1] for c in df.columns) == set(keypoints), f"{p.name}: keypoint set differs"
    combined = pd.concat([df[dfs[0].columns] for df in dfs])
    print(f"{len(session_dirs)} session-views, {len(combined)} frames, {len(keypoints)} keypoints")

    sessions = [Path(p).parts[-2] for p in combined.index]
    is_test = np.array([recording_of(s) in TEST_RECORDINGS for s in sessions])
    assert is_test.any(), "no test recording matched"
    for name, mask, csv_name in (
        ("train", ~is_test, "CollectedData.csv"),
        ("test", is_test, "CollectedData_test.csv"),
    ):
        combined[mask].to_csv(out_dir / csv_name)
        print(f"  {name}: {mask.sum()} frames -> {out_dir / csv_name}")

    n_copied = 0
    for rel_path in combined.index:
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(source_dir / Path(rel_path).relative_to("labeled-data"), dst)
            n_copied += 1
    print(f"  images: {n_copied} copied, {len(combined) - n_copied} already present")

    with open(out_dir / "project.yaml", "w") as f:
        yaml.safe_dump(
            {"keypoint_names": keypoints, "schema_version": 1, "view_names": []},
            f, default_flow_style=False, sort_keys=False,
        )
    (out_dir / "videos").mkdir(exist_ok=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
