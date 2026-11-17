#!/usr/bin/env python3
"""
Build _raw/aind-foraging-bot from the September 2022 DeepLabCut project at
_raw/_dlc/aind-forage/Foraging_Bot_UpdatingOld_20220927-LucasKinsey-2022-09-27.

The source is one standard DLC project: a bottom-up camera on a head-fixed mouse at a lick
spout, one `labeled-data/<video>/CollectedData_LucasKinsey.csv` per video (the scorer in the
CSV header is "Han_Lucas", not the project scorer). This script:

- keeps 7 of the 17 keypoints and renames them (RENAME below): nose tip, tongue tip, the two
  front tongue corners, jaw (-> lowerlip) and the two front paws. Dropped entirely (not copied):
  lickports, the back tongue points and all whisker points.
- the camera views the mouse from below, so the source's image-left labels (`L`) are the
  animal's right, and left/right are swapped in the output names.
- keeps native image sizes (crops differ per video: 272x272, 352x312, 304x272, 240x254, ...).
- normalizes the scorer to "aind-foraging-bot".
- splits at the video level (one video = one subject; there are no mouse ids), via
  mighty_mouse.preprocess.subject_split; the default seed is the first one whose test videos
  all have a video file in the source. Only the images referenced by a CSV row are copied.
- copies each video that exists in the source into `videos/` (train) or `videos_test/` (test);
  some labeled videos have no video file in the source and are skipped.

Usage:
    python scripts/preprocess/aind-foraging-bot/convert_aind_foraging_bot.py [--seed N]
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.subject_split import subject_split

SOURCE_PROJECT = "aind-forage/Foraging_Bot_UpdatingOld_20220927-LucasKinsey-2022-09-27"
SCORER = "aind-foraging-bot"

# source column -> output keypoint (output order). Image-left is the animal's right.
RENAME = {
    "nosetip": "nose_tip",
    "tongueTip": "tongue_tip",
    "tongueLeftFront": "tongue_end_right",
    "tongueRightFront": "tongue_end_left",
    "jaw": "lowerlip",
    "pawR": "paw_wrist_left",
    "pawL": "paw_wrist_right",
}

# Target ~10-15% of frames in test; midpoint, because the greedy split only ever overshoots.
TEST_FRACTION = 0.125
TEST_RANGE = (0.10, 0.15)


def load_source(source: Path) -> pd.DataFrame:
    """All labeled rows, indexed by (video, frame) with the 7 renamed keypoints."""
    frames = []
    for csv in sorted((source / "labeled-data").glob("*/CollectedData_*.csv")):
        df = pd.read_csv(csv, header=[0, 1, 2], index_col=[0, 1, 2])
        df.index = pd.MultiIndex.from_arrays(
            [[csv.parent.name] * len(df), df.index.get_level_values(2)],
            names=["video", "frame"],
        )
        df.columns = pd.MultiIndex.from_tuples([(SCORER, b, c) for _, b, c in df.columns])
        frames.append(df[[(SCORER, k, c) for k in RENAME for c in ("x", "y")]])
    df = pd.concat(frames)
    return df.rename(columns=RENAME, level=1)


def pick_split(
    counts: dict[str, int], seed: int | None, with_video: set[str]
) -> tuple[int, set[str], set[str]]:
    """Split with the given seed, or the first seed (from 0) landing in TEST_RANGE whose test
    videos all have a source video file (`with_video`)."""
    total = sum(counts.values())
    for s in [seed] if seed is not None else range(1000):
        train, test = subject_split(counts, s, TEST_FRACTION)
        frac = sum(counts[k] for k in test) / total
        if seed is not None or (TEST_RANGE[0] <= frac <= TEST_RANGE[1] and test <= with_video):
            return s, train, test
    raise RuntimeError("no seed landed in the target test range with all test videos present")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="split seed (default: first in range)"
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None, help="default: <raw_dir>/aind-foraging-bot"
    )
    args = parser.parse_args()

    raw_dir = Path(load_paths()["raw_dir"])
    source = raw_dir / "_dlc" / SOURCE_PROJECT
    out_dir = args.out_dir or raw_dir / "aind-foraging-bot"
    if (out_dir / "CollectedData.csv").exists():
        raise SystemExit(f"{out_dir} already converted; remove it first to rebuild")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_source(source)
    print(f"{len(df)} frames, {df.index.get_level_values('video').nunique()} videos")

    counts = df.groupby(level="video").size().to_dict()
    with_video = {v for v in counts if (source / "videos" / f"{v}.avi").exists()}
    seed, train_videos, test_videos = pick_split(counts, args.seed, with_video)
    print(f"split seed {seed}: {len(train_videos)} train / {len(test_videos)} test videos")
    print(f"  test videos: {sorted(test_videos)}")

    (out_dir / "videos").mkdir(exist_ok=True)
    (out_dir / "videos_test").mkdir(exist_ok=True)
    for name, split_videos in (("train", train_videos), ("test", test_videos)):
        split = df[df.index.get_level_values("video").isin(split_videos)]
        paths = [f"labeled-data/{v}/{f}" for v, f in split.index]
        for (v, f), p in zip(split.index, paths, strict=True):
            dst = out_dir / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "labeled-data" / v / f, dst)
        out = split.copy()
        out.index = pd.Index(paths)
        csv_name = "CollectedData.csv" if name == "train" else "CollectedData_test.csv"
        out.to_csv(out_dir / csv_name)
        print(f"  {name}: {len(split)} frames ({len(split) / len(df):.1%})")

        vid_dir = out_dir / ("videos" if name == "train" else "videos_test")
        n_copied = 0
        for v in sorted(split_videos):
            src_video = source / "videos" / f"{v}.avi"
            if src_video.exists():
                shutil.copy2(src_video, vid_dir / src_video.name)
                n_copied += 1
            else:
                print(f"  (no source video for {v})")
        print(f"  {name}: {n_copied}/{len(split_videos)} videos copied")

    project = {
        "keypoint_names": list(RENAME.values()),
        "schema_version": 1,
        "view_names": [],
    }
    (out_dir / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False))


if __name__ == "__main__":
    main()
