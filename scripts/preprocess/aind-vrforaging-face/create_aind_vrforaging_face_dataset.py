#!/usr/bin/env python3
"""
Build _raw/aind-vrforaging-face from the face-camera DeepLabCut project at
_raw/_dlc/aind-vrforaging/face-camera, as an empty-label LP project ready for pseudo-labeling.

The source has ~600 frames in `labeled-data/<session>/`, but only one session (20 frames) ever
got labels. Those 20 labels are discarded; this script keeps the frames only:

- copies every session's PNGs at native size (720x540 and 1440x1080 are mixed). The two
  `<session> (1)` folders hold different frames of the same sessions at another crop, and are
  merged into `<session>`.
- writes CollectedData.csv / CollectedData_test.csv with the KEYPOINTS columns, all empty.
  They are then filled by running the head-fixed model over the images:
      python scripts/preprocess/transfer_pseudo_labels.py --model_dir <model> \
          --target_dataset aind-vrforaging-face --keypoints <KEYPOINTS> ...
- splits by mouse (the id in `behavior_<mouse>_<date>`), pooled across sessions, via
  mighty_mouse.preprocess.subject_split; the default seed is the first landing in TEST_RANGE.
  Folders with no mouse id (`video`) count as their own subject.

Usage:
    python scripts/preprocess/aind-vrforaging-face/create_aind_vrforaging_face_dataset.py
        [--seed N]
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.subject_split import subject_split

SOURCE = "aind-vrforaging/face-camera"
SCORER = "aind-vrforaging-face"

# Canonical names (same as the head-fixed model's outputs), in labeling order.
KEYPOINTS = [
    "eye_back_right",
    "eye_bottom_right",
    "eye_front_right",
    "eye_top_right",
    "pupil_center_right",
    "nose_tip",
    "nose_top",
    "pad_top_right",
    "pad_side_right",
    "ear_top_right",
    "ear_tip_right",
    "ear_bottom_right",
    "ear_base_right",
    "wrist_left",
    "wrist_right",
]

TEST_FRACTION = 0.125  # midpoint of TEST_RANGE; the greedy split only ever overshoots
TEST_RANGE = (0.10, 0.15)


def mouse_of(session: str) -> str:
    """`behavior_754575_2024-12-06_13-20-24` -> `754575`; `video` -> `video`."""
    parts = session.split("_")
    return parts[1] if parts[0] == "behavior" and len(parts) > 1 else session


def list_frames(source: Path) -> dict[str, dict[str, Path]]:
    """session -> {PNG name: source path}. A `<session> (1)` folder (a second set of frames of
    the same session, at a different crop) is merged into `<session>`."""
    out: dict[str, dict[str, Path]] = {}
    for d in sorted((source / "labeled-data").iterdir()):
        if d.is_dir():
            frames = out.setdefault(d.name.removesuffix(" (1)"), {})
            for p in sorted(d.glob("*.png")):
                assert p.name not in frames, f"duplicate frame name {p.name} in {d.name}"
                frames[p.name] = p
    return out


def pick_split(
    frames: dict[str, dict[str, Path]], seed: int | None
) -> tuple[int, set[str], set[str]]:
    """(seed, train mice, test mice): the given seed, or the first landing in TEST_RANGE."""
    counts: dict[str, int] = {}
    for session, names in frames.items():
        counts[mouse_of(session)] = counts.get(mouse_of(session), 0) + len(names)
    total = sum(counts.values())
    for s in [seed] if seed is not None else range(1000):
        train, test = subject_split(counts, s, TEST_FRACTION)
        frac = sum(counts[m] for m in test) / total
        if seed is not None or TEST_RANGE[0] <= frac <= TEST_RANGE[1]:
            return s, train, test
    raise RuntimeError("no seed landed in the target test range")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="split seed (default: first in range)"
    )
    parser.add_argument("--out_dir", type=Path, default=None, help=f"default: <raw_dir>/{SCORER}")
    args = parser.parse_args()

    raw_dir = Path(load_paths()["raw_dir"])
    source = raw_dir / "_dlc" / SOURCE
    out_dir = args.out_dir or raw_dir / SCORER
    if (out_dir / "CollectedData.csv").exists():
        raise SystemExit(f"{out_dir} already converted; remove it first to rebuild")

    frames = list_frames(source)
    total = sum(map(len, frames.values()))
    seed, train_mice, test_mice = pick_split(frames, args.seed)
    print(f"{total} frames, {len(frames)} sessions; split seed {seed}")
    print(f"  test mice: {sorted(test_mice)}")

    columns = pd.MultiIndex.from_tuples(
        [(SCORER, k, c) for k in KEYPOINTS for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    for csv_name, mice in (
        ("CollectedData.csv", train_mice),
        ("CollectedData_test.csv", test_mice),
    ):
        paths = []
        for session, names in frames.items():
            if mouse_of(session) not in mice:
                continue
            (out_dir / "labeled-data" / session).mkdir(parents=True, exist_ok=True)
            for n, src in names.items():
                shutil.copy2(src, out_dir / "labeled-data" / session / n)
                paths.append(f"labeled-data/{session}/{n}")
        pd.DataFrame(index=pd.Index(paths), columns=columns, dtype=float).to_csv(
            out_dir / csv_name
        )
        print(f"  {csv_name}: {len(paths)} frames ({len(paths) / total:.1%})")

    for sub in ("videos", "videos_test"):
        (out_dir / sub).mkdir(exist_ok=True)
    project = {"keypoint_names": KEYPOINTS, "schema_version": 1, "view_names": []}
    (out_dir / "project.yaml").write_text(yaml.safe_dump(project, sort_keys=False))


if __name__ == "__main__":
    main()
