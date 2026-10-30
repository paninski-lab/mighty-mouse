#!/usr/bin/env python3
"""
Build _raw/hantman (side + front views combined) from the raw SLEAP export at
_raw/_dlc/hantman/{side,front}_v*.slp.

These .slp files are NOT .pkg.slp packages (no embedded frame pixels), so
`litpose convert` / lightning_pose.converters.sleap can't read them directly.
Frames must be pulled from the original .avi videos instead, which live locally
under _raw/_dlc/hantman/{side,front}/ (the .slp files reference them by their
original Windows recording-machine path, e.g. "D:/tracker_videos/side/...avi" --
only the basename is used to resolve them locally).

`side_v13.slp` additionally contains ~24k SLEAP-tracker *predicted* instances
mixed in with ~3k human-labeled ones (frame.user_instances vs. predicted);
only user instances are converted. `front_v29.slp` is 100% human-labeled.

Node naming is inconsistent between the two skeletons ("digit 4" in side vs.
"digit4" in front) -- normalized to "digit4" here so both views produce the
same source column name for configs/datasets/hantman.yaml.

Split is subject-level and pooled across both views, via mighty_mouse.preprocess.subject_split:
a subject is
the first '_'-delimited token in a session name ({subject}_{date}_{view}_{version}),
uppercased so casing slips in the source filenames (e.g. "jcr130" vs "JCR130")
don't split the same animal across train/test. All of a subject's sessions --
side and front alike -- land in the same split.

Usage:
    python scripts/preprocess/hantman-sleap/convert_hantman_sleap.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sleap_io as sio
import yaml
from PIL import Image

from mighty_mouse.paths import load_paths
from mighty_mouse.preprocess.subject_split import subject_of, subject_split

SCORER = "hantman"
TEST_FRACTION = 0.15

VIEWS = ["side", "front"]


def _find_slp(source_dir: Path, view: str) -> Path:
    matches = sorted(source_dir.glob(f"{view}_v*.slp"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {view}_v*.slp in {source_dir}, found {matches}")
    return matches[0]


def _load_view(source_dir: Path, view: str) -> tuple[list, dict]:
    """Returns (rows, images) where rows is a list of dicts with index/keypoints
    and images maps output relative path -> (H, W) uint8 array."""
    slp_file = _find_slp(source_dir, view)
    labels = sio.load_slp(slp_file)

    rows = []
    images = {}
    n_multi_instance = 0

    for lf in labels.labeled_frames:
        user_instances = lf.user_instances
        if len(user_instances) == 0:
            continue
        if len(user_instances) > 1:
            n_multi_instance += 1
        inst = user_instances[0]

        video_filename = lf.video.filename
        local_video = source_dir / view / Path(video_filename).name
        if not local_video.exists():
            raise FileNotFoundError(
                f"video not found locally: {local_video} (from {video_filename})"
            )
        session = local_video.stem

        frame_name = f"img{lf.frame_idx:08d}.png"
        rel_path = f"labeled-data/{session}/{frame_name}"

        if rel_path not in images:
            lf.video.replace_filename(str(local_video))
            img = lf.image
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            images[rel_path] = img.astype(np.uint8)

        row = {"index": rel_path, "session": session, "subject": subject_of(session)}
        for node, (x, y) in zip(inst.skeleton.nodes, inst.numpy(), strict=False):
            name = node.name.replace(" ", "")
            row[f"{name}_x"] = x
            row[f"{name}_y"] = y
        rows.append(row)

    print(f"  {view}: {len(rows)} labeled frames "
          f"({n_multi_instance} frames had >1 user instance; kept first)")
    return rows, images


def _rows_to_df(rows: list[dict], keypoints: list[str]) -> pd.DataFrame:
    tuples = [(SCORER, kp, coord) for kp in keypoints for coord in ("x", "y")]
    columns = pd.MultiIndex.from_tuples(tuples, names=["scorer", "bodyparts", "coords"])
    index = [r["index"] for r in rows]
    data = [
        [r.get(f"{kp}_{coord}", np.nan) for kp in keypoints for coord in ("x", "y")]
        for r in rows
    ]
    return pd.DataFrame(data, columns=columns, index=index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0, help="subject-split seed (default: 0)")
    args = parser.parse_args()

    paths = load_paths()
    raw_dir = Path(paths["raw_dir"])
    source_dir = raw_dir / "_dlc" / "hantman"

    all_rows = []
    all_images: dict = {}
    for view in VIEWS:
        print(f"\n── {view} ──────────────────────────────────────────────")
        rows, images = _load_view(source_dir, view)
        all_rows.extend(rows)
        all_images.update(images)

    keypoints = sorted({k[:-2] for r in all_rows for k in r if k.endswith("_x")})
    print(f"\nkeypoints found: {keypoints}")

    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["subject"]] = counts.get(r["subject"], 0) + 1
    train_subjects, test_subjects = subject_split(counts, args.seed, TEST_FRACTION)
    print(f"subjects: {len(train_subjects)} train, {len(test_subjects)} test "
          f"(pooled across both views)")

    out_dir = raw_dir / "hantman"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_subjects in (("train", train_subjects), ("test", test_subjects)):
        split_rows = [r for r in all_rows if r["subject"] in split_subjects]
        df = _rows_to_df(split_rows, keypoints)
        csv_name = "CollectedData.csv" if split_name == "train" else "CollectedData_test.csv"
        df.to_csv(out_dir / csv_name)
        print(f"  {split_name}: {len(df)} frames -> {out_dir / csv_name}")

    n_saved = 0
    for rel_path, img in all_images.items():
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            Image.fromarray(img).save(dst)
            n_saved += 1
    print(f"  images: {n_saved} saved, {len(all_images) - n_saved} already present")

    # project.yaml lets the LP labeling app open this as a project directly. The
    # source here (raw SLEAP) has no equivalent file, so this is written from
    # scratch -- schema_version/view_names follow the convention other _raw/
    # datasets' project.yaml files use.
    project = {"keypoint_names": keypoints, "schema_version": 1, "view_names": []}
    with open(out_dir / "project.yaml", "w") as f:
        yaml.safe_dump(project, f, default_flow_style=False, sort_keys=False)
    print(f"  project.yaml written ({len(keypoints)} keypoints, view_names: [])")

    (out_dir / "videos").mkdir(exist_ok=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
