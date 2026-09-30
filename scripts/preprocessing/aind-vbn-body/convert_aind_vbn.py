#!/usr/bin/env python3
"""
Build _raw/aind-vbn-body or _raw/aind-vbn-face from the per-session DLC projects in
_raw/_dlc/aind-vbn.

The source is one standard DLC project per (session, camera): directories named
"<session_id>_<mouse_id>_<date>.<behavior|face>-<scorer>-<label date>", each with its
own labeled-data/<session>.<camera>/CollectedData_<scorer>.csv and a (hours-long)
video. "behavior" is the body/side camera, "face" is the face camera; they're kept as
two parallel, independent datasets (aind-vbn-body / aind-vbn-face) rather than merged.
Pick one with --view.

Conversion is a concat of the per-session CSVs (scorer normalized to "aind-vbn", since
it varies by annotator and would otherwise misalign columns), copying only the
labeled images, plus a 15 s clip of each session's video.

Body only: each frame is cropped to the face + upper trunk and resized to 256x256.
With H = |nose_tip.x - eye_mid.x| (eye_mid = mean of eye_top_l/eye_bottom_l) and
C = mean(nose_tip, eye_mid), the crop box is [C-2H, C+3H] in both x and y. Frames
missing nose or eyes use the session's median H and C. Keypoints are shifted/scaled
with the crop and set to NaN if they land outside it. The *_lh, *_rh and tail_*
keypoints are dropped entirely. Each session's video clip is cropped with that session's
median box (median H, cx, cy over its labeled frames) and resized to 256x256.

Split is subject-level by mouse id (second '_'-delimited token of the session), via
mighty_mouse.subject_split. The test mice are always chosen from the *body* dataset's
frame counts, so aind-vbn-face holds out exactly the same mice.

Ignores source-side non-data: old/, time_stamps.csv, vbn_video_metadata.csv.

Usage:
    python scripts/preprocessing/aind-vbn-body/convert_aind_vbn.py --view body
    python scripts/preprocessing/aind-vbn-body/convert_aind_vbn.py --view face
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.subject_split import subject_split
from mighty_mouse.videos import make_video_snippet

# Source camera suffix for each output dataset.
VIEW_TO_CAMERA = {"body": "behavior", "face": "face"}
# Target ~10-15% of frames in test; aim at the midpoint since subject_split only overshoots.
TEST_FRACTION = 0.125
SCORER = "aind-vbn"
CLIP_LENGTH = 15
CROP_SIZE = 256
DROP_SUFFIXES = ("_lh", "_rh")
DROP_PREFIXES = ("tail_",)
CLIP_SKIP_START = 60.0  # skip camera setup/handling; the videos are ~2.7 h so no motion search


def mouse_of(session: str) -> str:
    """Mouse id from '<session_id>_<mouse_id>_<date>.<camera>'."""
    return session.split("_")[1]


def load_view(source_dir: Path, camera: str) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Concat every per-session CSV for `camera`; also return session -> video path."""
    dfs, videos = [], {}
    for proj in sorted(source_dir.glob(f"*.{camera}-*")):
        (session_dir,) = (proj / "labeled-data").iterdir()
        (csv,) = session_dir.glob("CollectedData_*.csv")
        df = pd.read_csv(csv, header=[0, 1, 2], index_col=[0, 1, 2])
        df.index = pd.Index(["/".join(i) for i in df.index])
        df.columns = pd.MultiIndex.from_tuples([(SCORER, b, c) for _, b, c in df.columns])
        df.columns.names = ["scorer", "bodyparts", "coords"]
        dfs.append(df)
        video = proj / "videos" / f"{session_dir.name}.mp4"
        assert video.exists(), video
        videos[session_dir.name] = video
    assert all(d.columns.equals(dfs[0].columns) for d in dfs), "keypoint schemas differ"
    return pd.concat(dfs), videos


def box_from(h: pd.Series, cx: pd.Series, cy: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"x0": cx - 2 * h, "y0": cy - 2 * h, "side": 5 * h})


def crop_boxes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-frame (x0, y0, side) crop boxes, and per-session boxes from the median H, C over
    each session's frames with nose + eyes. Frames lacking nose/eyes get the session median."""
    xy = lambda bp: (df[(SCORER, bp, "x")].to_numpy(), df[(SCORER, bp, "y")].to_numpy())
    (nx, ny), (ex1, ey1), (ex2, ey2) = xy("nose_tip"), xy("eye_top_l"), xy("eye_bottom_l")
    ex, ey = (ex1 + ex2) / 2, (ey1 + ey2) / 2
    boxes = pd.DataFrame(
        {"H": np.abs(nx - ex), "cx": (nx + ex) / 2, "cy": (ny + ey) / 2}, index=df.index
    )
    sessions = session_of_index(df.index)
    med = boxes.groupby(sessions).median()
    assert med.notna().all().all(), "a session has no frame with nose + eyes"
    boxes = boxes.fillna(boxes.groupby(sessions).transform("median"))
    return box_from(boxes.H, boxes.cx, boxes.cy), box_from(med.H, med.cx, med.cy)


def crop_video(src: Path, dst: Path, x0: float, y0: float, side: float) -> None:
    """Crop (black-padded if the box leaves the frame) and resize to CROP_SIZE, as h264 mp4."""
    pad = 200
    x0, y0, side = round(x0), round(y0), round(side)
    vf = (
        f"pad=iw+{2 * pad}:ih+{2 * pad}:{pad}:{pad}:black,"
        f"crop={side}:{side}:{x0 + pad}:{y0 + pad},scale={CROP_SIZE}:{CROP_SIZE}"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", "-preset", "medium", str(dst)],
        check=True,
    )


def crop_image(src: Path, x0: float, y0: float, side: float) -> np.ndarray:
    """Crop the (possibly out-of-bounds, zero-padded) box and resize to CROP_SIZE."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    scale = CROP_SIZE / side
    # affine warp handles sub-pixel offsets, out-of-bounds padding and resize in one step
    M = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    return cv2.warpAffine(img, M, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def crop_keypoints(df: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scale = CROP_SIZE / boxes.side
    for col in df.columns:
        off = boxes.x0 if col[2] == "x" else boxes.y0
        df[col] = (df[col] - off) * scale
    for bp in dict.fromkeys(df.columns.get_level_values("bodyparts")):
        x, y = df[(SCORER, bp, "x")], df[(SCORER, bp, "y")]
        outside = ~((x >= 0) & (x < CROP_SIZE) & (y >= 0) & (y < CROP_SIZE)) & (x.notna() | y.notna())
        df.loc[outside, [(SCORER, bp, "x"), (SCORER, bp, "y")]] = np.nan
    return df


def session_of_index(index: pd.Index) -> pd.Series:
    return pd.Series([Path(p).parts[-2] for p in index], index=index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", choices=VIEW_TO_CAMERA, required=True)
    parser.add_argument("--seed", type=int, default=0, help="subject-split seed (default: 0)")
    args = parser.parse_args()

    raw_dir = Path(load_paths()["raw_dir"])
    source_dir = raw_dir / "_dlc" / "aind-vbn"
    out_dir = raw_dir / f"aind-vbn-{args.view}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # test mice come from the body dataset so both views hold out the same animals
    body_df, _ = load_view(source_dir, VIEW_TO_CAMERA["body"])
    body_mice = session_of_index(body_df.index).map(mouse_of)
    train_mice, test_mice = subject_split(body_mice.value_counts().to_dict(), args.seed, TEST_FRACTION)
    print(f"mice: {len(train_mice)} train, {len(test_mice)} test ({sorted(test_mice)})")

    df, videos = load_view(source_dir, VIEW_TO_CAMERA[args.view])
    if args.view == "body":
        keep = [c for c in df.columns if not (c[1].endswith(DROP_SUFFIXES) or c[1].startswith(DROP_PREFIXES))]
        df = df[keep]
        boxes, session_boxes = crop_boxes(df)
        df = crop_keypoints(df, boxes)
        print(f"  cropped to {CROP_SIZE}x{CROP_SIZE}; kept {len(df.columns) // 2} keypoints")
    sessions = session_of_index(df.index)
    is_test = sessions.map(mouse_of).isin(test_mice)
    print(f"{args.view}: {len(df)} frames, {len(videos)} sessions, {len(df.columns) // 2} keypoints")

    for name, split_df in (("CollectedData.csv", df[~is_test]), ("CollectedData_test.csv", df[is_test])):
        split_df.to_csv(out_dir / name)
        print(f"  {name}: {len(split_df)} frames ({len(split_df) / len(df):.1%})")

    # copy only labeled images (cropped + resized for body)
    src_proj = {s: p.parents[1] for s, p in videos.items()}
    for rel in df.index:
        session = Path(rel).parts[-2]
        src, dst = src_proj[session] / rel, out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.view == "body":
            b = boxes.loc[rel]
            cv2.imwrite(str(dst), crop_image(src, b.x0, b.y0, b.side))
        else:
            shutil.copy2(src, dst)

    # 15 s clip per session video; test-mouse sessions go to videos_test/
    for session, video in videos.items():
        clip_dir = out_dir / ("videos_test" if mouse_of(session) in test_mice else "videos")
        if args.view == "body":
            tmp_dir = out_dir / ".clips_full"
            full, _, _ = make_video_snippet(
                video, tmp_dir, clip_length=CLIP_LENGTH, skip_start=CLIP_SKIP_START, from_start=True
            )
            clip_dir.mkdir(exist_ok=True)
            dst = clip_dir / full.name
            b = session_boxes.loc[session]
            crop_video(full, dst, b.x0, b.y0, b.side)
            full.unlink()
        else:
            dst, _, _ = make_video_snippet(
                video, clip_dir, clip_length=CLIP_LENGTH, skip_start=CLIP_SKIP_START, from_start=True
            )
        print(f"  {video.name} -> {dst.relative_to(out_dir)}")

    if args.view == "body":
        shutil.rmtree(out_dir / ".clips_full", ignore_errors=True)

    project = {
        "keypoint_names": [b for b in dict.fromkeys(df.columns.get_level_values("bodyparts"))],
        "schema_version": 1,
        "view_names": [],
    }
    with open(out_dir / "project.yaml", "w") as f:
        yaml.safe_dump(project, f, default_flow_style=False, sort_keys=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
