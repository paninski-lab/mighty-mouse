"""Stage 3 of frame selection: turning selected frames into pretraining images.

Selected frames are decoded from their videos, screened for junk (blank, black or blown-out
frames, which the embeddings can't tell from rare poses), downscaled so the short side is at
most `MIN_SIDE`, and saved as `<dataset>/<video_key>/img<frame>.png`. The selection of one
method is a `selection.csv` with columns `dataset`, `video_key`, `frame_index`.

Frames are BGR uint8 arrays, as OpenCV reads and writes them.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MIN_SIDE = 256                  # shortest side of saved frames (smaller frames stay as is)
N_DIGITS = 8                    # zero-padded width of the frame index in file names
SEEK_MIN_GAP = 300              # gaps of at least this many frames are seeked, not decoded

# a frame is junk if its gray std is below JUNK_MAX_STD (blank, black, flat) or more than
# JUNK_MIN_SATURATED of its pixels are brighter than JUNK_SATURATED_LEVEL (blown out)
JUNK_MAX_STD = 10.0
JUNK_SATURATED_LEVEL = 245
JUNK_MIN_SATURATED = 0.5
# a method's frame set is rejected if more than this share is junk, overall or in any dataset
MAX_JUNK_OVERALL = 0.005
MAX_JUNK_PER_DATASET = 0.02


def resize_min_side(frame: np.ndarray, min_side: int = MIN_SIDE) -> np.ndarray:
    """Downscale a frame so its shorter side is `min_side`, keeping the aspect ratio.

    Frames whose shorter side is already at most `min_side` are returned unchanged (never
    upscaled).
    """
    h, w = frame.shape[:2]
    short = min(h, w)
    if short <= min_side:
        return frame
    scale = min_side / short
    size = (max(round(w * scale), 1), max(round(h * scale), 1))  # cv2 takes (width, height)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def junk_stats(frame: np.ndarray) -> tuple[float, float]:
    """`(gray std, share of pixels brighter than JUNK_SATURATED_LEVEL)` of a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.std()), float((gray > JUNK_SATURATED_LEVEL).mean())


def is_junk(gray_std: float, saturated: float) -> bool:
    """Whether a frame with these `junk_stats` is blank/flat or blown out."""
    return gray_std < JUNK_MAX_STD or saturated > JUNK_MIN_SATURATED


def frame_path(root: Path, dataset: str, key: str, frame_idx: int) -> Path:
    """`<root>/<dataset>/<key>/img<frame_idx>.png`, the frame index zero-padded."""
    return Path(root) / dataset / key / f"img{frame_idx:0{N_DIGITS}d}.png"


def read_frames(
    video_file: Path, frame_idxs, min_gap: int = SEEK_MIN_GAP
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame index, BGR frame)` for the requested frames of a video, in order.

    A frame less than `min_gap` frames ahead of the current position is reached by decoding
    forward; a farther one by seeking, which only decodes from the nearest earlier keyframe.
    Frames were identical either way on every dataset (see the 100k extraction notes), so no
    per-video choice is needed. If the capture doesn't report the position it was asked to seek
    to, the video is reopened and read sequentially from there on.

    Raises:
        OSError: if the video can't be opened, or ends before a requested frame
    """
    wanted = sorted({int(i) for i in frame_idxs})
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise OSError(f"Error opening video file {video_file}")
    try:
        pos, can_seek = 0, True
        for idx in wanted:
            if can_seek and idx - pos >= min_gap:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == idx:
                    pos = idx
                else:  # seeking isn't trustworthy for this file: start over, sequentially
                    can_seek = False
                    cap.release()
                    cap = cv2.VideoCapture(str(video_file))
                    pos = 0
            while pos < idx:
                if not cap.grab():
                    raise OSError(f"{video_file} ends at frame {pos}, before frame {idx}")
                pos += 1
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"{video_file} ends at frame {pos}, before frame {idx}")
            pos += 1
            yield idx, frame
    finally:
        cap.release()


def save_frame(path: Path, frame: np.ndarray) -> None:
    """Write a PNG via a temporary name, so an interrupted run never leaves a partial image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise OSError(f"could not encode frame for {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(encoded.tobytes())
    os.replace(tmp, path)


def plan_extraction(
    selections: dict[str, pd.DataFrame], done: dict[str, set[tuple[str, str, int]]]
) -> dict[tuple[str, str], dict[int, list[str]]]:
    """Group the frames still to extract by video.

    Args:
        selections: method -> selection table (`dataset`, `video_key`, `frame_index`)
        done: method -> `(dataset, video_key, frame_index)` already extracted or dropped;
            these are left out

    Returns:
        `(dataset, video_key)` -> `{frame_index: [methods that selected it]}`. A frame chosen
        by several methods appears once, so it is decoded once.

    Raises:
        ValueError: if a video key belongs to more than one dataset (their frames would end
            up in different folders but be indistinguishable to the sampler)
    """
    plan: dict[tuple[str, str], dict[int, list[str]]] = {}
    owner: dict[str, str] = {}
    for method, table in selections.items():
        for dataset, key, idx in zip(
            table["dataset"], table["video_key"], table["frame_index"], strict=True
        ):
            if owner.setdefault(key, dataset) != dataset:
                raise ValueError(
                    f"video key {key!r} is in datasets {owner[key]!r} and {dataset!r}"
                )
            if (dataset, key, int(idx)) in done.get(method, ()):
                continue
            plan.setdefault((dataset, key), {}).setdefault(int(idx), []).append(method)
    return plan


def junk_counts(selected: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset and overall number of selected and dropped (junk) frames of one method.

    Args:
        selected: selection table (needs a `dataset` column)
        dropped: dropped frames (needs a `dataset` column)

    Returns:
        a frame indexed by dataset, plus a final `all` row, with `n_selected`, `n_dropped`
        and `junk_fraction`
    """
    counts = pd.DataFrame({
        "n_selected": selected["dataset"].value_counts(),
        "n_dropped": dropped["dataset"].value_counts(),
    }).fillna(0).astype(int).sort_index()
    counts.loc["all"] = counts.sum()
    counts["junk_fraction"] = counts["n_dropped"] / counts["n_selected"].clip(lower=1)
    return counts


def junk_violations(
    counts: pd.DataFrame,
    max_overall: float = MAX_JUNK_OVERALL,
    max_per_dataset: float = MAX_JUNK_PER_DATASET,
) -> list[str]:
    """Messages for each junk share in `junk_counts` output above its limit (empty if fine)."""
    messages = []
    for name, share in counts["junk_fraction"].items():
        limit = max_overall if name == "all" else max_per_dataset
        if share > limit:
            n = int(counts.loc[name, "n_dropped"])
            total = int(counts.loc[name, "n_selected"])
            messages.append(f"{name}: {n}/{total} frames are junk ({share:.2%} > {limit:.2%})")
    return messages
