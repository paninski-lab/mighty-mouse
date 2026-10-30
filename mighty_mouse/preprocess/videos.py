"""
Extract a fixed-length clip containing the most movement from a longer video.

Movement ("motion energy") is measured either from pose predictions (mean keypoint
displacement between frames, for keypoints above a likelihood threshold) or, when no
predictions are available, from raw pixel differences on a downsampled version of the
video. The output clip is always re-encoded to h264/yuv420p mp4, regardless of the
source codec/container, so downstream tools have one predictable target format.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from mighty_mouse.labels import read_labels_csv


def make_video_snippet(
    video_file: Path,
    out_dir: Path,
    preds_file: Path | None = None,
    clip_length: int = 15,
    likelihood_thresh: float = 0.9,
    skip_start: float = 60.0,
    from_start: bool = False,
    fps: float | None = None,
    crf: int = 23,
    preset: str = "medium",
) -> tuple[Path, int, float]:
    """Extract a `clip_length`-second clip from a video.

    Parameters
    ----------
    video_file: path to the source video
    out_dir: directory to save the clip in (created if it doesn't exist)
    preds_file: csv of pose predictions; if given, movement is measured from keypoint
        displacement instead of raw pixels. Ignored if from_start is True.
    clip_length: length of the clip in seconds
    likelihood_thresh: when using preds_file, only count keypoints with a likelihood
        above this threshold (0-1) toward the movement measure
    skip_start: ignore this many seconds at the start of the video -- when searching
        for the highest-motion window (e.g. to skip past camera setup/handling), or,
        with from_start, as the point the clip itself starts from
    from_start: if True, skip the motion-energy search and just take the
        `clip_length`-second window starting at skip_start
    fps: override the frame rate used for all time math (skip_start/clip_length <->
        frame index) and for reading the source video, for videos whose container
        reports the wrong frame rate (e.g. a camera's true capture rate rather than
        the rate it was saved at). Defaults to the frame rate reported by the video
        file itself.
    crf: h264 constant rate factor (lower = higher quality/larger file)
    preset: ffmpeg x264 preset (encode speed vs. compression efficiency tradeoff)

    Returns
    -------
    - path to the output clip
    - clip start: frame index
    - clip start: seconds

    """
    video = cv2.VideoCapture(str(video_file))
    native_fps = video.get(cv2.CAP_PROP_FPS)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    effective_fps = fps if fps is not None else native_fps
    win_len = int(effective_fps * clip_length)
    skip_frames = int(effective_fps * skip_start)
    n_frames_considered = n_frames - skip_frames

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{video_file.stem}.short.mp4"

    encode_flags = [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
    ]

    if from_start or win_len >= n_frames_considered:
        # either explicitly requested, or the remaining video (after skip_start) is
        # already shorter than the requested clip length -- keep it as is, just
        # re-encode to guarantee h264/yuv420p/mp4
        clip_start_idx = skip_frames
        clip_start_sec = skip_start
    else:
        # find the `clip_length`-second window with the highest average motion energy,
        # considering only frames at or after skip_start
        if preds_file is None:
            me = compute_video_motion_energy(video_file, start_frame=skip_frames)
        else:
            df = read_labels_csv(preds_file)
            me = compute_motion_energy_from_prediction_df(df, likelihood_thresh)
            me = me[skip_frames:]
        me_win = pd.Series(me).rolling(window=win_len, center=False).mean()
        # rolling() places each result at the window's right edge; shift back to the start
        clip_start_idx = int(me_win.argmax() - win_len) + skip_frames
        clip_start_sec = clip_start_idx / effective_fps
        if np.isnan(clip_start_sec) or clip_start_idx < skip_frames:
            # all predictions were below likelihood_thresh -- fall back to skip_start
            clip_start_idx, clip_start_sec = skip_frames, skip_start

    ffmpeg_cmd = ["ffmpeg", "-y"]
    if fps is not None:
        # ignore the container's (wrong) timestamps and regenerate them at the true
        # capture rate, both when reading the input and timing the output
        ffmpeg_cmd += ["-r", str(fps)]
    ffmpeg_cmd += ["-ss", str(clip_start_sec), "-i", str(video_file)]
    if from_start or win_len < n_frames_considered:
        ffmpeg_cmd += ["-t", str(clip_length)]
    ffmpeg_cmd += [*encode_flags, str(dst)]

    if not dst.exists():
        subprocess.run(ffmpeg_cmd, check=True)

    return dst, clip_start_idx, float(clip_start_sec)


def compute_video_motion_energy(
    video_file: Path,
    resize_dims: int = 32,
    start_frame: int = 0,
) -> np.ndarray:
    """Per-frame motion energy: summed absolute pixel difference from the prior frame."""
    frames = read_nth_frames(
        video_file=video_file, n=1, resize_dims=resize_dims, start_frame=start_frame
    )
    batches = frames.reshape(frames.shape[0], -1)

    diffs = np.concatenate([np.zeros((1, batches.shape[1])), np.diff(batches, axis=0)])
    me = np.sum(np.abs(diffs), axis=1)

    return me


def compute_motion_energy_from_prediction_df(
    df: pd.DataFrame,
    likelihood_thresh: float,
) -> np.ndarray:
    """Per-frame motion energy: mean keypoint displacement from the prior frame."""
    kps_and_conf = df.to_numpy().reshape(df.shape[0], -1, 3)
    kps = kps_and_conf[:, :, :2]
    conf = kps_and_conf[:, :, -1]
    conf2 = np.concatenate([conf[:, :, None], conf[:, :, None]], axis=2)

    kps[conf2 < likelihood_thresh] = np.nan

    me = np.nanmean(np.linalg.norm(kps[1:] - kps[:-1], axis=2), axis=-1)
    me = np.concatenate([[0], me])
    return me


def read_nth_frames(
    video_file: Path,
    n: int = 1,
    resize_dims: int = 64,
    start_frame: int = 0,
) -> np.ndarray:
    """Read every nth frame of a video from start_frame onward, resized to
    (resize_dims, resize_dims), as RGB."""
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise OSError(f"Error opening video file {video_file}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    frame_counter = 0
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    with tqdm(total=frame_total, desc=f"scanning {video_file.name}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_counter % n == 0:
                frame_resize = cv2.resize(frame, (resize_dims, resize_dims))
                frame_rgb = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb.astype(np.float16))
            frame_counter += 1
            pbar.update(1)

    cap.release()

    return np.array(frames)
