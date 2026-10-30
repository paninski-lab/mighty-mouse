"""
House style for overlay videos (skill `pose-video`, section "House style"): the standing keypoint-
group palette, text helpers, the square letterboxed panel and the h264 encode every overlay-video
renderer shares. Colours are BGR (cv2).
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

# ── keypoint groups: one saturated BGR hue each, nothing white or grey ──────────
GROUPS = [
    ("eye",          (255, 255,   0), ["eye_"]),
    ("pupil",        (0,   255, 255), ["pupil_"]),
    ("ear",          (255,   0, 255), ["ear_"]),
    ("nose",         (0,   140, 255), ["nose_"]),
    ("whisker pad",  (150, 190,   0), ["pad_"]),
    ("mouth/lips",   (255, 130,  60), ["mouth", "lowerlip", "upperlip_"]),
    ("tongue",       (170, 110, 255), ["tongue_"]),
    ("wrist",        (255,   0, 127), ["wrist_"]),
    # digit tips are split by side, so a left vs right channel is told apart: right green, left red
    ("digit tips R", (0,   255, 120), [f"d{i}_tip_right" for i in range(1, 5)]),
    ("digit tips L", (0,     0, 255), [f"d{i}_tip_left" for i in range(1, 5)]),
]
BANNER_H = 62
FONT     = cv2.FONT_HERSHEY_SIMPLEX


def color(kp: str) -> tuple[int, int, int]:
    """BGR colour of a keypoint's group (grey only for a name outside every group)."""
    for _, col, prefixes in GROUPS:
        if any(kp.startswith(p) for p in prefixes):
            return col
    return (200, 200, 200)


def put(img: np.ndarray, xy: tuple[int, int], text: str, fs: float = 0.5,
        col: tuple[int, int, int] = (235, 235, 235)) -> int:
    """Draw text at xy; return the x just after it."""
    cv2.putText(img, text, xy, FONT, fs, col, 1, cv2.LINE_AA)
    return xy[0] + cv2.getTextSize(text, FONT, fs, 1)[0][0]


def fit(text: str, width: int, fs: float = 0.5, floor: float = 0.28) -> float:
    """Largest font scale <= fs at which text fits in width pixels."""
    while cv2.getTextSize(text, FONT, fs, 1)[0][0] > width and fs > floor:
        fs -= 0.02
    return fs


def legend_row(canvas: np.ndarray, kps: list[str] | None = None, r: int = 5,
               fs: float | None = None, x: int = 8, y: int = 20) -> int:
    """Group swatches across the top of the banner; with kps, only groups that occur in it."""
    for name, col, prefixes in GROUPS:
        if kps is not None and not any(k.startswith(tuple(prefixes)) for k in kps):
            continue
        cv2.circle(canvas, (x + r, y - 4), r, col, -1, cv2.LINE_AA)
        size = fs if fs is not None else fit(name, 200, 0.42)
        x = put(canvas, (x + 2 * r + 3, y), name, size) + 12
    return x


def square(img: np.ndarray, size: int,
           interp: int = cv2.INTER_AREA) -> tuple[np.ndarray, int, int, float]:
    """Letterbox img to a black square and resize to size x size.

    Returns (panel, ox, oy, s): a point (x, y) of the original frame is drawn at
    ((x + ox) * s, (y + oy) * s).
    """
    h, w = img.shape[:2]
    side = max(h, w)
    sq = np.zeros((side, side, 3), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    sq[oy:oy + h, ox:ox + w] = img
    return cv2.resize(sq, (size, size), interpolation=interp), ox, oy, size / side


def header(panel: np.ndarray, text: str, fs: float = 0.45) -> None:
    """Black panel header bar with white text."""
    W = panel.shape[1]
    cv2.rectangle(panel, (0, 0), (W, 20), (0, 0, 0), -1)
    put(panel, (5, 15), text, fit(text, W - 10, fs), (255, 255, 255))


def footer(panel: np.ndarray, text: str, fs: float = 0.34) -> None:
    """Black panel footer bar (step i/n | frame | source)."""
    H, W = panel.shape[:2]
    cv2.rectangle(panel, (0, H - 16), (W, H), (0, 0, 0), -1)
    put(panel, (4, H - 4), text, fit(text, W - 8, fs), (200, 200, 200))


def video_path(out: str | Path) -> Path:
    """Delivery path for a video: always inside a `videos/` folder (no PNGs; skill `pose-video`).

    `<dir>/name.mp4` -> `<dir>/videos/name.mp4`; a path already inside `videos/` is kept.
    The sibling `<dir>/csv/` is where a renderer writes its tables.
    """
    out = Path(out).resolve()
    if out.parent.name != "videos":
        out = out.parent / "videos" / out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def csv_dir(video: Path) -> Path:
    d = video.parent.parent / "csv"
    d.mkdir(exist_ok=True)
    return d


def ffmpeg() -> str:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return "ffmpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()


class Writer:
    """mp4v writer to a temporary file, re-encoded to h264/yuv420p (plays in VS Code, browsers)."""

    def __init__(self, out: Path, fps: float, size: tuple[int, int], crf: int = 23):
        self.out, self.crf = Path(out), crf
        self.raw = self.out.with_name(self.out.stem + "_raw.mp4")
        self.vw  = cv2.VideoWriter(str(self.raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        self.vw.write(frame)

    def close(self) -> Path:
        self.vw.release()
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(self.raw),
                        "-c:v", "libx264", "-crf", str(self.crf), "-pix_fmt", "yuv420p",
                        str(self.out)], check=True)
        self.raw.unlink()
        return self.out
