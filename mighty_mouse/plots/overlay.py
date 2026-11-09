"""Overlay helpers shared by the qualitative renderers: frame to panel coordinates, and the
ground-truth and predicted points of one frame."""

import cv2
import numpy as np
import pandas as pd


def to_panel(x: float, y: float, ox: int, oy: int, s: float) -> tuple[int, int]:
    """Panel pixel of a point of the original frame; (ox, oy, s) as from `house_style.square`."""
    return int((x + ox) * s), int((y + oy) * s)


def frame_value(df: pd.DataFrame, frame: str, kp: str, coord: str) -> float:
    """One cell of a label / prediction frame; NaN if the frame or keypoint is missing."""
    try:
        return float(df.loc[frame, (df.columns[0][0], kp, coord)])
    except (KeyError, ValueError, TypeError):
        return np.nan


def predicted_point(
    df: pd.DataFrame, frame: str, kp: str, conf: float,
) -> tuple[float, float] | None:
    """(x, y) of a predicted keypoint, or None if missing or its likelihood is below conf."""
    x, y, c = (frame_value(df, frame, kp, coord) for coord in ("x", "y", "likelihood"))
    if not np.isfinite([x, y, c]).all() or c < conf:
        return None
    return x, y


def labeled_point(gt: pd.DataFrame, frame: str, kp: str) -> tuple[float, float] | None:
    """(x, y) of a ground-truth keypoint, or None unless it is labeled (visible == 2)."""
    scorer = gt.columns[0][0]
    visible = pd.to_numeric(pd.Series([gt.loc[frame, (scorer, kp, "visible")]]), errors="coerce")
    if visible.iloc[0] != 2:
        return None
    return float(gt.loc[frame, (scorer, kp, "x")]), float(gt.loc[frame, (scorer, kp, "y")])


def draw_ground_truth(panel: np.ndarray, gt: pd.DataFrame, frame: str, kps: list[str],
                      transform: tuple[int, int, float], color_of) -> None:
    """Ground truth of the labeled `kps` as x markers of the keypoint's colour.

    Args:
        panel: image to draw on
        gt: converted labels with (scorer, keypoint, x|y|visible) columns
        frame: image path in `gt`'s index
        kps: keypoints to draw
        transform: (ox, oy, s) of `house_style.square`
        color_of: keypoint name -> BGR colour
    """
    for k in kps:
        point = labeled_point(gt, frame, k)
        if point is not None:
            cv2.drawMarker(panel, to_panel(*point, *transform), color_of(k),
                           cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)
