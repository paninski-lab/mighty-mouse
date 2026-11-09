"""Before / after comparison video on one dataset's held-out test frames: one panel per model and
camera view, the same frame in every panel of a step. Draws the channels the models were trained
on (union of the `trainable` keypoints of the training datasets) at confidence >= conf, and the
dataset's own ground truth as x markers.
"""

import json
from pathlib import Path

import cv2
import pandas as pd

from mighty_mouse.labels import read_labels_csv
from mighty_mouse.plots.house_style import (
    BANNER_H,
    Writer,
    blank_canvas,
    color,
    fit,
    footer,
    header,
    legend_row,
    put,
    square,
    video_path,
)
from mighty_mouse.plots.overlay import draw_ground_truth, predicted_point, to_panel
from mighty_mouse.plots.views import sample_per_view


def compare_keypoints(
    pred: pd.DataFrame, inv: dict, train: list[str], target: str, gt: pd.DataFrame,
    kps: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Keypoints to draw as predictions and keypoints whose ground truth to draw.

    Args:
        pred: one model's predictions (fixes the keypoint order)
        inv: dataset inventory, `{dataset: {"trainable": [...], "direct": [...]}}`
        train: datasets the models were trained on; their `trainable` keypoints are drawn
        target: dataset whose test frames are shown
        gt: the target's test labels (used when the target is not in the inventory)
        kps: draw exactly these keypoints instead (trained or not)
    """
    trained = set().union(*[set(inv[d]["trainable"]) for d in train])
    drawn = [k for k in dict.fromkeys(pred.columns.get_level_values(1)) if k in trained]
    if kps:
        drawn = list(kps)
    if target in inv:
        own = sorted(inv[target]["direct"])
    else:                                           # derived set outside the corpus (ibl-digits)
        vis = gt.xs("visible", axis=1, level=2).apply(pd.to_numeric, errors="coerce") == 2
        own = sorted(vis.columns[vis.any()].get_level_values(1))
    if kps:
        own = [k for k in own if k in drawn]
    return drawn, own


def grid_size(n_views: int, n_models: int, panel: int, rows_models: bool) -> tuple[int, int]:
    """(width, height) in px of the panel grid, below the banner. One model puts the cameras
    side by side; several put models in columns and cameras in rows (rows_models swaps them)."""
    if rows_models:
        return panel * n_views, panel * n_models
    if n_models == 1:
        return panel * n_views, panel
    return panel * n_models, panel * n_views


def grid_cell(view_i: int, model_j: int, n_models: int, rows_models: bool) -> tuple[int, int]:
    """(row, column) of a view's panel for a model in the grid of `grid_size`."""
    if rows_models:
        return model_j, view_i
    return (0, view_i) if n_models == 1 else (view_i, model_j)


def banner_line(title: str, target: str, conf: float, no_gt: bool, n_models: int,
                rows_models: bool) -> str:
    """Banner text above the grid."""
    return (f"{title} | {target} held-out test frames | conf >= {conf} | "
            + ("no ground truth drawn" if no_gt else f"x = {target} ground truth")
            + ("" if n_models == 1 else (" | same frame in every row" if rows_models
                                         else " | same frame in every column")))


def render_compare(
    out: str | Path, models: list[tuple[str, Path]], train: list[str], target: str,
    data_dir: Path, *, title: str = "", conf: float = 0.7, frames: int = 60, panel: int = 420,
    fps: int = 3, kps: list[str] | None = None, rows_models: bool = False, no_gt: bool = False,
) -> Path:
    """Write the comparison video and return its path.

    Args:
        out: `<delivery dir>/<name>.mp4`; written into the delivery's videos/ folder
        models: (label, run directory with eval/<target>/predictions.csv) per model
        train: datasets the models were trained on
        target: dataset whose held-out test frames are shown
        data_dir: corpus directory with the inventory, test labels and frames
        title: banner prefix
        conf: minimum likelihood to draw a keypoint
        frames: frames sampled per camera view
        panel: panel side in px
        fps: video frame rate
        kps: draw exactly these keypoints (default: those the models were trained on)
        rows_models: one row per model and one column per view, instead of the reverse
        no_gt: do not draw the ground-truth markers
    """
    D = Path(data_dir)
    inv = json.load(open(D / "dataset_inventory.json"))["datasets"]
    preds = [(label, read_labels_csv(Path(run) / "eval" / target / "predictions.csv"))
             for label, run in models]
    gt = read_labels_csv(D / f"CollectedData_{target}_test.csv")
    kps, own = compare_keypoints(preds[0][1], inv, train, target, gt, kps)

    views = sample_per_view(sorted(preds[0][1].index), target, frames)
    n = max(len(fs) for fs in views.values())
    P, bh = panel, BANNER_H
    W, H = grid_size(len(views), len(preds), P, rows_models)
    OUT = video_path(out)
    vw = Writer(OUT, fps, (W, H + bh))
    for i in range(n):
        canvas = blank_canvas(W, H)
        legend_row(canvas)
        line = banner_line(title, target, conf, no_gt, len(preds), rows_models)
        put(canvas, (8, 48), line, fit(line, W - 16), (205, 205, 205))
        for r, (v, view_frames) in enumerate(views.items()):
            if i >= len(view_frames):               # this view has fewer frames: leave it dark
                continue
            f = view_frames[i]
            base, ox, oy, s = square(cv2.imread(str(D / f)), P)
            for j, (label, pr) in enumerate(preds):
                p = base.copy()
                # ground truth, x of the keypoint colour
                draw_ground_truth(p, gt, f, [] if no_gt else own, (ox, oy, s), color)
                nd = 0
                for k in kps:
                    point = predicted_point(pr, f, k, conf)
                    if point is None:
                        continue
                    cv2.circle(p, to_panel(*point, ox, oy, s), 5, color(k), -1, cv2.LINE_AA)
                    nd += 1
                header(p, f"{label} | {v} | {nd} kps")
                footer(p, f"{i % len(view_frames) + 1}/{len(view_frames)} "
                          f"{f.split('/')[2][:24]}/{Path(f).name}")
                cv2.line(p, (P - 1, 0), (P - 1, P), (60, 60, 60), 1)
                rr, cc = grid_cell(r, j, len(preds), rows_models)
                canvas[bh + rr * P: bh + (rr + 1) * P, cc * P:(cc + 1) * P] = p
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} steps, views {list(views)}, {len(preds)} models, "
          f"{len(kps)} channels)")
    return OUT
