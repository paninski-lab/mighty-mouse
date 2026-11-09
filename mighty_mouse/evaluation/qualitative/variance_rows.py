"""Ensemble-variance video: one row per variance definition, the same frames and the same centre
(median of all models) in every row; only the +-1 SD cross differs between rows. Columns are camera
views, with a strip beside each panel listing the SD per keypoint. Statistics: see
`mighty_mouse.evaluation.ensemble_stats`. Also writes the per-frame and summary tables to csv/.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

from mighty_mouse.evaluation.ensemble_stats import (
    EnsembleStats,
    ensemble_stats,
    labelled_keypoints,
    per_frame_keypoint_table,
    select_modes,
    usable_keypoints,
    variance_summary,
)
from mighty_mouse.labels import read_labels_csv
from mighty_mouse.plots.house_style import (
    BANNER_H,
    Writer,
    blank_canvas,
    color,
    csv_dir,
    fit,
    footer,
    header,
    legend_row,
    put,
    square,
    video_path,
)
from mighty_mouse.plots.median_frame import median_subtracted, session_median
from mighty_mouse.plots.overlay import draw_ground_truth, to_panel
from mighty_mouse.plots.views import sample_per_view


def load_keypoint_map(path: str | Path | None) -> dict[str, str]:
    """{raw keypoint name: canonical name} from a YAML map or a configs/datasets/<name>.yaml
    (its `keypoints:`); empty without a path. Only used to pick house colours."""
    if not path:
        return {}
    kmap = yaml.safe_load(Path(path).read_text())
    return kmap.get("keypoints", kmap)


def canvas_size(n_views: int, panel: int, strip: int, n_modes: int,
                median_panel: bool) -> tuple[int, int]:
    """(width, height) in px of the grid below the banner: a panel and a strip per view, a row
    per mode (two with the median-subtracted panel)."""
    return (panel + strip) * n_views, panel * n_modes * (2 if median_panel else 1)


def banner_line(
    title: str, target: str, subset: str, n_models: int, conf: float, bar_scale: float,
    n_modes: int, with_gt: bool, median_panel: bool,
) -> str:
    """Banner text above the grid."""
    shown_rule = ("no confidence masking: every keypoint drawn on every frame" if conf <= 0 else
                  f"shown if any model conf >= {conf}; ring = mean conf < {conf}")
    cross = "rows differ ONLY in the +-1 SD cross" if n_modes > 1 else "cross = +-1 SD"
    scale = "" if bar_scale == 1 else f" x{bar_scale:g}"
    return (f"{title} | {target} {subset} | dot = median of all {n_models} models; "
            f"{shown_rule} | {cross} (true length{scale}) | "
            + ("x = ground truth" if with_gt else "no ground truth drawn")
            + (" | lower panels: frame - session median "
               "(gray = unchanged, contrast stretched per frame)" if median_panel else ""))


def draw_variance_panel(
    base: np.ndarray, strip_width: int, stats: EnsembleStats, i: int, mode: str,
    kps: list[str], kcolor, transform: tuple[int, int, float], bar_scale: float, conf: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """One mode's panel for frame `i`: the median of each shown keypoint with its +-1 SD cross
    (ring if the mean confidence is below `conf`), and the strip listing the SDs.

    Returns:
        (panel, strip, number of keypoints drawn, mean SD in px)
    """
    P = base.shape[0]
    ox, oy, s = transform
    sx, sy = stats.sd[mode]
    p = base.copy()
    strip = np.full((P, strip_width, 3), 24, np.uint8)
    put(strip, (6, 16), f"{mode} SD (px) x / y", 0.4, (220, 220, 220))
    ln, nd, sds = 34, 0, []
    for j, k in enumerate(kps):
        if not stats.shown[i, j]:
            continue
        q = to_panel(stats.x[i, j], stats.y[i, j], ox, oy, s)
        cc = kcolor(k)
        nd += 1
        ex, ey = sx[i, j] * s * bar_scale, sy[i, j] * s * bar_scale
        th = 2 if stats.mean_conf[i, j] >= conf else 1
        cv2.line(p, (int(q[0] - ex), q[1]), (int(q[0] + ex), q[1]), cc, th, cv2.LINE_AA)
        cv2.line(p, (q[0], int(q[1] - ey)), (q[0], int(q[1] + ey)), cc, th, cv2.LINE_AA)
        cv2.circle(p, q, 4 if th == 2 else 5, cc, -1 if th == 2 else 1, cv2.LINE_AA)
        sds.append(np.hypot(sx[i, j], sy[i, j]))
        if ln < P - 8:
            cv2.circle(strip, (10, ln - 4), 4, cc, -1 if th == 2 else 1, cv2.LINE_AA)
            put(strip, (20, ln),
                f"{k[:16]:16s} {sx[i, j]:.1f}/{sy[i, j]:.1f}  c{stats.mean_conf[i, j]:.2f}",
                0.34, (215, 215, 215))
            ln += 13
    return p, strip, nd, (np.mean(sds) if sds else 0)


def render_variance_rows(
    out: str | Path, pred_csvs: list[Path], img_root: Path | None, target: str, *,
    gt_csv: Path | None = None, subset: str = "held-out test frames",
    keypoint_map: str | Path | None = None, title: str = "", conf: float = 0.7,
    bar_scale: float = 1.0, frames: int = 60, panel: int = 440, strip: int = 250, fps: int = 3,
    exclude: str = "", include: str = "", modes: str = "", no_gt: bool = False,
    median_panel: bool = False, median_frames: int = 0,
) -> Path:
    """Write the video and its csv tables, and return the video path.

    Args:
        out: `<delivery dir>/<name>.mp4`; written into the delivery's videos/ folder
        pred_csvs: one prediction CSV per model; image paths are relative to `img_root`
        img_root: root the frame paths are relative to
        target: dataset name, to tell the camera views apart
        gt_csv: ground-truth CSV; labels score the csv/ tables even with `no_gt`. None = none
        subset: what the frames are, for the banner
        keypoint_map: raw -> canonical keypoint names, for the house colours only
        title: banner prefix
        conf: a keypoint is drawn if any model's likelihood reaches this (0 = always)
        bar_scale: multiplier on the drawn SD cross
        frames: frames sampled per camera view
        panel, strip: panel side and strip width in px
        fps: video frame rate
        exclude, include: comma regexes selecting keypoints
        modes: comma list of variance rows to draw (default all)
        no_gt: do not draw the ground truth
        median_panel: add, under each panel, the overlays on the frame minus the session median
        median_frames: frames per session for that median (0 = every frame in the folder)
    """
    preds = [read_labels_csv(f) for f in pred_csvs]
    kmap = load_keypoint_map(keypoint_map)
    kcolor = lambda k: color(kmap.get(k, k))  # noqa: E731
    kps = usable_keypoints(preds[0], exclude, include)
    idx = preds[0].index
    stats = ensemble_stats(preds, kps, conf)
    mode_rows = select_modes(modes)
    gt = None if gt_csv is None else read_labels_csv(gt_csv).reindex(idx)
    labelled = labelled_keypoints(gt, kps)
    OUT = video_path(out)
    CSV = csv_dir(OUT)
    per = per_frame_keypoint_table(stats, idx, kps, mode_rows, gt)
    per.to_csv(CSV / (OUT.stem + "_per_frame_keypoint.csv"), index=False)
    variance_summary(per, mode_rows).to_csv(CSV / (OUT.stem + "_summary.csv"), index=False)

    position = {f: i for i, f in enumerate(idx)}
    views = {v: [position[f] for f in fs]
             for v, fs in sample_per_view(idx, target, frames).items()}
    n = max(len(v) for v in views.values())
    P, Sw, bh = panel, strip, BANNER_H
    nb = 2 if median_panel else 1   # panels per mode: frame (+ median-subtracted frame)
    W, H = canvas_size(len(views), P, Sw, len(mode_rows), median_panel)
    vw = Writer(OUT, fps, (W, H + bh))
    med_cache = {}
    line = banner_line(title, target, subset, len(preds), conf, bar_scale, len(mode_rows),
                       not (no_gt or gt is None), median_panel)
    for step in range(n):
        canvas = blank_canvas(W, H)
        legend_row(canvas, [kmap.get(k, k) for k in kps], fs=0.42)
        put(canvas, (8, 48), line, fit(line, W - 16), (205, 205, 205))
        for col_i, (v, view_frames) in enumerate(views.items()):
            i = view_frames[step % len(view_frames)]
            f = idx[i]
            img = cv2.imread(str(img_root / f))
            sources = [("", img)]
            if median_panel:
                med, n_med = session_median((img_root / f).parent, median_frames, med_cache)
                sources.append((f"frame - median of {n_med} | ", median_subtracted(img, med)))
            for b_i, (blabel, src) in enumerate(sources):
                base, ox, oy, s = square(src, P)
                if not no_gt:
                    draw_ground_truth(base, gt, f, labelled, (ox, oy, s), kcolor)
                for r_i, (mname, formula) in enumerate(mode_rows):
                    p, strip_img, nd, mean_sd = draw_variance_panel(
                        base, Sw, stats, i, mname, kps, kcolor, (ox, oy, s), bar_scale, conf)
                    header(p, f"{blabel}{mname}: {formula} | {v} | {nd} kps | mean SD "
                              f"{mean_sd:.2f} px")
                    footer(p, f"{step % len(view_frames) + 1}/{len(view_frames)} "
                              f"{Path(f).parent.name[:24]}/{Path(f).name}")
                    x0 = col_i * (P + Sw)
                    y0 = bh + (r_i * nb + b_i) * P
                    canvas[y0:y0 + P, x0:x0 + P] = p
                    canvas[y0:y0 + P, x0 + P:x0 + P + Sw] = strip_img
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} steps)")
    return OUT
