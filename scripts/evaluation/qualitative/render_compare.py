"""
Before / after comparison video on one dataset's held-out (OOD) test frames.

Grid: one column per model (e.g. trunk before fine-tuning | anchored LoRA N=50 | anchored LoRA all
frames), one row per camera view (ibl: left, right-flipped; kaufman: cam1, cam2). With
--rows_models, one ROW per model and one column per view. Same frame in every panel of a step.
Draws only the channels the models were trained on (union of `trainable` of --train datasets;
--kps overrides) at confidence >= --conf, standing group palette; the dataset's own ground truth
as x markers of the keypoint's colour unless --no_gt. Each panel header gives the model and how
many keypoints it shows; the footer the frame id. Writes <out dir>/videos/<name>.mp4.

    python scripts/evaluation/qualitative/render_compare.py --models "before=<run>" "N=50=<run>" \
        --train facemap,ibl,... --target ibl --title "..." \
        --out <delivery dir>/<name>_nogt.mp4 --no_gt
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mighty_mouse.paths import load_paths
from mighty_mouse.plots.house_style import (
    BANNER_H,
    Writer,
    color,
    fit,
    footer,
    header,
    legend_row,
    put,
    square,
    video_path,
)
from mighty_mouse.plots.views import view_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models",      required=True, nargs="+",
                    help="label=run_dir (run_dir has eval/<target>/)")
    ap.add_argument("--train",       required=True,
                    help="comma list of datasets whose channels to draw")
    ap.add_argument("--target",      default="ibl")
    ap.add_argument("--data_dir",    default=None,
                    help="corpus data dir (default: paths.yaml data_dir)")
    ap.add_argument("--title",       default="")
    ap.add_argument("--conf",        type=float, default=0.7)
    ap.add_argument("--frames",      type=int, default=60)
    ap.add_argument("--panel",       type=int, default=420)
    ap.add_argument("--fps",         type=int, default=3)
    ap.add_argument("--out",         required=True,
                    help="<delivery dir>/<name>.mp4 (written into videos/)")
    ap.add_argument("--kps",         default=None,
                    help="comma list: draw only these keypoints (default: all trained)")
    ap.add_argument("--rows_models", action="store_true",
                    help="one ROW per model, one column per camera view")
    ap.add_argument("--no_gt",       action="store_true",
                    help="do not draw the ground-truth x markers")
    args = ap.parse_args()

    D   = Path(args.data_dir or load_paths()["data_dir"])
    inv = json.load(open(D / "dataset_inventory.json"))["datasets"]
    trained = set().union(*[set(inv[d]["trainable"]) for d in args.train.split(",")])
    models = []
    for m in args.models:
        lab, run = m.rsplit("=", 1)
        pr = pd.read_csv(Path(run) / "eval" / args.target / "predictions.csv",
                         header=[0, 1, 2], index_col=0)
        models.append((lab, pr, pr.columns[0][0]))
    gt = pd.read_csv(D / f"CollectedData_{args.target}_test.csv", header=[0, 1, 2], index_col=0)
    sg = gt.columns[0][0]
    kps = [k for k in dict.fromkeys(models[0][1].columns.get_level_values(1)) if k in trained]
    if args.kps:
        kps = args.kps.split(",")                   # every model draws these, trained or not
    if args.target in inv:
        own = sorted(inv[args.target]["direct"])
    else:                                           # derived set outside the corpus (ibl-digits)
        vis = gt.xs("visible", axis=1, level=2).apply(pd.to_numeric, errors="coerce") == 2
        own = sorted(vis.columns[vis.any()].get_level_values(1))
    if args.kps:
        own = [k for k in own if k in kps]

    views = {}
    for f in sorted(models[0][1].index):
        views.setdefault(view_of(args.target, f.split("/")[2]), []).append(f)
    views = {v: [fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(args.frames, len(fs)))]
             for v, fs in sorted(views.items())}
    n = max(len(fs) for fs in views.values())
    P, bh = args.panel, BANNER_H
    horiz = len(models) == 1                        # one model: cameras side by side
    if args.rows_models:
        W, H = P * len(views), P * len(models)
    else:
        W, H = (P * len(views), P) if horiz else (P * len(models), P * len(views))
    OUT = video_path(args.out)
    vw  = Writer(OUT, args.fps, (W, H + bh))
    for i in range(n):
        canvas = np.full((H + bh, W, 3), 14, np.uint8)
        legend_row(canvas)
        line = (f"{args.title} | {args.target} held-out test frames | conf >= {args.conf} | "
                + ("no ground truth drawn" if args.no_gt else f"x = {args.target} ground truth")
                + ("" if horiz else (" | same frame in every row" if args.rows_models
                                     else " | same frame in every column")))
        put(canvas, (8, 48), line, fit(line, W - 16), (205, 205, 205))
        for r, (v, frames) in enumerate(views.items()):
            if i >= len(frames):                    # this view has fewer frames: leave it dark
                continue
            f = frames[i]
            base, ox, oy, s = square(cv2.imread(str(D / f)), P)
            for j, (lab, pr, sp) in enumerate(models):
                p = base.copy()
                for k in ([] if args.no_gt else own):     # ground truth, x of the keypoint colour
                    if pd.to_numeric(pd.Series([gt.loc[f, (sg, k, "visible")]]),
                                     errors="coerce").iloc[0] == 2:
                        q = (int((float(gt.loc[f, (sg, k, "x")]) + ox) * s),
                             int((float(gt.loc[f, (sg, k, "y")]) + oy) * s))
                        cv2.drawMarker(p, q, color(k), cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)
                nd = 0
                for k in kps:
                    c = float(pr.loc[f, (sp, k, "likelihood")])
                    if not np.isfinite(c) or c < args.conf:
                        continue
                    q = (int((float(pr.loc[f, (sp, k, "x")]) + ox) * s),
                         int((float(pr.loc[f, (sp, k, "y")]) + oy) * s))
                    cv2.circle(p, q, 5, color(k), -1, cv2.LINE_AA)
                    nd += 1
                header(p, f"{lab} | {v} | {nd} kps")
                footer(p, f"{i % len(frames) + 1}/{len(frames)} "
                          f"{f.split('/')[2][:24]}/{Path(f).name}")
                cv2.line(p, (P - 1, 0), (P - 1, P), (60, 60, 60), 1)
                rr, cc = (j, r) if args.rows_models else ((0, r) if horiz else (r, j))
                canvas[bh + rr * P: bh + (rr + 1) * P, cc * P:(cc + 1) * P] = p
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} steps, views {list(views)}, {len(models)} models, "
          f"{len(kps)} channels)")


if __name__ == "__main__":
    main()
