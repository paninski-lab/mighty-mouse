"""
Transfer-keypoint panel video: every dataset view of a corpus version, side by side.

One panel per camera view, held-out test frames, showing ONLY the keypoints that view's dataset
never supervises -- the transfer class -- at confidence >= 0.7. The grid grows with the corpus:
one panel per (dataset, view) in VIEW_ORDER. Colours are per keypoint group, one saturated hue
each, none white or grey; digit tips are split by side (right green, left red).

Transfer keypoint = not in the dataset's `trainable` set (its own labels plus the lateral partners
horizontal flips supervise). tongue_tip and pupil_center_right are drawn in every view on request;
where the dataset already supervises one the panel header says so. pupil_center_right is drawn
as a ring, never a filled dot.

    python scripts/evaluation/qualitative/render_panel.py --model <run dir with eval/> \
        --out <delivery dir>/<name>.mp4 \
        [--mode transfer|all|own] [--conf 0.7] [--frames 60]

Writes <out dir>/videos/<name>.mp4.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mighty_mouse.paths import load_paths
from mighty_mouse.plots.house_style import BANNER_H, GROUPS, Writer, fit, video_path
from mighty_mouse.plots.house_style import color as group_color
from mighty_mouse.plots.views import view_of

EXTRA = ["tongue_tip", "pupil_center_right"]   # always drawn, even where supervised
RING  = {"pupil_center_right"}                 # ring, not a filled dot
OWN   = {"tongue_tip": "tongue tip", "pupil_center_right": "R pupil"}

# (dataset, view) in panel order; views absent from a corpus version are skipped.
VIEW_ORDER = [("ibl", "left"), ("ibl", "right"), ("facemap", "cam0"), ("facemap", "cam1"),
              ("cazettes-side", "side"), ("kondo", "bottom"),
              ("hantman-mv", "front"), ("hantman-mv", "side"),
              ("kaufman", "cam1"), ("kaufman", "cam2"),
              ("cheese-2d", "BC"), ("cheese-2d", "L"), ("cheese-2d", "R"), ("cheese-2d", "TC"),
              ("cheese-2d", "TL"), ("cheese-2d", "TR"),
              ("cheese-3d", "BC"), ("cheese-3d", "L"), ("cheese-3d", "R"), ("cheese-3d", "TC"),
              ("cheese-3d", "TL"), ("cheese-3d", "TR"),
              ("aind-vbn-body", "body"), ("aind-vbn-face", "face"),
              ("cavelli", "left"), ("cavelli", "right"),
              ("tyssowski", "front"), ("tyssowski", "side"),
              ("kaufman-water", "cam1"), ("kaufman-water", "cam2")]


def put(canvas, x, y, text, color=(235, 235, 235), fs=0.5):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)
    return x + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]


def val(df, fr, k, c):
    try:
        return float(df.loc[fr, (df.columns[0][0], k, c)])
    except (KeyError, ValueError, TypeError):
        return np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--model",  required=True, help="run directory containing eval/<dataset>/")
    ap.add_argument("--out",    required=True,
                    help="<delivery dir>/<name>.mp4 (written into videos/)")
    ap.add_argument("--conf",   type=float, default=0.7)
    ap.add_argument("--mode",   default="transfer", choices=["transfer", "all", "own"],
                    help="transfer: only keypoints the dataset never supervises "
                         "(+ tongue tip, R pupil); all: every channel; "
                         "own: only the dataset's own labelled keypoints")
    ap.add_argument("--datasets", default=None, help="comma list to include (default: all)")
    # cheese-2d is left out by default; pass --exclude "" to include every dataset
    ap.add_argument("--exclude",  default="cheese-2d",
                    help="comma list of datasets to leave out (default: cheese-2d; '' for none)")
    ap.add_argument("--data_dir", default=None,
                    help="corpus data dir; pin it when rendering an older version than paths.yaml")
    ap.add_argument("--frames", type=int, default=60, help="max frames sampled per view")
    ap.add_argument("--panel",  type=int, default=300)
    ap.add_argument("--cols",   type=int, default=5)
    ap.add_argument("--fps",    type=int, default=3)
    ap.add_argument("--split",  default="test", choices=["test", "train", "both"],
                    help="test = eval/<ds>/predictions.csv (held-out test frames); "
                         "train = zeroshot/<ds>_train_predictions.csv in the run dir, when a "
                         "leave-one-out model was run on its left-out dataset's TRAIN frames; "
                         "both = one panel per view and split")
    args = ap.parse_args()

    P     = load_paths()
    D     = Path(args.data_dir) if args.data_dir else Path(P["data_dir"])
    MODEL = Path(args.model)
    OUT   = video_path(args.out)
    inv   = json.load(open(D / "dataset_inventory.json"))["datasets"]
    # name the run by the parts that identify it, not just the seed directory
    parts = [x for x in MODEL.parts if x not in ("supervised", "tf1")]
    label = "/".join(parts[parts.index("results") + 1:]) if "results" in parts else MODEL.name

    panels = {}
    keep_ds = set(args.datasets.split(",")) if args.datasets else set(inv)
    keep_ds -= set(x for x in args.exclude.split(",") if x)
    splits = ["test", "train"] if args.split == "both" else [args.split]
    for dataset, split in [(d, s) for d in inv if d in keep_ds for s in splits]:
        pred = (MODEL / "eval" / dataset / "predictions.csv" if split == "test"
                else MODEL / "zeroshot" / f"{dataset}_train_predictions.csv")
        if not pred.exists():
            print(f"  skip {dataset} ({split}): no {pred.name}")
            continue
        preds = pd.read_csv(pred, header=[0, 1, 2], index_col=0)
        if preds.index[0] == preds.index.name:
            preds = preds.iloc[1:]
        supervised = set(inv[dataset]["trainable"])
        all_kps  = [k for k in dict.fromkeys(preds.columns.get_level_values(1))
                    if k != "set" and not k.startswith("Unnamed")]
        labelled = set(inv[dataset]["direct"])
        if args.mode == "transfer":
            transfer = [k for k in all_kps if k not in supervised]
            extra    = [k for k in EXTRA if k in all_kps and k not in transfer]
        elif args.mode == "all":
            transfer, extra = all_kps, []
        else:  # own labels only, not flip-supervised partners
            transfer, extra = [k for k in all_kps if k in labelled], []
        by_view = {}
        for fr in preds.index:
            if isinstance(fr, str) and fr.startswith(f"labeled-data/{dataset}/"):
                v = view_of(dataset, fr.split("/")[2])
                if v:
                    by_view.setdefault(v, []).append(fr)
        for v, frames in by_view.items():
            frames = sorted(frames)
            if len(frames) > args.frames:
                frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, args.frames)]
            panels[(dataset, v if args.split != "both" else f"{split} {v}")] = dict(
                frames=frames, preds=preds, transfer=transfer, extra=extra, labelled=labelled)

    order = [k for k in VIEW_ORDER if k in panels]
    missing = sorted(set(panels) - set(order))
    if missing:
        print(f"  note: views not in VIEW_ORDER, appended: {missing}")
        order += missing

    PANEL, COLS = args.panel, args.cols
    ROWS = int(np.ceil(len(order) / COLS))
    n  = max(len(panels[k]["frames"]) for k in order)
    bh = BANNER_H
    gw, gh = PANEL * COLS, PANEL * ROWS
    vw = Writer(OUT, args.fps, (gw, gh + bh), crf=24)
    drawn = {k: 0 for k in order}
    for i in range(n):
        canvas = np.full((gh + bh, gw, 3), 14, np.uint8)
        x = 10
        for name, col, _ in GROUPS:
            cv2.circle(canvas, (x + 6, 20), 6, col, -1, cv2.LINE_AA)
            x = put(canvas, x + 16, 25, name) + 16
        cv2.circle(canvas, (x + 6, 20), 6, group_color("pupil_"), 2, cv2.LINE_AA)
        put(canvas, x + 16, 25, "right pupil (ring)")
        what = {"transfer": "ONLY keypoints the dataset never labels, inherited from the others "
                            "| tongue tip and right pupil shown in every view",
                "all":      "ALL keypoints the model outputs (the dataset's own and inherited)",
                "own":      "ONLY the keypoints the dataset itself labels"}[args.mode]
        where = {"test":  "held-out test frames",
                 "train": "TRAIN frames (zero-shot: never trained on them)",
                 "both":  "test and train frames (one panel each)"}[args.split]
        line = f"{label} on {where} | {what} | conf >= {args.conf}"
        # shrink the banner until it fits the canvas rather than running off the right edge
        put(canvas, 10, 50, line, (200, 200, 200), fit(line, gw - 20, 0.5, floor=0.3))
        for idx, key in enumerate(order):
            dataset, view = key
            r, c0 = divmod(idx, COLS)
            d = panels[key]
            fr = d["frames"][i % len(d["frames"])]
            img = cv2.imread(str(D / fr))
            if img is None:
                img = np.zeros((PANEL, PANEL, 3), np.uint8)
            h, w = img.shape[:2]
            side = max(h, w)
            sq = np.zeros((side, side, 3), np.uint8)
            oy, ox = (side - h) // 2, (side - w) // 2
            sq[oy:oy + h, ox:ox + w] = img
            sc = PANEL / side
            panel = cv2.resize(sq, (PANEL, PANEL), interpolation=cv2.INTER_CUBIC)
            for k in d["transfer"] + d["extra"]:
                px, py, cf = (val(d["preds"], fr, k, c) for c in ("x", "y", "likelihood"))
                if np.isnan(px) or np.isnan(cf) or cf < args.conf:
                    continue
                pt = (int((px + ox) * sc), int((py + oy) * sc))
                if k in RING:
                    cv2.circle(panel, pt, 5, group_color(k), 2, cv2.LINE_AA)
                else:
                    cv2.circle(panel, pt, 4, group_color(k), -1, cv2.LINE_AA)
                drawn[key] += 1
            cv2.rectangle(panel, (0, 0), (PANEL, 19), (0, 0, 0), -1)
            n_own = len([k for k in d["transfer"] if k in d["labelled"]])
            if args.mode == "transfer":
                head = f"{dataset} / {view}  ({len(d['transfer'])} inherited)"
            elif args.mode == "all":
                head = (f"{dataset} / {view}  ({n_own} labelled + "
                        f"{len(d['transfer']) - n_own} other)")
            else:
                head = f"{dataset} / {view}  ({n_own} labelled)"
            if d["extra"]:
                head += " + own " + ", ".join(OWN[k] for k in d["extra"])
            cv2.putText(panel, head, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1,
                        cv2.LINE_AA)
            step = i % len(d["frames"]) + 1
            cv2.rectangle(panel, (0, PANEL - 15), (PANEL, PANEL), (0, 0, 0), -1)
            cv2.putText(panel, f"{step}/{len(d['frames'])} {Path(fr).name}", (3, PANEL - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)
            canvas[bh + r * PANEL:bh + (r + 1) * PANEL, c0 * PANEL:(c0 + 1) * PANEL] = panel
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} steps at {args.fps} fps, {len(order)} views)")
    for key in order:
        d = panels[key]
        print(f"  {key[0] + '/' + key[1]:22s} frames={len(d['frames']):3d} "
              f"drawn-set={len(d['transfer']):2d} extra={d['extra']} markers={drawn[key]:5d}")


if __name__ == "__main__":
    main()
