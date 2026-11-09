"""Transfer-keypoint panel video: every dataset view of a corpus version, side by side, on held-out
test frames (or a leave-one-out model's train frames). By default only the keypoints a view's
dataset never supervises are drawn, with tongue tip and right pupil shown in every view.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from mighty_mouse.labels import read_labels_csv
from mighty_mouse.plots.house_style import (
    BANNER_H,
    GROUPS,
    Writer,
    blank_canvas,
    fit,
    footer,
    header,
    put,
    square,
    video_path,
)
from mighty_mouse.plots.house_style import color as group_color
from mighty_mouse.plots.overlay import predicted_point, to_panel
from mighty_mouse.plots.views import sample_per_view

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

WHAT = {"transfer": "ONLY keypoints the dataset never labels, inherited from the others "
                    "| tongue tip and right pupil shown in every view",
        "all":      "ALL keypoints the model outputs (the dataset's own and inherited)",
        "own":      "ONLY the keypoints the dataset itself labels"}
WHERE = {"test":  "held-out test frames",
         "train": "TRAIN frames (zero-shot: never trained on them)",
         "both":  "test and train frames (one panel each)"}


def run_label(model: Path) -> str:
    """Name of a run by the parts that identify it, not just the seed directory."""
    parts = [x for x in model.parts if x not in ("supervised", "tf1")]
    return "/".join(parts[parts.index("results") + 1:]) if "results" in parts else model.name


def panel_keypoints(
    mode: str, all_kps: list[str], supervised: set[str], labelled: set[str],
) -> tuple[list[str], list[str]]:
    """(keypoints to draw, extra keypoints drawn regardless) of one dataset.

    Args:
        mode: `transfer` (only keypoints the dataset never supervises, plus the extras),
            `all` (every output) or `own` (only the dataset's own labeled keypoints)
        all_kps: every keypoint the model outputs
        supervised: the dataset's `trainable` keypoints
        labelled: the dataset's `direct` keypoints
    """
    if mode == "transfer":
        transfer = [k for k in all_kps if k not in supervised]
        return transfer, [k for k in EXTRA if k in all_kps and k not in transfer]
    if mode == "all":
        return all_kps, []
    return [k for k in all_kps if k in labelled], []    # own labels only, not flip partners


def panel_order(keys) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(panels in VIEW_ORDER, panels not in VIEW_ORDER appended in sorted order)."""
    keys = set(keys)
    order = [k for k in VIEW_ORDER if k in keys]
    return order, sorted(keys - set(order))


def header_text(mode: str, dataset: str, view: str, transfer: list[str], extra: list[str],
                labelled: set[str]) -> str:
    """Header bar text of one panel."""
    n_own = len([k for k in transfer if k in labelled])
    if mode == "transfer":
        head = f"{dataset} / {view}  ({len(transfer)} inherited)"
    elif mode == "all":
        head = f"{dataset} / {view}  ({n_own} labelled + {len(transfer) - n_own} other)"
    else:
        head = f"{dataset} / {view}  ({n_own} labelled)"
    if extra:
        head += " + own " + ", ".join(OWN[k] for k in extra)
    return head


def load_panels(
    model: Path, inv: dict, *, datasets: list[str] | None, exclude: Iterable[str], mode: str,
    split: str, frames: int,
) -> dict[tuple[str, str], dict]:
    """The panels to draw: {(dataset, view): frames, predictions and keypoint sets}.

    Args:
        model: run directory with eval/<dataset>/predictions.csv (and zeroshot/ for train frames)
        inv: dataset inventory
        datasets: datasets to include (default: all in the inventory)
        exclude: datasets to leave out
        mode: see `panel_keypoints`
        split: `test`, `train` (a leave-one-out model's zero-shot train predictions) or `both`
        frames: max frames sampled per view
    """
    panels = {}
    keep = (set(datasets) if datasets else set(inv)) - set(exclude)
    splits = ["test", "train"] if split == "both" else [split]
    for dataset, sp in [(d, s) for d in inv if d in keep for s in splits]:
        pred = (model / "eval" / dataset / "predictions.csv" if sp == "test"
                else model / "zeroshot" / f"{dataset}_train_predictions.csv")
        if not pred.exists():
            print(f"  skip {dataset} ({sp}): no {pred.name}")
            continue
        preds = read_labels_csv(pred)
        all_kps = [k for k in dict.fromkeys(preds.columns.get_level_values(1))
                   if k != "set" and not k.startswith("Unnamed")]
        labelled = set(inv[dataset]["direct"])
        transfer, extra = panel_keypoints(mode, all_kps, set(inv[dataset]["trainable"]), labelled)
        own_frames = sorted(fr for fr in preds.index
                            if isinstance(fr, str) and fr.startswith(f"labeled-data/{dataset}/"))
        for v, view_frames in sample_per_view(own_frames, dataset, frames).items():
            panels[(dataset, v if split != "both" else f"{sp} {v}")] = dict(
                frames=view_frames, preds=preds, transfer=transfer, extra=extra,
                labelled=labelled)
    return panels


def render_panel(
    out: str | Path, model: Path, data_dir: Path, *, conf: float = 0.7, mode: str = "transfer",
    datasets: list[str] | None = None, exclude: tuple[str, ...] = ("cheese-2d",), frames: int = 60,
    panel: int = 300, cols: int = 5, fps: int = 3, split: str = "test",
) -> Path:
    """Write the panel video and return its path (arguments as for `load_panels`; `panel` is
    the panel side in px, `cols` the grid width, `conf` the minimum likelihood to draw)."""
    D, MODEL = Path(data_dir), Path(model)
    OUT = video_path(out)
    inv = json.load(open(D / "dataset_inventory.json"))["datasets"]
    label = run_label(MODEL)
    panels = load_panels(MODEL, inv, datasets=datasets, exclude=exclude, mode=mode, split=split,
                         frames=frames)
    order, missing = panel_order(panels)
    if missing:
        print(f"  note: views not in VIEW_ORDER, appended: {missing}")
        order += missing

    PANEL, COLS = panel, cols
    ROWS = int(np.ceil(len(order) / COLS))
    n  = max(len(panels[k]["frames"]) for k in order)
    bh = BANNER_H
    gw, gh = PANEL * COLS, PANEL * ROWS
    vw = Writer(OUT, fps, (gw, gh + bh), crf=24)
    drawn = {k: 0 for k in order}
    for i in range(n):
        canvas = blank_canvas(gw, gh)
        x = 10
        for name, col, _ in GROUPS:
            cv2.circle(canvas, (x + 6, 20), 6, col, -1, cv2.LINE_AA)
            x = put(canvas, (x + 16, 25), name) + 16
        cv2.circle(canvas, (x + 6, 20), 6, group_color("pupil_"), 2, cv2.LINE_AA)
        put(canvas, (x + 16, 25), "right pupil (ring)")
        line = f"{label} on {WHERE[split]} | {WHAT[mode]} | conf >= {conf}"
        # shrink the banner until it fits the canvas rather than running off the right edge
        put(canvas, (10, 50), line, fit(line, gw - 20, 0.5, floor=0.3), (200, 200, 200))
        for idx, key in enumerate(order):
            dataset, view = key
            r, c0 = divmod(idx, COLS)
            d = panels[key]
            fr = d["frames"][i % len(d["frames"])]
            img = cv2.imread(str(D / fr))
            if img is None:
                img = np.zeros((PANEL, PANEL, 3), np.uint8)
            tile, ox, oy, sc = square(img, PANEL, interp=cv2.INTER_CUBIC)
            for k in d["transfer"] + d["extra"]:
                point = predicted_point(d["preds"], fr, k, conf)
                if point is None:
                    continue
                pt = to_panel(*point, ox, oy, sc)
                if k in RING:
                    cv2.circle(tile, pt, 5, group_color(k), 2, cv2.LINE_AA)
                else:
                    cv2.circle(tile, pt, 4, group_color(k), -1, cv2.LINE_AA)
                drawn[key] += 1
            header(tile, header_text(mode, dataset, view, d["transfer"], d["extra"],
                                     d["labelled"]), 0.36)
            step = i % len(d["frames"]) + 1
            footer(tile, f"{step}/{len(d['frames'])} {Path(fr).name}", 0.3)
            canvas[bh + r * PANEL:bh + (r + 1) * PANEL, c0 * PANEL:(c0 + 1) * PANEL] = tile
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} steps at {fps} fps, {len(order)} views)")
    for key in order:
        d = panels[key]
        print(f"  {key[0] + '/' + key[1]:22s} frames={len(d['frames']):3d} "
              f"drawn-set={len(d['transfer']):2d} extra={d['extra']} markers={drawn[key]:5d}")
    return OUT
