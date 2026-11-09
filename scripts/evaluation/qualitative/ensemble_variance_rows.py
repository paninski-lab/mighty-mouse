"""
One video, one ROW per ensemble-variance definition, same frames and same centre in every row.

Centre in every row = median of ALL models (per keypoint, frame, coordinate). A keypoint is drawn
when at least one model has confidence >= --conf (same keypoints in every row); hollow ring when
the models' mean confidence < --conf. Only the +-1 SD cross differs between rows (all at true
length, x --bar_scale):
  row 1  ensemble var          var_x = Var_m(x_m)
  row 2  inv-conf-weighted     var_x = Var_m(x_m) / mean_m(c_m)
  row 3  mixture               var_x = mean_m(1 / c_m) + Var_m(x_m)
                               (components with variance 1/c, read as px^2)
(c floored at 1e-5; same for y). Columns = camera views. A strip beside each panel lists the SD
x / y per keypoint. Writes videos/<out>.mp4 and csv/<out>_per_frame_keypoint.csv (+ _summary.csv:
per keypoint and variance type, median SD, and on labelled cells the median error of the ensemble
median and Spearman corr(error, SD)). No PNGs.

    python scripts/evaluation/qualitative/ensemble_variance_rows.py --models <run> <run> <run> \
        --title "..." \
        --out <delivery dir>/<name>.mp4 [--exclude regexes] [--no_gt]
        [--modes mixture] [--include ear_] [--conf 0]
--modes keeps only the named rows (default all three); --include keeps only keypoints matching
any regex; --conf 0 disables confidence masking (every included keypoint is drawn on every frame,
solid).
--median_panel adds, under each panel, the same overlays on the frame minus its session's median
frame (per-pixel median of EVERY frame in that session folder, or of --median_frames evenly
spaced ones; gray = unchanged, contrast stretched per frame): shows the moving parts on dim
frames.

Raw-dataset models (e.g. the active pseudo-labeling ensembles, whose predictions
`active_labeling.py score` caches in results/<dataset>/active/round_XX/) pass the prediction CSVs
and the image root directly, and map the raw keypoint names to canonical ones for the house
colours (labels in the strip keep the raw names):

    python scripts/evaluation/qualitative/ensemble_variance_rows.py --preds <csv> <csv> <csv> \
        --image_dir <_raw/ibl> --no_gt --keypoint_map <map.yaml> \
        --subset "pool frames (not trained on)" --out <delivery dir>/<name>.mp4
"""

import argparse
from pathlib import Path

from mighty_mouse.evaluation.qualitative.variance_rows import render_variance_rows
from mighty_mouse.paths import load_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--models",   nargs="+",
                     help="corpus run dirs (reads eval/<target>/predictions.csv)")
    src.add_argument("--preds",    nargs="+",
                     help="prediction CSVs, one per model, image paths relative to --image_dir")
    ap.add_argument("--target",    default="ibl")
    ap.add_argument("--data_dir",  default=None,
                    help="corpus data dir (default: paths.yaml data_dir)")
    ap.add_argument("--image_dir", default=None,
                    help="root the frame paths are relative to (default: --data_dir)")
    ap.add_argument("--gt_csv",    default=None,
                    help="ground-truth CSV (default: <data_dir>/CollectedData_<target>_test.csv)")
    ap.add_argument("--subset",    default="held-out test frames",
                    help="what the frames are, for the banner")
    ap.add_argument("--keypoint_map", default=None,
                    help="YAML {raw name: canonical name}, or a configs/datasets/<name>.yaml "
                         "(its `keypoints:`), used only to pick the house colour of raw "
                         "keypoint names")
    ap.add_argument("--title",     default="")
    ap.add_argument("--conf",      type=float, default=0.7)
    ap.add_argument("--bar_scale", type=float, default=1.0)
    ap.add_argument("--frames",    type=int, default=60)
    ap.add_argument("--panel",     type=int, default=440)
    ap.add_argument("--strip",     type=int, default=250)
    ap.add_argument("--fps",       type=int, default=3)
    ap.add_argument("--exclude",   default="")
    ap.add_argument("--include",   default="", help="comma regexes: keep only matching keypoints")
    ap.add_argument("--modes",     default="",
                    help="comma list of rows to draw: 'ensemble var', "
                         "'inv-conf-weighted', 'mixture' (default all)")
    ap.add_argument("--no_gt",     action="store_true",
                    help="do not draw ground truth (labels still score the csv/ tables)")
    ap.add_argument("--median_panel", action="store_true",
                    help="under each panel, the same overlays on the frame minus its session's "
                         "median frame")
    ap.add_argument("--median_frames", type=int, default=0,
                    help="frames per session for the median "
                         "(default 0 = every frame in the session folder)")
    ap.add_argument("--out",       required=True,
                    help="<delivery dir>/<name>.mp4 (written into videos/)")
    args = ap.parse_args()

    D = (Path(args.data_dir) if args.data_dir
         else (None if args.image_dir else Path(load_paths()["data_dir"])))
    pred_csvs = args.preds or [Path(m) / "eval" / args.target / "predictions.csv"
                               for m in args.models]
    # labels score the csv/ tables even with --no_gt (which only stops drawing them); raw
    # predictions without --gt_csv or --data_dir have none
    gt_csv = (Path(args.gt_csv) if args.gt_csv
              else D / f"CollectedData_{args.target}_test.csv" if D else None)
    render_variance_rows(
        args.out, [Path(f) for f in pred_csvs], Path(args.image_dir) if args.image_dir else D,
        args.target, gt_csv=gt_csv, subset=args.subset, keypoint_map=args.keypoint_map,
        title=args.title, conf=args.conf, bar_scale=args.bar_scale, frames=args.frames,
        panel=args.panel, strip=args.strip, fps=args.fps, exclude=args.exclude,
        include=args.include, modes=args.modes, no_gt=args.no_gt,
        median_panel=args.median_panel, median_frames=args.median_frames,
    )


if __name__ == "__main__":
    main()
