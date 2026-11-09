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
from pathlib import Path

from mighty_mouse.evaluation.qualitative.compare import render_compare
from mighty_mouse.paths import load_paths


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

    models = []
    for m in args.models:
        label, run = m.rsplit("=", 1)
        models.append((label, Path(run)))
    render_compare(
        args.out, models, args.train.split(","), args.target,
        Path(args.data_dir or load_paths()["data_dir"]), title=args.title, conf=args.conf,
        frames=args.frames, panel=args.panel, fps=args.fps,
        kps=args.kps.split(",") if args.kps else None, rows_models=args.rows_models,
        no_gt=args.no_gt,
    )


if __name__ == "__main__":
    main()
