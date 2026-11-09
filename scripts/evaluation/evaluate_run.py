#!/usr/bin/env python3
"""
Evaluate a trained model against every per-dataset test CSV, then clean up its directory.

Writes <run>/eval/<dataset>/{predictions.csv, pixel_error.csv, comparison_metrics.json}, then
removes litpose's scratch predictions and, unless --keep_checkpoints, the model checkpoints.
Chained after `litpose train` in Lightning jobs (a remote job has no "come back to this process
after training" step); also useful standalone for finishing sweep jobs that trained successfully
but stalled before their chained eval step completed.

Usage:
    python scripts/evaluation/evaluate_run.py --run <run dir> [--keep_checkpoints]
        [--datasets ibl,facemap] [--data_dir <corpus dir>]
"""

import argparse
from pathlib import Path

from mighty_mouse.evaluation.run import clean_run_dir, evaluate_run
from mighty_mouse.paths import load_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", required=True, type=Path,
        help="Model output_dir passed to `litpose train`",
    )
    parser.add_argument(
        "--data_dir", default=None, type=Path,
        help="Corpus directory with the test CSVs (default: paths.yaml data_dir).",
    )
    parser.add_argument(
        "--keep_checkpoints", action="store_true",
        help="Keep *.ckpt files after evaluation (needed for zero-shot predictions and "
             "fine-tuning).",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated test sets to evaluate (default: every dataset in ALL_DATASETS).",
    )
    args = parser.parse_args()
    datasets = [d for d in args.datasets.split(",") if d] if args.datasets else None
    data_dir = args.data_dir or Path(load_paths()["data_dir"])
    print(f"run: {args.run}")
    evaluate_run(args.run, data_dir, datasets)
    clean_run_dir(args.run, args.keep_checkpoints)


if __name__ == "__main__":
    main()
