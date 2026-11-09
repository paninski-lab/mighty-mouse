"""Evaluate a trained Lightning Pose run on every dataset's test set, and tidy its directory.

`evaluate_run` writes <run_dir>/eval/<dataset>/{predictions.csv, pixel_error.csv,
comparison_metrics.json}; `clean_run_dir` then removes the scratch files and checkpoints that
are not needed after evaluation. Pixel errors count labeled cells only (see comparison.py).
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mighty_mouse.datasets import ALL_DATASETS
from mighty_mouse.evaluation.comparison import comparison_summary, visible_errors
from mighty_mouse.labels import read_labels_csv


def image_diagonals(index: pd.Index, data_dir: Path) -> np.ndarray:
    """Pixel diagonal of each image in `index` (paths relative to `data_dir`)."""
    diagonals = []
    for name in index:
        with Image.open(Path(data_dir) / name) as image:
            diagonals.append(float(np.linalg.norm(image.size)))
    return np.asarray(diagonals)


def score_predictions(
    labels_df: pd.DataFrame, preds_df: pd.DataFrame, diagonals: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Score predictions against labels.

    Args:
        labels_df: converted test labels
        preds_df: model predictions on those frames
        diagonals: image diagonal in pixels for each row of `labels_df`

    Returns:
        (predictions in the labels' row order, per-frame per-keypoint pixel errors,
        pooled summary from `comparison_summary`)
    """
    error_df = visible_errors(labels_df, preds_df)
    summary = comparison_summary(error_df, np.asarray(diagonals))
    return preds_df.loc[labels_df.index], error_df, summary


def evaluate_run(run_dir: Path, data_dir: Path, datasets: list[str] | None = None) -> None:
    """Evaluate a trained run against each per-dataset test CSV in `data_dir`.

    Args:
        run_dir: trained model directory
        data_dir: corpus directory with CollectedData_<dataset>_test.csv and the labeled frames
        datasets: test sets to evaluate (default: every dataset in ALL_DATASETS)
    """
    from lightning_pose.api import Model

    run_dir, data_dir = Path(run_dir), Path(data_dir)

    print("  Loading model...")
    model = Model.from_dir(run_dir)

    for eval_name in datasets or ALL_DATASETS:
        test_csv = data_dir / f"CollectedData_{eval_name}_test.csv"

        if not test_csv.exists():
            print(f"  WARNING: {test_csv} not found — skipping {eval_name} eval")
            continue

        print(f"  Predicting on {eval_name} test set ({test_csv.name})...")
        result = model.predict_on_label_csv(
            csv_file=test_csv,
            data_dir=data_dir,
            compute_metrics=False,
        )
        labels_df = read_labels_csv(test_csv)
        preds_df, error_df, summary = score_predictions(
            labels_df, result.predictions, image_diagonals(labels_df.index, data_dir),
        )

        save_dir = run_dir / "eval" / eval_name
        save_dir.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(save_dir / "predictions.csv")
        error_df.to_csv(save_dir / "pixel_error.csv")
        (save_dir / "comparison_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

        mean_err = summary["mean_px"]
        mean_txt = "n/a (no labeled cell)" if mean_err is None else f"{mean_err:.2f} px"
        print(f"    Mean pixel error: {mean_txt}  →  {save_dir}")


def clean_run_dir(run_dir: Path, keep_checkpoints: bool = False) -> None:
    """Remove the scratch prediction files litpose leaves in `run_dir` and, unless
    `keep_checkpoints`, its model checkpoints (*.ckpt). Evaluation results under
    run_dir/eval/<dataset>/ are what is kept long-term; checkpoints are needed for any model
    used after evaluation (zero-shot predictions, fine-tuning).
    """
    run_dir = Path(run_dir)
    for fname in ("predictions.csv", "predictions_pixel_error.csv"):
        p = run_dir / fname
        if p.exists():
            p.unlink()
    image_preds = run_dir / "image_preds"
    if image_preds.exists():
        shutil.rmtree(image_preds)

    if keep_checkpoints:
        n_kept = len(list(run_dir.rglob("*.ckpt")))
        print(f"  Kept {n_kept} checkpoint file(s) (--keep_checkpoints)")
    else:
        n_deleted = 0
        for ckpt in run_dir.rglob("*.ckpt"):
            ckpt.unlink()
            n_deleted += 1
        print(f"  Deleted {n_deleted} checkpoint file(s)")
