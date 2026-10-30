"""Run an ensemble of Lightning Pose models on a label CSV's images and combine them.

Shared by `scripts/preprocess/ensemble_pseudo_labels.py` and
`scripts/preprocess/active_labeling.py`. Nothing
here imports `lightning_pose`: `predict` takes an already-loaded `lightning_pose.api.Model`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.labels import read_labels_csv


def predict(model, csv_path: Path, data_dir: Path, tmp: Path, tag: str) -> pd.DataFrame:
    """Run `model` on the images listed in `csv_path`. Only the image paths matter, so a
    label-free CSV with the model's own keypoint columns is used as input (the target's
    columns needn't match the model's). Its basename is unique per target/csv, so it can't
    collide with the model's self-eval predictions (see transfer_pseudo_labels.py).

    Args:
        model: a loaded `lightning_pose.api.Model`
        csv_path: label CSV whose index lists the images to predict on
        data_dir: directory the CSV's image paths are relative to
        tmp: scratch directory for the label-free input CSV
        tag: basename of that input CSV; must differ from any CSV the model has predicted on

    Returns:
        Predictions (x, y, likelihood per keypoint), indexed by image path
    """
    index = read_labels_csv(csv_path).index
    cols = pd.MultiIndex.from_tuples(
        [("scorer", kp, xy) for kp in model.cfg.data.keypoint_names for xy in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )
    temp_csv = tmp / tag
    pd.DataFrame(np.nan, index=index, columns=cols).to_csv(temp_csv)
    result = model.predict_on_label_csv(
        csv_file=temp_csv, data_dir=data_dir, compute_metrics=False, add_train_val_test_set=False,
    )
    return result.predictions


def ensemble(
    orig: pd.DataFrame,
    preds: list[pd.DataFrame],
    keypoints: list[tuple[str, str]],
    threshold: float,
    agg: str = "mean",
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Return (new labels, {target_kp: n_surviving_members per row}).

    A prediction survives if its likelihood is >= threshold. The new (x, y) is the mean
    over survivors (the median if `agg="median"`), NaN if none survive. Only the target
    columns in `keypoints` (`(pred_name, target_name)` pairs) change.
    """
    new = orig.copy()
    scorer = orig.columns.get_level_values(0)[0]
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    reduce = np.nanmean if agg == "mean" else np.nanmedian
    n_surv = {}
    for pred_kp, kp in keypoints:
        xs, ys, oks = [], [], []
        for p in preds:
            p = p.loc[orig.index]
            scorer_p = p.columns.get_level_values(0)[0]
            ok = (p[(scorer_p, pred_kp, "likelihood")] >= threshold).to_numpy()
            xs.append(np.where(ok, p[(scorer_p, pred_kp, "x")].to_numpy(), np.nan))
            ys.append(np.where(ok, p[(scorer_p, pred_kp, "y")].to_numpy(), np.nan))
            oks.append(ok)
        n = np.sum(oks, axis=0)
        with np.errstate(all="ignore"):  # all-NaN slices -> NaN (the blank case)
            new[(scorer, kp, "x")] = reduce(xs, axis=0)
            new[(scorer, kp, "y")] = reduce(ys, axis=0)
        n_surv[kp] = n
    return new, n_surv
