"""
Pixel-error scoring for evaluation runs and the comparison tables.

Only labeled cells are scored: visible == 2 in the converted labels (1 = occluded and
0 = unlabeled are never scored). The tongue and lip keypoints are left out of every pixel-error
comparison by default (ALWAYS_EXCLUDED): they are still trained, predicted and drawn in videos.
"""

import numpy as np
import pandas as pd

LIP_KEYPOINTS    = frozenset({"upperlip_left", "upperlip_right", "lowerlip"})
TONGUE_KEYPOINTS = frozenset({"tongue_tip", "tongue_center", "tongue_end_left",
                              "tongue_end_right"})
ALWAYS_EXCLUDED: frozenset[str] = TONGUE_KEYPOINTS | LIP_KEYPOINTS


def visible_errors(
    labels: pd.DataFrame, predictions: pd.DataFrame, exclude: frozenset[str] = ALWAYS_EXCLUDED,
) -> pd.DataFrame:
    """Per-frame, per-keypoint pixel error over the labeled cells.

    Args:
        labels: converted labels with (scorer, keypoint, x|y|visible) columns; without a
            visible column a finite (x, y) counts as labeled
        predictions: (scorer, keypoint, x|y|likelihood) columns for the same frames
        exclude: keypoints never scored (their column is all NaN)

    Returns:
        DataFrame indexed like labels, one column per keypoint, NaN where not scored

    Raises:
        ValueError: if frame indices repeat or differ between labels and predictions, or a
            scored cell has a non-finite coordinate
    """
    if not labels.index.is_unique or not predictions.index.is_unique:
        raise ValueError("Frame indices must be unique")
    if set(labels.index) != set(predictions.index):
        raise ValueError("Prediction and label frames differ")
    predictions = predictions.loc[labels.index]
    result = {}
    for key in labels.columns.get_level_values(1).unique():
        gt      = labels.xs(key, axis=1, level=1).droplevel(0, axis=1)
        pred    = predictions.xs(key, axis=1, level=1).droplevel(0, axis=1)
        xy      = gt[["x", "y"]].to_numpy(dtype=float)
        actual  = pred[["x", "y"]].to_numpy(dtype=float)
        visible = gt["visible"].to_numpy() == 2 if "visible" in gt else np.isfinite(xy).all(1)
        if key in exclude:
            visible[:] = False
        if not np.isfinite(xy[visible]).all() or not np.isfinite(actual[visible]).all():
            raise ValueError(f"Nonfinite visible coordinates for {key}")
        error = np.linalg.norm(actual - xy, axis=1)
        error[~visible] = np.nan
        result[key] = error
    return pd.DataFrame(result, index=labels.index)


def comparison_summary(errors: pd.DataFrame, diagonals: np.ndarray) -> dict:
    """Pooled error of one evaluation, in pixels and relative to each frame's image diagonal.

    Args:
        errors: output of visible_errors (NaN where not scored)
        diagonals: image diagonal in pixels, one per row of errors

    Returns:
        {"points": scored cells, "mean_px", "median_px", "normalized_mean" (mean of
        error / diagonal), "equal_keypoint_normalized_mean" (the same, averaged per keypoint
        first)}; the means are None when no cell is scored

    Raises:
        ValueError: unless there is one finite, positive diagonal per frame
    """
    diagonals = np.asarray(diagonals, dtype=float)
    if (diagonals.shape != (len(errors),) or not np.isfinite(diagonals).all()
            or (diagonals <= 0).any()):
        raise ValueError("One finite positive image diagonal is required per frame")
    values    = errors.to_numpy(dtype=float)
    norm      = values / diagonals[:, None]
    count     = int(np.isfinite(values).sum())
    populated = np.isfinite(values).any(axis=0)
    if not count:
        return dict(points=0, mean_px=None, median_px=None, normalized_mean=None,
                    equal_keypoint_normalized_mean=None)
    return dict(
        points=count,
        mean_px=float(np.nanmean(values)),
        median_px=float(np.nanmedian(values)),
        normalized_mean=float(np.nanmean(norm)),
        equal_keypoint_normalized_mean=float(np.nanmean(np.nanmean(norm[:, populated], axis=0))),
    )


def trained_keypoints(datasets: set[str], inventory: dict) -> set[str]:
    """Keypoints a model trained on these datasets supervises: the union of their `trainable` sets.

    Args:
        datasets: names of the datasets in the model's training CSV (unknown names are ignored)
        inventory: the `datasets` dict of <data_dir>/dataset_inventory.json
            (python -m mighty_mouse.preprocess.inventory)

    Returns:
        the supervised keypoint names
    """
    return set().union(*[set(inventory[d]["trainable"]) for d in datasets if d in inventory])
