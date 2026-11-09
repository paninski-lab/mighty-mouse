"""Statistics of an ensemble's predictions: the median prediction, three definitions of the
ensemble's spread, and the per-frame / per-keypoint tables built from them.

The three variance modes (SD in px, x and y separately):
  ensemble var       Var(x) across members
  inv-conf-weighted  Var(x) / mean(c)
  mixture            mean(1/c) + Var(x)
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MODES = [("ensemble var", "Var(x)"), ("inv-conf-weighted", "Var(x) / mean(c)"),
         ("mixture", "mean(1/c) + Var(x)")]


@dataclass
class EnsembleStats:
    """Arrays of shape (frames, keypoints).

    Attributes:
        x, y: median prediction across members
        mean_conf: mean likelihood across members
        shown: whether any member's likelihood reaches the confidence threshold
        sd: {mode name: (SD in x, SD in y)} for every mode in MODES
    """

    x: np.ndarray
    y: np.ndarray
    mean_conf: np.ndarray
    shown: np.ndarray
    sd: dict[str, tuple[np.ndarray, np.ndarray]]


def sd_column(mode: str) -> str:
    """Name of a mode's SD column in the per-frame table."""
    return f"sd_{mode.replace(' ', '_')}"


def select_modes(requested: str = "") -> list[tuple[str, str]]:
    """The variance modes named in a comma list (every mode if empty), in MODES order.

    Raises:
        ValueError: if none of the names is a mode
    """
    if not requested:
        return list(MODES)
    want = [m.strip() for m in requested.split(",")]
    modes = [mode for mode in MODES if mode[0] in want]
    if not modes:
        raise ValueError(f"--modes {want} matches none of {[m for m, _ in MODES]}")
    return modes


def usable_keypoints(pred: pd.DataFrame, exclude: str = "", include: str = "") -> list[str]:
    """Keypoints of a prediction frame that have a likelihood column, in column order.

    Args:
        pred: predictions with (scorer, keypoint, x|y|likelihood) columns
        exclude: comma regexes; matching keypoints are dropped
        include: comma regexes; if given, only matching keypoints are kept
    """
    scorer = pred.columns[0][0]
    kps = [k for k in dict.fromkeys(pred.columns.get_level_values(1))
           if (scorer, k, "likelihood") in pred.columns]
    if exclude:
        kps = [k for k in kps if not any(re.search(r, k) for r in exclude.split(","))]
    if include:
        kps = [k for k in kps if any(re.search(r, k) for r in include.split(","))]
    return kps


def ensemble_stats(preds: list[pd.DataFrame], kps: list[str], conf: float) -> EnsembleStats:
    """Statistics over ensemble members, on the first member's frames.

    Args:
        preds: one prediction frame per member (same scorer name as the first)
        kps: keypoints to summarize
        conf: a keypoint counts as shown on a frame if any member's likelihood is >= conf
    """
    scorer = preds[0].columns[0][0]
    idx = preds[0].index
    preds = [p.loc[idx] for p in preds]
    # models x frames x kps
    x = np.stack([np.stack([p[(scorer, k, "x")].values for k in kps], 1) for p in preds])
    y = np.stack([np.stack([p[(scorer, k, "y")].values for k in kps], 1) for p in preds])
    c = np.stack([np.stack([p[(scorer, k, "likelihood")].values for k in kps], 1) for p in preds])
    shown = (c >= conf).any(0)
    mconf = c.mean(0)
    vx, vy = np.var(x, 0), np.var(y, 0)
    cf = np.maximum(c, 1e-5)
    comp = (1.0 / cf).mean(0)
    sd = {"ensemble var": (np.sqrt(vx), np.sqrt(vy)),
          "inv-conf-weighted": (np.sqrt(vx / np.maximum(mconf, 1e-5)),
                                np.sqrt(vy / np.maximum(mconf, 1e-5))),
          "mixture": (np.sqrt(comp + vx), np.sqrt(comp + vy))}
    return EnsembleStats(np.median(x, 0), np.median(y, 0), mconf, shown, sd)


def labelled_keypoints(gt: pd.DataFrame | None, kps: list[str]) -> list[str]:
    """Keypoints that have a visible column in the ground truth (none without ground truth)."""
    if gt is None:
        return []
    scorer = gt.columns[0][0]
    return [k for k in kps if (scorer, k, "visible") in gt.columns]


def per_frame_keypoint_table(
    stats: EnsembleStats, frames: pd.Index, kps: list[str], modes: list[tuple[str, str]],
    gt: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per shown (frame, keypoint): median position, mean confidence, the SD of each
    mode, and the median's pixel error against the ground truth where it is labeled
    (visible == 2; NaN elsewhere).

    Args:
        stats: from `ensemble_stats`
        frames: the frames the stats are for
        kps: the keypoints the stats are for
        modes: variance modes to include
        gt: ground truth reindexed to `frames`, or None
    """
    labelled = labelled_keypoints(gt, kps)
    scorer = gt.columns[0][0] if gt is not None else None
    rows = []
    for j, k in enumerate(kps):
        vis = ((pd.to_numeric(gt[(scorer, k, "visible")], errors="coerce") == 2).values
               if k in labelled else np.zeros(len(frames), bool))
        err = (np.hypot(stats.x[:, j] - gt[(scorer, k, "x")].astype(float).values,
                        stats.y[:, j] - gt[(scorer, k, "y")].astype(float).values)
               if k in labelled else np.full(len(frames), np.nan))
        err = np.where(vis, err, np.nan)
        for i, f in enumerate(frames):
            if stats.shown[i, j]:
                r = dict(frame=f, keypoint=k, mean_conf=stats.mean_conf[i, j],
                         x=stats.x[i, j], y=stats.y[i, j], err=err[i])
                for name, _ in modes:
                    a, b = stats.sd[name]
                    r[sd_column(name)] = float(np.hypot(a[i, j], b[i, j]))
                rows.append(r)
    # explicit columns: nothing above the threshold still gives a (header-only) table
    return pd.DataFrame(rows, columns=["frame", "keypoint", "mean_conf", "x", "y", "err"]
                        + [sd_column(name) for name, _ in modes])


def variance_summary(per: pd.DataFrame, modes: list[tuple[str, str]]) -> pd.DataFrame:
    """Per keypoint and mode: median SD, median error, and the Spearman correlation between
    error and SD over the labeled frames (NaN with fewer than 3)."""
    summ = []
    for k, g in per.groupby("keypoint", sort=False):
        for name, _ in modes:
            col = sd_column(name)
            lab = g[g.err.notna()]
            summ.append(dict(keypoint=k, variance=name, median_sd_px=g[col].median(),
                             median_err_px=lab.err.median(),
                             spearman_err_sd=(lab.err.corr(lab[col], method="spearman")
                                              if len(lab) > 2 else np.nan)))
    return pd.DataFrame(summ, columns=["keypoint", "variance", "median_sd_px", "median_err_px",
                                       "spearman_err_sd"])
