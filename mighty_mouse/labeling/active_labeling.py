"""Ensemble-disagreement frame selection for pseudo-labeling new keypoints.

The workflow (see `skills/active-pseudo-labeling/SKILL.md`): draw a few frames per session
into a `CollectedData_tolabel.csv`, label them by hand, train an ensemble on them, then
repeatedly add the unlabeled frames the ensemble disagrees about most (prefilled with the
ensemble mean) for the user to correct. Everything here operates on DataFrames; the
`scripts/active_labeling.py` CLI does the I/O and model inference.

A keypoint's *spread* in a frame is sqrt(var_x + var_y) across all ensemble members
(pixels). It only counts when the keypoint is *eligible*: at least one member predicts it
with likelihood above `min_conf`. Otherwise the members may simply disagree about an
occluded point they all (correctly) flag as low-confidence.
"""

import numpy as np
import pandas as pd

from mighty_mouse.labeling.ensemble import ensemble
from mighty_mouse.labels import frame_name, session_name

MANIFEST_COLUMNS = ["source", "round", "prefilled"]


def combine_originals(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Stack the train and test label frames (same columns) into one pool.

    Returns:
        (rows in train-then-test order, Series mapping image path -> "train" / "test")
    """
    if list(train.columns) != list(test.columns):
        raise ValueError("train and test CSVs have different columns")
    both = pd.concat([train, test])
    if not both.index.is_unique:
        raise ValueError("image paths are not unique across the train and test CSVs")
    source = pd.Series(["train"] * len(train) + ["test"] * len(test), index=both.index)
    return both, source


def sample_per_session(images: pd.Index, n: int, seed: int) -> pd.Index:
    """Draw up to `n` images at random from each session (parent directory).

    Returns:
        The drawn images, in their original order
    """
    rng = np.random.default_rng(seed)
    sessions = pd.Series([session_name(i) for i in images], index=images)
    chosen = []
    for _, group in sessions.groupby(sessions, sort=True):
        names = group.index.to_numpy()
        chosen.extend(rng.choice(names, size=min(n, len(names)), replace=False))
    return images[images.isin(chosen)]


def init_tolabel(
    originals: pd.DataFrame, sampled: pd.Index, keypoints: list[str], allow_existing: bool = False,
) -> pd.DataFrame:
    """The rows of `originals` in `sampled`.

    `keypoints` must be empty there unless `allow_existing` (e.g. rough cross-dataset
    pseudo-labels the user will correct); existing values are then carried over as is.
    """
    rows = originals.loc[sampled].copy()
    cols = [c for c in rows.columns if c[1] in keypoints]
    if len(cols) != 2 * len(keypoints):
        raise ValueError(f"expected x/y columns for every keypoint in {keypoints}")
    if not allow_existing and rows[cols].notna().any().any():
        raise ValueError(
            f"{keypoints} already have labels in the sampled rows (allow_existing to keep them)"
        )
    return rows


def make_manifest(
    index: pd.Index, source: pd.Series, round_: int, prefilled: bool,
) -> pd.DataFrame:
    """One manifest row per image: which CSV it came from, the round it was added, and
    whether its keypoints were prefilled with the ensemble mean."""
    return pd.DataFrame(
        {"source": source.loc[index].to_numpy(), "round": round_, "prefilled": prefilled},
        index=index,
    )[MANIFEST_COLUMNS]


def _member_values(preds: list[pd.DataFrame], index: pd.Index, kp: str) -> np.ndarray:
    """(n_members, n_images, 3) array of x, y, likelihood for one keypoint."""
    out = []
    for p in preds:
        scorer = p.columns.get_level_values(0)[0]
        cols = [(scorer, kp, c) for c in ("x", "y", "likelihood")]
        out.append(p.loc[index, cols].to_numpy(dtype=float))
    return np.stack(out)


def ensemble_spread(
    preds: list[pd.DataFrame], index: pd.Index, keypoints: list[str], min_conf: float = 0.9,
) -> pd.DataFrame:
    """Per-image, per-keypoint ensemble spread (pixels); NaN where not eligible."""
    cols = {}
    for kp in keypoints:
        v = _member_values(preds, index, kp)
        spread = np.sqrt(v[..., 0].var(axis=0) + v[..., 1].var(axis=0))
        eligible = (v[..., 2] > min_conf).any(axis=0)
        cols[kp] = np.where(eligible, spread, np.nan)
    return pd.DataFrame(cols, index=index)


def frame_scores(spread: pd.DataFrame) -> pd.Series:
    """Mean spread over each frame's eligible keypoints; NaN if it has none."""
    return spread.mean(axis=1)


def select_frames(scores: pd.Series, m: int | None) -> pd.Index:
    """The `m` highest-scoring frames, best first; NaN-scored (ineligible) frames never
    qualify. `m=None` returns every frame, NaN-scored ones last."""
    ranked = scores.sort_values(ascending=False, kind="stable", na_position="last")
    if m is None:
        return ranked.index
    return ranked.dropna().index[:m]


def order_by_session(chosen: pd.Index, originals_index: pd.Index) -> pd.Index:
    """Reorder `chosen` so each session's frames are contiguous, in frame order.

    Sessions follow their first appearance in `originals_index`; frames within a session
    are sorted by filename (zero-padded frame numbers, so lexicographic is temporal). This
    is the order a batch is shown in the LP app: labeling is more consistent when all of a
    video's frames come back to back.
    """
    first_seen: dict[str, int] = {}
    for i, img in enumerate(originals_index):
        first_seen.setdefault(session_name(img), i)
    return pd.Index(
        sorted(chosen, key=lambda img: (first_seen[session_name(img)], frame_name(img)))
    )


def prefill_rows(
    rows: pd.DataFrame,
    preds: list[pd.DataFrame],
    keypoints: list[str],
    threshold: float = 0.9,
    agg: str = "mean",
) -> pd.DataFrame:
    """Fill `keypoints` with the mean (or median, `agg="median"`) over members with
    likelihood >= threshold (empty if none)."""
    new, _ = ensemble(rows, preds, [(kp, kp) for kp in keypoints], threshold, agg)
    return new


def round_stats(
    spread_labeled: pd.DataFrame,
    spread_pool: pd.DataFrame,
    bin_edges: tuple[float, ...] = (1, 2, 4),
    px_thresholds: tuple[float, ...] = (3, 10, 20),
) -> pd.DataFrame:
    """Per-keypoint summary of how the unlabeled pool compares to the labeled frames.

    `baseline` is the mean spread over the labeled frames where the keypoint is eligible
    (in-sample, so optimistic). Pool frames are binned by spread / baseline into
    [0, e1), [e1, e2), ..., [e_last, inf); `pool_ineligible` counts frames with no
    high-confidence member, which are never candidates. The `n>Npx` columns count pool
    frames with spread above N pixels, which stay comparable across rounds even when the
    baseline moves (the ratio bins don't).
    """
    edges = [0.0, *bin_edges, np.inf]
    rows = []
    for kp in spread_labeled.columns:
        base = spread_labeled[kp].mean()
        pool = spread_pool[kp]
        row = {
            "keypoint": kp,
            "labeled_eligible": int(spread_labeled[kp].notna().sum()),
            "baseline": base,
            "pool_frames": len(pool),
            "pool_ineligible": int(pool.isna().sum()),
        }
        ratio = pool.dropna() / base if base and base > 0 else pd.Series(dtype=float)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            label = f"x{lo:g}-{hi:g}" if np.isfinite(hi) else f"x{lo:g}+"
            row[label] = int(((ratio >= lo) & (ratio < hi)).sum()) if len(ratio) else np.nan
        for t in px_thresholds:
            row[f"n>{t:g}px"] = int((pool > t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_training_csv(trained: pd.DataFrame, tolabel: pd.DataFrame) -> dict[str, int]:
    """How the CSV a round's models were trained on differs from the current tolabel rows.

    Row order is ignored (a `reorder` after `train-csv` is harmless).

    Returns:
        {"missing": tolabel rows the models never saw (e.g. `train-csv` was skipped after a
        `select`), "extra": trained rows no longer in tolabel, "edited": common rows whose
        values differ (labels changed after training)}
    """
    missing = tolabel.index.difference(trained.index)
    extra = trained.index.difference(tolabel.index)
    common = trained.index.intersection(tolabel.index)
    a = trained.loc[common].to_numpy(dtype=float)
    b = tolabel.loc[common].to_numpy(dtype=float)
    same = (np.isnan(a) == np.isnan(b)) & ~(np.abs(np.nan_to_num(a) - np.nan_to_num(b)) > 1e-6)
    return {"missing": len(missing), "extra": len(extra), "edited": int((~same.all(axis=1)).sum())}


def merge_back(
    originals: pd.DataFrame, tolabel: pd.DataFrame, keypoints: list[str],
) -> pd.DataFrame:
    """Copy the `keypoints` columns of `tolabel` into `originals`, keeping its rows and order.

    Raises:
        ValueError: if an original image is missing from `tolabel`, or `tolabel` differs
            from `originals` in any other column (it would silently be lost)
    """
    missing = originals.index.difference(tolabel.index)
    if len(missing):
        raise ValueError(f"{len(missing)} original images are not in tolabel, e.g. {missing[0]}")
    if list(tolabel.columns) != list(originals.columns):
        raise ValueError("tolabel and original columns differ")
    sub = tolabel.loc[originals.index]
    kp_cols = [c for c in originals.columns if c[1] in keypoints]
    other = [c for c in originals.columns if c[1] not in keypoints]
    o, t = originals[other].to_numpy(dtype=float), sub[other].to_numpy(dtype=float)
    same = (np.isnan(o) == np.isnan(t)) & ~(np.abs(np.nan_to_num(o) - np.nan_to_num(t)) > 1e-6)
    if not same.all():
        bad = sorted({other[j][1] for j in np.where(~same.all(axis=0))[0]})
        raise ValueError(f"tolabel differs from the originals in non-target keypoints: {bad}")
    new = originals.copy()
    new[kp_cols] = sub[kp_cols]
    return new
