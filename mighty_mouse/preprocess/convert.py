"""
Convert one raw labeled dataset into the standardized format for combined training.

Driven by configs/datasets/<dataset>.yaml:
  - session and keypoint exclusions
  - bilateral keypoint lateralization (adds _left / _right variants)
  - keypoint renaming to canonical names from configs/keypoints.yaml
  - visibility column (2=labeled, 1=occluded/no-coordinate, 0=not in dataset)

Visibility convention (per Lightning Pose's training.uniform_heatmaps_for_nan_keypoints):
  2 = keypoint is labeled in this frame        -> Gaussian heatmap target (standard supervision)
  1 = keypoint belongs to this dataset, but has no coordinate this frame (unlabeled,
      opposite side of a lateralized dataset, ...) -> trained as occluded: uniform
      heatmap target, teaches the model low confidence everywhere for this keypoint
  0 = keypoint is not part of this dataset (assigned at merge time) -> excluded from
      the loss entirely; the model is given no opinion on this keypoint for this frame

A vis=1 default is only correct when the keypoint really might be occluded/absent in
that frame. Where a keypoint is known to actually be visible but wasn't given a
coordinate for structural reasons (e.g. hantman-mv/kaufman's unlabeled-but-visible
left paw), the per-dataset post-processing below forces vis 1 -> 0 instead, so the
model isn't taught to expect low confidence on a keypoint that's really there.
"""

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.labels import COORDS, SCORER, frame_name, session_name

SIDES = ("left", "right")


# ── per-dataset post-processing ───────────────────────────────────────────────
#
# Each entry in POST_PROCESS maps a dataset name to a function:
#   fn(df: pd.DataFrame, config: dict) -> pd.DataFrame
#
# Applied by post_process() to the output of process_split() (canonical names, remapped
# index). To add a new dataset: define a function below and register it in POST_PROCESS.


def suppress_occluded(df: pd.DataFrame, keypoints: Iterable[str]) -> pd.DataFrame:
    """Force visibility 1 -> 0 for the given keypoints; 2 and 0 are left alone.

    Modifies ``df`` in place.

    Args:
        df: processed split with canonical (SCORER, keypoint, coord) columns
        keypoints: canonical keypoint names to suppress

    Returns:
        the same DataFrame
    """
    for kp in keypoints:
        vis_col = (SCORER, kp, "visible")
        vis = df[vis_col].to_numpy()
        df[vis_col] = np.where(vis == 1.0, 0.0, vis)
    return df


# hantman-mv: every session is lateralized to "right" (configs/datasets/hantman-mv.yaml) --
# the camera is mounted on the right side of the body, so the original annotator only ever
# labeled the right paw. The left paw is visible in these frames too, it just was never
# labeled: an annotation gap, not occlusion. process_split()'s default (vis=1) would train
# the model that this is an occluded keypoint (uniform heatmap target, teaches low
# confidence everywhere) for a paw that's actually there. Force vis=0 (excluded from the
# loss entirely, no opinion) for every _left column instead.
#
# Exceptions (left at the default vis=1, i.e. empty = occluded):
#  - ear_*_left: deliberately mapped with all-empty source columns (see
#    scripts/preprocess/hantman-mv/README.md) to get the vis=1 suppression signal on
#    *both* sides, matching cazettes-side/facemap.
#  - d{1-4}_tip_left on *front* rows: the left fingertips were later pseudo-labeled and
#    manually adjusted on every front-view frame, so empty cells there are real occlusions.
#    On *side* rows they were never labeled or reviewed, so they are forced to vis=0.
#  - wrist_left (both views): labeled/being labeled manually on both views.
_HANTMAN_MV_LEFT_SUPPRESS_EXEMPT = frozenset([
    "ear_top_left", "ear_tip_left", "ear_bottom_left", "ear_base_left", "wrist_left",
])
_HANTMAN_MV_LEFT_FINGERTIPS = ("d1_tip_left", "d2_tip_left", "d3_tip_left", "d4_tip_left")
_HANTMAN_MV_SIDE_SUFFIX = "_side"


def _post_process_hantman_mv(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    kps = [
        kp for kp in df.columns.get_level_values(1).unique()
        if kp.endswith("_left")
        and kp not in _HANTMAN_MV_LEFT_SUPPRESS_EXEMPT
        and kp not in _HANTMAN_MV_LEFT_FINGERTIPS
    ]
    suppress_occluded(df, kps)

    side_rows = np.array([session_name(p).endswith(_HANTMAN_MV_SIDE_SUFFIX) for p in df.index])
    for kp in _HANTMAN_MV_LEFT_FINGERTIPS:
        if kp not in df.columns.get_level_values(1):
            continue
        vis_col = (SCORER, kp, "visible")
        vis = df[vis_col].to_numpy()
        df[vis_col] = np.where(side_rows & (vis == 1.0), 0.0, vis)
    return df


# kaufman: every session is lateralized to "right" (configs/datasets/kaufman.yaml) -- same
# situation as hantman-mv above: only the right forepaw's fingers were ever labeled. The
# left forepaw is visible in these frames too, but only has a few coarse keypoints
# (LFPm/LFPl/LFPp, excluded) -- its fingers are an annotation gap, not occlusion.
# process_split()'s default (vis=1) would train the model that this is an
# occluded keypoint (uniform heatmap target, teaches low confidence everywhere) for a paw
# that's actually there. Force vis=0 (excluded from the loss entirely, no opinion) for
# the _left forepaw columns only.
#
# The lateralized face keypoints (eye_*, ear_*, pad_*) are deliberately left at the
# default vis=1: the left side of the face really is hidden behind the head from this
# camera, so the occlusion signal is correct there (same reasoning as hantman-mv's ear
# exemption).
_KAUFMAN_LEFT_FOREPAW = frozenset([
    "wrist_left", "d1_tip_left", "d2_tip_left", "d3_tip_left", "d4_tip_left",
])


def _post_process_kaufman(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    kps = [kp for kp in df.columns.get_level_values(1).unique() if kp in _KAUFMAN_LEFT_FOREPAW]
    return suppress_occluded(df, kps)


# cazettes-side: every session is lateralized to "left" (configs/datasets/cazettes-side.yaml)
# -- the camera is on the left side of the body, and only the near (left) paw's fingertips
# were labeled. The right paw's fingers are visible in some frames too, but were never
# labeled: an annotation gap, not occlusion. The `{side}` mapping would otherwise give the
# far-side d{1-4}_tip_right columns the default vis=1 (trained as occluded: uniform heatmap
# target, teaches low confidence everywhere). Force vis=0 (excluded from the loss entirely,
# no opinion) for them instead. Everything else keeps the default: the far-side face
# keypoints really are hidden, and wrist_right has its own labels (rightPaw).
_CAZETTES_SIDE_FAR_FINGERTIPS = frozenset([
    "d1_tip_right", "d2_tip_right", "d3_tip_right", "d4_tip_right",
])


def _post_process_cazettes_side(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    kps = [
        kp for kp in df.columns.get_level_values(1).unique()
        if kp in _CAZETTES_SIDE_FAR_FINGERTIPS
    ]
    return suppress_occluded(df, kps)


# tyssowski: every session is lateralized to "right" (configs/datasets/tyssowski.yaml) and
# only the right paw was labeled -- same situation as kaufman above. The left paw is an
# annotation gap, not occlusion, so force vis=0 for the left wrist/fingertips rather than
# the default vis=1. (No left-side face keypoints are mapped, so nothing else to handle.)
_TYSSOWSKI_LEFT_FOREPAW = _KAUFMAN_LEFT_FOREPAW


def _post_process_tyssowski(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    kps = [kp for kp in df.columns.get_level_values(1).unique() if kp in _TYSSOWSKI_LEFT_FOREPAW]
    return suppress_occluded(df, kps)


POST_PROCESS: dict[str, Callable[[pd.DataFrame, dict], pd.DataFrame]] = {
    "cazettes-side": _post_process_cazettes_side,
    "hantman-mv": _post_process_hantman_mv,
    "kaufman": _post_process_kaufman,
    "kaufman-water": _post_process_kaufman,
    "tyssowski": _post_process_tyssowski,
}


def post_process(df: pd.DataFrame, config: dict, dataset_name: str) -> pd.DataFrame:
    """Apply the dataset's POST_PROCESS hook, if it has one."""
    fn = POST_PROCESS.get(dataset_name)
    return df if fn is None else fn(df, config)


# ── paths ─────────────────────────────────────────────────────────────────────

def raw_dataset_dir(raw_dir: Path, dataset_name: str, config: dict) -> Path:
    """Return the raw folder for a dataset: ``raw_folder`` from its config if set (e.g.
    ``<name>-v2`` for relabeled data), otherwise the dataset name."""
    return raw_dir / (config.get("raw_folder") or dataset_name)


# ── validation ────────────────────────────────────────────────────────────────

def csv_keypoints(dfs: Iterable[pd.DataFrame | None]) -> set[str]:
    """Union of keypoint names across the given CSVs (None entries are skipped)."""
    kps: set[str] = set()
    for df in dfs:
        if df is not None:
            kps |= set(df.columns.get_level_values(1).unique())
    return kps


def csv_sessions(dfs: Iterable[pd.DataFrame | None]) -> set[str]:
    """Union of session names across the given CSVs (None entries are skipped)."""
    sessions: set[str] = set()
    for df in dfs:
        if df is not None:
            sessions |= {session_name(p) for p in df.index}
    return sessions


def check_exclusions(config: dict, kps: set[str], sessions: set[str]) -> list[str]:
    """Every excluded keypoint and session must exist in the CSVs."""
    errors = [
        f"exclude.keypoints: '{kp}' not found in any CSV"
        for kp in config["exclude"]["keypoints"] if kp not in kps
    ]
    errors += [
        f"exclude.sessions: '{s}' not found in any CSV"
        for s in config["exclude"]["sessions"] if s not in sessions
    ]
    return errors


def check_keypoint_sources(config: dict, kps: set[str]) -> list[str]:
    """Every source keypoint in the ``keypoints`` mapping must exist in the CSVs."""
    return [
        f"keypoints: source '{src}' not found in any CSV"
        for src in config["keypoints"] if src not in kps
    ]


def check_keypoint_targets(config: dict, canonical_kps: list[str]) -> list[str]:
    """Every mapping target (both sides, for ``{side}`` targets) must be canonical."""
    errors: list[str] = []
    for src, tgt in config["keypoints"].items():
        resolved = [tgt.format(side=side) for side in SIDES] if "{side}" in tgt else [tgt]
        for name in resolved:
            if name not in canonical_kps:
                errors.append(f"keypoints: '{src}' → '{name}' not in keypoints.yaml")
    return errors


def check_duplicate_targets(config: dict) -> list[str]:
    """No two source keypoints may resolve to the same canonical name (the later one would
    silently overwrite the earlier in map_keypoints)."""
    seen: dict[str, str] = {}
    errors: list[str] = []
    for src, tgt in config["keypoints"].items():
        resolved = [tgt.format(side=side) for side in SIDES] if "{side}" in tgt else [tgt]
        for name in resolved:
            if name in seen:
                errors.append(f"keypoints: '{src}' and '{seen[name]}' both map to '{name}'")
            seen[name] = src
    return errors


def check_sessions(config: dict, sessions: set[str]) -> list[str]:
    """``sessions`` entries must exist in the CSVs, and every CSV session must appear in
    either ``sessions`` or ``exclude.sessions``."""
    cfg_sessions = config.get("sessions") or {}
    errors = [f"sessions: '{s}' not found in any CSV" for s in cfg_sessions if s not in sessions]
    accounted = set(cfg_sessions) | set(config["exclude"]["sessions"])
    errors += [
        f"CSV session '{s}' missing from sessions and exclude.sessions"
        for s in sorted(sessions - accounted)
    ]
    return errors


def check_session_sides(config: dict) -> list[str]:
    """If any keypoint is lateralized, every non-excluded session's side must be left/right.

    map_keypoints() only routes a {side} keypoint's labels to a session whose side is
    exactly "left" or "right" -- any other value (null, "top", ...) would silently drop
    that session's labels for every lateralized keypoint and mark them visible=1.
    """
    exc_kps      = config["exclude"]["keypoints"]
    exc_sessions = config["exclude"]["sessions"]
    cfg_sessions = config.get("sessions") or {}
    side_kps = [
        src for src, tgt in config["keypoints"].items()
        if "{side}" in tgt and src not in exc_kps
    ]
    if not side_kps:
        return []
    return [
        f"sessions: '{s}' has side {side!r}, but {side_kps} are mapped to "
        "*_{side} names -- side must be 'left' or 'right'"
        for s, side in sorted(cfg_sessions.items())
        if s not in exc_sessions and side not in SIDES
    ]


def validate_dataset_config(
    config: dict,
    canonical_kps: list[str],
    dfs: Iterable[pd.DataFrame | None],
) -> list[str]:
    """Check a dataset config against its raw CSVs and the canonical keypoint list.

    Args:
        config: normalized dataset config (see configs.load_dataset_config)
        canonical_kps: keypoints from configs/keypoints.yaml
        dfs: the dataset's raw CSV splits; None entries (missing splits) are skipped

    Returns:
        list of error messages (empty if valid)
    """
    dfs = list(dfs)
    kps = csv_keypoints(dfs)
    sessions = csv_sessions(dfs)
    return (
        check_exclusions(config, kps, sessions)
        + check_keypoint_sources(config, kps)
        + check_keypoint_targets(config, canonical_kps)
        + check_duplicate_targets(config)
        + check_sessions(config, sessions)
        + check_session_sides(config)
    )


# ── processing ────────────────────────────────────────────────────────────────

def drop_excluded_sessions(df: pd.DataFrame, exc_sessions: Iterable[str]) -> pd.DataFrame:
    """Drop rows whose session is in ``exc_sessions``."""
    exc = set(exc_sessions)
    keep = [session_name(p) not in exc for p in df.index]
    return df.loc[keep]


def map_keypoints(
    df: pd.DataFrame,
    mapping: dict[str, str],
    session_sides: dict[str, str],
) -> pd.DataFrame:
    """Rename raw keypoints to canonical names, lateralize ``{side}`` targets, add visibility.

    A plain target copies x/y and sets visible=2 where labeled, 1 where NaN. A ``{side}``
    target produces both a _left and a _right keypoint: each row's coordinates go only to
    the side its session is assigned in ``session_sides``; the other side gets NaN and
    visible=1.

    Args:
        df: raw CSV split (any scorer name)
        mapping: raw keypoint name -> canonical name (optionally containing ``{side}``)
        session_sides: session name -> "left" / "right"

    Returns:
        DataFrame with (SCORER, canonical, coord) columns in mapping order, same index as df
    """
    orig_scorer = df.columns.get_level_values(0)[0]
    sides = pd.Series([session_name(p) for p in df.index]).map(session_sides)
    is_side = {side: (sides == side).to_numpy() for side in SIDES}

    arrays: dict[tuple[str, str, str], np.ndarray] = {}

    def _add(canonical: str, x: np.ndarray, y: np.ndarray, vis: np.ndarray) -> None:
        arrays[(SCORER, canonical, "x")]       = x
        arrays[(SCORER, canonical, "y")]       = y
        arrays[(SCORER, canonical, "visible")] = vis

    for kp, canonical_tmpl in mapping.items():
        x_orig  = df[(orig_scorer, kp, "x")].to_numpy(dtype=float)
        y_orig  = df[(orig_scorer, kp, "y")].to_numpy(dtype=float)
        labeled = ~np.isnan(x_orig)

        if "{side}" in canonical_tmpl:
            for side in SIDES:
                on_side = is_side[side]
                _add(
                    canonical_tmpl.format(side=side),
                    np.where(on_side, x_orig, np.nan),
                    np.where(on_side, y_orig, np.nan),
                    np.where(on_side & labeled, 2.0, 1.0),
                )
        else:
            _add(canonical_tmpl, x_orig, y_orig, np.where(labeled, 2.0, 1.0))

    names = ["scorer", "bodyparts", "coords"]
    if arrays:
        columns = pd.MultiIndex.from_tuples(list(arrays), names=names)
    else:
        columns = pd.MultiIndex.from_arrays([[], [], []], names=names)
    return pd.DataFrame(arrays, index=df.index, columns=columns)


def reindex_to_canonical(df: pd.DataFrame, canonical_kps: list[str]) -> pd.DataFrame:
    """Reorder columns to the canonical keypoint order, adding absent keypoints.

    Keypoints not present in ``df`` get NaN coordinates and visible=0. Columns for
    keypoints not in ``canonical_kps`` are dropped.
    """
    present = set(df.columns.get_level_values(1))
    all_cols = pd.MultiIndex.from_tuples(
        [(SCORER, kp, coord) for kp in canonical_kps for coord in COORDS],
        names=["scorer", "bodyparts", "coords"],
    )
    result = df.reindex(columns=all_cols)
    absent_vis = [(SCORER, kp, "visible") for kp in canonical_kps if kp not in present]
    result.loc[:, absent_vis] = 0.0
    return result


def remap_index(index: Iterable[str], dataset_name: str) -> pd.Index:
    """Rewrite image paths to ``labeled-data/<dataset_name>/<session>/<frame>``."""
    return pd.Index([
        str(Path("labeled-data") / dataset_name / session_name(p) / frame_name(p))
        for p in index
    ])


def process_split(
    df: pd.DataFrame,
    config: dict,
    dataset_name: str,
    canonical_kps: list[str],
) -> pd.DataFrame:
    """Apply exclusions, lateralization, renaming, and visibility to one CSV split.

    Does not apply the dataset's POST_PROCESS hook -- see post_process().

    Args:
        df: raw CSV split
        config: normalized dataset config
        dataset_name: used for the output index paths
        canonical_kps: keypoints from configs/keypoints.yaml

    Returns:
        DataFrame with every canonical keypoint's columns (absent keypoints get
        visible=0) and index paths remapped under labeled-data/<dataset_name>/
    """
    exc_kps = set(config["exclude"]["keypoints"])
    mapping = {k: v for k, v in config["keypoints"].items() if k not in exc_kps}

    df = drop_excluded_sessions(df, config["exclude"]["sessions"])
    result = map_keypoints(df, mapping, config.get("sessions") or {})
    result = reindex_to_canonical(result, canonical_kps)
    result.index = remap_index(result.index, dataset_name)
    return result


def copy_images(
    index: Iterable[str],
    raw_dataset_dir: Path,
    out_dir: Path,
    link: bool = False,
) -> int:
    """Copy each frame of a processed split from the raw dataset into ``out_dir``.

    With ``link``, symlink each frame to the raw file instead (resolved, so a raw folder
    whose labeled-data is itself a symlink to the original frames links to the real
    files). Frames already present in ``out_dir`` are skipped.

    Args:
        index: processed index paths (labeled-data/<dataset>/<session>/<frame>)
        raw_dataset_dir: raw dataset folder containing labeled-data/<session>/<frame>
        out_dir: output data directory
        link: symlink instead of copying

    Returns:
        number of files created
    """
    n_created = 0
    for new_path in index:
        src = raw_dataset_dir / "labeled-data" / session_name(new_path) / frame_name(new_path)
        dst = out_dir / new_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            if link:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            n_created += 1
    return n_created
