"""
Conventions for Lightning Pose label CSVs (and their project.yaml) shared across the repo.

LP label (and prediction) CSVs have a three-level column header (scorer / bodyparts /
coords) and an index of image paths ending in ``.../<session>/<frame>.png``.
"""

from pathlib import Path

import pandas as pd

# scorer name written into every converted / combined CSV
SCORER = "All"

# per-keypoint columns in converted CSVs
COORDS = ("x", "y", "visible")


def read_labels_csv(path: Path | str) -> pd.DataFrame:
    """Read an LP-format label or prediction CSV.

    Args:
        path: path to the CSV file

    Returns:
        DataFrame with a (scorer, bodyparts, coords) column MultiIndex, indexed by image path
    """
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)


def session_name(img_path: str) -> str:
    """Return the session (parent directory name) of a labeled-frame image path."""
    return Path(img_path).parts[-2]


def frame_name(img_path: str) -> str:
    """Return the frame filename of a labeled-frame image path."""
    return Path(img_path).parts[-1]


def _check_new_keypoints(keypoints: list[str], existing: list[str]) -> None:
    if not keypoints:
        raise ValueError("no keypoints given")
    dupes = sorted({kp for kp in keypoints if keypoints.count(kp) > 1})
    if dupes:
        raise ValueError(f"keypoints listed more than once: {dupes}")
    present = [kp for kp in keypoints if kp in existing]
    if present:
        raise ValueError(f"keypoints already present: {present}")


def append_keypoint_columns(csv_text: str, keypoints: list[str]) -> str:
    """Append empty x/y columns for new keypoints to the text of a raw LP label CSV.

    Edits the text line by line instead of round-tripping through pandas, so every
    existing byte is preserved: pandas' default float parser drops the last digit of some
    17-significant-digit values, and writing converts CRLF line endings to LF. Each
    line keeps its own line ending.

    Args:
        csv_text: full CSV contents (read with newline translation off)
        keypoints: new keypoint names, appended in this order after the existing columns

    Returns:
        the new CSV contents

    Raises:
        ValueError: if the header isn't a scorer/bodyparts/coords x/y header, or a
            keypoint is empty, repeated, or already present
    """
    lines = csv_text.splitlines(keepends=True)
    header = [line.rstrip("\r\n").split(",") for line in lines[:3]]
    if [h[0] for h in header] != ["scorer", "bodyparts", "coords"] or len(header[0]) < 3:
        raise ValueError("expected a 3-row scorer/bodyparts/coords header with keypoint columns")
    if set(header[2][1:]) != {"x", "y"}:
        raise ValueError(f"expected only x/y coords, found {sorted(set(header[2][1:]))}")
    _check_new_keypoints(keypoints, header[1][1:])

    scorer = header[0][-1]
    suffixes = [
        "".join(f",{scorer},{scorer}" for _ in keypoints),
        "".join(f",{kp},{kp}" for kp in keypoints),
        ",x,y" * len(keypoints),
    ]
    row_suffix = ",," * len(keypoints)

    out = []
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if not body:
            out.append(line)
            continue
        out.append(body + (suffixes[i] if i < 3 else row_suffix) + line[len(body):])
    return "".join(out)


def add_project_keypoints(project_cfg: dict, keypoints: list[str]) -> dict:
    """Return a copy of an LP project.yaml dict with ``keypoints`` appended to keypoint_names.

    Raises:
        ValueError: if a keypoint is repeated or already in keypoint_names
    """
    existing = list(project_cfg.get("keypoint_names") or [])
    _check_new_keypoints(keypoints, existing)
    return {**project_cfg, "keypoint_names": existing + list(keypoints)}
