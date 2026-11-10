"""
Loading and cross-checking the YAML files under configs/.

  configs/keypoints.yaml           canonical keypoint vocabulary (defines CSV column order)
  configs/model.yaml               Lightning Pose model config (names the output columns)
  configs/datasets/<name>.yaml     per-dataset conversion config (see README.md)
"""

import re
from pathlib import Path

import yaml

from mighty_mouse.paths import repo_root


def default_configs_dir() -> Path:
    """Return the repo's configs/ directory."""
    return repo_root() / "configs"


def load_canonical_keypoints(configs_dir: Path) -> list[str]:
    """Load the canonical keypoint list from configs/keypoints.yaml."""
    with open(configs_dir / "keypoints.yaml") as f:
        return yaml.safe_load(f)["keypoints"]


SIDE_RE = re.compile(r"_(left|right)$")
_REGION_RE = re.compile(r"^\s*#\s*─+\s*(.+?)\s*─+\s*$")
_KEYPOINT_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_]+)\s*(?:#.*)?$")


def collapse_side(name: str) -> str:
    """Keypoint name without its `_left` / `_right` suffix (left/right pairs become one)."""
    return SIDE_RE.sub("", name)


def hflip_partner(name: str) -> str | None:
    """The `_left` / `_right` counterpart of a lateralized keypoint, None for midline ones."""
    if name.endswith("_left"):
        return name[:-5] + "_right"
    if name.endswith("_right"):
        return name[:-6] + "_left"
    return None


def load_regions(configs_dir: Path) -> dict[str, str]:
    """Map each (uncommented) canonical keypoint to its region, per the comment headers of
    configs/keypoints.yaml (`# ── eye ──` lines)."""
    region, out = None, {}
    for line in (configs_dir / "keypoints.yaml").read_text().splitlines():
        if m := _REGION_RE.match(line):
            region = m.group(1)
        elif (m := _KEYPOINT_RE.match(line)) and region:
            out[m.group(1)] = region
    return out


def load_model_config(configs_dir: Path) -> dict:
    """Load configs/model.yaml as a plain dict."""
    with open(configs_dir / "model.yaml") as f:
        return yaml.safe_load(f)


def normalize_dataset_config(cfg: dict) -> dict:
    """Fill in optional dataset-config fields so downstream code can index them directly.

    ``exclude.sessions`` and ``exclude.keypoints`` become lists (YAML leaves an empty
    entry as None) and ``keypoints`` becomes a dict. Modifies ``cfg`` in place.

    Args:
        cfg: parsed configs/datasets/<name>.yaml

    Returns:
        the same dict, normalized
    """
    exc = cfg.get("exclude") or {}
    exc["sessions"]  = exc.get("sessions")  or []
    exc["keypoints"] = exc.get("keypoints") or []
    cfg["exclude"]   = exc
    cfg["keypoints"] = cfg.get("keypoints") or {}
    return cfg


def load_dataset_config(configs_dir: Path, dataset_name: str) -> dict:
    """Load and normalize configs/datasets/<dataset_name>.yaml.

    Raises:
        FileNotFoundError: if the dataset has no config file
    """
    path = configs_dir / "datasets" / f"{dataset_name}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return normalize_dataset_config(cfg)


def check_model_config(model_cfg: dict, canonical_kps: list[str]) -> list[str]:
    """Verify model.yaml's keypoint vocabulary matches keypoints.yaml exactly.

    The two files duplicate the same vocabulary and neither is derived from the other:
    output CSV column order comes from keypoints.yaml, while Lightning Pose names those
    columns from model.yaml's keypoint_names. The two ways they can disagree fail very
    differently, so both are checked before any CSV is written:

      - different length: LP raises at train time (this shipped once, in the commit that
        added cazettes: keypoints.yaml went to 43, model.yaml stayed at 41)
      - same keypoints, different order: nothing raises, ever. Every prediction is
        silently attributed to the wrong keypoint, and pixel_error compares wrong pairs.

    Args:
        model_cfg: parsed configs/model.yaml
        canonical_kps: keypoints from configs/keypoints.yaml

    Returns:
        list of error messages (empty if consistent)
    """
    model_kps  = model_cfg["data"]["keypoint_names"]
    n_declared = model_cfg["data"]["num_keypoints"]
    errors: list[str] = []

    missing = [k for k in canonical_kps if k not in model_kps]
    extra   = [k for k in model_kps if k not in canonical_kps]
    if missing:
        errors.append(f"model.yaml: keypoint_names is missing {missing}")
    if extra:
        errors.append(f"model.yaml: keypoint_names has unknown keypoint(s) {extra}")
    if not missing and not extra and model_kps != canonical_kps:
        errors.append(
            "model.yaml: keypoint_names holds the same keypoints as keypoints.yaml but in a "
            "different order — CSV columns would be labeled with the wrong keypoint names"
        )
    if n_declared != len(canonical_kps):
        errors.append(
            f"model.yaml: num_keypoints={n_declared} but keypoints.yaml has "
            f"{len(canonical_kps)} keypoints"
        )

    return errors
