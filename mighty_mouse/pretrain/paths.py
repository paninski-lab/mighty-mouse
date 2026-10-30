"""Layout of the pretraining directory.

The repo's paths.yaml only holds `pretrain_dir`, the directory everything else lives in;
the rest of the layout is fixed here.
"""

from pathlib import Path

VIDEOS_PATHS_FILE = "paths.yaml"  # not the repo's paths.yaml: lists the pretraining videos
EMBEDDINGS_DIR = "embeddings"
DEDUP_DIR = "dedup"


def pretrain_dir(paths: dict) -> Path:
    """The `pretrain_dir` directory from a loaded repo paths.yaml.

    Raises:
        KeyError: if paths.yaml has no `pretrain_dir` entry
    """
    if "pretrain_dir" not in paths:
        raise KeyError("paths.yaml has no `pretrain_dir` entry (the pretraining data directory)")
    return Path(paths["pretrain_dir"])


def videos_paths_file(root: Path) -> Path:
    """`<pretrain_dir>/paths.yaml`: dataset -> split -> video key -> {path, dims, frames}"""
    return root / VIDEOS_PATHS_FILE


def embeddings_dir(root: Path) -> Path:
    """`<pretrain_dir>/embeddings`: per-frame CLS embeddings"""
    return root / EMBEDDINGS_DIR


def dedup_dir(root: Path) -> Path:
    """`<pretrain_dir>/dedup`: stage-1 per-video selections"""
    return root / DEDUP_DIR
