"""Stage 1 of frame selection: per-video dedup of CLS embeddings by k-means.

At most `k` frames are kept per video, chosen so that redundant stretches (many frames in
the same cluster) collapse to few frames while rare poses/scenes keep theirs:

- a video with at most `k` frames is kept whole;
- a shorter video is clustered into `k` clusters with one random frame kept from each
  ("flat");
- a long video (at least `two_level_min_frames`) uses a two-level k-means with balanced
  sampling (`hierarchical.hierarchical_select`), which under-weights dense stretches more.

Each selected frame records the size of the cluster it was picked from, for later stages.

torch is imported inside the functions that need it.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_K = 15000
K1_FACTOR = 3               # two-level: level-1 clusters = K1_FACTOR * k
K2 = 1500                   # two-level: level-2 groups
TWO_LEVEL_MIN_FRAMES = 100_000
METHODS = ("auto", "flat", "two_level")

# leading frames that aren't video (e.g. an encoder-info placeholder), per dataset
SKIP_FIRST_FRAMES = {
    "aind-vbn-body": 1,
    "aind-vbn-face": 1,
}


def output_path(dedup_dir: Path, model_type: str, dataset: str, key: str) -> Path:
    """`<dedup_dir>/<model_type>/<dataset>/<key>.npz`, `key` being the video's key in the
    videos paths file."""
    return Path(dedup_dir) / model_type / dataset / f"{key}.npz"


def seed_from_key(key: str) -> int:
    """Stable 32-bit seed derived from a video key (not `hash()`, which varies per run)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def choose_method(
    n_frames: int,
    k: int = DEFAULT_K,
    method: str = "auto",
    two_level_min_frames: int = TWO_LEVEL_MIN_FRAMES,
) -> str:
    """The selection method used for a video with `n_frames` usable frames.

    Args:
        n_frames: frames available for selection (after skipping leading frames)
        k: maximum number of frames kept
        method: `"auto"`, `"flat"` or `"two_level"`
        two_level_min_frames: with `"auto"`, the smallest video that uses two-level

    Returns:
        `"whole"` (video has at most `k` frames), `"flat"` or `"two_level"`

    Raises:
        ValueError: if `method` is unknown
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choices: {METHODS}")
    if n_frames <= k:
        return "whole"
    if method == "auto":
        return "two_level" if n_frames >= two_level_min_frames else "flat"
    return method


def stored_method(stats: dict) -> str:
    """Method recorded in a saved selection's stats; files from before the `method` field
    existed were flat (or whole, if no k-means ran)."""
    return stats.get("method") or ("whole" if stats["n_iter"] == 0 else "flat")


def select_frames(
    embeddings: np.ndarray,
    k: int = DEFAULT_K,
    seed: int = 0,
    device: str = "cpu",
    method: str = "auto",
    two_level_min_frames: int = TWO_LEVEL_MIN_FRAMES,
    skip_first: int = 0,
    k1_factor: int = K1_FACTOR,
    k2: int = K2,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select up to `k` representative frames of a video from its embeddings.

    Args:
        embeddings: per-frame CLS embeddings, shape (n_frames, dim)
        k: maximum number of frames kept
        seed: seed for k-means and the within-cluster choices
        device: torch device to cluster on
        method: `"auto"` (see `choose_method`), `"flat"` or `"two_level"`
        two_level_min_frames: with `"auto"`, the smallest video that uses two-level
        skip_first: leading frames to ignore entirely (never selected, not clustered, not
            counted in the coverage stats); returned indices still index `embeddings`
        k1_factor: two-level level-1 clusters, as a multiple of `k`
        k2: two-level level-2 groups

    Returns:
        - sorted indices of the kept frames into `embeddings`, int32
        - size of the cluster each kept frame was picked from, int32 (1 if the video was
          kept whole). For two-level this is the level-1 cluster, and the sizes don't sum to
          the number of frames because unchosen clusters are dropped.
        - stats: `n_frames` (including skipped), `k`, `seed`, `skip_first`, `method`, `k1`,
          `k2` (0 unless two-level), `n_iter`, `n_iter_level2`, `n_empty_clusters`, and the
          cosine distance from every usable frame to its nearest kept frame
          (`coverage_mean`, `coverage_p95`, `coverage_max`)

    Raises:
        ValueError: if `method` is unknown, or two-level is requested for a video with
            fewer than `k1_factor * k` usable frames
    """
    import torch

    from mighty_mouse.pretraining.kmeans import (
        DEFAULT_CHUNK,
        normalize_rows,
        random_member_per_cluster,
        spherical_kmeans,
    )

    n_total = len(embeddings)
    n = n_total - skip_first
    chosen = choose_method(n, k, method, two_level_min_frames)
    k1 = k1_factor * k if chosen == "two_level" else 0
    stats = {
        "n_frames": n_total, "k": k, "seed": seed, "skip_first": skip_first, "method": chosen,
        "k1": k1, "k2": k2 if chosen == "two_level" else 0,
        "n_iter": 0, "n_iter_level2": 0, "n_empty_clusters": 0,
        "coverage_mean": 0.0, "coverage_p95": 0.0, "coverage_max": 0.0,
    }
    if chosen == "whole":
        return np.arange(skip_first, n_total, dtype=np.int32), np.ones(n, dtype=np.int32), stats
    if chosen == "two_level" and k1 > n:
        raise ValueError(f"two_level needs at least {k1} usable frames, video has {n}")

    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    # copy=True: normalize_rows works in place and must not modify the caller's array
    x = torch.from_numpy(embeddings[skip_first:]).to(device=device, dtype=dtype, copy=True)
    x = normalize_rows(x)

    if chosen == "flat":
        _, assign, stats["n_iter"] = spherical_kmeans(x, k, seed=seed)
        keep = random_member_per_cluster(assign, k, seed)
        counts = torch.bincount(assign, minlength=k)
        stats["n_empty_clusters"] = int((counts == 0).sum())
        keep, sizes = keep.cpu().numpy(), counts[assign[keep]].cpu().numpy()
    else:
        from mighty_mouse.pretraining.hierarchical import hierarchical_select

        keep, sizes, level_stats = hierarchical_select(x, k, k1, k2, seed)
        stats |= level_stats

    kept = x[torch.from_numpy(keep).to(x.device)].float()
    nearest = torch.empty(n, dtype=torch.float32, device=x.device)
    for start in range(0, n, DEFAULT_CHUNK):
        block = x[start:start + DEFAULT_CHUNK].float()
        nearest[start:start + DEFAULT_CHUNK] = (block @ kept.T).max(1).values
    coverage = (1 - nearest).clamp_min(0)
    stats |= {
        "coverage_mean": float(coverage.mean()),
        "coverage_p95": float(torch.quantile(coverage, 0.95)),
        "coverage_max": float(coverage.max()),
    }
    return (keep + skip_first).astype(np.int32), sizes.astype(np.int32), stats


def save_selection(
    path: Path,
    frame_idx: np.ndarray,
    cluster_size: np.ndarray,
    stats: dict,
) -> None:
    """Write a selection to `path` (.npz) atomically, so a killed run never leaves a
    truncated file that a skip-existing check would trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, frame_idx=frame_idx, cluster_size=cluster_size, **stats)
    os.replace(tmp, path)


def load_selection(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inverse of `save_selection`: `(frame_idx, cluster_size, stats)`."""
    with np.load(path) as f:
        stats = {k: f[k].item() for k in f.files if k not in ("frame_idx", "cluster_size")}
        return f["frame_idx"], f["cluster_size"], stats
