"""Stage 1 of frame selection: per-video dedup of CLS embeddings by k-means.

One random frame is kept from each of `k` clusters, so a video's redundant stretches
(many frames in the same cluster) collapse to few frames while rare poses/scenes keep
theirs. A video with at most `k` frames is kept whole. Each selected frame records how
many frames its cluster contained, for use by later stages.

torch is imported inside the functions that need it.
"""

import hashlib
import os
from pathlib import Path

import numpy as np

DEFAULT_K = 15000


def output_path(dedup_dir: Path, model_type: str, dataset: str, key: str) -> Path:
    """`<dedup_dir>/<model_type>/<dataset>/<key>.npz`, `key` being the video's key in the
    videos paths file."""
    return Path(dedup_dir) / model_type / dataset / f"{key}.npz"


def seed_from_key(key: str) -> int:
    """Stable 32-bit seed derived from a video key (not `hash()`, which varies per run)."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def select_frames(
    embeddings: np.ndarray,
    k: int = DEFAULT_K,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Keep one random frame per k-means cluster of a video's embeddings.

    Args:
        embeddings: per-frame CLS embeddings, shape (n_frames, dim)
        k: number of clusters (maximum number of frames kept)
        seed: seed for k-means and the within-cluster choice
        device: torch device to cluster on

    Returns:
        - sorted indices of the kept frames, int32
        - number of frames in each kept frame's cluster, int32 (all 1 if the video was
          kept whole)
        - stats: `n_frames`, `k`, `seed`, `n_iter`, `n_empty_clusters`, and the cosine
          distance from every frame to its nearest kept frame (`coverage_mean`,
          `coverage_p95`, `coverage_max`)
    """
    import torch

    from mighty_mouse.pretraining.kmeans import (
        DEFAULT_CHUNK,
        normalize_rows,
        spherical_kmeans,
    )

    n = len(embeddings)
    stats = {"n_frames": n, "k": k, "seed": seed}
    if n <= k:
        stats |= {"n_iter": 0, "n_empty_clusters": 0,
                  "coverage_mean": 0.0, "coverage_p95": 0.0, "coverage_max": 0.0}
        return np.arange(n, dtype=np.int32), np.ones(n, dtype=np.int32), stats

    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    x = normalize_rows(torch.from_numpy(embeddings).to(device=device, dtype=dtype))
    _, assign, n_iter = spherical_kmeans(x, k, seed=seed)

    # one random member per cluster: the member with the highest random rank (ranks are a
    # permutation, so exactly one row per non-empty cluster matches its cluster's maximum)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    rank = torch.randperm(n, generator=generator, device=x.device)
    top = torch.full((k,), -1, dtype=torch.int64, device=x.device)
    top.scatter_reduce_(0, assign, rank, reduce="amax")
    keep = (rank == top[assign]).nonzero().squeeze(1)
    counts = torch.bincount(assign, minlength=k)

    kept = x[keep].float()
    nearest = torch.empty(n, dtype=torch.float32, device=x.device)
    for start in range(0, n, DEFAULT_CHUNK):
        block = x[start:start + DEFAULT_CHUNK].float()
        nearest[start:start + DEFAULT_CHUNK] = (block @ kept.T).max(1).values
    coverage = (1 - nearest).clamp_min(0)

    stats |= {
        "n_iter": n_iter,
        "n_empty_clusters": int((counts == 0).sum()),
        "coverage_mean": float(coverage.mean()),
        "coverage_p95": float(torch.quantile(coverage, 0.95)),
        "coverage_max": float(coverage.max()),
    }
    return (
        keep.cpu().numpy().astype(np.int32),
        counts[assign[keep]].cpu().numpy().astype(np.int32),
        stats,
    )


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
