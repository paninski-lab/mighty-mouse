"""Stage 2 input: the frames stage 1 kept for one dataset, gathered across its videos.

Everything here is read from the stage-1 outputs (`dedup/`) and the embeddings (`embeddings/`);
videos without a stage-1 output yet are skipped and reported.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mighty_mouse.pretrain.dedup import load_selection
from mighty_mouse.pretrain.dedup import output_path as dedup_path
from mighty_mouse.pretrain.embed import output_path as embeddings_path


@dataclass
class Pool:
    """A dataset's stage-1 frames: row `i` is frame `frame_idx[i]` of `video_keys[video[i]]`."""

    embeddings: np.ndarray   # (n, dim) float16, unit length
    video: np.ndarray        # (n,) index into video_keys
    frame_idx: np.ndarray    # (n,) int32
    video_keys: list[str]
    raw_frames: int          # usable frames in the dataset's processed videos (all of stage 1's)
    missing: list[str]       # video keys with no stage-1 output


def pool_counts(
    dedup_root: Path, model_type: str, dataset: str, keys: list[str]
) -> tuple[int, int, list[str]]:
    """Cheap sizes of a dataset's pool, without reading embeddings.

    Args:
        dedup_root: `<pretrain_dir>/dedup`
        model_type: embedding model name (a subdirectory of the stage-1 outputs)
        dataset: dataset name
        keys: video keys of the dataset's train/unlabeled videos

    Returns:
        - frames in the processed videos, not counting each video's skipped leading frames
        - frames stage 1 kept
        - keys of videos without a stage-1 output
    """
    raw, kept, missing = 0, 0, []
    for key in keys:
        path = dedup_path(dedup_root, model_type, dataset, key)
        if not path.exists():
            missing.append(key)
            continue
        frame_idx, _, stats = load_selection(path)
        raw += int(stats["n_frames"]) - int(stats.get("skip_first", 0))
        kept += len(frame_idx)
    return raw, kept, missing


def load_pool(
    dedup_root: Path,
    embeddings_root: Path,
    model_type: str,
    dataset: str,
    keys: list[str],
) -> Pool:
    """Gather the normalized embeddings of every frame stage 1 kept in a dataset.

    Args:
        dedup_root: `<pretrain_dir>/dedup`
        embeddings_root: `<pretrain_dir>/embeddings`
        model_type: embedding model name
        dataset: dataset name
        keys: video keys of the dataset's train/unlabeled videos

    Raises:
        ValueError: if no video has a stage-1 output
    """
    embeddings, video, frame_idx, found, missing = [], [], [], [], []
    raw = 0
    for key in keys:
        path = dedup_path(dedup_root, model_type, dataset, key)
        if not path.exists():
            missing.append(key)
            continue
        idx, _, stats = load_selection(path)
        raw += int(stats["n_frames"]) - int(stats.get("skip_first", 0))
        emb = np.load(embeddings_path(embeddings_root, model_type, dataset, key), mmap_mode="r")
        rows = np.asarray(emb[idx], dtype=np.float32)
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-6)
        embeddings.append(rows.astype(np.float16))
        video.append(np.full(len(idx), len(found), dtype=np.int32))
        frame_idx.append(idx.astype(np.int32))
        found.append(key)
    if not found:
        raise ValueError(f"{dataset}: no videos with stage-1 outputs")
    return Pool(
        np.concatenate(embeddings), np.concatenate(video), np.concatenate(frame_idx),
        found, raw, missing,
    )
