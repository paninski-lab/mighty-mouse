"""Tests for mighty_mouse.pretrain.pool."""

import numpy as np
import pytest

from mighty_mouse.pretrain.dedup import output_path as dedup_path
from mighty_mouse.pretrain.dedup import save_selection
from mighty_mouse.pretrain.embed import output_path as embeddings_path
from mighty_mouse.pretrain.pool import load_pool, pool_counts

MODEL = "m"


def _write(tmp_path, dataset, key, n_frames, kept, skip_first=0, seed=0):
    """Fake embeddings for a video of `n_frames` and a stage-1 output keeping `kept`."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n_frames, 4)).astype(np.float16) * 7
    path = embeddings_path(tmp_path / "emb", MODEL, dataset, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    out = dedup_path(tmp_path / "dedup", MODEL, dataset, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = np.asarray(kept, dtype=np.int32)
    save_selection(
        out, kept, np.ones(len(kept), np.int32), {"n_frames": n_frames, "skip_first": skip_first}
    )
    return emb


class TestPoolCounts:
    """Test the function pool_counts."""

    def test_pool_counts_sums_videos_and_skips_leading_frames(self, tmp_path):
        _write(tmp_path, "d", "v1", 100, [1, 5, 9], skip_first=10)
        _write(tmp_path, "d", "v2", 50, [0, 2])
        raw, kept, missing = pool_counts(tmp_path / "dedup", MODEL, "d", ["v1", "v2"])
        assert (raw, kept, missing) == (140, 5, [])

    def test_pool_counts_reports_videos_without_output(self, tmp_path):
        _write(tmp_path, "d", "v1", 10, [1])
        raw, kept, missing = pool_counts(tmp_path / "dedup", MODEL, "d", ["v1", "v9"])
        assert (raw, kept, missing) == (10, 1, ["v9"])


class TestLoadPool:
    """Test the function load_pool."""

    def test_load_pool_gathers_selected_frames_normalized(self, tmp_path):
        emb1 = _write(tmp_path, "d", "v1", 20, [3, 7], seed=1)
        emb2 = _write(tmp_path, "d", "v2", 30, [0, 11, 25], seed=2)
        pool = load_pool(tmp_path / "dedup", tmp_path / "emb", MODEL, "d", ["v1", "v2"])
        assert pool.video_keys == ["v1", "v2"]
        assert pool.video.tolist() == [0, 0, 1, 1, 1]
        assert pool.frame_idx.tolist() == [3, 7, 0, 11, 25]
        assert pool.embeddings.dtype == np.float16
        norms = np.linalg.norm(pool.embeddings.astype(np.float32), axis=1)
        assert np.allclose(norms, 1, atol=1e-2)
        expected = emb1[3].astype(np.float32) / np.linalg.norm(emb1[3].astype(np.float32))
        assert np.allclose(pool.embeddings[0].astype(np.float32), expected, atol=1e-2)
        assert pool.raw_frames == 50
        assert emb2.shape == (30, 4)

    def test_load_pool_skips_and_reports_missing_videos(self, tmp_path):
        _write(tmp_path, "d", "v1", 20, [3])
        pool = load_pool(tmp_path / "dedup", tmp_path / "emb", MODEL, "d", ["v0", "v1"])
        assert pool.video_keys == ["v1"] and pool.missing == ["v0"]
        assert pool.video.tolist() == [0]

    def test_load_pool_no_outputs_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no videos"):
            load_pool(tmp_path / "dedup", tmp_path / "emb", MODEL, "d", ["v1"])
