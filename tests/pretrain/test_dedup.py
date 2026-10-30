"""Tests for mighty_mouse.pretrain.dedup."""

from pathlib import Path

import numpy as np
import pytest

from mighty_mouse.pretrain.dedup import (
    choose_method,
    load_selection,
    output_path,
    save_selection,
    seed_from_key,
    select_frames,
    stored_method,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def embeddings() -> np.ndarray:
    """300 frames in 3 tight groups of 100 near-identical frames."""
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8, dtype=np.float32) * 3
    return np.concatenate([c + 0.01 * rng.normal(size=(100, 8)) for c in centers]).astype(
        np.float32
    )


class TestOutputPath:
    """Test the function output_path."""

    def test_output_path_layout(self):
        assert output_path(Path("/d"), "m", "ibl", "key") == Path("/d/m/ibl/key.npz")


class TestSeedFromKey:
    """Test the function seed_from_key."""

    def test_seed_from_key_stable_and_distinct(self):
        assert seed_from_key("a") == seed_from_key("a")
        assert seed_from_key("a") != seed_from_key("b")
        assert 0 <= seed_from_key("a") < 2**32


class TestChooseMethod:
    """Test the function choose_method."""

    def test_choose_method_whole_when_at_most_k(self):
        assert choose_method(100, k=100, method="two_level") == "whole"

    def test_choose_method_auto_switches_at_threshold(self):
        assert choose_method(99_999, k=15000, method="auto") == "flat"
        assert choose_method(100_000, k=15000, method="auto") == "two_level"

    def test_choose_method_forced(self):
        assert choose_method(50_000, k=15000, method="two_level") == "two_level"
        assert choose_method(500_000, k=15000, method="flat") == "flat"

    def test_choose_method_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            choose_method(10, method="nope")


class TestStoredMethod:
    """Test the function stored_method."""

    def test_stored_method_reads_field(self):
        assert stored_method({"method": "two_level", "n_iter": 5}) == "two_level"

    def test_stored_method_infers_for_old_files(self):
        assert stored_method({"n_iter": 0}) == "whole"
        assert stored_method({"n_iter": 12}) == "flat"


class TestSelectFrames:
    """Test the function select_frames."""

    def test_select_frames_short_video_kept_whole(self, embeddings):
        idx, size, stats = select_frames(embeddings[:10], k=10)
        assert idx.tolist() == list(range(10))
        assert size.tolist() == [1] * 10
        assert stats["n_iter"] == 0

    def test_select_frames_one_per_cluster(self, embeddings):
        idx, size, stats = select_frames(embeddings, k=3, seed=0)
        assert len(idx) == 3
        # one kept frame from each group of 100
        assert sorted(idx // 100) == [0, 1, 2]
        assert size.tolist() == [100, 100, 100]
        assert stats["n_empty_clusters"] == 0

    def test_select_frames_indices_sorted_unique_in_range(self, embeddings):
        idx, size, _ = select_frames(embeddings, k=20, seed=1)
        assert np.all(np.diff(idx) > 0)
        assert idx.min() >= 0 and idx.max() < 300
        assert idx.dtype == np.int32 and size.dtype == np.int32

    def test_select_frames_cluster_sizes_sum_to_n(self, embeddings):
        idx, size, stats = select_frames(embeddings, k=7, seed=2)
        assert size.sum() == 300  # every frame belongs to exactly one kept cluster
        assert stats["n_empty_clusters"] == 7 - len(idx)

    def test_select_frames_deterministic_per_seed(self, embeddings):
        a = select_frames(embeddings, k=10, seed=5)[0]
        b = select_frames(embeddings, k=10, seed=5)[0]
        assert np.array_equal(a, b)

    def test_select_frames_does_not_modify_input(self, embeddings):
        before = embeddings.copy()
        select_frames(embeddings, k=10, seed=0)
        assert np.array_equal(before, embeddings)

    def test_select_frames_two_level(self, embeddings):
        idx, size, stats = select_frames(
            embeddings, k=20, seed=0, method="two_level", k1_factor=3, k2=6
        )
        assert len(idx) == 20 and np.all(np.diff(idx) > 0)
        assert stats["method"] == "two_level" and stats["k1"] == 60 and stats["k2"] == 6
        assert size.min() >= 1

    def test_select_frames_two_level_auto_uses_threshold(self, embeddings):
        _, _, stats = select_frames(
            embeddings, k=20, two_level_min_frames=100, k1_factor=3, k2=6
        )
        assert stats["method"] == "two_level"
        _, _, stats = select_frames(embeddings, k=20, two_level_min_frames=10_000)
        assert stats["method"] == "flat" and stats["k1"] == 0

    def test_select_frames_two_level_too_few_frames_raises(self, embeddings):
        with pytest.raises(ValueError, match="at least"):
            select_frames(embeddings[:50], k=20, method="two_level")

    @pytest.mark.parametrize("method", ["flat", "two_level"])
    def test_select_frames_skip_first_never_selected(self, embeddings, method):
        # make frame 0 a lone outlier that k-means would otherwise keep
        embeddings = embeddings.copy()
        embeddings[0] = -embeddings[0]
        idx, _, stats = select_frames(
            embeddings, k=20, seed=1, method=method, skip_first=1, k1_factor=3, k2=6
        )
        assert 0 not in idx and idx.max() < 300
        assert stats["skip_first"] == 1 and stats["n_frames"] == 300

    def test_select_frames_skip_first_whole_video(self, embeddings):
        idx, size, stats = select_frames(embeddings[:10], k=10, skip_first=2)
        assert idx.tolist() == list(range(2, 10)) and len(size) == 8
        assert stats["method"] == "whole"

    def test_select_frames_coverage_stats(self, embeddings):
        _, _, stats = select_frames(embeddings, k=3, seed=0)
        assert 0 <= stats["coverage_mean"] <= stats["coverage_p95"] <= stats["coverage_max"]
        assert stats["coverage_max"] < 0.01  # groups are tight


class TestSaveLoadSelection:
    """Test the functions save_selection and load_selection."""

    def test_save_load_selection_round_trip(self, tmp_path):
        path = tmp_path / "m" / "ds" / "key.npz"
        idx = np.array([1, 5, 9], dtype=np.int32)
        size = np.array([4, 2, 1], dtype=np.int32)
        stats = {"n_frames": 10, "k": 3, "coverage_mean": 0.5, "method": "two_level"}
        save_selection(path, idx, size, stats)
        idx2, size2, stats2 = load_selection(path)
        assert np.array_equal(idx, idx2) and np.array_equal(size, size2)
        assert stats2 == stats
        assert not list(path.parent.glob("*.tmp"))
