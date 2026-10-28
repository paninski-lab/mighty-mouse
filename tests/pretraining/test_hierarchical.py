"""Tests for mighty_mouse.pretraining.hierarchical."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mighty_mouse.pretraining.hierarchical import (  # noqa: E402
    balanced_counts,
    hierarchical_select,
)
from mighty_mouse.pretraining.kmeans import normalize_rows  # noqa: E402


@pytest.fixture
def blobs():
    """One huge tight blob (2400 rows) and six small ones (100 rows each), unit-length."""
    rng = np.random.default_rng(0)
    centers = np.eye(7, 12, dtype=np.float32)
    sizes = [2400] + [100] * 6
    x = np.concatenate([
        c + 0.01 * rng.normal(size=(s, 12)) for c, s in zip(centers, sizes, strict=True)
    ]).astype(np.float32)
    labels = np.repeat(np.arange(7), sizes)
    return normalize_rows(torch.from_numpy(x)), labels


class TestBalancedCounts:
    """Test the function balanced_counts."""

    def test_balanced_counts_small_groups_give_everything(self):
        # 2 + 5 + 11 + 11 = 29 <= 30, but quota 12 would give 31
        assert balanced_counts(np.array([2, 5, 30, 60]), 30) == 11

    def test_balanced_counts_never_exceeds_target(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            sizes = rng.integers(1, 50, size=30)
            target = int(rng.integers(1, sizes.sum()))
            n = balanced_counts(sizes, target)
            assert np.minimum(n, sizes).sum() <= target
            assert np.minimum(n + 1, sizes).sum() > target or n == sizes.max()

    def test_balanced_counts_target_exceeds_total(self):
        assert balanced_counts(np.array([3, 4]), 100) == 4


class TestHierarchicalSelect:
    """Test the function hierarchical_select."""

    def test_hierarchical_select_count_unique_sorted(self, blobs):
        x, _ = blobs
        idx, size, stats = hierarchical_select(x, 60, 180, 12, seed=0)
        assert len(idx) == 60 and len(set(idx.tolist())) == 60
        assert np.all(np.diff(idx) > 0) and idx.max() < len(x)
        assert len(size) == 60 and size.min() >= 1
        assert stats["n_iter"] >= 1

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_hierarchical_select_balances_dense_blob(self, blobs, seed):
        x, labels = blobs
        idx, _, _ = hierarchical_select(x, 60, 180, 12, seed=seed)
        picked = labels[idx]
        # the big blob holds 80% of the frames but must get well under that share,
        # and every small blob must be represented
        assert (picked == 0).mean() < 0.5
        assert set(picked) == set(range(7))

    def test_hierarchical_select_deterministic(self, blobs):
        x, _ = blobs
        a = hierarchical_select(x, 60, 180, 12, seed=4)[0]
        b = hierarchical_select(x, 60, 180, 12, seed=4)[0]
        assert np.array_equal(a, b)

    def test_hierarchical_select_k1_too_small_raises(self, blobs):
        with pytest.raises(ValueError, match="k1 must be"):
            hierarchical_select(blobs[0], 60, 60, 12, seed=0)

    def test_hierarchical_select_k1_too_large_raises(self, blobs):
        with pytest.raises(ValueError, match="k1 must be"):
            hierarchical_select(blobs[0], 60, 10_000, 12, seed=0)
