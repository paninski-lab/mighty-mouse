"""Tests for mighty_mouse.pretrain.selection."""

import numpy as np
import pytest

from mighty_mouse.pretrain.selection import hier_sizes, select_pool


class TestHierSizes:
    """Test the function hier_sizes."""

    def test_hier_sizes_three_times_the_selection(self):
        assert hier_sizes(100_000, 1000) == (3000, 100)

    def test_hier_sizes_limited_to_share_of_pool(self):
        k1, _ = hier_sizes(1000, 500)
        assert k1 == 900

    def test_hier_sizes_always_above_selection_and_within_pool(self):
        k1, _ = hier_sizes(1000, 980)
        assert 980 < k1 <= 1000

    def test_hier_sizes_minimum_groups(self):
        assert hier_sizes(10_000, 100)[1] == 50

    def test_hier_sizes_selecting_everything_raises(self):
        with pytest.raises(ValueError, match="n_select"):
            hier_sizes(100, 100)


class TestSelectPool:
    """Test the function select_pool."""

    def test_select_pool_all_rows_when_budget_covers_pool(self):
        x = np.zeros((10, 4))
        assert select_pool(x, 10, "hier", 0).tolist() == list(range(10))
        assert select_pool(x, 50, "fps", 0).tolist() == list(range(10))

    def test_select_pool_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            select_pool(np.zeros((10, 4)), 5, "random", 0)

    def test_select_pool_nonpositive_budget_raises(self):
        with pytest.raises(ValueError, match="positive"):
            select_pool(np.zeros((10, 4)), 0, "hier", 0)

    @pytest.mark.parametrize("method", ["hier", "fps"])
    def test_select_pool_returns_sorted_distinct_rows(self, method):
        torch = pytest.importorskip("torch")
        from mighty_mouse.pretrain.kmeans import normalize_rows

        x = normalize_rows(torch.randn(400, 8))
        idx = select_pool(x, 40, method, 0)
        assert 0 < len(idx) <= 40
        assert len(set(idx.tolist())) == len(idx) and (np.diff(idx) > 0).all()

    def test_select_pool_fps_keeps_the_rare_group(self):
        torch = pytest.importorskip("torch")
        from mighty_mouse.pretrain.kmeans import normalize_rows

        rng = np.random.default_rng(0)
        big = np.eye(3, 8)[0] + 0.01 * rng.normal(size=(500, 8))
        rare = np.eye(3, 8)[1] + 0.01 * rng.normal(size=(5, 8))
        x = normalize_rows(torch.from_numpy(np.vstack([big, rare])).float())
        assert (select_pool(x, 5, "fps", 0) >= 500).any()
