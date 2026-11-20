"""Tests for mighty_mouse.pretrain.farthest_point."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mighty_mouse.pretrain.farthest_point import farthest_point_select  # noqa: E402
from mighty_mouse.pretrain.kmeans import normalize_rows  # noqa: E402


class TestFarthestPointSelect:
    """Test the function farthest_point_select."""

    def test_farthest_point_select_picks_one_per_separated_group(self):
        centers = torch.eye(4, 8)
        x = normalize_rows(centers.repeat_interleave(25, dim=0) + 0.01 * torch.randn(100, 8))
        idx, _ = farthest_point_select(x, 4, seed=0)
        assert sorted(idx // 25) == [0, 1, 2, 3]

    def test_farthest_point_select_keeps_isolated_row(self):
        rng = np.random.default_rng(0)
        dense = np.eye(3, 8)[0] + 0.01 * rng.normal(size=(200, 8))
        x = normalize_rows(torch.from_numpy(np.vstack([dense, np.eye(3, 8)[2:3]])).float())
        idx, _ = farthest_point_select(x, 2, seed=1)
        assert 200 in idx

    def test_farthest_point_select_sorted_unique(self):
        x = normalize_rows(torch.randn(60, 8))
        idx, stats = farthest_point_select(x, 20, seed=0)
        assert len(set(idx.tolist())) == 20 and (np.diff(idx) > 0).all()
        assert stats["eps_last"] > 0

    def test_farthest_point_select_all_rows(self):
        x = normalize_rows(torch.randn(10, 4))
        idx, stats = farthest_point_select(x, 10, seed=0)
        assert idx.tolist() == list(range(10))
        assert stats["eps_last"] == 0.0

    def test_farthest_point_select_is_seeded(self):
        x = normalize_rows(torch.randn(80, 8))
        assert np.array_equal(
            farthest_point_select(x, 10, seed=3)[0], farthest_point_select(x, 10, seed=3)[0]
        )

    def test_farthest_point_select_bad_count_raises(self):
        x = normalize_rows(torch.randn(10, 4))
        with pytest.raises(ValueError, match="n_select"):
            farthest_point_select(x, 11, seed=0)
