"""Tests for mighty_mouse.pretraining.kmeans."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mighty_mouse.pretraining.kmeans import (  # noqa: E402
    assign_to_centroids,
    normalize_rows,
    spherical_kmeans,
)


@pytest.fixture
def blobs():
    """Three well-separated groups of unit vectors, 50 points each."""
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8, dtype=np.float32)
    x = np.concatenate([c + 0.05 * rng.normal(size=(50, 8)) for c in centers]).astype(np.float32)
    labels = np.repeat(np.arange(3), 50)
    return normalize_rows(torch.from_numpy(x)), labels


class TestNormalizeRows:
    """Test the function normalize_rows."""

    def test_normalize_rows_unit_length(self):
        x = normalize_rows(torch.randn(10, 4) * 5)
        assert torch.allclose(x.norm(dim=1), torch.ones(10), atol=1e-5)

    def test_normalize_rows_zero_row_stays_finite(self):
        x = normalize_rows(torch.zeros(2, 4))
        assert torch.isfinite(x).all()


class TestAssignToCentroids:
    """Test the function assign_to_centroids."""

    def test_assign_to_centroids_picks_most_similar(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        assign, best = assign_to_centroids(x, torch.tensor([[0.0, 1.0], [1.0, 0.0]]), chunk=1)
        assert assign.tolist() == [1, 0]
        assert torch.allclose(best, torch.ones(2))


class TestSphericalKmeans:
    """Test the function spherical_kmeans."""

    @pytest.mark.parametrize("init", ["kmeans++", "random"])
    def test_spherical_kmeans_recovers_blobs(self, blobs, init):
        x, labels = blobs
        _, assign, _ = spherical_kmeans(x, 3, seed=0, init=init)
        assign = assign.numpy()
        # each true group lands in a single cluster, and the three clusters differ
        groups = [set(assign[labels == i]) for i in range(3)]
        assert all(len(g) == 1 for g in groups)
        assert len(set.union(*groups)) == 3

    def test_spherical_kmeans_shapes_and_unit_centroids(self, blobs):
        x, _ = blobs
        centroids, assign, n_run = spherical_kmeans(x, 5, seed=1)
        assert centroids.shape == (5, 8)
        assert assign.shape == (150,)
        assert torch.allclose(centroids.norm(dim=1), torch.ones(5), atol=1e-5)
        assert n_run >= 1

    def test_spherical_kmeans_k_equals_n(self):
        x = normalize_rows(torch.randn(6, 4))
        _, assign, _ = spherical_kmeans(x, 6, seed=0)
        assert len(set(assign.tolist())) == 6

    def test_spherical_kmeans_deterministic(self, blobs):
        x, _ = blobs
        a = spherical_kmeans(x, 4, seed=3)[1]
        b = spherical_kmeans(x, 4, seed=3)[1]
        assert torch.equal(a, b)

    def test_spherical_kmeans_bad_k_raises(self, blobs):
        with pytest.raises(ValueError, match="k must be"):
            spherical_kmeans(blobs[0], 151)

    def test_spherical_kmeans_bad_init_raises(self, blobs):
        with pytest.raises(ValueError, match="unknown init"):
            spherical_kmeans(blobs[0], 3, init="nope")
