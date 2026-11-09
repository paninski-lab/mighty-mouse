"""Tests for mighty_mouse.pretrain.layout."""

from pathlib import Path

import pytest

from mighty_mouse.pretrain.layout import (
    dedup_dir,
    embeddings_dir,
    pretrain_dir,
    subset_dir,
    videos_paths_file,
)


class TestPretrainingDir:
    """Test the function pretrain_dir."""

    def test_pretraining_dir_reads_entry(self):
        assert pretrain_dir({"pretrain_dir": "/data/videos"}) == Path("/data/videos")

    def test_pretraining_dir_missing_raises(self):
        with pytest.raises(KeyError, match="pretrain_dir"):
            pretrain_dir({"raw_dir": "/x"})


class TestLayout:
    """Test the fixed layout under the pretraining dir."""

    def test_layout_is_relative_to_root(self):
        root = Path("/data/videos")
        assert videos_paths_file(root) == root / "paths.yaml"
        assert embeddings_dir(root) == root / "embeddings"
        assert dedup_dir(root) == root / "dedup"


class TestSubsetDir:
    """Test the function subset_dir."""

    def test_subset_dir_joins_size_and_method(self):
        root = Path("/data/videos")
        assert subset_dir(root, "100k", "A1") == root / "pretrain_datasets" / "100k_methodA1"
