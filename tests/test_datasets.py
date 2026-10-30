"""Tests for mighty_mouse.datasets."""

from pathlib import Path

import pytest

from mighty_mouse.datasets import ALL_DATASETS, SHORT_NAMES, tag_of


class TestAllDatasets:
    """Test the registry ALL_DATASETS."""

    def test_all_datasets_no_duplicates(self):
        assert len(ALL_DATASETS) == len(set(ALL_DATASETS))

    def test_all_datasets_sorted(self):
        assert ALL_DATASETS == sorted(ALL_DATASETS)

    def test_all_datasets_have_config(self):
        datasets_dir = Path(__file__).parents[1] / "configs" / "datasets"
        missing = [d for d in ALL_DATASETS if not (datasets_dir / f"{d}.yaml").exists()]
        assert missing == []


class TestShortNames:
    """Test the mapping SHORT_NAMES."""

    def test_short_names_cover_all_datasets(self):
        assert set(SHORT_NAMES) == set(ALL_DATASETS)

    def test_short_names_unique(self):
        assert len(set(SHORT_NAMES.values())) == len(SHORT_NAMES)


class TestTagOf:
    """Test the function tag_of."""

    def test_tag_of_fixed_order(self):
        assert tag_of(["cheese-2d", "facemap", "ibl"]) == "face+ibl+cheese"

    def test_tag_of_unknown_dataset(self):
        with pytest.raises(KeyError, match="SHORT_NAMES"):
            tag_of(["facemap", "new-lab"])
