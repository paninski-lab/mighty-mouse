"""Tests for mighty_mouse.datasets."""

from pathlib import Path

from mighty_mouse.datasets import ALL_DATASETS


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
