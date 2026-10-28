"""Tests for mighty_mouse.pretraining.video_list."""

from pathlib import Path

import pytest

from mighty_mouse.pretraining.video_list import list_videos


@pytest.fixture
def videos_paths() -> dict:
    return {
        "a": {
            "train": {"v1": {"path": "/x/v1.mp4", "frames": 10}, "v2": {"path": None}},
            "test": {"v3": {"path": "/x/v3.avi", "frames": 5}},
            "unlabeled": {},
        },
        "b": {"train": {"w1": {"path": "/y/w1.mp4", "frames": 7}}},
    }


class TestListVideos:
    """Test the function list_videos."""

    def test_list_videos_all_datasets(self, videos_paths):
        videos, skipped = list_videos(videos_paths)
        assert [(v.dataset, v.key) for v in videos] == [("a", "v1"), ("a", "v3"), ("b", "w1")]
        assert videos[0].path == Path("/x/v1.mp4")
        assert videos[0].n_frames == 10
        assert skipped == [("a", "v2")]

    def test_list_videos_single_dataset(self, videos_paths):
        videos, skipped = list_videos(videos_paths, dataset="b")
        assert [v.key for v in videos] == ["w1"]
        assert skipped == []

    def test_list_videos_unknown_dataset_raises(self, videos_paths):
        with pytest.raises(ValueError, match="unknown dataset"):
            list_videos(videos_paths, dataset="nope")

    def test_list_videos_same_file_stem_distinct_keys(self):
        vp = {"a": {"train": {"k1": {"path": "/x/v.mp4"}, "k2": {"path": "/y/v.mp4"}}}}
        videos, _ = list_videos(vp)
        assert [v.key for v in videos] == ["k1", "k2"]

    def test_list_videos_duplicate_key_across_splits_raises(self):
        vp = {"a": {"train": {"k": {"path": "/x/v.mp4"}}, "test": {"k": {"path": "/y/v.mp4"}}}}
        with pytest.raises(ValueError, match="more than one split"):
            list_videos(vp)
