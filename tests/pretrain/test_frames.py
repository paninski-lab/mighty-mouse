"""Tests for mighty_mouse.pretrain.frames."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from mighty_mouse.pretrain.frames import (
    frame_path,
    is_junk,
    junk_counts,
    junk_stats,
    junk_violations,
    plan_extraction,
    read_frames,
    resize_min_side,
    save_frame,
)


def _video(path: Path, levels: list[int], size=(32, 32)) -> Path:
    """A video whose i-th frame is flat gray `levels[i]`."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for level in levels:
        writer.write(np.full((size[1], size[0], 3), level, dtype=np.uint8))
    writer.release()
    return path


class TestResizeMinSide:
    """Test the function resize_min_side."""

    def test_resize_min_side_landscape(self):
        out = resize_min_side(np.zeros((480, 640, 3), np.uint8), 256)
        assert out.shape == (256, 341, 3)

    def test_resize_min_side_portrait(self):
        out = resize_min_side(np.zeros((640, 480, 3), np.uint8), 256)
        assert out.shape == (341, 256, 3)

    def test_resize_min_side_never_upscales(self):
        frame = np.zeros((200, 300, 3), np.uint8)
        assert resize_min_side(frame, 256) is frame

    def test_resize_min_side_exact_size_unchanged(self):
        frame = np.zeros((256, 400, 3), np.uint8)
        assert resize_min_side(frame, 256).shape == (256, 400, 3)


class TestJunkStats:
    """Test the function junk_stats."""

    def test_junk_stats_flat_frame(self):
        std, saturated = junk_stats(np.full((8, 8, 3), 100, np.uint8))
        assert std == pytest.approx(0.0)
        assert saturated == 0.0

    def test_junk_stats_saturated_share(self):
        frame = np.zeros((10, 10, 3), np.uint8)
        frame[:3] = 255
        std, saturated = junk_stats(frame)
        assert saturated == pytest.approx(0.3)
        assert std > 10


class TestIsJunk:
    """Test the function is_junk."""

    def test_is_junk_low_std(self):
        assert is_junk(9.9, 0.0)

    def test_is_junk_std_at_threshold_is_kept(self):
        assert not is_junk(10.0, 0.0)

    def test_is_junk_mostly_saturated(self):
        assert is_junk(40.0, 0.51)

    def test_is_junk_saturated_at_threshold_is_kept(self):
        assert not is_junk(40.0, 0.5)

    def test_is_junk_normal_frame(self):
        assert not is_junk(35.0, 0.02)


class TestFramePath:
    """Test the function frame_path."""

    def test_frame_path_pads_index(self):
        assert frame_path(Path("/out"), "ibl", "v1", 123) == Path("/out/ibl/v1/img00000123.png")


class TestReadFrames:
    """Test the function read_frames."""

    def test_read_frames_returns_requested_in_order(self, tmp_path):
        video = _video(tmp_path / "v.mp4", [20, 60, 100, 140, 180, 220])
        got = list(read_frames(video, [4, 1, 1]))
        assert [i for i, _ in got] == [1, 4]
        assert got[0][1].mean() == pytest.approx(60, abs=6)
        assert got[1][1].mean() == pytest.approx(180, abs=6)

    def test_read_frames_seeks_across_large_gaps(self, tmp_path):
        video = _video(tmp_path / "v.mp4", [10 * i for i in range(20)])
        got = dict(read_frames(video, [2, 9, 18], min_gap=3))
        for i in (2, 9, 18):
            assert got[i].mean() == pytest.approx(10 * i, abs=6)

    def test_read_frames_falls_back_when_seek_is_ignored(self, tmp_path):
        class NoSeekCapture:
            """A capture whose `set` does nothing, so `get` keeps the old position."""

            def __init__(self, path):
                self.frames = [np.full((4, 4, 3), 10 * i, np.uint8) for i in range(20)]
                self.pos = 0

            def isOpened(self):
                return True

            def set(self, prop, value):
                pass

            def get(self, prop):
                return self.pos

            def grab(self):
                self.pos += 1
                return self.pos <= len(self.frames)

            def read(self):
                ok = self.pos < len(self.frames)
                self.pos += 1
                return ok, self.frames[self.pos - 1] if ok else None

            def release(self):
                pass

        with patch("mighty_mouse.pretrain.frames.cv2.VideoCapture", NoSeekCapture):
            got = dict(read_frames(tmp_path / "x.mp4", [2, 9, 18], min_gap=3))
        assert [int(got[i].mean()) for i in (2, 9, 18)] == [20, 90, 180]

    def test_read_frames_past_end_raises(self, tmp_path):
        video = _video(tmp_path / "v.mp4", [20, 60, 100])
        with pytest.raises(OSError, match="ends at frame"):
            list(read_frames(video, [1, 10]))

    def test_read_frames_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError, match="opening"):
            list(read_frames(tmp_path / "missing.mp4", [0]))


class TestSaveFrame:
    """Test the function save_frame."""

    def test_save_frame_writes_png_without_leftovers(self, tmp_path):
        path = tmp_path / "ds" / "v" / "img00000001.png"
        frame = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
        save_frame(path, frame)
        assert np.array_equal(cv2.imread(str(path)), frame)
        assert [p.name for p in path.parent.iterdir()] == [path.name]


def _selection(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dataset", "video_key", "frame_index"])


class TestPlanExtraction:
    """Test the function plan_extraction."""

    def test_plan_extraction_merges_methods_per_frame(self):
        plan = plan_extraction(
            {
                "A": _selection([("d", "v1", 5), ("d", "v1", 7)]),
                "B": _selection([("d", "v1", 5), ("e", "w", 1)]),
            },
            done={},
        )
        assert plan == {("d", "v1"): {5: ["A", "B"], 7: ["A"]}, ("e", "w"): {1: ["B"]}}

    def test_plan_extraction_skips_done_per_method(self):
        plan = plan_extraction(
            {"A": _selection([("d", "v1", 5)]), "B": _selection([("d", "v1", 5)])},
            done={"A": {("d", "v1", 5)}},
        )
        assert plan == {("d", "v1"): {5: ["B"]}}

    def test_plan_extraction_key_in_two_datasets_raises(self):
        with pytest.raises(ValueError, match="v1"):
            plan_extraction({"A": _selection([("d", "v1", 0), ("e", "v1", 0)])}, done={})


class TestJunkCounts:
    """Test the function junk_counts."""

    def test_junk_counts_per_dataset_and_overall(self):
        selected = _selection([("a", "v", i) for i in range(8)] + [("b", "w", 0), ("b", "w", 1)])
        dropped = pd.DataFrame({"dataset": ["a", "a"]})
        counts = junk_counts(selected, dropped)
        assert counts.loc["a", ["n_selected", "n_dropped"]].tolist() == [8, 2]
        assert counts.loc["b", ["n_selected", "n_dropped"]].tolist() == [2, 0]
        assert counts.loc["all", "junk_fraction"] == pytest.approx(0.2)

    def test_junk_counts_no_drops(self):
        counts = junk_counts(_selection([("a", "v", 0)]), pd.DataFrame({"dataset": []}))
        assert counts.loc["all", "n_dropped"] == 0


class TestJunkViolations:
    """Test the function junk_violations."""

    def _counts(self, a_dropped: int, b_dropped: int) -> pd.DataFrame:
        selected = _selection(
            [("a", "v", i) for i in range(1000)] + [("b", "w", i) for i in range(1000)]
        )
        dropped = pd.DataFrame({"dataset": ["a"] * a_dropped + ["b"] * b_dropped})
        return junk_counts(selected, dropped)

    def test_junk_violations_within_limits(self):
        assert junk_violations(self._counts(5, 4)) == []

    def test_junk_violations_dataset_over_limit(self):
        messages = junk_violations(self._counts(0, 30), max_overall=0.5)
        assert len(messages) == 1 and messages[0].startswith("b:")

    def test_junk_violations_overall_over_limit(self):
        messages = junk_violations(self._counts(10, 10))
        assert [m.split(":")[0] for m in messages] == ["all"]
