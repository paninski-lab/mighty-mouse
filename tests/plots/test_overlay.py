"""Test the mighty_mouse.plots.overlay module."""

import numpy as np
import pandas as pd

from mighty_mouse.plots.overlay import (
    draw_ground_truth,
    frame_value,
    labeled_point,
    predicted_point,
    to_panel,
)


def red(kp: str) -> tuple[int, int, int]:
    return (0, 0, 255)


def labels() -> pd.DataFrame:
    """Two frames; nose labeled in a.png (10, 20), occluded in b.png; ear unlabeled."""
    cols = pd.MultiIndex.from_tuples(
        [("s", k, c) for k in ("nose", "ear") for c in ("x", "y", "visible")])
    rows = [[10.0, 20.0, 2.0, np.nan, np.nan, 0.0], [30.0, 40.0, 1.0, np.nan, np.nan, 0.0]]
    return pd.DataFrame(rows, index=["a.png", "b.png"], columns=cols)


def predictions() -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples(
        [("s", k, c) for k in ("nose", "ear") for c in ("x", "y", "likelihood")])
    rows = [[11.0, 21.0, 0.9, 5.0, 6.0, 0.2], [31.0, 41.0, 0.8, np.nan, 6.0, 0.9]]
    return pd.DataFrame(rows, index=["a.png", "b.png"], columns=cols)


class TestToPanel:
    """Test the function to_panel."""

    def test_to_panel_offset_then_scale(self):
        assert to_panel(10, 20, 5, 0, 0.5) == (7, 10)

    def test_to_panel_truncates_to_int(self):
        assert to_panel(1.9, 1.9, 0, 0, 1.0) == (1, 1)


class TestFrameValue:
    """Test the function frame_value."""

    def test_frame_value_reads_cell(self):
        assert frame_value(predictions(), "a.png", "nose", "x") == 11.0

    def test_frame_value_missing_is_nan(self):
        assert np.isnan(frame_value(predictions(), "nope.png", "nose", "x"))
        assert np.isnan(frame_value(predictions(), "a.png", "tail", "x"))


class TestPredictedPoint:
    """Test the function predicted_point."""

    def test_predicted_point_above_conf(self):
        assert predicted_point(predictions(), "a.png", "nose", 0.7) == (11.0, 21.0)

    def test_predicted_point_below_conf_is_none(self):
        assert predicted_point(predictions(), "a.png", "ear", 0.7) is None

    def test_predicted_point_nan_coordinate_is_none(self):
        assert predicted_point(predictions(), "b.png", "ear", 0.7) is None

    def test_predicted_point_missing_keypoint_is_none(self):
        assert predicted_point(predictions(), "a.png", "tail", 0.0) is None


class TestLabeledPoint:
    """Test the function labeled_point."""

    def test_labeled_point_visible_two(self):
        assert labeled_point(labels(), "a.png", "nose") == (10.0, 20.0)

    def test_labeled_point_occluded_or_unlabeled_is_none(self):
        assert labeled_point(labels(), "b.png", "nose") is None
        assert labeled_point(labels(), "a.png", "ear") is None


class TestDrawGroundTruth:
    """Test the function draw_ground_truth."""

    def test_draw_ground_truth_marks_only_labeled_keypoints(self):
        panel = np.zeros((50, 50, 3), np.uint8)
        draw_ground_truth(panel, labels(), "a.png", ["nose", "ear"], (0, 0, 1.0), red)
        assert panel.any()
        ys, xs = np.nonzero(panel.any(-1))
        assert abs(xs.mean() - 10) <= 6 and abs(ys.mean() - 20) <= 6

    def test_draw_ground_truth_nothing_labeled_leaves_panel(self):
        panel = np.zeros((50, 50, 3), np.uint8)
        draw_ground_truth(panel, labels(), "b.png", ["nose", "ear"], (0, 0, 1.0), red)
        assert not panel.any()
