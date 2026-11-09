"""Test the mighty_mouse.evaluation.qualitative.variance_rows module."""

import numpy as np
import pandas as pd

from mighty_mouse.evaluation.ensemble_stats import ensemble_stats
from mighty_mouse.evaluation.qualitative.variance_rows import (
    banner_line,
    canvas_size,
    draw_variance_panel,
    load_keypoint_map,
)


def member(x: float, conf: float) -> pd.DataFrame:
    """One member's prediction of keypoints a and b on one frame."""
    cols = pd.MultiIndex.from_tuples([("s", k, c) for k in "ab" for c in ("x", "y", "likelihood")])
    return pd.DataFrame([[x, 20.0, conf, 40.0, 20.0, conf]], index=["f.png"], columns=cols)


class TestLoadKeypointMap:
    """Test the function load_keypoint_map."""

    def test_load_keypoint_map_none_is_empty(self):
        assert load_keypoint_map(None) == {}

    def test_load_keypoint_map_plain_map(self, tmp_path):
        f = tmp_path / "map.yaml"
        f.write_text("wrist: paw_r\n")
        assert load_keypoint_map(f) == {"wrist": "paw_r"}

    def test_load_keypoint_map_dataset_config_keypoints_section(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("exclude: {}\nkeypoints:\n  wrist: paw_r\n")
        assert load_keypoint_map(f) == {"wrist": "paw_r"}


class TestCanvasSize:
    """Test the function canvas_size."""

    def test_canvas_size_panel_and_strip_per_view_row_per_mode(self):
        assert canvas_size(2, 100, 50, 3, False) == (300, 300)

    def test_canvas_size_median_panel_doubles_rows(self):
        assert canvas_size(2, 100, 50, 3, True) == (300, 600)


class TestBannerLine:
    """Test the function banner_line."""

    def test_banner_line_default(self):
        line = banner_line("T", "ibl", "pool frames", 3, 0.7, 1.0, 3, True, False)
        assert line == (
            "T | ibl pool frames | dot = median of all 3 models; shown if any model conf >= 0.7; "
            "ring = mean conf < 0.7 | rows differ ONLY in the +-1 SD cross (true length) | "
            "x = ground truth")

    def test_banner_line_variants(self):
        line = banner_line("T", "ibl", "s", 3, 0.0, 2.0, 1, False, True)
        assert "no confidence masking" in line and "cross = +-1 SD (true length x2)" in line
        assert "no ground truth drawn" in line and "lower panels" in line


class TestDrawVariancePanel:
    """Test the function draw_variance_panel."""

    def draw(self, conf=0.5, mean_conf=0.9):
        stats = ensemble_stats([member(10.0, mean_conf), member(14.0, mean_conf)], ["a", "b"], 0.5)
        base = np.zeros((100, 100, 3), np.uint8)
        out = draw_variance_panel(base, 80, stats, 0, "ensemble var", ["a", "b"],
                                  lambda k: (0, 255, 0), (0, 0, 1.0), 1.0, conf)
        return base, out

    def test_draw_variance_panel_draws_each_shown_keypoint_on_a_copy(self):
        base, (panel, strip, n_drawn, mean_sd) = self.draw()
        assert not base.any()                          # the input panel is untouched
        assert panel.shape == (100, 100, 3) and panel.any()
        assert strip.shape == (100, 80, 3)
        assert n_drawn == 2
        assert mean_sd == np.mean([2.0, 0.0])          # SD of x in {10, 14} is 2; y is constant

    def test_draw_variance_panel_hidden_keypoints_are_skipped(self):
        stats = ensemble_stats([member(10.0, 0.1), member(14.0, 0.1)], ["a", "b"], 0.5)
        _, _, n_drawn, mean_sd = draw_variance_panel(
            np.zeros((100, 100, 3), np.uint8), 80, stats, 0, "mixture", ["a", "b"],
            lambda k: (0, 255, 0), (0, 0, 1.0), 1.0, 0.5)
        assert n_drawn == 0 and mean_sd == 0
