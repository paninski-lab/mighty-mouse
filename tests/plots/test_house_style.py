"""Test the mighty_mouse.plots.house_style module."""

from pathlib import Path

import numpy as np

from mighty_mouse.plots.house_style import GROUPS, color, csv_dir, fit, square, video_path


class TestColor:
    """Test the function color."""

    def test_color_group_hue(self):
        assert color("ear_tip_left") == dict((n, c) for n, c, _ in GROUPS)["ear"]

    def test_color_digit_tips_split_by_side(self):
        assert color("d1_tip_right") != color("d1_tip_left")

    def test_color_unknown_keypoint_grey(self):
        assert color("tail") == (200, 200, 200)


class TestFit:
    """Test the function fit."""

    def test_fit_short_text_keeps_size(self):
        assert fit("ab", 500, 0.5) == 0.5

    def test_fit_long_text_shrinks_to_floor(self):
        assert fit("x" * 500, 50, 0.5) < 0.3


class TestSquare:
    """Test the function square."""

    def test_square_letterbox_offsets_and_scale(self):
        img = np.full((100, 200, 3), 255, np.uint8)

        panel, ox, oy, s = square(img, 50)

        assert panel.shape == (50, 50, 3)
        assert (ox, oy, s) == (0, 50, 0.25)
        assert panel[0, 25].sum() == 0 and panel[25, 25].sum() > 0


class TestVideoPath:
    """Test the functions video_path and csv_dir."""

    def test_video_path_goes_into_videos(self, tmp_path: Path):
        out = video_path(tmp_path / "x.mp4")

        assert out == tmp_path / "videos" / "x.mp4"
        assert out.parent.is_dir()

    def test_video_path_keeps_videos_folder(self, tmp_path: Path):
        assert video_path(tmp_path / "videos" / "x.mp4") == tmp_path / "videos" / "x.mp4"

    def test_csv_dir_is_sibling_of_videos(self, tmp_path: Path):
        assert csv_dir(video_path(tmp_path / "x.mp4")) == tmp_path / "csv"
