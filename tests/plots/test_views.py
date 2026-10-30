"""Test the mighty_mouse.plots.views module."""

from mighty_mouse.plots.views import view_of


class TestViewOf:
    """Test the function view_of."""

    def test_view_of_ibl_left_and_flipped_right(self):
        assert view_of("ibl", "6c6983ef-7383-4989-9183-32b1a300d17a_left") == "left"
        assert view_of("ibl", "6c6983ef-7383-4989-9183-32b1a300d17a_right_flipped") == "right"

    def test_view_of_ibl_legacy_sessions(self):
        assert view_of("ibl", "x_iblrig_leftCamera.paws_downsampled_20min") == "left"
        assert view_of("ibl", "x_iblrig_rightCamera.paws_downsampled.flipped_20min") == "right"

    def test_view_of_ibl_derived_set(self):
        assert view_of("ibl-digits", "a_left") == "left"

    def test_view_of_cheese_camera(self):
        assert view_of("cheese-3d", "20231031_B20_chew_bl_000_08-38-25_BC") == "BC"
        assert view_of("cheese-2d", "20250518_B1_ephys-record_awake_002_TL_x") == "TL"

    def test_view_of_kaufman_cameras_kept_apart(self):
        assert view_of("kaufman", "20241208-125516-cam1") == "cam1"
        assert view_of("kaufman-water", "20200820-143216-cam2") == "cam2"

    def test_view_of_front_side(self):
        assert view_of("hantman-mv", "KPC108_20241231_v029_front") == "front"
        assert view_of("tyssowski", "M399_20240528_v004_side") == "side"

    def test_view_of_cavelli_suffix(self):
        assert view_of("cavelli", "M1_2026-06-18_16-25-09_000_right") == "right"

    def test_view_of_single_view_dataset(self):
        assert view_of("kondo", "anything") == "bottom"

    def test_view_of_unknown_dataset_falls_back(self):
        assert view_of("new-lab", "s1-cam3") == "cam3"
        assert view_of("new-lab", "s1") == "camera"
