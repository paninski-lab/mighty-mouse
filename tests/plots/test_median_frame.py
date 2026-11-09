"""Test the mighty_mouse.plots.median_frame module."""

import cv2
import numpy as np

from mighty_mouse.plots.median_frame import median_subtracted, session_median


def write_frames(folder, values):
    folder.mkdir(exist_ok=True)
    for i, v in enumerate(values):
        cv2.imwrite(str(folder / f"img{i:03d}.png"), np.full((6, 8, 3), v, np.uint8))


class TestSessionMedian:
    """Test the function session_median."""

    def test_session_median_of_every_frame(self, tmp_path):
        write_frames(tmp_path, [10, 20, 90])
        med, n = session_median(tmp_path, 0, {})
        assert n == 3
        assert med.shape == (6, 8, 3)
        assert (med == 20).all()

    def test_session_median_n_evenly_spaced_frames(self, tmp_path):
        write_frames(tmp_path, [10, 20, 90])
        med, n = session_median(tmp_path, 2, {})        # first and last frame
        assert n == 2
        assert (med == 50).all()

    def test_session_median_cached_per_folder(self, tmp_path):
        write_frames(tmp_path, [10, 20, 90])
        cache = {}
        first = session_median(tmp_path, 0, cache)
        for f in tmp_path.glob("*.png"):
            f.unlink()
        assert session_median(tmp_path, 0, cache) is first

    def test_session_median_keeps_a_few_sessions(self, tmp_path):
        cache = {}
        for i in range(10):
            write_frames(tmp_path / f"s{i}", [10, 20])
            session_median(tmp_path / f"s{i}", 0, cache)
        assert len(cache) == 8
        assert tmp_path / "s0" not in cache


class TestMedianSubtracted:
    """Test the function median_subtracted."""

    def test_median_subtracted_unchanged_frame_is_gray(self):
        img = np.full((6, 8, 3), 100, np.uint8)
        out = median_subtracted(img, img.astype(np.float32))
        assert out.shape == img.shape and out.dtype == np.uint8
        assert (out == 128).all()

    def test_median_subtracted_removes_global_brightness_offset(self):
        med = np.full((6, 8, 3), 50, np.float32)
        out = median_subtracted(np.full((6, 8, 3), 90, np.uint8), med)
        assert (out == 128).all()

    def test_median_subtracted_moving_object_stands_out(self):
        med = np.full((6, 8, 3), 50, np.float32)
        img = np.full((6, 8, 3), 50, np.uint8)
        img[2:4, 2:4] = 200
        out = median_subtracted(img, med)
        assert out[3, 3, 0] > 200 and out[0, 0, 0] == 128

    def test_median_subtracted_resizes_median_to_frame(self):
        img = np.full((6, 8, 3), 100, np.uint8)
        out = median_subtracted(img, np.full((3, 4, 3), 100, np.float32))
        assert out.shape == (6, 8, 3)
