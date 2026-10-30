"""Tests for mighty_mouse.pretrain.embed."""

from pathlib import Path

import numpy as np
import pytest

from mighty_mouse.pretrain.embed import output_path, preprocess_frames


class TestOutputPath:
    """Test the function output_path."""

    def test_output_path_layout(self):
        p = output_path(Path("/e"), "dinov3_vits16", "ibl", "vid-key")
        assert p == Path("/e/dinov3_vits16/ibl/vid-key.npy")


class TestPreprocessFrames:
    """Test the function preprocess_frames."""

    def test_preprocess_frames_shape_and_dtype(self):
        frames = np.random.randint(0, 255, (3, 40, 60, 3), dtype=np.uint8)
        out = preprocess_frames(frames, 32)
        assert out.shape == (3, 3, 32, 32)
        assert out.dtype == np.float32

    def test_preprocess_frames_normalizes(self):
        frames = np.full((1, 16, 16, 3), 255, dtype=np.uint8)
        out = preprocess_frames(frames, 16)
        np.testing.assert_allclose(out[0, 0], (1 - 0.485) / 0.229, rtol=1e-5)

    def test_preprocess_frames_bad_resolution_raises(self):
        with pytest.raises(ValueError, match="multiple of 16"):
            preprocess_frames(np.zeros((1, 8, 8, 3), dtype=np.uint8), 20)
