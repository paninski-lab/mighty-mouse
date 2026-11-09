"""Test the mighty_mouse.evaluation.run module."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from mighty_mouse.evaluation.run import clean_run_dir, image_diagonals, score_predictions


def frame(values: dict[str, list[float]], index: list[str], visible: float = 2.0) -> pd.DataFrame:
    """One row per index entry; each keypoint at (v, v) with the given visible flag."""
    cols = pd.MultiIndex.from_tuples([("s", k, c) for k in values for c in ("x", "y", "visible")])
    rows = [sum(([v, v, visible] for v in row), []) for row in zip(*values.values(), strict=True)]
    return pd.DataFrame(rows, index=index, columns=cols)


class TestImageDiagonals:
    """Test the function image_diagonals."""

    def test_image_diagonals_reads_each_image_size(self, tmp_path):
        Image.new("RGB", (3, 4)).save(tmp_path / "a.png")
        Image.new("RGB", (6, 8)).save(tmp_path / "b.png")
        diagonals = image_diagonals(pd.Index(["a.png", "b.png"]), tmp_path)
        np.testing.assert_allclose(diagonals, [5, 10])

    def test_image_diagonals_missing_image_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            image_diagonals(pd.Index(["nope.png"]), tmp_path)


class TestScorePredictions:
    """Test the function score_predictions."""

    def test_score_predictions_errors_and_summary(self):
        labels = frame({"nose": [0.0, 0.0]}, ["a.png", "b.png"])
        preds = frame({"nose": [3.0, 4.0]}, ["a.png", "b.png"])
        _, errors, summary = score_predictions(labels, preds, np.array([10.0, 10.0]))
        np.testing.assert_allclose(errors["nose"], [3 * np.sqrt(2), 4 * np.sqrt(2)])
        assert summary["points"] == 2
        assert summary["mean_px"] == pytest.approx(3.5 * np.sqrt(2))

    def test_score_predictions_returns_predictions_in_label_order(self):
        labels = frame({"nose": [0.0, 0.0]}, ["a.png", "b.png"])
        preds = frame({"nose": [4.0, 3.0]}, ["b.png", "a.png"])
        out, _, _ = score_predictions(labels, preds, np.array([10.0, 10.0]))
        assert list(out.index) == ["a.png", "b.png"]

    def test_score_predictions_no_labeled_cell(self):
        labels = frame({"nose": [0.0, 0.0]}, ["a.png", "b.png"], visible=1.0)
        preds = frame({"nose": [3.0, 4.0]}, ["a.png", "b.png"])
        _, _, summary = score_predictions(labels, preds, np.array([10.0, 10.0]))
        assert summary["points"] == 0
        assert summary["mean_px"] is None

    def test_score_predictions_rejects_bad_diagonals(self):
        labels = frame({"nose": [0.0, 0.0]}, ["a.png", "b.png"])
        with pytest.raises(ValueError, match="diagonal"):
            score_predictions(labels, labels, np.array([10.0]))


class TestCleanRunDir:
    """Test the function clean_run_dir."""

    @staticmethod
    def make_run(tmp_path):
        (tmp_path / "predictions.csv").write_text("x")
        (tmp_path / "predictions_pixel_error.csv").write_text("x")
        (tmp_path / "image_preds" / "ibl").mkdir(parents=True)
        (tmp_path / "tb_logs" / "v0").mkdir(parents=True)
        (tmp_path / "tb_logs" / "v0" / "best.ckpt").write_text("x")
        (tmp_path / "eval" / "ibl").mkdir(parents=True)
        (tmp_path / "eval" / "ibl" / "pixel_error.csv").write_text("x")

    def test_clean_run_dir_removes_scratch_and_checkpoints(self, tmp_path):
        self.make_run(tmp_path)
        clean_run_dir(tmp_path)
        assert not (tmp_path / "predictions.csv").exists()
        assert not (tmp_path / "predictions_pixel_error.csv").exists()
        assert not (tmp_path / "image_preds").exists()
        assert not list(tmp_path.rglob("*.ckpt"))
        assert (tmp_path / "eval" / "ibl" / "pixel_error.csv").exists()

    def test_clean_run_dir_keep_checkpoints(self, tmp_path):
        self.make_run(tmp_path)
        clean_run_dir(tmp_path, keep_checkpoints=True)
        assert (tmp_path / "tb_logs" / "v0" / "best.ckpt").exists()
        assert not (tmp_path / "image_preds").exists()
