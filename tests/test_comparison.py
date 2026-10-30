"""Test the mighty_mouse.comparison module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mighty_mouse.comparison import (
    ALWAYS_EXCLUDED,
    comparison_summary,
    trained_keypoints,
    visible_errors,
)


def frame(values: dict[str, list[float]], visible: float = 2.0) -> pd.DataFrame:
    """Two frames; each keypoint at (v, v) with the given visible flag."""
    cols = pd.MultiIndex.from_tuples([("s", k, c) for k in values for c in ("x", "y", "visible")])
    rows = [sum(([v, v, visible] for v in row), []) for row in zip(*values.values(), strict=True)]
    return pd.DataFrame(rows, index=["a.png", "b.png"], columns=cols)


class TestVisibleErrors:
    """Test the function visible_errors."""

    def test_visible_errors_masks_tongue_and_lips(self):
        labels = frame({"nose": [0.0, 0.0], "tongue_tip": [0.0, 0.0], "lowerlip": [0.0, 0.0]})
        preds  = frame({"nose": [3.0, 4.0], "tongue_tip": [30.0, 40.0], "lowerlip": [5.0, 6.0]})
        err = visible_errors(labels, preds)
        np.testing.assert_allclose(err["nose"], [3 * np.sqrt(2), 4 * np.sqrt(2)])
        assert err["tongue_tip"].isna().all()
        assert err["lowerlip"].isna().all()

    def test_visible_errors_custom_exclude(self):
        labels = frame({"nose": [0.0, 0.0], "tongue_tip": [0.0, 0.0]})
        preds  = frame({"nose": [1.0, 1.0], "tongue_tip": [2.0, 2.0]})
        err = visible_errors(labels, preds, exclude=frozenset({"nose"}))
        assert err["nose"].isna().all()
        np.testing.assert_allclose(err["tongue_tip"], [2 * np.sqrt(2)] * 2)

    def test_visible_errors_skips_occluded(self):
        err = visible_errors(frame({"nose": [0.0, 0.0]}, visible=1.0), frame({"nose": [3.0, 4.0]}))
        assert err["nose"].isna().all()

    def test_visible_errors_excludes_every_tongue_and_lip_keypoint_of_the_vocabulary(self):
        path  = Path(__file__).parents[1] / "configs" / "keypoints.yaml"
        vocab = yaml.safe_load(path.read_text())
        for part in ("tongue", "lip"):
            names = {k for k in vocab["keypoints"] if part in k}
            assert names, f"no {part} keypoints found in configs/keypoints.yaml"
            assert names <= ALWAYS_EXCLUDED

    def test_visible_errors_rejects_mismatched_frames(self):
        labels = frame({"nose": [0.0, 0.0]})
        preds  = frame({"nose": [0.0, 0.0]}).rename(index={"b.png": "c.png"})
        with pytest.raises(ValueError, match="differ"):
            visible_errors(labels, preds)


class TestComparisonSummary:
    """Test the function comparison_summary."""

    def test_comparison_summary_pooled_and_normalized(self):
        errors = pd.DataFrame({"nose": [2.0, 4.0], "ear": [6.0, np.nan]})
        s = comparison_summary(errors, np.array([10.0, 20.0]))
        assert s["points"] == 3
        assert s["mean_px"] == pytest.approx(4.0)
        assert s["median_px"] == pytest.approx(4.0)
        assert s["normalized_mean"] == pytest.approx((0.2 + 0.2 + 0.6) / 3)
        assert s["equal_keypoint_normalized_mean"] == pytest.approx((0.2 + 0.6) / 2)

    def test_comparison_summary_nothing_scored(self):
        s = comparison_summary(pd.DataFrame({"nose": [np.nan]}), np.array([10.0]))
        assert s["points"] == 0
        assert s["mean_px"] is None

    def test_comparison_summary_rejects_bad_diagonals(self):
        with pytest.raises(ValueError, match="diagonal"):
            comparison_summary(pd.DataFrame({"nose": [1.0, 2.0]}), np.array([10.0]))


class TestTrainedKeypoints:
    """Test the function trained_keypoints."""

    def test_trained_keypoints_union(self):
        inv = {"a": {"trainable": ["nose", "ear"]}, "b": {"trainable": ["tongue"]}}
        assert trained_keypoints({"a", "b"}, inv) == {"nose", "ear", "tongue"}

    def test_trained_keypoints_ignores_unknown_dataset(self):
        assert trained_keypoints({"a", "missing"}, {"a": {"trainable": ["nose"]}}) == {"nose"}
