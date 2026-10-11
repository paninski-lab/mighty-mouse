"""Test the mighty_mouse.labeling.ensemble module."""

import numpy as np
import pytest

from mighty_mouse.labeling.ensemble import ensemble
from tests.labeling.conftest import label_frame, pred_frame


class TestEnsemble:
    """Test the function ensemble."""

    def test_ensemble_mean_of_surviving_members(self):
        imgs = ["a.png", "b.png"]
        orig = label_frame(imgs)
        preds = [
            pred_frame(imgs, {"ear_a": [(0, 0, 0.95), (1, 1, 0.5)]}),
            pred_frame(imgs, {"ear_a": [(2, 4, 0.95), (3, 3, 0.95)]}),
        ]

        new, n_surv = ensemble(orig, preds, [("ear_a", "ear_a")], 0.9)

        assert new[("All", "ear_a", "x")].tolist() == [1.0, 3.0]
        assert new[("All", "ear_a", "y")].tolist() == [2.0, 3.0]
        assert n_surv["ear_a"].tolist() == [2, 1]

    def test_ensemble_median_ignores_one_outlier_member(self):
        orig = label_frame(["a.png"])
        preds = [
            pred_frame(["a.png"], {"ear_a": [(1, 1, 0.95)]}),
            pred_frame(["a.png"], {"ear_a": [(2, 2, 0.95)]}),
            pred_frame(["a.png"], {"ear_a": [(30, 30, 0.95)]}),
        ]

        new, _ = ensemble(orig, preds, [("ear_a", "ear_a")], 0.9, agg="median")

        assert new[("All", "ear_a", "x")].tolist() == [2.0]

    def test_ensemble_median_blank_when_no_member_survives(self):
        orig = label_frame(["a.png"])
        preds = [pred_frame(["a.png"], {"ear_a": [(1, 1, 0.1)]})]

        new, _ = ensemble(orig, preds, [("ear_a", "ear_a")], 0.9, agg="median")

        assert np.isnan(new[("All", "ear_a", "x")]).all()

    def test_ensemble_rejects_unknown_agg(self):
        orig = label_frame(["a.png"])
        preds = [pred_frame(["a.png"], {"ear_a": [(1, 1, 0.99)]})]

        with pytest.raises(ValueError, match="agg"):
            ensemble(orig, preds, [("ear_a", "ear_a")], 0.9, agg="max")

    def test_ensemble_blank_when_no_member_survives(self):
        orig = label_frame(["a.png"])
        preds = [pred_frame(["a.png"], {"ear_a": [(1, 1, 0.1)]})]

        new, n_surv = ensemble(orig, preds, [("ear_a", "ear_a")], 0.9)

        assert np.isnan(new[("All", "ear_a", "x")]).all()
        assert n_surv["ear_a"].tolist() == [0]

    def test_ensemble_leaves_other_columns_alone(self):
        orig = label_frame(["a.png"])
        preds = [pred_frame(["a.png"], {"ear_a": [(1, 1, 0.99)]})]

        new, _ = ensemble(orig, preds, [("ear_a", "ear_a")], 0.9)

        assert new[("All", "nose", "x")].tolist() == [1.0]
        assert np.isnan(new[("All", "ear_b", "x")]).all()
