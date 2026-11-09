"""Test the mighty_mouse.evaluation.ensemble_stats module."""

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.evaluation.ensemble_stats import (
    MODES,
    ensemble_stats,
    labelled_keypoints,
    per_frame_keypoint_table,
    sd_column,
    select_modes,
    usable_keypoints,
    variance_summary,
)


def member(x, y, c, kp="nose", frames=("a.png",)) -> pd.DataFrame:
    """One member's prediction of a single keypoint on the given frames."""
    cols = pd.MultiIndex.from_tuples([("s", kp, k) for k in ("x", "y", "likelihood")])
    return pd.DataFrame(np.stack([x, y, c], 1), index=list(frames), columns=cols)


def ensemble(x, c):
    """Three members predicting one keypoint on one frame at (x_m, 0) with likelihood c_m."""
    return [member([xm], [0.0], [cm]) for xm, cm in zip(x, c, strict=True)]


class TestSelectModes:
    """Test the function select_modes."""

    def test_select_modes_default_all(self):
        assert select_modes() == MODES

    def test_select_modes_subset_keeps_canonical_order(self):
        assert [m for m, _ in select_modes("mixture, ensemble var")] == ["ensemble var", "mixture"]

    def test_select_modes_unknown_raises(self):
        with pytest.raises(ValueError, match="matches none"):
            select_modes("bogus")


class TestUsableKeypoints:
    """Test the function usable_keypoints."""

    @staticmethod
    def pred():
        cols = [("s", "nose", "x"), ("s", "nose", "y"), ("s", "nose", "likelihood"),
                ("s", "ear_l", "x"), ("s", "ear_l", "y"), ("s", "ear_l", "likelihood"),
                ("s", "ear_r", "x"), ("s", "ear_r", "y"), ("s", "ear_r", "likelihood"),
                ("s", "set", "x")]
        return pd.DataFrame([[0.0] * len(cols)], columns=pd.MultiIndex.from_tuples(cols))

    def test_usable_keypoints_requires_likelihood(self):
        assert usable_keypoints(self.pred()) == ["nose", "ear_l", "ear_r"]

    def test_usable_keypoints_include_and_exclude_regexes(self):
        assert usable_keypoints(self.pred(), include="ear_") == ["ear_l", "ear_r"]
        assert usable_keypoints(self.pred(), exclude="ear_r,nose") == ["ear_l"]
        assert usable_keypoints(self.pred(), exclude="ear_r", include="ear_") == ["ear_l"]


class TestEnsembleStats:
    """Test the function ensemble_stats."""

    def test_ensemble_stats_median_and_ensemble_var(self):
        stats = ensemble_stats(ensemble([0.0, 3.0, 6.0], [1.0, 1.0, 1.0]), ["nose"], 0.5)
        assert stats.x[0, 0] == 3.0
        assert stats.sd["ensemble var"][0][0, 0] == pytest.approx(np.sqrt(6.0))
        assert stats.sd["ensemble var"][1][0, 0] == 0.0

    def test_ensemble_stats_confidence_modes(self):
        stats = ensemble_stats(ensemble([0.0, 3.0, 6.0], [1.0, 0.5, 0.25]), ["nose"], 0.5)
        mean_c, mean_inv_c = np.mean([1.0, 0.5, 0.25]), np.mean([1.0, 2.0, 4.0])
        assert stats.mean_conf[0, 0] == pytest.approx(mean_c)
        assert stats.sd["inv-conf-weighted"][0][0, 0] == pytest.approx(np.sqrt(6.0 / mean_c))
        assert stats.sd["mixture"][0][0, 0] == pytest.approx(np.sqrt(mean_inv_c + 6.0))

    def test_ensemble_stats_shown_if_any_member_reaches_conf(self):
        members = ensemble([0.0, 1.0, 2.0], [0.2, 0.9, 0.1])
        assert ensemble_stats(members, ["nose"], 0.9).shown[0, 0]
        assert not ensemble_stats(members, ["nose"], 0.95).shown[0, 0]

    def test_ensemble_stats_zero_confidence_is_floored(self):
        stats = ensemble_stats(ensemble([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]), ["nose"], 0.0)
        assert np.isfinite(stats.sd["mixture"][0]).all()

    def test_ensemble_stats_follows_first_members_frames(self):
        first = member([1.0, 2.0], [0.0, 0.0], [1.0, 1.0], frames=("a.png", "b.png"))
        second = member([3.0, 4.0], [0.0, 0.0], [1.0, 1.0], frames=("b.png", "a.png"))
        stats = ensemble_stats([first, second], ["nose"], 0.5)
        np.testing.assert_allclose(stats.x[:, 0], [(1.0 + 4.0) / 2, (2.0 + 3.0) / 2])


class TestLabelledKeypoints:
    """Test the function labelled_keypoints."""

    def test_labelled_keypoints_none_without_ground_truth(self):
        assert labelled_keypoints(None, ["nose"]) == []

    def test_labelled_keypoints_needs_visible_column(self):
        cols = pd.MultiIndex.from_tuples([("s", "nose", "visible"), ("s", "ear", "x")])
        assert labelled_keypoints(pd.DataFrame([[2.0, 1.0]], columns=cols), ["nose", "ear"]) == [
            "nose"]


class TestPerFrameKeypointTable:
    """Test the function per_frame_keypoint_table."""

    FRAMES = pd.Index(["a.png", "b.png"])

    def stats(self, conf=0.5):
        members = [member([10.0, 10.0], [20.0, 20.0], [0.9, 0.1], frames=self.FRAMES)] * 2
        return ensemble_stats(members, ["nose"], conf)

    def gt(self, visible):
        cols = pd.MultiIndex.from_tuples([("s", "nose", k) for k in ("x", "y", "visible")])
        return pd.DataFrame([[13.0, 24.0, visible[0]], [13.0, 24.0, visible[1]]],
                            index=self.FRAMES, columns=cols)

    def test_per_frame_keypoint_table_rows_only_where_shown(self):
        table = per_frame_keypoint_table(self.stats(), self.FRAMES, ["nose"], MODES, None)
        assert list(table.frame) == ["a.png"]
        assert list(table.columns) == (["frame", "keypoint", "mean_conf", "x", "y", "err"]
                                       + [sd_column(m) for m, _ in MODES])

    def test_per_frame_keypoint_table_error_only_where_labeled(self):
        table = per_frame_keypoint_table(
            self.stats(conf=0.0), self.FRAMES, ["nose"], MODES, self.gt([2.0, 1.0]))
        assert table.err.iloc[0] == pytest.approx(5.0)    # (10, 20) vs (13, 24)
        assert np.isnan(table.err.iloc[1])

    def test_per_frame_keypoint_table_no_ground_truth_has_nan_error(self):
        table = per_frame_keypoint_table(self.stats(0.0), self.FRAMES, ["nose"], MODES, None)
        assert table.err.isna().all()

    def test_per_frame_keypoint_table_nothing_shown_is_header_only(self):
        table = per_frame_keypoint_table(self.stats(conf=2.0), self.FRAMES, ["nose"], MODES, None)
        assert table.empty
        assert "sd_mixture" in table.columns


class TestVarianceSummary:
    """Test the function variance_summary."""

    @staticmethod
    def per(errs):
        return pd.DataFrame({"keypoint": "nose", "err": errs,
                             "sd_ensemble_var": [1.0, 2.0, 3.0, 4.0][:len(errs)]})

    MODE = [("ensemble var", "Var(x)")]

    def test_variance_summary_medians_and_spearman(self):
        out = variance_summary(self.per([1.0, 2.0, 3.0, 4.0]), self.MODE).iloc[0]
        assert out.median_sd_px == 2.5
        assert out.median_err_px == 2.5
        assert out.spearman_err_sd == pytest.approx(1.0)

    def test_variance_summary_spearman_needs_three_labeled_frames(self):
        out = variance_summary(self.per([1.0, 2.0, np.nan, np.nan]), self.MODE).iloc[0]
        assert np.isnan(out.spearman_err_sd)
        assert out.median_err_px == 1.5
