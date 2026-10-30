"""Test the mighty_mouse.preprocess.labeling.active_labeling module."""

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.preprocess.labeling.active_labeling import (
    add_missing_keypoints,
    combine_originals,
    compare_training_csv,
    ensemble_spread,
    exclude_frames,
    frame_scores,
    init_tolabel,
    make_manifest,
    merge_back,
    order_by_session,
    prefill_rows,
    read_frame_ids,
    round_stats,
    sample_per_session,
    select_frames,
    sync_tolabel,
)
from tests.preprocess.labeling.conftest import EARS, label_frame, pred_frame


class TestCombineOriginals:
    """Test the function combine_originals."""

    def test_combine_originals_order_and_source(self, train_df, test_df):
        both, source = combine_originals(train_df, test_df)

        assert list(both.index) == list(train_df.index) + list(test_df.index)
        assert source.tolist() == ["train"] * len(train_df) + ["test"] * len(test_df)

    def test_combine_originals_rejects_duplicate_images(self, train_df):
        with pytest.raises(ValueError, match="not unique"):
            combine_originals(train_df, train_df)

    def test_combine_originals_rejects_column_mismatch(self, train_df, test_df):
        with pytest.raises(ValueError, match="different columns"):
            combine_originals(train_df, test_df.drop(columns="ear_b", level=1))


class TestAddMissingKeypoints:
    """Test the function add_missing_keypoints."""

    def test_add_missing_keypoints_appends_empty_columns(self, train_df):
        out = add_missing_keypoints(train_df, ["ear_a", "paw"])

        assert list(out.columns[: len(train_df.columns)]) == list(train_df.columns)
        n = len(train_df.columns)
        assert list(out.columns[n:]) == [("All", "paw", "x"), ("All", "paw", "y")]
        assert out[("All", "paw", "x")].isna().all()

    def test_add_missing_keypoints_noop_when_present(self, train_df):
        assert add_missing_keypoints(train_df, EARS) is train_df


class TestSyncTolabel:
    """Test the function sync_tolabel."""

    def test_sync_tolabel_takes_other_columns_from_originals(self, train_df):
        tolabel = train_df.iloc[[3, 0]].copy()
        tolabel[("All", "ear_a", "x")] = [7.0, 8.0]
        originals = train_df.copy()
        originals.loc[originals.index[0], ("All", "nose", "x")] = 42.0

        synced, changes = sync_tolabel(tolabel, originals, EARS)

        assert list(synced.index) == list(tolabel.index)
        assert synced[("All", "ear_a", "x")].tolist() == [7.0, 8.0]
        assert synced.loc[train_df.index[0], ("All", "nose", "x")] == 42.0
        assert changes[["image", "keypoint", "coord", "old", "new"]].values.tolist() == [
            [train_df.index[0], "nose", "x", 1.0, 42.0]
        ]

    def test_sync_tolabel_cleared_cell_is_a_change(self, train_df):
        originals = train_df.copy()
        originals.loc[originals.index[1], ("All", "nose", "y")] = np.nan

        _, changes = sync_tolabel(train_df, originals, EARS)

        assert len(changes) == 1 and np.isnan(changes["new"].iloc[0])

    def test_sync_tolabel_follows_originals_columns(self, train_df):
        tolabel = add_missing_keypoints(train_df, ["paw"])
        originals = add_missing_keypoints(train_df.drop(columns="ear_b", level=1), EARS + ["paw"])

        synced, changes = sync_tolabel(tolabel, originals, EARS + ["paw"])

        assert list(synced.columns) == list(originals.columns)
        assert changes.empty

    def test_sync_tolabel_rejects_unknown_images(self, train_df):
        with pytest.raises(ValueError, match="not in the originals"):
            sync_tolabel(train_df, train_df.iloc[:2], EARS)


class TestSamplePerSession:
    """Test the function sample_per_session."""

    def test_sample_per_session_n_from_each(self, train_df, test_df):
        both, _ = combine_originals(train_df, test_df)

        out = sample_per_session(both.index, n=2, seed=0)

        counts = pd.Series([i.split("/")[1] for i in out]).value_counts()
        assert counts.to_dict() == {"sA": 2, "sB": 2, "sC": 2}

    def test_sample_per_session_takes_all_of_small_sessions(self, train_df, test_df):
        both, _ = combine_originals(train_df, test_df)

        out = sample_per_session(both.index, n=3, seed=0)

        assert sum("/sC/" in i for i in out) == 2

    def test_sample_per_session_keeps_original_order_and_is_seeded(self, train_df, test_df):
        both, _ = combine_originals(train_df, test_df)

        a = sample_per_session(both.index, n=2, seed=1)
        b = sample_per_session(both.index, n=2, seed=1)

        assert list(a) == list(b)
        assert list(a) == [i for i in both.index if i in set(a)]


class TestInitTolabel:
    """Test the function init_tolabel."""

    def test_init_tolabel_keeps_rows_and_existing_labels(self, train_df):
        rows = init_tolabel(train_df, train_df.index[[1, 5]], EARS)

        assert list(rows.index) == list(train_df.index[[1, 5]])
        assert rows[("All", "nose", "x")].tolist() == [1.0, 1.0]

    def test_init_tolabel_rejects_already_labeled_keypoints(self, train_df):
        train_df.iloc[0, train_df.columns.get_loc(("All", "ear_a", "x"))] = 3.0

        with pytest.raises(ValueError, match="already have labels"):
            init_tolabel(train_df, train_df.index[:2], EARS)

    def test_init_tolabel_allow_existing_carries_labels_over(self, train_df):
        train_df.iloc[0, train_df.columns.get_loc(("All", "ear_a", "x"))] = 3.0

        rows = init_tolabel(train_df, train_df.index[:2], EARS, allow_existing=True)

        assert rows[("All", "ear_a", "x")].iloc[0] == 3.0

    def test_init_tolabel_rejects_unknown_keypoint(self, train_df):
        with pytest.raises(ValueError, match="x/y columns"):
            init_tolabel(train_df, train_df.index[:2], ["ear_z"])


class TestMakeManifest:
    """Test the function make_manifest."""

    def test_make_manifest_columns(self, train_df, test_df):
        both, source = combine_originals(train_df, test_df)
        idx = both.index[[0, 9]]

        m = make_manifest(idx, source, round_=2, prefilled=True)

        assert m.to_dict("list") == {
            "source": ["train", "test"], "round": [2, 2], "prefilled": [True, True],
        }


class TestEnsembleSpread:
    """Test the function ensemble_spread."""

    def test_ensemble_spread_value_and_eligibility(self):
        imgs = ["a.png", "b.png"]
        preds = [
            pred_frame(imgs, {"ear_a": [(0, 0, 0.95), (0, 0, 0.5)]}),
            pred_frame(imgs, {"ear_a": [(6, 8, 0.2), (9, 9, 0.5)]}),
        ]

        out = ensemble_spread(preds, pd.Index(imgs), ["ear_a"], min_conf=0.9)

        # members 3 and 4 away on x and y: var_x = 9, var_y = 16 -> sqrt(25) = 5
        assert out["ear_a"].iloc[0] == pytest.approx(5.0)
        assert np.isnan(out["ear_a"].iloc[1])  # no member above 0.9

    def test_ensemble_spread_threshold_is_strict(self):
        preds = [pred_frame(["a.png"], {"ear_a": [(0, 0, 0.9)]}),
                 pred_frame(["a.png"], {"ear_a": [(2, 0, 0.9)]})]

        out = ensemble_spread(preds, pd.Index(["a.png"]), ["ear_a"], min_conf=0.9)

        assert np.isnan(out["ear_a"].iloc[0])


class TestFrameScores:
    """Test the function frame_scores."""

    def test_frame_scores_mean_over_eligible_only(self):
        spread = pd.DataFrame({"a": [2.0, np.nan, np.nan], "b": [4.0, 6.0, np.nan]})

        out = frame_scores(spread)

        assert out.iloc[0] == 3.0 and out.iloc[1] == 6.0 and np.isnan(out.iloc[2])


class TestSelectFrames:
    """Test the function select_frames."""

    def test_select_frames_top_m_skips_nan(self):
        scores = pd.Series([1.0, np.nan, 5.0, 3.0], index=list("abcd"))

        assert list(select_frames(scores, 2)) == ["c", "d"]
        assert list(select_frames(scores, 10)) == ["c", "d", "a"]

    def test_select_frames_all_puts_nan_last(self):
        scores = pd.Series([1.0, np.nan, 5.0], index=list("abc"))

        assert list(select_frames(scores, None)) == ["c", "a", "b"]


class TestReadFrameIds:
    """Test the function read_frame_ids."""

    def test_read_frame_ids_skips_blanks_comments_duplicates(self):
        a, b = "labeled-data/s1/img1.png", "labeled-data/s1/img2.png"
        text = f"# reserved\n{a}\n\n  {b}  \n{a}\n"

        assert list(read_frame_ids(text)) == [a, b]

    def test_read_frame_ids_empty(self):
        assert len(read_frame_ids("")) == 0


class TestExcludeFrames:
    """Test the function exclude_frames."""

    def test_exclude_frames_drops_listed(self):
        scores = pd.Series([1.0, 5.0, 3.0], index=list("abc"))

        result = exclude_frames(scores, pd.Index(["b", "z"]))

        assert list(result.index) == ["a", "c"]
        assert list(select_frames(result, 1)) == ["c"]

    def test_exclude_frames_nothing_listed(self):
        scores = pd.Series([1.0, np.nan], index=list("ab"))

        pd.testing.assert_series_equal(exclude_frames(scores, pd.Index([])), scores)


class TestOrderBySession:
    """Test the function order_by_session."""

    def test_order_by_session_groups_sessions_and_sorts_frames(self, train_df, test_df):
        both, _ = combine_originals(train_df, test_df)
        chosen = pd.Index([
            "labeled-data/sB/img3.png", "labeled-data/sC/img0.png",
            "labeled-data/sA/img2.png", "labeled-data/sB/img1.png", "labeled-data/sA/img0.png",
        ])

        out = order_by_session(chosen, both.index)

        assert list(out) == [
            "labeled-data/sA/img0.png", "labeled-data/sA/img2.png",
            "labeled-data/sB/img1.png", "labeled-data/sB/img3.png", "labeled-data/sC/img0.png",
        ]

    def test_order_by_session_keeps_all_images(self, train_df):
        chosen = train_df.index[::-1]

        assert set(order_by_session(chosen, train_df.index)) == set(chosen)


class TestPrefillRows:
    """Test the function prefill_rows."""

    def test_prefill_rows_fills_only_requested_keypoints(self):
        rows = label_frame(["a.png"])
        preds = [pred_frame(["a.png"], {"ear_a": [(2, 4, 0.99)], "ear_b": [(9, 9, 0.99)]})]

        out = prefill_rows(rows, preds, ["ear_a"], threshold=0.9)

        assert out[("All", "ear_a", "x")].tolist() == [2.0]
        assert np.isnan(out[("All", "ear_b", "x")]).all()


class TestRoundStats:
    """Test the function round_stats."""

    def test_round_stats_bins_relative_to_baseline(self):
        labeled = pd.DataFrame({"k": [1.0, 3.0, np.nan]})  # baseline 2
        pool = pd.DataFrame({"k": [1.0, 3.0, 5.0, 9.0, np.nan]})  # ratios .5 1.5 2.5 4.5

        out = round_stats(labeled, pool).iloc[0]

        assert out["baseline"] == 2.0
        assert out["labeled_eligible"] == 2 and out["pool_ineligible"] == 1
        assert [out["x0-1"], out["x1-2"], out["x2-4"], out["x4+"]] == [1, 1, 1, 1]

    def test_round_stats_counts_pool_frames_above_pixel_thresholds(self):
        labeled = pd.DataFrame({"k": [1.0]})
        pool = pd.DataFrame({"k": [1.0, 3.5, 12.0, 25.0, np.nan]})

        out = round_stats(labeled, pool, px_thresholds=(3, 10, 20)).iloc[0]

        assert [out["n>3px"], out["n>10px"], out["n>20px"]] == [3, 2, 1]

    def test_round_stats_no_eligible_labeled_frames(self):
        out = round_stats(pd.DataFrame({"k": [np.nan]}), pd.DataFrame({"k": [1.0]})).iloc[0]

        assert np.isnan(out["baseline"]) and np.isnan(out["x0-1"])


class TestCompareTrainingCsv:
    """Test the function compare_training_csv."""

    def test_compare_training_csv_identical_ignoring_order(self, train_df):
        out = compare_training_csv(train_df.iloc[::-1], train_df)

        assert out == {"missing": 0, "extra": 0, "edited": 0}

    def test_compare_training_csv_counts_rows_added_since_training(self, train_df):
        out = compare_training_csv(train_df.iloc[:5], train_df)

        assert out == {"missing": len(train_df) - 5, "extra": 0, "edited": 0}

    def test_compare_training_csv_counts_dropped_and_edited_rows(self, train_df):
        edited = train_df.copy()
        edited.iloc[0, edited.columns.get_loc(("All", "ear_a", "x"))] = 7.0

        assert compare_training_csv(train_df, edited)["edited"] == 1
        assert compare_training_csv(train_df, train_df.iloc[:3])["extra"] == len(train_df) - 3


class TestMergeBack:
    """Test the function merge_back."""

    def test_merge_back_copies_target_columns_in_original_order(self, train_df):
        tolabel = train_df.iloc[::-1].copy()  # different order
        tolabel[("All", "ear_a", "x")] = np.arange(len(tolabel), dtype=float)

        out = merge_back(train_df, tolabel, ["ear_a"])

        assert list(out.index) == list(train_df.index)
        assert out[("All", "ear_a", "x")].tolist() == list(range(len(train_df)))[::-1]
        assert out[("All", "nose", "x")].tolist() == train_df[("All", "nose", "x")].tolist()

    def test_merge_back_adds_target_columns_the_originals_lack(self, train_df):
        originals = train_df.drop(columns="ear_b", level=1)
        tolabel = train_df.copy()
        tolabel[("All", "ear_b", "y")] = 3.0

        out = merge_back(originals, tolabel, EARS)

        assert list(out.columns) == list(train_df.columns)
        assert (out[("All", "ear_b", "y")] == 3.0).all()

    def test_merge_back_rejects_missing_images(self, train_df):
        with pytest.raises(ValueError, match="not in tolabel"):
            merge_back(train_df, train_df.iloc[:3], EARS)

    def test_merge_back_rejects_edited_non_target_column(self, train_df):
        tolabel = train_df.copy()
        tolabel[("All", "nose", "x")] += 1

        with pytest.raises(ValueError, match="nose"):
            merge_back(train_df, tolabel, EARS)
