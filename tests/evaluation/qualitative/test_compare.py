"""Test the mighty_mouse.evaluation.qualitative.compare module."""

import pandas as pd

from mighty_mouse.evaluation.qualitative.compare import (
    banner_line,
    compare_keypoints,
    grid_cell,
    grid_size,
)

INV = {"ibl": {"trainable": ["paw_l", "paw_r", "nose_tip"], "direct": ["paw_l", "nose_tip"]},
       "facemap": {"trainable": ["nose_tip", "ear_base"], "direct": ["nose_tip"]}}


def pred(kps):
    cols = pd.MultiIndex.from_tuples([("s", k, c) for k in kps for c in ("x", "y", "likelihood")])
    return pd.DataFrame([[1.0] * len(cols)], columns=cols)


def labels(visible: dict[str, float]):
    cols = pd.MultiIndex.from_tuples([("s", k, "visible") for k in visible])
    return pd.DataFrame([list(visible.values())], columns=cols)


class TestCompareKeypoints:
    """Test the function compare_keypoints."""

    def test_compare_keypoints_union_of_trained_in_prediction_order(self):
        p = pred(["ear_base", "tail", "paw_l", "nose_tip"])
        kps, _ = compare_keypoints(p, INV, ["ibl", "facemap"], "ibl", labels({"nose_tip": 2.0}))
        assert kps == ["ear_base", "paw_l", "nose_tip"]

    def test_compare_keypoints_own_from_inventory(self):
        _, own = compare_keypoints(pred(["paw_l"]), INV, ["ibl"], "ibl", labels({"nose_tip": 2.0}))
        assert own == ["nose_tip", "paw_l"]

    def test_compare_keypoints_own_derived_for_target_outside_inventory(self):
        gt = labels({"digit_a": 2.0, "digit_b": 1.0, "digit_c": 2.0})
        _, own = compare_keypoints(pred(["paw_l"]), INV, ["ibl"], "ibl-digits", gt)
        assert own == ["digit_a", "digit_c"]

    def test_compare_keypoints_override_draws_given_keypoints_and_restricts_own(self):
        gt = labels({"nose_tip": 2.0})
        kps, own = compare_keypoints(pred(["paw_l"]), INV, ["ibl"], "ibl", gt,
                                     kps=["nose_tip", "tail"])
        assert kps == ["nose_tip", "tail"]
        assert own == ["nose_tip"]


class TestGrid:
    """Test the functions grid_size and grid_cell."""

    def test_grid_one_model_puts_cameras_side_by_side(self):
        assert grid_size(2, 1, 100, False) == (200, 100)
        assert grid_cell(1, 0, 1, False) == (0, 1)

    def test_grid_several_models_are_columns_and_views_rows(self):
        assert grid_size(2, 3, 100, False) == (300, 200)
        assert grid_cell(1, 2, 3, False) == (1, 2)

    def test_grid_rows_models_swaps_the_axes(self):
        assert grid_size(2, 3, 100, True) == (200, 300)
        assert grid_cell(1, 2, 3, True) == (2, 1)

    def test_grid_cells_tile_the_grid_exactly(self):
        for rows_models in (False, True):
            for n_models in (1, 3):
                w, h = grid_size(2, n_models, 10, rows_models)
                cells = {grid_cell(v, m, n_models, rows_models)
                         for v in range(2) for m in range(n_models)}
                assert len(cells) == 2 * n_models
                assert max(r for r, _ in cells) + 1 == h // 10
                assert max(c for _, c in cells) + 1 == w // 10


class TestBannerLine:
    """Test the function banner_line."""

    def test_banner_line_ground_truth_and_layout_note(self):
        line = banner_line("T", "ibl", 0.7, False, 2, False)
        assert line == ("T | ibl held-out test frames | conf >= 0.7 | x = ibl ground truth"
                        " | same frame in every column")

    def test_banner_line_no_gt_rows_models(self):
        line = banner_line("T", "ibl", 0.7, True, 2, True)
        assert "no ground truth drawn" in line and line.endswith("same frame in every row")

    def test_banner_line_single_model_has_no_layout_note(self):
        assert "same frame" not in banner_line("T", "ibl", 0.7, False, 1, False)
