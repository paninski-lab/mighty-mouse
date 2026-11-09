"""Test the mighty_mouse.evaluation.qualitative.panel module."""

from pathlib import Path

import numpy as np
import pandas as pd

from mighty_mouse.evaluation.qualitative.panel import (
    VIEW_ORDER,
    header_text,
    load_panels,
    panel_keypoints,
    panel_order,
    run_label,
)

ALL_KPS = ["nose_tip", "paw_l", "ear_base", "tongue_tip", "pupil_center_right"]


class TestRunLabel:
    """Test the function run_label."""

    def test_run_label_from_results_dir_without_recipe_parts(self):
        run = Path("/x/results/head-fixed_v2/trunks/a_train/supervised/sampling-T2/tf1/seed0")
        assert run_label(run) == "head-fixed_v2/trunks/a_train/sampling-T2/seed0"

    def test_run_label_without_results_is_the_directory_name(self):
        assert run_label(Path("/elsewhere/run7")) == "run7"


class TestPanelKeypoints:
    """Test the function panel_keypoints."""

    SUPERVISED, LABELLED = {"nose_tip", "paw_l"}, {"nose_tip"}

    def test_panel_keypoints_transfer_is_unsupervised_plus_extras(self):
        transfer, extra = panel_keypoints("transfer", ALL_KPS, self.SUPERVISED, self.LABELLED)
        assert transfer == ["ear_base", "tongue_tip", "pupil_center_right"]
        assert extra == []      # extras already among the transfer keypoints

    def test_panel_keypoints_transfer_adds_supervised_extras(self):
        transfer, extra = panel_keypoints(
            "transfer", ALL_KPS, self.SUPERVISED | {"tongue_tip"}, self.LABELLED)
        assert "tongue_tip" not in transfer and extra == ["tongue_tip"]

    def test_panel_keypoints_all_and_own(self):
        assert panel_keypoints("all", ALL_KPS, self.SUPERVISED, self.LABELLED) == (ALL_KPS, [])
        assert panel_keypoints("own", ALL_KPS, self.SUPERVISED, self.LABELLED) == (
            ["nose_tip"], [])


class TestPanelOrder:
    """Test the function panel_order."""

    def test_panel_order_follows_view_order_then_appends_unknown_sorted(self):
        keys = [("kondo", "bottom"), ("ibl", "right"), ("zzz", "x"), ("ibl", "left"),
                ("aaa", "y")]
        order, missing = panel_order(keys)
        assert order == [("ibl", "left"), ("ibl", "right"), ("kondo", "bottom")]
        assert missing == [("aaa", "y"), ("zzz", "x")]
        assert order == [k for k in VIEW_ORDER if k in order]


class TestHeaderText:
    """Test the function header_text."""

    def test_header_text_modes(self):
        transfer, labelled = ["ear_base", "nose_tip", "paw_l"], {"nose_tip"}
        assert header_text("transfer", "ibl", "left", transfer, [], labelled) == (
            "ibl / left  (3 inherited)")
        assert header_text("all", "ibl", "left", transfer, [], labelled) == (
            "ibl / left  (1 labelled + 2 other)")
        assert header_text("own", "ibl", "left", ["nose_tip"], [], labelled) == (
            "ibl / left  (1 labelled)")

    def test_header_text_names_the_extras(self):
        text = header_text("transfer", "ibl", "left", [], ["tongue_tip", "pupil_center_right"],
                           set())
        assert text.endswith(" + own tongue tip, R pupil")


class TestLoadPanels:
    """Test the function load_panels."""

    @staticmethod
    def write_predictions(path: Path, frames: list[str], kps: list[str]) -> None:
        cols = pd.MultiIndex.from_tuples(
            [("s", k, c) for k in kps for c in ("x", "y", "likelihood")],
            names=["scorer", "bodyparts", "coords"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(np.ones((len(frames), len(cols))), index=frames, columns=cols).to_csv(path)

    INV = {"ibl": {"trainable": ["paw_l"], "direct": ["paw_l"]},
           "kondo": {"trainable": ["nose_tip"], "direct": ["nose_tip"]}}

    def frames(self, dataset, session, n):
        return [f"labeled-data/{dataset}/{session}/img{i:03d}.png" for i in range(n)]

    def test_load_panels_one_panel_per_view_with_sampled_frames(self, tmp_path):
        frames = self.frames("ibl", "a_left", 5) + self.frames("ibl", "a_right_flipped", 5)
        self.write_predictions(tmp_path / "eval" / "ibl" / "predictions.csv", frames,
                               ["paw_l", "nose_tip"])
        panels = load_panels(tmp_path, self.INV, datasets=None, exclude=["kondo"],
                             mode="transfer", split="test", frames=3)
        assert sorted(panels) == [("ibl", "left"), ("ibl", "right")]
        left = panels[("ibl", "left")]
        assert len(left["frames"]) == 3
        assert left["transfer"] == ["nose_tip"] and left["labelled"] == {"paw_l"}

    def test_load_panels_skips_missing_predictions_and_excluded_datasets(self, tmp_path, capsys):
        self.write_predictions(tmp_path / "eval" / "ibl" / "predictions.csv",
                               self.frames("ibl", "a_left", 2), ["paw_l"])
        panels = load_panels(tmp_path, self.INV, datasets=None, exclude=[], mode="all",
                             split="test", frames=5)
        assert list(panels) == [("ibl", "left")]
        assert "skip kondo (test)" in capsys.readouterr().out

    def test_load_panels_both_splits_prefix_the_view(self, tmp_path):
        frames = self.frames("ibl", "a_left", 2)
        self.write_predictions(tmp_path / "eval" / "ibl" / "predictions.csv", frames, ["paw_l"])
        self.write_predictions(tmp_path / "zeroshot" / "ibl_train_predictions.csv", frames,
                               ["paw_l"])
        panels = load_panels(tmp_path, self.INV, datasets=["ibl"], exclude=[], mode="all",
                             split="both", frames=5)
        assert sorted(panels) == [("ibl", "test left"), ("ibl", "train left")]
