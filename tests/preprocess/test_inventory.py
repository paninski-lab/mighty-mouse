"""Test the mighty_mouse.preprocess.inventory module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.preprocess.inventory import (
    column_layout,
    dataset_entry,
    direct_mask,
    hflip_partner,
    sessions,
    trainable_mask,
    unreferenced_raw_sessions,
)

CANONICAL = ["nose_tip", "ear_tip_left", "ear_tip_right", "tongue_tip"]


def label_df(frames: list[str], visible: dict[str, int]) -> pd.DataFrame:
    """Converted labels for `frames` (labeled-data/<ds>/<session>/<frame>) over CANONICAL;
    `visible` gives each keypoint's visible flag (absent = 0, unlabeled)."""
    cols = pd.MultiIndex.from_tuples(
        [("scorer", k, c) for k in CANONICAL for c in ("x", "y", "visible")]
    )
    df = pd.DataFrame(np.nan, index=frames, columns=cols)
    for k in CANONICAL:
        v = visible.get(k, 0)
        df[("scorer", k, "visible")] = v
        if v == 2:
            df[("scorer", k, "x")] = 1.0
            df[("scorer", k, "y")] = 2.0
    return df


def frames_of(session: str, n: int = 2) -> list[str]:
    return [f"labeled-data/ds/{session}/img{i}.png" for i in range(n)]


class TestColumnLayout:
    """Test the function column_layout."""

    def test_column_layout_column_order(self):
        assert column_layout(label_df(frames_of("s0"), {}))[:4] == [
            ("nose_tip", "x"), ("nose_tip", "y"), ("nose_tip", "visible"), ("ear_tip_left", "x"),
        ]


class TestSessions:
    """Test the function sessions."""

    def test_sessions_from_row_paths(self):
        df = label_df(frames_of("s0") + frames_of("s1"), {})
        assert sessions(df) == {"s0", "s1"}


class TestDirectMask:
    """Test the function direct_mask."""

    def test_direct_mask_only_visible_2(self):
        df = label_df(frames_of("s0"), {"nose_tip": 2, "ear_tip_left": 1})
        assert direct_mask(df) == {"nose_tip"}

    def test_direct_mask_empty_when_nothing_labeled(self):
        assert direct_mask(label_df(frames_of("s0"), {})) == set()


class TestHflipPartner:
    """Test the function hflip_partner."""

    def test_hflip_partner_swaps_side(self):
        assert hflip_partner("ear_tip_left") == "ear_tip_right"
        assert hflip_partner("ear_tip_right") == "ear_tip_left"

    def test_hflip_partner_none_for_midline(self):
        assert hflip_partner("nose_tip") is None


class TestTrainableMask:
    """Test the function trainable_mask."""

    def test_trainable_mask_adds_partners(self):
        assert trainable_mask({"ear_tip_left", "nose_tip"}, CANONICAL) == {
            "ear_tip_left", "ear_tip_right", "nose_tip",
        }

    def test_trainable_mask_skips_partners_outside_vocabulary(self):
        assert trainable_mask({"eye_left"}, CANONICAL) == {"eye_left"}


class TestUnreferencedRawSessions:
    """Test the function unreferenced_raw_sessions."""

    def test_unreferenced_raw_sessions_lists_unused_dirs(self, tmp_path):
        for s in ("s0", "s1", "extra"):
            (tmp_path / "ds" / "labeled-data" / s).mkdir(parents=True)
        assert unreferenced_raw_sessions("ds", {"s0", "s1"}, tmp_path) == ["extra"]

    def test_unreferenced_raw_sessions_no_raw_dir(self, tmp_path):
        assert unreferenced_raw_sessions("ds", {"s0"}, tmp_path) == []


class TestDatasetEntry:
    """Test the function dataset_entry."""

    @pytest.fixture
    def corpus(self, tmp_path) -> Path:
        (tmp_path / "configs" / "datasets").mkdir(parents=True)
        (tmp_path / "configs" / "datasets" / "ds.yaml").write_text(
            "exclude:\n  keypoints: [paw]\nsessions:\n  s0: left\n"
        )
        return tmp_path

    def write(self, corpus: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
        train.to_csv(corpus / "CollectedData_ds_train.csv")
        test.to_csv(corpus / "CollectedData_ds_test.csv")

    def test_dataset_entry_masks_and_counts(self, corpus):
        self.write(corpus, label_df(frames_of("s0", 3), {"ear_tip_left": 2}),
                   label_df(frames_of("s1"), {"nose_tip": 2}))
        with patch("mighty_mouse.preprocess.inventory.repo_root", return_value=corpus):
            e = dataset_entry("ds", CANONICAL, corpus, corpus / "raw")
        assert (e["train_frames"], e["test_frames"]) == (3, 2)
        assert e["direct"] == ["ear_tip_left"]
        assert e["trainable"] == ["ear_tip_left", "ear_tip_right"]
        assert e["eval"] == ["nose_tip"]
        assert e["views"] == {"s0": "left"}
        assert e["excluded"] == ["paw"]

    def test_dataset_entry_rejects_session_overlap(self, corpus):
        self.write(corpus, label_df(frames_of("s0"), {}), label_df(frames_of("s0"), {}))
        with patch("mighty_mouse.preprocess.inventory.repo_root", return_value=corpus), \
             pytest.raises(ValueError, match="overlap"):
            dataset_entry("ds", CANONICAL, corpus, corpus / "raw")

    def test_dataset_entry_rejects_keypoint_order(self, corpus):
        self.write(corpus, label_df(frames_of("s0"), {}), label_df(frames_of("s1"), {}))
        with patch("mighty_mouse.preprocess.inventory.repo_root", return_value=corpus), \
             pytest.raises(ValueError, match="canonical layout"):
            dataset_entry("ds", CANONICAL[::-1], corpus, corpus / "raw")

    def test_dataset_entry_rejects_missing_visible(self, corpus):
        test = label_df(frames_of("s1"), {}).drop(columns=("scorer", "tongue_tip", "visible"))
        self.write(corpus, label_df(frames_of("s0"), {}), test)
        with patch("mighty_mouse.preprocess.inventory.repo_root", return_value=corpus), \
             pytest.raises(ValueError, match="test CSV columns"):
            dataset_entry("ds", CANONICAL, corpus, corpus / "raw")
