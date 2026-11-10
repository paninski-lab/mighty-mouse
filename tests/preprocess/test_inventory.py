"""Test the mighty_mouse.preprocess.inventory module."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.preprocess.inventory import (
    column_layout,
    convert_in_memory,
    count_labels,
    dataset_entry,
    direct_mask,
    load_converted,
    make_table,
    run,
    sessions,
    trainable_mask,
    unreferenced_raw_sessions,
    write_tables,
)

CANONICAL = ["nose_tip", "ear_tip_left", "ear_tip_right", "tongue_tip"]
REGIONS = {"nose_tip": "nose", "ear_tip_left": "ears", "ear_tip_right": "ears",
           "tongue_tip": "mouth"}


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


class TestTrainableMask:
    """Test the function trainable_mask."""

    def test_trainable_mask_adds_partners(self):
        assert trainable_mask({"ear_tip_left", "nose_tip"}, CANONICAL) == {
            "ear_tip_left", "ear_tip_right", "nose_tip",
        }

    def test_trainable_mask_skips_partners_outside_vocabulary(self):
        assert trainable_mask({"eye_left"}, CANONICAL) == {"eye_left"}


class TestCountLabels:
    """Test the function count_labels."""

    def test_count_labels_counts_occluded_not_absent(self):
        df = label_df(frames_of("s0", 3), {"nose_tip": 2, "ear_tip_left": 1})
        assert count_labels(df) == 6


class TestUnreferencedRawSessions:
    """Test the function unreferenced_raw_sessions."""

    def test_unreferenced_raw_sessions_lists_unused_dirs(self, tmp_path):
        for s in ("s0", "s1", "extra"):
            (tmp_path / "labeled-data" / s).mkdir(parents=True)
        assert unreferenced_raw_sessions(tmp_path, {"s0", "s1"}) == ["extra"]

    def test_unreferenced_raw_sessions_no_raw_dir(self, tmp_path):
        assert unreferenced_raw_sessions(tmp_path / "missing", {"s0"}) == []


class TestDatasetEntry:
    """Test the function dataset_entry."""

    def test_dataset_entry_counts_and_masks(self):
        train = label_df(frames_of("s0", 3), {"ear_tip_left": 2})
        test = label_df(frames_of("s1"), {"nose_tip": 2})
        e = dataset_entry("ds", CANONICAL, train, test, REGIONS, "3")
        assert (e["train_frames"], e["test_frames"]) == (3, 2)
        assert (e["train_sessions"], e["test_sessions"]) == (1, 1)
        assert (e["train_labels"], e["test_labels"]) == (3, 2)
        assert e["version"] == "3"
        assert e["direct"] == ["ear_tip_left"]
        assert e["trainable"] == ["ear_tip_left", "ear_tip_right"]

    def test_dataset_entry_region_counts_collapse_sides(self):
        train = label_df(frames_of("s0"), {"ear_tip_left": 2, "ear_tip_right": 2, "nose_tip": 2})
        e = dataset_entry("ds", CANONICAL, train, label_df(frames_of("s1"), {}), REGIONS)
        assert e["regions"] == {"nose": 1, "ears": 1, "mouth": 0}

    def test_dataset_entry_rejects_session_overlap(self):
        df = label_df(frames_of("s0"), {})
        with pytest.raises(ValueError, match="overlap"):
            dataset_entry("ds", CANONICAL, df, df, REGIONS)

    def test_dataset_entry_rejects_keypoint_order(self):
        with pytest.raises(ValueError, match="canonical layout"):
            dataset_entry("ds", CANONICAL[::-1], label_df(frames_of("s0"), {}),
                          label_df(frames_of("s1"), {}), REGIONS)

    def test_dataset_entry_rejects_missing_visible(self):
        test = label_df(frames_of("s1"), {}).drop(columns=("scorer", "tongue_tip", "visible"))
        with pytest.raises(ValueError, match="test labels"):
            dataset_entry("ds", CANONICAL, label_df(frames_of("s0"), {}), test, REGIONS)


class TestLoadConverted:
    """Test the function load_converted."""

    def test_load_converted_reads_both_splits(self, tmp_path):
        label_df(frames_of("s0"), {"nose_tip": 2}).to_csv(tmp_path / "CollectedData_ds_train.csv")
        label_df(frames_of("s1"), {}).to_csv(tmp_path / "CollectedData_ds_test.csv")
        train, test = load_converted(tmp_path, "ds")
        assert sessions(train) == {"s0"} and sessions(test) == {"s1"}

    def test_load_converted_none_when_not_converted(self, tmp_path):
        assert load_converted(tmp_path, "ds") is None


def write_raw(folder, sessions_: dict[str, int]) -> None:
    """Raw DLC-style CSVs (one `nose` keypoint) with the given frames per train/test session."""
    folder.mkdir(parents=True, exist_ok=True)
    cols = pd.MultiIndex.from_tuples([("me", "nose", "x"), ("me", "nose", "y")])
    for csv_name, names in (("CollectedData.csv", ["s0"]), ("CollectedData_test.csv", ["s1"])):
        idx = [f"labeled-data/{s}/img{i}.png" for s in names for i in range(sessions_[s])]
        pd.DataFrame(1.0, index=idx, columns=cols).to_csv(folder / csv_name)


CONFIG = {"exclude": {"sessions": [], "keypoints": []}, "keypoints": {"nose": "nose_tip"}}


class TestConvertInMemory:
    """Test the function convert_in_memory."""

    def test_convert_in_memory_matches_conversion(self, tmp_path):
        write_raw(tmp_path / "raw", {"s0": 3, "s1": 2})
        train, test = convert_in_memory(tmp_path / "raw", "ds", CONFIG, CANONICAL)
        assert list(train.index)[0] == "labeled-data/ds/s0/img0.png"
        assert (len(train), len(test)) == (3, 2)
        assert direct_mask(train) == {"nose_tip"}

    def test_convert_in_memory_drops_excluded_sessions(self, tmp_path):
        write_raw(tmp_path / "raw", {"s0": 3, "s1": 2})
        config = {**CONFIG, "exclude": {"sessions": ["s0"], "keypoints": []}}
        train, _ = convert_in_memory(tmp_path / "raw", "ds", config, CANONICAL)
        assert len(train) == 0

    def test_convert_in_memory_none_without_csvs(self, tmp_path):
        assert convert_in_memory(tmp_path / "raw", "ds", CONFIG, CANONICAL) is None


def entry(frames: int = 3, nose: int = 1) -> dict:
    return {"version": "2", "train_frames": frames, "test_frames": 1, "train_sessions": 4,
            "test_sessions": 1, "train_labels": 10, "test_labels": 2,
            "regions": {"nose": nose, "ears": 0}, "direct": ["nose_tip"],
            "trainable": ["nose_tip", "x"]}


class TestMakeTable:
    """Test the function make_table."""

    def test_make_table_rows_and_total(self):
        header, rows = make_table({"a": entry(3, 1), "b": entry(5, 0)})
        assert header == ["dataset", "version", "train_frames", "test_frames", "train_sessions",
                          "test_sessions", "train_labels", "test_labels", "nose", "ears",
                          "direct", "trainable"]
        assert rows[0] == ["a", "2", 3, 1, 4, 1, 10, 2, 1, 0, 1, 2]
        assert rows[-1] == ["TOTAL", "", 8, 2, "", "", 20, 4, "1 / 2", "0 / 2", "", ""]
        assert all(len(r) == len(header) for r in rows)

    def test_make_table_empty(self):
        _, rows = make_table({})
        assert rows == [["TOTAL", "", 0, 0, "", "", 0, 0, "", ""]]


class TestWriteTables:
    """Test the function write_tables."""

    def test_write_tables_csv_and_md(self, tmp_path):
        written = write_tables(["a", "b"], [["x", 1]], tmp_path / "out", "t")
        assert [p.name for p in written] == ["t.csv", "t.md"]
        assert (tmp_path / "out" / "t.csv").read_text().splitlines() == ["a,b", "x,1"]
        assert (tmp_path / "out" / "t.md").read_text().splitlines() == [
            "| a | b |", "|---|---|", "| x | 1 |"]


class TestRun:
    """Test the function run."""

    @pytest.fixture
    def configs(self, tmp_path):
        cfg = tmp_path / "configs"
        (cfg / "datasets").mkdir(parents=True)
        (cfg / "keypoints.yaml").write_text(
            "keypoints:\n  # ── nose ──\n  - nose_tip\n  # ── ears ──\n  - ear_tip_left\n"
            "  - ear_tip_right\n  # ── mouth ──\n  - tongue_tip\n")
        for name in ("ds", "other"):
            (cfg / "datasets" / f"{name}.yaml").write_text("keypoints:\n  nose: nose_tip\n")
        with patch("mighty_mouse.preprocess.inventory.default_configs_dir", return_value=cfg), \
             patch("mighty_mouse.preprocess.inventory.ALL_DATASETS", ["ds", "other"]):
            yield cfg

    def test_run_raw_writes_tables_only(self, tmp_path, configs):
        write_raw(tmp_path / "raw" / "ds", {"s0": 3, "s1": 2})
        (tmp_path / "raw" / "ds" / "VERSION.txt").write_text("4\n")
        written = run("raw", tmp_path / "raw", out_dir=tmp_path / "assets")
        assert sorted(p.name for p in written) == ["corpus.csv", "corpus.md"]
        row = (tmp_path / "assets" / "corpus.csv").read_text().splitlines()[1].split(",")
        assert row[:4] == ["ds", "4", "3", "2"]

    def test_run_corpus_writes_json_next_to_data(self, tmp_path, configs):
        data = tmp_path / "data"
        data.mkdir()
        label_df(frames_of("s0", 3), {"nose_tip": 2}).to_csv(data / "CollectedData_ds_train.csv")
        label_df(frames_of("s1"), {}).to_csv(data / "CollectedData_ds_test.csv")
        written = run("corpus", tmp_path / "raw", data)
        assert sorted(p.name for p in written) == [
            "dataset_inventory.csv", "dataset_inventory.json", "dataset_inventory.md"]
        manifest = json.loads((data / "dataset_inventory.json").read_text())
        assert list(manifest["datasets"]) == ["ds"]
        assert manifest["datasets"]["ds"]["direct"] == ["nose_tip"]
        assert manifest["canonical_keypoints"] == CANONICAL

    def test_run_rejects_corpus_without_data_dir(self, tmp_path, configs):
        with pytest.raises(ValueError, match="data_dir"):
            run("corpus", tmp_path)

    def test_run_rejects_unknown_source(self, tmp_path, configs):
        with pytest.raises(ValueError, match="source"):
            run("elsewhere", tmp_path)
