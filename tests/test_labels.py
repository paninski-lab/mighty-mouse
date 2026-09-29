"""Test the mighty_mouse.labels module."""

from pathlib import Path

import pandas as pd
import pytest

from mighty_mouse.labels import (
    add_project_keypoints,
    append_keypoint_columns,
    frame_name,
    read_labels_csv,
    session_name,
)


class TestReadLabelsCsv:
    """Test the function read_labels_csv."""

    def test_read_labels_csv_roundtrip(self, tmp_path: Path, raw_df: pd.DataFrame):
        path = tmp_path / "CollectedData.csv"
        raw_df.to_csv(path)

        result = read_labels_csv(path)

        pd.testing.assert_frame_equal(result, raw_df, check_names=False)
        assert result.columns.nlevels == 3

    def test_read_labels_csv_accepts_str_path(self, tmp_path: Path, raw_df: pd.DataFrame):
        path = tmp_path / "CollectedData.csv"
        raw_df.to_csv(path)

        result = read_labels_csv(str(path))

        assert list(result.index) == list(raw_df.index)

    def test_read_labels_csv_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_labels_csv(tmp_path / "missing.csv")


class TestSessionName:
    """Test the function session_name."""

    def test_session_name_labeled_data_path(self):
        assert session_name("labeled-data/sessA/img0.png") == "sessA"

    def test_session_name_nested_prefix(self):
        assert session_name("labeled-data/facemap/sessA/img0.png") == "sessA"


class TestFrameName:
    """Test the function frame_name."""

    def test_frame_name_labeled_data_path(self):
        assert frame_name("labeled-data/sessA/img0.png") == "img0.png"

    def test_frame_name_nested_prefix(self):
        assert frame_name("labeled-data/facemap/sessA/img0.png") == "img0.png"


CRLF_CSV = (
    "scorer,orig,orig\r\n"
    "bodyparts,nose,nose\r\n"
    "coords,x,y\r\n"
    "labeled-data/sessA/img0.png,185.05215495078437,2.0\r\n"
    "labeled-data/sessA/img1.png,,\r\n"
)


class TestAppendKeypointColumns:
    """Test the function append_keypoint_columns."""

    def test_append_keypoint_columns_appends_empty_columns(self):
        result = append_keypoint_columns(CRLF_CSV, ["paw", "tail"])

        assert result.splitlines() == [
            "scorer,orig,orig,orig,orig,orig,orig",
            "bodyparts,nose,nose,paw,paw,tail,tail",
            "coords,x,y,x,y,x,y",
            "labeled-data/sessA/img0.png,185.05215495078437,2.0,,,,",
            "labeled-data/sessA/img1.png,,,,,,",
        ]

    def test_append_keypoint_columns_preserves_existing_bytes(self):
        # CRLF endings and a 17-digit float that pandas would round on a round trip
        result = append_keypoint_columns(CRLF_CSV, ["paw"])

        for old, new in zip(
            CRLF_CSV.splitlines(keepends=True), result.splitlines(keepends=True), strict=True,
        ):
            assert new.startswith(old.rstrip("\r\n"))
            assert new.endswith("\r\n")

    def test_append_keypoint_columns_lf_and_no_trailing_newline(self):
        text = "scorer,o,o\nbodyparts,a,a\ncoords,x,y\nimg.png,1,2"

        result = append_keypoint_columns(text, ["b"])

        assert result == "scorer,o,o,o,o\nbodyparts,a,a,b,b\ncoords,x,y,x,y\nimg.png,1,2,,"

    def test_append_keypoint_columns_readable(self, tmp_path: Path):
        path = tmp_path / "CollectedData.csv"
        path.write_text(append_keypoint_columns(CRLF_CSV, ["paw"]))

        df = read_labels_csv(path)

        assert list(df.columns.get_level_values(1)) == ["nose", "nose", "paw", "paw"]
        assert df[("orig", "paw", "x")].isna().all()

    def test_append_keypoint_columns_existing_keypoint(self):
        with pytest.raises(ValueError, match="already present"):
            append_keypoint_columns(CRLF_CSV, ["paw", "nose"])

    def test_append_keypoint_columns_repeated_keypoint(self):
        with pytest.raises(ValueError, match="more than once"):
            append_keypoint_columns(CRLF_CSV, ["paw", "paw"])

    def test_append_keypoint_columns_no_keypoints(self):
        with pytest.raises(ValueError, match="no keypoints"):
            append_keypoint_columns(CRLF_CSV, [])

    def test_append_keypoint_columns_bad_header(self):
        with pytest.raises(ValueError, match="3-row"):
            append_keypoint_columns("a,b\n1,2\n", ["paw"])

    def test_append_keypoint_columns_non_xy_coords(self):
        text = "scorer,o,o,o\nbodyparts,a,a,a\ncoords,x,y,likelihood\nimg.png,1,2,0.9\n"

        with pytest.raises(ValueError, match="only x/y"):
            append_keypoint_columns(text, ["b"])


class TestAddProjectKeypoints:
    """Test the function add_project_keypoints."""

    def test_add_project_keypoints_appends(self):
        cfg = {"keypoint_names": ["nose"], "schema_version": 1, "view_names": []}

        result = add_project_keypoints(cfg, ["paw", "tail"])

        assert result == {
            "keypoint_names": ["nose", "paw", "tail"], "schema_version": 1, "view_names": [],
        }

    def test_add_project_keypoints_does_not_mutate(self):
        cfg = {"keypoint_names": ["nose"]}

        add_project_keypoints(cfg, ["paw"])

        assert cfg == {"keypoint_names": ["nose"]}

    def test_add_project_keypoints_missing_names(self):
        assert add_project_keypoints({}, ["paw"]) == {"keypoint_names": ["paw"]}

    def test_add_project_keypoints_existing_keypoint(self):
        with pytest.raises(ValueError, match="already present"):
            add_project_keypoints({"keypoint_names": ["nose"]}, ["nose"])
