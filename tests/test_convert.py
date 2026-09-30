"""Test the mighty_mouse.convert module."""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mighty_mouse.convert import (
    check_exclusions,
    check_keypoint_sources,
    check_keypoint_targets,
    check_session_sides,
    check_sessions,
    copy_images,
    csv_keypoints,
    csv_sessions,
    drop_excluded_sessions,
    map_keypoints,
    post_process,
    process_split,
    raw_dataset_dir,
    reindex_to_canonical,
    remap_index,
    suppress_occluded,
    validate_dataset_config,
)
from mighty_mouse.labels import SCORER

NAN = np.nan


def _vis(df: pd.DataFrame, kp: str) -> list[float]:
    return df[(SCORER, kp, "visible")].tolist()


def _xy(df: pd.DataFrame, kp: str) -> list[list[float]]:
    return df[[(SCORER, kp, "x"), (SCORER, kp, "y")]].to_numpy().tolist()


class TestSuppressOccluded:
    """Test the function suppress_occluded."""

    def test_suppress_occluded_only_changes_vis_1(self, make_processed_df: Callable):
        df = make_processed_df({"paw_left": [2.0, 1.0, 0.0]})

        result = suppress_occluded(df, ["paw_left"])

        assert _vis(result, "paw_left") == [2.0, 0.0, 0.0]

    def test_suppress_occluded_leaves_other_keypoints(self, make_processed_df: Callable):
        df = make_processed_df({"paw_left": [1.0], "paw_right": [1.0]})

        result = suppress_occluded(df, ["paw_left"])

        assert _vis(result, "paw_right") == [1.0]

    def test_suppress_occluded_no_keypoints(self, make_processed_df: Callable):
        df = make_processed_df({"paw_left": [1.0]})

        result = suppress_occluded(df, [])

        assert _vis(result, "paw_left") == [1.0]

    def test_suppress_occluded_unknown_keypoint(self, make_processed_df: Callable):
        df = make_processed_df({"paw_left": [1.0]})

        with pytest.raises(KeyError):
            suppress_occluded(df, ["nope"])


class TestPostProcess:
    """Test the function post_process."""

    def test_post_process_no_hook(self, make_processed_df: Callable):
        df = make_processed_df({"paw_left": [1.0]})

        result = post_process(df, {}, "facemap")

        assert result is df
        assert _vis(result, "paw_left") == [1.0]

    def test_post_process_dispatches_registered_hook(self, make_processed_df: Callable):
        df = make_processed_df({"paw_left": [1.0]})
        hook = MagicMock(return_value="sentinel")

        with patch.dict("mighty_mouse.convert.POST_PROCESS", {"toy": hook}):
            result = post_process(df, {"a": 1}, "toy")

        hook.assert_called_once_with(df, {"a": 1})
        assert result == "sentinel"

    def test_post_process_hantman_mv(self, make_processed_df: Callable):
        df = make_processed_df({
            "paw_left": [1.0, 2.0],
            "paw_right": [1.0, 2.0],
            "ear_tip_left": [1.0, 1.0],   # exempt: keeps the occlusion signal
        })

        result = post_process(df, {}, "hantman-mv")

        assert _vis(result, "paw_left") == [0.0, 2.0]
        assert _vis(result, "paw_right") == [1.0, 2.0]
        assert _vis(result, "ear_tip_left") == [1.0, 1.0]

    def test_post_process_kaufman(self, make_processed_df: Callable):
        df = make_processed_df({
            "wrist_left": [1.0, 2.0],
            "d1_tip_left": [1.0, 1.0],
            "d1_tip_right": [1.0, 1.0],
            "eye_top_left": [1.0, 1.0],   # face: left side really is hidden
        })

        result = post_process(df, {}, "kaufman")

        assert _vis(result, "wrist_left") == [0.0, 2.0]
        assert _vis(result, "d1_tip_left") == [0.0, 0.0]
        assert _vis(result, "d1_tip_right") == [1.0, 1.0]
        assert _vis(result, "eye_top_left") == [1.0, 1.0]

    def test_post_process_aind_vbn_face(self, make_processed_df: Callable):
        df = make_processed_df({
            "eye_top_right": [1.0, 1.0],
            "pad_top_right": [1.0, 1.0],
            "ear_top_right": [1.0, 1.0],
            "eye_top_left": [1.0, 2.0],     # labeled side: untouched
            "d1_tip_right": [1.0, 2.0],     # right fingers really are labeled: untouched
        })

        result = post_process(df, {}, "aind-vbn-face")

        assert _vis(result, "eye_top_right") == [0.0, 0.0]
        assert _vis(result, "pad_top_right") == [0.0, 0.0]
        assert _vis(result, "ear_top_right") == [0.0, 0.0]
        assert _vis(result, "eye_top_left") == [1.0, 2.0]
        assert _vis(result, "d1_tip_right") == [1.0, 2.0]


class TestRawDatasetDir:
    """Test the function raw_dataset_dir."""

    def test_raw_dataset_dir_defaults_to_dataset_name(self, tmp_path: Path):
        assert raw_dataset_dir(tmp_path, "ibl", {}) == tmp_path / "ibl"

    def test_raw_dataset_dir_raw_folder_override(self, tmp_path: Path):
        result = raw_dataset_dir(tmp_path, "ibl", {"raw_folder": "ibl-v2"})

        assert result == tmp_path / "ibl-v2"

    def test_raw_dataset_dir_null_raw_folder(self, tmp_path: Path):
        assert raw_dataset_dir(tmp_path, "ibl", {"raw_folder": None}) == tmp_path / "ibl"


class TestCsvKeypoints:
    """Test the function csv_keypoints."""

    def test_csv_keypoints_union(self, make_raw_df: Callable):
        df_a = make_raw_df({"labeled-data/s/i.png": {"nose": (1.0, 1.0)}})
        df_b = make_raw_df({"labeled-data/s/i.png": {"paw": (1.0, 1.0)}})

        assert csv_keypoints([df_a, df_b]) == {"nose", "paw"}

    def test_csv_keypoints_skips_none(self, raw_df: pd.DataFrame):
        assert csv_keypoints([None, raw_df]) == {"nose", "paw", "tail"}

    def test_csv_keypoints_all_none(self):
        assert csv_keypoints([None, None]) == set()


class TestCsvSessions:
    """Test the function csv_sessions."""

    def test_csv_sessions_union(self, make_raw_df: Callable):
        df_a = make_raw_df({"labeled-data/s1/i.png": {"nose": (1.0, 1.0)}})
        df_b = make_raw_df({"labeled-data/s2/i.png": {"nose": (1.0, 1.0)}})

        assert csv_sessions([df_a, df_b]) == {"s1", "s2"}

    def test_csv_sessions_skips_none(self, raw_df: pd.DataFrame):
        assert csv_sessions([raw_df, None]) == {"sessA", "sessB", "sessX"}

    def test_csv_sessions_all_none(self):
        assert csv_sessions([None]) == set()


class TestCheckExclusions:
    """Test the function check_exclusions."""

    def test_check_exclusions_valid(self, dataset_config: dict):
        assert check_exclusions(dataset_config, {"tail"}, {"sessX"}) == []

    def test_check_exclusions_unknown_keypoint(self, dataset_config: dict):
        errors = check_exclusions(dataset_config, set(), {"sessX"})

        assert errors == ["exclude.keypoints: 'tail' not found in any CSV"]

    def test_check_exclusions_unknown_session(self, dataset_config: dict):
        errors = check_exclusions(dataset_config, {"tail"}, set())

        assert errors == ["exclude.sessions: 'sessX' not found in any CSV"]


class TestCheckKeypointSources:
    """Test the function check_keypoint_sources."""

    def test_check_keypoint_sources_valid(self, dataset_config: dict):
        assert check_keypoint_sources(dataset_config, {"nose", "paw"}) == []

    def test_check_keypoint_sources_missing(self, dataset_config: dict):
        errors = check_keypoint_sources(dataset_config, {"nose"})

        assert errors == ["keypoints: source 'paw' not found in any CSV"]


class TestCheckKeypointTargets:
    """Test the function check_keypoint_targets."""

    def test_check_keypoint_targets_valid(self, dataset_config: dict, canonical_kps: list[str]):
        assert check_keypoint_targets(dataset_config, canonical_kps) == []

    def test_check_keypoint_targets_unknown_plain(self, canonical_kps: list[str]):
        config = {"keypoints": {"snout": "snout"}}

        errors = check_keypoint_targets(config, canonical_kps)

        assert errors == ["keypoints: 'snout' → 'snout' not in keypoints.yaml"]

    def test_check_keypoint_targets_side_checks_both(self):
        config = {"keypoints": {"paw": "paw_{side}"}}

        errors = check_keypoint_targets(config, ["paw_left"])

        assert errors == ["keypoints: 'paw' → 'paw_right' not in keypoints.yaml"]


class TestCheckSessions:
    """Test the function check_sessions."""

    def test_check_sessions_valid(self, dataset_config: dict):
        assert check_sessions(dataset_config, {"sessA", "sessB", "sessX"}) == []

    def test_check_sessions_config_session_not_in_csv(self, dataset_config: dict):
        errors = check_sessions(dataset_config, {"sessA", "sessX"})

        assert errors == ["sessions: 'sessB' not found in any CSV"]

    def test_check_sessions_unaccounted_csv_sessions(self, dataset_config: dict):
        errors = check_sessions(dataset_config, {"sessA", "sessB", "sessX", "sessZ", "sessY"})

        # sorted, so the error order is deterministic
        assert errors == [
            "CSV session 'sessY' missing from sessions and exclude.sessions",
            "CSV session 'sessZ' missing from sessions and exclude.sessions",
        ]

    def test_check_sessions_no_sessions_block(self):
        config = {"exclude": {"sessions": [], "keypoints": []}, "keypoints": {}}

        errors = check_sessions(config, {"s1"})

        assert errors == ["CSV session 's1' missing from sessions and exclude.sessions"]


class TestCheckSessionSides:
    """Test the function check_session_sides."""

    def test_check_session_sides_valid(self, dataset_config: dict):
        assert check_session_sides(dataset_config) == []

    def test_check_session_sides_invalid_side(self, dataset_config: dict):
        dataset_config["sessions"]["sessB"] = "top"

        errors = check_session_sides(dataset_config)

        assert len(errors) == 1
        assert "'sessB' has side 'top'" in errors[0]
        assert "['paw']" in errors[0]

    def test_check_session_sides_null_side(self, dataset_config: dict):
        dataset_config["sessions"]["sessB"] = None

        errors = check_session_sides(dataset_config)

        assert len(errors) == 1
        assert "'sessB' has side None" in errors[0]

    def test_check_session_sides_ignores_excluded_session(self, dataset_config: dict):
        dataset_config["sessions"]["sessX"] = "top"

        assert check_session_sides(dataset_config) == []

    def test_check_session_sides_no_lateralized_keypoints(self, dataset_config: dict):
        dataset_config["keypoints"] = {"nose": "nose"}
        dataset_config["sessions"]["sessB"] = "top"

        assert check_session_sides(dataset_config) == []

    def test_check_session_sides_lateralized_keypoint_excluded(self, dataset_config: dict):
        dataset_config["exclude"]["keypoints"].append("paw")
        dataset_config["sessions"]["sessB"] = "top"

        assert check_session_sides(dataset_config) == []


class TestValidateDatasetConfig:
    """Test the function validate_dataset_config."""

    def test_validate_dataset_config_valid(
        self,
        dataset_config: dict,
        canonical_kps: list[str],
        raw_df: pd.DataFrame,
    ):
        assert validate_dataset_config(dataset_config, canonical_kps, [raw_df, None]) == []

    def test_validate_dataset_config_accepts_generator(
        self,
        dataset_config: dict,
        canonical_kps: list[str],
        raw_df: pd.DataFrame,
    ):
        # dfs is consumed twice internally (keypoints, sessions)
        dfs = (df for df in [raw_df])

        assert validate_dataset_config(dataset_config, canonical_kps, dfs) == []

    def test_validate_dataset_config_collects_all_errors(
        self,
        dataset_config: dict,
        canonical_kps: list[str],
        raw_df: pd.DataFrame,
    ):
        dataset_config["keypoints"]["whisker"] = "whisker"
        dataset_config["sessions"]["sessB"] = "top"

        errors = validate_dataset_config(dataset_config, canonical_kps, [raw_df])

        assert errors == [
            "keypoints: source 'whisker' not found in any CSV",
            "keypoints: 'whisker' → 'whisker' not in keypoints.yaml",
            "sessions: 'sessB' has side 'top', but ['paw'] are mapped to *_{side} names "
            "-- side must be 'left' or 'right'",
        ]


class TestDropExcludedSessions:
    """Test the function drop_excluded_sessions."""

    def test_drop_excluded_sessions(self, raw_df: pd.DataFrame):
        result = drop_excluded_sessions(raw_df, ["sessX"])

        assert list(result.index) == [
            "labeled-data/sessA/img0.png",
            "labeled-data/sessA/img1.png",
            "labeled-data/sessB/img0.png",
        ]

    def test_drop_excluded_sessions_none_excluded(self, raw_df: pd.DataFrame):
        result = drop_excluded_sessions(raw_df, [])

        pd.testing.assert_frame_equal(result, raw_df)

    def test_drop_excluded_sessions_all_excluded(self, raw_df: pd.DataFrame):
        result = drop_excluded_sessions(raw_df, ["sessA", "sessB", "sessX"])

        assert result.empty


class TestMapKeypoints:
    """Test the function map_keypoints."""

    def test_map_keypoints_plain(self, raw_df: pd.DataFrame):
        result = map_keypoints(raw_df, {"nose": "snout"}, {})

        assert list(result.columns) == [
            (SCORER, "snout", "x"), (SCORER, "snout", "y"), (SCORER, "snout", "visible"),
        ]
        assert result.columns.names == ["scorer", "bodyparts", "coords"]
        assert _vis(result, "snout") == [2.0, 1.0, 2.0, 2.0]
        np.testing.assert_array_equal(
            _xy(result, "snout"), [[1.0, 2.0], [NAN, NAN], [9.0, 10.0], [0.0, 0.0]],
        )

    def test_map_keypoints_lateralized(self, raw_df: pd.DataFrame):
        sides = {"sessA": "left", "sessB": "right"}

        result = map_keypoints(raw_df, {"paw": "paw_{side}"}, sides)

        # sessA -> left, sessB -> right (unlabeled there), sessX has no side
        np.testing.assert_array_equal(
            _xy(result, "paw_left"), [[3.0, 4.0], [7.0, 8.0], [NAN, NAN], [NAN, NAN]],
        )
        assert _vis(result, "paw_left") == [2.0, 2.0, 1.0, 1.0]
        assert np.isnan(_xy(result, "paw_right")).all()
        assert _vis(result, "paw_right") == [1.0, 1.0, 1.0, 1.0]

    def test_map_keypoints_preserves_index(self, raw_df: pd.DataFrame):
        result = map_keypoints(raw_df, {"nose": "nose"}, {})

        assert list(result.index) == list(raw_df.index)

    def test_map_keypoints_empty_mapping(self, raw_df: pd.DataFrame):
        result = map_keypoints(raw_df, {}, {})

        assert result.shape == (len(raw_df), 0)
        assert result.columns.nlevels == 3

    def test_map_keypoints_unknown_source(self, raw_df: pd.DataFrame):
        with pytest.raises(KeyError):
            map_keypoints(raw_df, {"whisker": "whisker"}, {})


class TestReindexToCanonical:
    """Test the function reindex_to_canonical."""

    def test_reindex_to_canonical_order(self, make_processed_df: Callable):
        df = make_processed_df({"b": [2.0], "a": [2.0]})

        result = reindex_to_canonical(df, ["a", "b"])

        assert list(result.columns.get_level_values(1)[::3]) == ["a", "b"]

    def test_reindex_to_canonical_absent_keypoint(self, make_processed_df: Callable):
        df = make_processed_df({"a": [2.0, 1.0]})

        result = reindex_to_canonical(df, ["a", "b"])

        assert _vis(result, "b") == [0.0, 0.0]
        assert np.isnan(_xy(result, "b")).all()
        assert _vis(result, "a") == [2.0, 1.0]

    def test_reindex_to_canonical_drops_non_canonical(self, make_processed_df: Callable):
        df = make_processed_df({"a": [2.0], "extra": [2.0]})

        result = reindex_to_canonical(df, ["a"])

        assert set(result.columns.get_level_values(1)) == {"a"}

    def test_reindex_to_canonical_column_names(self, make_processed_df: Callable):
        result = reindex_to_canonical(make_processed_df({"a": [2.0]}), ["a"])

        assert result.columns.names == ["scorer", "bodyparts", "coords"]


class TestRemapIndex:
    """Test the function remap_index."""

    def test_remap_index(self):
        index = ["labeled-data/sessA/img0.png", "some/other/root/sessB/img3.png"]

        result = remap_index(index, "toy")

        assert list(result) == [
            "labeled-data/toy/sessA/img0.png",
            "labeled-data/toy/sessB/img3.png",
        ]


class TestProcessSplit:
    """Test the function process_split."""

    def test_process_split_end_to_end(
        self,
        raw_df: pd.DataFrame,
        dataset_config: dict,
        canonical_kps: list[str],
    ):
        result = process_split(raw_df, dataset_config, "toy", canonical_kps)

        assert list(result.index) == [
            "labeled-data/toy/sessA/img0.png",
            "labeled-data/toy/sessA/img1.png",
            "labeled-data/toy/sessB/img0.png",
        ]
        assert list(result.columns.get_level_values(1)[::3]) == canonical_kps
        assert _vis(result, "nose") == [2.0, 1.0, 2.0]
        assert _vis(result, "paw_left") == [2.0, 2.0, 1.0]
        assert _vis(result, "paw_right") == [1.0, 1.0, 1.0]
        assert _vis(result, "ear") == [0.0, 0.0, 0.0]
        np.testing.assert_array_equal(
            _xy(result, "paw_left"), [[3.0, 4.0], [7.0, 8.0], [NAN, NAN]],
        )

    def test_process_split_excluded_keypoint_not_mapped(
        self,
        raw_df: pd.DataFrame,
        dataset_config: dict,
        canonical_kps: list[str],
    ):
        # tail is excluded, so its mapping onto a canonical name must be ignored
        dataset_config["keypoints"]["tail"] = "ear"

        result = process_split(raw_df, dataset_config, "toy", canonical_kps)

        assert _vis(result, "ear") == [0.0, 0.0, 0.0]

    def test_process_split_does_not_post_process(
        self,
        raw_df: pd.DataFrame,
        dataset_config: dict,
        canonical_kps: list[str],
    ):
        hook = MagicMock()

        with patch.dict("mighty_mouse.convert.POST_PROCESS", {"toy": hook}):
            process_split(raw_df, dataset_config, "toy", canonical_kps)

        hook.assert_not_called()


class TestCopyImages:
    """Test the function copy_images."""

    @pytest.fixture
    def raw_dir(self, tmp_path: Path) -> Path:
        raw = tmp_path / "raw"
        for session in ("sessA", "sessB"):
            (raw / "labeled-data" / session).mkdir(parents=True)
            (raw / "labeled-data" / session / "img0.png").write_bytes(session.encode())
        return raw

    @pytest.fixture
    def index(self) -> list[str]:
        return ["labeled-data/toy/sessA/img0.png", "labeled-data/toy/sessB/img0.png"]

    def test_copy_images_copies(self, tmp_path: Path, raw_dir: Path, index: list[str]):
        out = tmp_path / "out"

        n_created = copy_images(index, raw_dir, out)

        assert n_created == 2
        dst = out / "labeled-data/toy/sessA/img0.png"
        assert dst.read_bytes() == b"sessA"
        assert not dst.is_symlink()

    def test_copy_images_skips_existing(self, tmp_path: Path, raw_dir: Path, index: list[str]):
        out = tmp_path / "out"
        copy_images(index[:1], raw_dir, out)

        n_created = copy_images(index, raw_dir, out)

        assert n_created == 1

    def test_copy_images_link(self, tmp_path: Path, raw_dir: Path, index: list[str]):
        out = tmp_path / "out"

        n_created = copy_images(index, raw_dir, out, link=True)

        assert n_created == 2
        dst = out / "labeled-data/toy/sessB/img0.png"
        assert dst.is_symlink()
        assert dst.resolve() == (raw_dir / "labeled-data/sessB/img0.png").resolve()

    def test_copy_images_link_resolves_symlinked_raw(
        self,
        tmp_path: Path,
        raw_dir: Path,
        index: list[str],
    ):
        # a raw folder whose labeled-data is itself a symlink to the original frames
        linked_raw = tmp_path / "raw-v2"
        linked_raw.mkdir()
        (linked_raw / "labeled-data").symlink_to(raw_dir / "labeled-data")
        out = tmp_path / "out"

        copy_images(index, linked_raw, out, link=True)

        target = Path((out / index[0]).readlink())
        assert target == (raw_dir / "labeled-data/sessA/img0.png").resolve()

    def test_copy_images_missing_source(self, tmp_path: Path, raw_dir: Path):
        with pytest.raises(FileNotFoundError):
            copy_images(["labeled-data/toy/sessZ/img0.png"], raw_dir, tmp_path / "out")
