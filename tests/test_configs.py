"""Test the mighty_mouse.configs module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mighty_mouse.configs import (
    check_model_config,
    collapse_side,
    default_configs_dir,
    hflip_partner,
    load_canonical_keypoints,
    load_dataset_config,
    load_model_config,
    load_regions,
    normalize_dataset_config,
)


class TestDefaultConfigsDir:
    """Test the function default_configs_dir."""

    def test_default_configs_dir_under_repo_root(self, tmp_path: Path):
        with patch("mighty_mouse.configs.repo_root", return_value=tmp_path):
            assert default_configs_dir() == tmp_path / "configs"


class TestLoadCanonicalKeypoints:
    """Test the function load_canonical_keypoints."""

    def test_load_canonical_keypoints(self, configs_dir: Path, canonical_kps: list[str]):
        assert load_canonical_keypoints(configs_dir) == canonical_kps

    def test_load_canonical_keypoints_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_canonical_keypoints(tmp_path)


class TestLoadModelConfig:
    """Test the function load_model_config."""

    def test_load_model_config(self, configs_dir: Path, model_cfg: dict):
        assert load_model_config(configs_dir) == model_cfg

    def test_load_model_config_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_model_config(tmp_path)


class TestNormalizeDatasetConfig:
    """Test the function normalize_dataset_config."""

    def test_normalize_dataset_config_fills_missing_fields(self):
        result = normalize_dataset_config({})

        assert result == {"exclude": {"sessions": [], "keypoints": []}, "keypoints": {}}

    def test_normalize_dataset_config_replaces_nulls(self):
        cfg = {"exclude": {"sessions": None, "keypoints": None}, "keypoints": None}

        result = normalize_dataset_config(cfg)

        assert result == {"exclude": {"sessions": [], "keypoints": []}, "keypoints": {}}

    def test_normalize_dataset_config_null_exclude(self):
        result = normalize_dataset_config({"exclude": None})

        assert result["exclude"] == {"sessions": [], "keypoints": []}

    def test_normalize_dataset_config_preserves_values(self, dataset_config: dict):
        expected = {
            "exclude": {"sessions": ["sessX"], "keypoints": ["tail"]},
            "keypoints": {"nose": "nose", "paw": "paw_{side}"},
            "sessions": {"sessA": "left", "sessB": "right"},
        }

        result = normalize_dataset_config(dataset_config)

        assert result == expected

    def test_normalize_dataset_config_in_place(self):
        cfg: dict = {}

        result = normalize_dataset_config(cfg)

        assert result is cfg


class TestLoadDatasetConfig:
    """Test the function load_dataset_config."""

    def test_load_dataset_config_normalizes(self, configs_dir: Path):
        result = load_dataset_config(configs_dir, "toy")

        assert result["exclude"] == {"sessions": [], "keypoints": ["tail"]}
        assert result["keypoints"] == {"nose": "nose", "paw": "paw_{side}"}
        assert result["sessions"] == {"sessA": "left"}

    def test_load_dataset_config_empty_file(self, configs_dir: Path):
        (configs_dir / "datasets" / "empty.yaml").write_text("")

        result = load_dataset_config(configs_dir, "empty")

        assert result == {"exclude": {"sessions": [], "keypoints": []}, "keypoints": {}}

    def test_load_dataset_config_unknown_dataset(self, configs_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset_config(configs_dir, "nope")


class TestCheckModelConfig:
    """Test the function check_model_config."""

    def test_check_model_config_consistent(self, model_cfg: dict, canonical_kps: list[str]):
        assert check_model_config(model_cfg, canonical_kps) == []

    def test_check_model_config_missing_keypoint(self, canonical_kps: list[str]):
        model_cfg = {"data": {"keypoint_names": canonical_kps[:-1], "num_keypoints": 3}}

        errors = check_model_config(model_cfg, canonical_kps)

        assert len(errors) == 2
        assert "missing ['ear']" in errors[0]
        assert "num_keypoints=3" in errors[1]

    def test_check_model_config_extra_keypoint(self, canonical_kps: list[str]):
        names = [*canonical_kps, "tail"]
        model_cfg = {"data": {"keypoint_names": names, "num_keypoints": len(canonical_kps)}}

        errors = check_model_config(model_cfg, canonical_kps)

        assert errors == ["model.yaml: keypoint_names has unknown keypoint(s) ['tail']"]

    def test_check_model_config_reordered(self, canonical_kps: list[str]):
        # the silent failure mode: same set, different order
        names = list(reversed(canonical_kps))
        model_cfg = {"data": {"keypoint_names": names, "num_keypoints": len(canonical_kps)}}

        errors = check_model_config(model_cfg, canonical_kps)

        assert len(errors) == 1
        assert "different order" in errors[0]

    def test_check_model_config_num_keypoints_mismatch(self, canonical_kps: list[str]):
        model_cfg = {"data": {"keypoint_names": list(canonical_kps), "num_keypoints": 99}}

        errors = check_model_config(model_cfg, canonical_kps)

        assert errors == ["model.yaml: num_keypoints=99 but keypoints.yaml has 4 keypoints"]


class TestCollapseSide:
    """Test the function collapse_side."""

    def test_collapse_side_strips_suffix(self):
        assert collapse_side("ear_tip_left") == collapse_side("ear_tip_right") == "ear_tip"

    def test_collapse_side_keeps_midline_and_inner_side_words(self):
        assert collapse_side("nose_tip") == "nose_tip"
        assert collapse_side("left_ear_tip") == "left_ear_tip"


class TestHflipPartner:
    """Test the function hflip_partner."""

    def test_hflip_partner_swaps_side(self):
        assert hflip_partner("ear_tip_left") == "ear_tip_right"
        assert hflip_partner("ear_tip_right") == "ear_tip_left"

    def test_hflip_partner_none_for_midline(self):
        assert hflip_partner("nose_tip") is None


class TestLoadRegions:
    """Test the function load_regions."""

    def test_load_regions_maps_keypoints_to_header(self, tmp_path):
        (tmp_path / "keypoints.yaml").write_text(
            "keypoints:\n"
            "  # ── eye ──\n"
            "  - eye_top_left\n"
            "  - eye_top_right  # comment\n"
            "  # ── nose ──\n"
            "  - nose_tip\n"
            "  # - commented_out\n"
        )
        assert load_regions(tmp_path) == {
            "eye_top_left": "eye", "eye_top_right": "eye", "nose_tip": "nose",
        }

    def test_load_regions_ignores_keypoints_before_any_header(self, tmp_path):
        (tmp_path / "keypoints.yaml").write_text("keypoints:\n  - orphan\n  # ── a ──\n  - k\n")
        assert load_regions(tmp_path) == {"k": "a"}
