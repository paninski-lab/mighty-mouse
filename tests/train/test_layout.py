from pathlib import Path

import pytest

from mighty_mouse.train.layout import (
    RECIPE,
    parse_run_dir,
    recipe_run_dir,
    recipe_seed_dirs,
    run_dir,
    sweep_args,
    tag_of_train_name,
    train_name,
)


class TestTrainName:
    """Test the functions train_name and tag_of_train_name."""

    def test_train_name_round_trip(self):
        assert train_name("face+ibl") == "face+ibl_train"
        assert tag_of_train_name(train_name("face+ibl")) == "face+ibl"

    def test_tag_of_train_name_rejects_other_names(self):
        with pytest.raises(ValueError):
            tag_of_train_name("face+ibl_test")


class TestRunDir:
    """Test the function run_dir."""

    def test_run_dir_stock_layout(self):
        out = run_dir(Path("/r"), "ibl_train", [], None, 1, "vits_dinov3", 0)
        assert out == Path("/r/ibl_train/supervised/tf1/vits_dinov3/seed0")

    def test_run_dir_temperature_component(self):
        out = run_dir(Path("/r"), "ibl_train", [], 2, 1, "vits_dinov3", 0)
        assert out == Path("/r/ibl_train/supervised/sampling-T2/tf1/vits_dinov3/seed0")

    def test_run_dir_losses_sorted_and_backbone_sanitized(self):
        out = run_dir(Path("/r"), "ibl_train", ["b", "a"], "1", "all", "org/model.v2", "3")
        assert out == Path("/r/ibl_train/a+b/tfall/org_model_v2/seed3")


class TestParseRunDir:
    """Test the function parse_run_dir."""

    def test_parse_run_dir_inverts_run_dir(self):
        out = run_dir(Path("/r/trunks"), "ibl_train", ["a"], 2, 1, "vits_dinov3", 4)
        assert parse_run_dir(out) == {
            "train_name": "ibl_train", "losses_tag": "a", "temperature": "2",
            "train_frames": "1", "backbone": "vits_dinov3", "seed": "4",
        }

    def test_parse_run_dir_stock_has_no_temperature(self):
        out = run_dir(Path("/r"), "ibl_train", [], None, 1, "vits_dinov3", 0)
        parsed = parse_run_dir(out)
        assert parsed["temperature"] is None
        assert parsed["train_name"] == "ibl_train"

    def test_parse_run_dir_independent_of_root_depth(self):
        shallow = parse_run_dir("ibl_train/supervised/tf1/vits_dinov3/seed0")
        deep = parse_run_dir("/a/b/c/d/ibl_train/supervised/tf1/vits_dinov3/seed0")
        assert shallow == deep

    def test_parse_run_dir_rejects_other_paths(self):
        with pytest.raises(ValueError):
            parse_run_dir("/r/ibl_train/eval/ibl")
        with pytest.raises(ValueError):
            parse_run_dir("tf1/vits_dinov3/seed0")


class TestRecipeRunDir:
    """Test the function recipe_run_dir."""

    def test_recipe_run_dir_matches_existing_runs(self):
        # pinned: runs already on disk follow this layout
        out = recipe_run_dir(Path("/r/trunks"), "face+ibl")
        assert out == Path("/r/trunks/face+ibl_train/supervised/sampling-T2/tf1/vits_dinov3/seed0")

    def test_recipe_run_dir_seed(self):
        assert recipe_run_dir(Path("/r"), "ibl", 3).name == "seed3"


class TestRecipeSeedDirs:
    """Test the function recipe_seed_dirs."""

    def test_recipe_seed_dirs_lists_existing_seeds_only(self, tmp_path):
        for seed in (0, 1, 2):
            recipe_run_dir(tmp_path, "ibl", seed).mkdir(parents=True)
        (tmp_path / "ibl_train" / "supervised" / "tf1" / "vits_dinov3" / "seed0").mkdir(
            parents=True)  # a different recipe (no temperature)
        assert recipe_seed_dirs(tmp_path, "ibl") == [recipe_run_dir(tmp_path, "ibl", s)
                                                    for s in (0, 1, 2)]

    def test_recipe_seed_dirs_missing_tag_is_empty(self, tmp_path):
        assert recipe_seed_dirs(tmp_path, "nope") == []


class TestSweepArgs:
    """Test the function sweep_args."""

    def test_sweep_args_follow_recipe(self):
        assert sweep_args() == (
            f"--train_frames {RECIPE['train_frames']} --backbones {RECIPE['backbone']} "
            f"--sampling_temperatures {RECIPE['temperature']}"
        )
        assert sweep_args() == "--train_frames 1 --backbones vits_dinov3 --sampling_temperatures 2"
