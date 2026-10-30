"""Test the sweep helpers of the mighty_mouse.train.sweep module."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def train():
    """mighty_mouse.train.sweep imported against a fake paths.yaml (it reads paths at import)."""
    sys.modules.pop("mighty_mouse.train.sweep", None)
    paths = {"data_dir": "/data/corpus", "results_dir": "/results"}
    with patch("mighty_mouse.paths.load_paths", return_value=paths), \
         patch("mighty_mouse.paths.repo_root", return_value=Path(__file__).parents[2]):
        module = importlib.import_module("mighty_mouse.train.sweep")
    yield module
    sys.modules.pop("mighty_mouse.train.sweep", None)


def fake_lightning_pose(branch: bool) -> dict[str, types.ModuleType]:
    """sys.modules entries for a lightning-pose with (branch) or without the branch classes."""
    callbacks, samplers = types.ModuleType("callbacks"), types.ModuleType("samplers")
    if branch:
        callbacks.ResumeCheckpoint   = object
        samplers.TemperatureSampler = object
    return {"lightning_pose": types.ModuleType("lightning_pose"),
            "lightning_pose.callbacks": callbacks,
            "lightning_pose.data": types.ModuleType("data"),
            "lightning_pose.data.samplers": samplers}


class TestTemperatureTag:
    """Test the function temperature_tag."""

    def test_temperature_tag_stock_has_no_component(self, train):
        assert train.temperature_tag(None) is None
        assert train.temperature_tag(1) is None
        assert train.temperature_tag("1") is None

    def test_temperature_tag_non_stock(self, train):
        assert train.temperature_tag("2") == "sampling-T2"
        assert train.temperature_tag("inf") == "sampling-Tinf"


class TestMakeOutputDir:
    """Test the function make_output_dir."""

    def test_make_output_dir_stock_layout_unchanged(self, train):
        out = train.make_output_dir("CollectedData_ibl_train.csv", "vits_dinov3", "1", "0", [])
        assert out == Path("/results/ibl_train/supervised/tf1/vits_dinov3/seed0")

    def test_make_output_dir_temperature_component(self, train):
        out = train.make_output_dir("CollectedData_ibl_train.csv", "vits_dinov3", "1", "0", [],
                                    "2")
        assert out == Path("/results/ibl_train/supervised/sampling-T2/tf1/vits_dinov3/seed0")


class TestBuildCombos:
    """Test the function build_combos."""

    def test_build_combos_default_one_stock_run(self, train):
        combos = train.build_combos(["a.csv"], ["b"], ["1"], ["0"])
        assert combos == [("a.csv", "b", "1", "0", None)]

    def test_build_combos_crosses_temperatures(self, train):
        combos = train.build_combos(["a.csv"], ["b"], ["1"], ["0", "1"], ["1", "2"])
        assert len(combos) == 4
        assert {c[4] for c in combos} == {"1", "2"}


class TestNeedsLpBranch:
    """Test the function needs_lp_branch."""

    def test_needs_lp_branch_stock_run(self, train):
        assert not train.needs_lp_branch(None, None, ["training.max_steps=100"])

    def test_needs_lp_branch_temperature(self, train):
        assert train.needs_lp_branch("2")

    def test_needs_lp_branch_override(self, train):
        assert train.needs_lp_branch(None, None, ["+model.head_hidden_channels=256"])

    def test_needs_lp_branch_config_key(self, train, tmp_path):
        cfg = tmp_path / "zoom.yaml"
        cfg.write_text("training:\n  imgaug_per_dataset_zoom:\n    ibl: [-0.4, 0.15]\n")
        assert train.needs_lp_branch(None, cfg)

    def test_needs_lp_branch_null_config_keys_are_stock(self, train, tmp_path):
        cfg = tmp_path / "plain.yaml"
        cfg.write_text("data:\n  dataset_names: null\ntraining:\n  sampling_temperature: null\n")
        assert not train.needs_lp_branch(None, cfg)


class TestCheckLightningPose:
    """Test the function check_lightning_pose."""

    def test_check_lightning_pose_branch_installed(self, train):
        with patch.dict(sys.modules, fake_lightning_pose(branch=True)):
            train.check_lightning_pose()

    def test_check_lightning_pose_stock_exits(self, train):
        with patch.dict(sys.modules, fake_lightning_pose(branch=False)), \
             pytest.raises(SystemExit, match="post_sub_lp_delta"):
            train.check_lightning_pose()


class TestMakeTrainCommand:
    """Test the function make_train_command."""

    @staticmethod
    def overrides(cmd: list[str]) -> list[str]:
        return cmd[cmd.index("--overrides") + 1:]

    def test_make_train_command_stock(self, train):
        cmd = train.make_train_command("CollectedData_ibl_train.csv", "vits_dinov3", "1", "0", [],
                                       Path("/out"))
        assert cmd[:3] == ["litpose", "train", str(train.CONFIG_FILE)]
        assert not any("dataset_names" in o or "sampling" in o for o in self.overrides(cmd))

    def test_make_train_command_temperature(self, train):
        cmd = train.make_train_command("CollectedData_ibl_train.csv", "vits_dinov3", "1", "0", [],
                                       Path("/out"), temperature="2")
        ov = self.overrides(cmd)
        assert f"++data.dataset_names=[{','.join(train.EVAL_DATASETS)}]" in ov
        assert "++training.sampling_temperature='2'" in ov

    def test_make_train_command_zoom_config_passes_datasets(self, train, tmp_path):
        cfg = tmp_path / "zoom.yaml"
        cfg.write_text("training:\n  imgaug_per_dataset_zoom:\n    ibl: [-0.4, 0.15]\n")
        cmd = train.make_train_command("CollectedData_ibl_train.csv", "vits_dinov3", "1", "0", [],
                                       Path("/out"), config_file=cfg)
        assert cmd[2] == str(cfg)
        assert any(o.startswith("++data.dataset_names=") for o in self.overrides(cmd))

    def test_make_train_command_extra_overrides_last(self, train):
        extra = ["+model.head_hidden_channels=256", "training.max_steps=10"]
        cmd = train.make_train_command("CollectedData_ibl_train.csv", "vits_dinov3", "1", "0", [],
                                       Path("/out"), debug=True, extra_overrides=extra)
        assert self.overrides(cmd)[-2:] == extra
        assert "training.check_val_every_n_epoch=null" in self.overrides(cmd)
