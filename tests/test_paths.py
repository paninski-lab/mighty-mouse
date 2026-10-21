"""Tests for mighty_mouse.paths."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mighty_mouse.paths import load_paths, repo_root


class TestRepoRoot:
    """Test the function repo_root."""

    def test_repo_root_finds_paths_yaml(self, tmp_path: Path):
        (tmp_path / "paths.yaml").write_text("raw_dir: x\n")
        fake_file = tmp_path / "pkg" / "paths.py"
        fake_file.parent.mkdir()
        fake_file.touch()
        with patch("mighty_mouse.paths.__file__", str(fake_file)):
            assert repo_root() == tmp_path

    def test_repo_root_picks_nearest_ancestor(self, tmp_path: Path):
        (tmp_path / "paths.yaml").write_text("a: 1\n")
        inner = tmp_path / "inner"
        inner.mkdir()
        (inner / "paths.yaml").write_text("a: 2\n")
        fake_file = inner / "pkg" / "paths.py"
        fake_file.parent.mkdir()
        fake_file.touch()
        with patch("mighty_mouse.paths.__file__", str(fake_file)):
            assert repo_root() == inner

    def test_repo_root_missing(self, tmp_path: Path):
        fake_file = tmp_path / "pkg" / "paths.py"
        fake_file.parent.mkdir()
        fake_file.touch()
        # tmp_path's ancestors could in principle contain a paths.yaml; patch exists() instead
        with (
            patch("mighty_mouse.paths.__file__", str(fake_file)),
            patch.object(Path, "exists", return_value=False),
            pytest.raises(FileNotFoundError, match="paths.yaml"),
        ):
            repo_root()


class TestLoadPaths:
    """Test the function load_paths."""

    def test_load_paths_returns_dict(self, tmp_path: Path):
        (tmp_path / "paths.yaml").write_text("raw_dir: /r\ndata_dir: /d\nresults_dir: /o\n")
        with patch("mighty_mouse.paths.repo_root", return_value=tmp_path):
            assert load_paths() == {"raw_dir": "/r", "data_dir": "/d", "results_dir": "/o"}

    def test_load_paths_missing_file(self, tmp_path: Path):
        with (
            patch("mighty_mouse.paths.repo_root", return_value=tmp_path),
            pytest.raises(FileNotFoundError),
        ):
            load_paths()
