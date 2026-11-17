"""Tests for mighty_mouse.preprocess.history."""

from pathlib import Path
from unittest.mock import patch

from mighty_mouse.preprocess.history import (
    corpus_version_of,
    parse_history,
    record_corpus_version,
    render_history,
)

DATASETS = ["ds", "other"]


class TestCorpusVersionOf:
    """Test the function corpus_version_of."""

    def test_corpus_version_of_versioned_dir(self):
        assert corpus_version_of(Path("data/head-fixed_v12")) == "v12"

    def test_corpus_version_of_unversioned_dir(self):
        assert corpus_version_of(Path("data/head-fixed")) is None


class TestRenderAndParseHistory:
    """Test the functions render_history and parse_history."""

    def test_render_history_orders_versions_numerically_and_marks_missing(self):
        with patch("mighty_mouse.preprocess.history.ALL_DATASETS", DATASETS):
            text = render_history({"v10": {"ds": "2", "other": "1"}, "v9": {"ds": "h1"}})
        assert "| dataset | v9 | v10 |" in text
        assert "| ds | h1 | 2 |" in text
        assert "| other | – | 1 |" in text

    def test_parse_history_round_trips_render_history(self):
        history = {"v1": {"ds": "h1"}, "v2": {"ds": "1", "other": "0"}}
        with patch("mighty_mouse.preprocess.history.ALL_DATASETS", DATASETS):
            assert parse_history(render_history(history)) == history

    def test_parse_history_empty_text(self):
        assert parse_history("") == {}


class TestRecordCorpusVersion:
    """Test the function record_corpus_version."""

    def record(self, tmp_path, data_name, out, versions):
        with patch("mighty_mouse.preprocess.history.dataset_versions", return_value=versions), \
             patch("mighty_mouse.preprocess.history.ALL_DATASETS", DATASETS):
            return record_corpus_version(tmp_path / data_name, tmp_path / "raw", out_path=out)

    def test_record_corpus_version_adds_a_column(self, tmp_path):
        out = tmp_path / "history.md"
        self.record(tmp_path, "head-fixed_v1", out, {"ds": "1", "other": "0"})
        self.record(tmp_path, "head-fixed_v2", out, {"ds": "2", "other": "0"})
        with patch("mighty_mouse.preprocess.history.ALL_DATASETS", DATASETS):
            assert parse_history(out.read_text()) == {
                "v1": {"ds": "1", "other": "0"}, "v2": {"ds": "2", "other": "0"}}

    def test_record_corpus_version_overwrites_a_rebuilt_version(self, tmp_path):
        out = tmp_path / "history.md"
        self.record(tmp_path, "head-fixed_v1", out, {"ds": "1", "other": "0"})
        self.record(tmp_path, "head-fixed_v1", out, {"ds": "2", "other": "0"})
        assert parse_history(out.read_text()) == {"v1": {"ds": "2", "other": "0"}}

    def test_record_corpus_version_skips_unversioned_dir(self, tmp_path):
        out = tmp_path / "history.md"
        assert self.record(tmp_path, "head-fixed", out, {"ds": "1"}) is None
        assert not out.exists()
