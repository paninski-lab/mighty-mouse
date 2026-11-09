"""Test the mighty_mouse.evaluation.delivery module."""

import pytest

from mighty_mouse.evaluation.delivery import (
    KEYS,
    collect_deliveries,
    front_matter,
    front_matter_lines,
    index_markdown,
    parse_front_matter,
)

FIELDS = dict(title="Evaluation suite 2026-11-10", date="2026-11-10", data_version="head-fixed_v2",
              question="How do the models score?", models="all, LOO-ibl",
              outputs="summary.csv, per_keypoint.csv", finding="(fill in)")


class TestFrontMatterLines:
    """Test the functions front_matter_lines and front_matter."""

    def test_front_matter_lines_fenced_keys_in_order(self):
        lines = front_matter_lines(**FIELDS)
        assert lines[0] == lines[-1] == "---"
        assert [line.split(":")[0] for line in lines[1:-1]] == list(KEYS)
        assert lines[1] == "title: Evaluation suite 2026-11-10"
        assert lines[-2] == "finding: (fill in)"

    def test_front_matter_text_ends_with_newline(self):
        text = front_matter(**FIELDS)
        assert text == "\n".join(front_matter_lines(**FIELDS)) + "\n"

    def test_front_matter_lines_multiline_value_raises(self):
        with pytest.raises(ValueError, match="question"):
            front_matter_lines(**{**FIELDS, "question": "a\nb"})


class TestParseFrontMatter:
    """Test the function parse_front_matter."""

    def test_parse_front_matter_round_trip(self):
        assert parse_front_matter(front_matter(**FIELDS) + "\n# body\n") == FIELDS

    def test_parse_front_matter_value_may_contain_colons(self):
        text = "---\nquestion: Why: because: yes\n---\n"
        assert parse_front_matter(text) == {"question": "Why: because: yes"}

    def test_parse_front_matter_none_without_block(self):
        assert parse_front_matter("# just a title\n") is None
        assert parse_front_matter("---\ntitle: T\n") is None      # unterminated
        assert parse_front_matter("---\n---\nbody\n") is None     # empty


class TestCollectDeliveries:
    """Test the function collect_deliveries."""

    def test_collect_deliveries_sorted_with_and_without_front_matter(self, tmp_path):
        for name, text in (("b", front_matter(**FIELDS)), ("a", "# no front matter\n")):
            (tmp_path / name).mkdir()
            (tmp_path / name / "README.md").write_text(text)
        (tmp_path / "c").mkdir()                                    # no README
        out = collect_deliveries(tmp_path)
        assert [name for name, _ in out] == ["a", "b", "c"]
        assert [fm for _, fm in out] == [None, FIELDS, None]


class TestIndexMarkdown:
    """Test the function index_markdown."""

    def test_index_markdown_row_per_delivery_and_missing_list(self):
        fm = {**FIELDS, "question": "a | b?"}
        text = index_markdown("head-fixed_v2", [("x-11-10", fm), ("y", None)])
        assert text.startswith("# Qualitative evaluations — head-fixed_v2\n")
        assert "| [`x-11-10`](x-11-10/README.md) | 2026-11-10 | a \\| b? | all, LOO-ibl |" in text
        assert text.endswith("Folders without a README front matter (add one): `y`\n")

    def test_index_markdown_all_indexed_has_no_missing_line(self):
        text = index_markdown("r", [("x", FIELDS)])
        assert "Folders without" not in text
