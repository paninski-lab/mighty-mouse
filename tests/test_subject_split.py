"""Tests for mighty_mouse.subject_split."""

import pytest

from mighty_mouse.subject_split import subject_of, subject_split


class TestSubjectOf:
    """Test the function subject_of."""

    def test_subject_of_first_token(self):
        assert subject_of("JCR130_20260101_side") == "JCR130"

    def test_subject_of_uppercases(self):
        assert subject_of("jcr130_x") == subject_of("JCR130_x") == "JCR130"

    def test_subject_of_no_delimiter(self):
        assert subject_of("mouse1") == "MOUSE1"

    def test_subject_of_empty(self):
        assert subject_of("") == ""


class TestSubjectSplit:
    """Test the function subject_split."""

    COUNTS = {"A": 10, "B": 20, "C": 30, "D": 40}

    def test_subject_split_partitions_subjects(self):
        train, test = subject_split(self.COUNTS, seed=0, test_fraction=0.3)
        assert train | test == set(self.COUNTS)
        assert train & test == set()

    def test_subject_split_reaches_fraction(self):
        _, test = subject_split(self.COUNTS, seed=0, test_fraction=0.3)
        assert sum(self.COUNTS[s] for s in test) >= 0.3 * sum(self.COUNTS.values())

    def test_subject_split_stops_once_target_met(self):
        # removing the last-added subject must drop below the target
        _, test = subject_split(self.COUNTS, seed=0, test_fraction=0.3)
        target = 0.3 * sum(self.COUNTS.values())
        assert any(sum(self.COUNTS[s] for s in test) - self.COUNTS[s] < target for s in test)

    def test_subject_split_deterministic(self):
        a = subject_split(self.COUNTS, seed=7, test_fraction=0.3)
        b = subject_split(self.COUNTS, seed=7, test_fraction=0.3)
        assert a == b

    def test_subject_split_seed_changes_selection(self):
        tests = {
            frozenset(subject_split(self.COUNTS, seed=s, test_fraction=0.2)[1]) for s in range(20)
        }
        assert len(tests) > 1

    def test_subject_split_zero_fraction(self):
        train, test = subject_split(self.COUNTS, seed=0, test_fraction=0.0)
        assert test == set()
        assert train == set(self.COUNTS)

    def test_subject_split_full_fraction(self):
        train, test = subject_split(self.COUNTS, seed=0, test_fraction=1.0)
        assert train == set()
        assert test == set(self.COUNTS)

    def test_subject_split_empty_counts(self):
        assert subject_split({}, seed=0, test_fraction=0.5) == (set(), set())

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_subject_split_single_subject_overshoots(self, seed):
        train, test = subject_split({"A": 100}, seed=seed, test_fraction=0.1)
        assert test == {"A"}
        assert train == set()
