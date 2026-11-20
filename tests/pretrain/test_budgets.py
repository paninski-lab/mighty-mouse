"""Tests for mighty_mouse.pretrain.budgets."""

import pytest

from mighty_mouse.pretrain.budgets import dataset_budgets


class TestDatasetBudgets:
    """Test the function dataset_budgets."""

    def test_dataset_budgets_sums_to_total(self):
        raw = {"a": 1_000_000, "b": 100_000, "c": 5_000}
        pool = {"a": 500_000, "b": 90_000, "c": 4_000}
        budgets = dataset_budgets(raw, pool, 100_000, alpha=0.3)
        assert sum(budgets.values()) == 100_000

    def test_dataset_budgets_alpha_zero_is_equal_shares(self):
        raw = {"a": 1000, "b": 10}
        pool = {"a": 1000, "b": 1000}
        assert dataset_budgets(raw, pool, 100, alpha=0.0) == {"a": 50, "b": 50}

    def test_dataset_budgets_alpha_one_is_proportional(self):
        raw = {"a": 3000, "b": 1000}
        pool = {"a": 3000, "b": 1000}
        assert dataset_budgets(raw, pool, 400, alpha=1.0) == {"a": 300, "b": 100}

    def test_dataset_budgets_bigger_dataset_gets_more_but_not_proportionally(self):
        raw = {"a": 100_000, "b": 1_000}
        pool = {"a": 100_000, "b": 1_000}
        budgets = dataset_budgets(raw, pool, 1000, alpha=0.3)
        assert budgets["a"] > budgets["b"]
        assert budgets["a"] / budgets["b"] < 100

    def test_dataset_budgets_cap_redistributes_surplus(self):
        raw = {"a": 1000, "b": 1000, "c": 1000}
        pool = {"a": 10, "b": 1000, "c": 1000}
        budgets = dataset_budgets(raw, pool, 300, alpha=0.3)
        assert budgets == {"a": 10, "b": 145, "c": 145}

    def test_dataset_budgets_pools_smaller_than_total_take_everything(self):
        raw = {"a": 1000, "b": 500}
        pool = {"a": 30, "b": 20}
        assert dataset_budgets(raw, pool, 100) == {"a": 30, "b": 20}

    def test_dataset_budgets_never_exceed_pool(self):
        raw = {"a": 10**6, "b": 10**3, "c": 10**2}
        pool = {"a": 1000, "b": 50, "c": 20}
        budgets = dataset_budgets(raw, pool, 500)
        assert all(budgets[d] <= pool[d] for d in pool)
        assert sum(budgets.values()) == 500

    def test_dataset_budgets_mismatched_keys_raise(self):
        with pytest.raises(ValueError, match="same datasets"):
            dataset_budgets({"a": 1}, {"b": 1}, 1)

    def test_dataset_budgets_bad_total_raises(self):
        with pytest.raises(ValueError, match="total"):
            dataset_budgets({"a": 5}, {"a": 5}, 0)

    def test_dataset_budgets_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            dataset_budgets({"a": 5}, {"a": 5}, 3, alpha=-1)

    def test_dataset_budgets_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="without frames"):
            dataset_budgets({"a": 5, "b": 0}, {"a": 5, "b": 0}, 3)
