"""Tests for SumTree data structure."""
import numpy as np
import pytest

from src.training.sum_tree import SumTree


# ============================================================================
# Basic Construction
# ============================================================================


class TestSumTreeInit:
    """Tests for SumTree initialization."""

    def test_init_creates_correct_tree_size(self):
        tree = SumTree(capacity=8)
        assert len(tree.tree) == 2 * 8 - 1  # 15 nodes

    def test_init_creates_correct_data_size(self):
        tree = SumTree(capacity=8)
        assert len(tree.data) == 8

    def test_init_empty_tree_has_zero_total(self):
        tree = SumTree(capacity=4)
        assert tree.total() == 0.0

    def test_init_empty_tree_has_zero_length(self):
        tree = SumTree(capacity=4)
        assert len(tree) == 0

    def test_init_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            SumTree(capacity=0)
        with pytest.raises(ValueError):
            SumTree(capacity=-1)

    def test_init_capacity_one(self):
        tree = SumTree(capacity=1)
        assert len(tree.tree) == 1
        tree.add(5.0, "only")
        assert tree.total() == 5.0
        assert len(tree) == 1


# ============================================================================
# Add and Retrieve
# ============================================================================


class TestSumTreeAdd:
    """Tests for adding items and retrieving them."""

    def test_add_single_item(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "data_0")
        assert len(tree) == 1
        assert tree.total() == 1.0

    def test_add_multiple_items(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        assert len(tree) == 3
        assert tree.total() == pytest.approx(6.0)

    def test_add_fills_to_capacity(self):
        tree = SumTree(capacity=4)
        for i in range(4):
            tree.add(float(i + 1), f"data_{i}")
        assert len(tree) == 4
        assert tree.total() == pytest.approx(10.0)  # 1+2+3+4

    def test_get_retrieves_correct_data(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        # cumsum=0.5 should fall in first leaf (priority=1.0)
        idx, priority, data = tree.get(0.5)
        assert data == "a"
        assert priority == pytest.approx(1.0)
        assert idx == 0

    def test_get_boundary_between_leaves(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        # cumsum=1.0 should land on "a" (left child <= check)
        idx, priority, data = tree.get(1.0)
        assert data == "a"

    def test_get_second_leaf(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        # cumsum=1.5 should be in second leaf (1.0 < 1.5 <= 3.0)
        idx, priority, data = tree.get(1.5)
        assert data == "b"
        assert priority == pytest.approx(2.0)

    def test_get_last_leaf(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        # cumsum=5.5 should be in third leaf (3.0 < 5.5 <= 6.0)
        idx, priority, data = tree.get(5.5)
        assert data == "c"
        assert priority == pytest.approx(3.0)


# ============================================================================
# Ring Buffer Overwrite
# ============================================================================


class TestSumTreeRingBuffer:
    """Tests for ring buffer overwrite behavior."""

    def test_overwrite_oldest_entry(self):
        tree = SumTree(capacity=4)
        for i in range(4):
            tree.add(1.0, f"old_{i}")
        # Now add a 5th item — should overwrite position 0
        tree.add(10.0, "new_0")
        assert len(tree) == 4
        # Total: was 4.0, removed 1.0, added 10.0 -> 13.0
        assert tree.total() == pytest.approx(13.0)
        # Retrieving at cumsum near 0 should get the new item
        idx, priority, data = tree.get(0.1)
        assert data == "new_0"
        assert priority == pytest.approx(10.0)

    def test_overwrite_wraps_around(self):
        tree = SumTree(capacity=3)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        # Overwrite "a" at position 0
        tree.add(4.0, "d")
        # Overwrite "b" at position 1
        tree.add(5.0, "e")
        assert len(tree) == 3
        # Active data: d(4), e(5), c(3)
        assert tree.total() == pytest.approx(12.0)

    def test_overwrite_updates_size_correctly(self):
        tree = SumTree(capacity=2)
        tree.add(1.0, "a")
        assert len(tree) == 1
        tree.add(2.0, "b")
        assert len(tree) == 2
        tree.add(3.0, "c")  # overwrite
        assert len(tree) == 2  # still capped


# ============================================================================
# Priority Update
# ============================================================================


class TestSumTreeUpdate:
    """Tests for priority update propagation."""

    def test_update_changes_priority(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.update(0, 5.0)
        assert tree.total() == pytest.approx(7.0)  # 5+2

    def test_update_propagates_to_root(self):
        tree = SumTree(capacity=8)
        for i in range(8):
            tree.add(1.0, f"d{i}")
        assert tree.total() == pytest.approx(8.0)
        tree.update(3, 10.0)
        assert tree.total() == pytest.approx(17.0)  # 7*1 + 10

    def test_update_out_of_range_raises(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        with pytest.raises(IndexError):
            tree.update(4, 1.0)
        with pytest.raises(IndexError):
            tree.update(-1, 1.0)

    def test_update_affects_sampling(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(1.0, "b")
        # Set "a" priority very high
        tree.update(0, 100.0)
        # Almost all cumulative sums should land on "a"
        _, _, data = tree.get(50.0)
        assert data == "a"


# ============================================================================
# Total and Min/Max Priority
# ============================================================================


class TestSumTreeAggregates:
    """Tests for total(), min_priority(), max_priority."""

    def test_total_matches_sum_of_priorities(self):
        tree = SumTree(capacity=8)
        priorities = [0.5, 1.2, 3.4, 0.1, 2.7, 0.8, 1.1, 4.0]
        for i, p in enumerate(priorities):
            tree.add(p, i)
        assert tree.total() == pytest.approx(sum(priorities))

    def test_min_priority_empty(self):
        tree = SumTree(capacity=4)
        assert tree.min_priority() == 0.0

    def test_min_priority_single(self):
        tree = SumTree(capacity=4)
        tree.add(3.0, "a")
        assert tree.min_priority() == pytest.approx(3.0)

    def test_min_priority_tracks_minimum(self):
        tree = SumTree(capacity=4)
        tree.add(5.0, "a")
        tree.add(2.0, "b")
        tree.add(8.0, "c")
        assert tree.min_priority() == pytest.approx(2.0)

    def test_min_priority_after_update(self):
        tree = SumTree(capacity=4)
        tree.add(5.0, "a")
        tree.add(2.0, "b")
        tree.update(1, 10.0)
        assert tree.min_priority() == pytest.approx(5.0)

    def test_min_priority_with_overwrite(self):
        tree = SumTree(capacity=2)
        tree.add(1.0, "a")
        tree.add(5.0, "b")
        assert tree.min_priority() == pytest.approx(1.0)
        tree.add(10.0, "c")  # overwrites "a" (priority 1.0)
        assert tree.min_priority() == pytest.approx(5.0)

    def test_max_priority_tracks_max_seen(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(5.0, "b")
        tree.add(2.0, "c")
        assert tree.max_priority == pytest.approx(5.0)

    def test_max_priority_default(self):
        tree = SumTree(capacity=4)
        assert tree.max_priority == 1.0  # Default max priority

    def test_max_priority_updates_on_add(self):
        tree = SumTree(capacity=4)
        tree.add(3.0, "a")
        assert tree.max_priority == pytest.approx(3.0)
        tree.add(1.0, "b")
        assert tree.max_priority == pytest.approx(3.0)  # doesn't decrease
        tree.add(7.0, "c")
        assert tree.max_priority == pytest.approx(7.0)

    def test_max_priority_updates_on_update(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.update(0, 20.0)
        assert tree.max_priority == pytest.approx(20.0)


# ============================================================================
# Sampling Distribution (Statistical)
# ============================================================================


class TestSumTreeSamplingDistribution:
    """Statistical tests that sampling respects priorities."""

    def test_sampling_proportional_to_priorities(self):
        """Sample many times and check distribution roughly matches priorities."""
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")  # 10%
        tree.add(2.0, "b")  # 20%
        tree.add(3.0, "c")  # 30%
        tree.add(4.0, "d")  # 40%

        counts = {"a": 0, "b": 0, "c": 0, "d": 0}
        n_samples = 10000
        rng = np.random.default_rng(42)

        for _ in range(n_samples):
            s = rng.uniform(0, tree.total())
            _, _, data = tree.get(s)
            counts[data] += 1

        total_count = sum(counts.values())
        # Check proportions within tolerance
        assert counts["a"] / total_count == pytest.approx(0.1, abs=0.02)
        assert counts["b"] / total_count == pytest.approx(0.2, abs=0.02)
        assert counts["c"] / total_count == pytest.approx(0.3, abs=0.02)
        assert counts["d"] / total_count == pytest.approx(0.4, abs=0.02)

    def test_stratified_sampling(self):
        """Test stratified sampling pattern used in PER."""
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(1.0, "b")
        tree.add(1.0, "c")
        tree.add(1.0, "d")

        batch_size = 4
        segment = tree.total() / batch_size
        rng = np.random.default_rng(42)

        sampled_data = set()
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = rng.uniform(low, high)
            _, _, data = tree.get(s)
            sampled_data.add(data)

        # With uniform priorities and stratified sampling, should get all 4
        assert sampled_data == {"a", "b", "c", "d"}


# ============================================================================
# Edge Cases
# ============================================================================


class TestSumTreeEdgeCases:
    """Tests for edge cases."""

    def test_zero_priority(self):
        tree = SumTree(capacity=4)
        tree.add(0.0, "zero")
        tree.add(1.0, "one")
        assert tree.total() == pytest.approx(1.0)
        # Sampling should always return "one" since "zero" has no priority
        _, _, data = tree.get(0.5)
        assert data == "one"

    def test_all_zero_priorities(self):
        tree = SumTree(capacity=4)
        tree.add(0.0, "a")
        tree.add(0.0, "b")
        assert tree.total() == pytest.approx(0.0)

    def test_very_small_priorities(self):
        tree = SumTree(capacity=4)
        tree.add(1e-10, "a")
        tree.add(1e-10, "b")
        assert tree.total() == pytest.approx(2e-10)

    def test_very_large_priorities(self):
        tree = SumTree(capacity=4)
        tree.add(1e15, "a")
        tree.add(1e15, "b")
        assert tree.total() == pytest.approx(2e15)

    def test_single_element_get(self):
        tree = SumTree(capacity=1)
        tree.add(5.0, "only")
        idx, priority, data = tree.get(2.5)
        assert data == "only"
        assert priority == pytest.approx(5.0)
        assert idx == 0

    def test_large_capacity(self):
        capacity = 1024
        tree = SumTree(capacity=capacity)
        for i in range(capacity):
            tree.add(float(i + 1), i)
        expected_total = capacity * (capacity + 1) / 2
        assert tree.total() == pytest.approx(expected_total)
        assert len(tree) == capacity

    def test_data_can_be_any_type(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, {"state": [1, 2, 3], "action": 0})
        tree.add(1.0, (1, 2, 3))
        tree.add(1.0, 42)
        tree.add(1.0, None)
        assert len(tree) == 4

    def test_repeated_overwrites(self):
        """Ensure many cycles of overwrites maintain consistency."""
        tree = SumTree(capacity=4)
        for i in range(100):
            tree.add(float(i + 1), f"item_{i}")
        assert len(tree) == 4
        # Last 4 items added: 97, 98, 99, 100
        assert tree.total() == pytest.approx(97.0 + 98.0 + 99.0 + 100.0)

    def test_min_priority_with_unfilled_buffer(self):
        """Min should only consider actual entries, not empty slots."""
        tree = SumTree(capacity=8)
        tree.add(5.0, "a")
        tree.add(3.0, "b")
        # Empty slots have inf in min-tree; but they also have 0.0 priority.
        # Min of actual entries should be 3.0
        # However, empty leaf slots might have 0 priority in sum-tree...
        # The min-tree was initialized with inf, and only written slots get real values
        assert tree.min_priority() == pytest.approx(3.0)

    def test_get_returns_valid_data_index(self):
        """Data index from get() should be usable with update()."""
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        tree.add(3.0, "c")
        idx, _, _ = tree.get(4.0)
        # Should be able to update without error
        tree.update(idx, 10.0)
        assert tree.total() == pytest.approx(1.0 + 2.0 + 10.0)  # a + b + new_c
