"""SumTree data structure for O(log N) prioritized experience replay."""
import numpy as np
from typing import Any, Optional, Tuple


class SumTree:
    """
    Binary sum-tree stored in a flat numpy array for O(log N) prioritized sampling.

    Internal nodes store the sum of their children. Leaf nodes store individual
    priorities. Data is stored in a separate array with ring buffer semantics.

    Tree layout (capacity=4, array size=7):
        [0]          <- root = sum of all priorities
       /    \\
     [1]    [2]      <- internal nodes
    /  \\   /  \\
   [3] [4] [5] [6]   <- leaf nodes (indices capacity-1 to 2*capacity-2)
    """

    def __init__(self, capacity: int):
        """
        Initialize SumTree with fixed capacity.

        Args:
            capacity: Maximum number of leaf entries (experiences).
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.position = 0
        self.size = 0
        self._max_priority = 1.0
        self._min_tree: Optional[np.ndarray] = None
        # Separate min-tree for O(1) min queries
        self._min_tree = np.full(2 * capacity - 1, float("inf"), dtype=np.float64)

    def _leaf_index(self, data_index: int) -> int:
        """Convert a data array index to its corresponding tree leaf index."""
        return data_index + self.capacity - 1

    def _propagate_up(self, tree_index: int) -> None:
        """Propagate a priority change up to the root (sum-tree)."""
        parent = (tree_index - 1) // 2
        while parent >= 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self.tree[parent] = self.tree[left] + self.tree[right]
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _propagate_min_up(self, tree_index: int) -> None:
        """Propagate a priority change up to the root (min-tree)."""
        parent = (tree_index - 1) // 2
        while parent >= 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self._min_tree[parent] = min(self._min_tree[left], self._min_tree[right])
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, priority: float, data: Any) -> None:
        """
        Add an experience with given priority, using ring buffer semantics.

        Args:
            priority: Priority value for this experience (must be >= 0).
            data: The experience data to store.
        """
        tree_index = self._leaf_index(self.position)

        self.data[self.position] = data
        self._update_node(tree_index, priority)

        # Track max priority
        if priority > self._max_priority:
            self._max_priority = priority

        # Advance ring buffer pointer
        self.position = (self.position + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def _update_node(self, tree_index: int, priority: float) -> None:
        """Update a leaf node and propagate changes to both trees."""
        self.tree[tree_index] = priority
        self._propagate_up(tree_index)

        self._min_tree[tree_index] = priority
        self._propagate_min_up(tree_index)

    def update(self, data_index: int, priority: float) -> None:
        """
        Update the priority of an existing entry.

        Args:
            data_index: Index in the data array (0 to capacity-1).
            priority: New priority value.
        """
        if data_index < 0 or data_index >= self.capacity:
            raise IndexError(
                f"Data index {data_index} out of range [0, {self.capacity})"
            )

        tree_index = self._leaf_index(data_index)
        self._update_node(tree_index, priority)

        if priority > self._max_priority:
            self._max_priority = priority

    def get(self, cumulative_sum: float) -> Tuple[int, float, Any]:
        """
        Find the leaf corresponding to a cumulative sum value.

        Traverses from root to leaf: go left if the cumulative sum is less than
        or equal to the left child's value, otherwise subtract left and go right.

        Args:
            cumulative_sum: A value in [0, total()).

        Returns:
            Tuple of (data_index, priority, data).
        """
        node = 0  # Start at root

        while True:
            left = 2 * node + 1
            right = 2 * node + 2

            # Reached a leaf
            if left >= len(self.tree):
                break

            if cumulative_sum <= self.tree[left]:
                node = left
            else:
                cumulative_sum -= self.tree[left]
                node = right

        data_index = node - (self.capacity - 1)
        return data_index, self.tree[node], self.data[data_index]

    def total(self) -> float:
        """Return the total sum of all priorities (root node value). O(1)."""
        return float(self.tree[0])

    def min_priority(self) -> float:
        """
        Return the minimum priority among stored entries. O(1).

        Returns:
            Minimum priority, or 0.0 if tree is empty.
        """
        if self.size == 0:
            return 0.0
        val = float(self._min_tree[0])
        if val == float("inf"):
            return 0.0
        return val

    @property
    def max_priority(self) -> float:
        """Return the maximum priority ever seen (for default priority of new experiences)."""
        return self._max_priority

    def __len__(self) -> int:
        """Return current number of stored items."""
        return self.size
