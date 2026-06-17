"""Replay buffers for experience replay in RL algorithms."""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch

from src.core.device_manager import DeviceManager
from src.training.base_buffer import (
    BaseReplayBuffer,
    BatchDict,
    build_batch_dict,
    compute_priority,
    validate_next_action_mask,
)
from src.training.sum_tree import SumTree


def _validate_bulk_field_lengths(states, **fields) -> int:
    """Return row count after ensuring bulk replay fields are aligned."""
    row_count = len(states)
    mismatched = {name: len(values) for name, values in fields.items() if len(values) != row_count}
    if mismatched:
        details = ", ".join(f"{name}={count}" for name, count in sorted(mismatched.items()))
        raise ValueError(f"Replay bulk fields are misaligned: states={row_count}, {details}")
    return row_count


def _coerce_replay_priority(priority, default_priority: float, priority_eps: float) -> float:
    """Validate a replay priority before inserting it into the SumTree."""
    if priority is None:
        priority = default_priority
    if isinstance(priority, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError("priority must be finite and positive")
    try:
        value = float(priority)
    except (TypeError, ValueError) as exc:
        raise ValueError("priority must be finite and positive") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("priority must be finite and positive")
    return max(value, float(priority_eps))


def _unpack_replay_memory(memory_item, default_bootstrap_steps: int = 1):
    """Return a normalized replay tuple from legacy or current serialized memory."""
    if len(memory_item) == 9:
        (
            state,
            action,
            reward,
            next_state,
            done,
            priority,
            bootstrap_steps,
            next_action_mask,
            stream_id,
        ) = memory_item
    elif len(memory_item) == 8:
        (
            state,
            action,
            reward,
            next_state,
            done,
            priority,
            bootstrap_steps,
            next_action_mask,
        ) = memory_item
        stream_id = None
    elif len(memory_item) == 7:
        state, action, reward, next_state, done, priority, bootstrap_steps = memory_item
        next_action_mask = None
        stream_id = None
    elif len(memory_item) == 6:
        state, action, reward, next_state, done, priority = memory_item
        bootstrap_steps = default_bootstrap_steps
        next_action_mask = None
        stream_id = None
    else:
        raise ValueError(
            "Replay memory must have 6 fields "
            "(state, action, reward, next_state, done, priority) "
            "or 7 fields with bootstrap_steps "
            "or 8 fields with next_action_mask "
            "or 9 fields with stream_id"
        )

    return (
        state,
        action,
        reward,
        next_state,
        done,
        priority,
        int(bootstrap_steps),
        next_action_mask,
        stream_id,
    )


def restore_replay_memories(
    memory,
    memories,
    device: torch.device,
    clear: bool = True,
    default_bootstrap_steps: Optional[int] = None,
) -> int:
    """Restore already-materialized replay entries without recomputing n-step returns."""
    if not memories:
        return 0

    if default_bootstrap_steps is None:
        default_bootstrap_steps = int(getattr(memory, "n_step", 1))

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    priorities = []
    bootstrap_steps = []
    next_action_masks = []
    stream_ids = []

    for memory_item in memories:
        state, action, reward, next_state, done, priority, steps, next_action_mask, stream_id = (
            _unpack_replay_memory(
                memory_item,
                default_bootstrap_steps=default_bootstrap_steps,
            )
        )
        states.append(state.to(device) if torch.is_tensor(state) else state)
        next_states.append(next_state.to(device) if torch.is_tensor(next_state) else next_state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        priorities.append(priority)
        bootstrap_steps.append(steps)
        next_action_masks.append(next_action_mask)
        stream_ids.append(stream_id)

    if clear:
        memory.clear()

    if hasattr(memory, "add_bulk"):
        memory.add_bulk(
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=(
                next_action_masks if any(mask is not None for mask in next_action_masks) else None
            ),
            stream_ids=(
                stream_ids if any(stream_id is not None for stream_id in stream_ids) else None
            ),
        )
    else:
        for state, action, reward, next_state, done, priority, steps, next_action_mask in zip(
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps,
            next_action_masks,
        ):
            try:
                memory.add(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    priority,
                    bootstrap_steps=steps,
                    next_action_mask=next_action_mask,
                )
            except TypeError:
                memory.add(state, action, reward, next_state, done)

    return len(states)


class UniformReplayBuffer(BaseReplayBuffer):
    """
    Simple uniform sampling replay buffer for off-policy learning.

    Used as a baseline buffer when prioritized replay is not needed.
    """

    def __init__(self, capacity: int):
        """
        Initialize uniform replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: float = 1.0,
        bootstrap_steps: int = 1,
        next_action_mask=None,
    ) -> None:
        """Add transition to buffer. Priority is ignored for uniform sampling."""
        next_action_mask = validate_next_action_mask(next_action_mask)
        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
                max(1, int(bootstrap_steps)),
                next_action_mask,
            )
        )

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[BatchDict, List[int], torch.Tensor]:
        """
        Sample a batch of transitions uniformly.

        Args:
            batch_size: Number of samples
            device: Device to place tensors on

        Returns:
            Tuple of (batch_dict, indices, weights)
            - batch_dict: Dictionary with 'states', 'actions', 'rewards',
                         'next_states', 'dones' tensors
            - indices: List of sampled indices
            - weights: Uniform weights (all ones) for consistency with prioritized buffers
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} items from buffer "
                f"with only {len(self.buffer)} items"
            )

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        bootstrap_steps = []
        next_action_masks = []

        for i in indices:
            s, a, r, ns, d, steps, next_action_mask = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            bootstrap_steps.append(steps)
            next_action_masks.append(next_action_mask)

        batch_dict = build_batch_dict(
            states,
            actions,
            rewards,
            next_states,
            dones,
            device,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=(
                next_action_masks if any(mask is not None for mask in next_action_masks) else None
            ),
        )
        weights = torch.ones(batch_size, dtype=torch.float32, device=device)

        return batch_dict, list(indices), weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """No-op for uniform buffer - priorities are not used."""
        pass

    def __len__(self) -> int:
        return len(self.buffer)

    def get_all_memories(self) -> List:
        """Get all memories in buffer."""
        memories = []
        for (
            state,
            action,
            reward,
            next_state,
            done,
            bootstrap_steps,
            next_action_mask,
        ) in self.buffer:
            if next_action_mask is None:
                memories.append((state, action, reward, next_state, done, 1.0, bootstrap_steps))
            else:
                memories.append(
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        1.0,
                        bootstrap_steps,
                        next_action_mask,
                    )
                )
        return memories

    def clear(self) -> None:
        """Clear all memories from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """Prioritized experience replay buffer backed by SumTree for O(log N) ops."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_increment: float = 0.000001,
        priority_eps: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_end: Maximum importance sampling weight after annealing
            beta_increment: Beta increment per sample
            priority_eps: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_increment = beta_increment
        self.priority_eps = priority_eps

        self._tree = SumTree(capacity)
        self.device = DeviceManager.get_device()

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: Optional[float] = None,
        bootstrap_steps: int = 1,
        next_action_mask=None,
        stream_id=None,
    ) -> None:
        """
        Add experience to buffer.

        Args:
            state: State tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Terminal flag
            priority: Experience priority
            bootstrap_steps: Number of environment steps before bootstrapping
            stream_id: Optional producer stream metadata for persistence/debugging.
        """
        priority = _coerce_replay_priority(
            priority,
            default_priority=self._tree.max_priority,
            priority_eps=self.priority_eps,
        )
        next_action_mask = validate_next_action_mask(next_action_mask)

        self._tree.add(
            priority,
            (
                state,
                action,
                reward,
                next_state,
                done,
                max(1, int(bootstrap_steps)),
                next_action_mask,
                stream_id,
            ),
        )

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[BatchDict, List[int], torch.Tensor]:
        """
        Sample batch from buffer with proportional prioritization via SumTree.

        Uses stratified sampling: divide total priority into batch_size segments
        and sample uniformly within each segment for better coverage.

        Args:
            batch_size: Number of experiences to sample
            device: Device to place tensors on

        Returns:
            Tuple of (batch_dict, indices, weights_tensor)

        Raises:
            ValueError: If buffer doesn't have enough samples
        """
        buf_size = len(self._tree)
        if buf_size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} items from buffer " f"with only {buf_size} items"
            )

        total = self._tree.total()
        if total == 0:
            raise ValueError("Cannot sample from buffer with zero total priority")

        segment = total / batch_size
        indices = []
        priorities = []
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        bootstrap_steps = []
        next_action_masks = []

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx, pri, data = self._tree.get(s)
            indices.append(idx)
            priorities.append(pri)
            if len(data) == 8:
                state, action, reward, next_state, done, steps, next_action_mask, _stream_id = data
            elif len(data) == 7:
                state, action, reward, next_state, done, steps, next_action_mask = data
            elif len(data) == 6:
                state, action, reward, next_state, done, steps = data
                next_action_mask = None
            else:
                state, action, reward, next_state, done = data
                steps = 1
                next_action_mask = None
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            bootstrap_steps.append(steps)
            next_action_masks.append(next_action_mask)

        # Compute importance sampling weights
        self.beta = min(self.beta_end, self.beta + self.beta_increment)
        priorities_arr = np.array(priorities, dtype=np.float64)
        # P(i) = priority_i / total
        sampling_probs = priorities_arr / total
        # Clamp to avoid division by zero
        sampling_probs = np.maximum(sampling_probs, 1e-10)
        weights = (buf_size * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()

        batch_dict = build_batch_dict(
            states,
            actions,
            rewards,
            next_states,
            dones,
            device,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=(
                next_action_masks if any(mask is not None for mask in next_action_masks) else None
            ),
        )
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        return batch_dict, indices, weights_tensor

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of samples (data indices from SumTree)
            td_errors: TD errors for priority update
        """
        for idx, error in zip(indices, td_errors):
            priority = compute_priority(error, self.alpha, self.priority_eps)
            self._tree.update(idx, priority)

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._tree)

    def get_all_memories(self) -> List:
        """Get all memories with their current priorities for saving."""
        memories = []
        for data_index in range(len(self._tree)):
            data = self._tree.data[data_index]
            if data is None:
                continue
            priority = float(self._tree.tree[data_index + self._tree.capacity - 1])
            stream_id = None
            if len(data) == 8:
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    bootstrap_steps,
                    next_action_mask,
                    stream_id,
                ) = data
            elif len(data) == 7:
                state, action, reward, next_state, done, bootstrap_steps, next_action_mask = data
            elif len(data) == 6:
                state, action, reward, next_state, done, bootstrap_steps = data
                next_action_mask = None
            else:
                state, action, reward, next_state, done = data
                bootstrap_steps = 1
                next_action_mask = None
            if stream_id is not None:
                memories.append(
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        priority,
                        bootstrap_steps,
                        next_action_mask,
                        stream_id,
                    )
                )
            elif next_action_mask is None:
                memories.append(
                    (state, action, reward, next_state, done, priority, bootstrap_steps)
                )
            else:
                memories.append(
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        priority,
                        bootstrap_steps,
                        next_action_mask,
                    )
                )
        return memories

    def add_bulk(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        priorities,
        bootstrap_steps=None,
        next_action_masks=None,
        stream_ids=None,
    ) -> None:
        """Add multiple memories at once."""
        if bootstrap_steps is None:
            bootstrap_steps = [1] * len(states)
        if next_action_masks is None:
            next_action_masks = [None] * len(states)
        if stream_ids is None:
            stream_ids = [None] * len(states)
        _validate_bulk_field_lengths(
            states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            priorities=priorities,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=next_action_masks,
            stream_ids=stream_ids,
        )

        validated_memories = []
        for (
            state,
            action,
            reward,
            next_state,
            done,
            priority,
            steps,
            next_action_mask,
            stream_id,
        ) in zip(
            states,
            actions,
            rewards,
            next_states,
            dones,
            priorities,
            bootstrap_steps,
            next_action_masks,
            stream_ids,
        ):
            priority = _coerce_replay_priority(
                priority,
                default_priority=self._tree.max_priority,
                priority_eps=self.priority_eps,
            )
            next_action_mask = validate_next_action_mask(next_action_mask)
            validated_memories.append(
                (
                    priority,
                    (
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                        max(1, int(steps)),
                        next_action_mask,
                        stream_id,
                    ),
                )
            )

        for priority, memory in validated_memories:
            self._tree.add(priority, memory)

    def clear(self) -> None:
        """Clear all memories from buffer."""
        self._tree = SumTree(self.capacity)
