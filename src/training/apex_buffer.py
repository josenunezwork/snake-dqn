"""
Ape-X Distributed Prioritized Experience Replay Buffer.

Implements Option A: Central buffer process with queues for communication.
This design provides:
- Thread-safe concurrent actor writes and learner reads
- Prioritized sampling with importance sampling weights
- Priority updates from the learner
- Configurable capacity (1M for H100, smaller for Mac)

Architecture:
    Actors -> [Experience Queue] -> BufferProcess -> [Sample Queue] -> Learner
                                          ^
                                          |
                              [Priority Update Queue]

References:
    - Ape-X DQN: https://arxiv.org/abs/1803.00933
    - Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
"""

import platform
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from src.core.game_config import GameConfig
from src.training.action_mask import valid_action_mask_from_states
from src.training.base_buffer import (
    BaseReplayBuffer,
    BatchDict,
    compute_priority,
    validate_next_action_mask,
)
from src.training.sum_tree import SumTree

# =============================================================================
# Constants and Configuration
# =============================================================================

# Default capacities based on hardware
DEFAULT_CAPACITY_H100 = 1_000_000  # 1M for H100 GPUs
DEFAULT_CAPACITY_MAC = 100_000  # 100K for Mac (limited memory)
DEFAULT_CAPACITY_CPU = 250_000  # 250K for CPU-only systems


def _coerce_exact_action_mask(mask: Any) -> np.ndarray:
    """Validate a simulator-provided exact next-action mask.

    An all-false mask is valid: it means the simulator proved the next state has
    no legal actions, so bootstrapping should be suppressed.
    """
    if torch.is_tensor(mask):
        mask_array = mask.detach().cpu().numpy()
    else:
        mask_array = np.asarray(mask)

    expected_shape = (GameConfig.OUTPUT_SIZE,)
    if mask_array.shape != expected_shape:
        raise ValueError(
            "next_action_mask must have shape " f"{expected_shape}, got {mask_array.shape}"
        )
    validate_next_action_mask(mask_array, expected_size=GameConfig.OUTPUT_SIZE)
    mask_array = np.asarray(mask_array, dtype=np.bool_)
    return mask_array


def _validate_batch_field_lengths(states: List[Any], **fields: List[Any]) -> int:
    """Return row count after ensuring batched replay fields are aligned."""
    row_count = len(states)
    mismatched = {name: len(values) for name, values in fields.items() if len(values) != row_count}
    if mismatched:
        details = ", ".join(f"{name}={count}" for name, count in sorted(mismatched.items()))
        raise ValueError(f"Replay batch fields are misaligned: states={row_count}, {details}")
    return row_count


def _coerce_state_vector(value: Any, field_name: str, expected_size: int) -> np.ndarray:
    """Validate one flat observation vector for distributed replay."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    expected_shape = (int(expected_size),)
    if array.shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _coerce_action(action: Any) -> int:
    """Validate that a replay action matches the six-output action space."""
    if isinstance(action, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError(f"action must be an integer in [0, {GameConfig.OUTPUT_SIZE})")
    try:
        value = int(action)
        number = float(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be an integer in [0, {GameConfig.OUTPUT_SIZE})") from exc
    if not np.isfinite(number) or number != value or value < 0 or value >= GameConfig.OUTPUT_SIZE:
        raise ValueError(f"action must be an integer in [0, {GameConfig.OUTPUT_SIZE})")
    return value


def _coerce_reward(reward: Any) -> float:
    """Validate a replay reward before it reaches learner targets."""
    if isinstance(reward, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError("reward must be finite")
    try:
        value = float(reward)
    except (TypeError, ValueError) as exc:
        raise ValueError("reward must be finite") from exc
    if not np.isfinite(value):
        raise ValueError("reward must be finite")
    return value


def _coerce_done(done: Any) -> bool:
    """Validate a terminal flag without broad truthiness coercion."""
    if isinstance(done, bool):
        return done
    if isinstance(done, (str, bytes, bytearray, memoryview)):
        raise ValueError("done must be bool/0/1")
    try:
        number = float(done)
        value = int(done)
    except (TypeError, ValueError) as exc:
        raise ValueError("done must be bool/0/1") from exc
    if not np.isfinite(number) or number != value or value not in (0, 1):
        raise ValueError("done must be bool/0/1")
    return bool(value)


def _coerce_priority(priority: Any, priority_eps: float) -> float:
    """Validate a replay priority before inserting it into the SumTree."""
    if isinstance(priority, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError("priority must be finite and positive")
    try:
        value = float(priority)
    except (TypeError, ValueError) as exc:
        raise ValueError("priority must be finite and positive") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("priority must be finite and positive")
    return max(value, float(priority_eps))


def _coerce_bootstrap_steps(bootstrap_steps: Any) -> int:
    """Validate n-step replay horizon metadata without lossy truncation."""
    if isinstance(bootstrap_steps, (bool, str, bytes, bytearray, memoryview)):
        raise ValueError("bootstrap_steps must be an integer >= 1")
    try:
        steps = int(bootstrap_steps)
        number = float(bootstrap_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError("bootstrap_steps must be an integer >= 1") from exc
    if not np.isfinite(number) or number != steps or steps < 1:
        raise ValueError("bootstrap_steps must be an integer >= 1")
    return steps


def _coerce_priority_update_indices(
    indices: List[int], capacity: Optional[int] = None
) -> List[int]:
    """Validate sampled replay indices before updating SumTree priorities."""
    if isinstance(indices, (str, bytes, bytearray, memoryview)):
        raise ValueError("priority update indices must be a sequence of integers")
    validated = []
    for index in indices:
        if isinstance(index, (bool, str, bytes, bytearray, memoryview)):
            raise ValueError("priority update indices must be integers")
        try:
            value = int(index)
            number = float(index)
        except (TypeError, ValueError) as exc:
            raise ValueError("priority update indices must be integers") from exc
        if not np.isfinite(number) or number != value:
            raise ValueError("priority update indices must be integers")
        if value < 0 or (capacity is not None and value >= capacity):
            raise ValueError("priority update index out of range")
        validated.append(value)
    return validated


def _coerce_priority_update_td_errors(td_errors: Any) -> np.ndarray:
    """Validate learner TD errors before converting them to replay priorities."""
    if torch.is_tensor(td_errors):
        td_errors = td_errors.detach().cpu().numpy()
    try:
        values = np.asarray(td_errors, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("td_errors must be finite") from exc
    if not np.isfinite(values).all():
        raise ValueError("td_errors must be finite")
    return values


def _coerce_priority_update_payload(
    indices: List[int],
    td_errors: Any,
    capacity: Optional[int] = None,
) -> Tuple[List[int], np.ndarray]:
    """Validate priority-update fields as one atomic learner payload."""
    validated_indices = _coerce_priority_update_indices(indices, capacity=capacity)
    validated_errors = _coerce_priority_update_td_errors(td_errors)
    if len(validated_indices) != len(validated_errors):
        raise ValueError(
            "Priority update fields are misaligned: "
            f"indices={len(validated_indices)}, td_errors={len(validated_errors)}"
        )
    return validated_indices, validated_errors


def _mask_to_numpy(mask: Optional[Any], next_state: np.ndarray) -> np.ndarray:
    """Return a bool action mask, deriving one from next_state when absent."""
    if mask is not None:
        return _coerce_exact_action_mask(mask)

    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    return (
        valid_action_mask_from_states(next_state_tensor)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.bool_)
    )


def get_default_capacity() -> int:
    """
    Get default buffer capacity based on detected hardware.

    Returns:
        Appropriate buffer capacity for the system.
    """
    if torch.cuda.is_available():
        # Check if it's a high-end GPU (H100, A100, etc.)
        gpu_name = torch.cuda.get_device_name(0).lower()
        if any(x in gpu_name for x in ["h100", "a100", "a6000", "rtx 4090"]):
            return DEFAULT_CAPACITY_H100
        # Regular CUDA GPU
        return DEFAULT_CAPACITY_H100 // 2
    elif torch.backends.mps.is_available() or platform.system() == "Darwin":
        return DEFAULT_CAPACITY_MAC
    else:
        return DEFAULT_CAPACITY_CPU


# =============================================================================
# Message Types for Inter-Process Communication
# =============================================================================


class MessageType(Enum):
    """Message types for buffer communication."""

    ADD_EXPERIENCE = auto()  # Actor -> Buffer: Add single experience
    ADD_BATCH = auto()  # Actor -> Buffer: Add batch of experiences
    SAMPLE_REQUEST = auto()  # Learner -> Buffer: Request batch sample
    SAMPLE_RESPONSE = auto()  # Buffer -> Learner: Return sampled batch
    UPDATE_PRIORITIES = auto()  # Learner -> Buffer: Update priorities
    GET_SIZE = auto()  # Any -> Buffer: Get current buffer size
    SIZE_RESPONSE = auto()  # Buffer -> Any: Return buffer size
    GET_STATS = auto()  # Any -> Buffer: Get buffer statistics
    STATS_RESPONSE = auto()  # Buffer -> Any: Return statistics
    SHUTDOWN = auto()  # Signal buffer process to shutdown
    CLEAR = auto()  # Clear the buffer


@dataclass
class BufferMessage:
    """Message container for inter-process communication."""

    msg_type: MessageType
    data: Any = None
    sender_id: Optional[int] = None


# =============================================================================
# Shared Prioritized Replay Buffer (Core Implementation)
# =============================================================================


class SharedPrioritizedBuffer:
    """
    Core prioritized replay buffer with thread-safe operations.

    This buffer is designed to run inside a dedicated process and provides
    thread-safe access for concurrent operations.

    Uses a sum-tree data structure for O(log n) prioritized sampling.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 1_000_000,
        priority_eps: float = 1e-6,
        state_size: int = GameConfig.INPUT_SIZE,
    ):
        """
        Initialize the shared prioritized buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Number of frames to anneal beta
            priority_eps: Small constant to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.priority_eps = priority_eps
        self.state_size = int(state_size)

        # Current beta (annealed over time)
        self._beta = beta_start
        self._frame_count = 0

        # SumTree for O(log N) prioritized sampling
        self._tree = SumTree(capacity)

        # Thread lock for safe concurrent access
        self._lock = threading.RLock()

        # Statistics
        self._total_added = 0
        self._total_sampled = 0
        self._total_rejected_actor_messages = 0
        self._last_rejected_actor_message: Optional[str] = None
        self._total_rejected_priority_updates = 0
        self._last_rejected_priority_update: Optional[str] = None

    @property
    def beta(self) -> float:
        """Current beta value for importance sampling."""
        return self._beta

    def _update_beta(self) -> None:
        """Anneal beta towards 1.0 over training."""
        self._frame_count += 1
        fraction = min(1.0, self._frame_count / self.beta_frames)
        self._beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
        bootstrap_steps: int = 1,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a single experience to the buffer.

        Args:
            state: Current state (numpy array)
            action: Action taken
            reward: Reward received
            next_state: Next state (numpy array)
            done: Whether episode ended
            priority: Experience priority (uses max priority if None)
            bootstrap_steps: Number of environment steps before bootstrapping
            next_action_mask: Optional valid-action mask for next_state
        """
        with self._lock:
            state = _coerce_state_vector(state, "state", self.state_size)
            action = _coerce_action(action)
            reward = _coerce_reward(reward)
            next_state = _coerce_state_vector(next_state, "next_state", self.state_size)
            done = _coerce_done(done)

            # Use max priority for new experiences to ensure they're sampled
            if priority is None:
                priority = self._tree.max_priority

            # Supplied priorities are already PER tree priorities. Actors and
            # learner updates both call compute_priority(td_error, alpha, eps)
            # before this boundary, so exponentiating here would apply alpha
            # twice and skew fresh replay against learner-updated replay.
            priority = _coerce_priority(priority, self.priority_eps)
            bootstrap_steps = _coerce_bootstrap_steps(bootstrap_steps)
            if next_action_mask is not None:
                next_action_mask = _coerce_exact_action_mask(next_action_mask)

            # Add experience to SumTree
            experience = (
                state,
                action,
                reward,
                next_state,
                done,
                bootstrap_steps,
                next_action_mask,
            )
            self._tree.add(priority, experience)

            self._total_added += 1

    def add_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        priorities: Optional[List[float]] = None,
        bootstrap_steps: Optional[List[int]] = None,
        next_action_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Add a batch of experiences to the buffer.

        More efficient than calling add() repeatedly.

        Args:
            states: List of current states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            priorities: List of priorities (uses max priority if None)
            bootstrap_steps: Per-experience bootstrap horizons
            next_action_masks: Optional per-experience valid-action masks
        """
        with self._lock:
            n = len(states)
            if priorities is None:
                priorities = [self._tree.max_priority] * n
            if bootstrap_steps is None:
                bootstrap_steps = [1] * n
            if next_action_masks is None:
                next_action_masks = [None] * n
            _validate_batch_field_lengths(
                states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                priorities=priorities,
                bootstrap_steps=bootstrap_steps,
                next_action_masks=next_action_masks,
            )

            validated_experiences = []
            for i in range(n):
                state = _coerce_state_vector(states[i], "state", self.state_size)
                action = _coerce_action(actions[i])
                reward = _coerce_reward(rewards[i])
                next_state = _coerce_state_vector(next_states[i], "next_state", self.state_size)
                done = _coerce_done(dones[i])
                pri = priorities[i]
                if pri is None:
                    pri = self._tree.max_priority
                pri = _coerce_priority(pri, self.priority_eps)
                steps = _coerce_bootstrap_steps(bootstrap_steps[i])
                next_action_mask = next_action_masks[i]
                if next_action_mask is not None:
                    next_action_mask = _coerce_exact_action_mask(next_action_mask)
                experience = (
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    steps,
                    next_action_mask,
                )
                validated_experiences.append((pri, experience))

            for priority, experience in validated_experiences:
                self._tree.add(priority, experience)

            self._total_added += n

    def record_rejected_actor_message(self, exc: Exception) -> None:
        """Record an actor replay payload that could not be inserted."""
        with self._lock:
            self._total_rejected_actor_messages += 1
            self._last_rejected_actor_message = str(exc)

    def record_rejected_priority_update(self, exc: Exception) -> None:
        """Record a learner priority update that could not be applied."""
        with self._lock:
            self._total_rejected_priority_updates += 1
            self._last_rejected_priority_update = str(exc)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], List[int], np.ndarray]:
        """
        Sample a batch of experiences with stratified prioritized sampling.

        Uses SumTree for O(log N) per-sample lookup. Divides total priority
        into batch_size segments and samples one experience per segment.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (batch_dict, indices, weights):
            - batch_dict: Dictionary with 'states', 'actions', 'rewards',
                         'next_states', 'dones' as numpy arrays
            - indices: List of sampled data indices for priority updates
            - weights: Importance sampling weights (numpy array)

        Raises:
            ValueError: If buffer doesn't have enough samples
        """
        with self._lock:
            buffer_size = len(self._tree)

            if buffer_size < batch_size:
                raise ValueError(
                    f"Cannot sample {batch_size} items from buffer "
                    f"with only {buffer_size} items"
                )

            # Update beta for importance sampling
            self._update_beta()

            total = self._tree.total()
            if total == 0:
                raise ValueError("Cannot sample from buffer with zero total priority")

            segment = total / batch_size
            indices = []
            priorities_list = []
            states = []
            actions = []
            rewards = []
            next_states = []
            dones_list = []
            bootstrap_steps = []
            next_action_masks = []

            for i in range(batch_size):
                low = segment * i
                high = segment * (i + 1)
                s = np.random.uniform(low, high)
                idx, pri, data = self._tree.get(s)
                indices.append(idx)
                priorities_list.append(pri)
                if len(data) == 7:
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
                dones_list.append(done)
                bootstrap_steps.append(steps)
                next_action_masks.append(next_action_mask)

            # Compute importance sampling weights
            priorities_arr = np.array(priorities_list, dtype=np.float64)
            sampling_probs = priorities_arr / total
            sampling_probs = np.maximum(sampling_probs, 1e-10)
            weights = (buffer_size * sampling_probs) ** (-self._beta)
            weights = weights / weights.max()

            # Create batch dictionary with numpy arrays
            batch_dict = {
                "states": np.array(states),
                "actions": np.array(actions, dtype=np.int64),
                "rewards": np.array(rewards, dtype=np.float32),
                "next_states": np.array(next_states),
                "dones": np.array(dones_list, dtype=np.float32),
                "bootstrap_steps": np.array(bootstrap_steps, dtype=np.float32),
            }
            if any(mask is not None for mask in next_action_masks):
                batch_dict["next_action_masks"] = np.array(
                    [
                        _mask_to_numpy(mask, next_state)
                        for mask, next_state in zip(next_action_masks, next_states)
                    ],
                    dtype=np.bool_,
                )
                batch_dict["next_action_mask_present"] = np.array(
                    [mask is not None for mask in next_action_masks],
                    dtype=np.bool_,
                )

            self._total_sampled += batch_size

            return batch_dict, indices, weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors from the learner.

        Args:
            indices: Data indices of experiences to update (from SumTree)
            td_errors: TD errors for computing new priorities
        """
        with self._lock:
            indices, td_errors = _coerce_priority_update_payload(
                indices,
                td_errors,
                capacity=self._tree.capacity,
            )
            for idx, td_error in zip(indices, td_errors):
                # Compute new priority
                new_priority = compute_priority(td_error, self.alpha, self.priority_eps)

                # Update priority in SumTree
                self._tree.update(idx, new_priority)

    def __len__(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self._tree)

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        with self._lock:
            self._tree = SumTree(self.capacity)
            self._total_added = 0
            self._total_sampled = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Returns:
            Dictionary with buffer statistics.
        """
        with self._lock:
            tree_size = len(self._tree)
            return {
                "size": tree_size,
                "capacity": self.capacity,
                "fill_ratio": tree_size / self.capacity,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "total_rejected_actor_messages": self._total_rejected_actor_messages,
                "last_rejected_actor_message": self._last_rejected_actor_message,
                "total_rejected_priority_updates": self._total_rejected_priority_updates,
                "last_rejected_priority_update": self._last_rejected_priority_update,
                "max_priority": self._tree.max_priority,
                "current_beta": self._beta,
                "alpha": self.alpha,
            }


# =============================================================================
# Buffer Process (Runs in Separate Process)
# =============================================================================


class BufferProcess:
    """
    Manages the replay buffer in a dedicated process.

    Handles communication between actors (adding experiences) and
    learner (sampling and updating priorities) via multiprocessing queues.

    Usage:
        # Start buffer process
        buffer_proc = BufferProcess(capacity=1_000_000)
        buffer_proc.start()

        # Get client interfaces
        actor_client = buffer_proc.get_actor_client()
        learner_client = buffer_proc.get_learner_client()

        # Use clients from different processes/threads
        actor_client.add(state, action, reward, next_state, done)
        batch, indices, weights = learner_client.sample(batch_size=256)
        learner_client.update_priorities(indices, td_errors)

        # Shutdown
        buffer_proc.shutdown()
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 1_000_000,
        max_queue_size: int = 1000,
        state_size: int = GameConfig.INPUT_SIZE,
    ):
        """
        Initialize buffer process manager.

        Args:
            capacity: Buffer capacity (auto-detected if None)
            alpha: Priority exponent
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Frames to anneal beta
            max_queue_size: Maximum size for communication queues
            state_size: Expected flat state vector size.
        """
        self.capacity = capacity or get_default_capacity()
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.max_queue_size = max_queue_size
        self.state_size = int(state_size)

        # Create multiprocessing queues
        self._experience_queue: mp.Queue = mp.Queue(maxsize=max_queue_size)
        self._sample_request_queue: mp.Queue = mp.Queue(maxsize=100)
        self._sample_response_queue: mp.Queue = mp.Queue(maxsize=100)
        self._priority_update_queue: mp.Queue = mp.Queue(maxsize=max_queue_size)
        self._control_queue: mp.Queue = mp.Queue(maxsize=100)
        self._response_queue: mp.Queue = mp.Queue(maxsize=100)

        # Process handle
        self._process: Optional[mp.Process] = None
        self._shutdown_event = mp.Event()

    def start(self) -> None:
        """Start the buffer process."""
        if self._process is not None and self._process.is_alive():
            return

        self._shutdown_event.clear()

        self._process = mp.Process(
            target=self._run_buffer_loop,
            args=(
                self.capacity,
                self.alpha,
                self.beta_start,
                self.beta_end,
                self.beta_frames,
                self.state_size,
                self._experience_queue,
                self._sample_request_queue,
                self._sample_response_queue,
                self._priority_update_queue,
                self._control_queue,
                self._response_queue,
                self._shutdown_event,
            ),
            daemon=True,
        )
        self._process.start()

    @staticmethod
    def _run_buffer_loop(
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_end: float,
        beta_frames: int,
        state_size: int,
        experience_queue: mp.Queue,
        sample_request_queue: mp.Queue,
        sample_response_queue: mp.Queue,
        priority_update_queue: mp.Queue,
        control_queue: mp.Queue,
        response_queue: mp.Queue,
        shutdown_event: mp.Event,
    ) -> None:
        """
        Main loop for the buffer process.

        Runs in a separate process, handling all buffer operations.
        """
        # Create the buffer
        buffer = SharedPrioritizedBuffer(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
            state_size=state_size,
        )

        poll_interval = 0.001  # 1ms polling interval

        while not shutdown_event.is_set():
            processed_any = False

            # Process experience additions (from actors)
            try:
                while True:
                    msg = experience_queue.get_nowait()
                    processed_any = True

                    try:
                        if msg.msg_type == MessageType.ADD_EXPERIENCE:
                            if len(msg.data) == 8:
                                (
                                    state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    priority,
                                    steps,
                                    next_action_mask,
                                ) = msg.data
                            elif len(msg.data) == 7:
                                state, action, reward, next_state, done, priority, steps = msg.data
                                next_action_mask = None
                            else:
                                state, action, reward, next_state, done, priority = msg.data
                                steps = 1
                                next_action_mask = None
                            buffer.add(
                                state,
                                action,
                                reward,
                                next_state,
                                done,
                                priority,
                                bootstrap_steps=steps,
                                next_action_mask=next_action_mask,
                            )

                        elif msg.msg_type == MessageType.ADD_BATCH:
                            if len(msg.data) == 8:
                                (
                                    states,
                                    actions,
                                    rewards,
                                    next_states,
                                    dones,
                                    priorities,
                                    bootstrap_steps,
                                    next_action_masks,
                                ) = msg.data
                            elif len(msg.data) == 7:
                                (
                                    states,
                                    actions,
                                    rewards,
                                    next_states,
                                    dones,
                                    priorities,
                                    bootstrap_steps,
                                ) = msg.data
                                next_action_masks = None
                            else:
                                states, actions, rewards, next_states, dones, priorities = msg.data
                                bootstrap_steps = None
                                next_action_masks = None
                            buffer.add_batch(
                                states,
                                actions,
                                rewards,
                                next_states,
                                dones,
                                priorities,
                                bootstrap_steps=bootstrap_steps,
                                next_action_masks=next_action_masks,
                            )
                    except Exception as exc:
                        buffer.record_rejected_actor_message(exc)
            except Empty:
                pass

            # Process sample requests (from learner)
            try:
                while True:
                    msg = sample_request_queue.get_nowait()
                    processed_any = True

                    if msg.msg_type == MessageType.SAMPLE_REQUEST:
                        batch_size = msg.data
                        try:
                            batch, indices, weights = buffer.sample(batch_size)
                            response = BufferMessage(
                                MessageType.SAMPLE_RESPONSE, data=(batch, indices, weights)
                            )
                        except ValueError:
                            # Not enough samples
                            response = BufferMessage(MessageType.SAMPLE_RESPONSE, data=None)
                        sample_response_queue.put(response)
            except Empty:
                pass

            # Process priority updates (from learner)
            try:
                while True:
                    msg = priority_update_queue.get_nowait()
                    processed_any = True

                    if msg.msg_type == MessageType.UPDATE_PRIORITIES:
                        indices, td_errors = msg.data
                        try:
                            buffer.update_priorities(indices, td_errors)
                        except Exception as exc:
                            buffer.record_rejected_priority_update(exc)
            except Empty:
                pass

            # Process control messages
            try:
                while True:
                    msg = control_queue.get_nowait()
                    processed_any = True

                    if msg.msg_type == MessageType.GET_SIZE:
                        response = BufferMessage(MessageType.SIZE_RESPONSE, data=len(buffer))
                        response_queue.put(response)

                    elif msg.msg_type == MessageType.GET_STATS:
                        response = BufferMessage(
                            MessageType.STATS_RESPONSE, data=buffer.get_stats()
                        )
                        response_queue.put(response)

                    elif msg.msg_type == MessageType.CLEAR:
                        buffer.clear()

                    elif msg.msg_type == MessageType.SHUTDOWN:
                        shutdown_event.set()
            except Empty:
                pass

            # Small sleep to prevent CPU spinning when idle
            if not processed_any:
                time.sleep(poll_interval)

    def shutdown(self, timeout: float = 5.0) -> None:
        """
        Shutdown the buffer process gracefully.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if self._process is None or not self._process.is_alive():
            return

        # Signal shutdown
        self._shutdown_event.set()
        self._control_queue.put(BufferMessage(MessageType.SHUTDOWN))

        # Wait for process to terminate
        self._process.join(timeout=timeout)

        # Force terminate if still running
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

    def get_actor_client(self, actor_id: int = 0) -> "ActorBufferClient":
        """
        Get a client interface for actors to add experiences.

        Args:
            actor_id: Unique identifier for the actor

        Returns:
            ActorBufferClient instance
        """
        return ActorBufferClient(
            experience_queue=self._experience_queue,
            actor_id=actor_id,
            state_size=self.state_size,
        )

    def get_learner_client(self) -> "LearnerBufferClient":
        """
        Get a client interface for the learner.

        Returns:
            LearnerBufferClient instance
        """
        return LearnerBufferClient(
            sample_request_queue=self._sample_request_queue,
            sample_response_queue=self._sample_response_queue,
            priority_update_queue=self._priority_update_queue,
            control_queue=self._control_queue,
            response_queue=self._response_queue,
        )

    def get_size(self, timeout: float = 1.0) -> int:
        """
        Get current buffer size.

        Args:
            timeout: Maximum time to wait for response

        Returns:
            Current number of experiences in buffer
        """
        self._control_queue.put(BufferMessage(MessageType.GET_SIZE))
        try:
            response = self._response_queue.get(timeout=timeout)
            return response.data
        except Exception:
            return 0

    def get_stats(self, timeout: float = 1.0) -> Dict[str, Any]:
        """
        Get buffer statistics.

        Args:
            timeout: Maximum time to wait for response

        Returns:
            Dictionary with buffer statistics
        """
        self._control_queue.put(BufferMessage(MessageType.GET_STATS))
        try:
            response = self._response_queue.get(timeout=timeout)
            return response.data
        except Exception:
            return {}

    @property
    def is_alive(self) -> bool:
        """Check if buffer process is running."""
        return self._process is not None and self._process.is_alive()


# =============================================================================
# Client Interfaces
# =============================================================================


class ActorBufferClient:
    """
    Client interface for actors to add experiences to the shared buffer.

    Thread-safe and process-safe. Can be used from any process.
    """

    def __init__(
        self,
        experience_queue: mp.Queue,
        actor_id: int = 0,
        state_size: int = GameConfig.INPUT_SIZE,
    ):
        """
        Initialize actor client.

        Args:
            experience_queue: Queue for sending experiences
            actor_id: Unique identifier for this actor
            state_size: Expected flat state vector size.
        """
        self._queue = experience_queue
        self._actor_id = actor_id
        self._state_size = int(state_size)
        self._local_buffer: List = []  # For batching
        self._batch_threshold = 32  # Send in batches for efficiency
        self._queued_message_count = 0
        self._dropped_message_count = 0
        self._dropped_experience_count = 0
        self._last_drop_error: Optional[str] = None

    def get_stats(self) -> Dict[str, Any]:
        """Return actor-side IPC enqueue/drop counters."""
        return {
            "queued_message_count": self._queued_message_count,
            "dropped_message_count": self._dropped_message_count,
            "dropped_experience_count": self._dropped_experience_count,
            "last_drop_error": self._last_drop_error,
        }

    def _enqueue_message(self, msg: BufferMessage, experience_count: int) -> None:
        """Best-effort enqueue while tracking messages dropped before the buffer."""
        try:
            self._queue.put_nowait(msg)
            self._queued_message_count += 1
        except Exception as exc:
            # Queue full, drop experiences (acceptable in Ape-X), but make it observable.
            self._dropped_message_count += 1
            self._dropped_experience_count += int(experience_count)
            self._last_drop_error = str(exc)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
        flush: bool = False,
        bootstrap_steps: int = 1,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add experience to the shared buffer.

        Experiences are buffered locally and sent in batches for efficiency.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            priority: Experience priority
            flush: Force send local buffer immediately
            bootstrap_steps: Number of environment steps before bootstrapping
            next_action_mask: Optional valid-action mask for next_state
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        if torch.is_tensor(next_state):
            next_state = next_state.detach().cpu().numpy()
        if torch.is_tensor(next_action_mask):
            next_action_mask = next_action_mask.detach().cpu().numpy()
        state = _coerce_state_vector(state, "state", self._state_size)
        action = _coerce_action(action)
        reward = _coerce_reward(reward)
        next_state = _coerce_state_vector(next_state, "next_state", self._state_size)
        done = _coerce_done(done)
        if priority is not None:
            priority = _coerce_priority(priority, 0.0)
        bootstrap_steps = _coerce_bootstrap_steps(bootstrap_steps)
        if next_action_mask is not None:
            next_action_mask = _coerce_exact_action_mask(next_action_mask)

        self._local_buffer.append(
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

        # Flush on episode end or threshold reached
        if done or flush or len(self._local_buffer) >= self._batch_threshold:
            self.flush()

    def flush(self) -> None:
        """Send all buffered experiences to the shared buffer."""
        if not self._local_buffer:
            return

        if len(self._local_buffer) == 1:
            # Single experience
            msg = BufferMessage(
                MessageType.ADD_EXPERIENCE, data=self._local_buffer[0], sender_id=self._actor_id
            )
        else:
            # Batch of experiences
            states = [x[0] for x in self._local_buffer]
            actions = [x[1] for x in self._local_buffer]
            rewards = [x[2] for x in self._local_buffer]
            next_states = [x[3] for x in self._local_buffer]
            dones = [x[4] for x in self._local_buffer]
            priorities = [x[5] for x in self._local_buffer]
            bootstrap_steps = [x[6] for x in self._local_buffer]
            next_action_masks = [x[7] for x in self._local_buffer]

            msg = BufferMessage(
                MessageType.ADD_BATCH,
                data=(
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    priorities,
                    bootstrap_steps,
                    next_action_masks,
                ),
                sender_id=self._actor_id,
            )

        self._enqueue_message(msg, experience_count=len(self._local_buffer))

        self._local_buffer.clear()

    def add_batch(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        next_states: List[np.ndarray],
        dones: List[bool],
        priorities: Optional[List[float]] = None,
        bootstrap_steps: Optional[List[int]] = None,
        next_action_masks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """
        Add a batch of experiences directly.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            priorities: List of priorities
            bootstrap_steps: Per-experience bootstrap horizons
            next_action_masks: Optional per-experience valid-action masks
        """
        n = len(states)
        if priorities is None:
            priorities = [None] * n
        if bootstrap_steps is None:
            bootstrap_steps = [1] * n
        if next_action_masks is None:
            next_action_masks = [None] * n
        _validate_batch_field_lengths(
            states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            priorities=priorities,
            bootstrap_steps=bootstrap_steps,
            next_action_masks=next_action_masks,
        )
        if next_action_masks is not None:
            next_action_masks = [
                None if mask is None else _coerce_exact_action_mask(mask)
                for mask in next_action_masks
            ]
        states = [_coerce_state_vector(state, "state", self._state_size) for state in states]
        actions = [_coerce_action(action) for action in actions]
        rewards = [_coerce_reward(reward) for reward in rewards]
        next_states = [
            _coerce_state_vector(next_state, "next_state", self._state_size)
            for next_state in next_states
        ]
        dones = [_coerce_done(done) for done in dones]
        priorities = [
            None if priority is None else _coerce_priority(priority, 0.0) for priority in priorities
        ]
        bootstrap_steps = [_coerce_bootstrap_steps(steps) for steps in bootstrap_steps]
        msg = BufferMessage(
            MessageType.ADD_BATCH,
            data=(
                states,
                actions,
                rewards,
                next_states,
                dones,
                priorities,
                bootstrap_steps,
                next_action_masks,
            ),
            sender_id=self._actor_id,
        )

        self._enqueue_message(msg, experience_count=n)


class LearnerBufferClient:
    """
    Client interface for the learner to sample from and update the shared buffer.

    Provides prioritized sampling and priority updates.
    """

    def __init__(
        self,
        sample_request_queue: mp.Queue,
        sample_response_queue: mp.Queue,
        priority_update_queue: mp.Queue,
        control_queue: mp.Queue,
        response_queue: mp.Queue,
    ):
        """
        Initialize learner client.

        Args:
            sample_request_queue: Queue for sample requests
            sample_response_queue: Queue for sample responses
            priority_update_queue: Queue for priority updates
            control_queue: Queue for control messages
            response_queue: Queue for control responses
        """
        self._sample_request_queue = sample_request_queue
        self._sample_response_queue = sample_response_queue
        self._priority_update_queue = priority_update_queue
        self._control_queue = control_queue
        self._response_queue = response_queue

    def sample(
        self, batch_size: int, device: Optional[torch.device] = None, timeout: float = 5.0
    ) -> Optional[Tuple[BatchDict, List[int], torch.Tensor]]:
        """
        Sample a batch of experiences with prioritization.

        Args:
            batch_size: Number of experiences to sample
            device: Device to place tensors on (if None, returns numpy arrays)
            timeout: Maximum time to wait for response

        Returns:
            Tuple of (batch_dict, indices, weights) or None if not enough samples.
            - batch_dict: Dictionary with 'states', 'actions', 'rewards',
                         'next_states', 'dones' as tensors
            - indices: List of sampled indices
            - weights: Importance sampling weights
        """
        # Send sample request
        msg = BufferMessage(MessageType.SAMPLE_REQUEST, data=batch_size)
        self._sample_request_queue.put(msg)

        # Wait for response
        try:
            response = self._sample_response_queue.get(timeout=timeout)

            if response.data is None:
                return None

            batch, indices, weights = response.data

            # Convert to tensors if device specified
            if device is not None:
                raw_batch = batch
                batch = {
                    "states": torch.tensor(raw_batch["states"], dtype=torch.float32, device=device),
                    "actions": torch.tensor(raw_batch["actions"], dtype=torch.long, device=device),
                    "rewards": torch.tensor(
                        raw_batch["rewards"], dtype=torch.float32, device=device
                    ),
                    "next_states": torch.tensor(
                        raw_batch["next_states"], dtype=torch.float32, device=device
                    ),
                    "dones": torch.tensor(raw_batch["dones"], dtype=torch.float32, device=device),
                    "bootstrap_steps": torch.tensor(
                        raw_batch.get("bootstrap_steps", np.ones_like(raw_batch["dones"])),
                        dtype=torch.float32,
                        device=device,
                    ),
                }
                if "next_action_masks" in raw_batch:
                    batch["next_action_masks"] = torch.tensor(
                        raw_batch["next_action_masks"],
                        dtype=torch.bool,
                        device=device,
                    )
                if "next_action_mask_present" in raw_batch:
                    batch["next_action_mask_present"] = torch.tensor(
                        raw_batch["next_action_mask_present"],
                        dtype=torch.bool,
                        device=device,
                    )
                weights = torch.tensor(weights, dtype=torch.float32, device=device)

            return batch, indices, weights

        except Exception:
            return None

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for computing new priorities
        """
        # Convert tensor to numpy if needed
        indices, td_errors = _coerce_priority_update_payload(indices, td_errors)

        msg = BufferMessage(MessageType.UPDATE_PRIORITIES, data=(indices, td_errors))

        try:
            self._priority_update_queue.put_nowait(msg)
        except Exception:
            pass  # Queue full, skip update

    def get_size(self, timeout: float = 1.0) -> int:
        """Get current buffer size."""
        self._control_queue.put(BufferMessage(MessageType.GET_SIZE))
        try:
            response = self._response_queue.get(timeout=timeout)
            return response.data
        except Exception:
            return 0

    def get_stats(self, timeout: float = 1.0) -> Dict[str, Any]:
        """Get buffer statistics."""
        self._control_queue.put(BufferMessage(MessageType.GET_STATS))
        try:
            response = self._response_queue.get(timeout=timeout)
            return response.data
        except Exception:
            return {}


# =============================================================================
# Local Prioritized Buffer (Single-Process Alternative)
# =============================================================================


class LocalApexBuffer(BaseReplayBuffer):
    """
    Single-process Ape-X style buffer for testing or single-process training.

    Implements the same interface as the distributed buffer but runs
    in the same process. Useful for debugging or when multi-process
    overhead isn't justified.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 1_000_000,
        state_size: int = GameConfig.INPUT_SIZE,
    ):
        """
        Initialize local Ape-X buffer.

        Args:
            capacity: Buffer capacity (auto-detected if None)
            alpha: Priority exponent
            beta_start: Initial importance sampling weight
            beta_end: Final importance sampling weight
            beta_frames: Frames to anneal beta
            state_size: Expected flat state vector size.
        """
        self._buffer = SharedPrioritizedBuffer(
            capacity=capacity or get_default_capacity(),
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
            state_size=state_size,
        )

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: Optional[float] = None,
        bootstrap_steps: int = 1,
        next_action_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Add experience to buffer, using max priority when none is supplied."""
        # Convert tensors to numpy
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        if torch.is_tensor(next_state):
            next_state = next_state.detach().cpu().numpy()
        if torch.is_tensor(next_action_mask):
            next_action_mask = next_action_mask.detach().cpu().numpy()
        if next_action_mask is not None:
            next_action_mask = _coerce_exact_action_mask(next_action_mask)

        self._buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            priority,
            bootstrap_steps=bootstrap_steps,
            next_action_mask=next_action_mask,
        )

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[BatchDict, List[int], torch.Tensor]:
        """Sample batch with prioritization."""
        batch, indices, weights = self._buffer.sample(batch_size)

        # Convert to tensors
        batch_dict = {
            "states": torch.tensor(batch["states"], dtype=torch.float32, device=device),
            "actions": torch.tensor(batch["actions"], dtype=torch.long, device=device),
            "rewards": torch.tensor(batch["rewards"], dtype=torch.float32, device=device),
            "next_states": torch.tensor(batch["next_states"], dtype=torch.float32, device=device),
            "dones": torch.tensor(batch["dones"], dtype=torch.float32, device=device),
            "bootstrap_steps": torch.tensor(
                batch.get("bootstrap_steps", np.ones_like(batch["dones"])),
                dtype=torch.float32,
                device=device,
            ),
        }
        if "next_action_masks" in batch:
            batch_dict["next_action_masks"] = torch.tensor(
                batch["next_action_masks"],
                dtype=torch.bool,
                device=device,
            )
        if "next_action_mask_present" in batch:
            batch_dict["next_action_mask_present"] = torch.tensor(
                batch["next_action_mask_present"],
                dtype=torch.bool,
                device=device,
            )
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        return batch_dict, indices, weights_tensor

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        if torch.is_tensor(td_errors):
            td_errors = td_errors.detach().cpu().numpy()
        self._buffer.update_priorities(indices, td_errors)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def get_all_memories(self) -> List:
        """Return all memories (not recommended for large buffers)."""
        tree = self._buffer._tree
        return [tree.data[i] for i in range(len(tree)) if tree.data[i] is not None]

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return self._buffer.get_stats()


# =============================================================================
# Factory Function
# =============================================================================


def create_apex_buffer(
    distributed: bool = True,
    capacity: Optional[int] = None,
    alpha: float = 0.6,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    beta_frames: int = 1_000_000,
    state_size: int = GameConfig.INPUT_SIZE,
    **kwargs,
) -> Tuple[Any, Optional[ActorBufferClient], Optional[LearnerBufferClient]]:
    """
    Factory function to create an Ape-X buffer.

    Args:
        distributed: If True, creates distributed buffer with process
                    If False, creates local single-process buffer
        capacity: Buffer capacity (auto-detected if None)
        alpha: Priority exponent
        beta_start: Initial importance sampling weight
        beta_end: Final importance sampling weight
        beta_frames: Frames to anneal beta
        state_size: Expected flat state vector size.
        **kwargs: Additional arguments for BufferProcess

    Returns:
        Tuple of (buffer, actor_client, learner_client)
        - For distributed: (BufferProcess, ActorBufferClient, LearnerBufferClient)
        - For local: (LocalApexBuffer, None, None)
    """
    if distributed:
        buffer = BufferProcess(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
            state_size=state_size,
            **kwargs,
        )
        buffer.start()

        actor_client = buffer.get_actor_client()
        learner_client = buffer.get_learner_client()

        return buffer, actor_client, learner_client
    else:
        buffer = LocalApexBuffer(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_frames=beta_frames,
            state_size=state_size,
        )
        return buffer, None, None
