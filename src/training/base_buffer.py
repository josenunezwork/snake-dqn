"""Base classes and utilities for replay buffers."""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.game_config import GameConfig

# Type alias for batch dictionary
BatchDict = Dict[
    str, torch.Tensor
]  # keys: states, actions, rewards, next_states, dones, bootstrap_steps


class BaseReplayBuffer(ABC):
    """
    Abstract base class for all replay buffers.

    Provides a unified interface for adding transitions, sampling batches,
    and updating priorities. All buffer implementations should extend this class.
    """

    @abstractmethod
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
        """
        Add a transition to the buffer.

        Args:
            state: Current state tensor
            action: Action taken
            reward: Reward received
            next_state: Next state tensor
            done: Whether episode ended
            priority: Experience priority (ignored by uniform buffers)
            bootstrap_steps: Number of environment steps before bootstrapping
            next_action_mask: Optional valid-action mask for next_state
        """
        pass

    @abstractmethod
    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[BatchDict, List[int], torch.Tensor]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of samples to retrieve
            device: Device to place tensors on

        Returns:
            batch_dict: Dictionary with keys 'states', 'actions', 'rewards',
                       'next_states', 'dones', and optionally
                       'bootstrap_steps'. All values are tensors on the
                       specified device.
            indices: List of sampled indices (for priority updates)
            weights: Importance sampling weights tensor (ones for uniform sampling)
        """
        pass

    @abstractmethod
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of samples to update
            td_errors: TD errors for priority calculation

        Note: This is a no-op for uniform (non-prioritized) buffers.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return current buffer size."""
        pass

    @abstractmethod
    def get_all_memories(self) -> List:
        """Return all memories for saving."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the buffer."""
        pass


# ============================================================================
# Utility Functions
# ============================================================================


def _validate_batch_field_lengths(states: List, **fields: List) -> int:
    """Return row count after ensuring replay batch fields are aligned."""
    row_count = len(states)
    mismatched = {name: len(values) for name, values in fields.items() if len(values) != row_count}
    if mismatched:
        details = ", ".join(f"{name}={count}" for name, count in sorted(mismatched.items()))
        raise ValueError(f"Replay batch fields are misaligned: states={row_count}, {details}")
    return row_count


def compute_priority(td_error: float, alpha: float, eps: float = 1e-5) -> float:
    """
    Compute priority from TD error.

    Args:
        td_error: Temporal difference error
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        eps: Small constant to prevent zero priorities

    Returns:
        Priority value
    """
    return (abs(td_error) + eps) ** alpha


def validate_next_action_mask(mask, expected_size: int | None = None):
    """Validate and normalize an optional exact next-action mask.

    Exact masks are per-transition simulator facts, so they must be flat
    vectors aligned to the action dimension. Nested shapes with the right
    number of values are rejected instead of being sampled into malformed
    target masks.
    """
    if mask is None:
        return None

    if expected_size is None:
        expected_size = GameConfig.OUTPUT_SIZE
    expected_shape = (int(expected_size),)

    if torch.is_tensor(mask):
        if tuple(mask.shape) != expected_shape:
            flat = mask.detach().flatten()
            if flat.numel() != expected_size:
                raise ValueError(
                    f"next_action_mask must contain {expected_size} values, got {flat.numel()}"
                )
            raise ValueError(
                f"next_action_mask must have shape {expected_shape}, got {tuple(mask.shape)}"
            )
        flat = mask.detach()
        if flat.numel() != expected_size:
            raise ValueError(
                f"next_action_mask must contain {expected_size} values, got {flat.numel()}"
            )
        if flat.dtype == torch.bool:
            return mask.to(dtype=torch.bool)
        numeric = flat.to(dtype=torch.float32)
        if not bool(torch.isfinite(numeric).all()):
            raise ValueError("next_action_mask values must be finite 0/1 or bool")
        if not bool(((numeric == 0.0) | (numeric == 1.0)).all()):
            raise ValueError("next_action_mask values must be 0/1 or bool")
        return mask.to(dtype=torch.bool)

    if hasattr(mask, "tolist"):
        mask = mask.tolist()

    values = []

    def visit(item) -> None:
        if isinstance(item, (str, bytes, bytearray, memoryview)):
            raise ValueError("next_action_mask must contain boolean/integer values")
        if isinstance(item, Iterable):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool):
            values.append(bool(item))
            return
        try:
            numeric = float(item)
            integer = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("next_action_mask must contain boolean/integer values") from exc
        if not math.isfinite(numeric) or numeric != integer or integer not in (0, 1):
            raise ValueError("next_action_mask values must be 0/1 or bool")
        values.append(bool(integer))

    visit(mask)
    if len(values) != expected_size:
        raise ValueError(f"next_action_mask must contain {expected_size} values, got {len(values)}")
    if any(
        isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray, memoryview))
        for item in mask
    ):
        array_shape = tuple(np.asarray(mask, dtype=object).shape)
        raise ValueError(f"next_action_mask must have shape {expected_shape}, got {array_shape}")
    return tuple(values)


def build_batch_dict(
    states: List,
    actions: List,
    rewards: List,
    next_states: List,
    dones: List,
    device: torch.device,
    bootstrap_steps: Optional[List] = None,
    next_action_masks: Optional[List] = None,
) -> BatchDict:
    """
    Convert lists of transitions to a batch dictionary with tensors on device.

    Args:
        states: List of state tensors or arrays
        actions: List of action values
        rewards: List of reward values
        next_states: List of next state tensors or arrays
        dones: List of done flags
        device: Device to place tensors on
        bootstrap_steps: Optional per-sample step counts for n-step targets
        next_action_masks: Optional per-sample valid-action masks for next_states

    Returns:
        Dictionary with keys 'states', 'actions', 'rewards', 'next_states', 'dones',
        and optionally 'bootstrap_steps'/'next_action_masks', all as tensors on
        the specified device.
    """
    fields = {
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
    }
    if bootstrap_steps is not None:
        fields["bootstrap_steps"] = bootstrap_steps
    if next_action_masks is not None:
        fields["next_action_masks"] = next_action_masks
    _validate_batch_field_lengths(states, **fields)

    # Convert states
    state_tensors = []
    for s in states:
        if torch.is_tensor(s):
            state_tensors.append(s.to(device))
        else:
            state_tensors.append(torch.tensor(s, dtype=torch.float32, device=device))

    # Convert next_states
    next_state_tensors = []
    for ns in next_states:
        if torch.is_tensor(ns):
            next_state_tensors.append(ns.to(device))
        else:
            next_state_tensors.append(torch.tensor(ns, dtype=torch.float32, device=device))

    batch_dict: BatchDict = {
        "states": torch.stack(state_tensors),
        "actions": torch.tensor(actions, dtype=torch.long, device=device),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
        "next_states": torch.stack(next_state_tensors),
        "dones": torch.tensor(dones, dtype=torch.float32, device=device),
    }

    if bootstrap_steps is not None:
        batch_dict["bootstrap_steps"] = torch.tensor(
            bootstrap_steps,
            dtype=torch.float32,
            device=device,
        )

    if next_action_masks is not None:
        from src.training.action_mask import valid_action_mask_from_states

        mask_tensors = []
        mask_present = []
        for mask, next_state_tensor in zip(next_action_masks, next_state_tensors):
            if torch.is_tensor(mask):
                validated_mask = validate_next_action_mask(mask)
                mask_tensors.append(validated_mask.to(device=device, dtype=torch.bool))
                mask_present.append(True)
            elif mask is None:
                mask_tensors.append(
                    valid_action_mask_from_states(next_state_tensor.unsqueeze(0)).squeeze(0)
                )
                mask_present.append(False)
            else:
                validated_mask = validate_next_action_mask(mask)
                mask_tensors.append(torch.tensor(validated_mask, dtype=torch.bool, device=device))
                mask_present.append(True)
        batch_dict["next_action_masks"] = torch.stack(mask_tensors)
        batch_dict["next_action_mask_present"] = torch.tensor(
            mask_present,
            dtype=torch.bool,
            device=device,
        )

    return batch_dict
