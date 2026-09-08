"""Tensor utility functions for consistent tensor handling across the codebase."""

from typing import Any, List, Union

import numpy as np
import torch

from src.training.td_targets import (
    MASK_MODE_TERMINAL_NO_SUCCESSOR,
    validate_replay_mask_row,
)


def tensor_to_numpy(t: Any) -> np.ndarray:
    """
    Convert tensor to numpy array, handling non-tensor inputs gracefully.

    Args:
        t: Input that may be a tensor, numpy array, or other type

    Returns:
        numpy array representation of the input
    """
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    elif isinstance(t, np.ndarray):
        return t
    else:
        return np.array(t)


def ensure_tensor_on_device(
    t: Union[torch.Tensor, np.ndarray, list], device: torch.device
) -> torch.Tensor:
    """
    Ensure input is a tensor on the specified device.

    Args:
        t: Input tensor, numpy array, or list
        device: Target device

    Returns:
        Tensor on the specified device
    """
    if torch.is_tensor(t):
        return t.to(device)
    return torch.tensor(t, dtype=torch.float32, device=device)


def validate_replay_mask_metadata(
    done: object,
    next_state: object,
    next_action_mask: object,
    next_action_mask_mode: int,
) -> tuple[int, np.ndarray | None]:
    """Validate persisted successor-mask data and return its normalized form.

    Legacy replay may omit both mask and mode. Once a producer supplies an
    explicit mode, the row must match that mode's closed semantics so a
    persistence round trip cannot relabel advisory data as a simulator fact.
    """
    if torch.is_tensor(done):
        if done.numel() != 1:
            raise ValueError("done must be bool/0/1")
        done = done.detach().cpu().item()
    if isinstance(done, (bool, np.bool_)):
        done_flag = bool(done)
    elif isinstance(done, (int, np.integer)) and not isinstance(done, bool) and int(done) in (0, 1):
        done_flag = bool(done)
    else:
        raise ValueError("done must be bool/0/1")

    mask = None
    if next_action_mask is not None:
        mask = tensor_to_numpy(next_action_mask)
        if mask.shape != (6,):
            raise ValueError(f"next_action_mask must have shape (6,), got {mask.shape}")
        if mask.dtype == np.bool_:
            mask = mask.astype(bool, copy=False)
        else:
            try:
                numeric_mask = mask.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("next_action_mask values must be 0/1 or bool") from exc
            if not np.isfinite(numeric_mask).all() or not np.isin(numeric_mask, (0.0, 1.0)).all():
                raise ValueError("next_action_mask values must be 0/1 or bool")
            mask = numeric_mask.astype(bool)

    mode = validate_replay_mask_row(
        next_state,
        done_flag,
        mask,
        next_action_mask_mode,
    )
    if mode == MASK_MODE_TERMINAL_NO_SUCCESSOR and mask is not None and bool(mask.any()):
        raise ValueError("terminal_no_successor mode cannot carry a nonempty mask")

    return mode, mask


def memories_to_dicts(raw_tuples: List) -> List[dict]:
    """Convert raw replay buffer tuples to serializable dicts.

    Converts replay tuples into dictionaries with numpy arrays suitable for
    database storage. Supports the legacy 6-field tuple
    (state, action, reward, next_state, done, priority), the current 7-field
    tuple that appends bootstrap_steps, an 8-field tuple that also carries
    next_action_mask, a 9-field tuple with producer stream metadata, and the
    current 10-field tuple with mask mode before stream metadata.
    Used by ApexPolicy.

    Args:
        raw_tuples: List of replay tuples

    Returns:
        List of dicts with numpy state/next_state arrays
    """
    dict_list = []
    for raw_tuple in raw_tuples:
        next_action_mask = None
        next_action_mask_mode = 0
        stream_id = None
        if len(raw_tuple) == 10:
            (
                state,
                action,
                reward,
                next_state,
                done,
                priority,
                bootstrap_steps,
                next_action_mask,
                next_action_mask_mode,
                stream_id,
            ) = raw_tuple
        elif len(raw_tuple) == 9:
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
            ) = raw_tuple
        elif len(raw_tuple) == 8:
            state, action, reward, next_state, done, priority, bootstrap_steps, next_action_mask = (
                raw_tuple
            )
        elif len(raw_tuple) == 7:
            state, action, reward, next_state, done, priority, bootstrap_steps = raw_tuple
        elif len(raw_tuple) == 6:
            state, action, reward, next_state, done, priority = raw_tuple
            bootstrap_steps = 1
        else:
            raise ValueError(
                "Replay tuple must have 6 fields, 7 fields including bootstrap_steps, "
                "8 fields including next_action_mask, 9 fields including stream_id, "
                "or 10 fields including next_action_mask_mode and stream_id"
            )

        next_action_mask_mode, normalized_mask = validate_replay_mask_metadata(
            done,
            next_state,
            next_action_mask,
            next_action_mask_mode,
        )

        memory_dict = {
            "state": tensor_to_numpy(state),
            "action": action,
            "reward": reward,
            "next_state": tensor_to_numpy(next_state),
            "done": done,
            "priority": priority,
            "bootstrap_steps": int(bootstrap_steps),
            "next_action_mask_mode": next_action_mask_mode,
        }
        if normalized_mask is not None:
            memory_dict["next_action_mask"] = normalized_mask
        if stream_id is not None:
            memory_dict["stream_id"] = stream_id
            memory_dict["snake_id"] = stream_id
        dict_list.append(memory_dict)
    return dict_list
