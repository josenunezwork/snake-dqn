"""Tensor utility functions for consistent tensor handling across the codebase."""

from typing import Any, List, Union

import numpy as np
import torch


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


def memories_to_dicts(raw_tuples: List) -> List[dict]:
    """Convert raw replay buffer tuples to serializable dicts.

    Converts replay tuples into dictionaries with numpy arrays suitable for
    database storage. Supports the legacy 6-field tuple
    (state, action, reward, next_state, done, priority), the current 7-field
    tuple that appends bootstrap_steps, an 8-field tuple that also carries
    next_action_mask, and a 9-field tuple with producer stream metadata.
    Used by both OnlineTrainer and ApexPolicy.

    Args:
        raw_tuples: List of replay tuples

    Returns:
        List of dicts with numpy state/next_state arrays
    """
    dict_list = []
    for raw_tuple in raw_tuples:
        next_action_mask = None
        stream_id = None
        if len(raw_tuple) == 9:
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
                "8 fields including next_action_mask, or 9 fields including stream_id"
            )

        memory_dict = {
            "state": tensor_to_numpy(state),
            "action": action,
            "reward": reward,
            "next_state": tensor_to_numpy(next_state),
            "done": done,
            "priority": priority,
            "bootstrap_steps": int(bootstrap_steps),
        }
        if next_action_mask is not None:
            memory_dict["next_action_mask"] = tensor_to_numpy(next_action_mask).astype(bool)
        if stream_id is not None:
            memory_dict["stream_id"] = stream_id
            memory_dict["snake_id"] = stream_id
        dict_list.append(memory_dict)
    return dict_list
