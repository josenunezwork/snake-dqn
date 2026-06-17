"""Neural network utility functions for weight initialization and common operations."""
import torch
import torch.nn as nn


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """
    Hard update (copy) target network parameters.

    Args:
        target: Target network to update
        source: Source network to copy from
    """
    target.load_state_dict(source.state_dict())


def clip_gradients(model: nn.Module, max_norm: float = 10.0) -> None:
    """
    Clip gradients by global norm.

    Args:
        model: Model whose gradients to clip
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
