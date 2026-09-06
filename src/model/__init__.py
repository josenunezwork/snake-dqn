"""Model management modules.

This package provides neural network architectures for reinforcement learning:
- Ape-X networks for distributed training

Also provides checkpoint and network utility APIs.
"""

from .apex_network import ApexNetwork

# Base utilities and mixins
from .base_network import (
    BaseDQNVisualization,
    dueling_q,
    init_dueling_weights_orthogonal,
    init_weights_orthogonal,
    init_weights_xavier,
)
from .checkpoint_manager import CheckpointManager

__all__ = [
    # Core managers
    "CheckpointManager",
    # Ape-X networks
    "ApexNetwork",
    # Utilities
    "dueling_q",
    "init_weights_xavier",
    "init_weights_orthogonal",
    "init_dueling_weights_orthogonal",
    "BaseDQNVisualization",
]
