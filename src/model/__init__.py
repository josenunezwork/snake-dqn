"""Model management modules.

This package provides neural network architectures for reinforcement learning:
- Ape-X networks for distributed training

Also provides:
- CheckpointManager: Save/load model checkpoints
- Mixins and utilities for building custom networks
"""

from .apex_network import (
    ApexNetwork,
    create_apex_actor_network,
    create_apex_network_pair,
)

# Base utilities and mixins
from .base_network import (
    BaseDQNVisualization,
    VisualizationMixin,
    WeightManagementMixin,
    build_feature_layer,
    build_mlp,
    dueling_q,
    init_dueling_weights_orthogonal,
    init_weights_orthogonal,
    init_weights_xavier,
)
from .checkpoint_manager import MODEL_STATE_KEYS, CheckpointManager

__all__ = [
    # Core managers
    "CheckpointManager",
    "MODEL_STATE_KEYS",
    # Ape-X networks
    "ApexNetwork",
    "create_apex_network_pair",
    "create_apex_actor_network",
    # Mixins and utilities
    "WeightManagementMixin",
    "build_mlp",
    "build_feature_layer",
    "dueling_q",
    "init_weights_xavier",
    "init_weights_orthogonal",
    "init_dueling_weights_orthogonal",
    "VisualizationMixin",
    "BaseDQNVisualization",
]
