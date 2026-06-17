"""Model management modules.

This package provides neural network architectures for reinforcement learning:
- Ape-X networks for distributed training

Also provides:
- ModelFactory: Factory methods for creating model types
- CheckpointManager: Save/load model checkpoints
- Mixins and utilities for building custom networks
"""
from .checkpoint_manager import CheckpointManager, MODEL_STATE_KEYS
from .model_factory import ModelFactory
from .apex_network import ApexNetwork, create_apex_network_pair, create_apex_actor_network
from .gru_network import GruApexNetwork, create_gru_network_pair, create_gru_actor_network

# Base utilities and mixins
from .base_network import (
    build_mlp,
    build_feature_layer,
    dueling_q,
    init_weights_xavier,
    init_weights_orthogonal,
    init_dueling_weights_orthogonal,
    WeightManagementMixin,
    VisualizationMixin,
    BaseDQNVisualization,
)

__all__ = [
    # Core managers
    'CheckpointManager',
    'ModelFactory',
    'MODEL_STATE_KEYS',
    # Ape-X networks
    'ApexNetwork',
    'create_apex_network_pair',
    'create_apex_actor_network',
    # GRU/DRQN networks
    'GruApexNetwork',
    'create_gru_network_pair',
    'create_gru_actor_network',
    # Mixins and utilities
    'WeightManagementMixin',
    'build_mlp',
    'build_feature_layer',
    'dueling_q',
    'init_weights_xavier',
    'init_weights_orthogonal',
    'init_dueling_weights_orthogonal',
    'VisualizationMixin',
    'BaseDQNVisualization',
]
