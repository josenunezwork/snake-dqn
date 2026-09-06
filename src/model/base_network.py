"""Base network utilities and mixins for building neural networks.

Provides reusable building blocks for constructing neural network
architectures with less code duplication.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


def dueling_q(
    value_stream: torch.Tensor,
    advantage_stream: torch.Tensor,
) -> torch.Tensor:
    """Combine value and advantage streams into Q-values using dueling formula.

    Implements: Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a')).

    Args:
        value_stream: Value tensor with trailing action-dim size 1,
            e.g. shape (..., 1).
        advantage_stream: Advantage tensor with trailing action-dim size
            equal to the number of actions, e.g. shape (..., output_size).

    Returns:
        Q-values tensor with shape matching ``advantage_stream``.
    """
    return value_stream + (advantage_stream - advantage_stream.mean(dim=-1, keepdim=True))


def init_weights_xavier(module: nn.Module) -> None:
    """
    Initialize network weights using Xavier uniform initialization.

    Args:
        module: Network module to initialize
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


def init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize network weights using orthogonal initialization.

    Args:
        module: Network module to initialize
        gain: Scaling factor for weights
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)


def init_dueling_weights_orthogonal(
    feature_layer: nn.Sequential,
    value_stream: nn.Sequential,
    advantage_stream: nn.Sequential,
) -> None:
    """Initialize dueling network weights using orthogonal initialization.

    Applies sqrt(2) gain for ReLU layers and gain=1.0 for output layers.
    Used by ApexNetwork.

    Args:
        feature_layer: Shared feature extraction sequential module
        value_stream: Value stream sequential module
        advantage_stream: Advantage stream sequential module
    """
    for module in [feature_layer, value_stream, advantage_stream]:
        for i, layer in enumerate(module):
            if isinstance(layer, nn.Linear):
                is_output = i == len(module) - 1
                gain = 1.0 if is_output else 2.0**0.5
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)


class BaseDQNVisualization:
    """
    Specialized visualization mixin for DQN-style networks.

    Provides forward_with_activations() implementation that captures
    input, hidden (feature layer), and output activations.
    """

    feature_layer: nn.Sequential
    value_stream: nn.Sequential
    advantage_stream: nn.Sequential
    output_size: int

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass that also returns intermediate activations.

        Args:
            x: Input state tensor

        Returns:
            Tuple of (q_values, activations_dict)
            - q_values: Standard Q-value output
            - activations_dict: Dict with 'input', 'hidden', 'output' tensors
        """
        activations = {}

        # Capture input
        activations["input"] = x.detach().cpu()

        # Feature layer
        features = self.feature_layer(x)
        activations["hidden"] = features.detach().cpu()

        # Compute Q-values. For a dueling head, also surface the value/advantage
        # decomposition (V(s) + per-action A(s,a)) so the UI can show *why* the
        # Q-values look the way they do, not just the fused result.
        if hasattr(self, "value_stream") and hasattr(self, "advantage_stream"):
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
            activations["value"] = value.detach().cpu()
            activations["advantages"] = advantages.detach().cpu()
        else:
            q_values = self._compute_q_values_for_viz(features)

        # Capture output
        activations["output"] = q_values.detach().cpu()

        return q_values, activations

    def _compute_q_values_for_viz(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values from features for visualization.

        Subclasses should override if they need different computation
        (e.g., distributional RL returns expected Q-values).

        Args:
            features: Feature tensor from feature_layer

        Returns:
            Q-values tensor
        """
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + (advantages - advantages.mean(dim=-1, keepdim=True))
