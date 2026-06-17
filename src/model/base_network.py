"""Base network utilities and mixins for building neural networks.

Provides reusable building blocks for constructing neural network
architectures with less code duplication.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Type
from collections import OrderedDict


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
    return value_stream + (
        advantage_stream - advantage_stream.mean(dim=-1, keepdim=True)
    )


def build_mlp(
    sizes: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: bool = False
) -> nn.Sequential:
    """
    Build a Multi-Layer Perceptron (MLP) from a list of layer sizes.

    Args:
        sizes: List of layer sizes. E.g., [64, 128, 64, 4] creates
               a network with layers: 64->128, 128->64, 64->4
        activation: Activation function class (default: nn.ReLU)
        output_activation: Whether to add activation after final layer

    Returns:
        nn.Sequential containing the MLP layers

    Example:
        >>> mlp = build_mlp([58, 512, 256, 6])
        >>> # Creates: Linear(58,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,6)
    """
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Add activation after each layer except the last (unless output_activation=True)
        if i < len(sizes) - 2 or output_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


def build_feature_layer(
    input_size: int,
    hidden_size: int,
) -> nn.Sequential:
    """
    Build a standard feature extraction layer.

    Creates the common pattern: Linear -> ReLU -> Linear -> ReLU
    with sizes: input_size -> hidden_size -> hidden_size // 2

    Args:
        input_size: Input dimension
        hidden_size: First hidden layer size (second will be hidden_size // 2)

    Returns:
        nn.Sequential containing the feature layer
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU()
    )


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
    Used by both ApexNetwork and GruApexNetwork.

    Args:
        feature_layer: Shared feature extraction sequential module
        value_stream: Value stream sequential module
        advantage_stream: Advantage stream sequential module
    """
    for module in [feature_layer, value_stream, advantage_stream]:
        for i, layer in enumerate(module):
            if isinstance(layer, nn.Linear):
                is_output = i == len(module) - 1
                gain = 1.0 if is_output else 2.0 ** 0.5
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)


class WeightManagementMixin:
    """Mixin providing weight copy, sync, and sharing methods for DQN networks.

    Eliminates duplicated weight management code between ApexNetwork and
    GruApexNetwork. Both networks share identical implementations of these
    methods.

    Requires the class to be an nn.Module (provides state_dict, parameters, etc.).
    """

    def copy_weights_from(self, source_network: nn.Module) -> None:
        """Copy weights from another network (hard update for target network).

        Args:
            source_network: Network to copy weights from
        """
        self.load_state_dict(source_network.state_dict())

    def soft_update_from(self, source_network: nn.Module, tau: float = 0.005) -> None:
        """Soft update weights from source network (Polyak averaging).

        Updates: target = tau * source + (1 - tau) * target

        Args:
            source_network: Network to copy weights from
            tau: Interpolation parameter (0 < tau <= 1)
        """
        with torch.no_grad():
            for target_param, source_param in zip(
                self.parameters(), source_network.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

    def get_shareable_state_dict(self) -> OrderedDict:
        """Get state dict optimized for sharing with actors.

        Returns a CPU state dict that can be efficiently serialized
        and sent to actor processes.

        Returns:
            OrderedDict containing model weights on CPU
        """
        return OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in self.state_dict().items()
        )

    def load_shareable_state_dict(
        self,
        state_dict: OrderedDict,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load state dict received from learner (for actor synchronization).

        Args:
            state_dict: State dict (typically from get_shareable_state_dict())
            device: Device to move weights to (uses current device if None)
        """
        if device is None:
            device = next(self.parameters()).device

        self.load_state_dict(state_dict)
        self.to(device)


class VisualizationMixin:
    """
    Mixin that adds forward_with_activations() for network visualization.

    Provides a standardized way to capture intermediate activations
    during forward pass for debugging and visualization purposes.
    """


class BaseDQNVisualization(VisualizationMixin):
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
        self,
        x: torch.Tensor
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
        activations['input'] = x.detach().cpu()

        # Feature layer
        features = self.feature_layer(x)
        activations['hidden'] = features.detach().cpu()

        # Compute Q-values (subclass implements _compute_q_values)
        q_values = self._compute_q_values_for_viz(features)

        # Capture output
        activations['output'] = q_values.detach().cpu()

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
