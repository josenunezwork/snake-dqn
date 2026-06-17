"""GRU-enhanced Ape-X DQN Network Architecture (DRQN).

Implements a Dueling Network with GRU temporal memory for DRQN training.
The GRU layer enables the network to maintain temporal context across
sequential observations, improving performance in partially observable
environments.

Key features:
- GRU recurrent layer for temporal memory
- Dueling architecture: V(s) + A(s,a) - mean(A)
- Supports both single-step and sequence forward passes
- Hidden state management for episode boundaries
- Orthogonal weight initialization for stable distributed training
- Efficient weight sharing methods for actor synchronization

Reference: Hausknecht & Stone, "Deep Recurrent Q-Learning for POMDPs" (2015)
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from .base_network import (
    BaseDQNVisualization,
    WeightManagementMixin,
    dueling_q,
    init_weights_xavier,
    init_dueling_weights_orthogonal,
)


class GruApexNetwork(nn.Module, WeightManagementMixin, BaseDQNVisualization):
    """
    GRU-enhanced Ape-X DQN Network with Dueling Architecture.

    Extends the standard ApexNetwork with a GRU recurrent layer between
    the feature extractor and dueling streams. This allows the network
    to maintain temporal context across sequential observations.

    Architecture:
        Input (58) -> Hidden1 (512, ReLU) -> Hidden2 (256, ReLU)
                   -> GRU (256 -> 256, 1 layer)
                   -> Split:
                       -> Value Stream: (512) -> V(s) (1)
                       -> Advantage Stream: (512) -> A(s,a) (6)
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

    Attributes:
        input_size: State dimension (default: 58)
        hidden_size: Hidden layer size (default: 512)
        output_size: 6 actions: 3 directions x 2 speed modes (normal/boost)
        gru_hidden_size: GRU hidden state dimension (default: 256)
        num_gru_layers: Number of stacked GRU layers (default: 1)
        num_atoms: Always 1 for standard Q-values (not distributional)
    """

    def __init__(
        self,
        input_size: int = 58,
        hidden_size: int = 512,
        output_size: int = 6,
        gru_hidden_size: int = 256,
        num_gru_layers: int = 1,
        init_type: str = "orthogonal",
    ):
        """
        Initialize GRU-enhanced Ape-X DQN Network.

        Args:
            input_size: State dimension (default: 58 for Snake game)
            hidden_size: Hidden layer size (default: 512)
            output_size: Number of actions (default: 6, 3 dirs x 2 speed modes)
            gru_hidden_size: GRU hidden state size (default: 256)
            num_gru_layers: Number of stacked GRU layers (default: 1)
            init_type: Weight initialization type, "orthogonal" or "xavier"
        """
        super(GruApexNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru_hidden_size = gru_hidden_size
        self.num_gru_layers = num_gru_layers
        self.num_atoms = 1  # Standard Q-values (not distributional)
        self._init_type = init_type

        # Intermediate size for streams (half of hidden_size)
        self.stream_size = hidden_size // 2

        # Shared feature extractor: input -> hidden_size -> stream_size
        # Example with hidden_size=512: 58 -> 512 -> 256
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.stream_size),
            nn.ReLU(),
        )

        # GRU recurrent layer for temporal memory
        self.gru = nn.GRU(
            input_size=self.stream_size,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Dueling streams (split from GRU output)
        # Value stream: estimates V(s) - how good is this state
        self.value_stream = nn.Sequential(
            nn.Linear(gru_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Advantage stream: estimates A(s,a) - relative advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(gru_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # Apply weight initialization
        self._init_weights(init_type)

    def _init_weights(self, init_type: str) -> None:
        """
        Initialize network weights.

        Args:
            init_type: "orthogonal" for orthogonal init, "xavier" for Xavier init
        """
        if init_type == "orthogonal":
            init_dueling_weights_orthogonal(
                self.feature_layer, self.value_stream, self.advantage_stream
            )
            # Initialize GRU weights with orthogonal
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        elif init_type == "xavier":
            init_weights_xavier(self)
        else:
            raise ValueError(f"Unknown init_type: {init_type}. Use 'orthogonal' or 'xavier'")

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Q-values using GRU + dueling architecture.

        Args:
            x: Input state tensor. Supports two shapes:
               - Single step: (batch_size, input_size)
               - Sequence: (batch_size, seq_len, input_size)
            hidden: GRU hidden state of shape (num_layers, batch_size, gru_hidden_size).
                    If None, zeros are used.

        Returns:
            Tuple of (q_values, new_hidden):
            - q_values: Q-values for each action, shape (batch_size, output_size)
            - new_hidden: Detached hidden state for next call,
                          shape (num_layers, batch_size, gru_hidden_size)
        """
        # Handle single-step input: (batch, input_size) -> (batch, 1, input_size)
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size).to(x.device)

        # Extract features for each timestep
        # Reshape to (batch * seq_len, input_size) for feature layer
        x_flat = x.reshape(batch_size * seq_len, -1)
        features = self.feature_layer(x_flat)
        # Reshape back to (batch, seq_len, stream_size)
        features = features.reshape(batch_size, seq_len, self.stream_size)

        # Run through GRU
        gru_out, new_hidden = self.gru(features, hidden)

        # Use last timestep output for Q-value computation
        last_output = gru_out[:, -1, :]  # (batch, gru_hidden_size)

        # Compute value and advantages using dueling architecture
        value = self.value_stream(last_output)  # (batch, 1)
        advantages = self.advantage_stream(last_output)  # (batch, output_size)

        # Combine using dueling formula: Q = V + (A - mean(A))
        q_values = dueling_q(value, advantages)

        # Detach hidden state to prevent gradient leakage across calls
        new_hidden = new_hidden.detach()

        return q_values, new_hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize GRU hidden state with zeros.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Zero tensor of shape (num_gru_layers, batch_size, gru_hidden_size)
        """
        return torch.zeros(self.num_gru_layers, batch_size, self.gru_hidden_size)

    def _compute_q_values_for_viz(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from features for visualization.

        Note: This bypasses the GRU layer and operates directly on the
        feature layer output. Use forward() for proper temporal processing.
        """
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return dueling_q(value, advantages)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get parameter counts for network components.

        Returns:
            Dict with parameter counts for each component
        """

        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "total": count_params(self),
            "feature_layer": count_params(self.feature_layer),
            "gru": count_params(self.gru),
            "value_stream": count_params(self.value_stream),
            "advantage_stream": count_params(self.advantage_stream),
        }

    def reset_parameters(self) -> None:
        """Reset all parameters to initial values."""
        self._init_weights(self._init_type)

    def __repr__(self) -> str:
        """String representation with architecture details."""
        params = self.get_num_parameters()
        return (
            f"GruApexNetwork(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  output_size={self.output_size},\n"
            f"  gru_hidden_size={self.gru_hidden_size},\n"
            f"  num_gru_layers={self.num_gru_layers},\n"
            f"  init_type='{self._init_type}',\n"
            f"  total_params={params['total']:,}\n"
            f")"
        )


def create_gru_network_pair(
    input_size: int = 58,
    hidden_size: int = 512,
    output_size: int = 6,
    gru_hidden_size: int = 256,
    num_gru_layers: int = 1,
    device: torch.device = None,
    init_type: str = "orthogonal",
) -> Tuple[GruApexNetwork, GruApexNetwork]:
    """
    Create main and target GRU network pair.

    Args:
        input_size: State dimension (default: 58 for Snake)
        hidden_size: Hidden layer size (default: 512)
        output_size: Number of actions (default: 6, 3 dirs x 2 speed modes)
        gru_hidden_size: GRU hidden state size (default: 256)
        num_gru_layers: Number of stacked GRU layers (default: 1)
        device: Device to place networks on (defaults to CUDA if available)
        init_type: Weight initialization type

    Returns:
        Tuple of (main_network, target_network)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_network = GruApexNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        gru_hidden_size=gru_hidden_size,
        num_gru_layers=num_gru_layers,
        init_type=init_type,
    ).to(device)

    target_network = GruApexNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        gru_hidden_size=gru_hidden_size,
        num_gru_layers=num_gru_layers,
        init_type=init_type,
    ).to(device)

    target_network.copy_weights_from(main_network)
    target_network.eval()

    return main_network, target_network


def create_gru_actor_network(
    input_size: int = 58,
    hidden_size: int = 512,
    output_size: int = 6,
    gru_hidden_size: int = 256,
    num_gru_layers: int = 1,
    device: torch.device = None,
    init_type: str = "orthogonal",
) -> GruApexNetwork:
    """
    Create a single GRU network for actor processes.

    Actors only need one network (no target network) and typically
    run on CPU.

    Args:
        input_size: State dimension
        hidden_size: Hidden layer size
        output_size: Number of actions
        gru_hidden_size: GRU hidden state size
        num_gru_layers: Number of stacked GRU layers
        device: Device to place network on (defaults to CPU for actors)
        init_type: Weight initialization type

    Returns:
        GruApexNetwork configured for actor use
    """
    if device is None:
        device = torch.device("cpu")

    network = GruApexNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        gru_hidden_size=gru_hidden_size,
        num_gru_layers=num_gru_layers,
        init_type=init_type,
    ).to(device)

    network.eval()

    return network
