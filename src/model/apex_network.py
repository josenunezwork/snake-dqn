"""Ape-X DQN Network Architecture.

Implements a Dueling Network optimized for distributed Ape-X DQN training.
Designed for high throughput with torch.compile() support for H100 GPUs.

Key features:
- Dueling architecture: V(s) + A(s,a) - mean(A)
- Orthogonal weight initialization for stable distributed training
- Efficient weight sharing methods for actor synchronization
- torch.compile() compatible design

Reference: Horgan et al., "Distributed Prioritized Experience Replay" (2018)
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


class ApexNetwork(nn.Module, WeightManagementMixin, BaseDQNVisualization):
    """
    Ape-X DQN Network with Dueling Architecture.

    Optimized for distributed training with separate actors and a central learner.
    Uses a simple but effective MLP architecture for the 58D Snake state.

    Architecture:
        Input (58) -> Hidden1 (512, ReLU) -> Hidden2 (256, ReLU) -> Split
            -> Value Stream: (512) -> V(s) (1)
            -> Advantage Stream: (512) -> A(s,a) (6)
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))

    Features:
        - Dueling architecture for better value estimation
        - Orthogonal initialization for stable training
        - Efficient state dict sharing for actor synchronization
        - Compatible with torch.compile() for H100 optimization

    Attributes:
        input_size: State dimension (default: 58)
        hidden_size: Hidden layer size (default: 512)
        output_size: 6 actions: 3 directions × 2 speed modes (normal/boost)
        num_atoms: Always 1 for standard Q-values (not distributional)
    """

    def __init__(
        self,
        input_size: int = 58,
        hidden_size: int = 512,
        output_size: int = 6,
        init_type: str = "orthogonal"
    ):
        """
        Initialize Ape-X DQN Network.

        Args:
            input_size: State dimension (default: 58 for Snake game)
            hidden_size: Hidden layer size (default: 512)
            output_size: Number of actions (default: 6, 3 dirs × 2 speed modes)
            init_type: Weight initialization type, "orthogonal" or "xavier"
        """
        super(ApexNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_atoms = 1  # Standard Q-values (not distributional)
        self._init_type = init_type

        # Intermediate size for streams (half of hidden_size)
        self.stream_size = hidden_size // 2

        # Shared feature extractor: input -> hidden_size -> hidden_size // 2
        # Example with hidden_size=512: 58 -> 512 -> 256
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.stream_size),
            nn.ReLU()
        )

        # Dueling streams (split from shared features)
        # Value stream: estimates V(s) - how good is this state
        self.value_stream = nn.Sequential(
            nn.Linear(self.stream_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Advantage stream: estimates A(s,a) - relative advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.stream_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
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
        elif init_type == "xavier":
            init_weights_xavier(self)
        else:
            raise ValueError(f"Unknown init_type: {init_type}. Use 'orthogonal' or 'xavier'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values using dueling architecture.

        Args:
            x: Input state tensor of shape (batch_size, input_size)
               For Snake game: (batch_size, 58)

        Returns:
            Q-values for each action, shape (batch_size, output_size)
            For Snake game: (batch_size, 6)
        """
        # Extract shared features
        features = self.feature_layer(x)

        # Compute value and advantages
        value = self.value_stream(features)          # (batch, 1)
        advantages = self.advantage_stream(features)  # (batch, 6)

        # Combine using dueling formula: Q = V + (A - mean(A))
        return dueling_q(value, advantages)

    def _compute_q_values_for_viz(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from features for visualization."""
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return dueling_q(value, advantages)

    # =========================================================================
    # torch.compile() Compatibility Methods
    # =========================================================================

    def get_compiled_version(
        self,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False
    ) -> "ApexNetwork":
        """
        Get a torch.compile() optimized version of this network.

        Optimizes the forward pass for H100 GPUs using torch.compile().

        Args:
            mode: Compilation mode:
                - "default": Good balance of compile time and speedup
                - "reduce-overhead": Reduce framework overhead (good for inference)
                - "max-autotune": Maximum performance (longer compile time)
            fullgraph: If True, compile the full graph (stricter but faster)
            dynamic: If True, support dynamic batch sizes

        Returns:
            Compiled network (or self if torch.compile not available)
        """
        if not hasattr(torch, 'compile'):
            # PyTorch < 2.0, return self
            return self

        try:
            compiled = torch.compile(
                self,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic
            )
            return compiled
        except Exception as e:
            # Fallback if compilation fails
            print(f"Warning: torch.compile() failed: {e}. Using uncompiled network.")
            return self

    @staticmethod
    def compile_forward_only(network: "ApexNetwork", **compile_kwargs) -> callable:
        """
        Compile only the forward method for maximum compatibility.

        Useful when full network compilation causes issues.

        Args:
            network: Network instance to compile forward for
            **compile_kwargs: Arguments passed to torch.compile()

        Returns:
            Compiled forward function
        """
        if not hasattr(torch, 'compile'):
            return network.forward

        return torch.compile(network.forward, **compile_kwargs)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get parameter counts for network components.

        Returns:
            Dict with parameter counts:
            - 'total': Total parameters
            - 'feature_layer': Feature extractor parameters
            - 'value_stream': Value stream parameters
            - 'advantage_stream': Advantage stream parameters
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'total': count_params(self),
            'feature_layer': count_params(self.feature_layer),
            'value_stream': count_params(self.value_stream),
            'advantage_stream': count_params(self.advantage_stream)
        }

    def reset_parameters(self) -> None:
        """Reset all parameters to initial values."""
        self._init_weights(self._init_type)

    def __repr__(self) -> str:
        """String representation with architecture details."""
        params = self.get_num_parameters()
        return (
            f"ApexNetwork(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  output_size={self.output_size},\n"
            f"  init_type='{self._init_type}',\n"
            f"  total_params={params['total']:,}\n"
            f")"
        )


def create_apex_network_pair(
    input_size: int = 58,
    hidden_size: int = 512,
    output_size: int = 6,
    device: torch.device = None,
    init_type: str = "orthogonal",
) -> Tuple[ApexNetwork, ApexNetwork]:
    """
    Create main and target Ape-X network pair.

    Convenience function for creating the standard DQN network pair
    with proper initialization.

    Args:
        input_size: State dimension (default: 58 for Snake)
        hidden_size: Hidden layer size (default: 512)
        output_size: Number of actions (default: 6, 3 dirs × 2 speed modes)
        device: Device to place networks on (defaults to CUDA if available)
        init_type: Weight initialization type

    Returns:
        Tuple of (main_network, target_network)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create main network
    main_network = ApexNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        init_type=init_type
    ).to(device)

    # Create target network with same weights
    target_network = ApexNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        init_type=init_type
    ).to(device)

    # Initialize target with same weights as main
    target_network.copy_weights_from(main_network)
    target_network.eval()  # Target network is always in eval mode

    return main_network, target_network


def create_apex_actor_network(
    input_size: int = 58,
    hidden_size: int = 512,
    output_size: int = 6,
    device: torch.device = None,
    init_type: str = "orthogonal"
) -> ApexNetwork:
    """
    Create a single Ape-X network for actor processes.

    Actors only need one network (no target network) and typically
    run on CPU.

    Args:
        input_size: State dimension
        hidden_size: Hidden layer size
        output_size: Number of actions
        device: Device to place network on (defaults to CPU for actors)
        init_type: Weight initialization type

    Returns:
        ApexNetwork configured for actor use
    """
    if device is None:
        device = torch.device("cpu")  # Actors typically run on CPU

    network = ApexNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        init_type=init_type
    ).to(device)

    network.eval()  # Actors only do inference

    return network
