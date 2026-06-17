"""Factory for creating neural network models.

Simplified factory for Apex DQN networks only.
"""
import torch
from typing import Tuple

from .apex_network import ApexNetwork
from ..core.game_config import GameConfig


class ModelFactory:
    """Factory for creating Apex DQN models."""

    # Registry of model types - Apex only
    MODEL_REGISTRY = {
        'apex': ApexNetwork,
        'apex_dqn': ApexNetwork,  # Alias
    }

    @classmethod
    def create_model_by_type(
        cls,
        model_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        **kwargs
    ) -> ApexNetwork:
        """
        Create a model by type name.

        Args:
            model_type: Model type string (e.g., 'apex', 'apex_dqn')
            input_size: State dimension
            hidden_size: Hidden layer size
            output_size: Number of actions
            device: Device to place model on
            **kwargs: Additional model-specific arguments

        Returns:
            Instantiated ApexNetwork on specified device

        Raises:
            ValueError: If model_type is not recognized
        """
        model_type = model_type.lower().replace('-', '_')
        if model_type not in cls.MODEL_REGISTRY:
            available = ', '.join(sorted(set(cls.MODEL_REGISTRY.keys())))
            raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")

        model_class = cls.MODEL_REGISTRY[model_type]
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            **kwargs
        ).to(device)

        return model

    @classmethod
    def create_model_pair(
        cls,
        model_type: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        **kwargs
    ) -> Tuple[ApexNetwork, ApexNetwork]:
        """
        Create main and target model pair for DQN-style algorithms.

        Args:
            model_type: Model type string
            input_size: State dimension
            hidden_size: Hidden layer size
            output_size: Number of actions
            device: Device to place models on
            **kwargs: Additional model-specific arguments

        Returns:
            Tuple of (main_model, target_model)
        """
        main_model = cls.create_model_by_type(
            model_type, input_size, hidden_size, output_size, device, **kwargs
        )
        target_model = cls.create_model_by_type(
            model_type, input_size, hidden_size, output_size, device, **kwargs
        )

        target_model.load_state_dict(main_model.state_dict())
        target_model.eval()  # Target network is always in eval mode

        return main_model, target_model

    @staticmethod
    def create_model(
        config: GameConfig,
        device: torch.device
    ) -> Tuple[ApexNetwork, ApexNetwork]:
        """
        Create main and target Apex DQN models.

        Args:
            config: Configuration object
            device: Device to place models on

        Returns:
            Tuple of (main_model, target_model)
        """
        # Create main model
        main_model = ApexNetwork(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            output_size=config.OUTPUT_SIZE
        ).to(device)

        # Create target model and initialize with same weights
        target_model = ApexNetwork(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            output_size=config.OUTPUT_SIZE
        ).to(device)

        target_model.load_state_dict(main_model.state_dict())
        target_model.eval()  # Target network is always in eval mode

        return main_model, target_model

