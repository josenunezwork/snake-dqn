"""Base class for DQN-style policies with common functionality."""

from typing import Optional, Tuple

import torch


class BaseDQNPolicy:
    """
    Base class for value-based DQN policies.

    Provides common functionality for:
    - Epsilon-greedy exploration
    - State dict management with type verification
    - Memory cleanup
    - Update counter and reward tracking

    Subclasses should implement:
    - select_action(): Action selection logic
    - update(): Training update logic
    - get_state_dict(): Full state serialization
    - load_state_dict(): State loading
    """

    def __init__(self, policy_name: str, device: torch.device):
        """
        Initialize base DQN policy.

        Args:
            policy_name: Policy identifier string
            device: Device to place tensors on
        """
        self.device = device
        self._policy_name = policy_name
        self._epsilon = 1.0
        self.update_counter = 0
        self.total_reward = 0.0

    def get_policy_name(self) -> str:
        """Return policy type identifier."""
        return self._policy_name

    @property
    def epsilon(self) -> float:
        """Get current epsilon value."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        """Set epsilon value with clamping."""
        self._epsilon = max(0.0, min(1.0, value))

    def _verify_checkpoint_type(self, state_dict: dict, expected_type: str) -> None:
        """
        Validate checkpoint is for correct policy type.

        Args:
            state_dict: Checkpoint dictionary
            expected_type: Expected policy type string

        Raises:
            ValueError: If policy type doesn't match
        """
        checkpoint_type = state_dict.get("policy_type")
        if checkpoint_type not in [None, expected_type]:
            raise ValueError(
                f"Policy type mismatch: expected {expected_type}, " f"got {checkpoint_type}"
            )

    def _base_state_dict(self) -> dict:
        """
        Build common state dict fields.

        Returns:
            Dictionary with common policy state
        """
        return {
            "policy_type": self._policy_name,
            "epsilon": self._epsilon,
            "update_counter": self.update_counter,
            "total_reward": self.total_reward,
        }

    def _load_base_state(self, state_dict: dict) -> None:
        """
        Load common state fields from checkpoint.

        Args:
            state_dict: Checkpoint dictionary
        """
        self._epsilon = state_dict.get("epsilon", 1.0)
        self.update_counter = state_dict.get("update_counter", 0)
        self.total_reward = state_dict.get("total_reward", 0.0)

    def cleanup(self) -> None:
        """Release resources and clear memory."""
        if hasattr(self, "memory") and self.memory is not None:
            self.memory.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def select_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> int:
        """Select action using epsilon-greedy policy."""
        raise NotImplementedError

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: Optional[torch.Tensor],
        done: bool,
    ) -> Tuple[Optional[float], float]:
        """Update policy with transition."""
        raise NotImplementedError

    def get_state_dict(self) -> dict:
        """Get serializable state for checkpointing."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict) -> None:
        """Load from checkpoint."""
        raise NotImplementedError

    def get_all_memories(self) -> list:
        """Get all stored memories (default: empty)."""
        return []

    def prepare_memories_for_saving(self) -> list:
        """Prepare memories for database storage (default: empty)."""
        return []
