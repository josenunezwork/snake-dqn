"""Priority calculation and update utilities for Ape-X DQN.

This module provides utility functions for computing TD errors, priorities,
and importance sampling weights used in distributed prioritized experience replay.

Ape-X (Distributed Prioritized Experience Replay) architecture:
- Actors compute initial priorities from TD errors using local Q-networks
- Learner updates priorities after computing more accurate TD errors
- Priorities determine sampling probability in the replay buffer

References:
    Horgan et al., "Distributed Prioritized Experience Replay" (2018)
    https://arxiv.org/abs/1803.00933
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .action_mask import has_valid_actions, mask_invalid_q_values

# =============================================================================
# Constants
# =============================================================================

DEFAULT_EPSILON = 1e-6  # Small constant to avoid zero priorities
DEFAULT_ALPHA = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
DEFAULT_BETA_START = 0.4  # Initial importance sampling exponent
DEFAULT_BETA_END = 1.0  # Final importance sampling exponent


# =============================================================================
# TD Error Calculation
# =============================================================================


def compute_td_error(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    bootstrap_steps: Optional[torch.Tensor] = None,
    next_states: Optional[torch.Tensor] = None,
    next_action_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute TD errors for a batch of transitions.

    TD Error = |r + gamma * max_a Q(s', a) - Q(s, a)|

    This is the standard one-step TD error used for priority calculation.
    For actors, this provides initial priorities. For the learner, this
    provides updated priorities after training.

    Args:
        q_values: Q-values for current states, shape (batch_size, num_actions)
        actions: Actions taken, shape (batch_size,) or (batch_size, 1)
        rewards: Rewards received, shape (batch_size,)
        next_q_values: Q-values for next states, shape (batch_size, num_actions)
        dones: Terminal flags, shape (batch_size,)
        gamma: Discount factor
        bootstrap_steps: Optional per-sample bootstrap horizon
        next_states: Optional next-state rows used to mask invalid actions
        next_action_masks: Optional exact valid-action masks for next_states

    Returns:
        TD errors as tensor, shape (batch_size,)
    """
    # Handle action tensor shape
    if actions.dim() == 2:
        actions = actions.squeeze(1)

    # Get Q-value for the action taken
    current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Get maximum Q-value among actions the actor could execute next.
    if next_action_masks is not None or next_states is not None:
        next_q_values = mask_invalid_q_values(
            next_q_values,
            next_states,
            action_masks=next_action_masks,
        )
    valid_next_actions = has_valid_actions(
        next_q_values,
        next_states,
        action_masks=next_action_masks,
    )
    max_next_q = next_q_values.max(dim=1)[0]
    max_next_q = torch.where(valid_next_actions, max_next_q, torch.zeros_like(max_next_q))

    # Compute target Q-value with per-sample n-step horizons.
    if bootstrap_steps is None:
        bootstrap_steps = torch.ones_like(rewards)
    discounts = torch.pow(
        torch.full_like(rewards, gamma),
        bootstrap_steps.to(rewards.device),
    )
    target_q = rewards + discounts * max_next_q * (1.0 - dones)

    # TD error is the absolute difference
    td_error = torch.abs(current_q - target_q)

    return td_error


def compute_td_error_double_dqn(
    online_q_values: torch.Tensor,
    target_q_values: torch.Tensor,
    online_next_q_values: torch.Tensor,
    target_next_q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    bootstrap_steps: Optional[torch.Tensor] = None,
    next_states: Optional[torch.Tensor] = None,
    next_action_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute TD errors using Double DQN formulation.

    Double DQN uses the online network to select actions and the target
    network to evaluate them, reducing overestimation bias.

    TD Error = |r + gamma * Q_target(s', argmax_a Q_online(s', a)) - Q(s, a)|

    Args:
        online_q_values: Online network Q-values for current states
        target_q_values: Target network Q-values for current states (unused, for API consistency)
        online_next_q_values: Online network Q-values for next states
        target_next_q_values: Target network Q-values for next states
        actions: Actions taken, shape (batch_size,) or (batch_size, 1)
        rewards: Rewards received, shape (batch_size,)
        dones: Terminal flags, shape (batch_size,)
        gamma: Discount factor
        bootstrap_steps: Optional per-sample bootstrap horizon
        next_states: Optional next-state rows used to mask invalid actions
        next_action_masks: Optional exact valid-action masks for next_states

    Returns:
        TD errors as tensor, shape (batch_size,)
    """
    # Handle action tensor shape
    if actions.dim() == 2:
        actions = actions.squeeze(1)

    # Get Q-value for the action taken (from online network)
    current_q = online_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN: use online network to select a valid action, target network to evaluate
    if next_action_masks is not None or next_states is not None:
        online_next_q_values = mask_invalid_q_values(
            online_next_q_values,
            next_states,
            action_masks=next_action_masks,
        )
    valid_next_actions = has_valid_actions(
        online_next_q_values,
        next_states,
        action_masks=next_action_masks,
    )
    next_actions = online_next_q_values.argmax(dim=1)
    next_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    next_q = torch.where(valid_next_actions, next_q, torch.zeros_like(next_q))

    # Compute target Q-value
    if bootstrap_steps is None:
        bootstrap_steps = torch.ones_like(rewards)
    discounts = torch.pow(
        torch.full_like(rewards, gamma),
        bootstrap_steps.to(rewards.device),
    )
    target_q = rewards + discounts * next_q * (1.0 - dones)

    # TD error is the absolute difference
    td_error = torch.abs(current_q - target_q)

    return td_error


def compute_td_error_numpy(
    current_q: np.ndarray,
    reward: np.ndarray,
    next_max_q: np.ndarray,
    done: np.ndarray,
    gamma: float = 0.99,
    bootstrap_steps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute TD errors using numpy arrays.

    Simplified version for actors that already have the Q-values extracted.

    Args:
        current_q: Q-value for the action taken, shape (batch_size,)
        reward: Rewards received, shape (batch_size,)
        next_max_q: Maximum Q-value for next state, shape (batch_size,)
        done: Terminal flags, shape (batch_size,)
        gamma: Discount factor
        bootstrap_steps: Optional per-sample bootstrap horizon

    Returns:
        TD errors as numpy array, shape (batch_size,)
    """
    if bootstrap_steps is None:
        bootstrap_steps = np.ones_like(reward, dtype=np.float32)
    discounts = np.power(gamma, bootstrap_steps)
    target_q = reward + discounts * next_max_q * (1.0 - done.astype(np.float32))
    return np.abs(current_q - target_q)


# =============================================================================
# Priority Conversion
# =============================================================================


def td_error_to_priority(
    td_error: Union[float, np.ndarray, torch.Tensor],
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
) -> Union[float, np.ndarray]:
    """
    Convert TD error to priority value.

    priority = (|TD_error| + epsilon) ^ alpha

    Args:
        td_error: TD error value(s), can be scalar, numpy array, or torch tensor
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        epsilon: Small constant to avoid zero priorities

    Returns:
        Priority value(s) with same type as input (float or numpy array)
    """
    if isinstance(td_error, torch.Tensor):
        td_error = td_error.detach().cpu().numpy()

    td_error = np.asarray(td_error)
    priority = (np.abs(td_error) + epsilon) ** alpha

    # Return scalar if input was scalar
    if priority.ndim == 0:
        return float(priority)
    return priority


def priority_to_probability(priorities: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert priorities to sampling probabilities.

    P(i) = priority_i / sum(priorities)

    Note: Priorities should already have alpha applied via td_error_to_priority().

    Args:
        priorities: Array of priority values
        normalize: Whether to normalize to sum to 1.0

    Returns:
        Normalized probability distribution
    """
    if len(priorities) == 0:
        return np.array([])

    total = np.sum(priorities)
    if total == 0 or not normalize:
        return np.ones_like(priorities) / len(priorities)

    return priorities / total


# =============================================================================
# Importance Sampling Weights
# =============================================================================


def compute_importance_weights(
    buffer_size: int, probabilities: np.ndarray, indices: Union[List[int], np.ndarray], beta: float
) -> np.ndarray:
    """
    Compute importance sampling weights for bias correction.

    weight = (N * P(i)) ^ (-beta) / max_weight

    These weights correct for the bias introduced by non-uniform sampling.
    Beta starts at 0.4 and anneals to 1.0 during training, providing
    full correction as training progresses.

    Args:
        buffer_size: Total number of samples in buffer (N)
        probabilities: Sampling probability for each sample
        indices: Indices of sampled items
        beta: Importance sampling exponent (0 = no correction, 1 = full correction)

    Returns:
        Normalized importance sampling weights
    """
    if len(indices) == 0:
        return np.array([])

    indices = np.asarray(indices)

    # Compute raw importance sampling weights
    weights = (buffer_size * probabilities[indices]) ** (-beta)

    # Normalize by max weight for stability
    max_weight = weights.max()
    if max_weight > 0:
        weights = weights / max_weight

    return weights


def compute_importance_weights_from_priorities(
    priorities: np.ndarray, indices: Union[List[int], np.ndarray], beta: float
) -> np.ndarray:
    """
    Compute importance sampling weights directly from priorities.

    Convenience function that computes probabilities and then weights.

    Args:
        priorities: Array of all priority values in buffer
        indices: Indices of sampled items
        beta: Importance sampling exponent

    Returns:
        Normalized importance sampling weights
    """
    probabilities = priority_to_probability(priorities)
    return compute_importance_weights(len(priorities), probabilities, indices, beta)


# =============================================================================
# Beta Annealing
# =============================================================================


@dataclass
class BetaScheduler:
    """
    Scheduler for annealing beta from start value to 1.0.

    Beta controls the strength of importance sampling correction:
    - beta = 0: No correction (biased but higher variance)
    - beta = 1: Full correction (unbiased but lower variance)

    Attributes:
        beta_start: Initial beta value
        beta_end: Final beta value (typically 1.0)
        total_steps: Number of steps to anneal over
        current_step: Current step count
    """

    beta_start: float = DEFAULT_BETA_START
    beta_end: float = DEFAULT_BETA_END
    total_steps: int = 100000
    current_step: int = 0

    @property
    def beta(self) -> float:
        """Get current beta value."""
        if self.total_steps <= 0:
            return self.beta_end

        fraction = min(1.0, self.current_step / self.total_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * fraction

    def step(self, n: int = 1) -> float:
        """
        Advance scheduler by n steps and return current beta.

        Args:
            n: Number of steps to advance

        Returns:
            Current beta value after stepping
        """
        self.current_step = min(self.current_step + n, self.total_steps)
        return self.beta

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_step = 0

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
        }

    def load_state(self, state: dict) -> None:
        """Load scheduler state from checkpoint."""
        self.beta_start = state.get("beta_start", DEFAULT_BETA_START)
        self.beta_end = state.get("beta_end", DEFAULT_BETA_END)
        self.total_steps = state.get("total_steps", 100000)
        self.current_step = state.get("current_step", 0)


def compute_beta_linear(
    current_step: int,
    total_steps: int,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
) -> float:
    """
    Compute beta using linear annealing schedule.

    Args:
        current_step: Current training step
        total_steps: Total training steps
        beta_start: Initial beta value
        beta_end: Final beta value

    Returns:
        Current beta value
    """
    if total_steps <= 0:
        return beta_end

    fraction = min(1.0, current_step / total_steps)
    return beta_start + (beta_end - beta_start) * fraction


# =============================================================================
# Batch Priority Update
# =============================================================================


def update_priorities_batch(
    current_priorities: np.ndarray,
    indices: Union[List[int], np.ndarray],
    td_errors: Union[np.ndarray, torch.Tensor],
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """
    Update priorities for a batch of indices based on new TD errors.

    This is the main function used by the learner after computing new
    TD errors during training.

    Args:
        current_priorities: Current priority array (will be modified in-place)
        indices: Indices of samples to update
        td_errors: New TD errors for the samples
        alpha: Priority exponent
        epsilon: Small constant to avoid zero priorities

    Returns:
        Updated priorities array (same reference as input)
    """
    indices = np.asarray(indices)
    new_priorities = td_error_to_priority(td_errors, alpha, epsilon)

    current_priorities[indices] = new_priorities
    return current_priorities


def get_max_priority(priorities: np.ndarray, default: float = 1.0) -> float:
    """
    Get the maximum priority in the buffer.

    Used for assigning initial priority to new experiences.
    New experiences get max priority to ensure they are sampled at least once.

    Args:
        priorities: Array of current priorities
        default: Default value if array is empty

    Returns:
        Maximum priority value
    """
    if len(priorities) == 0:
        return default
    return float(np.max(priorities))


# =============================================================================
# Priority Statistics
# =============================================================================


@dataclass
class PriorityStatistics:
    """
    Statistics about priority distribution for debugging and monitoring.

    Attributes:
        min_priority: Minimum priority value
        max_priority: Maximum priority value
        mean_priority: Mean priority value
        std_priority: Standard deviation of priorities
        median_priority: Median priority value
        num_samples: Number of samples in buffer
        effective_samples: Effective sample size (measure of sample diversity)
    """

    min_priority: float = 0.0
    max_priority: float = 0.0
    mean_priority: float = 0.0
    std_priority: float = 0.0
    median_priority: float = 0.0
    num_samples: int = 0
    effective_samples: float = 0.0


def compute_priority_statistics(priorities: np.ndarray) -> PriorityStatistics:
    """
    Compute statistics about the priority distribution.

    Useful for debugging and monitoring the health of prioritized replay.
    High priority variance might indicate unstable learning.

    Args:
        priorities: Array of priority values

    Returns:
        PriorityStatistics dataclass with computed statistics
    """
    if len(priorities) == 0:
        return PriorityStatistics()

    # Basic statistics
    min_p = float(np.min(priorities))
    max_p = float(np.max(priorities))
    mean_p = float(np.mean(priorities))
    std_p = float(np.std(priorities))
    median_p = float(np.median(priorities))

    # Effective sample size: measures how uniform the distribution is
    # ESS = 1 / sum(p_i^2) where p_i are normalized probabilities
    # Higher ESS means more uniform sampling
    probs = priority_to_probability(priorities)
    if np.sum(probs**2) > 0:
        effective_samples = 1.0 / np.sum(probs**2)
    else:
        effective_samples = float(len(priorities))

    return PriorityStatistics(
        min_priority=min_p,
        max_priority=max_p,
        mean_priority=mean_p,
        std_priority=std_p,
        median_priority=median_p,
        num_samples=len(priorities),
        effective_samples=effective_samples,
    )


def log_priority_statistics(
    priorities: np.ndarray, prefix: str = "", logger=None
) -> Dict[str, float]:
    """
    Compute and optionally log priority statistics.

    Args:
        priorities: Array of priority values
        prefix: Prefix for log messages/dict keys
        logger: Optional logger instance (uses print if None)

    Returns:
        Dictionary of statistics for external logging
    """
    stats = compute_priority_statistics(priorities)

    prefix_str = f"{prefix}/" if prefix else ""

    stats_dict = {
        f"{prefix_str}priority_min": stats.min_priority,
        f"{prefix_str}priority_max": stats.max_priority,
        f"{prefix_str}priority_mean": stats.mean_priority,
        f"{prefix_str}priority_std": stats.std_priority,
        f"{prefix_str}priority_median": stats.median_priority,
        f"{prefix_str}buffer_size": stats.num_samples,
        f"{prefix_str}effective_samples": stats.effective_samples,
    }

    if logger is not None:
        for key, value in stats_dict.items():
            logger.info(f"{key}: {value:.6f}")

    return stats_dict


# =============================================================================
# Actor Priority Computation
# =============================================================================


def compute_actor_priorities(
    q_network: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    device: Optional[torch.device] = None,
    bootstrap_steps: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Compute initial priorities for actor-collected experiences.

    This is called by actors before sending experiences to the replay buffer.
    Uses the actor's local Q-network for TD error estimation.

    Args:
        q_network: Actor's Q-network
        states: Batch of states
        actions: Batch of actions
        rewards: Batch of rewards
        next_states: Batch of next states
        dones: Batch of done flags
        gamma: Discount factor
        alpha: Priority exponent
        epsilon: Small constant for priority
        device: Device for computation
        bootstrap_steps: Optional per-sample bootstrap horizon

    Returns:
        Array of initial priorities
    """
    q_network.eval()

    with torch.no_grad():
        if device is not None:
            states = states.to(device)
            next_states = next_states.to(device)

        # Get Q-values
        q_values = q_network(states)
        next_q_values = q_network(next_states)

        # Ensure other tensors are on same device
        if device is not None:
            actions = actions.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)
            if bootstrap_steps is not None:
                bootstrap_steps = bootstrap_steps.to(device)

        # Compute TD errors
        td_errors = compute_td_error(
            q_values,
            actions,
            rewards,
            next_q_values,
            dones,
            gamma,
            bootstrap_steps=bootstrap_steps,
            next_states=next_states,
        )

        # Convert to priorities
        priorities = td_error_to_priority(td_errors, alpha, epsilon)

    return priorities


# =============================================================================
# Learner Priority Update
# =============================================================================


def compute_learner_priorities(
    online_network: torch.nn.Module,
    target_network: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
    use_double_dqn: bool = True,
) -> np.ndarray:
    """
    Compute updated priorities after learner training step.

    This is called by the learner after each training batch to update
    priorities based on the most current Q-network.

    Args:
        online_network: Online Q-network (being trained)
        target_network: Target Q-network (for stability)
        batch: Batch dictionary with keys 'states', 'actions', 'rewards',
               'next_states', 'dones'
        gamma: Discount factor
        alpha: Priority exponent
        epsilon: Small constant for priority
        use_double_dqn: Whether to use Double DQN TD error computation

    Returns:
        Array of updated priorities
    """
    online_network.eval()
    target_network.eval()

    with torch.no_grad():
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        bootstrap_steps = batch.get("bootstrap_steps")
        next_action_masks = batch.get("next_action_masks")

        online_q = online_network(states)
        target_q = target_network(states)
        online_next_q = online_network(next_states)
        target_next_q = target_network(next_states)

        if use_double_dqn:
            td_errors = compute_td_error_double_dqn(
                online_q,
                target_q,
                online_next_q,
                target_next_q,
                actions,
                rewards,
                dones,
                gamma,
                bootstrap_steps=bootstrap_steps,
                next_states=next_states,
                next_action_masks=next_action_masks,
            )
        else:
            td_errors = compute_td_error(
                online_q,
                actions,
                rewards,
                target_next_q,
                dones,
                gamma,
                bootstrap_steps=bootstrap_steps,
                next_states=next_states,
                next_action_masks=next_action_masks,
            )

        priorities = td_error_to_priority(td_errors, alpha, epsilon)

    return priorities
