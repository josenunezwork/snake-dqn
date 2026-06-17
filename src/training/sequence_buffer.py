"""Sequence replay buffer for DRQN training.

Stores fixed-length sequences (trajectories) instead of individual transitions,
enabling recurrent networks (GRU/LSTM) to learn temporal dependencies.
Uses SumTree for prioritized sampling.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.game_config import GameConfig
from src.training.base_buffer import validate_next_action_mask
from src.training.sum_tree import SumTree

# Type alias for sequence batch
SequenceBatchDict = Dict[str, torch.Tensor]


def _coerce_exact_action_mask(mask) -> np.ndarray:
    """Validate and flatten a simulator-provided exact next-action mask."""
    validated_mask = validate_next_action_mask(mask, expected_size=GameConfig.OUTPUT_SIZE)
    if torch.is_tensor(validated_mask):
        mask_array = validated_mask.detach().cpu().numpy()
    else:
        mask_array = np.asarray(validated_mask)
    return mask_array.astype(np.bool_).reshape(-1)


class SequenceReplayBuffer:
    """Prioritized replay buffer that stores fixed-length sequences for DRQN.

    Each sequence is a trajectory of (state, action, reward, next_state, done)
    tuples, optionally followed by a next_action_mask, with a fixed length.
    Short sequences are zero-padded with masks.

    Burn-in support: the first burn_in_length steps are used to warm up the
    GRU hidden state and are excluded from loss computation via the masks tensor.
    """

    def __init__(
        self,
        capacity: int,
        sequence_length: int = 20,
        burn_in_length: int = 5,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 1_000_000,
        priority_eps: float = 1e-5,
    ):
        """Initialize sequence replay buffer.

        Args:
            capacity: Maximum number of sequences stored
            sequence_length: Fixed length of each stored sequence
            burn_in_length: Number of initial steps used for hidden state warmup
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction exponent
            beta_end: Final beta value (annealed over beta_frames)
            beta_frames: Number of frames over which to anneal beta
            priority_eps: Small constant to keep sequence priorities positive
        """
        if burn_in_length >= sequence_length:
            raise ValueError(
                f"burn_in_length ({burn_in_length}) must be less than "
                f"sequence_length ({sequence_length})"
            )

        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.priority_eps = priority_eps

        self._tree = SumTree(capacity)
        self._sequences: List[Optional[dict]] = [None] * capacity
        self._size = 0
        self._next_idx = 0
        self._frame_count = 0

    @property
    def beta(self) -> float:
        """Current beta value, linearly annealed from beta_start to beta_end."""
        fraction = min(1.0, self._frame_count / max(1, self.beta_frames))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def _build_sequence_dict(
        self,
        transitions: List[Tuple],
        start_index: int = 0,
    ) -> dict:
        """Build a padded sequence dict from a list of transitions.

        Args:
            transitions: List of (state, action, reward, next_state, done)
                tuples, optionally followed by next_action_mask. Length must fit
                between start_index and sequence_length.
            start_index: Sequence index where the first transition is written.

        Returns:
            Dictionary with numpy arrays for states, actions, rewards,
            next_states, dones, and masks.
        """
        if start_index < 0 or start_index >= self.sequence_length:
            raise ValueError(f"start_index ({start_index}) must be in [0, {self.sequence_length})")
        if len(transitions) > self.sequence_length - start_index:
            raise ValueError("transitions do not fit between start_index and sequence_length")

        valid_len = len(transitions)
        state_dim = len(transitions[0][0]) if hasattr(transitions[0][0], "__len__") else 1

        states = np.zeros((self.sequence_length, state_dim), dtype=np.float32)
        actions = np.zeros(self.sequence_length, dtype=np.int64)
        rewards = np.zeros(self.sequence_length, dtype=np.float32)
        next_states = np.zeros((self.sequence_length, state_dim), dtype=np.float32)
        dones = np.zeros(self.sequence_length, dtype=np.float32)
        masks = np.zeros(self.sequence_length, dtype=np.float32)
        next_action_masks = np.zeros(
            (self.sequence_length, GameConfig.OUTPUT_SIZE),
            dtype=np.bool_,
        )
        next_action_mask_present = np.zeros(self.sequence_length, dtype=np.float32)

        for offset, transition in enumerate(transitions):
            i = start_index + offset
            if len(transition) == 6:
                s, a, r, ns, d, next_action_mask = transition
            elif len(transition) == 5:
                s, a, r, ns, d = transition
                next_action_mask = None
            else:
                raise ValueError(
                    "Sequence transitions must have 5 fields "
                    "(state, action, reward, next_state, done) or 6 fields "
                    "with next_action_mask"
                )

            if torch.is_tensor(s):
                states[i] = s.detach().cpu().numpy()
            else:
                states[i] = np.asarray(s, dtype=np.float32)

            actions[i] = int(a)
            rewards[i] = float(r)

            if torch.is_tensor(ns):
                next_states[i] = ns.detach().cpu().numpy()
            else:
                next_states[i] = np.asarray(ns, dtype=np.float32)

            dones[i] = float(d)

            if next_action_mask is not None:
                mask_array = _coerce_exact_action_mask(next_action_mask)
                next_action_masks[i] = mask_array
                next_action_mask_present[i] = 1.0

            # Mask: 1 for valid steps that are past burn-in, 0 otherwise
            if i >= self.burn_in_length:
                masks[i] = 1.0

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "masks": masks,
            "next_action_masks": next_action_masks,
            "next_action_mask_present": next_action_mask_present,
            "valid_length": valid_len,
        }

    def add_sequence(
        self,
        sequence: List[Tuple],
        priority: Optional[float] = None,
        start_index: int = 0,
    ) -> None:
        """Add a single pre-built sequence to the buffer.

        Args:
            sequence: List of (state, action, reward, next_state, done) tuples.
                      Will be truncated or padded to sequence_length.
            priority: Initial priority. If None, uses max existing priority or 1.0.
            start_index: Sequence index where the first transition is written.
        """
        available_slots = self.sequence_length - start_index
        if available_slots <= 0:
            raise ValueError(
                f"start_index ({start_index}) must leave room for at least one transition"
            )
        # Truncate if too long. When right-aligning short episodes after the
        # burn-in window, keep the most recent transitions because they include
        # the terminal outcome.
        if len(sequence) > available_slots:
            if start_index > 0:
                sequence = sequence[-available_slots:]
            else:
                sequence = sequence[:available_slots]

        seq_dict = self._build_sequence_dict(sequence, start_index=start_index)

        if priority is None:
            max_p = self._tree.total() / max(1, self._size) if self._size > 0 else 1.0
            priority = max(max_p, 1.0)

        # Apply alpha exponent to priority
        tree_priority = (abs(priority) + self.priority_eps) ** self.alpha

        idx = self._next_idx
        self._sequences[idx] = seq_dict
        self._tree.add(tree_priority, idx)

        self._next_idx = (self._next_idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_episode(self, episode_transitions: List[Tuple]) -> None:
        """Split an episode into overlapping sequences and add them.

        Uses 50% overlap stride for continuity between sequences.
        Short episodes (< sequence_length) are added as a single padded sequence.

        Args:
            episode_transitions: Full list of transitions from one episode.
                Each is (state, action, reward, next_state, done), optionally
                followed by next_action_mask.
        """
        if not episode_transitions:
            return

        ep_len = len(episode_transitions)
        stride = max(1, self.sequence_length // 2)

        if ep_len <= self.burn_in_length:
            # Very short episodes used to be sampled with all-zero masks, which
            # wastes DRQN updates and hides early terminal lessons. Place their
            # available transitions after burn-in so they can contribute loss
            # with a zero recurrent context.
            self.add_sequence(episode_transitions, start_index=self.burn_in_length)
            return

        if ep_len <= self.sequence_length:
            # Single short sequence — add as-is (will be padded)
            self.add_sequence(episode_transitions)
            return

        # Split into overlapping windows
        start = 0
        while start < ep_len:
            end = min(start + self.sequence_length, ep_len)
            chunk = episode_transitions[start:end]
            self.add_sequence(chunk)

            # Advance by stride, but stop if we've covered the end
            start += stride
            if end == ep_len:
                break

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[SequenceBatchDict, List[int], torch.Tensor]:
        """Sample a batch of sequences with prioritized replay.

        Args:
            batch_size: Number of sequences to sample
            device: Device to place tensors on

        Returns:
            Tuple of (batch_dict, tree_indices, weights):
              - batch_dict: keys 'states', 'actions', 'rewards', 'next_states',
                           'dones', 'masks' with shapes (batch, seq_len, ...)
              - tree_indices: SumTree indices for priority updates
              - weights: Importance sampling weights tensor

        Raises:
            ValueError: If buffer has fewer sequences than batch_size
        """
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} sequences from buffer "
                f"with only {self._size} sequences"
            )

        self._frame_count += batch_size

        segment = self._tree.total() / batch_size
        tree_indices = []
        seq_indices = []
        priorities = []

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumsum = np.random.uniform(low, high)
            idx, priority, data_idx = self._tree.get(cumsum)
            tree_indices.append(idx)
            seq_indices.append(data_idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities_arr = np.array(priorities, dtype=np.float32)
        min_prob = priorities_arr.min() / self._tree.total()
        max_weight = (self._size * min_prob) ** (-self.beta)

        weights_arr = np.zeros(batch_size, dtype=np.float32)
        for i, p in enumerate(priorities_arr):
            prob = p / self._tree.total()
            w = (self._size * prob) ** (-self.beta)
            weights_arr[i] = w / max_weight

        # Assemble batch arrays
        first_seq = self._sequences[seq_indices[0]]
        state_dim = first_seq["states"].shape[1]

        batch_states = np.zeros((batch_size, self.sequence_length, state_dim), dtype=np.float32)
        batch_actions = np.zeros((batch_size, self.sequence_length), dtype=np.int64)
        batch_rewards = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
        batch_next_states = np.zeros(
            (batch_size, self.sequence_length, state_dim), dtype=np.float32
        )
        batch_dones = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
        batch_masks = np.zeros((batch_size, self.sequence_length), dtype=np.float32)
        batch_next_action_masks = np.zeros(
            (batch_size, self.sequence_length, GameConfig.OUTPUT_SIZE),
            dtype=np.bool_,
        )
        batch_next_action_mask_present = np.zeros(
            (batch_size, self.sequence_length),
            dtype=np.float32,
        )

        for i, si in enumerate(seq_indices):
            seq = self._sequences[si]
            batch_states[i] = seq["states"]
            batch_actions[i] = seq["actions"]
            batch_rewards[i] = seq["rewards"]
            batch_next_states[i] = seq["next_states"]
            batch_dones[i] = seq["dones"]
            batch_masks[i] = seq["masks"]
            batch_next_action_masks[i] = seq["next_action_masks"]
            batch_next_action_mask_present[i] = seq["next_action_mask_present"]

        batch_dict: SequenceBatchDict = {
            "states": torch.tensor(batch_states, dtype=torch.float32, device=device),
            "actions": torch.tensor(batch_actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(batch_rewards, dtype=torch.float32, device=device),
            "next_states": torch.tensor(batch_next_states, dtype=torch.float32, device=device),
            "dones": torch.tensor(batch_dones, dtype=torch.float32, device=device),
            "masks": torch.tensor(batch_masks, dtype=torch.float32, device=device),
        }
        if bool(batch_next_action_mask_present.any()):
            batch_dict["next_action_masks"] = torch.tensor(
                batch_next_action_masks,
                dtype=torch.bool,
                device=device,
            )
            batch_dict["next_action_mask_present"] = torch.tensor(
                batch_next_action_mask_present,
                dtype=torch.float32,
                device=device,
            )

        weights = torch.tensor(weights_arr, dtype=torch.float32, device=device)
        return batch_dict, tree_indices, weights

    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities using max TD error across each sequence.

        Args:
            tree_indices: SumTree indices returned by sample()
            td_errors: TD errors, shape (batch,) or (batch, seq_len).
                       If 2D, max across the sequence dimension is used.
        """
        if td_errors.ndim == 2:
            # Take max across sequence dimension
            max_errors = np.abs(td_errors).max(axis=1)
        else:
            max_errors = np.abs(td_errors)

        for idx, error in zip(tree_indices, max_errors):
            priority = (float(error) + self.priority_eps) ** self.alpha
            self._tree.update(idx, priority)

    def __len__(self) -> int:
        """Return number of stored sequences."""
        return self._size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough sequences for a batch."""
        return self._size >= batch_size

    def clear(self) -> None:
        """Clear all sequences from the buffer."""
        self._tree = SumTree(self.capacity)
        self._sequences = [None] * self.capacity
        self._size = 0
        self._next_idx = 0
        self._frame_count = 0
