"""Multi-step replay buffer for n-step returns."""

from collections import deque
from typing import Deque, Dict, Hashable, Optional, Tuple

import torch

from .replay_buffer import PrioritizedReplayBuffer


class MultiStepBuffer(PrioritizedReplayBuffer):
    """
    Prioritized replay buffer with n-step returns.

    Computes n-step returns: R_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γⁿV(s_{t+n})
    """

    def __init__(
        self,
        capacity: int,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_increment: float = 0.000001,
        priority_eps: float = 1e-6,
    ):
        """
        Initialize multi-step replay buffer.

        Args:
            capacity: Maximum buffer size
            n_step: Number of steps for n-step returns
            gamma: Discount factor
            alpha: Priority exponent
            beta_start: Initial importance sampling weight
            beta_end: Maximum importance sampling weight after annealing
            beta_increment: Beta increment per sample
            priority_eps: Small constant to prevent zero priorities
        """
        super().__init__(capacity, alpha, beta_start, beta_end, beta_increment, priority_eps)

        self.n_step = n_step
        self.gamma = gamma

        # N-step buffer to accumulate transitions
        self.n_step_buffer = deque(maxlen=n_step)
        self._stream_buffers: Dict[Hashable, Deque] = {}

    def _get_stream_buffer(self, stream_id: Optional[Hashable] = None) -> Deque:
        """Return the n-step buffer for one independent trajectory stream."""
        if stream_id is None:
            return self.n_step_buffer
        if stream_id not in self._stream_buffers:
            self._stream_buffers[stream_id] = deque(maxlen=self.n_step)
        return self._stream_buffers[stream_id]

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: Optional[float] = None,
        stream_id: Optional[Hashable] = None,
        next_action_mask=None,
    ) -> None:
        """
        Add transition and compute n-step return.

        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            done: Terminal flag
            priority: Experience priority
            stream_id: Optional independent trajectory id. Use this when
                multiple actors/snakes share one replay buffer so n-step
                returns do not mix transitions from different snakes.
            next_action_mask: Optional valid-action mask for next_state.
        """
        n_step_buffer = self._get_stream_buffer(stream_id)

        # Add to n-step buffer
        n_step_buffer.append((state, action, reward, next_state, done, next_action_mask))

        # If a short episode ends before the n-step window fills, still
        # materialize its terminal transitions. Early crashes are high-value
        # learning signal and must not vanish from replay.
        if len(n_step_buffer) < self.n_step:
            if done:
                self._flush_buffer(n_step_buffer, stream_id=stream_id)
                if stream_id is not None:
                    self._stream_buffers.pop(stream_id, None)
            return

        # Compute n-step return
        n_step_return, n_step_next_state, n_step_done, bootstrap_steps, next_action_mask = (
            self._compute_n_step_return(n_step_buffer)
        )

        # Get first transition for state and action
        first_state, first_action = n_step_buffer[0][0], n_step_buffer[0][1]

        # Add to main replay buffer with n-step return
        super().add(
            first_state,
            first_action,
            n_step_return,
            n_step_next_state,
            n_step_done,
            priority,
            bootstrap_steps=bootstrap_steps,
            next_action_mask=next_action_mask,
            stream_id=stream_id,
        )

        # If episode ended, flush remaining PARTIAL transitions
        if done:
            # The full n-step transition was already added above.
            # Remove it from the buffer before flushing to avoid duplicate.
            n_step_buffer.popleft()
            self._flush_buffer(n_step_buffer, stream_id=stream_id)
            if stream_id is not None:
                self._stream_buffers.pop(stream_id, None)

    def _compute_n_step_return(
        self, n_step_buffer: Optional[Deque] = None
    ) -> Tuple[float, torch.Tensor, bool, int, Optional[torch.Tensor]]:
        """
        Compute n-step return from buffer.

        Returns:
            Tuple of (n_step_return, n_step_next_state, n_step_done, bootstrap_steps,
            next_action_mask)
        """
        if n_step_buffer is None:
            n_step_buffer = self.n_step_buffer

        n_step_return = 0.0
        n_step_next_state = None
        n_step_done = False
        bootstrap_steps = 0
        n_step_next_action_mask = None

        for i, (state, action, reward, next_state, done, next_action_mask) in enumerate(
            n_step_buffer
        ):
            # Accumulate discounted rewards
            n_step_return += (self.gamma**i) * reward
            bootstrap_steps = i + 1

            # Terminal state encountered
            if done:
                n_step_next_state = next_state
                n_step_done = True
                n_step_next_action_mask = next_action_mask
                break

        # If no terminal state, use last next_state
        if not n_step_done:
            n_step_next_state = n_step_buffer[-1][3]
            n_step_done = n_step_buffer[-1][4]
            n_step_next_action_mask = n_step_buffer[-1][5]

        return (
            n_step_return,
            n_step_next_state,
            n_step_done,
            bootstrap_steps,
            n_step_next_action_mask,
        )

    def _flush_buffer(self, n_step_buffer: Optional[Deque] = None, stream_id=None):
        """
        Flush remaining transitions in n-step buffer when episode ends.

        This ensures all transitions are added even if buffer isn't full.
        """
        if n_step_buffer is None:
            n_step_buffer = self.n_step_buffer

        while len(n_step_buffer) > 0:
            # Compute partial n-step return with remaining transitions
            n_step_return, n_step_next_state, n_step_done, bootstrap_steps, next_action_mask = (
                self._compute_n_step_return(n_step_buffer)
            )
            first_state, first_action = n_step_buffer[0][0], n_step_buffer[0][1]

            super().add(
                first_state,
                first_action,
                n_step_return,
                n_step_next_state,
                n_step_done,
                priority=None,
                bootstrap_steps=bootstrap_steps,
                next_action_mask=next_action_mask,
                stream_id=stream_id,
            )

            # Remove first transition
            n_step_buffer.popleft()

    def _flush_episode_boundary(self, n_step_buffer: Deque, stream_id=None) -> None:
        """Flush pending live-episode tail transitions without duplicating emitted ones."""
        if len(n_step_buffer) == self.n_step:
            # The first transition was emitted when the buffer filled on add().
            n_step_buffer.popleft()
        self._flush_buffer(n_step_buffer, stream_id=stream_id)

    def flush_n_step_buffer(self, stream_id: Optional[Hashable] = None) -> None:
        """Flush pending partial n-step transitions for an episode boundary.

        Unlike reset_n_step_buffer(), this preserves short tail transitions when
        an episode is truncated by max-frames rather than ending with done=True.
        """
        if stream_id is None:
            self._flush_episode_boundary(self.n_step_buffer)
            for pending_stream_id, stream_buffer in list(self._stream_buffers.items()):
                self._flush_episode_boundary(stream_buffer, stream_id=pending_stream_id)
            self._stream_buffers.clear()
            return

        stream_buffer = self._stream_buffers.pop(stream_id, None)
        if stream_buffer is not None:
            self._flush_episode_boundary(stream_buffer, stream_id=stream_id)

    def reset_n_step_buffer(self, stream_id: Optional[Hashable] = None):
        """Reset one stream's n-step buffer, or all streams if omitted."""
        if stream_id is None:
            self.n_step_buffer.clear()
            self._stream_buffers.clear()
        else:
            self._stream_buffers.pop(stream_id, None)

    def clear(self) -> None:
        """Clear replay memory and all pending n-step transitions."""
        super().clear()
        self.reset_n_step_buffer()
