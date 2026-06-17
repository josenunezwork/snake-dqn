"""Tests for Ape-X priority target calculations."""

import torch

from src.core.game_config import StateIndices
from src.training.apex_priorities import compute_td_error, compute_td_error_double_dqn


def _next_states() -> torch.Tensor:
    states = torch.zeros((1, 58), dtype=torch.float32)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    states[:, danger_start:danger_end] = 0.0
    states[:, StateIndices.BOOST_AVAILABLE] = 0.0
    return states


def _trapped_next_states() -> torch.Tensor:
    states = _next_states()
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    states[:, danger_start:danger_end] = 1.0
    return states


def test_standard_td_error_masks_invalid_next_actions():
    """Actor priority estimates should not max over invalid boost actions."""
    td_error = compute_td_error(
        q_values=torch.zeros((1, 6)),
        actions=torch.tensor([0]),
        rewards=torch.zeros(1),
        next_q_values=torch.tensor([[0.0, 5.0, 0.0, 0.0, 50.0, 0.0]]),
        dones=torch.zeros(1),
        gamma=0.5,
        next_states=_next_states(),
    )

    assert td_error.tolist() == [2.5]


def test_standard_td_error_does_not_bootstrap_from_trapped_next_state():
    """Actor priority estimates should be reward-only when no next action is valid."""
    td_error = compute_td_error(
        q_values=torch.zeros((1, 6)),
        actions=torch.tensor([0]),
        rewards=torch.tensor([1.25]),
        next_q_values=torch.tensor([[0.0, 5.0, 0.0, 0.0, 50.0, 0.0]]),
        dones=torch.zeros(1),
        gamma=0.5,
        next_states=_trapped_next_states(),
    )

    assert td_error.tolist() == [1.25]


def test_double_dqn_td_error_masks_invalid_next_actions():
    """Learner priority estimates should select the best valid online action."""
    td_error = compute_td_error_double_dqn(
        online_q_values=torch.zeros((1, 6)),
        target_q_values=torch.zeros((1, 6)),
        online_next_q_values=torch.tensor([[0.0, 3.0, 0.0, 0.0, 100.0, 0.0]]),
        target_next_q_values=torch.tensor([[0.0, 5.0, 0.0, 0.0, 50.0, 0.0]]),
        actions=torch.tensor([0]),
        rewards=torch.zeros(1),
        dones=torch.zeros(1),
        gamma=0.5,
        next_states=_next_states(),
    )

    assert td_error.tolist() == [2.5]


def test_double_dqn_td_error_does_not_bootstrap_from_trapped_next_state():
    """Learner priority estimates should be reward-only when no next action is valid."""
    td_error = compute_td_error_double_dqn(
        online_q_values=torch.zeros((1, 6)),
        target_q_values=torch.zeros((1, 6)),
        online_next_q_values=torch.tensor([[0.0, 3.0, 0.0, 0.0, 100.0, 0.0]]),
        target_next_q_values=torch.tensor([[0.0, 5.0, 0.0, 0.0, 50.0, 0.0]]),
        actions=torch.tensor([0]),
        rewards=torch.tensor([1.25]),
        dones=torch.zeros(1),
        gamma=0.5,
        next_states=_trapped_next_states(),
    )

    assert td_error.tolist() == [1.25]
