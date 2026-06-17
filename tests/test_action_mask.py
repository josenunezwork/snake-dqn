"""Tests for action-mask helpers used by target-value calculations."""

import pytest
import torch

from src.core.game_config import StateIndices
from src.training.action_mask import (
    has_valid_actions,
    mask_invalid_q_values,
    summarize_next_action_quality,
)


def _full_state_batch(batch_size: int = 1) -> torch.Tensor:
    states = torch.zeros((batch_size, 58), dtype=torch.float32)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    states[:, danger_start:danger_end] = 0.0
    states[:, StateIndices.BOOST_AVAILABLE] = 0.0
    return states


def _trapped_state_batch(batch_size: int = 1) -> torch.Tensor:
    states = _full_state_batch(batch_size)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    states[:, danger_start:danger_end] = 1.0
    return states


def test_summarize_next_action_quality_ignores_masked_samples():
    """Target-action health should only describe rows that can bootstrap."""
    next_states = torch.cat((_full_state_batch(), _trapped_state_batch()), dim=0)
    next_action_masks = torch.tensor(
        [
            [True, False, False, False, False, False],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    metrics = summarize_next_action_quality(
        next_states,
        output_size=6,
        next_action_masks=next_action_masks,
        next_action_mask_present=torch.ones(2),
        sample_mask=torch.tensor([0.0, 1.0]),
    )

    assert metrics["valid_next_action_fraction"] == pytest.approx(0.0)
    assert metrics["trapped_next_state_fraction"] == pytest.approx(1.0)
    assert metrics["exact_next_action_mask_fraction"] == pytest.approx(1.0)


def test_mask_invalid_q_values_allows_legacy_targets_without_state_or_exact_mask():
    """Priority helpers without compact states should keep old all-action behavior."""
    q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])

    masked_q = mask_invalid_q_values(q_values, states=None)
    valid_actions = has_valid_actions(q_values, states=None)

    assert torch.equal(masked_q, q_values)
    assert valid_actions.tolist() == [True]


def test_mask_invalid_q_values_uses_exact_mask_without_state_rows():
    """Exact simulator masks should be sufficient even when compact states are absent."""
    q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])
    action_masks = torch.tensor([[False, True, False, False, False, False]])

    masked_q = mask_invalid_q_values(q_values, states=None, action_masks=action_masks)
    valid_actions = has_valid_actions(q_values, states=None, action_masks=action_masks)

    assert masked_q.argmax(dim=1).tolist() == [1]
    assert masked_q[0, 4].item() < -1.0e8
    assert valid_actions.tolist() == [True]


def test_mask_invalid_q_values_accepts_numeric_zero_one_exact_mask():
    """Stored numeric 0/1 masks should behave like boolean exact masks."""
    q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])
    action_masks = torch.tensor([[0, 1, 0, 0, 0, 0]])

    masked_q = mask_invalid_q_values(q_values, states=None, action_masks=action_masks)

    assert masked_q.argmax(dim=1).tolist() == [1]
    assert masked_q[0, 4].item() < -1.0e8


def test_mask_invalid_q_values_rejects_non_binary_exact_mask():
    """Malformed exact masks should not mark target actions as valid."""
    q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])
    action_masks = torch.tensor([[0, 1, 0, 0, 2, 0]])

    with pytest.raises(ValueError, match="action_mask values must be 0/1 or bool"):
        mask_invalid_q_values(q_values, states=None, action_masks=action_masks)


def test_mask_invalid_q_values_rejects_wrong_shaped_exact_mask():
    """Malformed exact masks should fail instead of falling back to approximate state masks."""
    q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])
    states = _full_state_batch()
    wrong_shape_mask = torch.tensor([[True, True, True]])

    with pytest.raises(ValueError, match="action_mask shape must match q_values shape"):
        mask_invalid_q_values(q_values, states=states, action_masks=wrong_shape_mask)


def test_has_valid_actions_rejects_wrong_shaped_exact_mask():
    """Wrong-shaped exact masks should not hide behind state-derived dead-end detection."""
    q_values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])
    states = _trapped_state_batch()
    wrong_shape_mask = torch.tensor([[True, True, True]])

    with pytest.raises(ValueError, match="action_mask shape must match q_values shape"):
        has_valid_actions(q_values, states=states, action_masks=wrong_shape_mask)
