"""Public replay/target transport checks for persisted action-mask modes."""

import numpy as np
import pytest
import torch

from src.training.apex_buffer import LocalApexBuffer
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.td_targets import (
    MASK_MODE_LEGACY_ADVISORY,
    MASK_MODE_RASTER_RESOLVED_V3,
    MASK_MODE_TERMINAL_NO_SUCCESSOR,
    resolve_bootstrap_action_masks,
)


def _state(boost: float) -> np.ndarray:
    value = np.zeros(58, dtype=np.float32)
    value[54:57] = 0.0
    value[57] = boost
    return value


def test_mixed_modes_survive_local_replay_and_resolve_rowwise():
    buffer = LocalApexBuffer(capacity=8, state_size=58)
    # Legacy advice asks only for boost; unavailable boost falls back to legal normal actions.
    buffer.add(
        _state(0),
        0,
        1.0,
        _state(0),
        False,
        next_action_mask=[0, 0, 0, 1, 1, 1],
        next_action_mask_mode=MASK_MODE_LEGACY_ADVISORY,
    )
    # A resolved mode stays an actual six-wide successor fact.
    buffer.add(
        _state(1),
        1,
        2.0,
        _state(1),
        False,
        next_action_mask=[0, 0, 0, 1, 0, 0],
        next_action_mask_mode=MASK_MODE_RASTER_RESOLVED_V3,
    )
    batch, _, _ = buffer.sample(2, torch.device("cpu"))
    assert set(batch["next_action_mask_modes"].tolist()) == {0, 1}
    resolved = resolve_bootstrap_action_masks(
        batch["next_states"], batch["next_action_masks"], batch["next_action_mask_modes"]
    )
    legacy = batch["next_action_mask_modes"] == MASK_MODE_LEGACY_ADVISORY
    assert bool(resolved[legacy, :3].any())


def test_unknown_mode_and_bad_width_fail_closed():
    with pytest.raises(ValueError, match="unknown"):
        resolve_bootstrap_action_masks(
            torch.zeros(1, 58), torch.ones(1, 6, dtype=torch.bool), torch.tensor([99])
        )
    with pytest.raises(ValueError, match="shape"):
        resolve_bootstrap_action_masks(
            torch.zeros(1, 58), torch.ones(1, 5, dtype=torch.bool), torch.tensor([0])
        )


def test_local_buffer_rejects_impossible_live_resolved_empty_mask():
    buffer = LocalApexBuffer(capacity=8, state_size=58)
    with pytest.raises(ValueError, match="nonterminal resolved"):
        buffer.add(
            _state(1),
            0,
            0.0,
            _state(1),
            False,
            next_action_mask=[False] * 6,
            next_action_mask_mode=MASK_MODE_RASTER_RESOLVED_V3,
        )


def test_local_buffer_rejects_terminal_resolved_mode_before_target_sampling():
    buffer = LocalApexBuffer(capacity=8, state_size=58)
    with pytest.raises(ValueError, match="explicit nonterminal"):
        buffer.add(
            _state(1),
            0,
            0.0,
            _state(1),
            True,
            next_action_mask=[True, False, False, False, False, False],
            next_action_mask_mode=MASK_MODE_RASTER_RESOLVED_V3,
        )


def test_local_buffer_rejects_terminal_mode_with_mask():
    buffer = LocalApexBuffer(capacity=8, state_size=58)
    with pytest.raises(ValueError, match="requires no next_action_mask"):
        buffer.add(
            _state(1),
            0,
            0.0,
            _state(1),
            True,
            next_action_mask=[False] * 6,
            next_action_mask_mode=2,
        )


def test_local_replay_terminal_only_sample_keeps_explicit_mode():
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.add(
        torch.from_numpy(_state(0)),
        0,
        0.0,
        torch.from_numpy(_state(0)),
        True,
        next_action_mask_mode=MASK_MODE_TERMINAL_NO_SUCCESSOR,
    )

    batch, _, _ = replay.sample(1, torch.device("cpu"))

    assert batch["next_action_mask_modes"].tolist() == [MASK_MODE_TERMINAL_NO_SUCCESSOR]
    assert "next_action_masks" in batch


@pytest.mark.parametrize("invalid_steps", [0, -1, 1.5, True])
def test_local_replay_rejects_lossy_bootstrap_steps_on_add(invalid_steps):
    replay = PrioritizedReplayBuffer(capacity=8)
    with pytest.raises(ValueError, match="bootstrap_steps must be an integer >= 1"):
        replay.add(
            torch.from_numpy(_state(0)),
            0,
            0.0,
            torch.from_numpy(_state(0)),
            True,
            bootstrap_steps=invalid_steps,
        )


@pytest.mark.parametrize("invalid_steps", [0, -1, 1.5, True])
def test_local_replay_rejects_lossy_bootstrap_steps_on_bulk_add(invalid_steps):
    replay = PrioritizedReplayBuffer(capacity=8)
    with pytest.raises(ValueError, match="bootstrap_steps must be an integer >= 1"):
        replay.add_bulk(
            [torch.from_numpy(_state(0))],
            [0],
            [0.0],
            [torch.from_numpy(_state(0))],
            [True],
            [1.0],
            bootstrap_steps=[invalid_steps],
        )
