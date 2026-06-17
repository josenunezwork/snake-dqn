"""Tests for the legacy OnlineTrainer compatibility path."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.core.game_config import (
    ApexSettings,
    AppConfig,
    GameConfig,
    StateIndices,
    TrainingSettings,
    get_config,
    initialize_config,
)
from src.training.base_buffer import compute_priority
from src.training.online_trainer import OnlineTrainer

pytestmark = pytest.mark.usefixtures("setup_config")


class FixedQNetwork(nn.Module):
    """Tiny Q-network with trainable per-action values."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self.q_values = nn.Parameter(torch.tensor(q_values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.q_values.unsqueeze(0).expand(states.shape[0], -1)


def make_state() -> torch.Tensor:
    """Create a valid 58D state with all normal actions initially safe."""
    state = torch.zeros(GameConfig.INPUT_SIZE, dtype=torch.float32)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    state[danger_start:danger_end] = 0.0
    return state


def make_trapped_state() -> torch.Tensor:
    """Create a valid 58D state with no state-derived normal actions."""
    state = make_state()
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    state[danger_start:danger_end] = 1.0
    return state


def make_trainer() -> OnlineTrainer:
    """Create a small OnlineTrainer for compatibility tests."""
    return OnlineTrainer(GameConfig.INPUT_SIZE, 64, GameConfig.OUTPUT_SIZE)


def test_greedy_action_masks_invalid_boost_q_values():
    """Greedy legacy inference should not choose boost from state-derived masks."""
    trainer = make_trainer()
    trainer.epsilon = 0.0
    trainer.dqn = FixedQNetwork([0.0, 1.0, 2.0, 0.0, 99.0, 0.0])
    state = make_state()

    action = trainer.get_action(state)

    assert action == 2


def test_greedy_action_falls_back_to_best_normal_action_when_trapped():
    """Greedy legacy inference should avoid boost when no state action is valid."""
    trainer = make_trainer()
    trainer.epsilon = 0.0
    trainer.dqn = FixedQNetwork([0.0, 2.0, 1.0, 0.0, 99.0, 0.0])

    action = trainer.get_action(make_trapped_state())

    assert action == 1


def test_random_action_uses_state_mask():
    """Random legacy exploration should sample only known-valid normal actions."""
    trainer = make_trainer()
    trainer.epsilon = 1.0
    state = make_state()
    state[StateIndices.PER_ACTION_DANGER_START] = 1.0
    state[StateIndices.PER_ACTION_DANGER_START + 2] = 1.0

    actions = {trainer.get_action(state) for _ in range(20)}

    assert actions == {1}


def test_update_memory_preserves_exact_next_action_mask():
    """Callers using OnlineTrainer should not lose simulator next-action masks."""
    trainer = make_trainer()
    state = make_state()
    next_state = make_state()
    next_action_mask = torch.tensor([False, True, False, False, True, False])

    trainer.update_memory(
        state,
        action=1,
        reward=0.5,
        next_state=next_state,
        done=False,
        next_action_mask=next_action_mask,
    )

    memories = trainer.get_all_memories()

    assert len(memories) == 1
    assert len(memories[0]) == 8
    assert torch.equal(memories[0][7], next_action_mask)


def test_train_batch_size_one_uses_apex_gamma_without_scalar_shape_bug():
    """A one-row update should use Apex gamma and keep TD errors vector-shaped."""
    original_config = get_config()
    config = AppConfig(
        training=TrainingSettings(batch_size=1, memory_size=1000, gamma=0.99),
        apex=ApexSettings(min_buffer_size=1, learning_rate=0.001, gamma=0.5),
    )

    try:
        initialize_config(config)
        trainer = make_trainer()
        trainer.dqn = FixedQNetwork([0.0, 4.0, 1.0, 9.0, 9.0, 9.0])
        trainer.target_dqn = FixedQNetwork([0.0, 4.0, 1.0, 9.0, 9.0, 9.0])
        trainer.optimizer = optim.AdamW(trainer.dqn.parameters(), lr=GameConfig.APEX_LEARNING_RATE)
        trainer.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode="max",
            factor=0.5,
            patience=1000,
        )

        state = make_state()
        next_state = make_state()
        next_action_mask = torch.tensor([False, True, False, False, False, False])
        trainer.memory.add(
            state,
            0,
            1.0,
            next_state,
            False,
            priority=1.0,
            next_action_mask=next_action_mask,
        )

        loss, epsilon = trainer.train()
        priority = trainer.get_all_memories()[0][5]

        assert loss is not None
        assert epsilon < GameConfig.EPSILON_START
        assert trainer._last_train_metrics["valid_next_action_fraction"] == pytest.approx(1.0)
        assert trainer._last_train_metrics["trapped_next_state_fraction"] == pytest.approx(0.0)
        assert trainer._last_train_metrics["exact_next_action_mask_fraction"] == pytest.approx(1.0)
        assert priority == pytest.approx(
            compute_priority(3.0, GameConfig.PRIORITY_ALPHA, trainer.memory.priority_eps)
        )
    finally:
        initialize_config(original_config)
