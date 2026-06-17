"""Tests for local Apex policy training behavior."""

import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import (
    ApexSettings,
    AppConfig,
    NetworkSettings,
    RewardSettings,
    StateIndices,
    TrainingSettings,
    initialize_config,
)
from src.core.reward_contract import current_reward_contract
from src.training.apex_policy import ApexPolicy


class ConstantQ(torch.nn.Module):
    """Tiny deterministic Q-network for target calculation tests."""

    def __init__(self, value: float, output_size: int):
        super().__init__()
        self.value = value
        self.output_size = output_size

    def forward(self, states):
        return torch.full(
            (states.shape[0], self.output_size),
            self.value,
            dtype=torch.float32,
            device=states.device,
        )


class FixedQ(torch.nn.Module):
    """Return the same deterministic action values for every state."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.to(states.device).expand(states.shape[0], -1)


def _full_state_batch(batch_size: int = 1) -> torch.Tensor:
    """Create 58D next-state rows with every normal direction initially safe."""
    states = torch.zeros((batch_size, 58), dtype=torch.float32)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    states[:, danger_start:danger_end] = 0.0
    states[:, StateIndices.BOOST_AVAILABLE] = 0.0
    return states


def _trapped_state_batch(batch_size: int = 1) -> torch.Tensor:
    """Create 58D next-state rows with every normal direction immediately unsafe."""
    states = _full_state_batch(batch_size)
    danger_start = StateIndices.PER_ACTION_DANGER_START
    danger_end = StateIndices.PER_ACTION_DANGER_END
    states[:, danger_start:danger_end] = 1.0
    return states


def test_local_policy_uses_small_replay_warmup():
    """Local training should not inherit the large distributed Ape-X warmup."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(batch_size=2, min_buffer_size=50, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        assert policy._min_replay_size() == 8

        loss = None
        for i in range(8):
            state = torch.full((4,), float(i))
            next_state = torch.full((4,), float(i + 1))
            loss, _ = policy.update(
                state=state,
                action=i % 3,
                reward=0.1,
                next_state=next_state,
                done=False,
                snake_id=0,
            )

        assert loss is not None
        assert policy.update_counter > 0
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_loss_uses_per_sample_bootstrap_steps():
    """N-step and partial-tail replay rows should not share one fixed discount."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.5),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=3)
        policy.dqn = ConstantQ(0.0, output_size=3)
        policy.target_dqn = ConstantQ(1.0, output_size=3)

        _, td_errors = policy._compute_double_dqn_loss(
            states=torch.zeros((2, 4)),
            actions=torch.tensor([0, 0]),
            rewards=torch.zeros(2),
            next_states=torch.ones((2, 4)),
            dones=torch.zeros(2),
            weights=torch.ones(2),
            bootstrap_steps=torch.tensor([1.0, 3.0]),
        )

        assert td_errors.tolist() == [0.5, 0.125]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_loss_masks_invalid_next_actions():
    """Double DQN targets should not bootstrap from actions the actor cannot take."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001, gamma=0.5),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        policy.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])

        _, td_errors = policy._compute_double_dqn_loss(
            states=torch.zeros((1, 58)),
            actions=torch.tensor([0]),
            rewards=torch.zeros(1),
            next_states=_full_state_batch(),
            dones=torch.zeros(1),
            weights=torch.ones(1),
            bootstrap_steps=torch.tensor([1.0]),
        )

        assert td_errors.tolist() == [2.5]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_loss_does_not_bootstrap_from_trapped_next_state():
    """A next state with no valid actions should use a reward-only target."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001, gamma=0.5),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        policy.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])

        _, td_errors = policy._compute_double_dqn_loss(
            states=torch.zeros((1, 58)),
            actions=torch.tensor([0]),
            rewards=torch.tensor([1.25]),
            next_states=_trapped_state_batch(),
            dones=torch.zeros(1),
            weights=torch.ones(1),
            bootstrap_steps=torch.tensor([1.0]),
        )

        assert td_errors.tolist() == [1.25]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_greedy_masks_invalid_state_actions():
    """Direct greedy policy inference should follow the same state-derived mask as targets."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(batch_size=1, min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 0.0
        policy.dqn = FixedQ([1.0, 100.0, 2.0, 30.0, 40.0, 50.0])
        state = _full_state_batch().squeeze(0)
        danger_start = StateIndices.PER_ACTION_DANGER_START
        danger_end = StateIndices.PER_ACTION_DANGER_END
        state[danger_start:danger_end] = torch.tensor([0.0, 1.0, 0.0])

        assert policy.select_action(state) == 2
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_greedy_falls_back_to_best_normal_action_when_trapped():
    """Direct greedy inference should avoid boost when state masks have no valid actions."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(batch_size=1, min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 0.0
        policy.dqn = FixedQ([1.0, 100.0, 2.0, 30.0, 1_000.0, 50.0])
        state = _trapped_state_batch().squeeze(0)

        assert policy.select_action(state) == 1
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_random_uses_state_valid_action_mask():
    """Direct random policy inference should not explore known-invalid actions."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 1.0
        state = _full_state_batch().squeeze(0)
        danger_start = StateIndices.PER_ACTION_DANGER_START
        danger_end = StateIndices.PER_ACTION_DANGER_END
        state[danger_start:danger_end] = torch.tensor([1.0, 1.0, 0.0])

        actions = {policy.select_action(state) for _ in range(10)}

        assert actions == {2}
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_greedy_prefers_exact_action_mask_for_safe_boost():
    """Exact simulator masks should let direct greedy selection choose safe boost actions."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 0.0
        policy.dqn = FixedQ([1.0, 2.0, 3.0, 4.0, 100.0, 6.0])
        state = _full_state_batch().squeeze(0)
        state[StateIndices.BOOST_AVAILABLE] = 1.0
        exact_mask = torch.tensor([False, False, False, False, True, False])

        assert policy.select_action(state, action_mask=exact_mask) == 4
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_random_prefers_exact_action_mask_for_safe_boost():
    """Exact simulator masks should let direct random selection explore safe boost actions."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 1.0
        state = _full_state_batch().squeeze(0)
        state[StateIndices.BOOST_AVAILABLE] = 1.0
        exact_mask = torch.tensor([False, False, False, False, True, False])

        actions = {policy.select_action(state, action_mask=exact_mask) for _ in range(10)}

        assert actions == {4}
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_rejects_non_binary_exact_action_mask():
    """Direct policy selection should not coerce malformed exact masks to valid actions."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 0.0
        policy.dqn = FixedQ([1.0, 2.0, 3.0, 4.0, 100.0, 6.0])
        state = _full_state_batch().squeeze(0)
        exact_mask = torch.tensor([0, 1, 0, 0, 2, 0])

        with pytest.raises(ValueError, match="action_mask values must be 0/1 or bool"):
            policy.select_action(state, action_mask=exact_mask)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_select_action_rejects_wrong_shaped_exact_action_mask():
    """Direct policy selection should not ignore malformed exact-mask shapes."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.epsilon = 0.0
        policy.dqn = FixedQ([1.0, 2.0, 3.0, 4.0, 100.0, 6.0])
        state = _full_state_batch().squeeze(0)
        wrong_shape_mask = torch.tensor([True, False, True])

        with pytest.raises(ValueError, match="action_mask shape must match q_values shape"):
            policy.select_action(state, action_mask=wrong_shape_mask)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_loss_prefers_exact_replay_action_mask():
    """Exact replay masks should override approximate masks inferred from state features."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001, gamma=0.5),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        policy.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        next_states = _full_state_batch()
        next_states[:, StateIndices.BOOST_AVAILABLE] = 1.0
        exact_masks = torch.tensor([[False, True, False, False, False, False]])

        _, td_errors = policy._compute_double_dqn_loss(
            states=torch.zeros((1, 58)),
            actions=torch.tensor([0]),
            rewards=torch.zeros(1),
            next_states=next_states,
            dones=torch.zeros(1),
            weights=torch.ones(1),
            bootstrap_steps=torch.tensor([1.0]),
            next_action_masks=exact_masks,
        )

        assert td_errors.tolist() == [2.5]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_policy_update_preserves_exact_next_action_mask():
    """Direct ApexPolicy.update callers should not drop simulator action masks."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=50, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        mask = torch.tensor([False, True, False, False, False, False])

        policy.update(
            state=torch.zeros(58),
            action=1,
            reward=0.5,
            next_state=torch.ones(58),
            done=False,
            snake_id=0,
            next_action_mask=mask,
        )

        stored = policy.memory.get_all_memories()
        assert len(stored) == 1
        assert len(stored[0]) == 9
        assert torch.equal(stored[0][7], mask)
        assert stored[0][8] == 0
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_local_train_step_records_target_action_quality_metrics():
    """Local/headless policy updates should expose replay target-action health."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(batch_size=1, min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        exact_mask = torch.tensor([False, True, False, False, False, False])

        loss, _epsilon = policy.update(
            state=torch.zeros(58),
            action=1,
            reward=0.5,
            next_state=torch.ones(58),
            done=False,
            snake_id=0,
            next_action_mask=exact_mask,
        )

        assert loss is not None
        assert policy._last_train_metrics["valid_next_action_fraction"] == pytest.approx(1.0)
        assert policy._last_train_metrics["trapped_next_state_fraction"] == pytest.approx(0.0)
        assert policy._last_train_metrics["exact_next_action_mask_fraction"] == pytest.approx(1.0)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_gru_train_step_records_target_action_quality_metrics():
    """DRQN metrics should ignore padded sequence slots after burn-in."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(
                input_size=58,
                hidden_size=32,
                output_size=6,
                use_gru=True,
                gru_hidden_size=16,
                sequence_length=5,
                burn_in_length=1,
            ),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(batch_size=1, min_buffer_size=1, learning_rate=0.001),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=32, output_size=6, use_gru=True, n_step=1)
        exact_trapped_mask = torch.zeros(6, dtype=torch.bool)
        terminal_valid_mask = torch.tensor([True, False, False, False, False, False])
        loss = None

        for step in range(3):
            state = torch.full((58,), float(step))
            next_state = torch.full((58,), float(step + 1))
            next_action_mask = terminal_valid_mask if step == 2 else exact_trapped_mask
            loss, _epsilon = policy.update(
                state=state,
                action=0,
                reward=0.0,
                next_state=next_state,
                done=step == 2,
                snake_id=0,
                next_action_mask=next_action_mask,
            )

        assert loss is not None
        assert policy._last_train_metrics["valid_next_action_fraction"] == pytest.approx(0.0)
        assert policy._last_train_metrics["trapped_next_state_fraction"] == pytest.approx(1.0)
        assert policy._last_train_metrics["exact_next_action_mask_fraction"] == pytest.approx(1.0)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_get_priorities_prefers_exact_replay_action_mask():
    """Policy-side priority estimates should use exact masks when replay has them."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=58, hidden_size=64, output_size=6),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(
                min_buffer_size=1,
                learning_rate=0.001,
                gamma=0.5,
                priority_alpha=1.0,
                priority_epsilon=0.0,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=58, hidden_size=64, output_size=6, n_step=1)
        policy.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        policy.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        next_state = _full_state_batch().squeeze(0)
        next_state[StateIndices.BOOST_AVAILABLE] = 1.0
        exact_mask = torch.tensor([False, True, False, False, False, False])

        priorities = policy.get_priorities(
            [
                (
                    torch.zeros(58),
                    0,
                    0.0,
                    next_state,
                    False,
                    1.0,
                    1,
                    exact_mask,
                )
            ]
        )

        assert priorities.tolist() == [pytest.approx(2.5)]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_local_policy_replay_uses_apex_priority_settings():
    """ApexPolicy replay should follow Apex PER config, not generic training alpha."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000, priority_alpha=1.0),
            apex=ApexSettings(
                min_buffer_size=2,
                learning_rate=0.001,
                priority_alpha=0.5,
                priority_beta_start=0.2,
                priority_beta_end=0.7,
                priority_epsilon=0.02,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)

        assert policy.memory.alpha == pytest.approx(0.5)
        assert policy.memory.beta == pytest.approx(0.2)
        assert policy.memory.beta_end == pytest.approx(0.7)
        assert policy.memory.priority_eps == pytest.approx(0.02)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_gru_replay_uses_apex_priority_settings():
    """GRU Apex replay should follow Apex PER config, not sequence defaults."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(
                batch_size=2,
                buffer_size=32,
                min_buffer_size=2,
                learning_rate=0.001,
                priority_alpha=0.5,
                priority_beta_start=0.2,
                priority_beta_end=0.7,
                priority_epsilon=0.03,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, use_gru=True)

        assert policy.memory.alpha == pytest.approx(0.5)
        assert policy.memory.beta_start == pytest.approx(0.2)
        assert policy.memory.beta_end == pytest.approx(0.7)
        assert policy.memory.priority_eps == pytest.approx(0.03)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_checkpoint_records_effective_apex_config_snapshot():
    """Apex checkpoints should preserve the replay/training contract used for learning."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(
                batch_size=1,
                memory_size=1000,
                priority_beta_increment=0.123,
            ),
            rewards=RewardSettings(death=-40.0, food_base=2.0),
            apex=ApexSettings(
                num_actors=3,
                batch_size=2,
                buffer_size=32,
                actor_update_freq=17,
                min_buffer_size=20,
                learning_rate=0.001,
                gamma=0.5,
                n_step=5,
                target_update_freq=11,
                priority_alpha=0.55,
                priority_beta_start=0.25,
                priority_beta_end=0.75,
                priority_epsilon=0.04,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=4)

        state_dict = policy.get_state_dict()
        apex_config = state_dict["apex_config"]
        reward_contract = current_reward_contract()

        assert state_dict["reward_contract"] == reward_contract
        assert state_dict["reward_death"] == pytest.approx(-40.0)
        assert state_dict["reward_food_base"] == pytest.approx(2.0)
        assert apex_config == {
            "actor_update_freq": 17,
            "batch_size": 2,
            "buffer_size": 32,
            "distributed": False,
            "gamma": 0.5,
            "learning_rate": 0.001,
            "min_replay_size": 8,
            "n_step": 4,
            "num_actors": 3,
            "priority_alpha": 0.55,
            "priority_beta_current": 0.25,
            "priority_beta_end": 0.75,
            "priority_beta_increment": 0.123,
            "priority_beta_start": 0.25,
            "priority_epsilon": 0.04,
            "reward_contract": reward_contract,
            "reward_death": -40.0,
            "reward_food_base": 2.0,
            "target_update_freq": 11,
            "use_gru": False,
        }
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_gru_checkpoint_records_sequence_replay_config():
    """GRU Apex checkpoints should include sequence replay dimensions."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(
                input_size=4,
                hidden_size=64,
                output_size=3,
                sequence_length=12,
                burn_in_length=3,
                gru_hidden_size=48,
            ),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(
                batch_size=2,
                buffer_size=32,
                min_buffer_size=2,
                learning_rate=0.001,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, use_gru=True)

        apex_config = policy.get_state_dict()["apex_config"]

        assert apex_config["use_gru"] is True
        assert apex_config["sequence_length"] == 12
        assert apex_config["burn_in_length"] == 3
        assert apex_config["gru_hidden_size"] == 48
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_local_policy_replay_uses_apex_gamma():
    """Local/offline Apex should use Apex gamma even if generic training gamma differs."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000, gamma=0.9),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.5),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=3)

        assert policy.gamma == pytest.approx(0.5)
        assert policy.memory.gamma == pytest.approx(0.5)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_replay_capacity_uses_apex_buffer_size():
    """Local/offline feedforward Apex should honor apex.buffer_size."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(
                batch_size=2,
                buffer_size=32,
                min_buffer_size=2,
                learning_rate=0.001,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)

        assert policy.memory.capacity == 32
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_gru_replay_capacity_uses_apex_buffer_size():
    """Local/offline GRU Apex should honor apex.buffer_size."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(
                batch_size=2,
                buffer_size=32,
                min_buffer_size=2,
                learning_rate=0.001,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, use_gru=True)

        assert policy.memory.capacity == 32
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_train_step_uses_apex_batch_size(monkeypatch):
    """Local feedforward Apex sampling should ignore generic training.batch_size."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(
                batch_size=2,
                buffer_size=32,
                min_buffer_size=2,
                learning_rate=0.001,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        sampled_batch_sizes = []
        original_sample = policy.memory.sample

        def sample_with_recording(batch_size, device):
            sampled_batch_sizes.append(batch_size)
            return original_sample(batch_size, device)

        monkeypatch.setattr(policy.memory, "sample", sample_with_recording)

        for i in range(2):
            policy.update(
                state=torch.full((4,), float(i)),
                action=i % 3,
                reward=0.1,
                next_state=torch.full((4,), float(i + 1)),
                done=False,
                snake_id=0,
            )

        assert sampled_batch_sizes == [2]
        assert policy.update_counter == 1
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_gru_train_step_uses_apex_batch_size(monkeypatch):
    """Local GRU Apex readiness checks should ignore generic training.batch_size."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(
                batch_size=2,
                buffer_size=32,
                min_buffer_size=2,
                learning_rate=0.001,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, use_gru=True)
        checked_batch_sizes = []

        def is_ready_with_recording(batch_size):
            checked_batch_sizes.append(batch_size)
            return False

        monkeypatch.setattr(policy.memory, "is_ready", is_ready_with_recording)

        loss, _epsilon = policy.train_step()

        assert loss is None
        assert checked_batch_sizes == [2]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_target_sync_uses_apex_frequency(monkeypatch):
    """Local feedforward Apex target sync should follow apex.target_update_freq."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(
                batch_size=2,
                memory_size=1000,
                target_update_frequency=999,
            ),
            apex=ApexSettings(
                batch_size=2,
                min_buffer_size=2,
                learning_rate=0.001,
                target_update_freq=1,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        sync_calls = []
        monkeypatch.setattr(
            "src.training.apex_policy.hard_update",
            lambda target, source: sync_calls.append((target, source)),
        )

        for i in range(2):
            policy.update(
                state=torch.full((4,), float(i)),
                action=i % 3,
                reward=0.1,
                next_state=torch.full((4,), float(i + 1)),
                done=False,
                snake_id=0,
            )

        assert policy.update_counter == 1
        assert len(sync_calls) == 1
        assert sync_calls[0] == (policy.target_dqn, policy.dqn)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_gru_target_sync_helper_uses_apex_frequency(monkeypatch):
    """Shared Apex target sync helper should ignore generic training frequency."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(
                batch_size=2,
                memory_size=1000,
                target_update_frequency=999,
            ),
            apex=ApexSettings(
                min_buffer_size=2,
                learning_rate=0.001,
                target_update_freq=2,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, use_gru=True)
        sync_calls = []
        monkeypatch.setattr(
            "src.training.apex_policy.hard_update",
            lambda target, source: sync_calls.append((target, source)),
        )

        policy.update_counter = 1
        policy._maybe_sync_target_network()
        assert sync_calls == []

        policy.update_counter = 2
        policy._maybe_sync_target_network()
        assert sync_calls == [(policy.target_dqn, policy.dqn)]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_policy_default_n_step_uses_apex_config():
    """ApexPolicy should default to the configured Apex n-step horizon."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, n_step=5),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3)

        assert policy.n_step == 5
        assert policy.memory.n_step == 5
        assert policy._local_buffer.maxlen == 5
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_state_dict_syncs_feedforward_replay_hyperparameters():
    """Checkpoint n-step/gamma metadata should update the live local replay wrapper."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        state_dict = policy.get_state_dict()

        assert state_dict["input_size"] == 4
        assert state_dict["hidden_size"] == 64

        policy.memory.n_step_buffer.append((torch.zeros(4), 0, 1.0, torch.ones(4), False))
        state_dict["n_step"] = 4
        state_dict["gamma"] = 0.5
        state_dict["apex_config"]["n_step"] = 4
        state_dict["apex_config"]["gamma"] = 0.5

        policy.load_state_dict(state_dict)

        assert policy.n_step == 4
        assert policy.gamma == pytest.approx(0.5)
        assert policy.memory.n_step == 4
        assert policy.memory.gamma == pytest.approx(0.5)
        assert policy.memory.n_step_buffer.maxlen == 4
        assert len(policy.memory.n_step_buffer) == 0
        assert policy._local_buffer.maxlen == 4
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_state_dict_uses_nested_apex_contract_when_top_level_metadata_missing():
    """Distributed-style checkpoints should still restore target semantics into local replay."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        state_dict = policy.get_state_dict()
        state_dict.pop("n_step")
        state_dict.pop("gamma")
        state_dict["apex_config"]["n_step"] = 4
        state_dict["apex_config"]["gamma"] = 0.5

        policy.load_state_dict(state_dict)

        assert policy.n_step == 4
        assert policy.gamma == pytest.approx(0.5)
        assert policy.memory.n_step == 4
        assert policy.memory.gamma == pytest.approx(0.5)
        assert policy.memory.n_step_buffer.maxlen == 4
        assert policy._local_buffer.maxlen == 4
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_state_dict_rejects_inconsistent_checkpoint_contract_before_replay_mutation():
    """A checkpoint with conflicting target semantics should fail before replay is rescaled."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        state_dict = policy.get_state_dict()
        policy.memory.n_step_buffer.append((torch.zeros(4), 0, 1.0, torch.ones(4), False))
        state_dict["n_step"] = 4
        state_dict["gamma"] = 0.5

        with pytest.raises(ValueError, match="n_step=1.*n_step=4"):
            policy.load_state_dict(state_dict)

        assert policy.n_step == 1
        assert policy.gamma == pytest.approx(0.9)
        assert policy.memory.n_step == 1
        assert policy.memory.gamma == pytest.approx(0.9)
        assert policy.memory.n_step_buffer.maxlen == 1
        assert len(policy.memory.n_step_buffer) == 1
        assert policy._local_buffer.maxlen == 1
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_state_dict_rejects_reward_contract_mismatch_before_replay_mutation():
    """A checkpoint trained with a different reward scale should not resume silently."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        state_dict = policy.get_state_dict()
        policy.memory.n_step_buffer.append((torch.zeros(4), 0, 1.0, torch.ones(4), False))
        stale_contract = dict(state_dict["apex_config"]["reward_contract"])
        stale_contract["survival"] = float(stale_contract["survival"]) + 1.0
        state_dict["apex_config"]["reward_contract"] = stale_contract

        with pytest.raises(ValueError, match="reward_contract.survival"):
            policy.load_state_dict(state_dict)

        assert policy.n_step == 1
        assert policy.gamma == pytest.approx(0.9)
        assert policy.memory.n_step == 1
        assert len(policy.memory.n_step_buffer) == 1
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_checkpoint_uses_nested_apex_contract_when_top_level_metadata_missing(tmp_path):
    """File-based policy loading should keep distributed checkpoint target semantics."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        writer = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        checkpoint = writer.get_state_dict()
        checkpoint.pop("n_step")
        checkpoint.pop("gamma")
        checkpoint["apex_config"]["n_step"] = 4
        checkpoint["apex_config"]["gamma"] = 0.5
        checkpoint_path = tmp_path / "nested_contract.pth"
        torch.save(checkpoint, checkpoint_path)

        reader = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)

        assert reader.load_checkpoint(str(checkpoint_path)) is True
        assert reader.n_step == 4
        assert reader.gamma == pytest.approx(0.5)
        assert reader.memory.n_step == 4
        assert reader.memory.gamma == pytest.approx(0.5)
        assert reader._local_buffer.maxlen == 4
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_checkpoint_rejects_inconsistent_contract_before_replay_mutation(tmp_path):
    """File-based policy loading should not bypass checkpoint contract validation."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        writer = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        checkpoint = writer.get_state_dict()
        checkpoint["n_step"] = 4
        checkpoint["gamma"] = 0.5
        checkpoint_path = tmp_path / "conflicting_contract.pth"
        torch.save(checkpoint, checkpoint_path)

        reader = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        reader.memory.n_step_buffer.append((torch.zeros(4), 0, 1.0, torch.ones(4), False))

        assert reader.load_checkpoint(str(checkpoint_path)) is False
        assert reader.n_step == 1
        assert reader.gamma == pytest.approx(0.9)
        assert reader.memory.n_step == 1
        assert reader.memory.gamma == pytest.approx(0.9)
        assert reader.memory.n_step_buffer.maxlen == 1
        assert len(reader.memory.n_step_buffer) == 1
        assert reader._local_buffer.maxlen == 1
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_load_checkpoint_without_epsilon_preserves_inference_epsilon(tmp_path):
    """Legacy inference checkpoints should not force full exploration when epsilon is absent."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        writer = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        checkpoint = writer.get_state_dict()
        checkpoint.pop("epsilon")
        checkpoint_path = tmp_path / "legacy_no_epsilon.pth"
        torch.save(checkpoint, checkpoint_path)

        reader = ApexPolicy(
            input_size=4,
            hidden_size=64,
            output_size=3,
            n_step=1,
            training=False,
            inference_epsilon=0.05,
        )

        assert reader.load_checkpoint(str(checkpoint_path)) is True
        assert reader.epsilon == pytest.approx(0.05)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_constructor_checkpoint_path_raises_when_load_fails(tmp_path):
    """Explicit checkpoint paths should not leave a random policy behind after load failure."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(min_buffer_size=2, learning_rate=0.001, gamma=0.9),
        )
    )

    try:
        writer = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=1)
        checkpoint = writer.get_state_dict()
        checkpoint["n_step"] = 4
        checkpoint["gamma"] = 0.5
        checkpoint_path = tmp_path / "bad_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(RuntimeError, match="Failed to load checkpoint"):
            ApexPolicy(
                input_size=4,
                hidden_size=64,
                output_size=3,
                n_step=1,
                checkpoint_path=str(checkpoint_path),
            )
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_get_priorities_returns_per_scaled_priorities():
    """Actor-side priority helper should return tree priorities, not raw TD errors."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=2, memory_size=1000),
            apex=ApexSettings(
                min_buffer_size=2,
                learning_rate=0.001,
                gamma=0.5,
                priority_alpha=0.5,
                priority_epsilon=0.0,
            ),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=3)
        policy.dqn = ConstantQ(0.0, output_size=3)
        policy.target_dqn = ConstantQ(1.0, output_size=3)
        experiences = [
            (
                torch.zeros(4),
                0,
                0.0,
                torch.ones(4),
                False,
                1.0,
                2,
                None,
                "snake-0",
            )
        ]

        priorities = policy.get_priorities(experiences)

        assert priorities.shape == (1,)
        assert priorities[0] == pytest.approx(0.5)
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()


def test_feedforward_loss_preserves_single_sample_batch_shape():
    """A batch of one should still produce one TD error, not a scalar side effect."""
    DeviceManager.override_device(torch.device("cpu"))
    initialize_config(
        AppConfig(
            network=NetworkSettings(input_size=4, hidden_size=64, output_size=3),
            training=TrainingSettings(batch_size=1, memory_size=1000),
            apex=ApexSettings(min_buffer_size=1, learning_rate=0.001, gamma=0.5),
        )
    )

    try:
        policy = ApexPolicy(input_size=4, hidden_size=64, output_size=3, n_step=3)
        policy.dqn = ConstantQ(0.0, output_size=3)
        policy.target_dqn = ConstantQ(1.0, output_size=3)

        loss, td_errors = policy._compute_double_dqn_loss(
            states=torch.zeros((1, 4)),
            actions=torch.tensor([0]),
            rewards=torch.zeros(1),
            next_states=torch.ones((1, 4)),
            dones=torch.zeros(1),
            weights=torch.ones(1),
            bootstrap_steps=torch.tensor([1.0]),
        )

        assert loss.dim() == 0
        assert td_errors.shape == (1,)
        assert td_errors.tolist() == [0.5]
    finally:
        DeviceManager.reset_for_testing()
        initialize_config()
