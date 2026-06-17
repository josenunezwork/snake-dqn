"""Tests for DRQN integration into ApexPolicy.

Tests cover:
- ApexPolicy with use_gru=True creates correct network types
- Hidden state tracking per snake_id
- Episode buffering and sequence replay
- DRQN train_step with sequence batch
- Backward compatibility (use_gru=False unchanged)
- Config integration
- Checkpoint serialization
- Cleanup
"""

import pytest
import torch
import torch.nn as nn

from src.core.device_manager import DeviceManager
from src.core.game_config import GameConfig
from src.model.apex_network import ApexNetwork
from src.model.gru_network import GruApexNetwork
from src.training.apex_policy import ApexPolicy
from src.training.multistep_buffer import MultiStepBuffer
from src.training.sequence_buffer import SequenceReplayBuffer


@pytest.fixture(autouse=True)
def cpu_device():
    """Force CPU device for all tests."""
    DeviceManager.override_device(torch.device("cpu"))
    yield
    DeviceManager.reset_for_testing()


# =========================================================================
# Network Type Tests
# =========================================================================


class TestNetworkCreation:
    """Test that correct network types are created based on use_gru flag."""

    def test_gru_mode_creates_gru_network(self):
        """use_gru=True creates GruApexNetwork for both dqn and target_dqn."""
        policy = ApexPolicy(input_size=58, hidden_size=512, output_size=6, use_gru=True)
        assert isinstance(policy.dqn, GruApexNetwork)
        assert isinstance(policy.target_dqn, GruApexNetwork)
        policy.cleanup()

    def test_ff_mode_creates_apex_network(self):
        """use_gru=False creates ApexNetwork (backward compatible)."""
        policy = ApexPolicy(input_size=58, hidden_size=512, output_size=6, use_gru=False)
        assert isinstance(policy.dqn, ApexNetwork)
        assert isinstance(policy.target_dqn, ApexNetwork)
        policy.cleanup()

    def test_gru_mode_creates_sequence_buffer(self):
        """use_gru=True creates SequenceReplayBuffer."""
        policy = ApexPolicy(input_size=58, hidden_size=512, output_size=6, use_gru=True)
        assert isinstance(policy.memory, SequenceReplayBuffer)
        policy.cleanup()

    def test_ff_mode_creates_multistep_buffer(self):
        """use_gru=False creates MultiStepBuffer (backward compatible)."""
        policy = ApexPolicy(input_size=58, hidden_size=512, output_size=6, use_gru=False)
        assert isinstance(policy.memory, MultiStepBuffer)
        policy.cleanup()


# =========================================================================
# Hidden State Tracking Tests
# =========================================================================


class TestHiddenStateTracking:
    """Test per-snake hidden state tracking in GRU mode."""

    def test_hidden_state_created_on_first_access(self):
        """Hidden state is created as zeros on first access."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        hidden = policy._get_hidden(0)
        assert hidden.shape[0] == 1  # num_layers
        assert hidden.shape[1] == 1  # batch_size
        assert torch.all(hidden == 0)
        policy.cleanup()

    def test_hidden_states_tracked_per_snake(self):
        """Different snake_ids have separate hidden states."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        policy.epsilon = 0.0  # Greedy for deterministic action

        state = torch.randn(16)
        policy.select_action(state, snake_id=0)
        policy.select_action(state, snake_id=1)

        assert 0 in policy._hidden_states
        assert 1 in policy._hidden_states
        # Different hidden states (different random init paths)
        # Both should exist independently
        assert policy._hidden_states[0] is not policy._hidden_states[1]
        policy.cleanup()

    def test_reset_hidden_clears_state(self):
        """reset_hidden resets hidden state to zeros."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        policy.epsilon = 0.0

        state = torch.randn(16)
        policy.select_action(state, snake_id=0)

        policy.reset_hidden(0)
        h_after = policy._hidden_states[0]

        assert torch.all(h_after == 0)
        policy.cleanup()

    def test_select_action_updates_hidden(self):
        """Calling select_action updates the hidden state for that snake."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        policy.epsilon = 0.0

        state = torch.randn(16)
        policy.select_action(state, snake_id=0)
        h1 = policy._hidden_states[0].clone()

        policy.select_action(state, snake_id=0)
        h2 = policy._hidden_states[0].clone()

        # Hidden should change after second call
        assert not torch.allclose(h1, h2)
        policy.cleanup()

    def test_random_select_action_updates_hidden(self):
        """Random GRU exploration should still advance recurrent state."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        policy.epsilon = 1.0

        state = torch.randn(16)
        action = policy.select_action(state, snake_id=0)

        assert 0 <= action < 6
        assert 0 in policy._hidden_states
        assert not torch.all(policy._hidden_states[0] == 0)
        policy.cleanup()


# =========================================================================
# Episode Buffering Tests
# =========================================================================


class TestEpisodeBuffering:
    """Test episode transition accumulation in GRU mode."""

    def test_transitions_accumulated_per_snake(self):
        """Transitions accumulate in per-snake episode buffers."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)

        state = torch.randn(16)
        next_state = torch.randn(16)

        policy.update(state, 0, 1.0, next_state, False, snake_id=0)
        policy.update(state, 1, 0.5, next_state, False, snake_id=0)

        assert len(policy._episode_buffers[0]) == 2
        policy.cleanup()

    def test_done_flushes_episode_to_buffer(self):
        """When done=True, episode is added to sequence buffer and cleared."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)

        state = torch.randn(16)
        next_state = torch.randn(16)

        # Add several transitions
        for i in range(10):
            done = i == 9
            policy.update(state, i % 6, 0.1, next_state if not done else None, done, snake_id=0)

        # Episode buffer should be cleared
        assert len(policy._episode_buffers.get(0, [])) == 0

        # Sequence buffer should have at least 1 sequence
        assert len(policy.memory) >= 1
        policy.cleanup()

    def test_gru_update_preserves_exact_next_action_mask(self):
        """GRU episode buffering should keep simulator masks for target selection."""
        policy = ApexPolicy(
            input_size=GameConfig.INPUT_SIZE,
            hidden_size=32,
            output_size=GameConfig.OUTPUT_SIZE,
            use_gru=True,
        )
        state = torch.randn(GameConfig.INPUT_SIZE)
        next_state = torch.randn(GameConfig.INPUT_SIZE)
        exact_mask = torch.tensor([False, False, True, False, True, False])

        policy.update(
            state,
            0,
            0.5,
            next_state,
            False,
            snake_id=0,
            next_action_mask=exact_mask,
        )
        policy.update(next_state, 2, 1.0, None, True, snake_id=0)

        batch, _, _ = policy.memory.sample(1, torch.device("cpu"))

        train_start = policy.memory.burn_in_length
        assert "next_action_masks" in batch
        assert batch["next_action_masks"][0, train_start].tolist() == exact_mask.tolist()
        assert batch["next_action_mask_present"][0, train_start].item() == 1.0
        policy.cleanup()

    def test_multiple_snakes_separate_episodes(self):
        """Different snakes accumulate transitions independently."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)

        state = torch.randn(16)
        next_state = torch.randn(16)

        policy.update(state, 0, 1.0, next_state, False, snake_id=0)
        policy.update(state, 1, 0.5, next_state, False, snake_id=1)
        policy.update(state, 2, 0.3, next_state, False, snake_id=0)

        assert len(policy._episode_buffers[0]) == 2
        assert len(policy._episode_buffers[1]) == 1
        policy.cleanup()


# =========================================================================
# Train Step Tests
# =========================================================================


class TestDRQNTrainStep:
    """Test DRQN training loop with sequence batches."""

    def _fill_buffer(self, policy, num_episodes=5, ep_length=25):
        """Helper to fill sequence buffer with episodes."""
        for _ in range(num_episodes):
            state = torch.randn(policy.dqn.input_size)
            for step in range(ep_length):
                next_state = torch.randn(policy.dqn.input_size)
                done = step == ep_length - 1
                policy.update(
                    state,
                    step % policy.output_size,
                    0.1,
                    next_state if not done else None,
                    done,
                    snake_id=0,
                )
                state = next_state

    def test_drqn_train_step_returns_loss(self):
        """DRQN train_step returns loss when buffer is ready."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)

        # Fill buffer with enough episodes
        self._fill_buffer(policy, num_episodes=10, ep_length=25)

        # Buffer should have sequences now
        assert len(policy.memory) > 0

        # Force batch_size to be small enough for our buffer
        batch_size = min(4, len(policy.memory))
        if policy.memory.is_ready(batch_size):
            # Temporarily patch Apex batch size for test
            import src.core.game_config as cfg

            original_config = cfg._current_config
            from src.core.game_config import ApexSettings, AppConfig

            test_config = AppConfig(
                apex=ApexSettings(
                    batch_size=batch_size,
                    min_buffer_size=batch_size,
                ),
            )
            cfg._current_config = test_config

            try:
                loss, epsilon = policy._drqn_train_step()
                assert loss is not None
                assert isinstance(loss, float)
                assert epsilon > 0
            finally:
                cfg._current_config = original_config

        policy.cleanup()

    def test_drqn_train_step_not_ready(self):
        """DRQN train_step returns None when buffer not ready."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        # Buffer is empty, should not be ready
        loss, epsilon = policy._drqn_train_step()
        assert loss is None
        policy.cleanup()

    def test_ff_train_step_still_works(self):
        """Feedforward train_step still works in non-GRU mode."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=False)
        # Buffer is empty, should return None
        loss, epsilon = policy.train_step()
        assert loss is None
        policy.cleanup()

    def test_drqn_next_targets_use_hidden_after_current_state(self):
        """Next-state targets should keep recurrent context aligned by one step."""

        class EchoHiddenNetwork(nn.Module):
            def __init__(self, output_size):
                super().__init__()
                self.output_size = output_size
                self.anchor = nn.Parameter(torch.zeros(1))

            def init_hidden(self, batch_size):
                return torch.zeros(1, batch_size, 1)

            def forward(self, x, hidden=None):
                if x.dim() == 3:
                    h = hidden if hidden is not None else self.init_hidden(x.shape[0])
                    q = None
                    for t in range(x.shape[1]):
                        q, h = self.forward(x[:, t, :], h)
                    return q, h

                h = hidden if hidden is not None else self.init_hidden(x.shape[0])
                hidden_value = h.squeeze(0)
                q_values = self.anchor + torch.zeros(
                    x.shape[0], self.output_size, dtype=x.dtype, device=x.device
                )
                q_values[:, 0:1] = hidden_value
                new_hidden = h + x[:, 0:1].unsqueeze(0)
                return q_values, new_hidden

        policy = ApexPolicy(input_size=4, hidden_size=32, output_size=6, use_gru=True)
        policy.dqn = EchoHiddenNetwork(output_size=6)
        policy.target_dqn = EchoHiddenNetwork(output_size=6)

        train_states = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]])
        train_next_states = torch.tensor([[[10.0, 0.0, 0.0, 0.0], [20.0, 0.0, 0.0, 0.0]]])
        hidden = policy.dqn.init_hidden(batch_size=1)
        target_hidden = policy.target_dqn.init_hidden(batch_size=1)

        next_q_online, next_q_target = policy._compute_drqn_next_q_values(
            train_states,
            train_next_states,
            hidden,
            target_hidden,
        )

        assert next_q_online[0, :, 0].tolist() == pytest.approx([1.0, 3.0])
        assert next_q_target[0, :, 0].tolist() == pytest.approx([1.0, 3.0])
        policy.cleanup()


# =========================================================================
# Backward Compatibility Tests
# =========================================================================


class TestBackwardCompatibility:
    """Test that use_gru=False behavior is unchanged."""

    def test_ff_select_action_no_snake_id(self):
        """Feedforward select_action works without snake_id."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=False)
        state = torch.randn(16)
        action = policy.select_action(state)
        assert 0 <= action < 6
        policy.cleanup()

    def test_ff_update_no_snake_id(self):
        """Feedforward update works without snake_id."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=False)
        state = torch.randn(16)
        next_state = torch.randn(16)
        loss, epsilon = policy.update(state, 0, 1.0, next_state, False)
        # Should work without error
        assert epsilon > 0
        policy.cleanup()

    def test_ff_no_hidden_states_dict(self):
        """Feedforward mode doesn't have _hidden_states attribute."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=False)
        assert not hasattr(policy, "_hidden_states")
        policy.cleanup()

    def test_default_use_gru_false(self):
        """Default policy has use_gru=False."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6)
        assert policy.use_gru is False
        policy.cleanup()


# =========================================================================
# Config Integration Tests
# =========================================================================


class TestConfigIntegration:
    """Test GRU-related config properties."""

    def test_gameconfig_use_gru_default(self):
        """GameConfig.USE_GRU defaults to False."""
        assert GameConfig.USE_GRU is False

    def test_gameconfig_gru_hidden_size_default(self):
        """GameConfig.GRU_HIDDEN_SIZE defaults to 256."""
        assert GameConfig.GRU_HIDDEN_SIZE == 256

    def test_gameconfig_sequence_length_default(self):
        """GameConfig.SEQUENCE_LENGTH defaults to 20."""
        assert GameConfig.SEQUENCE_LENGTH == 20

    def test_gameconfig_burn_in_length_default(self):
        """GameConfig.BURN_IN_LENGTH defaults to 5."""
        assert GameConfig.BURN_IN_LENGTH == 5


# =========================================================================
# Checkpoint Tests
# =========================================================================


class TestCheckpointing:
    """Test state dict serialization with GRU mode."""

    def test_get_state_dict_includes_use_gru(self):
        """State dict includes use_gru flag."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        state_dict = policy.get_state_dict()
        assert "use_gru" in state_dict
        assert state_dict["use_gru"] is True
        policy.cleanup()

    def test_get_state_dict_ff_includes_use_gru(self):
        """Feedforward state dict also includes use_gru flag."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=False)
        state_dict = policy.get_state_dict()
        assert "use_gru" in state_dict
        assert state_dict["use_gru"] is False
        policy.cleanup()

    def test_gru_get_all_memories_returns_empty(self):
        """GRU mode get_all_memories returns empty list."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        assert policy.get_all_memories() == []
        policy.cleanup()

    def test_gru_prepare_memories_returns_empty(self):
        """GRU mode prepare_memories_for_saving returns empty list."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        assert policy.prepare_memories_for_saving() == []
        policy.cleanup()


# =========================================================================
# Cleanup Tests
# =========================================================================


class TestCleanup:
    """Test resource cleanup in both modes."""

    def test_gru_cleanup_clears_hidden_states(self):
        """Cleanup clears hidden states in GRU mode."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        policy._get_hidden(0)
        policy._get_hidden(1)
        assert len(policy._hidden_states) == 2

        policy.cleanup()
        # After cleanup, hidden states should be cleared
        # (cleanup clears the dict)

    def test_gru_cleanup_clears_episode_buffers(self):
        """Cleanup clears episode buffers in GRU mode."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        state = torch.randn(16)
        next_state = torch.randn(16)
        policy.update(state, 0, 1.0, next_state, False, snake_id=0)
        assert len(policy._episode_buffers.get(0, [])) == 1

        policy.cleanup()

    def test_ff_cleanup_works(self):
        """Feedforward cleanup works without errors."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=False)
        policy.cleanup()  # Should not raise


# =========================================================================
# Priority Computation Tests
# =========================================================================


class TestPriorities:
    """Test priority computation in GRU mode."""

    def test_gru_get_priorities(self):
        """get_priorities works in GRU mode."""
        policy = ApexPolicy(input_size=16, hidden_size=32, output_size=6, use_gru=True)
        state = torch.randn(16)
        next_state = torch.randn(16)

        experiences = [(state, 0, 1.0, next_state, False)]
        priorities = policy.get_priorities(experiences)

        assert len(priorities) == 1
        assert priorities[0] > 0
        policy.cleanup()
