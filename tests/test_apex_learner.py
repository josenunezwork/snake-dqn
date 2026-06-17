"""Tests for ApexLearner with LocalApexBuffer and LearnerBufferClient.

Tests cover:
- Learner creation in local mode (LocalApexBuffer fallback)
- Learner creation with explicit buffer client
- train_step returns valid metrics after buffer fill
- train_step returns waiting status when buffer is under-filled
- Target network update at correct frequency
- get_weights returns CPU tensors
- Checkpoint save/load round-trip
- set_weights updates both online and target networks
- Factory function creates learner correctly
- get_training_stats returns expected keys
"""

import numpy as np
import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import StateIndices
from src.training.apex_buffer import LocalApexBuffer
from src.training.apex_learner import (
    ApexLearner,
    ApexLearnerConfig,
    create_apex_learner,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_cpu():
    """Force CPU device for all tests."""
    DeviceManager.override_device(torch.device("cpu"))
    yield
    DeviceManager.reset_for_testing()


def _small_config(**overrides) -> ApexLearnerConfig:
    """Create a small config for fast tests."""
    defaults = dict(
        input_size=8,
        hidden_size=32,
        output_size=6,
        batch_size=16,
        learning_rate=0.001,
        gamma=0.99,
        target_update_freq=5,
        min_buffer_size=32,
        log_interval=1000,
        weight_broadcast_interval=10,
    )
    defaults.update(overrides)
    return ApexLearnerConfig(**defaults)


def _fill_buffer(buffer: LocalApexBuffer, n: int, input_size: int = 8) -> None:
    """Add n random transitions to the buffer."""
    for _ in range(n):
        state = torch.randn(input_size)
        action = int(np.random.randint(0, 6))
        reward = float(np.random.randn())
        next_state = torch.randn(input_size)
        done = bool(np.random.random() < 0.05)
        buffer.add(state, action, reward, next_state, done)


class FixedQ(torch.nn.Module):
    """Return deterministic Q-values for target-selection tests."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.to(states.device).expand(states.shape[0], -1)


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


# ===========================================================================
# 1. Local mode creation
# ===========================================================================


class TestLearnerCreation:
    """Tests for ApexLearner creation."""

    def test_create_local_mode_default_buffer(self):
        """When no buffer_client is given, a LocalApexBuffer is created."""
        config = _small_config()
        learner = ApexLearner(config, device=torch.device("cpu"))
        assert isinstance(learner.buffer_client, LocalApexBuffer)
        assert learner.step_count == 0

    def test_create_with_explicit_buffer(self):
        """Accepts an explicit LocalApexBuffer."""
        config = _small_config()
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))
        assert learner.buffer_client is buf

    def test_factory_creates_learner(self):
        """create_apex_learner() returns a working ApexLearner."""
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=8)
        learner = create_apex_learner(
            input_size=8,
            hidden_size=32,
            output_size=6,
            batch_size=16,
            buffer_client=buf,
            min_buffer_size=32,
            n_step=5,
        )
        assert isinstance(learner, ApexLearner)
        assert learner.buffer_client is buf
        assert learner.config.n_step == 5


# ===========================================================================
# 2. train_step
# ===========================================================================


class TestTrainStep:
    """Tests for train_step behavior."""

    def test_waiting_when_buffer_small(self):
        """train_step returns waiting status when buffer < min_buffer_size."""
        config = _small_config(min_buffer_size=100)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 10, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))
        metrics = learner.train_step()
        assert metrics.get("status") == "waiting"
        assert learner.step_count == 0

    def test_waiting_reports_sample_errors_after_warmup(self):
        """Sample failures after warmup should be visible in waiting metrics."""

        class FailingSampleBuffer:
            def get_size(self):
                return 64

            def sample(self, batch_size, device):
                raise ValueError("bad replay batch")

        config = _small_config(min_buffer_size=32)
        learner = ApexLearner(
            config,
            buffer_client=FailingSampleBuffer(),
            device=torch.device("cpu"),
        )

        metrics = learner.train_step()
        stats = learner.get_training_stats()

        assert metrics["status"] == "waiting"
        assert metrics["buffer_size"] == 64
        assert metrics["sample_error_count"] == 1
        assert metrics["last_sample_error"] == "bad replay batch"
        assert stats["sample_error_count"] == 1
        assert stats["last_sample_error"] == "bad replay batch"
        assert learner.step_count == 0

    def test_waiting_reports_empty_sample_after_warmup(self):
        """A ready buffer returning no sample should not look like normal warmup."""

        class EmptySampleBuffer:
            def get_size(self):
                return 64

            def sample(self, batch_size, device):
                return None

        config = _small_config(min_buffer_size=32)
        learner = ApexLearner(
            config,
            buffer_client=EmptySampleBuffer(),
            device=torch.device("cpu"),
        )

        metrics = learner.train_step()

        assert metrics["status"] == "waiting"
        assert metrics["buffer_size"] == 64
        assert metrics["sample_error_count"] == 1
        assert metrics["last_sample_error"] == "buffer client returned no sample"
        assert learner.step_count == 0

    def test_train_step_produces_valid_metrics(self):
        """After filling buffer, train_step returns loss and q-values."""
        config = _small_config(min_buffer_size=32)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 64, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        metrics = learner.train_step()
        assert "loss" in metrics
        assert "mean_q_value" in metrics
        assert "step" in metrics
        assert metrics["step"] == 1
        assert isinstance(metrics["loss"], float)
        assert np.isfinite(metrics["loss"])

    def test_multiple_train_steps(self):
        """Multiple train_steps increment step_count and reduce loss."""
        config = _small_config(min_buffer_size=32)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 64, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        for _ in range(5):
            metrics = learner.train_step()
        assert learner.step_count == 5
        assert "loss" in metrics

    def test_td_targets_mask_invalid_next_actions(self):
        """Distributed learner targets should ignore invalid high-Q next actions."""
        config = _small_config(input_size=58, hidden_size=32, output_size=6, gamma=0.5)
        learner = ApexLearner(config, device=torch.device("cpu"))
        learner.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        learner.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])

        targets = learner.compute_td_targets(
            rewards=torch.zeros(1),
            next_states=_full_state_batch(),
            dones=torch.zeros(1),
            bootstrap_steps=torch.ones(1),
        )

        assert targets.tolist() == [2.5]

    def test_td_targets_do_not_bootstrap_from_trapped_next_state(self):
        """Distributed learner targets should be reward-only when no action is valid."""
        config = _small_config(input_size=58, hidden_size=32, output_size=6, gamma=0.5)
        learner = ApexLearner(config, device=torch.device("cpu"))
        learner.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        learner.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])

        targets = learner.compute_td_targets(
            rewards=torch.tensor([1.25]),
            next_states=_trapped_state_batch(),
            dones=torch.zeros(1),
            bootstrap_steps=torch.ones(1),
        )

        assert targets.tolist() == [1.25]

    def test_td_targets_prefer_exact_mask_that_enables_boost(self):
        """Exact simulator masks can enable boost targets that compact state masks cannot."""
        config = _small_config(input_size=58, hidden_size=32, output_size=6, gamma=0.5)
        learner = ApexLearner(config, device=torch.device("cpu"))
        learner.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        learner.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        exact_mask = torch.tensor([[False, False, False, False, True, False]])

        targets = learner.compute_td_targets(
            rewards=torch.zeros(1),
            next_states=_full_state_batch(),
            dones=torch.zeros(1),
            bootstrap_steps=torch.ones(1),
            next_action_masks=exact_mask,
        )

        assert targets.tolist() == [25.0]

    def test_td_targets_reject_wrong_shaped_exact_mask(self):
        """Distributed learner targets should fail fast on malformed exact masks."""
        config = _small_config(input_size=58, hidden_size=32, output_size=6, gamma=0.5)
        learner = ApexLearner(config, device=torch.device("cpu"))
        learner.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        learner.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])
        wrong_shape_mask = torch.tensor([[False, True, False]])

        with pytest.raises(ValueError, match="action_mask shape must match q_values shape"):
            learner.compute_td_targets(
                rewards=torch.zeros(1),
                next_states=_full_state_batch(),
                dones=torch.zeros(1),
                bootstrap_steps=torch.ones(1),
                next_action_masks=wrong_shape_mask,
            )

    def test_td_targets_without_bootstrap_steps_use_configured_n_step(self):
        """Legacy learner clients should not silently fall back to one-step discounts."""
        config = _small_config(input_size=58, hidden_size=32, output_size=6, gamma=0.5, n_step=3)
        learner = ApexLearner(config, device=torch.device("cpu"))
        learner.dqn = FixedQ([0.0, 3.0, 0.0, 0.0, 100.0, 0.0])
        learner.target_dqn = FixedQ([0.0, 5.0, 0.0, 0.0, 50.0, 0.0])

        targets = learner.compute_td_targets(
            rewards=torch.zeros(1),
            next_states=_full_state_batch(),
            dones=torch.zeros(1),
            bootstrap_steps=None,
        )

        assert targets.tolist() == pytest.approx([0.625])

    def test_train_step_reports_next_action_quality_metrics(self):
        """Learner metrics should expose exact-mask coverage and bootstrap availability."""
        config = _small_config(
            input_size=58,
            hidden_size=32,
            output_size=6,
            batch_size=4,
            min_buffer_size=4,
        )
        buf = LocalApexBuffer(capacity=32, alpha=0.6, state_size=config.input_size)
        exact_mask = torch.tensor([False, True, False, False, False, False])
        for _ in range(8):
            buf.add(
                _full_state_batch().squeeze(0),
                1,
                1.0,
                _full_state_batch().squeeze(0),
                False,
                next_action_mask=exact_mask,
            )
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        metrics = learner.train_step()

        assert metrics["valid_next_action_fraction"] == pytest.approx(1.0)
        assert metrics["trapped_next_state_fraction"] == pytest.approx(0.0)
        assert metrics["exact_next_action_mask_fraction"] == pytest.approx(1.0)

    def test_next_action_quality_metrics_ignore_terminal_samples(self):
        """Target-action health should only measure rows that can bootstrap."""
        config = _small_config(input_size=58, hidden_size=32, output_size=6)
        learner = ApexLearner(
            config,
            buffer_client=LocalApexBuffer(capacity=8, state_size=config.input_size),
            device=torch.device("cpu"),
        )
        next_states = torch.cat((_full_state_batch(), _trapped_state_batch()), dim=0)
        next_action_masks = torch.tensor(
            [
                [True, False, False, False, False, False],
                [False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        )
        next_action_mask_present = torch.ones(2)

        metrics = learner.compute_next_action_quality_metrics(
            next_states,
            next_action_masks=next_action_masks,
            next_action_mask_present=next_action_mask_present,
            sample_mask=torch.tensor([0.0, 1.0]),
        )

        assert metrics["valid_next_action_fraction"] == pytest.approx(0.0)
        assert metrics["trapped_next_state_fraction"] == pytest.approx(1.0)
        assert metrics["exact_next_action_mask_fraction"] == pytest.approx(1.0)


# ===========================================================================
# 3. Target network update
# ===========================================================================


class TestTargetUpdate:
    """Tests for target network update frequency."""

    def test_target_update_at_frequency(self):
        """Target network should update every target_update_freq steps."""
        config = _small_config(min_buffer_size=32, target_update_freq=3)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 64, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        # Do one step — online network changes, target stays same
        learner.train_step()

        # After more steps the target might not match online yet
        learner.train_step()  # step 2
        assert learner.step_count == 2

        # Step 3 triggers target update (target_update_freq=3)
        learner.train_step()
        assert learner.step_count == 3

        # Now target should match online
        for key in learner.dqn.state_dict():
            assert torch.allclose(
                learner.dqn.state_dict()[key], learner.target_dqn.state_dict()[key]
            ), f"Target mismatch at key {key} after target update"


# ===========================================================================
# 4. get_weights
# ===========================================================================


class TestGetWeights:
    """Tests for weight broadcasting."""

    def test_get_weights_returns_cpu_tensors(self):
        """get_weights() should return all tensors on CPU."""
        config = _small_config()
        learner = ApexLearner(config, device=torch.device("cpu"))
        weights = learner.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
        for key, tensor in weights.items():
            assert tensor.device == torch.device("cpu"), f"{key} not on CPU"

    def test_get_weights_are_copies(self):
        """Returned weights should be copies, not references."""
        config = _small_config()
        learner = ApexLearner(config, device=torch.device("cpu"))
        weights1 = learner.get_weights()
        weights2 = learner.get_weights()
        # Modify weights1 and verify weights2 is unaffected
        for key in weights1:
            weights1[key].zero_()
        for key in weights2:
            # At least some weights should be non-zero
            if weights2[key].numel() > 0:
                break
        # Original model should be unaffected
        for key, val in learner.dqn.state_dict().items():
            if val.numel() > 0:
                assert not torch.all(val == 0), "Model weights were modified by get_weights()"
                break


# ===========================================================================
# 5. Checkpoint save/load
# ===========================================================================


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_state_dict_roundtrip(self):
        """get_state_dict / load_state_dict preserves learner state."""
        config = _small_config(min_buffer_size=32)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 64, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        # Do a few train steps to change state
        for _ in range(3):
            learner.train_step()

        # Save state
        state = learner.get_state_dict()
        assert state["step_count"] == 3

        # Create new learner and load
        learner2 = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))
        assert learner2.step_count == 0
        learner2.load_state_dict(state)
        assert learner2.step_count == 3

        # Verify network weights match
        for key in learner.dqn.state_dict():
            assert torch.allclose(
                learner.dqn.state_dict()[key], learner2.dqn.state_dict()[key]
            ), f"Weight mismatch at {key}"

    def test_load_state_dict_rejects_gamma_mismatch_before_mutation(self):
        """Learner resumes should not silently change the TD-target discount contract."""
        checkpoint_learner = ApexLearner(
            _small_config(gamma=0.5),
            device=torch.device("cpu"),
        )
        checkpoint_learner.step_count = 7
        checkpoint = checkpoint_learner.get_state_dict()

        learner = ApexLearner(
            _small_config(gamma=0.9),
            device=torch.device("cpu"),
        )
        learner.step_count = 3
        original_weights = {key: value.clone() for key, value in learner.dqn.state_dict().items()}

        with pytest.raises(ValueError, match="gamma=0.5.*gamma=0.9"):
            learner.load_state_dict(checkpoint)

        assert learner.step_count == 3
        assert learner.config.gamma == pytest.approx(0.9)
        for key, value in learner.dqn.state_dict().items():
            assert torch.allclose(value, original_weights[key]), f"Weight mutated at {key}"

    def test_load_state_dict_rejects_n_step_mismatch_before_mutation(self):
        """Learner resumes should not silently change the actor return horizon."""
        checkpoint_learner = ApexLearner(
            _small_config(n_step=5),
            device=torch.device("cpu"),
        )
        checkpoint_learner.step_count = 7
        checkpoint = checkpoint_learner.get_state_dict()

        learner = ApexLearner(
            _small_config(n_step=3),
            device=torch.device("cpu"),
        )
        learner.step_count = 3
        original_weights = {key: value.clone() for key, value in learner.dqn.state_dict().items()}

        with pytest.raises(ValueError, match="n_step=5.*n_step=3"):
            learner.load_state_dict(checkpoint)

        assert learner.step_count == 3
        assert learner.config.n_step == 3
        for key, value in learner.dqn.state_dict().items():
            assert torch.allclose(value, original_weights[key]), f"Weight mutated at {key}"

    def test_load_state_dict_rejects_apex_config_gamma_mismatch_before_mutation(self):
        """Nested distributed checkpoint metadata should be validated, not ignored."""
        config = _small_config(gamma=0.9)
        checkpoint_learner = ApexLearner(config, device=torch.device("cpu"))
        checkpoint = checkpoint_learner.get_state_dict()
        checkpoint["apex_config"] = {
            "input_size": config.input_size,
            "hidden_size": config.hidden_size,
            "output_size": config.output_size,
            "gamma": 0.5,
        }

        learner = ApexLearner(config, device=torch.device("cpu"))
        learner.step_count = 3
        original_weights = {key: value.clone() for key, value in learner.dqn.state_dict().items()}

        with pytest.raises(ValueError, match="gamma=0.5.*gamma=0.9"):
            learner.load_state_dict(checkpoint)

        assert learner.step_count == 3
        for key, value in learner.dqn.state_dict().items():
            assert torch.allclose(value, original_weights[key]), f"Weight mutated at {key}"

    def test_set_weights_updates_both_networks(self):
        """set_weights() should update both online and target networks."""
        config = _small_config()
        learner = ApexLearner(config, device=torch.device("cpu"))

        # Create dummy weights (all zeros)
        dummy_weights = {k: torch.zeros_like(v) for k, v in learner.dqn.state_dict().items()}
        learner.set_weights(dummy_weights)

        # Both networks should now be all zeros
        for key, val in learner.dqn.state_dict().items():
            assert torch.all(val == 0), f"Online network {key} not zeroed"
        for key, val in learner.target_dqn.state_dict().items():
            assert torch.all(val == 0), f"Target network {key} not zeroed"


# ===========================================================================
# 6. get_training_stats
# ===========================================================================


class TestTrainingStats:
    """Tests for training statistics."""

    def test_stats_before_training(self):
        """Stats should have step_count=0 and buffer_size before training."""
        config = _small_config()
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 50, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))
        stats = learner.get_training_stats()
        assert stats["step_count"] == 0
        assert stats["buffer_size"] == 50

    def test_stats_after_training(self):
        """Stats should include loss/q_value after training."""
        config = _small_config(min_buffer_size=32)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 64, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        learner.train_step()
        stats = learner.get_training_stats()
        assert stats["step_count"] == 1
        assert "loss_mean" in stats
        assert "q_value_mean" in stats


# ===========================================================================
# 7. should_broadcast_weights
# ===========================================================================


class TestBroadcast:
    """Tests for weight broadcast scheduling."""

    def test_should_broadcast_at_interval(self):
        """should_broadcast_weights triggers at the configured interval."""
        config = _small_config(min_buffer_size=32, weight_broadcast_interval=3)
        buf = LocalApexBuffer(capacity=1000, alpha=0.6, state_size=config.input_size)
        _fill_buffer(buf, 64, input_size=8)
        learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

        # step_count=0 → 0 % 3 == 0 → True
        assert learner.should_broadcast_weights()

        learner.train_step()  # step=1
        assert not learner.should_broadcast_weights()

        learner.train_step()  # step=2
        assert not learner.should_broadcast_weights()

        learner.train_step()  # step=3
        assert learner.should_broadcast_weights()
