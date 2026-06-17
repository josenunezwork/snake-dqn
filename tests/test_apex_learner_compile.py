"""Tests for ApexLearner compile gating and CPU safety contract."""

import types

import pytest
import torch

from src.core.device_manager import DeviceManager
from src.core.game_config import get_config, initialize_config
from src.model.apex_network import ApexNetwork
from src.training.apex_buffer import LocalApexBuffer
from src.training.apex_learner import ApexLearner, ApexLearnerConfig


@pytest.fixture(autouse=True)
def _reset_runtime_state():
    """Keep device and global config deterministic for these tests."""
    original_config = get_config()
    DeviceManager.override_device(torch.device("cpu"))
    yield
    DeviceManager.reset_for_testing()
    initialize_config(original_config)


def _small_config(**overrides) -> ApexLearnerConfig:
    defaults = dict(
        input_size=8,
        hidden_size=32,
        output_size=6,
        batch_size=8,
        learning_rate=0.001,
        gamma=0.99,
        target_update_freq=5,
        min_buffer_size=8,
        log_interval=1000,
        weight_broadcast_interval=10,
    )
    defaults.update(overrides)
    return ApexLearnerConfig(**defaults)


def _fill_buffer(buffer: LocalApexBuffer, n: int, input_size: int = 8) -> None:
    """Add random transitions to a local learner buffer."""
    for _ in range(n):
        state = torch.randn(input_size)
        action = int(torch.randint(0, 6, (1,)).item())
        reward = float(torch.randn(()).item())
        next_state = torch.randn(input_size)
        done = False
        buffer.add(state, action, reward, next_state, done)


def _normalize_compiled_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        normalized[key.removeprefix("_orig_mod.")] = value
    return normalized


def test_compiled_state_dict_key_parity_and_broadcast_contract():
    """A compiled model should not break checkpoint/broadcast key compatibility."""
    compiled = torch.compile(ApexNetwork(58, 32, 6))
    compiled_keys = set(compiled.state_dict().keys())
    baseline_keys = set(ApexNetwork(58, 32, 6).state_dict().keys())

    if compiled_keys != baseline_keys:
        assert all(key.startswith("_orig_mod.") for key in compiled_keys)

    cleaned_compiled = _normalize_compiled_state_dict(compiled.state_dict())
    assert set(cleaned_compiled.keys()) == baseline_keys

    unloaded = ApexNetwork(58, 32, 6)
    load_result = unloaded.load_state_dict(cleaned_compiled)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    # Primary safety assertion for the actual learner code path: broadcasted weights
    # must remain compatible with a normal eager ApexNetwork.
    config = _small_config(input_size=58, hidden_size=32)
    buf = LocalApexBuffer(capacity=256, alpha=config.priority_alpha, state_size=config.input_size)
    _fill_buffer(buf, 32, input_size=config.input_size)
    learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))
    fresh = ApexNetwork(config.input_size, config.hidden_size, config.output_size)
    fresh.load_state_dict(learner.get_weights())


def test_maybe_compile_online_only_compiles_on_cuda():
    """Only CUDA + flag should allow compiled online forward."""
    config = _small_config(use_compile=True, min_buffer_size=1)
    learner = ApexLearner(config, device=torch.device("cpu"))
    assert learner._online_forward is learner.dqn

    learner.device = types.SimpleNamespace(type="mps")
    assert learner._maybe_compile_online() is learner.dqn


def test_train_step_and_td_targets_keep_weights_loadable():
    """Train-step path should stay unchanged on CPU and broadcast payload stays clean."""
    config = _small_config(min_buffer_size=8, input_size=8)
    buf = LocalApexBuffer(capacity=256, alpha=config.priority_alpha, state_size=config.input_size)
    _fill_buffer(buf, 32, input_size=config.input_size)
    learner = ApexLearner(config, buffer_client=buf, device=torch.device("cpu"))

    for _ in range(3):
        metrics = learner.train_step()
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))

    # Direct td target smoke test with explicit CPU tensors.
    states = torch.zeros((4, config.input_size), dtype=torch.float32)
    td_targets = learner.compute_td_targets(
        rewards=torch.zeros(4),
        next_states=states,
        dones=torch.zeros(4),
        bootstrap_steps=torch.ones(4),
    )
    assert torch.isfinite(td_targets).all()

    ApexNetwork(
        config.input_size,
        config.hidden_size,
        config.output_size,
    ).load_state_dict(learner.get_weights())
