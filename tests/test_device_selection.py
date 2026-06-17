"""Tests for environment-driven device selection."""

import pytest
import torch

from src.core.device_manager import DeviceManager


@pytest.fixture(autouse=True)
def _reset_device_manager_state(monkeypatch):
    """Ensure each test starts and ends with a clean DeviceManager state."""
    DeviceManager.reset_for_testing()
    monkeypatch.delenv("SNAKE_DQN_DEVICE", raising=False)
    yield
    DeviceManager.reset_for_testing()
    monkeypatch.delenv("SNAKE_DQN_DEVICE", raising=False)


def test_device_manager_uses_cpu_when_env_requests_cpu(monkeypatch):
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    assert DeviceManager.get_device().type == "cpu"


def test_device_manager_auto_never_selects_mps(monkeypatch):
    # Policy: auto-detection prefers CUDA then CPU; MPS is never auto-selected
    # because it is ~5x slower than CPU for this small model on Apple Silicon.
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "auto")
    device = DeviceManager.get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}
    assert device.type != "mps"
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_device_manager_explicit_mps_is_honored(monkeypatch):
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "mps")
    device = DeviceManager.get_device()
    assert isinstance(device, torch.device)
    if torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        # No MPS hardware: falls back to auto (cuda or cpu), never crashes.
        assert device.type in {"cuda", "cpu"}


def test_device_manager_cuda_request_falls_back_without_cuda(monkeypatch):
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cuda")
    device = DeviceManager.get_device()
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        # Fallback uses auto detection, which never selects MPS.
        assert device.type == "cpu"


def test_device_manager_reverts_to_auto_on_unrecognized_value(monkeypatch):
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "something-random")
    device = DeviceManager.get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "cpu"}
    assert device.type != "mps"


def test_device_override_wins_over_environment_variable(monkeypatch):
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    override_device = (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    DeviceManager.override_device(override_device)
    assert DeviceManager.get_device() == override_device
