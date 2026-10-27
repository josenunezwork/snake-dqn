"""CPU compiler-adapter tests; CUDA validation remains a runtime self-check."""

from __future__ import annotations

import copy

import pytest
import torch

from research.sandbox_a_resident.training_compile import TrainingCompile
from research.sandbox_a_resident.training_model import B8Actor
from research.sandbox_a_resident.training_task import ResidentTrainingTask, TrainingTaskConfig
from src.model.ego2s_network import Ego2sNet


def _inputs(rows: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randint(0, 256, (rows, 7, 31, 31), dtype=torch.uint8),
        torch.randint(0, 256, (rows, 4, 37, 37), dtype=torch.uint8),
        torch.rand(rows, 60),
    )


def test_compile_adapter_keeps_actor_state_and_export_surface_unwrapped() -> None:
    torch.set_num_threads(1)
    actor = B8Actor(b8_model=Ego2sNet(local_channels=7, n_scalars=60))
    before = copy.deepcopy(actor.state_dict())
    compiled = TrainingCompile(actor, mode="default", backend="eager")
    local, glob, scalars = _inputs()
    torch.testing.assert_close(
        compiled.actor_forward(local, glob, scalars), actor(local, glob, scalars)
    )
    assert not any("_orig_mod" in key for key in actor.state_dict())
    assert before.keys() == actor.state_dict().keys()
    assert compiled.gradient_probe(local, glob, scalars)["status"] == "compiled"


def test_compile_adapter_runtime_self_check_is_small_and_resume_safe() -> None:
    torch.set_num_threads(1)
    actor = B8Actor(b8_model=Ego2sNet(local_channels=7, n_scalars=60))
    adapter = TrainingCompile(actor, mode="default", backend="eager")
    state = copy.deepcopy(actor.state_dict())
    result = adapter.runtime_self_check(32)
    assert result["rows"] == 32
    actor.load_state_dict(state)
    local, glob, scalars = _inputs(2)
    adapter.actor_forward(local, glob, scalars).sum().backward()
    assert all(
        parameter.grad is not None for parameter in actor.parameters() if parameter.requires_grad
    )


def test_task_observe_uses_compiled_pure_tensor_v8_and_observation_ports() -> None:
    torch.set_num_threads(1)
    task = ResidentTrainingTask(TrainingTaskConfig(worlds=3, tiny=True), device="cpu")
    actor = B8Actor(b8_model=Ego2sNet(local_channels=7, n_scalars=60))
    adapter = TrainingCompile(actor, task, mode="default", backend="eager")
    eager, compiled = task.observe(), adapter.observe()
    for expected, actual in zip(eager, compiled, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert adapter.report["v8_features"]["status"] == "compiled"
    assert adapter.report["observation"]["status"] == "compiled"


def test_none_mode_never_calls_torch_compile_and_reports_disabled(monkeypatch) -> None:
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("mode=none must not call torch.compile")

    monkeypatch.setattr(torch, "compile", forbidden)
    actor = B8Actor(b8_model=Ego2sNet(local_channels=7, n_scalars=60))
    adapter = TrainingCompile(actor, mode="none")
    local, glob, scalars = _inputs(2)
    torch.testing.assert_close(
        adapter.actor_forward(local, glob, scalars), actor(local, glob, scalars)
    )
    assert adapter.runtime_self_check(32)["status"] == "disabled_eager"
    assert adapter.report["actor"]["fallback_reason"] == "requested mode=none"


def test_runtime_check_validates_real_collection_shape_without_gradients(monkeypatch):
    actor = B8Actor(b8_model=Ego2sNet(local_channels=7, n_scalars=60))
    adapter = TrainingCompile(actor, mode="default", backend="eager")
    observed = []
    original = adapter.actor_forward

    def forward(local, glob, scalars):
        observed.append((local.shape[0], torch.is_grad_enabled()))
        return original(local, glob, scalars)

    monkeypatch.setattr(adapter, "actor_forward", forward)
    monkeypatch.setattr(
        adapter, "gradient_probe", lambda *args: pytest.fail("unused backward graph")
    )
    before = torch.get_rng_state().clone()
    assert adapter.runtime_self_check(5)["rows"] == 5
    assert observed == [(5, False)]
    assert torch.equal(before, torch.get_rng_state())
    assert all(p.grad is None for p in actor.parameters())
