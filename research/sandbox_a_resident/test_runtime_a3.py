"""A-3 sparse ownership, profiler and explicit compiler fallback contracts."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from research.sandbox_a_resident.core import BOOTSTRAP_FIELDS, Recipe, ResidentAssay
from research.sandbox_a_resident.runtime_tools import CompiledBlock, PhaseProfiler
from src.model.ego2s_network import Ego2sNet


def test_lazy_compiler_failure_records_persistent_eager_fallback(monkeypatch) -> None:
    calls = []

    def compile_block(eager, **kwargs):
        assert kwargs == {"mode": "default", "dynamic": False, "fullgraph": True}

        def failed(value):
            calls.append("compile")
            raise RuntimeError("offline compiler unavailable")

        return failed

    monkeypatch.setattr(torch, "compile", compile_block)
    report = {}
    block = CompiledBlock(lambda x: x + 1, "default", report, "anchors")
    assert block(torch.tensor(1)).item() == 2
    assert block(torch.tensor(2)).item() == 3
    assert calls == ["compile"]
    assert report["anchors"]["status"] == "eager_fallback"
    assert "offline compiler unavailable" in report["anchors"]["fallback_reason"]


def test_compile_success_and_setup_failure_are_distinguished(monkeypatch) -> None:
    report = {}
    monkeypatch.setattr(torch, "compile", lambda eager, **kwargs: eager)
    block = CompiledBlock(lambda x: x * 2, "default", report, "ActorCritic")
    assert block(torch.tensor(3)).item() == 6
    assert report["ActorCritic"]["status"] == "compiled"

    def unavailable(*args, **kwargs):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch, "compile", unavailable)
    fallback = CompiledBlock(lambda x: x, "default", report, "v8_decide")
    assert fallback(9) == 9
    assert report["v8_decide"]["status"] == "eager_fallback"


def test_cuda_events_synchronize_only_after_last_phase(monkeypatch) -> None:
    operations = []

    class Event:
        def __init__(self, **kwargs):
            assert kwargs == {"enable_timing": True}

        def record(self):
            operations.append("record")

        def synchronize(self):
            operations.append("sync")

        def elapsed_time(self, end):
            assert operations.count("sync") == 1
            return 2.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    profiler = PhaseProfiler(SimpleNamespace(type="cuda"), 2)
    for _ in range(2):
        with profiler.phase("t_step"):
            assert "sync" not in operations
    report = profiler.finish()
    assert operations == ["record"] * 4 + ["sync"]
    assert report["mode"] == "cuda_event"
    assert report["ms_per_tick"] == {"t_step": 2.0}


def test_sparse_bank_owns_only_captured_rows_and_all_phases_reported(monkeypatch) -> None:
    torch.set_num_threads(1)
    runner = ResidentAssay(
        Recipe(worlds=6, horizon=3, minibatch=3, episode_limit=2, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    runner.age.copy_(torch.tensor([1, 0, 0, 0, 0, 0]))
    original_step = runner.sim.step

    def survive(actions):
        original_step(actions)
        runner.sim._last_done.zero_()

    monkeypatch.setattr(runner.sim, "step", survive)
    rollout = runner.collect()
    bank = runner._bootstrap_bank
    count = int(rollout["truncated"].sum())
    assert count > 0
    assert count < bank["valid"].numel()
    for name in BOOTSTRAP_FIELDS:
        assert bank[name].shape[:2] == (1, count), name
    assert set(runner.phase_profile["ms_per_tick"]) == {
        "t_obs",
        "t_actor",
        "t_anchor",
        "t_step",
        "t_bank_reset",
        "t_post",
    }
    assert runner.phase_profile["mode"] == "cpu_wall"


def test_empty_reset_performs_no_world_writes() -> None:
    torch.set_num_threads(1)
    runner = ResidentAssay(Recipe(worlds=2, tiny=True), Ego2sNet(local_channels=7, n_scalars=60))
    versions = {
        name: value._version
        for name, value in runner.sim.__dict__.items()
        if isinstance(value, torch.Tensor)
    }
    runner._reset(torch.zeros(2, dtype=torch.bool))
    assert versions == {name: getattr(runner.sim, name)._version for name in versions}


def test_h128_ep512_stagger_uses_four_cohorts() -> None:
    torch.set_num_threads(1)
    runner = ResidentAssay(
        Recipe(worlds=8, horizon=128, episode_limit=512, tiny=True, stagger_initial_age=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    assert runner.age.tolist() == [0, 128, 256, 384] * 2


def test_compiled_output_mismatch_falls_back_before_consumption(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda eager, **kwargs: lambda x: x + 2)
    report = {}
    block = CompiledBlock(lambda x: x + 1, "default", report, "anchors")
    assert block(torch.tensor(1)).item() == 2
    assert report["anchors"]["status"] == "eager_fallback"


def test_actor_backward_compiler_failure_recomputes_minibatch_eagerly(monkeypatch) -> None:
    torch.set_num_threads(1)
    runner = ResidentAssay(
        Recipe(worlds=2, horizon=2, minibatch=4, tiny=True),
        Ego2sNet(local_channels=7, n_scalars=60),
    )
    rollout = runner.collect()

    class BrokenBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            raise RuntimeError("compiler backward failed")

    def compile_block(eager, **kwargs):
        def forward(*args):
            logits, value = eager(*args)
            return BrokenBackward.apply(logits), BrokenBackward.apply(value)

        return forward

    monkeypatch.setattr(torch, "compile", compile_block)
    runner._forward = CompiledBlock(runner.actor, "default", runner.compile_report, "ActorCritic")
    result = runner.update(rollout)
    assert torch.isfinite(result["loss"])
    assert runner.compile_report["ActorCritic"]["status"] == "eager_fallback"
    assert "compiler backward failed" in runner.compile_report["ActorCritic"]["fallback_reason"]
