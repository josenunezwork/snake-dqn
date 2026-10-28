"""Focused checks for the sandbox-local CPU distribution reference."""

from __future__ import annotations

import inspect

import torch

from research.sandbox_a_resident.training_cpu_reference import (
    CpuRosterTask,
    _checkpoint_path,
    _CpuRosterRuntime,
    _seed,
)
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    DeviceRosterRuntime,
    proxy_rosters,
)
from research.sandbox_a_resident.training_sim import TrainingDeviceSim
from src.model.ego2s_network import load_ego2s_checkpoint
from src.model.inference_agent import InferenceAgent


def test_cpu_reference_uses_public_mix_partition_roots() -> None:
    assert _seed("scripted", 0) == _seed("scripted", 0)
    assert _seed("scripted", 0) != _seed("frozen", 0)
    assert _seed("mixed", 31) != _seed("mixed", 31, 1)
    assert _seed("scripted", 0, root_offset=1_600_000_000) != _seed("scripted", 0)
    task = CpuRosterTask("scripted", 2, 5000, checkpoint_paths=_paths(), root_offset=1_600_000_000)
    assert task._shard.seeds == [
        _seed("scripted", 0, root_offset=1_600_000_000),
        _seed("scripted", 1, root_offset=1_600_000_000),
    ]


def test_cpu_reference_closes_over_sandbox_and_src_only() -> None:
    source = inspect.getsource(
        __import__("research.sandbox_a_resident.training_cpu_reference", fromlist=["CpuRosterTask"])
    )
    assert "research.bl_ppo" not in source
    assert "research.t34" not in source


def test_installed_frozen_checkpoint_loads_through_existing_inference_loader() -> None:
    agent = InferenceAgent.from_checkpoint(
        _checkpoint_path(FRP3_S12_SHA256), device=torch.device("cpu")
    )
    assert agent.input_size == 61


def _paths() -> dict[str, str]:
    return {
        FRP3_S12_SHA256: _checkpoint_path(FRP3_S12_SHA256),
        CHAMPION_SHA256: _checkpoint_path(CHAMPION_SHA256),
    }


def test_cpu_task_steps_each_gate_mix_with_actual_roster_dispatch() -> None:
    for mix in ("scripted", "frozen", "mixed"):
        task = CpuRosterTask(mix, 1, 5000, checkpoint_paths=_paths())
        local, glob, scalars, legal = task.observe()
        assert (local.shape[0], glob.shape[0], scalars.shape, legal.shape) == (
            1,
            1,
            (1, 60),
            (1, 6),
        )
        event = task.step(torch.ones(1, dtype=torch.long))
        assert event.mass.shape == event.terminal.shape == event.reset.shape == (1,)


def test_cpu_task_does_not_reset_or_respawn_a_terminal_hero() -> None:
    task = CpuRosterTask("scripted", 1, 5000, checkpoint_paths=_paths())
    task.observe()
    task._shard.sim.alive[0, 0] = False
    task._shard.sim.respawn_timer[0, 0] = 0
    event = task.step(torch.ones(1, dtype=torch.long))
    assert event.terminal.item() is True
    assert event.reset.item() is False
    assert task._shard.active.item() is False


def test_cpu_observe_rasterizes_only_active_first_life_worlds(monkeypatch) -> None:
    """Retired worlds get inert fallback rows instead of repeated raster work."""
    import src.simd_env.ego_raster as raster

    task = CpuRosterTask("scripted", 2, 5000, checkpoint_paths=_paths())
    task._shard.active[:] = (True, False)
    real = raster.build_ego_raster
    seen = {}

    def observed_rows(*args, **kwargs):
        seen["rows"] = kwargs["rows"].copy()
        return real(*args, **kwargs)

    monkeypatch.setattr(raster, "build_ego_raster", observed_rows)
    local, glob, scalars, legal = task.observe()
    assert seen["rows"].tolist() == [[0, 0]]
    assert local.shape[0] == glob.shape[0] == scalars.shape[0] == legal.shape[0] == 2
    assert legal[1].tolist() == [True, True, True, False, False, False]


def test_cpu_reference_can_route_frozen_inference_to_cuda_semantics(monkeypatch) -> None:
    """The CPU simulator may select batched accelerator-only frozen forwards."""
    import src.simd_env.vector61_policy as vector_policy
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.grid_sim import GridBatchSim

    created = []

    class FakeNetwork:
        def to(self, device):
            self.last_device = torch.device(device)
            return self

    class FakeVectorPolicy:
        def __init__(self, *_args, forward, **_kwargs):
            self.forward = forward
            self.network = FakeNetwork()
            self.device = torch.device("cpu")
            created.append(self)

    monkeypatch.setattr(vector_policy, "Vector61SimdPolicy", FakeVectorPolicy)
    sim = GridBatchSim(BatchSimConfig(num_envs=1, num_snakes=6), seeds=[1], train_mode=False)
    cpu_runtime = _CpuRosterRuntime(
        sim,
        proxy_rosters("frozen", 1),
        [1],
        _paths(),
        inference_device="cpu",
    )
    assert cpu_runtime.inference_device.type == "cpu"
    assert created and all(policy.forward == "rowwise" for policy in created)

    created.clear()
    cuda_runtime = _CpuRosterRuntime(
        sim, proxy_rosters("frozen", 1), [1], _paths(), inference_device="cuda"
    )
    assert cuda_runtime.inference_device.type == "cuda"
    assert created and all(policy.forward == "batched" for policy in created)
    assert all(policy.network.last_device.type == "cuda" for policy in created)


def test_cpu_and_device_rosters_match_actions_on_three_copied_states_per_mix() -> None:
    """This is a copied-state action check, not a closed-loop simulator claim."""
    paths = _paths()
    root = _checkpoint_path(FRP3_S12_SHA256).rsplit("/saved_snakes/", 1)[0]
    b8 = load_ego2s_checkpoint(f"{root}/saved_snakes/b8_s3_t7_20261016.pth", "cpu").eval()
    frozen = {
        reference: InferenceAgent.from_checkpoint(path, device=torch.device("cpu")).network.eval()
        for reference, path in paths.items()
    }
    for mix in ("scripted", "frozen", "mixed"):
        task = CpuRosterTask(mix, 1, 5000, checkpoint_paths=paths)
        for _tick in range(3):
            task.observe()
            cpu_actions = torch.ones(1, dtype=torch.long).numpy()
            grid = torch.ones((1, 6), dtype=torch.long).numpy()
            grid[:, 0] = cpu_actions
            task._shard.runtime.actions(task._shard.sim, grid, task._shard.active)
            device = TrainingDeviceSim.from_reference(task._shard.sim, device="cpu")
            device.capacity_overflow = torch.zeros((1, 6), dtype=torch.bool)
            runtime = DeviceRosterRuntime(
                device,
                task._shard.rosters,
                b8_network=b8,
                checkpoint_networks=frozen,
                seeds=task._shard.seeds,
            )
            device_actions = runtime.actions(device, torch.ones(1, dtype=torch.long)).cpu().numpy()
            assert (device_actions == grid).all()
            task._shard.sim.step(grid, active_env_mask=task._shard.active)
            task._shard.runtime.after_step(task._shard.sim, task._shard.active)
