"""Non-gating acceptance diagnostics for the resident training adapter.

The checks use explicit, frozen actions and state fixtures.  They distinguish
exact discrete common-state mechanics from distributional simulator-D screens;
neither result is a Mac gate, release result, or CPU closed-loop identity claim.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.v8_device import v8_decide as device_v8
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.v8_nb import v8_decide as cpu_v8
from src.simd_env.vector61_featurizer import Vector61Featurizer, Vector61Memory, Vector61Params

_DISCRETE = ("bodies", "head_ptr", "seg_count", "length", "alive", "direction")


def forced_action_common_state(actions: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Replay forced actions from one manually supplied, RNG-free world state."""
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=2,
        game_width=200,
        game_height=100,
        initial_food=0,
        max_food=0,
        max_capacity=32,
        mechanics_version=1,
    )
    cpu = GridBatchSim(cfg, seeds=[17], train_mode=True, allow_respawn=False, jit=False)
    cpu.bodies[0, :, :5] = ((5, 5), (4, 5), (3, 5), (2, 5), (1, 5)), (
        (12, 5),
        (13, 5),
        (14, 5),
        (15, 5),
        (16, 5),
    )
    cpu.head_ptr[:] = 0
    cpu.seg_count[:] = 5
    cpu.length[:] = 5
    cpu.alive[:] = True
    cpu.direction[0] = (1, 3)
    cpu._rebuild_env_grids(0)
    device = DeviceSim.from_reference(cpu, device="cpu")
    for action in actions:
        values = np.asarray(action, dtype=np.int64).reshape(1, 2)
        cpu.step(values)
        device.step(torch.as_tensor(values))
        for name in _DISCRETE:
            expected = np.asarray(getattr(cpu, name))
            actual = getattr(device, name).detach().cpu().numpy()
            if not np.array_equal(actual, expected):
                return {"status": "MISMATCH", "field": name, "action": list(action)}
        if int(device.food_count[0]) != len(cpu.food_cells[0]):
            return {"status": "MISMATCH", "field": "food_count", "action": list(action)}
    return {"status": "PASS", "frames": len(actions), "claim": "common-state discrete mechanics"}


def vector_v8_common_state(sim: GridBatchSim, q: np.ndarray) -> dict[str, Any]:
    """Compare CPU/device masks and v8 result on a copied state and fixed Q rows."""
    device = DeviceSim.from_reference(sim, device="cpu")
    rows_np = np.column_stack((np.zeros(sim.S, dtype=np.int64), np.arange(sim.S, dtype=np.int64)))
    rows = torch.as_tensor(rows_np)
    cpu_mask = sim.get_advisory_action_mask()[0]
    device_mask = device.advisory_mask()[0].cpu().numpy()
    if not np.array_equal(cpu_mask, device_mask):
        return {"status": "MISMATCH", "field": "advisory_mask"}
    expected = cpu_v8(sim, rows_np, q, cpu_mask, features=True)
    actual = device_v8(
        device, rows, torch.as_tensor(q), torch.as_tensor(device_mask), features=True
    )
    if not np.array_equal(actual.action.cpu().numpy(), expected.action):
        return {"status": "MISMATCH", "field": "v8_action"}
    for name in ("base", "v5_action", "v7_action", "outcome", "counts", "area", "landing"):
        if not np.array_equal(getattr(actual, name).cpu().numpy(), getattr(expected, name)):
            return {"status": "MISMATCH", "field": f"v8_{name}"}
    return {"status": "PASS", "slots": sim.S, "claim": "common-state mask/v8"}


def vector_carry_common_state(sim: GridBatchSim) -> dict[str, Any]:
    """Check vector61 rows and their nearest-enemy carry on one copied state."""
    device = DeviceSim.from_reference(sim, device="cpu")
    params = replace(Vector61Params.from_game_config(sim.cfg), use_free_space=True)
    rows_np = np.column_stack((np.zeros(sim.S, dtype=np.int64), np.arange(sim.S, dtype=np.int64)))
    cpu = Vector61Featurizer(params)
    memory = Vector61Memory(1, sim.S)
    expected = cpu.featurize(sim, rows_np, memory, update_memory=True)
    runtime = DeviceVectorRuntime(params)
    runtime.bind(device)
    actual = runtime.features(device, update_memory=True).cpu().numpy()
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-6):
        return {"status": "MISMATCH", "field": "vector61_initial"}
    # Re-featurize without a reset.  This specifically exercises carry, rather
    # than accepting a stateless row-only comparison.
    expected = cpu.featurize(sim, rows_np, memory, update_memory=True)
    actual = runtime.features(device, update_memory=True).cpu().numpy()
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-6):
        return {"status": "MISMATCH", "field": "vector61_carry"}
    return {"status": "PASS", "slots": sim.S, "claim": "common-state vector carry"}


def distribution_screen_descriptor(
    *, worlds_per_mix: int = 32, horizon: int = 256, test_only: bool = False
) -> dict[str, Any]:
    """Describe the required pre-training simulator-D gross-error screen.

    Caller runs B8 on separate CPU and device fresh roots and records support,
    reset/food/collision rates and mass.  This deliberately contains no test
    outcomes and cannot be interpreted as a promotion comparison.
    """
    if (worlds_per_mix < 32 and not test_only) or worlds_per_mix < 1 or horizon not in (256, 5000):
        raise ValueError("screen requires E32 and a 256-frame prefix or complete H5000 lives")
    return {
        "schema": "sandbox-a-distribution-screen/v2",
        "task_horizon": 5000,
        "coverage": "complete first lives" if horizon == 5000 else "early-life prefix only",
        "test_only": test_only,
        "worlds_per_mix": worlds_per_mix,
        "horizon": horizon,
        "mixes": ["scripted", "frozen", "mixed"],
        "compare": ["spawn_spacing", "food_refill", "collisions", "respawns", "mass", "occupancy"],
        "status_meaning": "simulator-D engineering diagnostic only; not a gate",
    }


@torch.inference_mode()
def run_distribution_screen(
    *,
    cpu_factory: Callable[[str, int, int], Any],
    device_factory: Callable[[str, int, int], Any],
    cpu_actor: Any,
    device_actor: Any,
    worlds_per_mix: int = 32,
    horizon: int = 256,
    mixes: Sequence[str] = ("scripted", "frozen", "mixed"),
    test_only: bool = False,
    on_progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the required pre-training CPU/device gross-error screen.

    Factories own fresh domain-separated roots. The result reports separate
    distributions and is explicitly simulator-D engineering evidence only.
    """
    descriptor = distribution_screen_descriptor(
        worlds_per_mix=worlds_per_mix, horizon=horizon, test_only=test_only
    )
    if (
        not mixes
        or len(set(mixes)) != len(mixes)
        or any(mix not in ("scripted", "frozen", "mixed") for mix in mixes)
    ):
        raise ValueError("known unique screen mixes required")
    if not test_only and tuple(mixes) != ("scripted", "frozen", "mixed"):
        raise ValueError("production screen requires all three mixes")

    def progress(stage: str, **values: Any) -> None:
        print(f"DISTRIBUTION_SCREEN stage={stage} {values}", flush=True)
        if on_progress is not None:
            on_progress(f"distribution-screen-{stage}", values)

    def one(factory: Callable[[str, int, int], Any], actor: Any, mix: str) -> dict[str, Any]:
        progress("start", mix=mix, engine=str(factory), frames=horizon)
        # A short diagnostic is a prefix of the natural H5000 task, never a
        # shortened episode/reset or a claim about late-life distributions.
        task = factory(mix, worlds_per_mix, 5000)
        last_progress = time.monotonic()
        # The screen reads terminal-frame causes and opponent state after
        # step().  Its single-life contract therefore forbids resident
        # automatic replacement worlds.
        if getattr(getattr(task, "config", None), "auto_reset", False):
            raise ValueError("distribution screen requires auto_reset=False")
        parameters = getattr(actor, "parameters", None)
        actor_device = (
            next(iter(parameters()), torch.empty(0)).device
            if callable(parameters)
            else torch.device("cpu")
        )
        sim = getattr(task, "sim", None)
        if sim is None:
            sim = task._shard.sim

        def tensor(value: Any) -> torch.Tensor:
            return (
                torch.as_tensor(value, device="cpu").clone()
                if not isinstance(value, torch.Tensor)
                else value.detach().cpu()
            )

        def food_points() -> torch.Tensor:
            raw = sim.food_cells
            if isinstance(raw, torch.Tensor):
                return raw[sim.food_present].detach().cpu().float()
            return torch.tensor(
                [point for world in raw for point in world], dtype=torch.float32
            ).reshape(-1, 2)

        task.observe()
        heads = tensor(sim.bodies)[
            torch.arange(worlds_per_mix)[:, None], torch.arange(sim.S)[None], tensor(sim.head_ptr)
        ]
        distance = (heads[:, :, None] - heads[:, None]).float().square().sum(-1).sqrt()
        distance.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
        food = food_points()
        if food.shape[0] < 2 or not torch.isfinite(food).all():
            raise RuntimeError("distribution screen has invalid initial food support")
        correlation = float(torch.corrcoef(food.T)[0, 1])
        initial = {
            "nearest_spawn_mean_cells": float(distance.min(-1).values.mean()),
            "headings": torch.bincount(tensor(sim.direction).reshape(-1), minlength=4).tolist(),
            "initial_food_mean": float(len(food) / worlds_per_mix),
            "food_xy_correlation": correlation,
        }
        mass = torch.zeros(worlds_per_mix)
        alive_ticks = torch.zeros_like(mass)
        first_life = torch.ones(worlds_per_mix, dtype=torch.bool)
        length_bins = torch.zeros(5, dtype=torch.long)
        causes: dict[str, int] = {}
        deaths = resets = respawns = food_eaten = 0
        previous_alive = tensor(sim.alive).bool()
        for tick in range(horizon):
            local, glob, scalars, legal = task.observe()
            # The CPU reference deliberately keeps its simulator and features
            # on CPU, but may use the allocated GPU for B8 inference.  The
            # transfer is per tensor and does not alter CPU world mechanics.
            local, glob, scalars, legal = (
                local.to(actor_device),
                glob.to(actor_device),
                scalars.to(actor_device),
                legal.to(actor_device),
            )
            logits = actor(local, glob, scalars)
            if isinstance(logits, tuple):
                logits = logits[0]
            fallback = torch.tensor((True, True, True, False, False, False), device=legal.device)
            legal = torch.where(legal.any(1, keepdim=True), legal, fallback)
            event = task.step(logits.masked_fill(~legal, -torch.inf).argmax(1))
            logical_mass = tensor(event.mass).float()
            terminal = tensor(event.terminal).bool() & first_life
            alive = tensor(sim.alive).bool()
            mass += logical_mass * first_life
            alive_ticks += first_life & ~terminal
            bins = (logical_mass / 200).long().clamp(0, 4)
            length_bins += torch.bincount(bins[first_life & ~terminal], minlength=5)
            deaths += int(terminal.sum())
            resets += int((tensor(event.reset).bool() & first_life).sum())
            for cause in tensor(sim._last_death_cause)[:, 0][terminal].tolist():
                causes[str(cause)] = causes.get(str(cause), 0) + 1
            # A task is permitted to reset immediately after reporting a hero
            # terminal.  Retire that world's first life before observing any
            # post-step opponent state, otherwise a reset frame can turn old
            # dead opponents into spurious respawns (and food into a new-life
            # observation) for the completed world.
            first_life &= ~terminal
            respawns += int(((~previous_alive[:, 1:]) & alive[:, 1:] & first_life[:, None]).sum())
            food_eaten += int((tensor(sim._last_food_ate).bool() & first_life[:, None]).sum())
            previous_alive = alive
            if not torch.isfinite(mass).all():
                raise RuntimeError("nonfinite distribution-screen mass")
            now = time.monotonic()
            if (tick + 1) % 256 == 0 or now - last_progress >= 60:
                progress(
                    "advance", mix=mix, engine=type(task).__name__, frame=tick + 1, horizon=horizon
                )
                last_progress = now
            # H5000 is the registered maximum horizon, not work that has to be
            # done after every first-life hero has terminated.  All remaining
            # contributions are identically zero and neither simulator is
            # advanced further.
            if not bool(first_life.any()):
                break
        progress("complete", mix=mix, engine=type(task).__name__, frame=tick + 1, horizon=horizon)
        return {
            **initial,
            "simulated_frames": tick + 1,
            "mean_mass_integral": float(mass.mean() / horizon),
            "alive_fraction": float(alive_ticks.mean() / horizon),
            "no_death_fraction": float(first_life.float().mean()),
            "death_events": deaths,
            "reset_events": resets,
            "opponent_respawns": respawns,
            "food_consumption_events": food_eaten,
            "hero_death_causes": causes,
            "alive_length_occupancy_bins_200": length_bins.tolist(),
        }

    results = {
        mix: {
            "cpu": one(cpu_factory, cpu_actor, mix),
            "device": one(device_factory, device_actor, mix),
        }
        for mix in mixes
    }
    gaps = []
    for mix, pair in results.items():
        for name in (
            "nearest_spawn_mean_cells",
            "initial_food_mean",
            "mean_mass_integral",
            "alive_fraction",
        ):
            cpu, gpu = pair["cpu"][name], pair["device"][name]
            if abs(gpu - cpu) > max(abs(cpu), 1e-8) * 0.25:
                gaps.append({"mix": mix, "metric": name, "cpu": cpu, "device": gpu})
    progress("compared", mixes=list(mixes), material_gap_count=len(gaps))
    return {
        **descriptor,
        "status": "COMPLETE_SIMULATOR_D_SCREEN",
        "results": results,
        "material_gap_rule": "descriptive relative difference above 25 percent; "
        "not an efficacy gate",
        "material_gaps": gaps,
        "interpretation": "simulator-D pretraining; CPU reset rejection and counter-based device "
        "sampling use different random streams. Material gaps lower transfer confidence. "
        "This screen "
        "checks gross support and records task behavior; it does not certify distributional or "
        "closed-loop identity.",
    }
