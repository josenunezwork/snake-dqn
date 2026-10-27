"""Source-closed CPU roster reference for the simulator-D distribution screen.

This small sandbox-local adapter does not import the historical ``bl_ppo``
collector.  It catches roster/reset regressions before the separately seeded
GPU H5000 proxy screen; it does not establish CPU/GPU closed-loop identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.sandbox_a_resident.training_rosters import (
    MIX_IDS,
    TrainingRoster,
    proxy_rosters,
    vector61_params,
)

_ROOT = "a100-dev/sandbox-a-train01"
_ROOT_OFFSET = 600_000_000


def _seed(mix: str, world: int, episode: int = 0) -> int:
    """Public root: ``600000000 + mixOrdinal*32 + world`` for first lives."""
    if mix not in MIX_IDS or not 0 <= world < 32 or episode < 0:
        raise ValueError("CPU roots require a known mix, world in [0, 32), and nonnegative life")
    index = _ROOT_OFFSET + MIX_IDS[mix] * 32 + world + episode * 96
    return int.from_bytes(hashlib.sha256(f"{_ROOT}/{index}".encode()).digest()[:8], "big")


def _checkpoint_path(reference: str) -> str:
    """Resolve only installed bytes matching the immutable roster digest."""
    root = Path(__file__).resolve().parents[3] / "snake-dqn" / "saved_snakes"
    for path in sorted(root.glob("*.pth")):
        if hashlib.sha256(path.read_bytes()).hexdigest() == reference:
            return str(path)
    raise FileNotFoundError(f"installed checkpoint with digest {reference} is unavailable")


def _prepare_watch_phase(sim: Any, active: np.ndarray) -> None:
    """Prepare exactly the active rows for one public Watch decision seam."""
    if getattr(sim, "_policy_phase_prepared", False):
        return
    sim._active_env_mask = active
    try:
        sim.frame[active] += 1
        sim._maintain_food()
        if sim.allow_respawn:
            sim._respawn_dead()
        sim._refresh_action_masks(active)
        sim._policy_phase_prepared = True
    finally:
        del sim._active_env_mask


class _CpuRosterRuntime:
    """Local Vector61/v8 and scripted-opponent grouping over GridBatchSim."""

    def __init__(
        self,
        sim: Any,
        rosters: Sequence[TrainingRoster],
        seeds: Sequence[int],
        checkpoint_paths: Mapping[str, str],
        inference_device: torch.device | str = "cpu",
    ) -> None:
        from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
        from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

        if len(rosters) != sim.E or len(seeds) != sim.E:
            raise ValueError("CPU roster cardinality must match simulator environments")
        controlled = np.zeros((sim.E, sim.S), dtype=bool)
        vectors: dict[tuple[str, int], list[tuple[int, int]]] = {}
        scripted: dict[str, list[tuple[int, int]]] = {}
        for env, roster in enumerate(rosters):
            for slot, (kind, ref) in enumerate(roster.slots, start=1):
                if kind == "checkpoint":
                    controlled[env, slot] = True
                    vectors.setdefault((ref, slot), []).append((env, slot))
                else:
                    scripted.setdefault(ref, []).append((env, slot))
        params = vector61_params(sim.cfg)
        self.inference_device = torch.device(inference_device)
        self.runtime = Vector61Runtime(params=params, backend="numba") if controlled.any() else None
        if self.runtime is not None:
            self.runtime.bind(sim, controlled)
        self.scripted_rows = {
            key: np.asarray(rows, dtype=np.int64) for key, rows in scripted.items()
        }
        self.scripted = {key: FastProfileAnchorSimdPolicy(key, seeds) for key in scripted}
        self.vector_rows = {key: np.asarray(rows, dtype=np.int64) for key, rows in vectors.items()}
        if any(reference not in checkpoint_paths for reference, _slot in vectors):
            raise KeyError("CPU roster requires explicit paths for every frozen checkpoint")
        self.vector = {
            key: Vector61SimdPolicy(
                checkpoint_paths[key[0]],
                self.runtime,
                veto_slots=(key[1],),
                veto_variant="v8",
                veto_lambda=8.0,
                # This diagnostic compares distributions, not byte-exact
                # trajectories.  GPU batches remove up to 160 tiny CPU
                # checkpoint forwards per tick while CPU remains the parity
                # reference's default call shape.
                forward="batched" if self.inference_device.type == "cuda" else "rowwise",
            )
            for key in vectors
        }
        # CPU world dynamics/features stay untouched.  Only frozen-network
        # inference may move to the allocated accelerator.
        for policy in self.vector.values():
            policy.network.to(self.inference_device)
            policy.device = self.inference_device

    def actions(self, sim: Any, grid: np.ndarray, active: np.ndarray) -> None:
        masks, alive = sim.get_resolved_action_mask(), sim.get_alive()
        if self.runtime is not None:
            self.runtime.prepare(sim, active)
        for ref, rows in self.scripted_rows.items():
            rows = rows[active[rows[:, 0]] & alive[rows[:, 0], rows[:, 1]]]
            if len(rows):
                grid[rows[:, 0], rows[:, 1]] = self.scripted[ref].actions(
                    masks[rows[:, 0], rows[:, 1]], sim, rows
                )
        for key, rows in self.vector_rows.items():
            rows = rows[active[rows[:, 0]] & alive[rows[:, 0], rows[:, 1]]]
            if len(rows):
                grid[rows[:, 0], rows[:, 1]] = self.vector[key].actions(
                    masks[rows[:, 0], rows[:, 1]], sim, rows
                )

    def after_step(self, sim: Any, active: np.ndarray) -> None:
        if self.runtime is not None:
            self.runtime.observe_step(sim, active)

    def reset(self, mask: np.ndarray, seeds: Sequence[int]) -> None:
        """Clear vector carry only on reset worlds; other rows remain carried."""
        from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

        self.scripted = {key: FastProfileAnchorSimdPolicy(key, seeds) for key in self.scripted}
        if self.runtime is None:
            return
        mask = np.asarray(mask, dtype=bool)
        self.runtime.memory.reset_where(
            np.broadcast_to(mask[:, None], self.runtime.controlled.shape)
        )
        for name in ("_carry_valid", "sel_ready", "sel_fresh", "_alive_post"):
            getattr(self.runtime, name)[mask] = False
        for policy in self.vector.values():
            for env in np.flatnonzero(mask):
                policy.veto_counters.pop(int(env), None)
                policy.boost_counters.pop(int(env), None)
                policy.live_vetoes.pop(int(env), None)


@dataclass
class _CpuShard:
    mix: str
    rosters: tuple[TrainingRoster, ...]
    horizon: int
    checkpoint_paths: Mapping[str, str]
    inference_device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        from src.simd_env.batch_sim import BatchSimConfig
        from src.simd_env.grid_sim import GridBatchSim

        worlds = len(self.rosters)
        self.seeds = [_seed(self.mix, index) for index in range(worlds)]
        self.sim = GridBatchSim(
            BatchSimConfig(
                num_envs=worlds,
                num_snakes=6,
                mechanics_version=1,
                max_capacity=10_400,
                body_storage_capacity=10_400,
            ),
            seeds=self.seeds,
            train_mode=False,
            allow_respawn=True,
        )
        self.runtime = _CpuRosterRuntime(
            self.sim,
            self.rosters,
            self.seeds,
            self.checkpoint_paths,
            self.inference_device,
        )
        self.age = np.zeros(worlds, dtype=np.int64)
        self.active = np.ones(worlds, dtype=bool)

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        _prepare_watch_phase(self.sim, self.active)
        # Retired first-life worlds never receive another action.  Rasterising
        # them (and running their hero forward) dominated the CPU reference
        # screen after natural deaths, while the active-world subset preserves
        # every live world and its fixed H5000 semantics.
        active_worlds = np.flatnonzero(self.active)
        rows = np.column_stack((active_worlds, np.zeros(len(active_worlds), dtype=np.int64)))
        obs = build_ego_raster(self.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        output: list[torch.Tensor] = []
        for key in ("local", "global", "scalars"):
            source = torch.from_numpy(np.ascontiguousarray(obs[key]))
            full = torch.zeros((self.sim.E,) + tuple(source.shape[1:]), dtype=source.dtype)
            full[torch.from_numpy(active_worlds)] = source
            output.append(full)
        legal = torch.zeros((self.sim.E, 6), dtype=torch.bool)
        legal[:, :3] = True
        if len(active_worlds):
            legal[torch.from_numpy(active_worlds)] = torch.from_numpy(
                np.ascontiguousarray(self.sim.get_advisory_action_mask()[active_worlds, 0])
            )
        return tuple(output) + (legal,)

    def step(self, actions: torch.Tensor) -> Any:
        hero = actions.detach().cpu().numpy().astype(np.int64, copy=False).reshape(self.sim.E)
        grid = np.ones((self.sim.E, self.sim.S), dtype=np.int64)
        grid[:, 0] = hero
        self.runtime.actions(self.sim, grid, self.active)
        self.sim.step(grid, active_env_mask=self.active)
        self.runtime.after_step(self.sim, self.active)
        self.age[self.active] += 1
        terminal = (self.sim.get_done()[:, 0] | ~self.sim.get_alive()[:, 0]) & self.active
        cut = self.active & (self.age == self.horizon)
        mass = np.where(self.sim.get_alive()[:, 0], self.sim.length[:, 0], 0).astype(np.float32)
        # H5000 screen worlds are first-life only: a dead/cut hero freezes its
        # complete world, rather than silently creating another population.
        self.active &= ~(terminal | cut)
        return type(
            "CpuEvent",
            (),
            {
                "mass": torch.from_numpy(mass),
                "terminal": torch.from_numpy(terminal.copy()),
                "cut": torch.from_numpy(cut.copy()),
                "reset": torch.zeros_like(torch.from_numpy(terminal), dtype=torch.bool),
            },
        )()


class CpuRosterTask:
    """CPU adapter accepted by ``run_distribution_screen``."""

    def __init__(
        self,
        mix: str,
        worlds: int,
        horizon: int,
        *,
        checkpoint_paths: Mapping[str, str] | None = None,
        inference_device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        del seed  # Kept for legacy direct smoke callers; factory roots are fixed.
        if horizon != 5000 or not 1 <= worlds <= 32:
            raise ValueError("CPU distribution reference requires 1..32 worlds at H5000")
        self.device, self.mix = torch.device("cpu"), mix
        self.inference_device = torch.device(inference_device)
        if checkpoint_paths is None:
            # Direct local test helper only. The dispatched factory below requires
            # transferred paths, so a Colab runtime never discovers a Mac sibling.
            checkpoint_paths = {
                reference: _checkpoint_path(reference)
                for reference in {
                    key
                    for row in proxy_rosters(mix, worlds)
                    for _kind, key in row.slots
                    if _kind == "checkpoint"
                }
            }
        self._shard = _CpuShard(
            mix, proxy_rosters(mix, worlds), horizon, checkpoint_paths, self.inference_device
        )

    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._shard.observe()

    def step(self, actions: torch.Tensor) -> Any:
        return self._shard.step(actions)


def cpu_distribution_factory(
    checkpoint_paths: Mapping[str, str], *, inference_device: torch.device | str = "cpu"
):
    """Return the CPU callback accepted by the distribution screen."""
    return lambda mix, worlds, horizon: CpuRosterTask(
        mix, worlds, horizon, checkpoint_paths=checkpoint_paths, inference_device=inference_device
    )
