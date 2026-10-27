"""Natural H5000 task boundary for the sandbox GPU-resident PPO run.

This adapter is intentionally separate from A-3's timing assay.  It retains
the certified simulator bytes and supplies only sandbox-local reset, terminal
and mass-objective semantics.  Device and CPU RNG streams differ, so this is a
distributional adapter, not a claim of closed-loop CPU/GPU identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np
import torch

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide
from research.sandbox_a_resident.training_rosters import (
    DeviceRosterRuntime,
    TrainingRoster,
    proxy_rosters,
    roster_ids,
    training_rosters,
)
from research.sandbox_a_resident.training_sim import TrainingDeviceSim
from src.simd_env.batch_sim import BatchSimConfig


@dataclass(frozen=True)
class TrainingTaskConfig:
    """TRAIN-01's fixed world task, independent of PPO implementation."""

    worlds: int = 1024
    horizon: int = 5000
    max_capacity: int = 10400
    namespace: str = "a100-dev/sandbox-a-train01"
    allocation_offset: int = 0
    max_episodes: int = 100_000
    tiny: bool = False
    auto_reset: bool = True

    def __post_init__(self) -> None:
        if self.worlds < 3 or self.horizon != 5000 or self.max_capacity < 10400:
            raise ValueError("TRAIN-01 requires E>=3, H5000 and full 10400 body capacity")
        if (
            not self.namespace.startswith("a100-dev/")
            or self.allocation_offset < 0
            or self.max_episodes < 1
            or type(self.auto_reset) is not bool
        ):
            raise ValueError("sandbox task requires an a100-dev namespace")


class TaskStep(NamedTuple):
    """Transition accounting captured before automatic natural-world reset."""

    reward: torch.Tensor
    terminal: torch.Tensor
    cut: torch.Tensor
    reset: torch.Tensor
    mass: torch.Tensor


def _key(namespace: str, index: int) -> int:
    raw = hashlib.sha256(f"{namespace}/{index}".encode()).digest()[:8]
    value = int.from_bytes(raw, "big") & 0x7FFF_FFFF_FFFF_FFFF
    if value == 0:
        raise ValueError("canonical seed has forbidden effective low-63-bit zero")
    return value


def _full_key(namespace: str, index: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}/{index}".encode()).digest()[:8], "big")


_ENV_FIELDS = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "rng_counter",
    "rng_key",
    "food_cells",
    "food_present",
    "corpse_food",
    "food_count",
    "_last_reward",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
    "_boosted_this_step",
    "terminal_slots",
    "capacity_overflow",
)


def _row_view(sim: Any, rows: torch.Tensor) -> Any:
    """Mutable compact view with an explicit environment-field contract."""
    out = object.__new__(type(sim))
    out.__dict__ = sim.__dict__.copy()
    out.E = int(rows.numel())
    out.cfg = replace(sim.cfg, num_envs=out.E)
    for name in _ENV_FIELDS:
        setattr(out, name, getattr(sim, name).index_select(0, rows))
    r2_cache.inherit(out, sim)
    return out


def _copy_rows(target: Any, source: Any, rows: torch.Tensor) -> None:
    for name in _ENV_FIELDS:
        getattr(target, name).index_copy_(0, rows, getattr(source, name))


class ResidentTrainingTask:
    """Device-resident natural episodes: death or H5000 causes a fresh world."""

    def __init__(
        self, config: TrainingTaskConfig = TrainingTaskConfig(), *, device: str = "cpu"
    ) -> None:
        self.config, self.device = config, torch.device("cuda:0" if device == "cuda" else device)
        cfg = BatchSimConfig(
            num_envs=config.worlds,
            num_snakes=6,
            max_capacity=config.max_capacity,
            body_storage_capacity=config.max_capacity,
            mechanics_version=1,
            **(
                dict(game_width=120, game_height=120, initial_food=2, max_food=3)
                if config.tiny
                else {}
            ),
        )
        self.sim = TrainingDeviceSim(
            cfg,
            device=self.device,
            train_mode=False,
            allow_respawn=True,
            move_backend="indexed" if self.device.type == "cuda" else "reference",
        )
        self.sim.capacity_overflow = torch.zeros(
            (self.sim.E, self.sim.S), dtype=torch.bool, device=self.device
        )
        r2_cache.enable(self.sim)
        self.rosters: tuple[TrainingRoster, ...] = training_rosters(config.worlds)
        self.roster_id = roster_ids(self.rosters, self.device)
        self.episode = torch.zeros(config.worlds, dtype=torch.long, device=self.device)
        self.age = torch.zeros_like(self.episode)
        self._world = torch.arange(config.worlds, device=self.device)
        # Setup-only materialization of the finite SHA allocation.  Hot reset
        # uses only a tensor gather; it never calls a CPU RNG or policy.
        keys = np.empty((config.max_episodes, config.worlds), dtype=np.int64)
        for episode in range(config.max_episodes):
            start = config.allocation_offset + episode * config.worlds
            keys[episode] = [
                _key(config.namespace, start + world) for world in range(config.worlds)
            ]
        self._episode_keys = torch.as_tensor(keys, dtype=torch.long, device=self.device)
        self._pending_reset = torch.ones(config.worlds, dtype=torch.bool, device=self.device)
        self.opponents: Any | None = None
        self.reset()

    def _keys(self, rows: torch.Tensor) -> torch.Tensor:
        # Tensor-only counter PRF after the domain-separated SHA root.  Reset
        # frequency therefore cannot introduce a per-reset host policy/RNG path.
        episode = self.episode.index_select(0, rows)
        # Exact public allocation: a100-dev/sandbox-a-train01/<offset + episode*E + world>.
        return self._episode_keys[episode, rows]

    def _full_keys(self) -> list[int]:
        return [
            _full_key(
                self.config.namespace,
                self.config.allocation_offset + int(self.episode[index]) * self.sim.E + index,
            )
            for index in range(self.sim.E)
        ]

    def attach_opponents(self, runtime: Any) -> None:
        """Install the all-device roster runtime after verified checkpoint loading."""
        if getattr(runtime, "device", self.device) != self.device:
            raise ValueError("opponent runtime device differs from task")
        self.opponents = runtime
        runtime.reset(
            torch.ones(self.sim.E, dtype=torch.bool, device=self.device), self._full_keys()
        )

    def proxy_factory(self, source: DeviceRosterRuntime) -> Any:
        """Construct independent, paired H5000 device proxy tasks by mix."""
        frozen: dict[str, Any] = {}
        b8: Any | None = None
        for rows, policy, _random_slots in source._groups:
            b8 = policy.b8_network if b8 is None else b8
            roster = self.rosters[int(rows[0])]
            for (kind, reference), network in zip(roster.slots, policy.opponent_networks):
                if kind == "checkpoint":
                    frozen[reference] = network
        if b8 is None:
            raise ValueError("proxy factory needs an attached roster runtime")

        def make(mix: str, worlds: int, horizon: int, seed: int) -> "ResidentTrainingTask":
            if worlds < 1 or (horizon != 5000 and not self.config.tiny):
                raise ValueError("GPU proxy is H5000 except for a tiny test-only task")
            count = max(3, worlds)
            task = ResidentTrainingTask(
                TrainingTaskConfig(
                    worlds=count,
                    horizon=5000,
                    namespace=self.config.namespace,
                    allocation_offset=seed,
                    max_episodes=2,
                    tiny=self.config.tiny,
                    auto_reset=False,
                ),
                device=str(self.device),
            )
            selected = proxy_rosters(mix, worlds)
            if count > worlds:
                selected = selected + (selected[-1],) * (count - worlds)
            task.rosters = selected
            task.roster_id = roster_ids(selected, task.device)
            task.attach_opponents(
                DeviceRosterRuntime(task.sim, selected, b8_network=b8, checkpoint_networks=frozen)
            )
            return task

        return make

    @torch.inference_mode()
    def reset(
        self, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create actual fresh worlds via the engine's normal respawn sampler.

        This does not assign an artificial episode clock to a short body.  It
        deliberately leaves dead opponents eligible for normal per-frame
        respawn after the world begins, while slot zero is terminal on death.
        """
        if mask is None:
            mask = torch.ones(self.sim.E, dtype=torch.bool, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if mask.shape != (self.sim.E,):
            raise ValueError("reset mask must have shape (E,)")
        rows = mask.nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            return self.observe()
        for name in _ENV_FIELDS:
            getattr(self.sim, name)[rows] = 0
        self.sim.alive[rows] = False
        self.sim.length[rows] = 1
        self.sim._reward_prev_length[rows] = 1
        self.sim.rng_key.index_copy_(0, rows, self._keys(rows))
        self.sim.terminal_slots[rows] = False
        self.sim._prepared = False
        # This is the certified engine's complete bounded respawn sampler, not
        # A-3's adjacent-cell shortcut.  It also preserves its counter-RNG rules.
        local = _row_view(self.sim, rows)
        local.reset_fresh_worlds()
        _copy_rows(self.sim, local, rows)
        self.sim.terminal_slots[rows, 0] = True
        self.sim._prepared = False
        self.age[rows] = 0
        self._pending_reset[rows] = True
        if self.opponents is not None:
            self.opponents.reset(mask, self._full_keys())
        r2_cache.bump(self.sim)
        if bool(self.sim.capacity_overflow.any()):
            raise RuntimeError("body storage capacity overflow during reset")
        return self.observe()

    @torch.inference_mode()
    def observe(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the exact B8 bv8 tensors and pre-action hero legal mask."""
        self.sim.prepare()
        rows = torch.stack((self._world, torch.zeros_like(self._world)), dim=1)
        legal = self.sim.advisory_mask()[:, 0]
        v8 = (
            v8_decide(
                self.sim,
                rows,
                torch.zeros((self.sim.E, 6), device=self.device),
                legal,
                8.0,
                features=True,
            )
            .feature_planes()
            .reshape(self.sim.E, 30)
        )
        local, glob, scalars = build_observation(self.sim, rows, v8)
        return local, glob, scalars, legal

    @property
    def hero_length(self) -> torch.Tensor:
        """Current logical mass for proxy diagnostics; B8 scalar bytes stay untouched."""
        return self.sim.length[:, 0]

    @torch.inference_mode()
    def step(self, hero_actions: torch.Tensor) -> TaskStep:
        """Advance one frame and reset only after recording terminal/cut semantics."""
        hero_actions = torch.as_tensor(hero_actions, device=self.device, dtype=torch.long)
        if hero_actions.shape != (self.sim.E,):
            raise ValueError("hero actions must have shape (E,)")
        if self.opponents is None:
            actions = torch.ones((self.sim.E, self.sim.S), dtype=torch.long, device=self.device)
            actions[:, 0] = hero_actions
        else:
            actions = self.opponents.actions(self.sim, hero_actions)
        self._pending_reset.zero_()
        self.sim.step(actions)
        if self.opponents is not None:
            self.opponents.after_step(self.sim)
        terminal = self.sim._last_done[:, 0].clone()
        self.age += 1
        cut = self.age.eq(self.config.horizon) & ~terminal
        mass = torch.where(
            self.sim.alive[:, 0],
            self.sim.length[:, 0].float(),
            torch.zeros_like(self.age, dtype=torch.float),
        )
        reward = mass / 1000.0
        reset = terminal | cut
        self._pending_reset.copy_(reset)
        if bool(self.sim.capacity_overflow.any()):
            raise RuntimeError("body storage capacity overflow")
        if self.config.auto_reset and bool(reset.any()):
            self.episode += reset.long()
            self.reset(reset)
        return TaskStep(reward, terminal, cut, reset, mass)

    def state_dict(self) -> dict[str, Any]:
        """Complete mutable sandbox task state for a same-package resume."""
        sim = {
            name: getattr(self.sim, name).detach().clone()
            for name in (
                "bodies",
                "head_ptr",
                "seg_count",
                "length",
                "alive",
                "direction",
                "boost_frames",
                "frames_since_food",
                "respawn_timer",
                "_reward_prev_length",
                "frame",
                "rng_counter",
                "rng_key",
                "food_cells",
                "food_present",
                "corpse_food",
                "food_count",
                "_last_reward",
                "_last_done",
                "_last_death_cause",
                "_last_kills",
                "_last_food_ate",
                "_last_ambient_food_ate",
                "_last_corpse_food_ate",
                "_last_transition_valid",
                "_boosted_this_step",
                "terminal_slots",
                "capacity_overflow",
            )
        }
        return {
            "schema": "sandbox-a-training-task/v1",
            "sim": sim,
            "episode": self.episode.detach().clone(),
            "age": self.age.detach().clone(),
            "pending_reset": self._pending_reset.detach().clone(),
            # A snapshot can occur after collect's final observe().  In that
            # state prepare() has already advanced frame/food RNG for the next
            # policy decision; restoring it as false would advance it twice.
            "prepared": bool(self.sim._prepared),
            "rosters": tuple(row.identity for row in self.rosters),
            "opponents": None if self.opponents is None else self.opponents.state_dict(),
        }

    @torch.inference_mode()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "sandbox-a-training-task/v1":
            raise ValueError("foreign training task state")
        if tuple(state.get("rosters", ())) != tuple(row.identity for row in self.rosters):
            raise ValueError("training roster identity differs on resume")
        for name, value in state["sim"].items():
            target = getattr(self.sim, name)
            if target.shape != value.shape:
                raise ValueError(f"training task state shape mismatch: {name}")
            target.copy_(value.to(self.device))
        for name in ("episode", "age", "pending_reset"):
            getattr(self, "_pending_reset" if name == "pending_reset" else name).copy_(
                state[name].to(self.device)
            )
        if state["opponents"] is not None:
            if self.opponents is None:
                raise ValueError("resume needs attached opponent runtime")
            self.opponents.load_state_dict(state["opponents"])
        self.sim._prepared = bool(state.get("prepared", False))
        r2_cache.bump(self.sim)
