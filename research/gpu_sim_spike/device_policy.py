"""Full resident B8/vector61/v8/scripted policy loop and timed CPU diagnostic bridge."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Protocol

import numpy as np
import torch
from torch import nn

from research.gpu_sim_spike.anchors_device import choose_anchor, random_table
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide


def _masked_argmax(q: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    fallback = torch.arange(6, device=q.device)[None] < 3
    allowed = torch.where(legal.any(1, keepdim=True), legal, fallback)
    return q.masked_fill(~allowed, -torch.inf).argmax(1)


@dataclass
class DevicePolicy:
    """All mutable policy state belongs to this rollout branch batch."""

    device: torch.device
    b8_network: nn.Module
    opponent_networks: tuple[Any, ...] = ()
    anchor_kinds: tuple[str | None, ...] = ()
    anchor_tables: tuple[Any, ...] = ()
    vector_runtime: Any = None
    vector_prepared: bool = True
    reference_bridge: Any = None
    host_stages: dict[str, float] = field(default_factory=dict)
    port_limitations: tuple[str, ...] = ()
    v8_lambda: float = 8.0

    @classmethod
    def from_world(
        cls,
        world: Any,
        device: str | torch.device = "cpu",
        *,
        backend: str = "device",
        sim: Any = None,
    ) -> "DevicePolicy":
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda:0")
        source = getattr(world, "b8_network", None)
        if source is None:
            source = getattr(getattr(world, "hero", None), "net", None)
        if source is None:
            raise ValueError("DevicePolicy requires the frozen B8 network; no heuristic fallback")
        model = copy.deepcopy(source).to(target).eval()
        if backend not in ("device", "reference"):
            raise ValueError("backend must be device or reference")
        out = cls(target, model)
        if backend == "reference":
            out.reference_bridge = getattr(world, "reference_bridge", None)
            if out.reference_bridge is None:
                out.reference_bridge = CpuReferenceBridge.from_captured_world(world, model)
            out.port_limitations = ("reference policy backend retains BV8/v8/vector61 on host",)
            return out
        from research.gpu_sim_spike.device_sim import DeviceSim
        from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

        if sim is None:
            sim = DeviceSim.from_reference(world.sim, device=target)
        out.vector_runtime = DeviceVectorRuntime.from_reference(world.runtime, sim)
        nets, kinds, tables = [], [], []
        memo: dict[int, nn.Module] = {}
        for slot, policy in enumerate(world.opponents, start=1):
            net = getattr(policy, "network", None)
            kind = getattr(policy, "kind", None) if net is None else None
            if net is None and kind not in ("greedy_food", "random_safe"):
                raise TypeError(f"unported opponent type: {type(policy).__name__}")
            if net is not None:
                if id(net) not in memo:
                    memo[id(net)] = copy.deepcopy(net).to(target).eval()
                net = memo[id(net)]
            nets.append(net)
            kinds.append(kind)
            tables.append(random_table(world.seed, slot, target) if kind == "random_safe" else None)
        out.opponent_networks, out.anchor_kinds, out.anchor_tables = (
            tuple(nets),
            tuple(kinds),
            tuple(tables),
        )
        return out

    def repeat_interleave(self, repeats: int) -> "DevicePolicy":
        if repeats < 1:
            raise ValueError("repeats must be positive")
        return replace(
            self,
            vector_runtime=(
                self.vector_runtime.repeat_interleave(repeats)
                if self.vector_runtime is not None
                else None
            ),
            reference_bridge=(
                self.reference_bridge.repeat_interleave(repeats)
                if self.reference_bridge is not None
                else None
            ),
        )

    @torch.inference_mode()
    def evaluate(self, sim: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reference_bridge is not None:
            q, legal = self.reference_bridge.hero_q_and_legal(sim, self.device, self.host_stages)
            return q, legal, _masked_argmax(q, legal)
        rows = torch.stack(
            (
                torch.arange(sim.E, device=self.device),
                torch.zeros(sim.E, dtype=torch.long, device=self.device),
            ),
            dim=1,
        )
        legal = sim.advisory_mask()[:, 0]
        features = (
            v8_decide(
                sim,
                rows,
                torch.zeros((sim.E, 6), device=self.device),
                legal,
                self.v8_lambda,
                features=True,
            )
            .feature_planes()
            .reshape(sim.E, 30)
        )
        q = self.b8_network(*build_observation(sim, rows, features)).float()
        return q, legal, _masked_argmax(q, legal)

    @torch.inference_mode()
    def actions(self, sim: Any, hero_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.reference_bridge is not None:
            return self.reference_bridge.all_actions(
                sim, hero_actions, self.device, self.host_stages
            )
        if hero_actions is None:
            _, _, hero_actions = self.evaluate(sim)
        out = torch.ones((sim.E, sim.S), dtype=torch.long, device=self.device)
        out[:, 0] = hero_actions
        if not self.vector_prepared:
            self.vector_runtime.prepare(sim)
            self.vector_prepared = True
        for slot, (net, kind, table) in enumerate(
            zip(self.opponent_networks, self.anchor_kinds, self.anchor_tables), start=1
        ):
            if kind is not None:
                out[:, slot] = choose_anchor(sim, slot, kind, table)
                continue
            rows = torch.stack(
                (
                    torch.arange(sim.E, device=self.device),
                    torch.full((sim.E,), slot, dtype=torch.long, device=self.device),
                ),
                dim=1,
            )
            states, masks = self.vector_runtime.selection(rows)
            q = net(states).float()
            out[:, slot] = v8_decide(sim, rows, q, masks, self.v8_lambda).action
        return out

    def after_step(self, sim: Any) -> None:
        if self.reference_bridge is not None:
            self.reference_bridge.after_step(sim, self.host_stages)
        else:
            self.vector_runtime.observe_step(sim)
            self.vector_prepared = False


class ReferencePolicyBridge(Protocol):
    """Exact-host adapter used while a policy component remains unported.

    Implementations own one cloned CPU reference world per device environment.
    They must synchronize the ``DeviceSim`` snapshot before evaluating and use
    the project's existing ``build_ego_raster(... bv8)`` / v8 runtime rather
    than a reimplementation.  Returned tensors are already transferred once to
    ``device``.  Timing keys are accumulated, never reset, so repeat branches
    remain visible in benchmark receipts.
    """

    def repeat_interleave(self, repeats: int) -> "ReferencePolicyBridge": ...

    def hero_q_and_legal(
        self, sim: Any, device: torch.device, timings: dict[str, float]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def all_actions(
        self,
        sim: Any,
        hero_actions: Optional[torch.Tensor],
        device: torch.device,
        timings: dict[str, float],
    ) -> torch.Tensor: ...

    def after_step(self, sim: Any, timings: dict[str, float]) -> None: ...


@dataclass
class CpuReferenceBridge:
    """Exact CPU policy seam for a device rollout that has unported policy work.

    Each entry is a complete ``CapturedWorld`` clone, including its own runtime
    carry and v8 hook state.  The only traffic is an explicit snapshot copy at
    a decision seam; all such work is accumulated under ``host_*`` timing keys.
    That makes a run using this bridge an end-to-end *upper bound*, never an
    all-device claim.
    """

    worlds: list[Any]
    b8_network: nn.Module
    prepared: list[bool]

    @classmethod
    def from_captured_world(cls, source: Any, b8_network: nn.Module) -> "CpuReferenceBridge":
        from research.t34_shadow.shadow import clone_world_exact

        # Captured S0 roots are already at the selector seam.  Do not call
        # ``runtime.prepare`` again before their first decision: it advances
        # stateful nearest-enemy/v8 carry.
        return cls([clone_world_exact(source)], b8_network, [True])

    def repeat_interleave(self, repeats: int) -> "CpuReferenceBridge":
        if repeats < 1:
            raise ValueError("repeats must be positive")
        from research.t34_shadow.shadow import clone_world_exact

        return CpuReferenceBridge(
            [clone_world_exact(world) for world in self.worlds for _ in range(repeats)],
            self.b8_network,
            [ready for ready in self.prepared for _ in range(repeats)],
        )

    @staticmethod
    def _charge(timings: dict[str, float], key: str, started: float) -> None:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - started)

    def _world(self, sim: Any, env: int, timings: dict[str, float]) -> Any:
        if len(self.worlds) != int(sim.E):
            raise ValueError("reference bridge world count must match DeviceSim environments")
        started = time.perf_counter()
        sim.copy_env_to_reference(self.worlds[env].sim, env)
        self._charge(timings, "host_snapshot_sync_s", started)
        return self.worlds[env]

    def hero_q_and_legal(
        self, sim: Any, device: torch.device, timings: dict[str, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        qs: list[torch.Tensor] = []
        legal: list[torch.Tensor] = []
        for env in range(int(sim.E)):
            world = self._world(sim, env, timings)
            started = time.perf_counter()
            # The reference runtime owns its decision-time masks and v8 carry.
            # ``prepare`` does not advance a prepared snapshot; it refreshes
            # runtime selection state from the copied simulator fields.
            if not self.prepared[env]:
                world.runtime.prepare(world.sim)
                self.prepared[env] = True
            obs = build_ego_raster(
                world.sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            self._charge(timings, "host_bv8_observation_s", started)
            started = time.perf_counter()
            local = torch.from_numpy(np.ascontiguousarray(obs["local"])).to(device)
            glob = torch.from_numpy(np.ascontiguousarray(obs["global"])).to(device)
            scalars = torch.from_numpy(np.ascontiguousarray(obs["scalars"])).to(device)
            with torch.inference_mode():
                qs.append(self.b8_network(local, glob, scalars).float()[0])
            self._charge(timings, "host_bv8_transfer_and_forward_s", started)
            mask = np.asarray(world.sim.get_advisory_action_mask()[0, 0], dtype=bool).copy()
            if not mask.any():
                mask[:3] = True
            legal.append(torch.from_numpy(mask).to(device))
        return torch.stack(qs), torch.stack(legal)

    def all_actions(
        self,
        sim: Any,
        hero_actions: Optional[torch.Tensor],
        device: torch.device,
        timings: dict[str, float],
    ) -> torch.Tensor:
        from research.t34_exit_b8.policy import action_grid

        if hero_actions is None:
            _q, _legal = self.hero_q_and_legal(sim, device, timings)
            hero_actions = _masked_argmax(_q, _legal)
        heroes = hero_actions.detach().to("cpu", dtype=torch.long).tolist()
        grids = []
        for env, hero in enumerate(heroes):
            world = self._world(sim, env, timings)
            started = time.perf_counter()
            if not self.prepared[env]:
                world.runtime.prepare(world.sim)
                self.prepared[env] = True
            # action_grid invokes the original scripted/vector policies; v8 is
            # therefore exactly the source hook, with independent clone carry.
            grids.append(
                action_grid(world, int(hero), device="cpu" if device.type == "cpu" else str(device))
            )
            self._charge(timings, "host_opponent_dispatch_v8_s", started)
        started = time.perf_counter()
        result = torch.as_tensor(np.concatenate(grids, axis=0), dtype=torch.long, device=device)
        self._charge(timings, "host_action_transfer_s", started)
        return result

    def after_step(self, sim: Any, timings: dict[str, float]) -> None:
        for env in range(int(sim.E)):
            world = self._world(sim, env, timings)
            started = time.perf_counter()
            world.runtime.observe_step(world.sim)
            self.prepared[env] = False
            self._charge(timings, "host_runtime_carry_s", started)
