"""Held-out development probe for the ratified fit's epoch selection/tripwires.

``probe_factory`` returns ``hook(student_model, epoch) -> ProbeResult`` exactly
as ``research.t34_exit_b8_fit.diagnostics`` expects: fresh development worlds
(never gate data), each played to completion (death or H) by the current
student with the frozen v8 opponents, recording at every decision the student's
action, the *previous* student's action on the same observation, and the hero
length.  ``survival`` is the fraction of worlds alive at H.

Production shape is 32 worlds x H5000 (validated by the ratified
``validate_probe``).  Smaller shapes exist only for local smoke wiring and are
stamped as such by :mod:`research.t34_claim_a.pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.t34_exit_b8_fit.diagnostics import ProbeResult

PROBE_FACTORY = "research.t34_claim_a.probe:probe_factory"
PRODUCTION_WORLDS = 32
PRODUCTION_HORIZON = 5000
_LAST: list["DevProbe"] = []


def _q(model: torch.nn.Module, obs: Mapping[str, np.ndarray], device: torch.device) -> np.ndarray:
    from research.t34_claim_a.student import full_fp32, needs_full_sight
    from src.model.ego2s_network import obs_tensors

    tensors = obs_tensors(obs, device)
    with torch.no_grad(), full_fp32():
        if needs_full_sight(model):
            sight = torch.from_numpy(np.ascontiguousarray(obs["full_sight"])).to(device)
            values = model(*tensors, sight)
        else:
            values = model(*tensors)
    return values.float().cpu().numpy().astype(np.float32)


@dataclass
class DevProbe:
    """Lockstep student rollouts over a fixed development world bank."""

    carrier_checkpoint: Path
    previous_model: torch.nn.Module
    rows: Sequence[Mapping[str, Any]]
    horizon: int = PRODUCTION_HORIZON
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1 <= int(self.horizon) <= PRODUCTION_HORIZON:
            raise ValueError("probe horizon must lie in [1, 5000]")
        ids = [str(row["world_id"]) for row in self.rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("probe needs unique development world ids")
        if any("/probe" not in world_id for world_id in ids):
            raise ValueError("probe worlds must come from a dedicated probe namespace")

    def __call__(self, model: torch.nn.Module, epoch: int) -> ProbeResult:
        from research.t34_claim_a.student import _evaluation, full_sight_planes, needs_full_sight
        from research.t34_exit_b8.policy import B8Policy, action_grid
        from research.t34_exit_b8.trajectory import build_world, prepare_world
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        started = perf_counter()
        was_training = model.training
        model.eval()
        self.previous_model.eval()
        device = next(model.parameters()).device
        sight = needs_full_sight(model) or needs_full_sight(self.previous_model)
        dispatcher = B8Policy(device="cpu")
        worlds = [build_world(dict(row), self.carrier_checkpoint) for row in self.rows]
        traces = [{"previous_actions": [], "actions": [], "lengths": []} for _ in worlds]
        alive = [True] * len(worlds)
        root = np.asarray([[0, 0]], dtype=np.int64)
        forward_seconds = 0.0
        for _step in range(int(self.horizon)):
            active = [index for index, flag in enumerate(alive) if flag]
            if not active:
                break
            parts = []
            for index in active:
                prepare_world(worlds[index])
                obs = dict(
                    build_ego_raster(
                        worlds[index].sim,
                        EgoRasterConfig(version="bv8"),
                        backend="numba",
                        rows=root,
                    )
                )
                if sight:
                    obs["full_sight"] = full_sight_planes(worlds[index].sim, root)
                parts.append(obs)
            batch = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
            tick = perf_counter()
            current = _q(model, batch, device)
            previous = _q(self.previous_model, batch, device)
            forward_seconds += perf_counter() - tick
            for slot, index in enumerate(active):
                world = worlds[index]
                action = _evaluation(world.sim, current[slot]).greedy_action()
                prior = _evaluation(world.sim, previous[slot]).greedy_action()
                traces[index]["actions"].append(action)
                traces[index]["previous_actions"].append(prior)
                traces[index]["lengths"].append(int(world.sim.length[0, 0]))
                grid = action_grid(world, action, cache=dispatcher)
                world.sim.step(grid)
                world.runtime.observe_step(world.sim)
                if not bool(world.sim.alive[0, 0]):
                    alive[index] = False
        survived = [bool(flag) for flag in alive]
        trajectories = [
            {
                "world_id": str(row["world_id"]),
                "mix": str(row.get("mix", "")),
                "horizon": int(self.horizon),
                "complete": True,
                "survived": survived[index],
                **traces[index],
            }
            for index, row in enumerate(self.rows)
        ]
        survival = float(np.mean(survived))
        self.calls.append(
            {
                "epoch": int(epoch),
                "seconds": perf_counter() - started,
                "forward_seconds": forward_seconds,
                "decisions": int(sum(len(t["actions"]) for t in traces)),
                "survival": survival,
                "worlds": len(worlds),
                "horizon": int(self.horizon),
            }
        )
        if was_training:
            model.train()
        return ProbeResult(survival=survival, trajectories=trajectories, is_gate_data=False)


def probe_rows(namespace: str, worlds: int, mixes: Sequence[str] | None = None) -> list[dict]:
    """Development world bank from the ratified prep-world generator."""
    from research.t34_exit_b8.trajectory import make_world_rows

    if "/probe" not in namespace:
        raise ValueError("probe namespace must contain '/probe' (never a label or gate bank)")
    config: dict[str, Any] = {"namespace": namespace}
    if mixes:
        config["mixes"] = tuple(mixes)
    return make_world_rows(config, int(worlds))


def probe_factory(*, binding: Mapping[str, Any], previous_model: torch.nn.Module) -> DevProbe:
    """``module:function`` named by ``binding['probe_factory']``."""
    settings = dict(binding.get("claim_a") or {})
    probe = dict(settings.get("probe") or {})
    carrier = settings.get("carrier_checkpoint")
    if not carrier:
        raise ValueError("binding.claim_a.carrier_checkpoint is required for the probe")
    namespace = str(probe.get("namespace") or "")
    if not namespace:
        raise ValueError("binding.claim_a.probe.namespace must be preregistered")
    rows = probe_rows(
        namespace, int(probe.get("worlds", PRODUCTION_WORLDS)), probe.get("mixes") or None
    )
    hook = DevProbe(
        carrier_checkpoint=Path(str(carrier)),
        previous_model=previous_model,
        rows=rows,
        horizon=int(probe.get("horizon", PRODUCTION_HORIZON)),
    )
    _LAST.append(hook)
    return hook


def last_probe() -> DevProbe | None:
    return _LAST[-1] if _LAST else None


def validate_smoke_probe(probe: ProbeResult, *, worlds: int, horizon: int) -> dict[str, Any]:
    """The ratified ``validate_probe`` checks at a declared non-production shape.

    Used only by the local smoke (stamped ``SMOKE_PROBE_SHAPE``); production
    runs keep the ratified 32-world / H5000 validator untouched.
    """
    from research.t34_exit_b8_fit.diagnostics import trajectory_diagnostics

    if probe.is_gate_data:
        raise ValueError("selection probe must never use gate data")
    if len(probe.trajectories) != worlds:
        raise ValueError(f"required {worlds} smoke probe worlds, got {len(probe.trajectories)}")
    old, new, length = [], [], []
    for record in probe.trajectories:
        if int(record.get("horizon", 0)) != horizon or not bool(record.get("complete", False)):
            raise ValueError("smoke probe requires complete trajectories at its declared horizon")
        old.extend(record["previous_actions"])
        new.extend(record["actions"])
        length.extend(record["lengths"])
    return {
        "survival": float(probe.survival),
        "smoke_probe_shape": {"worlds": worlds, "horizon": horizon},
        **trajectory_diagnostics(
            previous_actions=np.asarray(old), actions=np.asarray(new), lengths=np.asarray(length)
        ),
    }
