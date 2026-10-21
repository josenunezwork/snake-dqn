"""Real T3 batch adapter for the frozen constructed Watch6 ruler fixtures."""

from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_served_search import ceiling, ruler
from research.bl_served_search.fixture_geometry import install
from research.bl_served_search.ruler_timing import STAGES, StageTimer
from research.redesign_scope_20261007.grid_h5000_identity import FRP3_S12
from research.t1_commitment_20261015.replay import configure_cpu
from research.t3_rollout_labels_20261015.bank_engine import build_world
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from src.simd_env.vector61_policy import Vector61SimdPolicy

CARRY_FIELDS = ("live_vetoes", "veto_counters", "boost_counters", "last_veto")


def _jsonable(value: Any) -> Any:
    """Serialize mutable numerical/runtime state without object addresses."""
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": _jsonable(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, float) and not math.isfinite(value):
        return {"float_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {"type": type(value).__qualname__, "state": _jsonable(vars(value))}
    raise TypeError(f"unsupported frozen runtime state: {type(value).__qualname__}")


def _clone_root(world: Any) -> Any:
    """Preserve before-selector phase and isolate all mutable v8 carry."""
    result = world.clone()
    result.sim._policy_phase_prepared = bool(getattr(world.sim, "_policy_phase_prepared", False))
    result.sim._policy_step_in_progress = False
    for original, cloned in zip([world.hero, *world.opponents], [result.hero, *result.opponents]):
        for name in CARRY_FIELDS:
            if hasattr(original, name):
                setattr(cloned, name, copy.deepcopy(getattr(original, name)))
        if hasattr(cloned, "_rule_config_checked"):
            delattr(cloned, "_rule_config_checked")
    return result


def _root_bytes(world: Any) -> bytes:
    """Bind bodies, ordered food, RNG, counters, phase and opponent carry."""
    sim = world.sim
    small_arrays = {
        name: _jsonable(value)
        for name, value in vars(sim).items()
        if isinstance(value, np.ndarray) and value.size <= sim.S * 20
    }
    value = {
        "bodies": [sim.get_bodies(0, slot) for slot in range(sim.S)],
        "food": sim.get_food(0),
        "corpse_food": sim.get_corpse_food(0),
        "rng": [_jsonable(getattr(item, "_rng", item).getstate()) for item in sim._rngs],
        "counters_and_masks": small_arrays,
        "phase_prepared": bool(getattr(sim, "_policy_phase_prepared", False)),
        "opponent_carry": {
            "runtime": {
                key: _jsonable(value)
                for key, value in vars(world.runtime).items()
                if key == "memory" or isinstance(value, np.ndarray)
            },
            "policies": [
                {
                    name: _jsonable(getattr(policy, name))
                    for name in CARRY_FIELDS
                    if hasattr(policy, name)
                }
                for policy in world.opponents
            ],
        },
        "opponent_veto": "FRP3-s12+v8(lambda=8); scripted slots unchanged",
    }
    return ruler.canonical(_jsonable(value)).encode()


def _fixture_plan(plan: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(plan)
    for index, spec in enumerate(result["specs"]):
        if spec[0] == "checkpoint":
            spec[1] = FRP3_S12[0]
            result["member_sha256s"][index] = FRP3_S12[1]
    return result


class ExactT3Operator:
    """Use the same dense operator on both hosts; timing fixtures stay separate."""

    def __init__(self, native_mode: str = "stock") -> None:
        self.native_mode = native_mode
        self._roots: dict[str, Any] = {}

    def _world(self, row: dict[str, Any]) -> Any:
        checkpoint = os.environ.get("RULER_B8_PATH", row.get("b8_path", ""))
        path = Path(checkpoint)
        if not path.is_file() or ceiling.digest_file(path) != ceiling.B8_SHA256:
            raise ValueError("ruler requires the pinned uploaded B8 checkpoint")
        checkpoint_root = Path(
            os.environ.get("SNAKE_CKPT_DIR", row.get("checkpoint_dir", "saved_snakes"))
        )
        plan = ceiling.portable_plan(row["plan"], checkpoint_root)
        for spec, checksum in zip(plan["specs"], plan["member_sha256s"]):
            if spec[0] == "checkpoint" and ceiling.digest_file(Path(spec[1])) != checksum:
                raise ValueError("ruler opponent checkpoint drift")
        configure_cpu()
        world = build_world("b8_s3", plan, hero_checkpoint=path)
        seen = set()
        for policy in world.opponents:
            if isinstance(policy, Vector61SimdPolicy) and id(policy) not in seen:
                seen.add(id(policy))
                policy._configure_veto(tuple(range(1, 6)), "v8", 8.0, None)
        return world

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        request = {**request, "plan": _fixture_plan(request["plan"])}
        world = self._world(request)
        geometry = install(world, int(request["target_length"]))
        raw = _root_bytes(world)
        fixture_id = str(request["fixture_id"])
        self._roots[fixture_id] = _clone_root(world)
        return {
            "fixture_id": fixture_id,
            "mix": str(request["mix"]),
            "target_length": int(request["target_length"]),
            "root_frame": int(world.sim.frame[0]),
            "root_length": int(world.sim.length[0, 0]),
            "legal_actions": np.flatnonzero(world.sim.get_advisory_action_mask()[0, 0]).tolist(),
            "roster_sha256": hashlib.sha256(ruler.canonical(request["plan"]).encode()).hexdigest(),
            "checkpoint_sha256": ceiling.B8_SHA256,
            "plan": request["plan"],
            "b8_path": Path(request["b8_path"]).name,
            "construction": geometry,
            "root_bytes": raw,
            "opponent_veto": "v8",
            "opponent_veto_lambda": 8.0,
        }

    def _restore(self, fixture: dict[str, Any]) -> Any:
        fixture_id = str(fixture["fixture_id"])
        if fixture_id not in self._roots:
            world = self._world(fixture)
            install(world, int(fixture["target_length"]))
            if hashlib.sha256(_root_bytes(world)).hexdigest() != fixture["root_sha256"]:
                raise RuntimeError("complete frozen root reconstruction digest mismatch")
            self._roots[fixture_id] = _clone_root(world)
        root = _clone_root(self._roots[fixture_id])
        if hashlib.sha256(_root_bytes(root)).hexdigest() != fixture["root_sha256"]:
            raise RuntimeError("cached frozen root state drift")
        return root

    def batch(self, fixture: dict[str, Any]) -> dict[str, Any]:
        from research.bl_k1.native import RowStats, configure

        stats = RowStats()
        with configure(self.native_mode, stats), StageTimer() as timer:
            with timer.measure("clone_digest"):
                root = self._restore(fixture)
            result = ceiling._step(
                root,
                int(fixture["root_frame"]),
                DenseTrigger(min_length=600, interval=5),
                dense=True,
                horizon=int(fixture["root_frame"]) + ruler.HORIZON,
            )
        if not result.get("fired"):
            raise RuntimeError("ruler root did not fire the unchanged dense trigger")
        outcomes = result["outcomes"]
        semantic = {
            "fixture_id": fixture["fixture_id"],
            "bare": int(result["bare"]),
            "chosen": int(result["chosen"]),
            "fired": True,
            "outcomes": {str(key): value for key, value in sorted(outcomes.items())},
            "rollout_branch_frames": sum(
                int(value.get("frames_executed", value.get("frames_survived", 300)))
                for value in outcomes.values()
            ),
        }
        stages = {key: float(timer.seconds[key]) for key in STAGES}
        stages.update(submit_visible=0.0, submit_consumed=0.0)
        return {
            "semantic": semantic,
            "stage_seconds": stages,
            "native_v8": {"mode": self.native_mode, **stats.as_dict()},
        }


def create_operator() -> ExactT3Operator:
    """Factory shared by the Mac and qualified L4 runtime."""
    return ExactT3Operator()
