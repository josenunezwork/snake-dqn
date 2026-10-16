"""Compact, data-only boundary snapshots and deterministic actor-world grafting."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "t4-snapshot-store/v2"
# Views are reconstructed from the exact padded storage; helpers are lazy caches.
SIM_SKIP = {"cfg", "_owner", "_slot", "_food", "_det", "_kernels"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _encode(value: Any, arrays: dict) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {"object_array": _encode(value.tolist(), arrays), "shape": value.shape}
        key = f"a{len(arrays)}"
        arrays[key] = value
        return {"array": key}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, random.Random):
        return {"random": _encode(value.getstate(), arrays)}
    from src.simd_env.rng import EnvRng

    if isinstance(value, EnvRng):
        return {"env_rng": _encode(vars(value), arrays)}
    if isinstance(value, dict):
        return {"dict": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {type(value).__name__: [_encode(v, arrays) for v in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported snapshot value: {type(value)}")


def _decode(value: Any, arrays: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "array" in value:
        return arrays[value["array"]].copy()
    if "object_array" in value:
        out = np.empty(value["shape"], dtype=object)
        rows = _decode(value["object_array"], arrays)
        for idx in np.ndindex(out.shape):
            item = rows
            for i in idx:
                item = item[i]
            out[idx] = item
        return out
    if "random" in value:
        rng = random.Random()
        rng.setstate(_decode(value["random"], arrays))
        return rng
    if "env_rng" in value:
        from src.simd_env.rng import EnvRng

        rng = EnvRng.__new__(EnvRng)
        rng.__dict__.update(_decode(value["env_rng"], arrays))
        return rng
    if "dict" in value:
        return {_decode(k, arrays): _decode(v, arrays) for k, v in value["dict"]}
    for name, cls in (("list", list), ("tuple", tuple), ("set", set)):
        if name in value:
            return cls(_decode(v, arrays) for v in value[name])
    raise ValueError("unknown snapshot encoding")


def capture(sim: Any, runtime: Any, specs: list, provenance: dict | None = None) -> dict:
    """Capture E=1 immediately before Watch prepares the next decision."""
    if sim.E != 1 or getattr(sim, "_policy_phase_prepared", False):
        raise ValueError("snapshot must be an unprepared single-world boundary")
    if getattr(sim, "_policy_step_in_progress", False):
        raise ValueError("cannot snapshot during selection")
    if not sim.alive[0, 0]:
        raise ValueError("restart snapshot requires a living hero")
    from src.simd_env.eval_engine import _TerminalHeroMixin

    state = {k: v for k, v in vars(sim).items() if k not in SIM_SKIP}
    rt = {k: v for k, v in vars(runtime).items() if isinstance(v, np.ndarray)}
    return copy.deepcopy(
        dict(
            schema=SCHEMA,
            sim=state,
            config=asdict(sim.cfg),
            terminal_hero=isinstance(sim, _TerminalHeroMixin),
            jit=sim._kernels is not None,
            runtime=rt,
            memory=vars(runtime.memory),
            runtime_counts=[runtime.fresh_rows, runtime.carried_rows],
            backend=runtime.backend,
            runtime_params=asdict(runtime.featurizer.p),
            specs=[list(s) for s in specs],
            provenance=provenance or {},
        )
    )


def save_snapshot(path: Path, snapshot: dict) -> dict:
    arrays: dict = {}
    tree = _encode(snapshot, arrays)
    arrays["metadata"] = np.frombuffer(json.dumps(tree).encode(), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        np.savez_compressed(fh, **arrays)
    return {"file": path.name, "sha256": sha256(path), "frame": int(snapshot["sim"]["frame"][0])}


def load_snapshot(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as arrays:
        state = _decode(json.loads(arrays["metadata"].tobytes()), arrays)
    if state.get("schema") != SCHEMA:
        raise ValueError("unsupported snapshot schema")
    return state


def restore(snapshot: dict) -> tuple:
    """Restore source semantics, including terminal hero and carried opponent state."""
    from research.t3_rollout_labels_20261015.engine import _restore_grid_views
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.eval_engine import _terminal_hero_sim_class
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.vector61_featurizer import Vector61Params
    from src.simd_env.vector61_policy import Vector61Runtime

    cls = _terminal_hero_sim_class("grid") if snapshot["terminal_hero"] else GridBatchSim
    sim = cls.__new__(cls)
    sim.__dict__.update(copy.deepcopy(snapshot["sim"]))
    sim.cfg = BatchSimConfig(**snapshot["config"])
    sim._det = None
    from src.simd_env import grid_sim_nb

    sim._kernels = grid_sim_nb if snapshot["jit"] else None
    _restore_grid_views(sim)
    runtime = Vector61Runtime(
        params=Vector61Params(**snapshot["runtime_params"]), backend=snapshot["backend"]
    )
    runtime.bind(sim, snapshot["runtime"]["controlled"])
    for key, value in snapshot["runtime"].items():
        setattr(runtime, key, value.copy())
    for key, value in snapshot["memory"].items():
        setattr(runtime.memory, key, value.copy())
    runtime.fresh_rows, runtime.carried_rows = snapshot["runtime_counts"]
    policies, groups = make_policies(sim, runtime, [snapshot["specs"]])
    return sim, runtime, policies, groups


def make_policies(sim: Any, runtime: Any, rosters: list, forward: str = "rowwise") -> tuple:
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    groups: dict = {}
    for e, roster in enumerate(rosters):
        for slot, spec in enumerate(roster, 1):
            groups.setdefault(tuple(spec), []).append((e, slot))
    policies = {}
    for spec in groups:
        if spec[0] == "scripted":
            policies[spec] = FastProfileAnchorSimdPolicy(spec[1], sim.get_world_seeds())
        else:
            policies[spec] = Vector61SimdPolicy(spec[1], runtime, forward=forward)
    return policies, {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def step_recorded(sim: Any, runtime: Any, actions: np.ndarray) -> None:
    def choose(prepared: Any) -> np.ndarray:
        runtime.prepare(prepared)
        return actions.copy()

    sim.step_with_policy(choose)
    runtime.observe_step(sim)


def _graft(sim: Any, runtime: Any, e: int, snapshot: dict) -> None:
    # Reward levers/schedules belong to the actor. Every dynamics field must match.
    reward_fields = {"num_envs", "gamma", "kill_scale", "death_value"}
    for key, value in snapshot["config"].items():
        if key not in reward_fields and getattr(sim.cfg, key) != value:
            raise ValueError(f"snapshot arena/config mismatch: {key}")
    if asdict(runtime.featurizer.p) != snapshot["runtime_params"]:
        raise ValueError("snapshot opponent featurizer parameters mismatch")
    for key, value in snapshot["sim"].items():
        if key in {"E", "_env_index", "_snake_index"}:
            continue
        target = getattr(sim, key, None)
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == 1:
            if not isinstance(target, np.ndarray) or target.shape[1:] != value.shape[1:]:
                raise ValueError(f"snapshot simulator shape mismatch: {key}")
            target[e] = copy.deepcopy(value[0])
        elif isinstance(value, list) and len(value) == 1:
            target[e] = copy.deepcopy(value[0])
        elif key not in {"_policy_step_in_progress"} and target != value:
            raise ValueError(f"snapshot simulator invariant mismatch: {key}")
    hero_controlled = bool(runtime.controlled[e, 0])
    for key, value in snapshot["runtime"].items():
        getattr(runtime, key)[e] = value[0].copy()
    runtime.controlled[e, 0] = hero_controlled
    for key, value in snapshot["memory"].items():
        getattr(runtime.memory, key)[e] = value[0].copy()
    sim._det = None


class SnapshotStore:
    """Hash-validated immutable snapshots; selection uses a separate per-segment RNG."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA or not self.manifest.get("snapshots"):
            raise ValueError("invalid or empty restart snapshot bank")
        self.entries = self.manifest["snapshots"]
        for entry in self.entries:
            name = entry["file"]
            if Path(name).name != name or sha256(self.path / name) != entry["sha256"]:
                raise ValueError("snapshot hash/path mismatch")
            snap = load_snapshot(self.path / name)
            if int(snap["sim"]["frame"][0]) != entry["frame"] or entry["frame"] < 0:
                raise ValueError("snapshot frame mismatch")
        for path_, digest in self.manifest.get("checkpoints", {}).items():
            if sha256(Path(path_)) != digest:
                raise ValueError(f"snapshot opponent checkpoint changed: {path_}")

    def validate_segment(self, segment_frames: int) -> None:
        if any(e["frame"] >= segment_frames for e in self.entries):
            raise ValueError("snapshot frame must precede segment end")

    def apply(
        self,
        sim: Any,
        runtime: Any,
        policies: dict,
        groups: dict,
        fraction: float,
        seed: int,
        segment: int,
    ) -> tuple:
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("restart fraction must lie in [0, 1]")
        restarted = np.zeros(sim.E, dtype=bool)
        count = int(round(sim.E * fraction))
        if not count:
            return policies, groups, restarted
        rng = np.random.default_rng([seed, segment, 20261016])
        worlds = rng.choice(sim.E, count, replace=False)
        choices = rng.integers(len(self.entries), size=count)
        rosters = [[None] * (sim.S - 1) for _ in range(sim.E)]
        for spec, cells in groups.items():
            for e, s in cells:
                rosters[e][s - 1] = spec
        for e, index in zip(worlds, choices):
            snap = load_snapshot(self.path / self.entries[int(index)]["file"])
            _graft(sim, runtime, int(e), snap)
            rosters[e] = snap["specs"]
            restarted[e] = True
        # Preserve the actor's existing numerical execution mode. Standalone
        # source parity uses rowwise inference; the actor already acts batched.
        policies, groups = make_policies(sim, runtime, rosters, forward="batched")
        return policies, groups, restarted
