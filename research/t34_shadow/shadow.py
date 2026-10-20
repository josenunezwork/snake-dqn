"""Exact, bounded shadow continuations for T34.

This module deliberately owns no simulator.  It drives the established T3
``CapturedWorld`` seam, so policy dispatch, Vector61 carry and GridBatchSim
remain the existing tested implementations.
"""

from __future__ import annotations

import copy
import hashlib
import random
import time
import types
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class ShadowDeadlineExceeded(TimeoutError):
    """Raised when a bounded shadow operation reaches its registered deadline."""


def _update_value(digest: "hashlib._Hash", value: Any, seen: set[int]) -> None:
    """Hash mutable runtime state without relying on repr or object addresses."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
        return
    if isinstance(value, np.generic):
        _update_value(digest, value.item(), seen)
        return
    if isinstance(value, types.ModuleType):
        digest.update(b"module:")
        digest.update(value.__name__.encode())
        return
    if isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        digest.update(b"callable:")
        digest.update(str(getattr(value, "__qualname__", value.__name__)).encode())
        return
    if isinstance(value, np.random.Generator):
        _update_value(digest, value.bit_generator.state, seen)
        return
    if isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        if value.dtype.hasobject:
            # Object-array bytes are pointer values, not stable state.
            for item in value.flat:
                _update_value(digest, item, seen)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    # Torch is optional at import time; tensors are part of policy carry when present.
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
        return
    identity = id(value)
    if identity in seen:
        digest.update(b"cycle")
        return
    seen.add(identity)
    if isinstance(value, dict):
        digest.update(b"dict")
        digest.update(str(len(value)).encode())
        for key in sorted(value, key=lambda item: repr(item)):
            if isinstance(key, str) and (
                key.endswith("_seconds")
                or key.endswith("_seconds_total")
                or key.endswith("_seconds_max")
                or key in {"_rule_config_checked"}
            ):
                continue
            _update_value(digest, key, seen)
            _update_value(digest, value[key], seen)
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        digest.update(str(len(value)).encode())
        for item in value:
            _update_value(digest, item, seen)
    elif isinstance(value, (set, frozenset)):
        digest.update(type(value).__name__.encode())
        digest.update(str(len(value)).encode())
        for item in sorted(value, key=repr):
            _update_value(digest, item, seen)
    elif hasattr(value, "getstate") and callable(value.getstate):
        digest.update(type(value).__name__.encode())
        _update_value(digest, value.getstate(), seen)
    elif hasattr(value, "__dict__"):
        digest.update(f"{type(value).__module__}.{type(value).__qualname__}".encode())
        # Network parameters are immutable under inference.  Their bound checkpoint is
        # included separately; including modules here would duplicate hundreds of MB.
        for key, item in sorted(vars(value).items()):
            if key.endswith(("_seconds", "_seconds_total", "_seconds_max")):
                continue
            if key in {
                "net",
                "network",
                "model",
                "_kernels",
                "_det",
                # Vector61FeaturizerNB's food cache is derived from ``food_cells``.
                # Its sim id is an address, so neither is semantic trajectory state.
                "_food_sim_id",
                "_food_lists",
                "_food_buf",
                "_food_n",
                "_rule_config_checked",
                "apply_seconds_total",
                "apply_seconds_max",
                "head_seconds_total",
                "head_seconds_max",
            }:
                continue
            digest.update(str(key).encode())
            _update_value(digest, item, seen)
    else:
        digest.update(f"opaque:{type(value).__module__}.{type(value).__qualname__}".encode())
    # Alias topology of immutable policies/params is not world state. Retain
    # cycle protection only on the active recursion path, so an equal copied
    # value hashes like an equal shared value.
    seen.remove(identity)


def _ordered_bodies(sim: Any) -> list[list[list[int]]]:
    """Return ordered head-to-tail bodies; ring-buffer storage order is not state order."""
    bodies: list[list[list[int]]] = []
    for env in range(int(sim.E)):
        for slot in range(int(sim.S)):
            count = int(sim.seg_count[env, slot])
            indices = (int(sim.head_ptr[env, slot]) - np.arange(count)) % int(sim.cap)
            bodies.append(np.asarray(sim.bodies[env, slot, indices], dtype=np.int64).tolist())
    return bodies


def full_digest(
    world: Any, *, checkpoint: Any = None, global_rng_state: object | None = None
) -> str:
    """Return an exact state digest for a T3 captured world.

    The payload intentionally includes all mutable simulator arrays, ordered food
    lists, each per-world MT stream, Vector61/v8 runtime carry, policy carry,
    frame/life state and the isolated CPython global RNG.  ``checkpoint`` is a
    caller-provided immutable checkpoint identity (normally a SHA-256 string).
    """
    sim = world.sim
    payload = {
        "ordered_bodies": _ordered_bodies(sim),
        "food_order": getattr(sim, "food_cells", None),
        "corpse_food": getattr(sim, "corpse_cells", None),
        "sim": {
            key: value
            for key, value in vars(sim).items()
            if key not in {"bodies", "food_cells", "corpse_cells", "_kernels", "_det"}
        },
        "env_rng": [getattr(rng, "_rng", rng).getstate() for rng in getattr(sim, "_rngs", [])],
        "runtime": getattr(world, "runtime", None),
        "hero": getattr(world, "hero", None),
        "opponents": getattr(world, "opponents", None),
        "checkpoint": checkpoint,
        "global_random": random.getstate() if global_rng_state is None else global_rng_state,
    }
    result = hashlib.sha256()
    _update_value(result, payload, set())
    return result.hexdigest()


def clone_world_exact(source: Any) -> Any:
    """Clone a T3 world and restore its actual pre/post-step lifecycle state.

    ``CapturedWorld.clone`` is intentionally branch-oriented and assumes a
    selector-prepared source.  A long-lived shadow is often cloned immediately
    *after* a step, so blindly retaining that assumption would skip the next
    prepare/food/respawn phase.  Normalize the clone to the source flags here.
    """
    clone = source.clone()
    # T3 retains immutable Torch modules while shallow-copying policy dictionaries.
    # Restore independent stateful carry that a shadow can mutate (notably v8 hooks).
    source_runtime, clone_runtime = source.runtime, clone.runtime
    runtime_memo: dict[int, Any] = {}
    for name, value in vars(source_runtime).items():
        if name == "trace":
            setattr(clone_runtime, name, value)
            continue
        try:
            copied = copy.deepcopy(value, runtime_memo)
        except (TypeError, RuntimeError):
            copied = copy.copy(value)
        if name in {"featurizer", "free_space_featurizer"} and copied is not None:
            # Rebuildable NB caches cannot point at the source simulator.
            for cache_name, empty in (
                ("_food_sim_id", None),
                ("_food_lists", {}),
                ("_food_buf", np.zeros((0, 0, 2), dtype=np.int64)),
                ("_food_n", np.zeros(0, dtype=np.int64)),
            ):
                if hasattr(copied, cache_name):
                    setattr(copied, cache_name, empty)
        setattr(clone_runtime, name, copied)
    policy_memo: dict[int, Any] = {}
    for original, copied in zip([source.hero, *source.opponents], [clone.hero, *clone.opponents]):
        if not hasattr(original, "__dict__"):
            continue
        state = {}
        for name, value in vars(original).items():
            if name in {"net", "network", "model", "_torch", "device"}:
                state[name] = value
            elif name == "runtime":
                state[name] = clone_runtime
            else:
                try:
                    state[name] = copy.deepcopy(value, policy_memo)
                except (TypeError, RuntimeError):
                    state[name] = copy.copy(value)
        copied.__dict__.clear()
        copied.__dict__.update(state)
    for name in ("_policy_phase_prepared", "_active_env_mask"):
        if hasattr(source.sim, name):
            value = getattr(source.sim, name)
            setattr(
                clone.sim,
                name,
                value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value),
            )
        elif hasattr(clone.sim, name):
            delattr(clone.sim, name)
    # ``_policy_step_in_progress`` is an execution-stack sentinel, never durable
    # state.  A clone made from inside a selector must be independently step-able.
    clone.sim._policy_step_in_progress = False
    return clone


@dataclass(frozen=True)
class ShadowFrame:
    """One predicted post-transition state."""

    action: int
    digest: str
    frame: int
    hero_alive: bool
    hero_length: float


class ShadowTrajectory:
    """A bounded exact greedy continuation rooted at a captured T3 world."""

    def __init__(
        self,
        depth: int = 300,
        *,
        checkpoint: Any = None,
        stepper: Callable[[Any], tuple[np.ndarray, np.ndarray, Any]] | None = None,
    ) -> None:
        if not 1 <= int(depth) <= 500:
            raise ValueError("shadow depth must be in 1..500")
        self.depth = int(depth)
        self.checkpoint = checkpoint
        self._stepper = stepper
        self.world: Any | None = None
        self.source_digest: str | None = None
        self.frames: list[ShadowFrame] = []
        self.invalid_reason: str | None = None
        self._global_rng_state: object | None = None

    @property
    def confirmed(self) -> bool:
        return self.world is not None and self.invalid_reason is None

    def fork(self, source: Any, *, global_rng_state: object | None = None) -> None:
        """Re-fork from an exact source state and clear any stale predictions."""
        self.world = clone_world_exact(source)
        self._global_rng_state = (
            random.getstate() if global_rng_state is None else copy.deepcopy(global_rng_state)
        )
        self.source_digest = full_digest(
            self.world, checkpoint=self.checkpoint, global_rng_state=self._global_rng_state
        )
        self.frames = []
        self.invalid_reason = None

    def invalidate(self, reason: str) -> None:
        """Mark every outstanding prediction stale; it can no longer authorize work."""
        self.invalid_reason = str(reason)
        self.frames = []

    def _choose_and_step(self, world: Any) -> tuple[np.ndarray, np.ndarray, Any]:
        if self._stepper is not None:
            return self._stepper(world)
        from research.t3_rollout_labels_20261015.bank_engine import choose_and_step

        return choose_and_step(world, capture_obs=False)

    def catch_up(self, *, deadline: float | None = None) -> int:
        """Advance to the configured horizon, stopping before a hard deadline."""
        if not self.confirmed or self.world is None:
            return 0
        made = 0
        ambient = random.getstate()
        try:
            if self._global_rng_state is not None:
                random.setstate(self._global_rng_state)
            while len(self.frames) < self.depth and bool(self.world.sim.alive[0, 0]):
                if deadline is not None and time.monotonic() >= deadline:
                    raise ShadowDeadlineExceeded("shadow catch-up deadline reached")
                actions, _advisory, _obs = self._choose_and_step(self.world)
                self._global_rng_state = random.getstate()
                self.frames.append(
                    ShadowFrame(
                        action=int(actions[0, 0]),
                        digest=full_digest(
                            self.world,
                            checkpoint=self.checkpoint,
                            global_rng_state=self._global_rng_state,
                        ),
                        frame=int(self.world.sim.frame[0]),
                        hero_alive=bool(self.world.sim.alive[0, 0]),
                        hero_length=float(self.world.sim.length[0, 0]),
                    )
                )
                made += 1
        finally:
            random.setstate(ambient)
        return made

    def confirm(self, live: Any) -> bool:
        """Confirm one real transition against the next prediction, consuming it on match."""
        if not self.confirmed or not self.frames:
            return False
        actual = full_digest(live, checkpoint=self.checkpoint)
        expected = self.frames[0]
        if actual != expected.digest:
            self.invalidate("full_digest_mismatch")
            return False
        self.frames.pop(0)
        return True
