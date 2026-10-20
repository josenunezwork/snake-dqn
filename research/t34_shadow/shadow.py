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


# ``full_digest`` is carried by Path B controller bindings. Consumers that
# persist one must record this value beside it; versions are purposely
# incomparable as v2 removed recursive/repr serialization and v3 added
# content fragments for repeated carry aliases.
FULL_DIGEST_SCHEMA = "t34-shadow-full-digest/v3"
_FAST_DIGEST_VERSION = FULL_DIGEST_SCHEMA.encode("ascii") + b"\0"
_SIM_EXCLUDED_FIELDS = frozenset({"bodies", "food_cells", "corpse_cells", "_kernels", "_det"})
_STATIC_OBJECT_FIELDS = frozenset(
    {
        "model",
        "net",
        "network",
        "_det",
        "_food_buf",
        "_food_lists",
        "_food_n",
        "_food_sim_id",
        "_kernels",
        "_rule_config_checked",
    }
)


def _digest_label(digest: "hashlib._Hash", label: str) -> None:
    data = label.encode("utf-8")
    digest.update(len(data).to_bytes(4, "little"))
    digest.update(data)


def _digest_array(digest: "hashlib._Hash", label: str, value: np.ndarray) -> None:
    """Append one typed C-contiguous array with unambiguous boundaries."""
    original = np.asarray(value)
    array = np.ascontiguousarray(original)
    _digest_label(digest, label)
    digest.update(b"A")
    dtype = original.dtype.str.encode("ascii")
    digest.update(len(dtype).to_bytes(2, "little"))
    digest.update(dtype)
    digest.update(original.ndim.to_bytes(2, "little"))
    digest.update(np.asarray(original.shape, dtype="<i8").tobytes())
    if original.dtype.hasobject:
        for index, item in enumerate(original.flat):
            _digest_value(digest, str(index), item)
    else:
        digest.update(array.tobytes())


def _digest_mt_state(digest: "hashlib._Hash", label: str, state: object) -> None:
    """Encode CPython MT state as fixed-width bytes instead of 625 repr calls."""
    if not isinstance(state, tuple) or len(state) != 3 or not isinstance(state[1], tuple):
        _digest_value(digest, label, state)
        return
    version, words_and_index, gaussian = state
    if not isinstance(version, int) or len(words_and_index) < 2:
        _digest_value(digest, label, state)
        return
    words = np.asarray(words_and_index[:-1], dtype="<u4")
    index = int(words_and_index[-1])
    _digest_label(digest, label)
    digest.update(b"M")
    digest.update(int(version).to_bytes(8, "little", signed=True))
    digest.update(len(words).to_bytes(4, "little"))
    digest.update(words.tobytes())
    digest.update(index.to_bytes(8, "little", signed=True))
    _digest_value(digest, "gaussian", gaussian)


def _digest_value(
    digest: "hashlib._Hash", label: str, value: Any, seen: set[int] | None = None
) -> None:
    """Encode bounded carry/metadata without object addresses or ``repr`` walks."""
    if seen is None:
        seen = set()
    tracked = isinstance(value, (tuple, list, set, frozenset, dict)) or hasattr(value, "__dict__")
    identity = id(value)
    if tracked:
        if identity in seen:
            _digest_label(digest, label)
            digest.update(b"Q")
            return
        seen.add(identity)
    if isinstance(value, np.ndarray):
        _digest_array(digest, label, value)
    elif value is None:
        _digest_label(digest, label)
        digest.update(b"N")
    elif isinstance(value, bool):
        _digest_label(digest, label)
        digest.update(b"B" + bytes((value,)))
    elif isinstance(value, int):
        _digest_label(digest, label)
        digest.update(b"I" + str(value).encode("ascii") + b"\0")
    elif isinstance(value, float):
        _digest_label(digest, label)
        digest.update(b"F" + np.asarray([value], dtype="<f8").tobytes())
    elif isinstance(value, str):
        _digest_label(digest, label)
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "little") + encoded)
    elif isinstance(value, bytes):
        _digest_label(digest, label)
        digest.update(b"Y" + len(value).to_bytes(8, "little") + value)
    elif isinstance(value, np.generic):
        _digest_value(digest, label, value.item(), seen)
    elif isinstance(value, random.Random):
        _digest_mt_state(digest, label, value.getstate())
    elif isinstance(value, np.random.Generator):
        _digest_value(digest, label, value.bit_generator.state, seen)
    elif hasattr(value, "getstate") and callable(value.getstate):
        _digest_value(digest, label, value.getstate(), seen)
    elif type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu().contiguous()
        _digest_label(digest, label)
        digest.update(b"Z")
        _digest_array(digest, "tensor", tensor.numpy())
    elif isinstance(value, types.ModuleType):
        _digest_label(digest, label)
        digest.update(b"U" + value.__name__.encode("utf-8"))
    elif isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        _digest_label(digest, label)
        digest.update(b"C" + getattr(value, "__qualname__", value.__name__).encode("utf-8"))
    elif isinstance(value, (tuple, list)):
        _digest_label(digest, label)
        digest.update(b"T" if isinstance(value, tuple) else b"L")
        digest.update(len(value).to_bytes(8, "little"))
        for index, item in enumerate(value):
            _digest_value(digest, str(index), item, seen)
    elif isinstance(value, (set, frozenset)):
        _digest_label(digest, label)
        digest.update(b"E")
        # Cell sets are semantically unordered. Their integer tuple keys are stable.
        ordered = sorted(value)
        digest.update(len(ordered).to_bytes(8, "little"))
        coordinate_width = len(ordered[0]) if ordered and isinstance(ordered[0], tuple) else 0
        if coordinate_width and all(
            isinstance(item, tuple)
            and len(item) == coordinate_width
            and all(type(part) is int and -(2**63) <= part < 2**63 for part in item)
            for item in ordered
        ):
            # ``food_set`` is commonly hundreds of lattice coordinate tuples.
            # Retain set/tuple/arity framing, then move its scalar payload in
            # one contiguous NumPy buffer instead of recursive Python visits.
            digest.update(b"P")
            digest.update(coordinate_width.to_bytes(4, "little"))
            _digest_array(digest, "coordinates", np.asarray(ordered, dtype=np.int64))
        else:
            for index, item in enumerate(ordered):
                _digest_value(digest, str(index), item, seen)
    elif isinstance(value, dict):
        _digest_label(digest, label)
        digest.update(b"D")
        ordered = sorted(
            (
                key
                for key in value
                if not (
                    isinstance(key, str)
                    and (
                        key.endswith(("_seconds", "_seconds_total", "_seconds_max"))
                        or key == "_rule_config_checked"
                    )
                )
            ),
            key=lambda item: (type(item).__qualname__, str(item)),
        )
        digest.update(len(ordered).to_bytes(8, "little"))
        for key in ordered:
            _digest_value(digest, "key", key, seen)
            _digest_value(digest, "value", value[key], seen)
    elif hasattr(value, "__dict__"):
        _digest_label(digest, label)
        digest.update(f"O:{type(value).__module__}.{type(value).__qualname__}".encode())
        fields = [
            (name, item)
            for name, item in sorted(vars(value).items())
            if name not in _STATIC_OBJECT_FIELDS
            and not name.endswith(("_seconds", "_seconds_total", "_seconds_max"))
        ]
        digest.update(len(fields).to_bytes(8, "little"))
        for name, item in fields:
            _digest_value(digest, name, item, seen)
    else:
        _digest_label(digest, label)
        digest.update(f"X:{type(value).__module__}.{type(value).__qualname__}".encode())
    if tracked:
        seen.remove(identity)


def _ordered_body_bytes(sim: Any) -> np.ndarray:
    """Materialize only live ring-buffer cells in canonical head-to-tail order."""
    chunks = []
    for env in range(int(sim.E)):
        for slot in range(int(sim.S)):
            count = int(sim.seg_count[env, slot])
            indices = (int(sim.head_ptr[env, slot]) - np.arange(count)) % int(sim.cap)
            chunks.append(np.asarray(sim.bodies[env, slot, indices], dtype=np.int64))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.int64)


def _ordered_cells(cells: Any) -> tuple[np.ndarray, np.ndarray]:
    """Encode per-environment ordered food/corpse cells with explicit offsets."""
    chunks, offsets = [], [0]
    for env_cells in cells:
        ordered = sorted(env_cells) if isinstance(env_cells, (set, frozenset)) else list(env_cells)
        chunk = np.asarray(ordered, dtype=np.int64).reshape(-1, 2)
        chunks.append(chunk)
        offsets.append(offsets[-1] + len(chunk))
    joined = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2), dtype=np.int64)
    return joined, np.asarray(offsets, dtype=np.int64)


def _digest_runtime(digest: "hashlib._Hash", runtime: Any) -> None:
    """Hash only runtime carry that can affect the next selected action.

    Featurizer/grid caches are recomputable from the simulator and include
    simulator identities.  They are deliberately outside the shadow contract.
    """
    if runtime is None:
        _digest_value(digest, "runtime", None)
        return
    _digest_label(digest, "runtime")
    digest.update(b"R")
    fields = [
        (name, value)
        for name, value in sorted(vars(runtime).items())
        if name not in _STATIC_OBJECT_FIELDS
        and not name.endswith(("_seconds", "_seconds_total", "_seconds_max"))
    ]
    digest.update(len(fields).to_bytes(8, "little"))
    for name, value in fields:
        _digest_value(digest, name, value)


def _digest_fragment(label: str, value: Any) -> bytes:
    """Return a content-addressed encoding for repeated runtime state.

    The fragment is keyed by bytes, never object identity: a deep-copied but
    equal runtime stays indistinguishable from an alias, as it was under the
    legacy serializer.  It saves hashing the same large carry arrays through
    every opponent policy in a six-snake world.
    """
    fragment = hashlib.sha256()
    _digest_value(fragment, label, value)
    return fragment.digest()


def _digest_runtime_fragment(
    digest: "hashlib._Hash", label: str, value: Any, memo: dict[int, bytes]
) -> None:
    _digest_label(digest, label)
    digest.update(b"H")
    identity = id(value)
    fragment = memo.get(identity)
    if fragment is None:
        # The inner label is deliberately independent of its outer policy path:
        # aliases and separately copied equal runtimes serialize identically.
        fragment = _digest_fragment("runtime-content", value)
        memo[identity] = fragment
    digest.update(fragment)


def _digest_policy(
    digest: "hashlib._Hash", label: str, policy: Any, runtime_memo: dict[int, bytes]
) -> None:
    """Hash policy carry while compacting only semantically equal runtime aliases."""
    if policy is None or not hasattr(policy, "__dict__"):
        _digest_value(digest, label, policy)
        return
    _digest_label(digest, label)
    digest.update(f"O:{type(policy).__module__}.{type(policy).__qualname__}".encode())
    fields = [
        (name, value)
        for name, value in sorted(vars(policy).items())
        if name not in _STATIC_OBJECT_FIELDS
        and not name.endswith(("_seconds", "_seconds_total", "_seconds_max"))
    ]
    digest.update(len(fields).to_bytes(8, "little"))
    for name, value in fields:
        if name == "runtime":
            _digest_runtime_fragment(digest, name, value, runtime_memo)
        else:
            _digest_value(digest, name, value)


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
    result = hashlib.sha256(_FAST_DIGEST_VERSION)
    _digest_array(result, "ordered_bodies", _ordered_body_bytes(sim))
    for label, cells in (("food", sim.food_cells), ("corpse", sim.corpse_cells)):
        joined, offsets = _ordered_cells(cells)
        _digest_array(result, f"{label}_cells", joined)
        _digest_array(result, f"{label}_offsets", offsets)
    # The legacy digest included every simulator attribute other than these
    # five. Preserve that divergence surface, but hash ndarray fields directly
    # in fixed lexical order instead of recursively repr-walking them.
    for name, value in sorted(vars(sim).items()):
        if name not in _SIM_EXCLUDED_FIELDS:
            _digest_value(result, f"sim.{name}", value)
    for index, rng in enumerate(getattr(sim, "_rngs", ())):
        _digest_mt_state(result, f"env_rng.{index}", getattr(rng, "_rng", rng).getstate())
    runtime = getattr(world, "runtime", None)
    runtime_memo: dict[int, bytes] = {}
    _digest_runtime_fragment(result, "runtime", runtime, runtime_memo)
    _digest_policy(result, "hero", getattr(world, "hero", None), runtime_memo)
    opponents = getattr(world, "opponents", None)
    if isinstance(opponents, list):
        _digest_label(result, "opponents")
        result.update(b"L")
        result.update(len(opponents).to_bytes(8, "little"))
        for index, opponent in enumerate(opponents):
            _digest_policy(result, str(index), opponent, runtime_memo)
    else:
        _digest_value(result, "opponents", opponents)
    _digest_value(result, "checkpoint", checkpoint)
    _digest_mt_state(
        result, "global_random", random.getstate() if global_rng_state is None else global_rng_state
    )
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
