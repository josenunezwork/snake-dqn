"""Explicit flat T34 state encoder, opt-in only; v3 remains the reference.

The stream is a sequence of typed, length-delimited records. Arrays carry their
dtype, shape and C-order bytes; cells carry environment offsets. Runtime objects
retain v3's field surface and exclusions. Only deeply immutable frozen dataclasses
and immutable checkpoint values are cached between calls. No mutable state or
object-address bytes are cached. Instances belong to one lane (not thread shared).
"""

from __future__ import annotations

import hashlib
import random
import struct
import types
from dataclasses import is_dataclass
from typing import Any

import numpy as np

FULL_DIGEST_SCHEMA = "t34-shadow-full-digest/v4"
_VERSION = FULL_DIGEST_SCHEMA.encode("ascii") + b"\0"
_U64 = struct.Struct("<Q").pack
_F64 = struct.Struct("<d").pack
_MT = struct.Struct("<625I").pack
_EXCLUDED = frozenset(
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
_SIM_EXCLUDED = frozenset({"bodies", "food_cells", "corpse_cells", "_kernels", "_det", "_rngs"})
_UNSET = object()


def _ignored(name: str) -> bool:
    return name.endswith(("_seconds", "_seconds_total", "_seconds_max"))


def _blob(tag: bytes, data: bytes) -> bytes:
    return tag + _U64(len(data)) + data


def _immutable(value: Any) -> bool:
    """Admit only values whose full transitive contents cannot mutate normally."""
    if type(value) in (type(None), bool, int, float, str, bytes):
        return True
    if type(value) in (tuple, frozenset):
        return all(_immutable(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type) and hasattr(value, "__dict__"):
        # An undecorated subclass inherits frozen metadata, but may add mutable
        # attributes. Only the class that declares the frozen contract is safe.
        params = type(value).__dict__.get("__dataclass_params__")
        if params is not None and params.frozen:
            return all(_immutable(item) for item in vars(value).values())
    return False


class DigestV4:
    """Lane-local serializer with once-per-run immutable content fragments.

    ``checkpoint`` in the constructor is captured by value. A per-call override
    remains available for the v3 call seam; mutable overrides are encoded afresh.
    Create a new instance when the run's immutable configuration changes.
    """

    def __init__(self, *, checkpoint: Any = None) -> None:
        self._static: dict[int, tuple[Any, bytes]] = {}
        self._array_headers: dict[tuple[Any, ...], bytes] = {}
        self._field_names: dict[tuple[str, ...], tuple[str, ...]] = {}
        self._checkpoints: dict[int, tuple[Any, bytes]] = {}
        self._object_memo: dict[tuple[int, frozenset[int]], bytes] = {}
        self._array_memo: dict[int, tuple[np.ndarray, bytes]] = {}
        self._sim_names: dict[tuple[str, ...], tuple[str, ...]] = {}
        self._checkpoint = self._fragment(checkpoint)

    def _array(self, value: np.ndarray) -> bytes:
        if value.dtype.hasobject:
            # Object pointers are not semantic bytes.
            return (
                _blob(b"O", self._encode(value.tolist()))
                + _blob(b"d", value.dtype.str.encode())
                + self._encode(value.shape)
            )
        key = (value.dtype.str, value.shape)
        header = self._array_headers.get(key)
        if header is None:
            dtype = value.dtype.str.encode("ascii")
            header = b"A" + _blob(b"d", dtype) + _U64(value.ndim)
            header += struct.pack(f"<{value.ndim}q", *value.shape) + _U64(value.nbytes)
            self._array_headers[key] = header
        if value.nbytes >= 4096:
            cached = self._array_memo.get(id(value))
            if cached is not None and cached[0] is value:
                return header + b"H" + cached[1]
            payload = memoryview(value) if value.flags.c_contiguous else value.tobytes(order="C")
            fragment = hashlib.sha256(payload).digest()
            self._array_memo[id(value)] = (value, fragment)
            return header + b"H" + fragment
        return header + b"V" + value.tobytes(order="C")

    def _write_array(self, digest: Any, value: np.ndarray) -> None:
        """Stream bulk sim arrays without joining/copying a multi-MB state blob."""
        digest.update(self._array(value))

    def _names(self, value: Any) -> tuple[str, ...]:
        keys = tuple(vars(value))
        names = self._field_names.get(keys)
        if names is None:
            names = tuple(
                name for name in sorted(keys) if name not in _EXCLUDED and not _ignored(name)
            )
            self._field_names[keys] = names
        return names

    def _object(self, value: Any, seen: set[int], omit: frozenset[str] = frozenset()) -> bytes:
        names = tuple(name for name in self._names(value) if name not in omit)
        parts = [
            _blob(b"o", f"{type(value).__module__}.{type(value).__qualname__}".encode()),
            _U64(len(names)),
        ]
        for name in names:
            parts.extend((_blob(b"k", name.encode()), self._encode(getattr(value, name), seen)))
        return b"".join(parts)

    def _mt(self, state: Any) -> bytes:
        if (
            isinstance(state, tuple)
            and len(state) == 3
            and state[0] == 3
            and isinstance(state[1], tuple)
            and len(state[1]) == 625
        ):
            return b"M" + _MT(*state[1]) + self._encode(state[2])
        return b"m" + self._encode(state)

    def _fragment(self, value: Any) -> bytes:
        return hashlib.sha256(self._encode(value)).digest()

    def _checkpoint_fragment(self, value: Any) -> bytes:
        cached = self._checkpoints.get(id(value))
        if cached is not None and cached[0] is value:
            return cached[1]
        fragment = self._fragment(value)
        if _immutable(value):
            if len(self._checkpoints) >= 64:
                self._checkpoints.clear()
            self._checkpoints[id(value)] = (value, fragment)
        return fragment

    def _encode(self, value: Any, seen: set[int] | None = None) -> bytes:
        """Fallback for bounded metadata; bulk arrays/coordinates never scalar-walk."""
        kind = type(value)
        if isinstance(value, np.ndarray):
            return self._array(value)
        if value is None:
            return b"N"
        if isinstance(value, bool):
            return b"B1" if value else b"B0"
        if isinstance(value, int):
            return _blob(b"I", str(value).encode("ascii"))
        if isinstance(value, float):
            return b"F" + _F64(value)
        if isinstance(value, str):
            return _blob(b"S", value.encode("utf-8"))
        if isinstance(value, bytes):
            return _blob(b"Y", value)
        if isinstance(value, np.generic):
            return self._encode(value.item(), seen)
        if isinstance(value, random.Random):
            return self._mt(value.getstate())
        if isinstance(value, np.random.Generator):
            return b"G" + self._encode(value.bit_generator.state, seen)
        if isinstance(value, types.ModuleType):
            return _blob(b"U", value.__name__.encode())
        if isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
            return _blob(b"C", getattr(value, "__qualname__", value.__name__).encode())
        if kind.__module__.startswith("torch") and hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            return b"Z" + self._array(tensor.numpy())
        identity = id(value)
        frozen = self._static.get(identity)
        if frozen is not None and frozen[0] is value:
            return b"H" + frozen[1]
        if seen is None:
            seen = set()
        if identity in seen:
            return b"Q"
        memo_key = (identity, frozenset(seen))
        if memo_key in self._object_memo:
            return self._object_memo[memo_key]
        seen.add(identity)
        try:
            if isinstance(value, (tuple, list, set, frozenset)):
                tag = (
                    b"T" if isinstance(value, tuple) else b"L" if isinstance(value, list) else b"E"
                )
                items = sorted(value) if tag == b"E" else value
                # Coordinate collections dominate the old recursive traversal.
                # Exact Python ints only: preserve bool/int and tuple/list distinctions.
                if items and all(
                    type(item) is tuple
                    and len(item) == 2
                    and all(type(part) is int and -(2**63) <= part < 2**63 for part in item)
                    for item in items
                ):
                    result = tag + b"P" + self._array(np.asarray(items, dtype="<i8"))
                else:
                    result = (
                        tag
                        + b"V"
                        + _U64(len(items))
                        + b"".join(self._encode(item, seen) for item in items)
                    )
            elif isinstance(value, dict):
                keys = sorted(
                    (
                        key
                        for key in value
                        if not (
                            isinstance(key, str)
                            and (_ignored(key) or key == "_rule_config_checked")
                        )
                    ),
                    key=lambda item: (type(item).__qualname__, str(item)),
                )
                result = (
                    b"D"
                    + _U64(len(keys))
                    + b"".join(
                        self._encode(key, seen) + self._encode(value[key], seen) for key in keys
                    )
                )
            elif hasattr(value, "getstate") and callable(value.getstate):
                result = b"g" + self._encode(value.getstate(), seen)
            elif hasattr(value, "__dict__"):
                result = b"h" + hashlib.sha256(self._object(value, seen)).digest()
            else:
                result = _blob(b"X", f"{kind.__module__}.{kind.__qualname__}".encode())
        finally:
            seen.remove(identity)
        if is_dataclass(value) and _immutable(value):
            fragment = hashlib.sha256(result).digest()
            if len(self._static) >= 128:
                self._static.clear()
            self._static[identity] = (value, fragment)
            return b"H" + fragment
        if hasattr(value, "__dict__"):
            self._object_memo[memo_key] = result
        return result

    def __call__(
        self, world: Any, *, checkpoint: Any = _UNSET, global_rng_state: object | None = None
    ) -> str:
        """Hash all semantic state every call; never sample confirmation state."""
        sim = world.sim
        self._object_memo = {}
        self._array_memo = {}
        result = hashlib.sha256(_VERSION)
        parts = []
        # Explicit fixed ordering, with lengths framing every live body.
        self._write_array(result, np.asarray(sim.seg_count))
        for env in range(int(sim.E)):
            for slot in range(int(sim.S)):
                count = int(sim.seg_count[env, slot])
                indices = (int(sim.head_ptr[env, slot]) - np.arange(count)) % int(sim.cap)
                self._write_array(result, np.asarray(sim.bodies[env, slot, indices], dtype="<i8"))
        for cells in (sim.food_cells, sim.corpse_cells):
            result.update(_U64(len(cells)))
            for row in cells:
                ordered = sorted(row) if isinstance(row, (set, frozenset)) else row
                self._write_array(result, np.asarray(ordered, dtype="<i8").reshape(-1, 2))
        keys = tuple(vars(sim))
        names = self._sim_names.get(keys)
        if names is None:
            names = tuple(sorted(name for name in keys if name not in _SIM_EXCLUDED))
            self._sim_names[keys] = names
        result.update(_U64(len(names)))
        for name in names:
            result.update(_blob(b"k", name.encode()))
            value = getattr(sim, name)
            if isinstance(value, np.ndarray):
                self._write_array(result, value)
            else:
                result.update(self._encode(value))
        rngs = getattr(sim, "_rngs", ())
        parts.append((b"L" if isinstance(rngs, list) else b"T") + _U64(len(rngs)))
        for rng in rngs:
            generator = getattr(rng, "_rng", rng)
            parts.append(self._mt(generator.getstate()))
            # Preserve any wrapper metadata that v3 also saw, without its second MT.
            parts.append(
                self._object(rng, set(), frozenset({"_rng"})) if rng is not generator else b"N"
            )
        memo: dict[int, bytes] = {}

        def runtime_fragment(runtime: Any) -> bytes:
            key = id(runtime)
            if key not in memo:
                memo[key] = self._fragment(runtime)
            return memo[key]

        parts.append(runtime_fragment(getattr(world, "runtime", None)))
        policies = [getattr(world, "hero", None)]
        opponents = getattr(world, "opponents", None)
        if isinstance(opponents, list):
            policies.extend(opponents)
        parts.append(_U64(len(policies)))
        for policy in policies:
            if policy is None or not hasattr(policy, "__dict__"):
                parts.append(self._encode(policy))
                continue
            parts.append(self._object(policy, {id(policy)}, frozenset({"runtime"})))
            parts.append(
                b"R" + runtime_fragment(policy.runtime) if hasattr(policy, "runtime") else b"N"
            )
        parts.append(b"L" if isinstance(opponents, list) else self._encode(opponents))
        parts.append(
            self._checkpoint if checkpoint is _UNSET else self._checkpoint_fragment(checkpoint)
        )
        parts.append(self._mt(random.getstate() if global_rng_state is None else global_rng_state))
        result.update(b"".join(parts))
        self._object_memo.clear()
        self._array_memo.clear()
        return result.hexdigest()


def full_digest_v4(
    world: Any, *, checkpoint: Any = None, global_rng_state: object | None = None
) -> str:
    """Stateless convenience seam; use a lane-local DigestV4 for static reuse."""
    return DigestV4(checkpoint=checkpoint)(world, global_rng_state=global_rng_state)
