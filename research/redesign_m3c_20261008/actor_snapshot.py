"""Detached actor-process state for M3-C full-state checkpoints.

The learner owns the checkpoint writer.  Actors only use this module while they
are stopped at a message boundary: it copies the state required to resume the
same segment without re-seeding an environment or an n-step window.  The result
contains ordinary Python values, CPU tensors, and NumPy arrays, so it is safe to
send through the actor checkpoint FIFO and to pickle in the background writer.

``restore_actor_state`` deliberately restores into a freshly constructed actor
segment.  This keeps shared-memory handles, compiled forwards, and policy
objects process-local while restoring every mutable simulator/runtime field.
Callers can put pool-specific state (for example a pool RNG state) in ``extra``.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np

SCHEMA = "m3c-actor-state/v1"

# These are process-local caches or callables recreated by ``build_segment``.
# In particular GridBatchSim's numba module cannot be copied/pickled.
_SIM_SKIP = {"_det", "_kernels", "_owner", "_slot", "_food"}
_RUNTIME_SKIP = {"trace", "featurizer", "free_space_featurizer"}
_TAG = "__m3c_actor_snapshot_type__"


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _copy_value(value: Any) -> Any:
    """Deep-copy values while preserving PRNG state and moving tensors to CPU."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.random.Generator):
        return {_TAG: "numpy-generator", "state": copy.deepcopy(value.bit_generator.state)}
    if isinstance(value, random.Random):
        return {_TAG: "python-random", "state": value.getstate()}
    # Importing torch at module import time makes simulator-only tests needlessly
    # expensive.  The ``is_tensor`` predicate is also safe for torch-less tools.
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().to(device="cpu").clone()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {_copy_value(k): _copy_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_value(v) for v in value]
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_copy_value(v) for v in value]}
    if isinstance(value, set):
        return {_TAG: "set", "items": [_copy_value(v) for v in value]}
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if hasattr(value, "__dict__"):
        return {
            _TAG: "object",
            "class": _class_name(value),
            "state": {key: _copy_value(item) for key, item in vars(value).items()},
        }
    # Dataclass-like immutable configuration values and enums land here.  They
    # have no mutable actor state, but still must be detached from the live actor.
    return copy.deepcopy(value)


def _restore_value(current: Any, saved: Any) -> Any:
    """Restore ``saved`` recursively, reusing a compatible live object when possible."""
    if isinstance(saved, dict) and _TAG in saved:
        kind = saved[_TAG]
        if kind == "numpy-generator":
            if current is None:
                bit_generator = getattr(np.random, saved["state"]["bit_generator"], None)
                if bit_generator is None:
                    raise ValueError("actor snapshot has an unknown NumPy bit generator")
                current = np.random.Generator(bit_generator())
            if not isinstance(current, np.random.Generator):
                raise ValueError("actor snapshot RNG type does not match the live actor")
            current.bit_generator.state = copy.deepcopy(saved["state"])
            return current
        if kind == "python-random":
            if current is None:
                current = random.Random()
            if not isinstance(current, random.Random):
                raise ValueError("actor snapshot Python RNG type does not match the live actor")
            current.setstate(saved["state"])
            return current
        if kind == "tuple":
            current_items = current if isinstance(current, tuple) else (None,) * len(saved["items"])
            return tuple(
                _restore_value(current_items[i], item) for i, item in enumerate(saved["items"])
            )
        if kind == "set":
            return {_restore_value(None, item) for item in saved["items"]}
        if kind == "object":
            if current is None or not hasattr(current, "__dict__"):
                raise ValueError("actor snapshot needs a freshly constructed compatible object")
            if _class_name(current) != saved["class"]:
                raise ValueError("actor snapshot object type does not match the live actor")
            _restore_object_state(current, saved["state"])
            return current
        raise ValueError(f"unknown actor snapshot tag {kind!r}")
    if isinstance(saved, np.ndarray):
        if isinstance(current, np.ndarray):
            if current.shape != saved.shape or current.dtype != saved.dtype:
                raise ValueError("actor snapshot array shape/dtype does not match the live actor")
            current[...] = saved
            return current
        return saved.copy()
    try:
        import torch

        if torch.is_tensor(saved):
            if torch.is_tensor(current):
                if current.shape != saved.shape or current.dtype != saved.dtype:
                    raise ValueError(
                        "actor snapshot tensor shape/dtype does not match the live actor"
                    )
                current.copy_(saved.to(device=current.device))
                return current
            return saved.detach().clone()
    except ImportError:
        pass
    if isinstance(saved, dict):
        old = current if isinstance(current, dict) else {}
        return {key: _restore_value(old.get(key), value) for key, value in saved.items()}
    if isinstance(saved, list):
        old = current if isinstance(current, list) else []
        return [
            _restore_value(old[i] if i < len(old) else None, item) for i, item in enumerate(saved)
        ]
    return copy.deepcopy(saved)


def _capture_object_state(obj: Any, skip: Iterable[str] = ()) -> Dict[str, Any]:
    excluded = set(skip)
    return {key: _copy_value(value) for key, value in vars(obj).items() if key not in excluded}


def _restore_object_state(obj: Any, state: Mapping[str, Any]) -> None:
    for key, saved in state.items():
        setattr(obj, key, _restore_value(getattr(obj, key, None), saved))


def _network_state(net: Any) -> Dict[str, Any]:
    """Return a fully detached CPU state dict without retaining autograd storage."""
    return {name: _copy_value(value) for name, value in net.state_dict().items()}


def _restore_network_state(net: Any, state: Mapping[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - callers always have torch
        raise RuntimeError("restoring actor network state requires torch") from exc
    copied = {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in state.items()
    }
    net.load_state_dict(copied, strict=True)


def capture_actor_state(
    sim: Any,
    runtime: Any,
    nsb: Any,
    rng: np.random.Generator,
    net: Any,
    segment: int,
    frame: int,
    last: int,
    live_seats: Optional[np.ndarray] = None,
    pool_seats: Optional[np.ndarray] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture all mutable state of one actor at a frame/message boundary.

    ``sim`` includes the per-world :class:`EnvRng` objects and their seed
    streams.  ``runtime`` includes its selection/carry arrays and trend memory;
    ``nsb`` includes incomplete n-step windows and completed-but-unsent chunks.
    The actor's NumPy generator is encoded independently, rather than relying
    on ambient NumPy or Python global RNG state.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    return {
        "schema": SCHEMA,
        "segment": int(segment),
        "frame": int(frame),
        "last": int(last),
        "sim": _capture_object_state(sim, _SIM_SKIP),
        "runtime": _capture_object_state(runtime, _RUNTIME_SKIP),
        "nsb": _capture_object_state(nsb),
        "rng": _copy_value(rng),
        "net": _network_state(net),
        "live_seats": None if live_seats is None else np.asarray(live_seats).copy(),
        "pool_seats": None if pool_seats is None else np.asarray(pool_seats).copy(),
        "extra": _copy_value(dict(extra or {})),
    }


def restore_actor_state(
    snapshot: Mapping[str, Any],
    sim: Any,
    runtime: Any,
    nsb: Any,
    rng: np.random.Generator,
    net: Any,
) -> Dict[str, Any]:
    """Restore a captured actor into an equivalent freshly-built segment.

    Returns the scalar and league state the caller must install around the
    restored objects.  It intentionally never restores process-local caches;
    the simulator's ``_det`` cache is cleared so the next step rebuilds it.
    """
    if snapshot.get("schema") != SCHEMA:
        raise ValueError("unsupported actor-state snapshot")
    _restore_object_state(sim, snapshot["sim"])
    if hasattr(sim, "_det"):
        sim._det = None
    _restore_object_state(runtime, snapshot["runtime"])
    _restore_object_state(nsb, snapshot["nsb"])
    _restore_value(rng, snapshot["rng"])
    _restore_network_state(net, snapshot["net"])
    live = snapshot.get("live_seats")
    pool = snapshot.get("pool_seats")
    return {
        "segment": int(snapshot["segment"]),
        "frame": int(snapshot["frame"]),
        "last": int(snapshot["last"]),
        "live_seats": None if live is None else np.asarray(live).copy(),
        "pool_seats": None if pool is None else np.asarray(pool).copy(),
        "extra": _restore_value(None, snapshot.get("extra", {})),
    }
