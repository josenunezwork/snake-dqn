"""I0's isolated, all-roster engine arm factory.

This module deliberately only wires components that earned benchmark admission:
MOVE2's indexed movement and the retained vector61 geometry adapter.  It does
not select the pending v8 bundle route, and it keeps the existing static flood
unless a future packet supplies a separately hash-bound sparse admission.

The benchmark owns timing, parity execution, and its immutable packet.  This
module owns only reproducible arm construction and state-inventory helpers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import torch

from research.gpu_sim_spike import a100_ruler as ruler

ARMS = ("R", "I")
"""R is the static/reference engine; I is the admitted I0 combined engine."""

# The geometry route is retained only as an engine component.  Its v8 indexed
# bundle is intentionally not part of either arm until V8-SPLIT selects it.
_FLAGS = {
    "R": {
        "move_backend": "reference",
        "geometry": "torch",
        "v8_indexed": "0",
        "vector_flood": "static",
    },
    "I": {
        "move_backend": "indexed",
        "geometry": "native",
        "v8_indexed": "0",
        "vector_flood": "static",
    },
}

# Clocks and indexed adapter temporary views are diagnostic/transient, not
# rollout state.  A nonempty transient view is a carry leak and is refused.
INSTRUMENTATION_FIELDS = {"_timing", "host_stages", "stage_timing"}
TRANSIENT_FIELDS = {"_geometry_sim", "_geometry_body_view"}


class IntegrationEngineError(RuntimeError):
    """The requested arm cannot preserve I0's fixed component boundary."""


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    """Explicit component selection stored with an I0 benchmark receipt."""

    name: str
    move_backend: str
    geometry: str
    v8_indexed: bool
    vector_flood: str


def arm_spec(name: str) -> ArmSpec:
    """Return the immutable component selection for an I0 arm."""
    if name not in ARMS:
        raise ValueError(f"unknown integration arm: {name}")
    flags = _FLAGS[name]
    return ArmSpec(
        name=name,
        move_backend=flags["move_backend"],
        geometry=flags["geometry"],
        v8_indexed=flags["v8_indexed"] == "1",
        vector_flood=flags["vector_flood"],
    )


@contextlib.contextmanager
def _arm(name: str) -> Iterator[None]:
    """Set process-scoped native-route flags for one isolated arm invocation."""
    flags = arm_spec(name)
    with patch.dict(
        os.environ,
        {
            "GPU_SIM_VECTOR_GEOMETRY": flags.geometry,
            "GPU_SIM_V8_INDEXED": "1" if flags.v8_indexed else "0",
            "GPU_SIM_VECTOR_FLOOD": flags.vector_flood,
        },
    ):
        yield


def _set_move_backend(sim: Any, backend: str) -> None:
    """Set an already-created simulator's selected MOVE implementation."""
    if getattr(sim, "move_backend", None) not in ("reference", "indexed"):
        raise IntegrationEngineError("ruler returned a simulator without MOVE backend selection")
    sim.move_backend = backend


def _install_geometry(policy: Any, geometry: str) -> None:
    """Install the retained geometry runtime without altering policy batching."""
    from research.gpu_sim_spike.vector_geometry import IndexedVectorRuntime

    if geometry == "native":
        if policy.vector_runtime is None:
            raise IntegrationEngineError("device policy has no vector61 runtime")
        policy.vector_runtime = IndexedVectorRuntime.from_runtime(policy.vector_runtime)
    if isinstance(policy.vector_runtime, IndexedVectorRuntime) != (geometry == "native"):
        raise IntegrationEngineError("geometry runtime does not match declared I0 arm")


def _case(checkpoint: Path, mix: str, worlds: int, device: str, name: str) -> tuple:
    """Build a fresh six-tuple compatible with the I0 benchmark worker.

    Policy construction deliberately continues through ``ruler._build_case``:
    opponent identities, forward microbatch shapes, and collection layout stay
    equal across arms.  Only the admitted engine routes change afterwards.
    """
    spec = arm_spec(name)
    with _arm(name):
        sim, policy, template, policy_template = ruler._build_case(
            checkpoint, "long-" + mix, worlds, device
        )
        for target in (sim, template):
            _set_move_backend(target, spec.move_backend)
        for target in (policy, policy_template):
            _install_geometry(target, spec.geometry)
        if bool(os.environ["GPU_SIM_V8_INDEXED"] == "1"):
            raise IntegrationEngineError("I0 must not enable the pending v8-bundle route")
    collection = ruler._collection(1, worlds, sim.device)
    return sim, policy, template, policy_template, collection, name


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def _value(value: Any) -> Any:
    """Typed, deterministic state inventory with no value normalization."""
    if isinstance(value, torch.Tensor):
        raw = _tensor_bytes(value)
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__qualname__, "fields": _value(dataclasses.asdict(value))}
    if isinstance(value, torch.device):
        return {"type": "device", "value": str(value)}
    if isinstance(value, torch.nn.Module):
        return {"type": type(value).__qualname__, "configuration_and_state": _value(vars(value))}
    if isinstance(value, Path):
        return {"type": "Path", "value": str(value)}
    if isinstance(value, dict):
        return {
            str(key): _value(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, set):
        return {"type": "set", "items": [_value(item) for item in sorted(value, key=repr)]}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [_value(item) for item in value]}
    if isinstance(value, float):
        return {"type": "float", "hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return {"type": type(value).__name__, "value": value}
    raise IntegrationEngineError(f"uninventoried persistent type: {type(value).__qualname__}")


def inventory(obj: Any) -> dict[str, Any]:
    """Inventory persistent state and reject a retained indexed temporary view."""
    result = {}
    for key, value in sorted(vars(obj).items()):
        if key in INSTRUMENTATION_FIELDS:
            continue
        if key in TRANSIENT_FIELDS:
            if value is not None:
                raise IntegrationEngineError("stale indexed carry: " + key)
            continue
        result[key] = _value(value)
    return result


def snapshot(case: tuple) -> dict[str, Any]:
    """Return full state evidence for a benchmark/oracle comparison."""
    sim, policy, _template, _policy_template, collection, _name = case
    policy_fields = {key: value for key, value in vars(policy).items() if key != "vector_runtime"}
    return {
        "sim": inventory(sim),
        "runtime": inventory(policy.vector_runtime),
        "policy": _value(
            {
                key: value
                for key, value in policy_fields.items()
                if key not in INSTRUMENTATION_FIELDS
            }
        ),
        "collection": _value(collection),
    }


def digest(case: tuple) -> str:
    """Return a stable digest of the complete post-tick state inventory."""
    import json

    return hashlib.sha256(json.dumps(snapshot(case), sort_keys=True).encode()).hexdigest()


def assert_oracle_separation(left: tuple, right: tuple) -> None:
    """Refuse an A/B proof that aliases mutable candidate and oracle state."""
    labels = ("sim", "policy", "sim_template", "policy_template", "collection")
    for index, label in enumerate(labels):
        if left[index] is right[index]:
            raise IntegrationEngineError("candidate and oracle share mutable " + label)
    for case in (left, right):
        if case[0] is case[2] or case[1] is case[3]:
            raise IntegrationEngineError("case aliases live state and reset template")
        if getattr(case[1], "vector_runtime", None) is getattr(case[3], "vector_runtime", None):
            raise IntegrationEngineError("case aliases live and template vector runtime")
