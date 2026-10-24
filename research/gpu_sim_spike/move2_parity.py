"""Byte-level frozen-device identity helpers for the prospective MOVE2 packet.

These helpers deliberately compare representation bytes, including padding and
Python lifecycle state.  They are not a CPU-correctness oracle.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

FROZEN_DEVICE_SIM = Path(__file__).with_name("fixtures") / "move" / "device_sim_pre_move.py"
FROZEN_DEVICE_SIM_SHA256 = "816ebd5fbd4016bb01eda5fe31ee7739f8cb7bf021b2885684f769ea63dc58ae"
_SKIP = {"device", "cfg", "move_backend", "stage_timing", "_timing"}


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_inventory(sim: Any) -> dict[str, dict[str, Any]]:
    """Return lossless tensor metadata; no values are normalized or omitted."""
    result: dict[str, dict[str, Any]] = {}
    for name, value in sorted(vars(sim).items()):
        if isinstance(value, torch.Tensor):
            raw = value.detach().cpu().contiguous().numpy().tobytes()
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.detach().cpu().numpy().dtype),
                "bytes": len(raw),
                "sha256": digest_bytes(raw),
            }
    return result


def _non_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return None
    if dataclasses.is_dataclass(value):
        return {key: _non_tensor(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _non_tensor(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_non_tensor(item) for item in value]
    raise TypeError(f"unserializable persistent state: {type(value).__qualname__}")


def _type_inventory(value: Any) -> Any:
    """Parallel type record preserves distinctions JSON values intentionally erase."""
    if dataclasses.is_dataclass(value):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                key: _type_inventory(item) for key, item in dataclasses.asdict(value).items()
            },
        }
    if isinstance(value, dict):
        return {
            str(key): _type_inventory(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, tuple):
        return {"container": "tuple", "items": [_type_inventory(item) for item in value]}
    if isinstance(value, list):
        return {"container": "list", "items": [_type_inventory(item) for item in value]}
    if isinstance(value, float):
        return {"type": "float", "hex": value.hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return {"type": type(value).__qualname__}
    if isinstance(value, Path):
        return {"type": "Path"}
    if isinstance(value, torch.Tensor):
        return None
    raise TypeError(f"unserializable persistent state: {type(value).__qualname__}")


def state_descriptor(sim: Any) -> dict[str, Any]:
    """Small deterministic descriptor sufficient to reproduce an exact probe."""
    fields = {
        name: _non_tensor(value)
        for name, value in sorted(vars(sim).items())
        if name not in _SKIP and not isinstance(value, torch.Tensor)
    }
    # Timing is diagnostic-only but its enablement changes execution.  Do not
    # serialize clocks/counters or object reprs; assert it is disabled instead.
    timing = getattr(sim, "stage_timing", getattr(sim, "_timing", None))
    if timing is not None:
        fields["stage_timing"] = {
            "enabled": bool(getattr(timing, "enabled", False)),
            "disabled": not bool(getattr(timing, "enabled", False)),
        }
    fields["cfg"] = _non_tensor(getattr(sim, "cfg", None))
    types = {
        name: _type_inventory(value)
        for name, value in sorted(vars(sim).items())
        if name not in _SKIP and not isinstance(value, torch.Tensor)
    }
    types["cfg"] = _type_inventory(getattr(sim, "cfg", None))
    if timing is not None:
        types["stage_timing"] = {"type": type(timing).__qualname__, "enabled": "bool"}
    return {
        "tensors": tensor_inventory(sim),
        "non_tensors": fields,
        "non_tensor_type_inventory": types,
    }


def compare(left: Any, right: Any) -> dict[str, Any]:
    """Compare every tensor and all persistent Python state byte-for-byte."""
    left_descriptor, right_descriptor = state_descriptor(left), state_descriptor(right)
    left_non_tensor = json.dumps(
        {
            "values": left_descriptor["non_tensors"],
            "types": left_descriptor["non_tensor_type_inventory"],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    right_non_tensor = json.dumps(
        {
            "values": right_descriptor["non_tensors"],
            "types": right_descriptor["non_tensor_type_inventory"],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if left_non_tensor != right_non_tensor:
        return {
            "match": False,
            "stage": "persistent_non_tensor_state",
            "left": left_descriptor["non_tensors"],
            "right": right_descriptor["non_tensors"],
        }
    left_tensors, right_tensors = left_descriptor["tensors"], right_descriptor["tensors"]
    if left_tensors.keys() != right_tensors.keys():
        return {
            "match": False,
            "stage": "tensor_inventory",
            "left": sorted(left_tensors),
            "right": sorted(right_tensors),
        }
    for name in left_tensors:
        if left_tensors[name] != right_tensors[name]:
            return {
                "match": False,
                "stage": "tensor",
                "field": name,
                "left": left_tensors[name],
                "right": right_tensors[name],
            }
    encoded = json.dumps(right_descriptor, sort_keys=True, separators=(",", ":")).encode()
    return {"match": True, "fields": len(right_tensors), "state_sha256": digest_bytes(encoded)}


def field_record(cpu: Any, device: Any) -> dict[str, Any]:
    """Record original array dtypes and bytes for an immutable discrepancy entry."""
    left, right = np.asarray(cpu), np.asarray(device)
    return {
        "match": left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes(),
        "cpu": {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "sha256": digest_bytes(left.tobytes()),
        },
        "device": {
            "shape": list(right.shape),
            "dtype": str(right.dtype),
            "sha256": digest_bytes(right.tobytes()),
        },
    }
