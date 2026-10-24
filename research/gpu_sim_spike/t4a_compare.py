"""Device-byte comparison primitives for the frozen T4A field inventory."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

TRANSIENT = {"_geometry_sim", "_geometry_body_view", "_timing", "host_stages", "stage_timing"}
SIM_WORLD = {
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "rng_counter",
    "rng_key",
    "food_cells",
    "food_present",
    "corpse_food",
    "food_count",
    "_last_reward",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
    "_boosted_this_step",
    "terminal_slots",
    "capacity_overflow",
}
RUNTIME_WORLD = {
    "controlled",
    "prev_dist",
    "prev_id",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
}
RUNTIME_CONSTANTS = {
    "device",
    "params",
    "_params",
    "_cpu_sector_lut",
    "_cpu_sector_rx",
    "_cpu_sector_ry",
    "_free_ratio",
    "fresh_rows",
    "carried_rows",
}
SIM_CONSTANTS = {
    "device",
    "cfg",
    "E",
    "S",
    "cap",
    "grid_w",
    "grid_h",
    "food_capacity",
    "v2",
    "train_mode",
    "allow_respawn",
    "move_backend",
    "rng_metadata",
    "terminal_slots",
    "_prepared",
}


@dataclass(frozen=True)
class Mismatch:
    field: str
    byte_offset: int
    expected: int
    actual: int


def _tensor(value: Any, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"unknown non-tensor persistent field {field}")
    return value.contiguous()


def persistent_tensors(owner: Any, prefix: str) -> dict[str, torch.Tensor]:
    """Inventory every tensor persistent field and refuse stale transient views."""
    result: dict[str, torch.Tensor] = {}
    for name, value in sorted(vars(owner).items()):
        if name in {"advisory_mask", "prepare"} and getattr(
            value, "__t4a_instrumentation__", False
        ):
            continue
        if name in TRANSIENT:
            if value is not None and name.startswith("_geometry"):
                raise ValueError(f"nonempty transient field {prefix}.{name}")
            continue
        if prefix == "runtime" and name in RUNTIME_CONSTANTS:
            continue
        if isinstance(value, torch.Tensor):
            expected = SIM_WORLD if prefix == "sim" else RUNTIME_WORLD
            if name not in expected:
                raise ValueError(f"unknown persistent tensor {prefix}.{name}")
            result[f"{prefix}.{name}"] = value.contiguous()
        elif name in SIM_CONSTANTS | RUNTIME_CONSTANTS | {
            "vector_prepared",
            "v8_lambda",
            "anchor_kinds",
            "anchor_tables",
            "b8_network",
            "opponent_networks",
            "reference_bridge",
            "port_limitations",
        }:
            continue
        elif value is not None:
            raise ValueError(f"unknown persistent field {prefix}.{name}")
    return result


def world_tensors(owner: Any, prefix: str, worlds: int) -> dict[str, torch.Tensor]:
    """Return the frozen complete E-indexed inventory and reject omissions."""
    persistent_tensors(owner, prefix)  # Reject unknown scalar/object state too.
    expected = SIM_WORLD if prefix == "sim" else RUNTIME_WORLD
    result: dict[str, torch.Tensor] = {}
    for name in expected:
        value = getattr(owner, name, None)
        if not isinstance(value, torch.Tensor) or not value.ndim or int(value.shape[0]) != worlds:
            raise ValueError(f"missing or malformed {prefix} world field {name}")
        result[f"{prefix}.{name}"] = value.contiguous()
    for name, value in vars(owner).items():
        if isinstance(value, torch.Tensor) and value.ndim and int(value.shape[0]) == worlds:
            if name not in expected and name not in {"_cpu_sector_lut", "_free_ratio"}:
                raise ValueError(f"unknown {prefix} E-indexed tensor {name}")
    return result


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()


def compare(left: torch.Tensor, right: torch.Tensor, field: str) -> Mismatch | None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return Mismatch(field + ".shape_or_dtype", 0, 0, 0)
    a, b = left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(
        torch.uint8
    )
    unequal = torch.ne(a, b).flatten()
    if not bool(unequal.any().item()):
        return None
    index = int(unequal.nonzero(as_tuple=False)[0].item())
    return Mismatch(field, index, int(a.flatten()[index].item()), int(b.flatten()[index].item()))


def compare_inventory(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> Mismatch | None:
    if set(left) != set(right):
        return Mismatch("inventory", 0, 0, 0)
    for field in sorted(left):
        mismatch = compare(left[field], right[field], field)
        if mismatch is not None:
            return mismatch
    return None


def compact_inventory(values: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": digest(value)}
        for name, value in sorted(values.items())
    }
