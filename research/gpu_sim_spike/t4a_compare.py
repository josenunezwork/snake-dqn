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


@dataclass(frozen=True)
class SchemaIncompatibility:
    """The two captures cannot support a byte-level scientific comparison.

    This is deliberately separate from :class:`Mismatch`: an absent field or
    changed tensor representation is harness evidence failure, not evidence
    that two otherwise comparable engine executions produced unequal bytes.
    """

    field: str
    reason: str
    expected_dtype: str | None = None
    actual_dtype: str | None = None
    expected_shape: tuple[int, ...] | None = None
    actual_shape: tuple[int, ...] | None = None


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


BODY_DIGEST_ALGORITHM = "t4a-device-mix64x4-v1"
BODY_DIGEST_CHUNK_BYTES = 1024**2
# Four live int64 vectors (bytes, positions, mix, shift), plus bounded scalars.
BODY_DIGEST_SCRATCH_BYTES = 4 * 8 * BODY_DIGEST_CHUNK_BYTES + 256


def _body_digest_chunk(raw: torch.Tensor, offset: int) -> torch.Tensor:
    """Reduce one byte window using its GLOBAL positions and signed shifts."""
    raw = raw.to(torch.int64)
    positions = torch.arange(offset, offset + raw.numel(), dtype=torch.int64, device=raw.device)
    lanes = []
    for salt in (17, 257, 65537, 16777259):
        value = positions.clone()
        value.add_(salt).mul_(6364136223846793005).add_(raw)
        value.bitwise_xor_(value >> 29).mul_(2862933555777941757)
        value.bitwise_xor_(value >> 31).mul_(3202034522624059733)
        value.bitwise_xor_(value >> 27)
        lanes.append(value.sum())
        del value
    return torch.stack(lanes)


def _body_digest_lanes(tensor: torch.Tensor) -> torch.Tensor:
    """Bound scratch independently of ring size; keep all bytes on device."""
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    lanes = torch.zeros(4, dtype=torch.int64, device=tensor.device)
    for offset in range(0, raw.numel(), BODY_DIGEST_CHUNK_BYTES):
        lanes.add_(_body_digest_chunk(raw[offset : offset + BODY_DIGEST_CHUNK_BYTES], offset))
    return lanes


def body_digest(tensor: torch.Tensor) -> str:
    """Fingerprint ring bytes on device, transferring exactly 32 bytes.

    Four position-sensitive integer reductions are diagnostic fingerprints,
    not cryptographic hashes or the equality oracle. Raw byte comparison stays
    on device in ``compare``. Signed int64 overflow is intentional modulo 2**64;
    no floating arithmetic or host body-ring serialization is involved. Chunk
    sums compose exactly in Z/(2**64); positions never restart at a boundary.
    Contiguous rings need at most BODY_DIGEST_SCRATCH_BYTES tensor scratch
    (excluding backend reduction workspace/allocator reservation). As before,
    noncontiguous callers additionally require a contiguous input copy.
    """
    return _body_digest_lanes(tensor).view(torch.uint8).cpu().numpy().tobytes().hex()


def compare(
    left: torch.Tensor, right: torch.Tensor, field: str
) -> Mismatch | SchemaIncompatibility | None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return SchemaIncompatibility(
            field,
            "dtype_or_shape",
            str(left.dtype),
            str(right.dtype),
            tuple(left.shape),
            tuple(right.shape),
        )
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
) -> Mismatch | SchemaIncompatibility | None:
    if set(left) != set(right):
        return SchemaIncompatibility(
            "inventory",
            "field_inventory",
        )
    # Schema admission is a precondition for *every* byte comparison.  Do not
    # report an earlier valid byte delta while a later field has incompatible
    # representation: that would turn a harness defect into a scientific
    # self-consistency verdict.
    for field in sorted(left):
        if left[field].dtype != right[field].dtype or tuple(left[field].shape) != tuple(
            right[field].shape
        ):
            return SchemaIncompatibility(
                field,
                "dtype_or_shape",
                str(left[field].dtype),
                str(right[field].dtype),
                tuple(left[field].shape),
                tuple(right[field].shape),
            )
    for field in sorted(left):
        mismatch = compare(left[field], right[field], field)
        if mismatch is not None:
            return mismatch
    return None


def compact_inventory(values: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            **(
                {"digest_algorithm": BODY_DIGEST_ALGORITHM, "digest": body_digest(value)}
                if name == "sim.bodies"
                else {"sha256": digest(value)}
            ),
        }
        for name, value in sorted(values.items())
    }
