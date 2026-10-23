"""Immutable natural-collection allocation evidence for P1 Amendment 04."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from research.bl_ppo.collector import CollectorConfig, _world_seed
from research.bl_ppo.exports import export_json, verify_export
from research.bl_ppo.late_body import FIXED_HOLDOUT_WORLD_OFFSET, SUPPLEMENT_WORLD_OFFSET

_NATURAL_WORLDS = 128
_RESERVED_RANGES = (
    (SUPPLEMENT_WORLD_OFFSET, SUPPLEMENT_WORLD_OFFSET + _NATURAL_WORLDS),
    (FIXED_HOLDOUT_WORLD_OFFSET, FIXED_HOLDOUT_WORLD_OFFSET + _NATURAL_WORLDS),
)


def validate_offset(offset: int) -> None:
    """Refuse offsets that alter the registered split or overlap reserved banks."""
    if type(offset) is not int or offset <= 0 or offset % 5:
        raise ValueError("natural collection offset must be a positive multiple of five")
    end = offset + _NATURAL_WORLDS
    if any(offset < reserved_end and start < end for start, reserved_end in _RESERVED_RANGES):
        raise ValueError("natural collection offset overlaps a supplement/fixed-holdout range")


def _seed_digest(seeds: Iterable[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in sorted(set(int(seed) for seed in seeds))).encode()
    ).hexdigest()


def build_allocation_audit(
    *, offset: int, burned_roots: Iterable[int], episodes: int = 8, seed: int = 20261021
) -> dict[str, Any]:
    """Materialize the complete pre-data root allocation and prove no collision.

    Episodes are inclusive (0..8). Nine 5k-frame lives per world bound at least
    5.76M collector transitions, exceeding the 4M natural-refit cap even at the
    conservative one-active-life-per-world accounting.
    """
    validate_offset(offset)
    if type(episodes) is not int or episodes < 8:
        raise ValueError("allocation audit requires episodes 0..E with E >= 8")
    burned = sorted(set(int(seed_value) for seed_value in burned_roots))
    if any(type(seed_value) is not int for seed_value in burned_roots):
        raise ValueError("burned roots must be integers")
    raw_config = CollectorConfig(worlds=_NATURAL_WORLDS, processes=1, seed=seed, preflight=True)
    allocated_config = CollectorConfig(
        worlds=_NATURAL_WORLDS,
        processes=1,
        seed=seed,
        excluded_seeds=frozenset(burned),
        preflight=True,
    )
    raw_roots, allocated_roots = [], []
    for world in range(_NATURAL_WORLDS):
        for episode in range(episodes + 1):
            world_id = offset + world
            raw_roots.append(_world_seed(raw_config, world_id, episode))
            allocated_roots.append(_world_seed(allocated_config, world_id, episode))
    collisions = sorted(set(raw_roots).intersection(burned))
    final_collisions = sorted(set(allocated_roots).intersection(burned))
    if collisions or final_collisions or len(set(allocated_roots)) != len(allocated_roots):
        raise ValueError("collection allocation has a non-disjoint natural root")
    return {
        "schema": "bl-p1-collection-allocation-audit/v1",
        "seed": seed,
        "natural_collection_world_id_offset": offset,
        "worlds": _NATURAL_WORLDS,
        "episodes_inclusive": {"start": 0, "end": episodes},
        "coverage": {
            "minimum_lives": _NATURAL_WORLDS * (episodes + 1),
            "maximum_transition_capacity": _NATURAL_WORLDS * (episodes + 1) * 5000,
            "statement": (
                "finite roots 0..E are audited before collection; runtime extends episodes "
                "through the exclusion loop as needed for the 4M cap; inaccessible "
                "protected-study history was not opened"
            ),
        },
        "burned_root_count": len(burned),
        "burned_roots": burned,
        "burned_roots_sha256": _seed_digest(burned),
        "raw_root_collision_count": len(collisions),
        "raw_root_collisions": collisions,
        "allocated_root_count": len(allocated_roots),
        "allocated_roots_sha256": _seed_digest(allocated_roots),
        "collision_count": len(collisions),
        "collision_roots": collisions,
        "mechanism": "collector._world_seed attempt loop with burned roots in excluded_seeds",
        "reserved_world_id_ranges": [list(item) for item in _RESERVED_RANGES],
    }


def publish_allocation_audit(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Create-only publish the audit and completion sidecars before dispatch."""
    return export_json(path, build_allocation_audit(**kwargs))


def validate_collection_control(
    control: dict[str, Any], directory: Path, *, actual_seed: int = 20261021
) -> None:
    """Bind runtime collection inputs to the immutable audited allocation."""
    offset = control.get("natural_collection_world_id_offset")
    validate_offset(offset)
    burned = control.get("natural_collection_burned_roots")
    if (
        not isinstance(burned, list)
        or burned != sorted(set(burned))
        or not all(type(value) is int for value in burned)
    ):
        raise ValueError("P1 Amendment-04 requires a sorted burned-root exclusion inventory")
    expected = control.get("collection_allocation_audit_sha256")
    audit_name = control.get("collection_allocation_audit_path", "collection-allocation-audit.json")
    if not isinstance(audit_name, str) or Path(audit_name).name != audit_name:
        raise ValueError("P1 Amendment-04 allocation audit name is unsafe")
    audit_path = directory / audit_name
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or verify_export(audit_path) != expected
    ):
        raise ValueError("P1 Amendment-04 collection allocation audit is not bound")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema") != "bl-p1-collection-allocation-audit/v1"
        or audit.get("natural_collection_world_id_offset") != offset
        or audit.get("seed") != actual_seed
        or audit.get("burned_roots") != burned
        or audit.get("collision_count") != 0
        or audit.get("allocated_root_count", 0) < _NATURAL_WORLDS * 9
    ):
        raise ValueError(
            "P1 Amendment-04 collection allocation audit does not prove control identity"
        )
