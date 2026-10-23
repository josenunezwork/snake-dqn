"""Immutable natural-collection allocation evidence for P1 Amendment 04."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research.bl_ppo.collector import CollectorConfig, _world_seed
from research.bl_ppo.exports import export_json, verify_export
from research.bl_ppo.late_body import FIXED_HOLDOUT_WORLD_OFFSET, SUPPLEMENT_WORLD_OFFSET

_NATURAL_WORLDS = 128
_FAMILY_EPISODE_END = 127
_FAMILY_ATTEMPT_END = 1
_OFFSET_ADVANCE = 10_000
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


def _root_at_attempt(seed: int, world: int, episode: int, attempt: int) -> int:
    """Return the collector root for one explicit, finite family tuple."""
    return int(np.random.SeedSequence([seed, 0xE2, world, episode, attempt]).generate_state(1)[0])


def enumerate_seed_family(
    collector_seed_ranges: Iterable[dict[str, int]],
    *,
    episode_end: int = _FAMILY_EPISODE_END,
    attempt_end: int = _FAMILY_ATTEMPT_END,
) -> list[int]:
    """Enumerate the frozen historical collector-root family without runtime state."""
    if episode_end != _FAMILY_EPISODE_END or attempt_end != _FAMILY_ATTEMPT_END:
        raise ValueError("C3 requires the fixed E=127 and A=1 seed-family enumeration")
    roots: set[int] = set()
    for item in collector_seed_ranges:
        seed, start, end = item.get("seed"), item.get("world_start"), item.get("world_end")
        if not all(type(value) is int for value in (seed, start, end)) or start < 0 or end < start:
            raise ValueError(
                "collector seed range must contain non-negative integer seed/world bounds"
            )
        for world in range(start, end + 1):
            for episode in range(episode_end + 1):
                for attempt in range(attempt_end + 1):
                    roots.add(_root_at_attempt(seed, world, episode, attempt))
    return sorted(roots)


def _family_tuple_count(collector_seed_ranges: Iterable[dict[str, int]]) -> int:
    """Count declared tuples before uint32 root deduplication."""
    return sum(
        (item["world_end"] - item["world_start"] + 1)
        * (_FAMILY_EPISODE_END + 1)
        * (_FAMILY_ATTEMPT_END + 1)
        for item in collector_seed_ranges
    )


def advance_allocation_offsets(canary_offset: int, source_offset: int) -> tuple[int, int]:
    """Advance both rejected lanes together, preserving their registered separation."""
    validate_offset(canary_offset)
    validate_offset(source_offset)
    if source_offset - canary_offset != 1_000:
        raise ValueError("C3 requires the fixed 1,000 world-id lane separation")
    return canary_offset + _OFFSET_ADVANCE, source_offset + _OFFSET_ADVANCE


def build_allocation_audit(
    *,
    offset: int,
    burned_roots: Iterable[int],
    episodes: int = _FAMILY_EPISODE_END,
    seed: int = 20261021,
    collector_seed_ranges: Iterable[dict[str, int]] = (),
) -> dict[str, Any]:
    """Materialize the complete pre-data root allocation and prove no collision.

    C3 fixes the audit to all 128 worlds, episodes 0..127, and attempts 0..1.
    Existing evaluator/supplement/fixed-holdout exclusions are supplied through
    ``burned_roots``; the complete historical natural family is added here.
    """
    validate_offset(offset)
    if type(episodes) is not int or episodes < 8:
        raise ValueError("allocation audit requires episodes 0..E with E >= 8")
    is_v3 = episodes == _FAMILY_EPISODE_END
    if collector_seed_ranges and not is_v3:
        raise ValueError("collector seed-family ranges require the fixed C3 E=127 audit")
    ranges = [dict(item) for item in collector_seed_ranges]
    family_roots = enumerate_seed_family(ranges) if is_v3 else []
    burned = sorted(set(int(seed_value) for seed_value in burned_roots) | set(family_roots))
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
        "schema": (
            "bl-p1-collection-allocation-audit/v3"
            if is_v3
            else "bl-p1-collection-allocation-audit/v1"
        ),
        "seed": seed,
        "natural_collection_world_id_offset": offset,
        "worlds": _NATURAL_WORLDS,
        "episodes_inclusive": {"start": 0, "end": episodes},
        "enumeration": {
            "collector_seed_ranges": ranges,
            "episode_start": 0,
            "episode_end": _FAMILY_EPISODE_END,
            "attempt_start": 0,
            "attempt_end": _FAMILY_ATTEMPT_END,
            "family_tuple_count": _family_tuple_count(ranges),
            "family_root_count": len(family_roots),
        },
        "coverage": {
            "minimum_lives": _NATURAL_WORLDS * (episodes + 1),
            "maximum_transition_capacity": _NATURAL_WORLDS * (episodes + 1) * 5000,
            "statement": (
                "C3 enumerates every declared historical natural root for E=127/A=1; "
                "runtime keeps the collector exclusion loop for later roots. Protected "
                "t34-round0-study-01 was not opened."
            ),
        },
        "burned_root_count": len(burned),
        "burned_roots": burned,
        "burned_roots_sha256": _seed_digest(burned),
        "raw_root_collision_count": len(collisions),
        "raw_root_collisions": collisions,
        "allocated_root_count": len(allocated_roots),
        "allocated_roots": sorted(allocated_roots),
        "allocated_roots_sha256": _seed_digest(allocated_roots),
        "collision_count": len(collisions),
        "collision_roots": collisions,
        "mechanism": "collector._world_seed attempt loop with burned roots in excluded_seeds",
        "offset_advance_rule": {
            "advance_both_offsets_by": _OFFSET_ADVANCE,
            "lane_separation": 1_000,
            "trigger": (
                "any enumerated collision rejects the whole allocation; never replace worlds"
            ),
        },
        "reserved_world_id_ranges": [list(item) for item in _RESERVED_RANGES],
    }


def build_allocation_audit_pair(
    *,
    canary_offset: int,
    source_offset: int,
    burned_roots: Iterable[int],
    collector_seed_ranges: Iterable[dict[str, int]],
    canary_seed: int = 2026102203,
    source_seed: int = 20261021,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build both C3 lanes and bind their exact cross-lane intersection proof."""
    if source_offset - canary_offset != 1_000:
        raise ValueError("C3 requires the fixed 1,000 world-id lane separation")
    kwargs = {"burned_roots": burned_roots, "collector_seed_ranges": collector_seed_ranges}
    canary = build_allocation_audit(offset=canary_offset, seed=canary_seed, **kwargs)
    source = build_allocation_audit(offset=source_offset, seed=source_seed, **kwargs)
    raw = sorted(set(canary["allocated_roots"]).intersection(source["allocated_roots"]))
    if raw:
        raise ValueError("collection allocation has a cross-lane collision")
    canary["cross_lane"] = {
        "other_lane": "source",
        "other_lane_seed": source_seed,
        "other_lane_offset": source_offset,
        "raw_root_intersection_count": 0,
        "allocated_root_intersection_count": 0,
    }
    source["cross_lane"] = {
        "other_lane": "canary",
        "other_lane_seed": canary_seed,
        "other_lane_offset": canary_offset,
        "raw_root_intersection_count": 0,
        "allocated_root_intersection_count": 0,
    }
    return canary, source


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
        audit.get("schema")
        not in {
            "bl-p1-collection-allocation-audit/v1",
            "bl-p1-collection-allocation-audit/v3",
        }
        or audit.get("natural_collection_world_id_offset") != offset
        or audit.get("seed") != actual_seed
        or audit.get("burned_roots") != burned
        or audit.get("collision_count") != 0
        or audit.get("allocated_root_count", 0) < _NATURAL_WORLDS * 9
    ):
        raise ValueError(
            "P1 Amendment-04 collection allocation audit does not prove control identity"
        )
    if (
        audit["schema"].endswith("/v3")
        and audit.get("cross_lane") is not None
        and (
            audit.get("enumeration", {}).get("episode_end") != _FAMILY_EPISODE_END
            or audit.get("enumeration", {}).get("attempt_end") != _FAMILY_ATTEMPT_END
            or audit.get("cross_lane", {}).get("raw_root_intersection_count") != 0
            or audit.get("cross_lane", {}).get("allocated_root_intersection_count") != 0
        )
    ):
        raise ValueError("C3 v3 audit does not prove the fixed family/cross-lane closure")
