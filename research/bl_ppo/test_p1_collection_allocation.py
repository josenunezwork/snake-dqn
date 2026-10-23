"""Focused C3 allocation contracts."""

from __future__ import annotations

import json

import pytest

from research.bl_ppo.exports import digest
from research.bl_ppo.p1_collection_allocation import (
    advance_allocation_offsets,
    build_allocation_audit,
    build_allocation_audit_pair,
    enumerate_seed_family,
    publish_allocation_audit,
    validate_collection_control,
)


def test_collection_audit_is_disjoint_and_binds_its_control(tmp_path) -> None:
    path = tmp_path / "collection-allocation-audit.json"
    publish_allocation_audit(path, offset=3_001_000, burned_roots=[], episodes=127)
    control = {
        "natural_collection_world_id_offset": 3_001_000,
        "natural_collection_burned_roots": [],
        "collection_allocation_audit_sha256": digest(path),
    }
    validate_collection_control(control, tmp_path)
    audit = json.loads(path.read_text())
    assert audit["collision_count"] == 0
    assert audit["allocated_root_count"] == 128 * 128


def test_raw_collision_rejects_the_fixed_offset() -> None:
    # A real raw root from the fixed allocation must reject rather than silently retry it.
    from research.bl_ppo.collector import CollectorConfig, _world_seed

    raw = _world_seed(CollectorConfig(worlds=128, processes=1, preflight=True), 3_001_000, 0)
    with pytest.raises(ValueError, match="non-disjoint"):
        build_allocation_audit(offset=3_001_000, burned_roots=[raw], episodes=127)


def test_offset_rejects_reserved_ranges() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        build_allocation_audit(offset=1_000_000, burned_roots=[])


def test_audit_seed_is_bound_to_execution_context_not_control(tmp_path) -> None:
    path = tmp_path / "collection-allocation-audit.json"
    publish_allocation_audit(path, offset=3_001_000, burned_roots=[], episodes=127, seed=2026102203)
    control = {
        "natural_collection_world_id_offset": 3_001_000,
        "natural_collection_seed": 2026102203,
        "natural_collection_burned_roots": [],
        "collection_allocation_audit_sha256": digest(path),
    }
    with pytest.raises(ValueError, match="does not prove"):
        validate_collection_control(control, tmp_path, actual_seed=20261021)


def test_c3_enumerates_full_seed_family_and_freezes_pair_advance() -> None:
    ranges = [{"seed": 20261021, "world_start": 0, "world_end": 127}]
    roots = enumerate_seed_family(ranges)
    # Roots are uint32; a recorded tuple may share a hash with another tuple.
    assert 0 < len(roots) <= 128 * 128 * 2
    canary, source = build_allocation_audit_pair(
        canary_offset=3_000_000,
        source_offset=3_001_000,
        burned_roots=[],
        collector_seed_ranges=ranges,
    )
    assert canary["cross_lane"]["allocated_root_intersection_count"] == 0
    assert source["enumeration"]["attempt_end"] == 1
    assert advance_allocation_offsets(3_000_000, 3_001_000) == (3_010_000, 3_011_000)
