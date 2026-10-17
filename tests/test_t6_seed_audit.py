"""Focused contracts for the T6 restart fresh-world audit."""

from __future__ import annotations

import json
from pathlib import Path

from research.t6_restart_20261016 import seed_audit

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
SNAPSHOT_STORE = ARTIFACTS / "t4b-snapshot-restart-20261016/full"
T4E_STORE = ARTIFACTS / "t4e-return-bank-20261016"


def test_full_snapshot_store_is_bound_and_source_seeds_are_explicit() -> None:
    receipt, seeds = seed_audit.snapshot_store_audit(SNAPSHOT_STORE)

    assert receipt["pass"] is True
    assert receipt["manifest_sha256"] == seed_audit.SNAPSHOT_MANIFEST_SHA256
    assert receipt["snapshot_count"] == seed_audit.SNAPSHOT_COUNT
    assert receipt["source_world_seed_count"] == seed_audit.SNAPSHOT_SOURCE_WORLD_COUNT
    assert receipt["source_world_seeds"] == sorted(seeds)


def test_snapshot_manifest_fails_closed_when_a_snapshot_filename_is_not_bound(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "t4-snapshot-store/v2", "snapshots": [{}]}))

    receipt, seeds = seed_audit.snapshot_store_audit(tmp_path)

    assert receipt["pass"] is False
    assert receipt["malformed_snapshot_files"] == ["None"]
    assert seeds == set()


def test_t4e_collection_audit_binds_all_planned_worlds() -> None:
    receipt, seeds = seed_audit.t4e_collection_audit(T4E_STORE)

    assert receipt["pass"] is True
    assert receipt["planned_world_count"] == seed_audit.COLLECTION_WORLD_COUNT
    assert len(seeds) == seed_audit.COLLECTION_WORLD_COUNT


def test_t6_rosters_are_distinct_and_exclude_snapshot_and_collection_worlds() -> None:
    from research.redesign_m2_20261008 import ni_spec

    _snapshot, snapshot_seeds = seed_audit.snapshot_store_audit(SNAPSHOT_STORE)
    _collection, collection_seeds = seed_audit.t4e_collection_audit(T4E_STORE)
    initial = set(ni_spec.distill_seeds(96, round_index=seed_audit.INITIAL_ROUND))
    extension = set(ni_spec.distill_seeds(96, round_index=seed_audit.EXTENSION_ROUND))

    assert len(initial) == len(extension) == 96
    assert not (initial & extension)
    assert not (initial | extension) & snapshot_seeds
    assert not (initial | extension) & collection_seeds


def test_t6_scanner_finds_reserved_seed_and_metadata_in_ignored_and_hidden_paths(
    tmp_path: Path,
) -> None:
    ignored = tmp_path / "logs" / "result.json"
    hidden = tmp_path / ".hidden" / "result.json"
    ignored.parent.mkdir()
    hidden.parent.mkdir()
    (tmp_path / ".gitignore").write_text("logs/\n")
    ignored.write_text('{"seed": 123456789}')
    hidden.write_text('{"round": 72}')

    hits, _progress = seed_audit.concrete_text_hits(tmp_path, ["123456789"])
    found_metadata, _metadata_progress = seed_audit._reserved_round_metadata(tmp_path, ())

    assert hits == {"123456789"}
    assert found_metadata is True


def test_training_inventory_checks_fixed_t6_budget_against_candidate_rosters() -> None:
    from research.redesign_m2_20261008 import ni_spec

    candidates = set(ni_spec.distill_seeds(96, round_index=72))
    receipt = seed_audit.training_namespace_audit(Path.cwd(), ARTIFACTS, candidates)

    assert receipt["pass"] is True
    assert receipt["candidate_collisions"] == []
    assert any(row["run_seed"] == 10 for row in receipt["records"])
