"""Focused contracts for the T6 restart fresh-world audit."""

from __future__ import annotations

import json
from pathlib import Path

from research.t6_restart_20261016 import seed_audit

ARTIFACTS = Path("artifacts-not-required-for-training-inventory-test")
COMMITTED_T6_RECEIPT = Path("docs/research/t6_seed_audit_2026-10-16.json")


def _snapshot_store(monkeypatch, tmp_path: Path) -> tuple[Path, set[int]]:
    """Build a small, independently hashed snapshot manifest for the audit contract."""
    seeds = {101, 202}
    rows = [
        {"file": f"b8_s0-scripted-c0-i{index}-s{seed}-d0.npz"}
        for index, seed in enumerate(sorted(seeds))
    ]
    store = tmp_path / "snapshot"
    store.mkdir()
    manifest = store / "manifest.json"
    manifest.write_text(json.dumps({"schema": "t4-snapshot-store/v2", "snapshots": rows}))
    monkeypatch.setattr(seed_audit, "SNAPSHOT_MANIFEST_SHA256", seed_audit.sha256(manifest))
    monkeypatch.setattr(seed_audit, "SNAPSHOT_COUNT", len(rows))
    monkeypatch.setattr(seed_audit, "SNAPSHOT_SOURCE_WORLD_COUNT", len(seeds))
    return store, seeds


def _collection_store(monkeypatch, tmp_path: Path) -> tuple[Path, set[int]]:
    """Build a complete mini T4e bank, including plans and disjoint published splits."""
    seeds = {301, 302}
    store = tmp_path / "collection"
    plans = store / "plans"
    plans.mkdir(parents=True)
    jobs = [{"plan_index": 0, "seed": 301}, {"plan_index": 1, "seed": 302}]
    for cycle in range(6):
        (plans / f"cycle-{cycle:04d}.json").write_text(
            json.dumps({"jobs": jobs if cycle == 0 else []})
        )
    for kind, receipt in (
        ("train", "world-00000-B8_s0-scripted-s301-outside.npz"),
        ("holdout", "world-00001-B8_s0-scripted-s302-outside.npz"),
    ):
        manifest = store / "banks" / kind / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "collection_schema": "t4e-return-bank/v1",
                    "schema_version": 1,
                    "publication": {"world_receipts": {receipt: {}}},
                }
            )
        )
    monkeypatch.setattr(seed_audit, "COLLECTION_WORLD_COUNT", len(jobs))
    monkeypatch.setattr(seed_audit.t4f_audit, "COLLECTION_WORLD_COUNT", len(jobs))
    return store, seeds


def _committed_t4e_collection_seeds() -> set[int]:
    """Rebuild the six recorded T4e allocation namespaces without external artifacts."""
    from research.t4e_return_20261016.run_collection import _candidate_seed

    root_name = "t4e-return-bank-20261016"
    prefix = f"t4e-return-collection/2026-10-16/{root_name}"
    return {
        _candidate_seed(f"{prefix}/cycle-{cycle}", index)
        for cycle in range(6)
        for index in range(200)
    }


def test_production_snapshot_constants_match_the_committed_t6_receipt() -> None:
    receipt = json.loads(COMMITTED_T6_RECEIPT.read_text())
    snapshot = receipt["snapshot_store"]

    assert seed_audit.SNAPSHOT_MANIFEST_SHA256 == snapshot["manifest_sha256"]
    assert seed_audit.SNAPSHOT_COUNT == snapshot["snapshot_count"]
    assert seed_audit.SNAPSHOT_SOURCE_WORLD_COUNT == snapshot["source_world_seed_count"]
    assert seed_audit.COLLECTION_WORLD_COUNT == receipt["t4e_collection"]["planned_world_count"]
    assert (
        seed_audit.t4f_audit.COLLECTION_WORLD_COUNT
        == receipt["t4e_collection"]["planned_world_count"]
    )


def test_full_snapshot_store_is_bound_and_source_seeds_are_explicit(
    monkeypatch, tmp_path: Path
) -> None:
    store, expected_seeds = _snapshot_store(monkeypatch, tmp_path)
    receipt, seeds = seed_audit.snapshot_store_audit(store)

    assert receipt["pass"] is True
    assert receipt["manifest_sha256"] == seed_audit.SNAPSHOT_MANIFEST_SHA256
    assert receipt["snapshot_count"] == seed_audit.SNAPSHOT_COUNT
    assert receipt["source_world_seed_count"] == seed_audit.SNAPSHOT_SOURCE_WORLD_COUNT
    assert receipt["source_world_seeds"] == sorted(expected_seeds)
    assert seeds == expected_seeds


def test_snapshot_manifest_fails_closed_when_a_snapshot_filename_is_not_bound(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "t4-snapshot-store/v2", "snapshots": [{}]}))

    receipt, seeds = seed_audit.snapshot_store_audit(tmp_path)

    assert receipt["pass"] is False
    assert receipt["malformed_snapshot_files"] == ["None"]
    assert seeds == set()


def test_t4e_collection_audit_binds_all_planned_worlds(monkeypatch, tmp_path: Path) -> None:
    store, expected_seeds = _collection_store(monkeypatch, tmp_path)
    receipt, seeds = seed_audit.t4e_collection_audit(store)

    assert receipt["pass"] is True
    assert receipt["planned_world_count"] == seed_audit.COLLECTION_WORLD_COUNT
    assert seeds == expected_seeds


def test_t6_rosters_are_distinct_and_exclude_snapshot_and_collection_worlds(
    monkeypatch, tmp_path: Path
) -> None:
    from research.redesign_m2_20261008 import ni_spec

    snapshot_store, snapshot_seeds = _snapshot_store(monkeypatch, tmp_path)
    collection_store, expected_collection_seeds = _collection_store(monkeypatch, tmp_path)
    _snapshot, actual_snapshot_seeds = seed_audit.snapshot_store_audit(snapshot_store)
    _collection, collection_seeds = seed_audit.t4e_collection_audit(collection_store)
    initial = set(ni_spec.distill_seeds(96, round_index=seed_audit.INITIAL_ROUND))
    extension = set(ni_spec.distill_seeds(96, round_index=seed_audit.EXTENSION_ROUND))

    assert len(initial) == len(extension) == 96
    assert not (initial & extension)
    assert actual_snapshot_seeds == snapshot_seeds
    assert collection_seeds == expected_collection_seeds
    assert not (initial | extension) & snapshot_seeds
    assert not (initial | extension) & collection_seeds


def test_t6_rosters_exclude_the_committed_registered_snapshot_collection_and_prior_rounds() -> None:
    """Keep the actual registered exclusion separate from the portable mini-store fixture."""
    from research.redesign_m2_20261008 import ni_spec

    receipt = json.loads(COMMITTED_T6_RECEIPT.read_text())
    snapshot = set(receipt["snapshot_store"]["source_world_seeds"])
    collection = _committed_t4e_collection_seeds()
    initial = set(ni_spec.distill_seeds(96, round_index=seed_audit.INITIAL_ROUND))
    extension = set(ni_spec.distill_seeds(96, round_index=seed_audit.EXTENSION_ROUND))
    prior_rounds = set().union(
        *(set(ni_spec.distill_seeds(96, round_index=round_index)) for round_index in range(68, 72))
    )

    assert seed_audit.seed_digest(snapshot) == receipt["snapshot_store"]["source_world_seed_digest"]
    assert len(collection) == receipt["t4e_collection"]["planned_world_count"]
    assert (
        seed_audit.seed_digest(collection) == receipt["t4e_collection"]["source_world_seed_digest"]
    )
    assert not (initial | extension) & snapshot
    assert not (initial | extension) & collection
    assert not (initial | extension) & prior_rounds


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
