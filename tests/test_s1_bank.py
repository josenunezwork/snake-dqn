"""Focused checks for the S1 immutable allocation boundary."""

from __future__ import annotations

import json

import pytest

from research.s1_mature_20261018 import bank


def test_shared_final_look_has_640_worlds_and_1920_policy_episodes() -> None:
    value = bank.build_world_bank()
    worlds = [row for seed in bank.TRAINING_SEEDS for row in bank.worlds_for_seed(value, seed)]
    assert len(worlds) == 640
    assert len({row["world_seed"] for row in worlds}) == 640
    assert len(worlds) * 3 == 1920  # N3 final, N1 final, and shared B8 comparator.
    assert len(value["timing_worlds"]) == 10
    assert not {row["world_seed"] for row in worlds} & {
        row["world_seed"] for row in value["timing_worlds"]
    }


def test_each_seed_has_registered_mix_counts_and_collection_namespaces() -> None:
    value = bank.build_world_bank()
    for seed in bank.TRAINING_SEEDS:
        rows = bank.worlds_for_seed(value, seed)
        assert {
            mix: sum(row["mix"] == mix for row in rows) for mix in bank.MIX_COUNTS
        } == bank.MIX_COUNTS
        assert bank.training_namespace("N3", seed).endswith(f"/seed-{seed}")
        assert bank.training_namespace("N1", seed).startswith("q1-task-aligned-n1-20261017/")


def test_loader_rejects_tampered_semantic_manifest(tmp_path) -> None:
    value = bank.build_world_bank()
    value["worlds"][str(bank.TRAINING_SEEDS[0])][0]["world_seed"] += 1
    path = tmp_path / "WORLD_BANK.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="semantic digest"):
        bank.load_world_bank(path)


def test_checked_in_audit_passes_and_records_required_coverage() -> None:
    audit = json.loads(bank.AUDIT_OUTPUT.read_text())
    assert audit["pass"] is True
    assert audit["candidate_count"] == 640
    assert audit["registered_collision_seeds"] == []
    assert not any(audit["namespace_collision_files"].values())
    assert any("Q1" in entry for entry in audit["coverage"])
