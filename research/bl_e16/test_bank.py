"""Synthetic-only finite-bank, sealed traversal and fail-closed scanner checks."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from research.bl_e16 import bank


@pytest.fixture(scope="module")
def allocation() -> dict:
    return bank.materialize()


def test_frozen_full_allocation_and_no_uint32_truncation(allocation: dict) -> None:
    bank.validate_bank(allocation)
    assert len(allocation["rows"]) == 288
    assert len(allocation["excluded_timing_rows"]) == 9
    assert len(allocation["excluded_preflight_rows"]) == 3
    assert len(bank.all_seeds(allocation)) == 24_876
    assert all(2**48 <= seed < 2**49 < 2**63 for seed in bank.all_seeds(allocation))
    assert bank.materialize(1)["allocation_sha256"] != allocation["allocation_sha256"]
    from src.simd_env.rng import EnvRng

    seed = allocation["rows"][0]["world_seed"]
    left, right = EnvRng(seed, 1450, 830, 10, 0), EnvRng(seed + 2**32, 1450, 830, 10, 0)
    assert [left.get_random_position() for _ in range(5)] != [
        right.get_random_position() for _ in range(5)
    ]


def test_mutation_duplicate_and_partial_vectors_rejected(allocation: dict) -> None:
    for kind in ("roster", "seed", "missing"):
        value = copy.deepcopy(allocation)
        if kind == "roster":
            value["rows"][0]["opponents"][0][1] = "bad"
        elif kind == "seed":
            value["rows"][0]["world_seed"] = value["rows"][1]["world_seed"]
        else:
            value["phase_seeds"]["actor"]["scripted"].pop()
        with pytest.raises(bank.SeedAuditFailed):
            bank.validate_bank(value)


def test_finite_training_worlds_unique_pairable_and_exhaustible(allocation: dict) -> None:
    worlds = 128
    values = [
        bank.training_seed(allocation, "actor", world, episode, worlds)
        for episode in range(90)
        for world in range(worlds)
    ]
    assert len(values) == len(set(values))
    assert not set(values) & {row["world_seed"] for row in allocation["rows"]}
    assert bank.training_seed(allocation, "actor", 0, 0) == values[0]
    assert bank.training_seed(allocation, "critic", 0, 0) != values[0]
    with pytest.raises(bank.SeedAuditFailed, match="exhausted"):
        bank.training_seed(allocation, "actor", 0, 96)


def test_safe_files_never_descends_sealed_or_follows_symlinks(tmp_path: Path, monkeypatch) -> None:
    visible, sealed = tmp_path / "safe.json", tmp_path / bank.SEALED_DIRECTORY
    visible.write_text('{"seeds": [123456]}')
    sealed.mkdir()
    (sealed / "hidden.json").write_text('{"seeds": [999999]}')
    (tmp_path / "alias").symlink_to(sealed, target_is_directory=True)
    (tmp_path / "alias.json").symlink_to(sealed / "hidden.json")
    original = os.scandir

    def guarded(path):
        assert bank.SEALED_DIRECTORY not in Path(path).parts
        assert "alias" not in Path(path).parts
        return original(path)

    monkeypatch.setattr(os, "scandir", guarded)
    files = bank._safe_files([tmp_path])
    assert files == [visible]
    assert bank._scan_history({123456, 999999}, files)["hits"] == [123456]


def test_literal_scan_matches_numeric_tokens_only(tmp_path: Path) -> None:
    path = tmp_path / "values.json"
    path.write_text("[123456, 11234567, 999999]")
    result = bank._scan_history({123456, 234567, 999999}, [path])
    assert result["hits"] == [123456, 999999]
    path.unlink()
    with pytest.raises(bank.SeedAuditFailed, match="scan failed"):
        bank._scan_history({123456}, [path])


def test_audit_rejects_collision_and_broken_positive_control(
    allocation: dict, tmp_path: Path, monkeypatch
) -> None:
    for rel in ("repo/docs", "repo/research", "artifacts"):
        (tmp_path / rel).mkdir(parents=True)
    seed = allocation["rows"][0]["world_seed"]
    path = tmp_path / "repo/docs/bank.json"
    path.write_text(f'{{"seeds": [1, {seed}]}}')
    monkeypatch.setattr(bank, "_registered_history", lambda *_: ({}, {1}))
    with pytest.raises(bank.SeedAuditFailed, match="whole allocation rejected"):
        bank.audit(allocation, tmp_path / "repo", tmp_path / "artifacts")
    path.write_text('{"seeds": [2]}')
    with pytest.raises(bank.SeedAuditFailed, match="positive control"):
        bank.audit(allocation, tmp_path / "repo", tmp_path / "artifacts")


def test_missing_roots_registration_and_noninteger_requests_fail(
    tmp_path: Path, allocation: dict
) -> None:
    with pytest.raises(bank.SeedAuditFailed, match="roots unavailable"):
        bank.audit(allocation, tmp_path, tmp_path / "missing")
    with pytest.raises(bank.SeedAuditFailed, match="committed"):
        bank.require_committed_registration(tmp_path)
    for args in (("actor", "scripted", True, 0), ("actor", "scripted", 1, -1)):
        with pytest.raises(ValueError):
            bank.namespace_seeds(*args)


def test_alias_coverage_requires_explicit_safe_target_and_rejects_sealed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "bank.json"
    ordinary = tmp_path / "public.json"
    ordinary.write_text('{"seeds": [123456]}')
    alias = {"path": str(path), "target": str(ordinary), "directory": False}
    files, records = bank._cover_aliases([alias], [], [root], ())
    assert files == [ordinary]
    assert records[0]["target_sha256"] == bank.sha256(ordinary)
    ordinary.unlink()
    with pytest.raises(bank.SeedAuditFailed, match="missing"):
        bank._cover_aliases([alias], [], [root], ())
    alias["target"] = str(tmp_path / bank.SEALED_DIRECTORY / "bank.json")
    with pytest.raises(bank.SeedAuditFailed, match="forbidden"):
        bank._cover_aliases([alias], [], [root], ())


def test_external_repository_alias_is_covered_without_walking_alias(tmp_path: Path) -> None:
    root, external = tmp_path / "root", tmp_path / "external"
    root.mkdir()
    (external / "research").mkdir(parents=True)
    inventory = external / "research/BANK.json"
    inventory.write_text('{"seeds": [123456]}')
    path = root / "repo"
    path.symlink_to(external, target_is_directory=True)
    files, records = bank._cover_aliases(
        [{"path": str(path), "target": str(external), "directory": True}], [], [root], ()
    )
    assert files == [inventory]
    assert records[0]["coverage"] == "explicit ordinary source roots"


def test_removed_s1_alias_requires_bound_public_inventory(tmp_path: Path, monkeypatch) -> None:
    import json

    monkeypatch.setattr(bank, "ROOT", tmp_path)
    package = tmp_path / "research/s1_mature_20261018"
    package.mkdir(parents=True)
    (package / "WORLD_BANK.json").write_text('{"seeds": [222222]}')
    (package / "SEED_BANK.json").write_text(
        json.dumps(
            {
                "schema": "s1-seed-bank/v1",
                "seeds": [123456],
                "collision_audit": "research/s1_mature_20261018/SEED_AUDIT.json",
            }
        )
    )
    audit_path = package / "SEED_AUDIT.json"
    audit_path.write_text(
        json.dumps(
            {
                "pass": True,
                "training_seeds": [123456],
                "world_bank_file": {"sha256": bank.sha256(package / "WORLD_BANK.json")},
            }
        )
    )
    target = tmp_path / "removed/research/s1_mature_20261018/SEED_BANK.json"
    assert bank._public_s1_alias(target) == (
        package / "SEED_BANK.json",
        audit_path,
        package / "WORLD_BANK.json",
    )
    (package / "WORLD_BANK.json").write_text("{}")
    with pytest.raises(bank.SeedAuditFailed, match="binding invalid"):
        bank._public_s1_alias(target)
