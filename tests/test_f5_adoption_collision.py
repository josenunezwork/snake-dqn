"""Outcome-free tests for F5 reconciliation, whole rounds and custodian gates."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from research.bl_residual import f5_r10a_annex as original
from research.bl_residual.f5_adoption_collision_20261026 import reconcile as closure


def test_frozen_whole_allocation_identity() -> None:
    bank = original.allocate(0)
    assert len(bank) == 684
    assert sum(r["timing_only"] for r in bank) == 12
    assert (
        closure.digest(closure.canonical(bank))
        == "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
    )


def test_reject_whole_round_before_accepting_next() -> None:
    bank = original.allocate(0)
    history = [{"parent_id": "historical-reserved", "seed": bank[0]["seed"], "aliases": []}]
    result = closure.audit(history, {bank[0]["seed"]})
    assert result["candidate_round"] == 1
    assert len(result["candidate_allocation"]) == 684
    assert result["candidate_allocation"] == original.allocate(1)
    assert result["accepted_round"] is None
    assert len(result["rejected_rounds"]) == 1
    assert result["rejected_rounds"][0]["round"] == 0


def test_pruned_auditor_preserves_all_original_reasons() -> None:
    bank = original.allocate(0)
    history = [
        {"parent_id": "one", "seed": bank[0]["seed"], "aliases": []},
        {"parent_id": "one", "seed": bank[0]["seed"], "aliases": ["alias"]},
        {"parent_id": "two", "seed": 2**63, "aliases": [bank[1]["parent_id"]]},
        {"parent_id": "irrelevant", "seed": -20, "aliases": []},
    ]
    expected = original.collision_audit(bank, closure.transitive_aliases(history))
    actual = closure.audit(history, set())["rejected_rounds"][0]["reasons"]
    assert actual == expected
    assert len(actual) == 3


def test_transitive_namespace_alias_collision() -> None:
    candidate = original.allocate(0)[0]
    history = [
        {"parent_id": "root", "seed": -1, "aliases": ["bridge"]},
        {"parent_id": "bridge", "seed": -2, "ancestry": [candidate["parent_id"]]},
    ]
    result = closure.audit(history, set())
    assert result["candidate_round"] == 1
    assert len(result["rejected_rounds"][0]["reasons"]) == 2


def test_no_unsupported_signed_or_uint32_normalization() -> None:
    seed = original.allocate(0)[0]["seed"]
    result = closure.audit([], {seed + 2**32, seed - 2**32, -seed})
    assert result["candidate_round"] == 0


def test_internal_collision_rejects_entire_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    allocate = original.allocate

    def with_internal_collision(round_index: int) -> list[dict]:
        rows = allocate(round_index)
        if round_index == 0:
            rows[-1]["seed"] = rows[0]["seed"]
        return rows

    monkeypatch.setattr(original, "allocate", with_internal_collision)
    result = closure.audit([], set())
    assert result["candidate_round"] == 1
    assert result["rejected_rounds"][0]["reasons"][0]["kind"] == "internal_collision"


@pytest.mark.parametrize(
    "name",
    [
        "D4096/metadata.json",
        "t34-round0-study-01/metadata.json",
        "runtime-output/metadata.json",
        "lives/metadata.json",
    ],
)
def test_refuse_protected_and_payload_paths_before_access(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        closure.read({"path": str(tmp_path / name), "sha256": "0" * 64})


def test_refuse_symlink_and_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        closure.read({"path": str(link), "sha256": closure.digest(b"{}")})
    with pytest.raises(ValueError, match="hash"):
        closure.read({"path": str(source), "sha256": "0" * 64})


def test_ancestor_symlink_is_refused_before_leaf_metadata_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    leaf = alias / "never-inspect.json"
    checked = []
    original_check = Path.is_symlink

    def record_check(path: Path) -> bool:
        checked.append(path)
        return original_check(path)

    monkeypatch.setattr(Path, "is_symlink", record_check)
    with pytest.raises(ValueError, match="symlink"):
        closure.safe_path(leaf)
    assert alias in checked
    assert leaf not in checked


def test_preparation_cli_cannot_accept_a_self_issued_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reconcile",
            "--plan",
            str(tmp_path / "plan.json"),
            "--out",
            str(tmp_path / "out"),
            "--certificate",
            str(tmp_path / "self-issued.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        closure.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


def test_template_does_not_supply_custodian_authority() -> None:
    evaluation = closure.audit([], set())
    manifest = {"complete": False, "unresolved": ["protected coverage"]}
    cert = closure.certificate_template(manifest, evaluation)
    assert all(x is None for x in cert["assertions"].values())
    assert cert["omissions"] is None
    with pytest.raises(ValueError, match="unattested"):
        closure.validate_certificate_structure(cert, manifest, evaluation)


def _certificate(tmp_path: Path) -> tuple[dict, dict, dict]:
    evaluation = closure.audit([], set())
    manifest = {"complete": False, "unresolved": ["custodian-dependent coverage"]}
    cert = closure.certificate_template(manifest, evaluation)
    cert.update(
        status="ATTESTED",
        issuer="synthetic test custodian",
        existing_authority="synthetic test only",
        attested_at_utc="2026-10-26T00:00:00Z",
        register_snapshot_cutoff_utc="2026-10-25T23:00:00Z",
        omissions=[],
        collision_result_all_684="NO_COLLISION",
        assertions={key: True for key in closure.ASSERTIONS},
        protected_register_catalog_bindings=[
            {"bank": bank, "register_sha256": "1" * 64, "catalog_sha256": "2" * 64}
            for bank in ("D4096", "t34-round0-study-01")
        ],
    )
    approval = {
        "issuer": cert["issuer"],
        "approved": True,
        "certificate_content_sha256": closure.digest(
            closure.canonical({k: v for k, v in cert.items() if k != "approval_record"})
        ),
    }
    path = tmp_path / "synthetic-approval.json"
    raw = closure.canonical(approval)
    path.write_bytes(raw)
    cert["approval_record"] = {"path": str(path), "sha256": closure.digest(raw)}
    return cert, manifest, evaluation


def test_bound_certificate_structure_is_not_issuer_authentication(tmp_path: Path) -> None:
    cert, manifest, evaluation = _certificate(tmp_path)
    closure.validate_certificate_structure(cert, manifest, evaluation)


@pytest.mark.parametrize(
    "change",
    [
        "omission",
        "stale_allocation",
        "missing_protected",
        "missing_assertion",
        "unknown_collision",
        "prior_rejections",
        "foreign_approval",
    ],
)
def test_certificate_refuses_unknown_stale_or_incomplete_coverage(
    tmp_path: Path, change: str
) -> None:
    cert, manifest, evaluation = _certificate(tmp_path)
    changed = copy.deepcopy(cert)
    if change == "omission":
        changed["omissions"] = ["unknown aliases"]
    elif change == "stale_allocation":
        changed["candidate_allocation_sha256"] = "0" * 64
    elif change == "missing_protected":
        changed["protected_register_catalog_bindings"].pop()
    elif change == "missing_assertion":
        changed["assertions"].pop(closure.ASSERTIONS[0])
    elif change == "unknown_collision":
        changed["collision_result_all_684"] = "UNKNOWN"
    elif change == "prior_rejections":
        changed["prior_round_rejections"] = [{"round": 0, "reasons": []}]
    else:
        changed["issuer"] = "another issuer"
    with pytest.raises(ValueError):
        closure.validate_certificate_structure(changed, manifest, evaluation)


def test_create_only_publication_and_inventory(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    closure.publish(output, {"metadata.json": {"no_outcomes": True}})
    inventory = json.loads((output / "COMPLETED.json").read_bytes())
    for name, expected in inventory["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match="create-only"):
        closure.publish(output, {"metadata.json": {"no_outcomes": False}})


def test_missing_physical_root_hash_refuses_recovery() -> None:
    value = {
        "schema": "bl-p1-collection-allocation-audit/v1",
        "burned_roots": [],
        "burned_root_count": 0,
        "burned_roots_sha256": closure.digest(b""),
        "allocated_roots": [42],
        "allocated_root_count": 1,
        "allocated_roots_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="allocated roots identity"):
        closure.collector_roots(value)
