"""Receipt-only tests for Mac historical inventory admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.q0q1_20261017 import audit_q0_bank, inventory_receipt, q1_bank


def _receipt(path: Path, *, source=None, q0=None, q1=None) -> tuple[dict[str, str], dict]:
    q0_bank = audit_q0_bank.build_world_bank()
    q1_world_bank = q1_bank.build_world_bank()
    value = {
        "schema": inventory_receipt.SCHEMA,
        "source_sha256": inventory_receipt.source_manifest(),
        "audits": {
            inventory_receipt.KIND_Q0: {
                "world_bank_semantic_sha256": q0_bank["allocation_semantic_sha256"],
                "audit": {
                    "pass": True,
                    "world_bank_semantic_sha256": q0_bank["allocation_semantic_sha256"],
                    "registered_collision_seeds": [],
                },
            },
            inventory_receipt.KIND_Q1: {
                "world_bank_semantic_sha256": q1_world_bank["allocation_semantic_sha256"],
                "audit": {
                    "pass": True,
                    "world_bank_semantic_sha256": q1_world_bank["allocation_semantic_sha256"],
                    "registered_collision_seeds": [],
                },
            },
        },
    }
    if source is not None:
        value["source_sha256"] = source
    if q0 is not None:
        value["audits"][inventory_receipt.KIND_Q0]["audit"].update(q0)
    if q1 is not None:
        value["audits"][inventory_receipt.KIND_Q1]["audit"].update(q1)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {
        inventory_receipt.RECEIPT_ENV: str(path),
        inventory_receipt.RECEIPT_SHA256_ENV: hashlib.sha256(path.read_bytes()).hexdigest(),
    }, value


def test_q0_job_uses_bound_mac_receipt(tmp_path: Path) -> None:
    env, _ = _receipt(tmp_path / "receipt.json")
    env["Q0Q1_JOB_KIND"] = "q0_t23_mass"
    audit = audit_q0_bank.audit_world_bank(env=env)
    assert audit["pass"] is True


def test_missing_or_mismatched_receipt_fails_closed_for_colab_job(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing hash-bound"):
        audit_q0_bank.audit_world_bank(env={"Q0Q1_JOB_KIND": "q0_t23_mass"})
    env, _ = _receipt(tmp_path / "receipt.json")
    env["Q0Q1_JOB_KIND"] = "q0_t23_mass"
    env[inventory_receipt.RECEIPT_SHA256_ENV] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatched"):
        audit_q0_bank.audit_world_bank(env=env)


def test_receipt_rejects_stale_source_and_bank_audit_failure(tmp_path: Path) -> None:
    env, _ = _receipt(tmp_path / "stale.json", source={})
    with pytest.raises(ValueError, match="source binding is stale"):
        inventory_receipt.load_receipt_audit(
            inventory_receipt.KIND_Q1, q1_bank.build_world_bank(), env=env
        )
    env, _ = _receipt(tmp_path / "failed.json", q1={"pass": False})
    with pytest.raises(ValueError, match="failing or incomplete"):
        inventory_receipt.load_receipt_audit(
            inventory_receipt.KIND_Q1, q1_bank.build_world_bank(), env=env
        )


def test_receipt_rejects_different_recomputed_bank(tmp_path: Path) -> None:
    env, _ = _receipt(tmp_path / "receipt.json")
    bank = q1_bank.build_world_bank()
    bank["allocation_semantic_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not bind this world bank"):
        inventory_receipt.load_receipt_audit(inventory_receipt.KIND_Q1, bank, env=env)


def test_producer_does_not_hide_missing_inventory_or_collision(monkeypatch) -> None:
    monkeypatch.setattr(
        audit_q0_bank,
        "audit_world_bank",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing inventory")),
    )
    with pytest.raises(FileNotFoundError, match="missing inventory"):
        inventory_receipt.create_receipt()

    monkeypatch.setattr(
        audit_q0_bank,
        "audit_world_bank",
        lambda *args, **kwargs: {
            "pass": True,
            "world_bank_semantic_sha256": audit_q0_bank.build_world_bank()[
                "allocation_semantic_sha256"
            ],
            "registered_collision_seeds": [],
        },
    )
    monkeypatch.setattr(
        q1_bank,
        "audit_world_bank",
        lambda *args, **kwargs: {"pass": False, "registered_collision_seeds": [123]},
    )
    with pytest.raises(ValueError, match="did not pass"):
        inventory_receipt.create_receipt()


def test_cli_refuses_to_overwrite_receipt(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "receipt.json"
    monkeypatch.setattr(inventory_receipt, "create_receipt", lambda *args: {"schema": "fixture"})
    assert inventory_receipt.main(["--out", str(out)]) == 0
    with pytest.raises(FileExistsError, match="already exists"):
        inventory_receipt.main(["--out", str(out)])
    assert (tmp_path / "receipt.json.sha256").is_file()
