"""Offline new-bank namespace, packet, and outcome-blind retry admission checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.watch_replay import prepare_l1_os as prepare
from research.watch_replay import seed_audit
from research.watch_replay.run import allocation, identity
from tools.ops.package_spec import digest, empty_ledger, publish_json, unit_id


def contract() -> dict:
    return {
        **allocation(seed_audit.L1_OS_BANK_NAMESPACES[0], 44, "exact-stop-l1-os"),
        "commit": "a" * 40,
    }


def receipt(
    tmp_path: Path, value: dict, *, attempt: int = 1, exports: list | None = None, **updates
) -> Path:
    root = tmp_path / f"attempt-{attempt}"
    root.mkdir()
    manifest = {
        "commit": value["commit"],
        "script": prepare.SCRIPT,
        "watch_replay": {"allocation": identity(value), "attempt": attempt},
    }
    publish_json(root / "manifest.json", manifest)
    row = {
        "manifest_sha256": digest(root / "manifest.json"),
        "stop_verified": True,
        "status": "FAIL",
        "error": "TRANSFER timeout",
        "exports": exports or [],
        **updates,
    }
    publish_json(root / "receipt.json", row)
    return root / "receipt.json"


def test_pure_seed_audit_reads_only_committed_seed_files(monkeypatch):
    real_read = Path.read_text
    read = []

    def tracked(path, *args, **kwargs):
        assert str(path.relative_to(seed_audit.ROOT)) in seed_audit.COMMITTED_BANK_FILES
        read.append(path)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked)
    monkeypatch.setattr(seed_audit.subprocess, "run", lambda *a, **k: pytest.fail("no scans"))
    report = seed_audit.l1_os_bank_disjointness()
    assert report["status"] == "CLEAR"
    assert report["total_seeds"] == 132 + 3 + 8
    assert len(read) == len(seed_audit.COMMITTED_BANK_FILES)


def test_pure_seed_audit_detects_prior_collision(monkeypatch):
    monkeypatch.setattr(
        seed_audit, "L1_OS_NAMESPACES", ((seed_audit.BANK_V2_CONFIRM_NAMESPACE, 1),)
    )
    with pytest.raises(ValueError, match="namespace reuse"):
        seed_audit.l1_os_bank_disjointness()


def test_old_bank_v2_registration_unchanged():
    report = seed_audit.bank_v2_disjointness()
    assert set(report["namespaces"]) == {ns for ns, _ in seed_audit.BANK_V2_NAMESPACES}


def test_forecast_export_and_envelope_bounds():
    plan = prepare.forecast()
    assert plan["high_minutes"] == 189
    assert plan["effective_timeout_minutes"] == 378
    assert plan["effective_unit_cap"] == 2 * plan["base_unit_cap"]
    assert 44 * prepare.MAX_SESSION_BYTES + prepare.MAX_CHECKPOINT_BYTES < prepare.MAX_EXPORT_BYTES


@pytest.mark.parametrize(
    "receipts,ledger",
    [([Path("one")], None), ([], Path("ledger")), ([Path("one")] * 3, Path("ledger"))],
)
def test_retry_limits_fail_before_reading(receipts, ledger):
    with pytest.raises(ValueError, match="at most two"):
        prepare._lineage(contract(), ledger, receipts)


@pytest.mark.parametrize(
    "updates", [{"stop_verified": False}, {"status": "PASS"}, {"error": "poor performance"}]
)
def test_retry_requires_verified_mechanical_stop(tmp_path, updates):
    value = contract()
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(value["namespace"], "sessions", identity(value)))
    prior = receipt(tmp_path, value, **updates)
    with pytest.raises(ValueError, match="stopped mechanically"):
        prepare._lineage(value, ledger, [prior])


def test_retry_ledger_exact_union(tmp_path):
    value = contract()
    row = {
        "index": 0,
        "unit_id": unit_id(value["namespace"], "sessions", 0),
        "size": 42,
        "sha256": "b" * 64,
    }
    ledger = empty_ledger(value["namespace"], "sessions", identity(value))
    ledger["units"] = [row]
    path = tmp_path / "ledger.json"
    publish_json(path, ledger)
    prior = receipt(
        tmp_path,
        value,
        exports=[
            {"path": "sessions/000000.watch.gz", "sha256": row["sha256"], "size": row["size"]}
        ],
    )
    assert prepare._lineage(value, path, [prior]) == {0}
    empty = tmp_path / "empty.json"
    publish_json(empty, empty_ledger(value["namespace"], "sessions", identity(value)))
    with pytest.raises(ValueError, match="exactly all"):
        prepare._lineage(value, empty, [prior])


@pytest.mark.parametrize(
    "reason", ["session budget", "signal SIGTERM", "signal SIGTERM; stop grace exceeded"]
)
def test_retry_accepts_supervisor_child_nonzero_interrupt(tmp_path, reason):
    value = contract()
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, empty_ledger(value["namespace"], "sessions", identity(value)))
    prior = receipt(
        tmp_path,
        value,
        failure_class="CHILD_NONZERO_EXIT",
        error=f"remote child exited 3: INTERRUPT: {reason}",
    )
    assert prepare._lineage(value, ledger, [prior]) == set()


def test_checkpoint_source_rejects_foreign_contract(tmp_path):
    publish_json(tmp_path / "contract.json", {**contract(), "commit": "b" * 40})
    with pytest.raises(ValueError, match="foreign checkpoint source"):
        prepare._checkpoint_roots([tmp_path], contract())


def test_packet_identical_inputs_and_unsealed_main(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "_check_pin", lambda _: None)
    monkeypatch.setattr(prepare, "CHECKPOINTS", {})
    monkeypatch.setattr(
        prepare,
        "release_bundle",
        lambda path, pins: path.write_bytes(b"synthetic release evidence"),
    )
    monkeypatch.setattr(
        prepare, "check_uploads", lambda manifest: {"status": "synthetic-source-pin-check"}
    )
    out = tmp_path / "packet"
    prepare.prepare(out, "a" * 40, packet=2)
    probe = json.loads((out / "canary.json").read_text())
    main = json.loads((out / "main-unsealed.json").read_text())
    assert probe["inputs"] == main["inputs"]
    assert probe["script"] == main["script"] == prepare.SCRIPT
    assert main["env"]["WATCH_REPLAY_WORKERS"] == "44"
    assert probe["env"]["WATCH_REPLAY_CHECKPOINT_SECONDS"] == "45"
    assert main["env"]["WATCH_REPLAY_CHECKPOINT_SECONDS"] == "1200"
    assert main["resume"]["max_outcome_blind_retries"] == 2
    assert main["requirements"] == {
        "gpu": "G4",
        "allowed_gpus": ["G4"],
        "min_vcpu": 48,
        "est_minutes": 189,
    }
    assert "sessions/000000.watch.gz" in probe["required_exports"]
    assert not (out / "main.json").exists()
    assert (
        json.loads((out / "main-contract.json").read_text())["namespace"]
        == seed_audit.L1_OS_BANK_NAMESPACES[1]
    )
    assert (
        json.loads((out / "canary-contract.json").read_text())["namespace"]
        == seed_audit.L1_OS_CANARY_NAMESPACES[1]
    )


def test_main_requires_successful_canary_check(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "_check_pin", lambda _: None)
    monkeypatch.setattr(prepare, "CHECKPOINTS", {})
    monkeypatch.setattr(
        prepare, "release_bundle", lambda path, pins: path.write_bytes(b"synthetic")
    )
    monkeypatch.setattr(prepare, "check_uploads", lambda manifest: {})

    def refuse(*args):
        raise ValueError("unqualified receipt")

    monkeypatch.setattr(prepare.canary, "check", refuse)
    prior = tmp_path / "receipt.json"
    prior.write_text("{}")
    with pytest.raises(ValueError, match="unqualified receipt"):
        prepare.prepare(tmp_path / "packet", "a" * 40, canary_receipt=prior)
    assert not (tmp_path / "packet/main.json").exists()
