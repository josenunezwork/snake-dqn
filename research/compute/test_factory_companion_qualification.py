"""Unqualified concurrency cannot allocate, publish or acquire a Mac lease."""

from __future__ import annotations

import hashlib
import json

import pytest

from research.compute import factory_companion
from research.compute import factory_companion_qualification as qualification


def test_missing_evidence_is_explicit_hold(tmp_path) -> None:
    report = qualification.qualification_status(tmp_path / "absent.json")
    assert report["verdict"] == "HOLD"
    assert any("native_descendant_backend_unavailable" in r for r in report["reasons"])
    assert "missing_immutable_qualification_evidence" in report["reasons"]
    assert {f"acceptance_{i}_not_qualified" for i in range(1, 8)} <= set(report["reasons"])


def test_manual_pass_flags_and_environment_cannot_activate(tmp_path, monkeypatch) -> None:
    path = tmp_path / "qualification.json"
    evidence = {
        "verdict": "PASS",
        "concurrency_policy_sha256": qualification.POLICY_SHA256,
        "operational_source_closure_sha256": qualification.source_closure_sha256(),
        "acceptance": {str(i): {"verdict": "PASS"} for i in range(1, 8)},
    }
    payload = json.dumps(evidence).encode()
    path.write_bytes(payload)
    path.with_name(path.name + ".sha256").write_text(hashlib.sha256(payload).hexdigest())
    path.with_name(path.name + ".complete").write_text("complete\n")
    monkeypatch.setenv("FACTORY_COMPANION_QUALIFIED", "1")
    monkeypatch.setenv("SKIP_COMPANION_QUALIFICATION", "1")
    report = qualification.qualification_status(path)
    assert report["verdict"] == "HOLD"
    assert len(report["reasons"]) == 1
    with pytest.raises(qualification.QualificationError, match="activation HOLD"):
        qualification.require_activation()


@pytest.mark.parametrize("mode", ["monitor", "client", "tool"])
def test_cli_refuses_before_supervisor_allocation(mode: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        factory_companion,
        "FactoryCompanionSupervisor",
        lambda *_args, **_kwargs: pytest.fail("supervisor allocated before qualification"),
    )
    args = [
        "--root",
        str(tmp_path),
        "--policy-sha",
        qualification.POLICY_SHA256,
        "--lease-identity",
        "pending",
        mode,
    ]
    if mode == "tool":
        args += ["--class", "light", "--", "cat", "file"]
    with pytest.raises(qualification.QualificationError, match="activation HOLD"):
        factory_companion.main(args)
    assert list(tmp_path.iterdir()) == []


def test_factory_startup_fails_before_monitor_or_registry_mutation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        factory_companion,
        "host_check_snapshot",
        lambda **_kwargs: pytest.fail("observation before qualification"),
    )
    with pytest.raises(qualification.QualificationError, match="activation HOLD"):
        factory_companion.factory_startup_guard(
            "attempt", policy_sha=qualification.POLICY_SHA256, root=tmp_path
        )
    assert list(tmp_path.iterdir()) == []


def test_companion_receipt_reference_is_hash_of_actual_immutable_file(tmp_path) -> None:
    from research.compute.test_factory_companion import Clock, supervisor

    subject = supervisor(tmp_path, Clock(), {})
    subject._write(subject._empty())
    value = subject.finalize_receipt(
        attempt_identity="excluded", policy_sha=qualification.POLICY_SHA256
    )
    from pathlib import Path

    path = Path(value["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"]
    assert path.with_name(path.name + ".complete").read_text() == "complete\n"


@pytest.mark.parametrize(
    "operation", ["monitor", "register", "helper", "launch", "sample", "bind", "control"]
)
def test_direct_live_root_methods_cannot_bypass_activation(operation: str, monkeypatch) -> None:
    subject = factory_companion.FactoryCompanionSupervisor()
    monkeypatch.setattr(
        subject, "process_table", lambda: pytest.fail("native probe before activation")
    )
    monkeypatch.setattr(subject, "_read", lambda: pytest.fail("registry read before activation"))
    with pytest.raises(qualification.QualificationError, match="activation HOLD"):
        if operation == "monitor":
            subject.claim_monitor()
        elif operation == "register":
            subject.register_client(
                policy_sha="p", lease_identity="a", worktree="w", executable="x", argv=["x"]
            )
        elif operation == "helper":
            with subject.control_helper(policy_sha="p", lease_identity="a", argv=["x"]):
                pytest.fail("helper activated")
        elif operation == "launch":
            subject.launch("light", policy_sha="p", lease_identity="a", worktree="w", argv=["cat"])
        elif operation == "sample":
            subject.sample(monitor=True)
        elif operation == "bind":
            subject.bind_factory_attempt(identity="a", policy_sha="p")
        else:
            subject.update_factory_state(identity="a", protected=True, streams={})
