"""Offline adversarial checks; never query provider or allocate a runtime."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools import small_bench_operator as op


def evidence(
    tmp_path: Path,
    sessions: str = "No active sessions found on server.\n",
    rate: float = 0,
    assignments: int = 0,
) -> tuple[dict, dict]:
    account = tmp_path / "account"
    account.mkdir()
    usage = tmp_path / "usage.txt"
    usage.write_text(
        f"Current balance: 1090\nUsage rate: {rate}\nActive assignments: {assignments}\n"
    )
    session_file = tmp_path / "sessions.txt"
    session_file.write_text(sessions)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"stop_verified": True, "endpoint": "old-a100"}))
    config = {"account_root": str(account), "previous_releases": [op.binding(receipt)]}
    return config, {"usage": op.binding(usage), "sessions": op.binding(session_file)}


def test_empty_live_account(tmp_path: Path) -> None:
    config, proof = evidence(tmp_path)
    result = op.live_checks(config, proof)
    assert result["balance"] == 1090
    assert result["rates"] == {}


@pytest.mark.parametrize(
    "sessions",
    [
        "unknown response",
        "[test] endpoint | Hardware: A100\n",
        "[test] endpoint | Hardware: L4\n",
        "[bad\n",
        "No active sessions found on server.\n[test] endpoint | Hardware: CPU\n",
    ],
)
def test_unknown_or_competing_assignment_refused(tmp_path: Path, sessions: str) -> None:
    config, proof = evidence(tmp_path, sessions, assignments=1)
    with pytest.raises(ValueError):
        op.live_checks(config, proof)


def test_pending_allocation_refused(tmp_path: Path) -> None:
    config, proof = evidence(tmp_path)
    (Path(config["account_root"]) / "pending-test.json").write_text("{}")
    with pytest.raises(ValueError, match="pending"):
        op.live_checks(config, proof)


def test_new_owned_assignment_still_requires_full_reservation() -> None:
    live = {"assignments": [{"endpoint": "new-l4", "full_cap": 4}]}
    with pytest.raises(ValueError, match="coverage"):
        op.assignment_coverage({"external_commitments_units": 0}, live)
    op.assignment_coverage({"external_commitments_units": 4}, live)


def test_admission_snapshot_has_scope_balances_holds_and_no_double_reservation() -> None:
    finances = {
        "outstanding": {
            "own": {
                "pair": {"units": 20, "wall_minutes": 120},
                "scopes": ["global"],
                "account_units": 20,
            },
            "other": {
                "pair": {"units": 5, "wall_minutes": 30},
                "scopes": ["global"],
                "account_units": 5,
            },
        },
        "scopes": {
            "global": {
                "ceiling": {"units": 100, "wall_minutes": 600},
                "spent": {"units": 10, "wall_minutes": 60},
            }
        },
        "external_commitments_units": 2,
        "unsettled_debit_holds_units": 3,
        "reservations": {"own": {}},
    }
    config = {
        "reservation_id": "own",
        "scopes": ["global"],
        "cap": {"units": 20, "wall_minutes": 120},
        "outside_tail": {"units": 0, "wall_minutes": 0},
    }
    snapshot = op.financial_snapshot(finances, config, 100)
    assert snapshot["scopes"]["global"]["available_before_reservation"]["units"] == 85
    assert snapshot["live_available_before_reservation_units"] == 90
    assert snapshot["same_reservation_excluded_once"] is True


def test_owned_calibrated_assignment(tmp_path: Path) -> None:
    config, proof = evidence(tmp_path, "[cpu] ep | Hardware: CPU\n", 0.08, 1)
    path = Path(config["account_root"]) / "active-cpu.json"
    path.write_text(
        json.dumps(
            {
                "endpoint": "ep",
                "session": "cpu",
                "unit_cap": 4,
                "observed_rate_units_per_hour": 0.08,
            }
        )
    )
    result = op.live_checks(config, proof)
    assert result["assignments"][0]["full_cap"] == 4
    assert result["rates"] == {"ep": 0.08}


@pytest.mark.parametrize("cap", [0, -1, True, float("nan"), float("inf")])
def test_active_assignment_requires_finite_positive_cap(tmp_path: Path, cap) -> None:
    config, proof = evidence(tmp_path, "[cpu] ep | Hardware: CPU\n", 0.08, 1)
    path = Path(config["account_root"]) / "active-cpu.json"
    path.write_text(
        json.dumps(
            {
                "endpoint": "ep",
                "session": "cpu",
                "unit_cap": cap,
                "observed_rate_units_per_hour": 0.08,
            }
        )
    )
    with pytest.raises(ValueError, match="finite"):
        op.live_checks(config, proof)


@pytest.mark.parametrize(
    "change",
    [
        {"session": "foreign"},
        {"observed_rate_units_per_hour": 0},
        {"observed_rate_units_per_hour": 0.09},
        {"accounting_exclusive": True},
    ],
)
def test_owned_assignment_bad_identity_rate_exclusive(tmp_path: Path, change: dict) -> None:
    config, proof = evidence(tmp_path, "[cpu] ep | Hardware: CPU\n", 0.08, 1)
    value = {
        "endpoint": "ep",
        "session": "cpu",
        "unit_cap": 4,
        "observed_rate_units_per_hour": 0.08,
        **change,
    }
    (Path(config["account_root"]) / "active-cpu.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        op.live_checks(config, proof)


def test_changed_capture_refused(tmp_path: Path) -> None:
    config, proof = evidence(tmp_path)
    Path(proof["usage"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        op.live_checks(config, proof)


def test_unverified_previous_release(tmp_path: Path) -> None:
    config, proof = evidence(tmp_path)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"stop_verified": False}))
    config["previous_releases"] = [op.binding(path)]
    with pytest.raises(ValueError, match="release"):
        op.live_checks(config, proof)


def test_publish_is_create_only_and_detects_partial_or_changed(tmp_path: Path) -> None:
    path = tmp_path / "OPERATOR_ADMISSION.json"
    op.publish_json(path, {"value": 1})
    assert op.verified(path) == {"value": 1}
    with pytest.raises(FileExistsError):
        op.publish_json(path, {"value": 2})
    path.write_text("{}")
    with pytest.raises(ValueError):
        op.verified(path)


def test_capture_executes_only_two_read_only_cli_calls(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["timeout"] == 60
        return type("Result", (), {"returncode": 0, "stdout": b"raw\n", "stderr": b""})()

    monkeypatch.setattr(op.subprocess, "run", run)
    value = op.capture({"colab": "/colab"}, tmp_path / "capture")
    assert calls == [
        ["/colab", "--auth", "oauth2", "usage"],
        ["/colab", "--auth", "oauth2", "sessions"],
    ]
    assert op.bound(value["usage"]).read_text() == "raw\n"


def test_failed_refresh_invalidates_previous_admission_pointer(tmp_path: Path, monkeypatch) -> None:
    prior = tmp_path / "immutable-prior-admission.json"
    prior.write_text("prior preserved")
    pointer = tmp_path / "OPERATOR_ADMISSION.current.json"
    pointer.write_text(json.dumps({"path": str(prior)}))
    config = {
        "register": str(tmp_path / "register"),
        "reservation_lock": str(tmp_path / "account-reservation.lock"),
        "account_root": str(tmp_path / "account"),
        "admission_pointer": str(pointer),
        "admissions_dir": str(tmp_path / "captures"),
    }
    monkeypatch.setattr(op, "configuration", lambda path: config)
    monkeypatch.setattr(op, "run_check", lambda value: None)
    monkeypatch.setattr(op.register, "read", lambda path: [{"operational_baseline": {}}])
    monkeypatch.setattr(op, "fingerprint", lambda value: {})

    def failed_capture(*args):
        raise ValueError("fresh account check failed")

    monkeypatch.setattr(op, "capture", failed_capture)
    with pytest.raises(ValueError, match="fresh account"):
        op.admit(tmp_path / "config.json")
    assert not pointer.exists()
    assert prior.read_text() == "prior preserved"


def close_event(tmp_path: Path, receipt: dict) -> dict:
    terminal = tmp_path / "receipt.json"
    terminal.write_text(json.dumps(receipt))
    exports = tmp_path / "exports.json"
    exports.write_text("{}")
    sessions = tmp_path / "sessions.txt"
    sessions.write_text("No active sessions found on server.\n")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "sessions": op.binding(sessions),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    )
    return {
        "terminal_receipt": op.binding(terminal),
        "verified_exports": op.binding(exports),
        "release_disappearance": op.binding(release),
        "charged": {"units": 10},
        "remaining_hold": {"units": 0},
    }


def test_release_alone_cannot_claim_provider_payment(tmp_path: Path) -> None:
    event = close_event(tmp_path, {"job_id": "main", "stop_verified": True, "endpoint": "ep"})
    op.settlement_evidence(event)  # default retains charged units as unpaid debit
    event["remaining_account_units"] = 0
    with pytest.raises(KeyError, match="provider_settlement"):
        op.settlement_evidence(event)


@pytest.mark.parametrize("new_was_requested", [False, True])
def test_verified_preallocation_failure_can_close_but_unknown_allocation_cannot(
    tmp_path: Path, new_was_requested: bool
) -> None:
    event = close_event(
        tmp_path,
        {
            "job_id": "main",
            "stop_verified": False,
            "endpoint": None,
            "generation": None,
            "failure_class": "UPLOAD_CAP",
            "elapsed_seconds": 0,
            "cli_calls": [{"command": "new"}] if new_was_requested else [],
        },
    )
    proof = tmp_path / "no-allocation-proof.json"
    proof.write_text("{}")
    event.update(
        allocation_never_requested=True,
        no_allocation_attestation="Verified no CLI request",
        no_allocation_proof=op.binding(proof),
        charged={"units": 0},
    )
    if new_was_requested:
        with pytest.raises(ValueError, match="pre-allocation"):
            op.settlement_evidence(event)
    else:
        op.settlement_evidence(event)


@pytest.mark.parametrize("seconds", [-1, 600, 601])
def test_stale_or_future_capture_refused(tmp_path: Path, monkeypatch, seconds: int) -> None:
    now = datetime(2026, 10, 24, tzinfo=timezone.utc)
    value = {
        "config": {"path": "/fake", "sha256": "x"},
        "register_row_hash": "r",
        "capture": {"started_utc": (now - timedelta(seconds=seconds)).isoformat()},
    }
    monkeypatch.setattr(op, "configuration", lambda p: {"register": "/register"})
    monkeypatch.setattr(op, "current_admission", lambda c: value)
    monkeypatch.setattr(op, "binding", lambda p: value["config"])
    monkeypatch.setattr(op.register, "read", lambda p: [{"row_hash": "r"}])
    with pytest.raises(ValueError, match="stale"):
        op.check_admission(tmp_path / "config", now)
