"""Synthetic A3 continuation proofs; no factory, labels, or outcome look."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from research.bl_labels import d4096_resume as resume
from research.bl_labels import mac_gates
from research.bl_labels.exports import _verify, empty_ledger, publish_json


def pack_fixture(tmp_path: Path) -> tuple[Path, dict]:
    pack = tmp_path / "pack"
    control = {
        "namespace": "d4096-synthetic",
        "source_commit": resume.SCIENCE_PIN,
        "allocation": [{"allocation_id": f"p{i}", "seed": i} for i in range(3)],
        "mac_gate": {
            "stage": "d4096",
            "status": "HOLD",
            "source_commit": resume.SCIENCE_PIN,
            "deadline_seconds": resume.CAP,
        },
    }
    publish_json(pack / "d4096-control.json", control)
    admitted = copy.deepcopy(control)
    admitted["mac_gate"].update(
        status="APPROVED",
        package_root=str(pack),
        state_dir=str(pack / "d4096-state-mac-01"),
        original_control_path=str(pack / "d4096-control.json"),
        original_control_sha256=_verify(pack / "d4096-control.json"),
        collector_source_commit=resume.OLD_PIN,
        collector_root=str(tmp_path / "old"),
        bindings={"speed_equivalence": {"status": "FAIL_OLD_FALLBACK"}},
    )
    publish_json(pack / "d4096-admitted.json", admitted)
    publish_json(
        pack / "d4096-verified-ledger.json",
        empty_ledger(control["namespace"], control["allocation"]),
    )
    return pack, admitted


def add_attempt(
    pack: Path,
    control: dict,
    index: int,
    *,
    workers: int,
    elapsed: float = 100.0,
    prior_elapsed: float = 0.0,
    prior_worker_seconds: float = 0.0,
    terminal: bool = True,
    start_change: dict | None = None,
    terminal_change: dict | None = None,
) -> Path:
    state = pack / "d4096-state-mac-01"
    output = pack / f"d4096-output-mac-{index:02d}"
    start_path = state / f"attempt-{index:02d}.start.json"
    start = {
        "schema": "bl-labels-f0-mac-attempt-start/v1",
        "namespace": control["namespace"],
        "source_commit": resume.SCIENCE_PIN,
        "allocation_sha256": resume.canonical_sha(control["allocation"]),
        "control_sha256": resume.canonical_sha(control),
        "deadline_seconds": resume.CAP,
        "original_wall_cap_seconds": resume.CAP,
        "state_dir": str(state),
        "output": str(output),
        "workers": workers,
        "owner_w12_ok": workers == 12,
        "w12_calibration_receipt_sha256": "a" * 64 if workers == 12 else None,
        "w12_calibration_source_commit": resume.OLD_PIN,
        "operational_source_commit": "b" * 40,
        "funding_state": "PENDING_SEALED",
        "judgment_sha256": resume.d4096_pack.JUDGMENT_SHA,
        **(start_change or {}),
    }
    publish_json(start_path, start)
    os.utime(start_path, ns=(index * 10**9, index * 10**9))
    output.mkdir(exist_ok=True)
    if not terminal:
        return output
    sealed = state / f"attempt-{index:02d}.rate-sealed.json"
    publish_json(sealed, {"events": []})
    result = {
        "schema": "bl-labels-f0-mac-driver/v1",
        "namespace": control["namespace"],
        "status": "PREEMPTED",
        "invalid_reason": None,
        "workers": workers,
        "owner_w12_ok": workers == 12,
        "w12_calibration_receipt_sha256": start["w12_calibration_receipt_sha256"],
        "w12_calibration_source_commit": resume.OLD_PIN,
        "collector_source_commit": resume.OLD_PIN,
        "operational_source_commit": start["operational_source_commit"],
        "stop_reason": "operational synthetic interruption",
        "registered_stop_reason": None,
        "life_boundary_drained": True,
        "output": str(output),
        "state_dir": str(state),
        "start_receipt": start_path.name,
        "start_sha256": _verify(start_path),
        "elapsed_seconds": elapsed,
        "prior_elapsed_seconds": prior_elapsed,
        "cumulative_elapsed_seconds": prior_elapsed + elapsed,
        "worker_seconds_denominator": workers * elapsed,
        "prior_worker_seconds_denominator": prior_worker_seconds,
        "cumulative_worker_seconds_denominator": prior_worker_seconds + workers * elapsed,
        "attempts_ledger": [f"attempt-{item:02d}.terminal.json" for item in range(1, index + 1)],
        "rate_sealed_path": str(sealed),
        **(terminal_change or {}),
    }
    publish_json(state / f"attempt-{index:02d}.terminal.json", result)
    publish_json(output / "runtime-result.json", {**result, "mac_lease": {}})
    return output


@pytest.fixture(autouse=True)
def no_real_admission(monkeypatch):
    monkeypatch.setattr(mac_gates, "require_admission", lambda *_: None)


def test_initial_empty_ledger_and_no_attempts_are_valid(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    assert resume.attempt_chain(pack, control) == ([], 0.0)
    assert resume.initial_ledger(pack, control) == pack / "d4096-verified-ledger.json"


def test_w6_to_w12_and_back_charges_each_terminal_at_its_own_width(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1, workers=6, elapsed=10)
    second = add_attempt(
        pack, control, 2, workers=12, elapsed=20, prior_elapsed=10, prior_worker_seconds=60
    )
    third = add_attempt(
        pack,
        control,
        3,
        workers=6,
        elapsed=30,
        prior_elapsed=30,
        prior_worker_seconds=300,
    )
    attempts, elapsed = resume.attempt_chain(pack, control)
    assert attempts == [first, second, third]
    assert elapsed == 60


@pytest.mark.parametrize(
    ("workers", "change", "message"),
    [
        (6, {"funding_state": "PASS"}, "start identity"),
        (6, {"judgment_sha256": "0" * 64}, "start identity"),
        (6, {"owner_w12_ok": True}, "start identity"),
        (12, {"owner_w12_ok": False}, "start identity"),
        (12, {"w12_calibration_receipt_sha256": None}, "start identity"),
        (6, {"workers": 8}, "worker width"),
    ],
)
def test_start_receipt_refusals_are_fail_closed(tmp_path, workers, change, message) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, workers=workers, start_change=change)
    with pytest.raises(ValueError, match=message):
        resume.attempt_chain(pack, control)


@pytest.mark.parametrize(
    "change",
    [
        {"worker_seconds_denominator": 61},
        {"prior_worker_seconds_denominator": 1},
        {"cumulative_worker_seconds_denominator": 601},
        {"cumulative_elapsed_seconds": 101},
        {"status": "COMPLETE"},
        {"status": "INVALID"},
        {"registered_stop_reason": "registered"},
    ],
)
def test_terminal_accounting_and_status_refusals(tmp_path, change) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, workers=6, terminal_change=change)
    with pytest.raises(ValueError):
        resume.attempt_chain(pack, control)


def test_every_terminal_counts_even_when_it_exported_no_lives(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, workers=6, elapsed=resume.CAP - 1)
    add_attempt(
        pack,
        control,
        2,
        workers=12,
        elapsed=1,
        prior_elapsed=resume.CAP - 1,
        prior_worker_seconds=6 * (resume.CAP - 1),
    )
    with pytest.raises(ValueError, match="exhausted"):
        resume.attempt_chain(pack, control)


def test_union_rejects_omitted_duplicate_and_foreign_attempt_sets(tmp_path, monkeypatch) -> None:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1, workers=6)
    second = add_attempt(pack, control, 2, workers=12, prior_elapsed=100, prior_worker_seconds=600)
    monkeypatch.setattr(resume, "original_control", lambda _: control)
    with pytest.raises(ValueError, match="attempts"):
        resume.publish_ledger(pack, [first], pack / "one.json")
    with pytest.raises(ValueError, match="attempts"):
        resume.publish_ledger(pack, [first, second, second], pack / "duplicate.json")


def test_final_chain_allows_only_the_last_complete_terminal(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, workers=6)
    add_attempt(
        pack,
        control,
        2,
        workers=12,
        prior_elapsed=100,
        prior_worker_seconds=600,
        terminal_change={"status": "COMPLETE", "pending_parents": []},
    )
    assert resume.attempt_chain(pack, control, final=True)[1] == 200
    with pytest.raises(ValueError, match="non-preempted"):
        resume.attempt_chain(pack, control)


def test_width_change_refuses_forced_mid_life_drain(tmp_path):
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, workers=12, terminal_change={"life_boundary_drained": False})
    resume.require_boundary_change(pack, 12)
    with pytest.raises(ValueError, match="life-boundary drain"):
        resume.require_boundary_change(pack, 6)
    add_attempt(pack, control, 2, workers=6, prior_elapsed=100, prior_worker_seconds=1200)
    with pytest.raises(ValueError, match="life-boundary drain"):
        resume.attempt_chain(pack, control)


def test_mac_complete_delegated_terminal_requires_verified_old_tail_record(tmp_path):
    from research.bl_labels import l4_delegation

    pack, control = pack_fixture(tmp_path)
    directory = pack / "delegation"
    publish_json(
        directory / "delegation-01.json",
        {
            "schema": l4_delegation.DELEGATION_SCHEMA,
            "delegation_id": "01",
            "namespace": control["namespace"],
            "allocation_sha256": l4_delegation.allocation_sha256(control["allocation"]),
            "collector_source_commit": resume.OLD_PIN,
            "parents": ["p2"],
        },
    )
    add_attempt(
        pack,
        control,
        1,
        workers=6,
        terminal_change={
            "status": "COMPLETE_EXCEPT_DELEGATED",
            "pending_parents": [],
            "delegation_dir": str(directory),
            "delegated_parents": ["p2"],
        },
    )
    assert resume.attempt_chain(pack, control)[1] == 100
    assert resume.attempt_chain(pack, control, final=True)[1] == 100


def _patch_launch(monkeypatch, pack: Path, control: dict, *, effective: int = 6) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(resume, "PACK", pack)
    monkeypatch.setattr(resume.d4096_sealed, "validate_readiness", lambda *_: {})
    monkeypatch.setattr(resume.d4096_sealed, "validate_projection", lambda *_: {})
    monkeypatch.setattr(resume.d4096_sealed, "require_not_discarded", lambda *_: None)
    monkeypatch.setattr(resume.d1024_w12, "_prepare_science_root", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        resume.d1024_w12,
        "_validate_inputs",
        lambda *_args, **_kw: (control, 12 if effective == 12 else 6, [], True),
    )
    monkeypatch.setattr(resume, "original_control", lambda _: control)
    monkeypatch.setattr(resume.mac_driver, "_prepare_jobs", lambda *_: ("ns", [], [object()]))
    monkeypatch.setattr(
        __import__("research.bl_labels.w12_runtime", fromlist=["resolve_resume_workers"]),
        "resolve_resume_workers",
        lambda workers, _state: (effective, ["synthetic"] if workers != effective else []),
    )
    monkeypatch.setattr(resume.subprocess, "run", lambda command, **_kw: calls.append(command))
    return calls


def test_check_without_admission_prints_plan_and_never_writes_receipt(
    tmp_path, monkeypatch, capsys
) -> None:
    pack, control = pack_fixture(tmp_path)
    (pack / "d4096-admitted.json").unlink()
    calls = _patch_launch(monkeypatch, pack, control)
    resume.launch(
        pack,
        tmp_path / "eq",
        tmp_path / "cal",
        tmp_path / "old",
        None,
        None,
        tmp_path / "b8",
        tmp_path / "frp3",
        sealed_readiness=tmp_path / "readiness",
        projection_receipt=tmp_path / "projection",
        check=True,
    )
    event = json.loads(capsys.readouterr().out)
    assert event["workers"] == 6 and event["prior_attempts"] == 0
    assert not (pack / "d4096-admitted.json").exists()
    assert calls == []


def test_existing_admission_w12_command_carries_old_width_authorization(
    tmp_path, monkeypatch
) -> None:
    pack, control = pack_fixture(tmp_path)
    calls = _patch_launch(monkeypatch, pack, control, effective=12)
    calibration = tmp_path / "cal"
    resume.launch(
        pack,
        tmp_path / "eq",
        calibration,
        tmp_path / "old",
        None,
        None,
        tmp_path / "b8",
        tmp_path / "frp3",
        workers=12,
        owner_w12_ok=True,
        w12_calibration_receipt=calibration,
        sealed_readiness=tmp_path / "readiness",
        projection_receipt=tmp_path / "projection",
        check=True,
    )
    command = calls[-1]
    assert command[command.index("--workers") + 1] == "12"
    assert command[command.index("--w12-calibration-receipt") + 1] == str(calibration)
    assert "--owner-w12-ok" in command


def test_launch_refuses_nonempty_next_output_before_driver(tmp_path, monkeypatch) -> None:
    pack, control = pack_fixture(tmp_path)
    calls = _patch_launch(monkeypatch, pack, control)
    output = pack / "d4096-output-mac-01"
    output.mkdir()
    (output / "foreign").write_text("x")
    with pytest.raises(ValueError, match="output directory"):
        resume.launch(
            pack,
            tmp_path / "eq",
            tmp_path / "cal",
            tmp_path / "old",
            None,
            None,
            tmp_path / "b8",
            tmp_path / "frp3",
            sealed_readiness=tmp_path / "readiness",
            projection_receipt=tmp_path / "projection",
            check=True,
        )
    assert calls == []
