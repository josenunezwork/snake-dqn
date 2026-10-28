"""Offline checks for TRAIN-01's Mac-side resume-packet builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_resume import (
    ResumeRefused,
    build_ledger,
    completion_packet,
    verify_mechanical_stop,
)
from research.sandbox_a_resident.training_units import (
    TRAIN01_07_SOURCE_COMMIT,
    TRAIN01_11_SOURCE_COMMIT,
    allocation,
    unit_id,
)
from research.sandbox_a_resident.units import publish


def control() -> dict:
    return {"study": "sandbox-a-train01", "source_commit": "a" * 40, "outer_iterations": 4}


def snapshot(
    root: Path,
    iteration: int,
    *,
    phase: str | None = None,
    prepare_phase: str | None = None,
    header_control: dict | None = None,
) -> None:
    header_control = control() if header_control is None else header_control
    state = publish(
        root / f"iteration-{iteration:04d}.pt", lambda path: path.write_bytes(b"opaque-state")
    )
    metadata = {
        "allocation": allocation(header_control),
        "source_commit": header_control["source_commit"],
    }
    if phase is not None:
        metadata["phase"] = phase
    if prepare_phase is not None:
        metadata["prepare_phase"] = prepare_phase
    info = {
        "schema": "sandbox-a-train-snapshot/v1",
        "id": unit_id(iteration),
        "iteration": iteration,
        "snapshot": {"id": unit_id(iteration), **state},
        "metadata": metadata,
    }
    publish(
        root / f"iteration-{iteration:04d}.json", lambda path: path.write_text(json.dumps(info))
    )
    publish(root / f"iteration-{iteration:04d}-actor.pth", lambda path: path.write_bytes(b"actor"))


def manifest() -> dict:
    return {"job_id": "train-01", "commit": "a" * 40, "output_dir": "/tmp/train-01", "inputs": []}


def receipt() -> dict:
    return {
        "job_id": "train-01",
        "commit": "a" * 40,
        "terminal": None,
        "generation": "g-1",
        "manifest_sha256": "b" * 64,
        "status": "INCOMPLETE",
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "stop_verified": True,
        "cleanup_error": None,
    }


def test_builds_relative_hash_bound_triples_without_opening_torch_payload(tmp_path: Path) -> None:
    snapshot(tmp_path, 0, phase="pre-actor", prepare_phase="ready")
    snapshot(tmp_path, 4)
    ledger, records, latest = build_ledger(tmp_path, control())
    assert latest == 4
    assert ledger["allocation"] == allocation(control())
    assert [row["iteration"] for row in ledger["units"]] == [0, 4]
    assert all(item["dest"].startswith("resume/") for item in records)
    assert ledger["units"][-1]["exports"][0]["path"] == "resume/iteration-0004.pt"
    assert {
        item["dest"] for item in records if not item["dest"].endswith((".sha256", ".complete"))
    } == {
        "resume/iteration-0000.pt",
        "resume/iteration-0000.json",
        "resume/iteration-0000-actor.pth",
        "resume/iteration-0004.pt",
        "resume/iteration-0004.json",
        "resume/iteration-0004-actor.pth",
    }
    assert len(records) == 18  # each payload plus its immutable SHA and completion marker


def test_refuses_partial_or_unsafe_prepare_snapshot(tmp_path: Path) -> None:
    snapshot(tmp_path, 0, phase="calibrated-boundary")
    with pytest.raises(ResumeRefused, match="safe continuation"):
        build_ledger(tmp_path, control())
    clean = tmp_path / "partial"
    snapshot(clean, 4)
    for suffix in (".sha256", ".complete"):
        (clean / f"iteration-0004-actor.pth{suffix}").unlink()
    # An incomplete triple cannot become evidence merely because its metadata remains.
    with pytest.raises(ResumeRefused, match="no complete"):
        build_ledger(clean, control())


def test_completion_packet_requires_released_mechanical_failure_and_fresh_identity(
    tmp_path: Path,
) -> None:
    snapshot(tmp_path, 2)
    packet = completion_packet(
        manifest(),
        receipt(),
        tmp_path,
        control(),
        completion_index=1,
        job_id="train-01-completion-1",
        output_dir=Path("/tmp/train-01-completion-1"),
        original_manifest_sha256="b" * 64,
    )
    assert packet.latest_iteration == 2
    assert packet.manifest["completion_attempt"]["manual"] is True
    assert packet.manifest["inputs"] == packet.input_records
    failed = receipt()
    failed["failure_class"] = "CHILD_NONZERO_EXIT"
    with pytest.raises(ResumeRefused, match="lacks a mechanical timeout/lease terminal reason"):
        completion_packet(
            manifest(),
            failed,
            tmp_path,
            control(),
            completion_index=1,
            job_id="x",
            output_dir=Path("/tmp/x"),
            original_manifest_sha256="b" * 64,
        )
    with pytest.raises(ResumeRefused, match="only two"):
        completion_packet(
            manifest(),
            receipt(),
            tmp_path,
            control(),
            completion_index=3,
            job_id="x",
            output_dir=Path("/tmp/x"),
            original_manifest_sha256="b" * 64,
        )


def test_completion_packet_directly_filters_prior_runtime_inputs_and_uses_latest_unit(
    tmp_path: Path,
) -> None:
    snapshot(tmp_path, 1, phase="calibrated-boundary")
    snapshot(tmp_path, 2)
    previous = manifest()
    previous["inputs"] = [
        {"path": "/tmp/plain", "dest": "b8-s3.pth", "sha256": "c" * 64},
        {"path": "/tmp/old-ledger", "dest": "verified-ledger.json", "sha256": "d" * 64},
        {"path": "/tmp/old-state", "dest": "resume/iteration-0001.pt", "sha256": "e" * 64},
    ]
    packet = completion_packet(
        previous,
        receipt(),
        tmp_path,
        control(),
        completion_index=2,
        job_id="train-01-completion-2",
        output_dir=Path("/tmp/train-01-completion-2"),
        original_manifest_sha256="b" * 64,
    )
    assert [row["iteration"] for row in packet.ledger["units"]] == [2]
    assert all(row["dest"] != "verified-ledger.json" for row in packet.manifest["inputs"])
    assert all(
        not row["dest"].startswith("resume/iteration-0001") for row in packet.manifest["inputs"]
    )
    assert {row["dest"] for row in packet.input_records} == {
        f"resume/iteration-0002{suffix}"
        for suffix in (
            ".pt",
            ".json",
            "-actor.pth",
            ".pt.sha256",
            ".json.sha256",
            "-actor.pth.sha256",
            ".pt.complete",
            ".json.complete",
            "-actor.pth.complete",
        )
    }


def test_build_ledger_ignores_earlier_nonresumable_phase_but_keeps_integrity_refusals(
    tmp_path: Path,
) -> None:
    snapshot(tmp_path, 1, phase="calibrated-boundary")
    snapshot(tmp_path, 2)
    ledger, _records, latest = build_ledger(tmp_path, control())
    assert latest == 2
    assert [row["iteration"] for row in ledger["units"]] == [2]

    broken = tmp_path / "iteration-0001.json.sha256"
    broken.write_text("0" * 64)
    with pytest.raises(ResumeRefused, match="corrupt export"):
        build_ledger(tmp_path, control())


def test_nonresumable_snapshot_still_verifies_state_payload_before_skip(tmp_path: Path) -> None:
    snapshot(tmp_path, 1, phase="calibrated-boundary")
    # A nonresumable phase may be skipped only after all three immutable
    # payloads have passed their sidecar checks.
    (tmp_path / "iteration-0001.pt.sha256").write_text("0" * 64)
    with pytest.raises(ResumeRefused, match=r"corrupt export: iteration-0001\.pt"):
        build_ledger(tmp_path, control())


def test_build_ledger_refuses_nonresumable_foreign_or_later_snapshot(tmp_path: Path) -> None:
    snapshot(tmp_path, 2)
    snapshot(
        tmp_path,
        1,
        phase="calibrated-boundary",
        header_control={**control(), "source_commit": "c" * 40},
    )
    with pytest.raises(ResumeRefused, match="header differs"):
        build_ledger(tmp_path, control())

    other = tmp_path / "later"
    snapshot(other, 2)
    snapshot(other, 3, phase="calibrated-boundary")
    with pytest.raises(ResumeRefused, match="safe continuation"):
        build_ledger(other, control())


def test_refuses_completed_snapshot_outside_frozen_boundary_cadence(tmp_path: Path) -> None:
    frozen = {**control(), "snapshot_interval": 2}
    snapshot(tmp_path, 3, header_control=frozen)
    with pytest.raises(ResumeRefused, match="frozen boundary cadence"):
        build_ledger(tmp_path, frozen)


def test_skips_partial_trio_and_refuses_complete_endpoint_or_header_mismatch(
    tmp_path: Path,
) -> None:
    snapshot(tmp_path, 2)
    (tmp_path / "iteration-0002-actor.pth.complete").unlink()
    with pytest.raises(ResumeRefused, match="no complete"):
        build_ledger(tmp_path, control())
    snapshot(tmp_path / "endpoint", 4)
    with pytest.raises(ResumeRefused, match="frozen endpoint"):
        completion_packet(
            manifest(),
            receipt(),
            tmp_path / "endpoint",
            control(),
            completion_index=1,
            job_id="train-01-completion-1",
            output_dir=Path("/tmp/train-01-completion-1"),
            original_manifest_sha256="b" * 64,
        )


def test_refuses_foreign_header_and_nonmatching_terminal_generation(tmp_path: Path) -> None:
    snapshot(tmp_path, 2, header_control={**control(), "source_commit": "c" * 40})
    with pytest.raises(ResumeRefused, match="header differs"):
        build_ledger(tmp_path, control())
    other = tmp_path / "receipt"
    snapshot(other, 2)
    observed = receipt()
    observed["terminal"] = {"job_id": "train-01", "generation": "different"}
    with pytest.raises(ResumeRefused, match="terminal identity"):
        completion_packet(
            manifest(),
            observed,
            other,
            control(),
            completion_index=1,
            job_id="train-01-completion-1",
            output_dir=Path("/tmp/train-01-completion-1"),
            original_manifest_sha256="b" * 64,
        )


def test_build_ledger_binds_train01_07_snapshot_to_exact_source_only_successor(
    tmp_path: Path,
) -> None:
    predecessor = {**control(), "source_commit": TRAIN01_07_SOURCE_COMMIT}
    successor = {**predecessor, "source_commit": "b" * 40}
    snapshot(tmp_path, 2, header_control=predecessor)
    ledger, _records, latest = build_ledger(tmp_path, successor, source_control=predecessor)
    assert latest == 2
    migration = ledger["source_migration"]
    assert migration["from_source_commit"] == TRAIN01_07_SOURCE_COMMIT
    assert migration["to_source_commit"] == successor["source_commit"]
    assert ledger["allocation"] == allocation(successor)


def test_build_ledger_refuses_source_migration_that_changes_recipe_or_foreign_predecessor(
    tmp_path: Path,
) -> None:
    predecessor = {**control(), "source_commit": TRAIN01_07_SOURCE_COMMIT}
    snapshot(tmp_path, 2, header_control=predecessor)
    with pytest.raises(ValueError, match="only source_commit"):
        build_ledger(
            tmp_path,
            {**predecessor, "source_commit": "b" * 40, "outer_iterations": 5},
            source_control=predecessor,
        )
    with pytest.raises(ValueError, match="only source_commit"):
        build_ledger(
            tmp_path,
            {**predecessor, "source_commit": "b" * 40},
            source_control={**predecessor, "source_commit": "c" * 40},
        )


def test_completion_packet_rebinds_only_verified_train01_07_snapshot_to_template_pin(
    tmp_path: Path,
) -> None:
    predecessor = {**control(), "source_commit": TRAIN01_07_SOURCE_COMMIT}
    successor = {**predecessor, "source_commit": "b" * 40}
    snapshot(tmp_path, 2, header_control=predecessor)
    prior = {**manifest(), "commit": predecessor["source_commit"]}
    prior_receipt = receipt()
    prior_receipt["commit"] = predecessor["source_commit"]
    target_input = {
        "path": "/target-detached-checkout/target-input",
        "dest": "target-input",
        "sha256": "d" * 64,
    }
    target = {**manifest(), "commit": successor["source_commit"], "inputs": [target_input]}
    packet = completion_packet(
        prior,
        prior_receipt,
        tmp_path,
        predecessor,
        completion_index=1,
        job_id="sandbox-a-train01-08",
        output_dir=Path("/tmp/sandbox-a-train01-08"),
        original_manifest_sha256="b" * 64,
        successor_manifest=target,
        successor_control=successor,
    )
    assert packet.manifest["commit"] == successor["source_commit"]
    assert packet.manifest["inputs"][0] == target_input
    assert packet.ledger["source_migration"]["from_source_commit"] == TRAIN01_07_SOURCE_COMMIT
    assert packet.ledger["source_migration"]["to_source_commit"] == successor["source_commit"]


def test_build_ledger_binds_train01_11_post_preparation_snapshot_to_successor(
    tmp_path: Path,
) -> None:
    predecessor = {**control(), "source_commit": TRAIN01_11_SOURCE_COMMIT}
    successor = {**predecessor, "source_commit": "e" * 40}
    snapshot(tmp_path, 0, phase="pre-actor", prepare_phase="ready", header_control=predecessor)
    ledger, records, latest = build_ledger(tmp_path, successor, source_control=predecessor)
    assert latest == 0
    assert [row["iteration"] for row in ledger["units"]] == [0]
    assert len(records) == 9
    migration = ledger["source_migration"]
    assert migration["from_source_commit"] == TRAIN01_11_SOURCE_COMMIT
    assert migration["to_source_commit"] == successor["source_commit"]
    assert ledger["allocation"] == allocation(successor)


@pytest.mark.parametrize(
    "eligible_receipt",
    [
        # Host whole-session wall cap (SUPERVISOR, terminal=None)
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('whole-session wall cap reached')",
        },
        # Host deadline / unit cap / lease / read-retry SUPERVISOR stops
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('whole-session deadline reached')",
        },
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('priced session unit cap reached')",
        },
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('retention lease elapsed')",
        },
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('read retry budget exhausted')",
        },
        # VM runtime timeout terminating child (CHILD_NONZERO_EXIT with reason=timeout)
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": -15,
                "reason": "timeout",
            },
        },
        # VM lease expiry / credential expiry
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": 0,
                "reason": "lease_expired",
            },
        },
        {
            "failure_class": "SUPERVISOR",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": -9,
                "reason": "release_credential_expiring",
            },
        },
        # RPC poll timeout with matching mechanical terminal
        {
            "failure_class": "RPC_EXEC_POLL_TIMEOUT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": -15,
                "reason": "timeout",
            },
        },
    ],
)
def test_completion_packet_admits_legitimate_wall_and_child_timeout_receipts(
    tmp_path: Path, eligible_receipt: dict
) -> None:
    snapshot(tmp_path, 2)
    observed = {**receipt(), **eligible_receipt}
    packet = completion_packet(
        manifest(),
        observed,
        tmp_path,
        control(),
        completion_index=1,
        job_id="train-01-timeout-c1",
        output_dir=Path("/tmp/train-01-timeout-c1"),
        original_manifest_sha256="b" * 64,
    )
    assert packet.latest_iteration == 2
    assert packet.manifest["completion_attempt"]["resume_from_iteration"] == 2


@pytest.mark.parametrize(
    "ineligible_receipt",
    [
        # Scientific tripwire / StallWatchdog os._exit(1) (reason is None)
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": 1,
                "reason": None,
            },
        },
        # Operator KeyboardInterrupt
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "KeyboardInterrupt()",
        },
        # Missing required export on completed child
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('whole-session wall cap reached; missing required export')",
        },
        # Bare SUPERVISOR with no mechanical marker
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('unexpected state corruption')",
        },
        # Completed terminal cannot be continued
        {
            "failure_class": "SUPERVISOR",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "completed",
                "returncode": 0,
                "reason": "timeout",
            },
        },
        # Non-mechanical failure classes
        {"failure_class": "UPLOAD_CAP"},
        {"failure_class": "ACCOUNTING"},
        {"failure_class": "FOREIGN_GENERATION"},
        # Unverified stop or dirty cleanup
        {
            "failure_class": "SUPERVISOR",
            "error": "whole-session wall cap reached",
            "stop_verified": False,
        },
        {
            "failure_class": "SUPERVISOR",
            "error": "whole-session wall cap reached",
            "cleanup_error": "unassign failed",
        },
        {
            "failure_class": "SUPERVISOR",
            "error": "whole-session wall cap reached",
            "status": "PASS",
        },
    ],
)
def test_completion_packet_refuses_non_mechanical_or_unreleased_receipts(
    tmp_path: Path, ineligible_receipt: dict
) -> None:
    snapshot(tmp_path, 2)
    observed = {**receipt(), **ineligible_receipt}
    with pytest.raises(ResumeRefused):
        completion_packet(
            manifest(),
            observed,
            tmp_path,
            control(),
            completion_index=1,
            job_id="train-01-refused-c1",
            output_dir=Path("/tmp/train-01-refused-c1"),
            original_manifest_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    "conflicting_receipt",
    [
        # Timeout race: CHILD_NONZERO_EXIT reason="timeout" + FloatingPointError tripwire
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": 1,
                "reason": "timeout",
                "last_traceback_line": "FloatingPointError: non-finite loss tripwire",
            },
        },
        # SUPERVISOR wall-cap error string combined with FloatingPointError tripwire
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": (
                "SupervisorError('whole-session wall cap reached; "
                "FloatingPointError: non-finite loss tripwire')"
            ),
        },
        # Timeout race: CHILD_NONZERO_EXIT reason="timeout" + PreparationFailure
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": 1,
                "reason": "timeout",
                "last_traceback_line": "PreparationFailure: critic holdout late EV below 0.10",
            },
        },
        # SUPERVISOR wall-cap error string combined with PreparationFailure
        {
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": (
                "Refused('whole-session wall cap reached; "
                "PreparationFailure: invalid calibrated temperature')"
            ),
        },
        # Lease expiry race with action collapse RuntimeError
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": -15,
                "reason": "lease_expired",
                "last_error_line": (
                    "RuntimeError: action collapse exceeds 90 percent "
                    "over a complete 100k-row window"
                ),
            },
        },
        # Stall watchdog returncode 124 despite timeout reason
        {
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": 124,
                "reason": "timeout",
            },
        },
        # RPC poll timeout with non-empty stall-watchdog faulthandler dump in receipt exports
        {
            "failure_class": "RPC_EXEC_POLL_TIMEOUT",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": -15,
                "reason": "timeout",
            },
            "exports": [
                {"path": "training/stall-watchdog/faulthandler-stacks.log", "size": 512},
            ],
        },
        # SUPERVISOR wall-cap stop with distribution-screen-failed in terminal job_log_tail
        {
            "failure_class": "SUPERVISOR",
            "error": "SupervisorError('whole-session wall cap reached')",
            "terminal": {
                "job_id": "train-01",
                "generation": "g-1",
                "status": "failed",
                "returncode": -15,
                "reason": "timeout",
                "job_log_tail": 'TRAIN01_PROGRESS {"stage": "distribution-screen-failed"}\n',
            },
        },
    ],
)
def test_scientific_failure_evidence_takes_precedence_over_timeout_markers(
    tmp_path: Path, conflicting_receipt: dict
) -> None:
    snapshot(tmp_path, 2)
    observed = {**receipt(), **conflicting_receipt}
    with pytest.raises(ResumeRefused, match="scientific|watchdog"):
        verify_mechanical_stop(observed, manifest(), "b" * 64)
    with pytest.raises(ResumeRefused, match="scientific|watchdog"):
        completion_packet(
            manifest(),
            observed,
            tmp_path,
            control(),
            completion_index=1,
            job_id="train-01-scientific-refused",
            output_dir=Path("/tmp/train-01-scientific-refused"),
            original_manifest_sha256="b" * 64,
        )


def test_verify_mechanical_stop_inspects_output_root_telemetry_without_false_positives(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "attempt/output"
    exports = output_root / "exports/training"
    watchdog_dir = exports / "stall-watchdog"
    watchdog_dir.mkdir(parents=True)
    snapshot(exports, 2)

    # Healthy run seals an empty 0-byte faulthandler-stacks.log and emits first_nonfinite_term=null
    # and STALL_WATCHDOG_DIAGNOSTICS sealed_dir=... on clean exit.
    (watchdog_dir / "faulthandler-stacks.log").write_bytes(b"")
    (exports / "progress.json").write_text(json.dumps({"stage": "sigusr1-graceful-stop"}) + "\n")
    (output_root / "poll-000100.json").write_text(
        json.dumps(
            {
                "job_log_tail": (
                    'TRAIN01_PROGRESS {"stage": "update", "first_nonfinite_term": null}\n'
                    'TRAIN01_PROGRESS {"schema": "sandbox-a-train-sigusr1-checkpoint/v1"}\n'
                    "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/content/out/stall-watchdog\n"
                ),
                "exports": [
                    {"path": "training/stall-watchdog/faulthandler-stacks.log", "size": 0},
                ],
            }
        )
        + "\n"
    )
    clean_wall_cap = {
        **receipt(),
        "failure_class": "SUPERVISOR",
        "terminal": None,
        "error": "SupervisorError('whole-session wall cap reached')",
    }
    verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)

    # Non-empty faulthandler stack dump in output_root refuses continuation.
    (watchdog_dir / "faulthandler-stacks.log").write_text("Thread 0x1 (most recent call first):\n")
    with pytest.raises(ResumeRefused, match="faulthandler-stacks.log is non-empty"):
        verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)
    (watchdog_dir / "faulthandler-stacks.log").write_bytes(b"")

    # Failed progress stage in output_root refuses continuation.
    (exports / "progress.json").write_text(json.dumps({"stage": "preparation-failed"}) + "\n")
    with pytest.raises(ResumeRefused, match="progress.json records failed stage"):
        verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)
    (exports / "progress.json").write_text(json.dumps({"stage": "update"}) + "\n")

    # Explicit non-finite report in latest poll log refuses continuation.
    (output_root / "poll-000101.json").write_text(
        json.dumps(
            {
                "job_log_tail": (
                    'TRAIN01_PROGRESS {"finite": false, ' '"first_nonfinite_term": "policy_loss"}\n'
                ),
            }
        )
        + "\n"
    )
    with pytest.raises(ResumeRefused, match="explicit scientific/non-mechanical fault"):
        verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)


def test_scientific_failure_precedence_scans_all_retained_history_beyond_eight_polls(
    tmp_path: Path,
) -> None:
    """Astra adversarial regression: fault in poll 1 + >8 healthy polls must still REFUSE."""
    output_root = tmp_path / "attempt/output"
    exports = output_root / "exports/training"
    exports.mkdir(parents=True)
    snapshot(exports, 2)

    gen = "59e68cce673b4f90a8c2e1d0b7c6a5f4"
    clean_wall_cap = {
        **receipt(),
        "generation": gen,
        "failure_class": "SUPERVISOR",
        "terminal": None,
        "error": "SupervisorError('whole-session wall cap reached')",
    }
    # Poll 1 has FloatingPointError; 20 newer same-generation polls have healthy progress.
    (output_root / "poll-000001.json").write_text(
        json.dumps(
            {
                "job": "train-01",
                "generation": gen,
                "status": "running",
                "job_log_tail": "FloatingPointError: non-finite loss tripwire\n",
                "exports": [],
            }
        )
        + "\n"
    )
    for index in range(2, 22):
        (output_root / f"poll-{index:06d}.json").write_text(
            json.dumps(
                {
                    "job": "train-01",
                    "generation": gen,
                    "status": "running",
                    "job_log_tail": (
                        f'TRAIN01_PROGRESS {{"stage": "ppo", "iteration": {index}, '
                        f'"first_nonfinite_term": null}}\n'
                    ),
                    "exports": [],
                }
            )
            + "\n"
        )
    with pytest.raises(ResumeRefused, match="explicit scientific/non-mechanical fault"):
        verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)
    with pytest.raises(ResumeRefused, match="explicit scientific/non-mechanical fault"):
        completion_packet(
            manifest(),
            clean_wall_cap,
            exports,
            control(),
            completion_index=1,
            job_id="train-01-retained-fault-c1",
            output_dir=Path("/tmp/train-01-retained-fault-c1"),
            original_manifest_sha256="b" * 64,
            output_root=output_root,
        )

    # Replace poll-000001 with healthy telemetry and confirm baseline admits, then verify
    # an early fault in supervisor.log (followed by 20 healthy lines) also refuses closed.
    (output_root / "poll-000001.json").write_text(
        json.dumps(
            {
                "job": "train-01",
                "generation": gen,
                "status": "running",
                "job_log_tail": 'TRAIN01_PROGRESS {"stage": "ppo", "iteration": 1}\n',
                "exports": [],
            }
        )
        + "\n"
    )
    verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)

    sup_log = tmp_path / "ops-supervisor.log"
    sup_lines = [
        json.dumps(
            {
                "event": "poll",
                "job": "train-01",
                "job_log_tail": "FloatingPointError: non-finite loss tripwire\n",
            }
        )
    ] + [
        json.dumps(
            {
                "event": "poll",
                "job": "train-01",
                "job_log_tail": f'TRAIN01_PROGRESS {{"stage": "ppo", "iteration": {idx}}}\n',
            }
        )
        for idx in range(2, 22)
    ]
    sup_log.write_text("\n".join(sup_lines) + "\n")
    with pytest.raises(ResumeRefused, match="explicit scientific/non-mechanical fault"):
        verify_mechanical_stop(
            clean_wall_cap,
            manifest(),
            "b" * 64,
            output_root=output_root,
            supervisor_log_path=sup_log,
        )

    # Corrupt early poll or foreign generation early poll must also fail closed.
    saved_poll1 = (output_root / "poll-000001.json").read_text()
    (output_root / "poll-000001.json").write_text("{corrupt-json\n")
    with pytest.raises(ResumeRefused, match="corrupt poll evidence"):
        verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)
    (output_root / "poll-000001.json").write_text(
        json.dumps({"job": "train-01", "generation": "f" * 32, "status": "running"}) + "\n"
    )
    with pytest.raises(ResumeRefused, match="foreign generation"):
        verify_mechanical_stop(clean_wall_cap, manifest(), "b" * 64, output_root=output_root)
    (output_root / "poll-000001.json").write_text(saved_poll1)


def _setup_enospc_fixture(tmp_path: Path) -> dict:
    from research.sandbox_a_resident.training_resume import (
        ACCOUNT_RELEASE_CAPTURE_SCHEMA,
        HOST_ENOSPC_FAILURE_CLASS,
        HOST_ENOSPC_RECOVERY_SCHEMA,
        compute_retained_evidence_inventory,
    )
    from research.sandbox_a_resident.training_units import NAMESPACE
    from research.sandbox_a_resident.units import digest

    attempt = tmp_path / "train01-attempt-15"
    state_dir = attempt / "state"
    output_root = attempt / "output"
    exports = output_root / "exports/training"
    account_root = tmp_path / "colab-account/snake"
    recovery_dir = tmp_path / "enospc-recovery"
    for directory in (state_dir, exports, account_root, recovery_dir):
        directory.mkdir(parents=True)

    ctrl = control()
    snapshot(exports, 2)
    gen = "59e68cce673b4f90a8c2e1d0b7c6a5f4"
    job_id = "sandbox-a-train01-20261027-15"
    session = f"snake-{job_id}-{gen[:12]}"
    endpoint = "m-s-1b2c3d4e5f6a"
    exec_manifest = {
        "job_id": job_id,
        "commit": ctrl["source_commit"],
        "output_dir": str(output_root.resolve()),
        "inputs": [],
    }
    exec_manifest_path = output_root / "manifest.json"
    exec_manifest_path.write_text(json.dumps(exec_manifest, sort_keys=True) + "\n")
    exec_manifest_sha = digest(exec_manifest_path)

    assignment = {
        "endpoint": endpoint,
        "generation": gen,
        "session": session,
        "output": str(output_root.resolve()),
    }
    assignment_path = state_dir / "assignment.json"
    assignment_path.write_text(json.dumps(assignment, sort_keys=True) + "\n")
    active_path = account_root / f"active-{session}.json"
    active_path.write_text(json.dumps(assignment, sort_keys=True) + "\n")

    for idx in range(1, 13):
        (output_root / f"poll-{idx:06d}.json").write_text(
            json.dumps(
                {
                    "job": job_id,
                    "generation": gen,
                    "status": "running",
                    "state": {"job_id": job_id, "generation": gen, "status": "running"},
                    "job_log_tail": f'TRAIN01_PROGRESS {{"stage": "ppo", "iteration": {idx}}}\n',
                    "exports": [],
                }
            )
            + "\n"
        )
    (output_root / "spend-000001.json").write_text(json.dumps({"spent": 12.5}) + "\n")
    (output_root / "usage-000001.json").write_text(json.dumps({"balance": 1050.0}) + "\n")

    sup_log = tmp_path / "ops-logs" / f"{job_id}.log"
    sup_log.parent.mkdir(parents=True)
    log_lines = (
        ["PREFLIGHT_OK"]
        + [
            json.dumps(
                {
                    "event": "poll",
                    "job": job_id,
                    "index": idx,
                    "job_log_tail": f'TRAIN01_PROGRESS {{"stage": "ppo", "iteration": {idx}}}\n',
                }
            )
            for idx in range(1, 13)
        ]
        + [
            f"[Errno 28] No space left on device: '{output_root.resolve()}/.receipt-ihix0kmn'",
            "EXIT=1",
        ]
    )
    sup_log.write_text("\n".join(log_lines) + "\n")
    captured_ts = sup_log.stat().st_mtime + 10.0

    capture_path = recovery_dir / "account-release-status.json"
    capture_payload = {
        "schema": ACCOUNT_RELEASE_CAPTURE_SCHEMA,
        "captured_at_utc": "2026-10-28T13:00:00Z",
        "captured_unix_seconds": captured_ts,
        "status": {
            "sessions": {},
            "usage": {"assignments": 0, "balance": 1050.0, "rate": 0.0},
            "unresolved": [],
        },
    }
    publish(capture_path, lambda dst: dst.write_text(json.dumps(capture_payload) + "\n"))

    ledger, _uploads, latest = build_ledger(exports, ctrl)
    latest_exp = ledger["units"][-1]["exports"][0]
    src_pkt_sha = "c" * 64

    def write_cert(name: str, payload: dict, *, directory: Path = recovery_dir) -> Path:
        target = directory / name
        publish(target, lambda dst: dst.write_text(json.dumps(payload, sort_keys=True) + "\n"))
        return target

    base_cert = {
        "schema": HOST_ENOSPC_RECOVERY_SCHEMA,
        "original_attempt_state": {
            "prior_attempt_status": "INCOMPLETE",
            "failure_class": HOST_ENOSPC_FAILURE_CLASS,
            "original_receipt_present": False,
            "original_terminal_status": "UNKNOWN_UNAVAILABLE",
            "original_supervisor_error": "UNKNOWN_UNAVAILABLE",
            "original_cleanup_error": "UNKNOWN_UNAVAILABLE",
            "post_hoc_release_verified": True,
        },
        "identities": {
            "study": ctrl["study"],
            "namespace": NAMESPACE,
            "job_id": job_id,
            "source_packet_id": "packet-train01-15",
            "source_packet_manifest_sha256": src_pkt_sha,
            "executed_manifest_sha256": exec_manifest_sha,
            "source_commit": ctrl["source_commit"],
            "control_allocation": allocation(ctrl),
            "output_dir": str(output_root.resolve()),
            "generation": gen,
            "endpoint": endpoint,
            "session": session,
            "assignment_record": {
                "path": str(assignment_path.resolve()),
                "size": assignment_path.stat().st_size,
                "sha256": digest(assignment_path),
            },
        },
        "supervisor_log": {
            "path": str(sup_log.resolve()),
            "size": sup_log.stat().st_size,
            "sha256": digest(sup_log),
            "line_count": len(log_lines),
            "terminal_enospc_line": log_lines[-2],
            "terminal_exit_line": "EXIT=1",
        },
        "retained_evidence": compute_retained_evidence_inventory(output_root),
        "release_evidence": {
            "account_status_capture": {
                "path": str(capture_path.resolve()),
                "size": capture_path.stat().st_size,
                "sha256": digest(capture_path),
                "captured_at_utc": capture_payload["captured_at_utc"],
                "captured_unix_seconds": captured_ts,
            },
            "broker_reconciliation": {
                "mode": "verified_disappearance_with_bound_stale_active",
                "account_root": str(account_root.resolve()),
                "stale_active_record": {
                    "path": str(active_path.resolve()),
                    "size": active_path.stat().st_size,
                    "sha256": digest(active_path),
                },
                "operator_reconcile_command": (
                    f"python -m research.colab_cli.supervisor recover {active_path.resolve()}"
                ),
            },
        },
        "resume_binding": {
            "verified_snapshot_iterations": [2],
            "latest_eligible_iteration": latest,
            "latest_eligible_unit": unit_id(latest),
            "latest_full_state_size": latest_exp["size"],
            "latest_full_state_sha256": latest_exp["sha256"],
        },
    }
    return {
        "ctrl": ctrl,
        "output_root": output_root,
        "exports": exports,
        "exec_manifest": exec_manifest,
        "exec_manifest_sha": exec_manifest_sha,
        "src_pkt_sha": src_pkt_sha,
        "sup_log": sup_log,
        "capture_path": capture_path,
        "capture_payload": capture_payload,
        "captured_ts": captured_ts,
        "account_root": account_root,
        "active_path": active_path,
        "session": session,
        "endpoint": endpoint,
        "gen": gen,
        "job_id": job_id,
        "base_cert": base_cert,
        "write_cert": write_cert,
    }


def test_host_enospc_recovery_admits_valid_evidence_and_preserves_absent_receipt(
    tmp_path: Path,
) -> None:
    from research.sandbox_a_resident.training_resume import verify_host_enospc_recovery

    fx = _setup_enospc_fixture(tmp_path)
    cert_path = fx["write_cert"]("cert-valid.json", fx["base_cert"])
    now_ts = fx["captured_ts"] + 30.0

    verified = verify_host_enospc_recovery(
        cert_path,
        fx["exec_manifest"],
        fx["exec_manifest_sha"],
        output_root=fx["output_root"],
        source_control=fx["ctrl"],
        source_packet_id="packet-train01-15",
        source_packet_manifest_sha256=fx["src_pkt_sha"],
        now_unix_seconds=now_ts,
    )
    assert verified["prior_attempt_status"] == "INCOMPLETE"
    assert verified["original_receipt_present"] is False
    assert verified["original_terminal_status"] == "UNKNOWN_UNAVAILABLE"
    assert verified["original_supervisor_error"] == "UNKNOWN_UNAVAILABLE"
    assert verified["original_cleanup_error"] == "UNKNOWN_UNAVAILABLE"
    assert verified["post_hoc_release_verified"] is True
    assert verified["stale_active_present_now"] is True
    assert verified["polls_scanned"] == 12
    assert not (fx["output_root"] / "receipt.json").exists()

    pkt = completion_packet(
        fx["exec_manifest"],
        None,
        fx["exports"],
        fx["ctrl"],
        completion_index=1,
        job_id="sandbox-a-train01-20261028-15-c1-i0002",
        output_dir=tmp_path / "attempt-15-c1/output",
        original_manifest_sha256=fx["exec_manifest_sha"],
        output_root=fx["output_root"],
        recovery_evidence_path=cert_path,
        source_packet_id="packet-train01-15",
        source_packet_manifest_sha256=fx["src_pkt_sha"],
        max_recovery_capture_age_seconds=None,
    )
    assert pkt.latest_iteration == 2
    assert not (fx["output_root"] / "receipt.json").exists()

    # Simulating `supervisor recover` removing active-<session>.json WITHOUT recovery-*.json fails;
    # WITH verified recovery-*.json in output_root, the same sealed certificate remains valid.
    saved_active = fx["active_path"].read_bytes()
    fx["active_path"].unlink()
    with pytest.raises(ResumeRefused, match="without a verified recovery-"):
        verify_host_enospc_recovery(
            cert_path,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=now_ts,
        )
    (fx["output_root"] / "recovery-abcd1234.json").write_text(
        json.dumps({"endpoint": fx["endpoint"], "stop_verified": True}) + "\n"
    )
    verified_after_recover = verify_host_enospc_recovery(
        cert_path,
        fx["exec_manifest"],
        fx["exec_manifest_sha"],
        output_root=fx["output_root"],
        source_control=fx["ctrl"],
        now_unix_seconds=now_ts,
    )
    assert verified_after_recover["stale_active_present_now"] is False
    fx["active_path"].write_bytes(saved_active)
    (fx["output_root"] / "recovery-abcd1234.json").unlink()


def test_host_enospc_recovery_refuses_wrong_identities_faults_stale_release_and_mutations(
    tmp_path: Path,
) -> None:
    import copy

    from research.sandbox_a_resident.training_resume import (
        compute_retained_evidence_inventory,
        verify_host_enospc_recovery,
    )
    from research.sandbox_a_resident.units import digest

    fx = _setup_enospc_fixture(tmp_path)
    now_ts = fx["captured_ts"] + 30.0
    counter = 0

    def check_refused(cert_payload: dict, match: str, *, now: float = now_ts) -> None:
        nonlocal counter
        counter += 1
        cpath = fx["write_cert"](f"cert-bad-{counter}.json", cert_payload)
        with pytest.raises(ResumeRefused, match=match):
            verify_host_enospc_recovery(
                cpath,
                fx["exec_manifest"],
                fx["exec_manifest_sha"],
                output_root=fx["output_root"],
                source_control=fx["ctrl"],
                source_packet_id="packet-train01-15",
                source_packet_manifest_sha256=fx["src_pkt_sha"],
                now_unix_seconds=now,
                max_capture_age_seconds=3600.0,
            )

    # 1. Inventing cleanup_error=None or wrong attempt status refuses.
    invented_cleanup = copy.deepcopy(fx["base_cert"])
    invented_cleanup["original_attempt_state"]["original_cleanup_error"] = None
    check_refused(invented_cleanup, "without inventing cleanup_error=null")

    # 2. Wrong identities (job_id, source_packet_id, generation, session, endpoint, commit).
    for field, bad_val, pat in (
        ("job_id", "wrong-job", "identities do not match"),
        ("source_packet_id", "packet-train01-16", "source_packet_id mismatch"),
        ("source_packet_manifest_sha256", "0" * 64, "source_packet_manifest_sha256 mismatch"),
        ("executed_manifest_sha256", "0" * 64, "identities do not match"),
        ("source_commit", "f" * 40, "identities do not match"),
        ("generation", "f" * 32, "identities do not match"),
        ("session", "snake-wrong", "identities do not match"),
        ("endpoint", "m-s-wrong", "assignment.json disagrees"),
    ):
        bad = copy.deepcopy(fx["base_cert"])
        bad["identities"][field] = bad_val
        check_refused(bad, pat)

    # 3. Certificate named receipt.json or inside output_root refuses; existing receipt refuses.
    named_receipt = fx["write_cert"]("receipt.json", fx["base_cert"])
    with pytest.raises(ResumeRefused, match="never receipt.json"):
        verify_host_enospc_recovery(
            named_receipt,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=now_ts,
        )
    (fx["output_root"] / "receipt.json").write_text("{}\n")
    valid_cert = fx["write_cert"]("cert-when-receipt-present.json", fx["base_cert"])
    with pytest.raises(ResumeRefused, match="receipt.json exists"):
        verify_host_enospc_recovery(
            valid_cert,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=now_ts,
        )
    (fx["output_root"] / "receipt.json").unlink()

    # 4. Stale / future release capture timestamps refuse.
    check_refused(fx["base_cert"], "exceeds maximum admission age", now=fx["captured_ts"] + 7200.0)
    check_refused(fx["base_cert"], "lies in the future", now=fx["captured_ts"] - 120.0)

    # 5. Unverified release (sessions != {}, assignments != 0, rate != 0, unresolved != []).
    for idx_rel, bad_status in enumerate(
        [
            {
                "sessions": {"m-s-active": {}},
                "usage": {"assignments": 0, "balance": 100.0, "rate": 0.0},
                "unresolved": [],
            },
            {
                "sessions": {},
                "usage": {"assignments": 1, "balance": 100.0, "rate": 0.0},
                "unresolved": [],
            },
            {
                "sessions": {},
                "usage": {"assignments": 0, "balance": 100.0, "rate": 5.3},
                "unresolved": [],
            },
            {
                "sessions": {},
                "usage": {"assignments": 0, "balance": 100.0, "rate": 0.0},
                "unresolved": ["pending-x"],
            },
        ]
    ):
        bad_cap_path = tmp_path / f"bad-cap-{idx_rel}.json"
        bad_cap_payload = {**fx["capture_payload"], "status": bad_status}
        publish(bad_cap_path, lambda dst, val=bad_cap_payload: dst.write_text(json.dumps(val)))
        bad_cert = copy.deepcopy(fx["base_cert"])
        bad_cert["release_evidence"]["account_status_capture"].update(
            {
                "path": str(bad_cap_path.resolve()),
                "size": bad_cap_path.stat().st_size,
                "sha256": digest(bad_cap_path),
            }
        )
        check_refused(bad_cert, "does not prove empty server sessions")

    pending_file = fx["account_root"] / f"pending-{fx['session']}.json"
    pending_file.write_text("{}\n")
    check_refused(fx["base_cert"], "unresolved broker pending record")
    pending_file.unlink()

    # 6. Conflicting scientific fault in poll-000001 (even with 11 newer healthy polls and re-hashed
    # inventory) or in supervisor.log refuses.
    poll1 = fx["output_root"] / "poll-000001.json"
    saved_poll1 = poll1.read_text()
    poll1.write_text(
        json.dumps(
            {
                "job": fx["job_id"],
                "generation": fx["gen"],
                "status": "running",
                "job_log_tail": "FloatingPointError: non-finite loss tripwire\n",
                "exports": [],
            }
        )
        + "\n"
    )
    # Without re-hashing -> caught by retained_evidence hash binding!
    check_refused(fx["base_cert"], "retained output_root evidence inventory/hashes drifted")
    # Even WITH re-hashed retained_evidence -> caught by full-history scientific fault scanner!
    rehashed_fault_cert = copy.deepcopy(fx["base_cert"])
    rehashed_fault_cert["retained_evidence"] = compute_retained_evidence_inventory(
        fx["output_root"]
    )
    check_refused(rehashed_fault_cert, "explicit scientific/non-mechanical fault")
    poll1.write_text(saved_poll1)

    # 7. Mutated supervisor.log hash, non-ENOSPC supervisor.log tail, or mutated certificate body.
    saved_log = fx["sup_log"].read_text()
    fx["sup_log"].write_text(saved_log + "# trailing comment\n")
    check_refused(fx["base_cert"], "supervisor_log size or SHA-256 differs")

    non_enospc_lines = saved_log.splitlines()[:-2] + ["RuntimeError: disk check", "EXIT=1"]
    fx["sup_log"].write_text("\n".join(non_enospc_lines) + "\n")
    bad_tail_cert = copy.deepcopy(fx["base_cert"])
    bad_tail_cert["supervisor_log"].update(
        {
            "size": fx["sup_log"].stat().st_size,
            "sha256": digest(fx["sup_log"]),
            "line_count": len(non_enospc_lines),
            "terminal_enospc_line": non_enospc_lines[-2],
        }
    )
    check_refused(bad_tail_cert, "does not end with receipt-write ENOSPC")
    fx["sup_log"].write_text(saved_log)

    tampered_cert = fx["write_cert"]("cert-tampered.json", fx["base_cert"])
    tampered_cert.write_text(
        tampered_cert.read_text().replace("2026-10-28T13:00:00Z", "2026-10-28T13:00:01Z")
    )
    with pytest.raises(ResumeRefused, match="corrupt export"):
        verify_host_enospc_recovery(
            tampered_cert,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=now_ts,
        )
