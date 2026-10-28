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
    with pytest.raises(ResumeRefused, match=r"progress.json.*records failed stage"):
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
    with pytest.raises(
        ResumeRefused, match="corrupt poll evidence|explicit scientific/non-mechanical fault"
    ):
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

    active_path.unlink()
    recovery_path = output_root / "recovery-qualified.json"
    recovery_path.write_text(json.dumps({"endpoint": endpoint, "stop_verified": True}) + "\n")

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
                "mode": "qualified_recover_receipt",
                "account_root": str(account_root.resolve()),
                "recover_receipt": {
                    "path": str(recovery_path.resolve()),
                    "size": recovery_path.stat().st_size,
                    "sha256": digest(recovery_path),
                },
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
    assert verified["stale_active_present_now"] is False
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

    # A later arbitrary receipt cannot replace the exact sealed recovery proof.
    (fx["output_root"] / "recovery-arbitrary.json").write_text(
        json.dumps({"endpoint": fx["endpoint"], "stop_verified": True}) + "\n"
    )
    with pytest.raises(ResumeRefused, match="inventory/hashes drifted"):
        verify_host_enospc_recovery(
            cert_path,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=now_ts,
        )


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


@pytest.mark.parametrize(
    "case",
    [
        "retained-child-nonzero",
        "retained-state-runtime-failure",
        "retained-output-log-tripwire",
        "retained-partial-progress-fault",
        "retained-failed-header-field",
        "unhashed-recovery-receipt",
        "nested-text-fault",
        "truncated-partial-fault",
        "unexplained-state-failed",
        "foreign-terminal-generation",
        "foreign-terminal-job",
        "foreign-state-generation",
        "mechanical-reason-with-fault",
        "malformed-returncode",
        "unbound-recovery-mode",
        "missing-bound-recovery",
        "mutated-bound-recovery",
        "pending-reconciliation",
    ],
)
def test_review_admission_probes_refuse(tmp_path: Path, case: str) -> None:
    """Regression probes from astra-review-continuation3 plus adjacent refusal cases."""
    import copy

    from research.sandbox_a_resident.training_resume import (
        compute_retained_evidence_inventory,
        verify_host_enospc_recovery,
    )

    fx = _setup_enospc_fixture(tmp_path)
    cert = copy.deepcopy(fx["base_cert"])
    poll_path = fx["output_root"] / "poll-000001.json"
    poll = json.loads(poll_path.read_text())
    terminal = {
        "job_id": fx["job_id"],
        "generation": fx["gen"],
        "status": "failed",
        "returncode": 7,
        "reason": None,
        "last_traceback_line": "EXIT=7",
        "exports": [],
    }
    if case == "retained-child-nonzero":
        poll["terminal"] = terminal
    elif case == "retained-state-runtime-failure":
        poll["state"].update(status="failed", reason="runtime_error", returncode=7)
    elif case == "unexplained-state-failed":
        poll["state"].update(status="failed", reason=None, returncode=0)
    elif case.startswith("foreign-terminal-"):
        terminal.update(reason="timeout", returncode=-15)
        terminal["generation" if case.endswith("generation") else "job_id"] = "foreign"
        poll["terminal"] = terminal
    elif case == "foreign-state-generation":
        poll["state"]["generation"] = "foreign"
    elif case == "malformed-returncode":
        poll["state"].update(returncode="7")
    elif case == "mechanical-reason-with-fault":
        terminal.update(reason="timeout", error="FloatingPointError: non-finite loss tripwire")
        poll["terminal"] = terminal
    elif case in {"retained-output-log-tripwire", "nested-text-fault"}:
        path = fx["exports"] / ("job.log" if case.startswith("retained") else "nested/notes.txt")
        path.parent.mkdir(exist_ok=True)
        path.write_text("FloatingPointError: non-finite loss tripwire\n")
    elif case in {"retained-partial-progress-fault", "truncated-partial-fault"}:
        text = json.dumps(
            {"stage": "distribution-screen-failed", "error": "material distribution gap"}
        )
        (fx["exports"] / "progress.json.partial").write_text(
            text if case.startswith("retained") else text[:-2]
        )
    elif case == "retained-failed-header-field":
        (fx["exports"] / "calibrated-boundary.json").write_text(
            json.dumps(
                {
                    "metadata": {"phase": "ppo"},
                    "error": "FloatingPointError: non-finite loss tripwire",
                }
            )
        )
    elif case == "unhashed-recovery-receipt":
        (fx["output_root"] / "recovery-arbitrary.json").write_text(
            json.dumps({"endpoint": fx["endpoint"], "stop_verified": True})
        )
    elif case == "unbound-recovery-mode":
        cert["release_evidence"]["broker_reconciliation"][
            "mode"
        ] = "verified_disappearance_with_bound_stale_active"
    elif case == "pending-reconciliation":
        fx["active_path"].write_text("{}")
    else:
        recovery = fx["output_root"] / "recovery-qualified.json"
        if case == "missing-bound-recovery":
            recovery.unlink()
        elif case == "mutated-bound-recovery":
            recovery.write_text(json.dumps({"endpoint": "foreign", "stop_verified": True}))
    poll_path.write_text(json.dumps(poll))
    # Rehash deliberately: refusal must come from semantic validation, not just drift.
    cert["retained_evidence"] = compute_retained_evidence_inventory(fx["output_root"])
    path = fx["write_cert"]("probe-certificate.json", cert)
    with pytest.raises(ResumeRefused):
        verify_host_enospc_recovery(
            path,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            source_packet_id="packet-train01-15",
            source_packet_manifest_sha256=fx["src_pkt_sha"],
            now_unix_seconds=fx["captured_ts"] + 30,
        )


def test_inventory_binds_every_retained_file(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import compute_retained_evidence_inventory

    fx = _setup_enospc_fixture(tmp_path)
    path = fx["exports"] / "job.log.partial"
    path.write_text("healthy progress")
    inventory = compute_retained_evidence_inventory(fx["output_root"])
    expected = {
        str(p.relative_to(fx["output_root"])) for p in fx["output_root"].rglob("*") if p.is_file()
    }
    assert {row["path"] for row in inventory["files"]} == expected
    path.write_text("changed progress")
    assert compute_retained_evidence_inventory(fx["output_root"]) != inventory


@pytest.mark.parametrize("reason", ["timeout", "lease_expired"])
def test_host_enospc_recovery_keeps_narrow_mechanical_exception(
    tmp_path: Path, reason: str
) -> None:
    from research.sandbox_a_resident.training_resume import (
        compute_retained_evidence_inventory,
        verify_host_enospc_recovery,
    )

    fx = _setup_enospc_fixture(tmp_path)
    poll_path = fx["output_root"] / "poll-000001.json"
    poll = json.loads(poll_path.read_text())
    poll["terminal"] = {
        "job_id": fx["job_id"],
        "generation": fx["gen"],
        "status": "failed",
        "returncode": -15,
        "reason": reason,
        "last_traceback_line": None,
    }
    poll_path.write_text(json.dumps(poll))
    cert = fx["base_cert"]
    cert["retained_evidence"] = compute_retained_evidence_inventory(fx["output_root"])
    cert_path = fx["write_cert"]("mechanical-cert.json", cert)
    result = verify_host_enospc_recovery(
        cert_path,
        fx["exec_manifest"],
        fx["exec_manifest_sha"],
        output_root=fx["output_root"],
        source_control=fx["ctrl"],
        now_unix_seconds=fx["captured_ts"] + 30,
    )
    assert result["prior_attempt_status"] == "INCOMPLETE"
    assert result["original_terminal_status"] == "UNKNOWN_UNAVAILABLE"


@pytest.mark.parametrize(
    "name",
    [
        "receipt.json.partial",
        "poll-001280.json.partial",
        "poll-001280.json.tmp123",
        "status.json.part",
        "status.jsonl",
        "status.jsonl.partial",
        "status.json.tmp123",
    ],
)
@pytest.mark.parametrize("defect", ["failed", "completed", "foreign-job", "foreign-generation"])
def test_review_r3_execution_filename_variants_refuse(
    tmp_path: Path, name: str, defect: str
) -> None:
    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    record = {"job_id": "train-01", "generation": "gen", "status": "running"}
    if defect == "failed":
        record.update(status="failed", returncode=7, reason=None)
    elif defect == "completed":
        record.update(status="completed", returncode=0)
    else:
        record["job_id" if defect == "foreign-job" else "generation"] = "foreign"
    text = json.dumps(record)
    if ".jsonl" in Path(name).suffixes:
        text = json.dumps({"message": "healthy first line"}) + "\n" + text
    (tmp_path / name).write_text(text + "\n")
    message = {
        "failed": "unknown execution status or failed status",
        "completed": "completed status",
        "foreign-job": "foreign job_id",
        "foreign-generation": "foreign generation",
    }[defect]
    with pytest.raises(ResumeRefused, match=message):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="gen"
        )


@pytest.mark.parametrize("suffix", [".json.tmp1", ".json.part", ".jsonl", ".jsonl.partial"])
@pytest.mark.parametrize(
    "record",
    [
        {"stage": "prep-failed"},
        {"finite": False},
        {"material_gaps": [1]},
        {"last_traceback_line": "EXIT=7"},
    ],
)
def test_review_r3_structured_diagnostics_refuse(tmp_path: Path, suffix: str, record: dict) -> None:
    from research.sandbox_a_resident.training_resume import _scan_retained_text

    path = tmp_path / ("progress" + suffix)
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(
        ResumeRefused,
        match=(
            "records failed stage"
            if "stage" in record
            else "explicit scientific/non-mechanical fault"
        ),
    ):
        _scan_retained_text(path, output_root=tmp_path)


@pytest.mark.parametrize("suffix", [".gz", ".xz", ".zip", ".bz2", ".partial", ""])
def test_review_r3_compressed_diagnostics_refuse(tmp_path: Path, suffix: str) -> None:
    import gzip

    from research.sandbox_a_resident.training_resume import _scan_retained_text

    path = tmp_path / ("job.log" + suffix)
    path.write_bytes(gzip.compress(b"FloatingPointError: non-finite loss tripwire"))
    with pytest.raises(
        ResumeRefused, match="compressed|scientific/non-mechanical fault|failed stage"
    ):
        _scan_retained_text(path, output_root=tmp_path)


@pytest.mark.parametrize("name", ["job.log", "job.log.partial", "diagnostic.txt", "stderr.err"])
@pytest.mark.parametrize("prefix", [b"\x00", b"\xe2\x82"])
def test_review_r3_binary_or_truncated_diagnostics_still_scan(
    tmp_path: Path, name: str, prefix: bytes
) -> None:
    from research.sandbox_a_resident.training_resume import _scan_retained_text

    path = tmp_path / name
    path.write_bytes(prefix + b"FloatingPointError: non-finite loss tripwire\n\xe2\x82")
    with pytest.raises(ResumeRefused, match="explicit scientific/non-mechanical fault"):
        _scan_retained_text(path, output_root=tmp_path)


@pytest.mark.parametrize("variant", [".partial", ".tmp123", ".part", "l", "l.partial"])
def test_review_r3_any_receipt_variant_refuses_host_enospc(tmp_path: Path, variant: str) -> None:
    from research.sandbox_a_resident.training_resume import verify_host_enospc_recovery

    fx = _setup_enospc_fixture(tmp_path)
    (fx["output_root"] / ("receipt.json" + variant)).write_text("{}\n")
    cert = fx["write_cert"]("receipt-variant-cert.json", fx["base_cert"])
    with pytest.raises(ResumeRefused, match="receipt.json"):
        verify_host_enospc_recovery(
            cert,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=fx["captured_ts"] + 30,
        )


@pytest.mark.parametrize("location", ["exports/calibrated-boundary.json", "supervisor.log"])
def test_review_r3_compound_mechanical_error_refuses(tmp_path: Path, location: str) -> None:
    from research.sandbox_a_resident.training_resume import (
        _check_supervisor_log_fault_evidence,
        _scan_retained_text,
    )

    path = tmp_path / location
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"error": "unexplained scientific error; whole-session wall cap reached"})
    )
    with pytest.raises(ResumeRefused):
        if path.name == "supervisor.log":
            _check_supervisor_log_fault_evidence(path)
        else:
            _scan_retained_text(path, output_root=tmp_path)


def test_review_r3_nested_supervisor_identity_refuses(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import _check_supervisor_log_fault_evidence

    path = tmp_path / "supervisor.log"
    path.write_text(
        json.dumps(
            {
                "job": "train-01",
                "poll": {
                    "terminal": {
                        "job_id": "foreign",
                        "generation": "gen",
                        "status": "running",
                    }
                },
            }
        )
    )
    with pytest.raises(ResumeRefused, match="foreign job_id"):
        _check_supervisor_log_fault_evidence(
            path, expected_job_id="train-01", expected_generation="gen"
        )


def test_review_r3_required_identity_is_required_without_expected_values() -> None:
    from research.sandbox_a_resident.training_resume import _check_retained_execution

    with pytest.raises(ResumeRefused, match="identity"):
        _check_retained_execution(
            {"status": "running"},
            label="terminal",
            expected_job_id=None,
            expected_generation=None,
            require_identity=True,
        )


def test_review_r3_recovery_descriptor_accepts_noncanonical_prefix(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import verify_host_enospc_recovery

    fx = _setup_enospc_fixture(tmp_path)
    cert = fx["base_cert"]
    descriptor = cert["release_evidence"]["broker_reconciliation"]["recover_receipt"]
    path = Path(descriptor["path"])
    descriptor["path"] = str(path.parent / ".." / path.parent.name / path.name)
    cert_path = fx["write_cert"]("noncanonical-cert.json", cert)
    result = verify_host_enospc_recovery(
        cert_path,
        fx["exec_manifest"],
        fx["exec_manifest_sha"],
        output_root=fx["output_root"],
        source_control=fx["ctrl"],
        now_unix_seconds=fx["captured_ts"] + 30,
    )
    assert result["post_hoc_release_verified"] is True


def test_review_r3_healthy_truncated_utf8_log_is_scanned(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import _scan_retained_text

    path = tmp_path / "job.log.partial"
    path.write_bytes(b"healthy progress\x00\xe2\x82")
    _scan_retained_text(path, output_root=tmp_path)


def test_review_r3_partial_poll_requires_full_poll_structure(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    (tmp_path / "poll-001280.json.tmp1").write_text(json.dumps({"state": "not-a-record"}))
    with pytest.raises(ResumeRefused, match="state is malformed"):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("tamper", [False, True])
def test_review_r3_source_archive_exception_requires_git_bytes(
    tmp_path: Path, tamper: bool
) -> None:
    import hashlib
    import io
    import subprocess
    import tarfile

    from research.sandbox_a_resident.training_resume import _scan_retained_text
    from research.sandbox_a_resident.units import digest

    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    name = "src/core/game_config.py"
    data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
    if tamper:
        data += b"\n# diagnostic hidden in source archive\n"
    staging = tmp_path / "staging"
    staging.mkdir()
    archive_path = staging / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    (tmp_path / "manifest.json").write_text(json.dumps({"commit": commit}))
    # Even a self-consistent, rehashed forged source plan must fail against Git.
    (staging / "source-plan.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "fallback_archive": archive_path.name,
                "fallback_sha256": digest(archive_path),
                "fallback_size": archive_path.stat().st_size,
                "files": {name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
            }
        )
    )
    if tamper:
        with pytest.raises(ResumeRefused, match="pinned Git bytes"):
            _scan_retained_text(archive_path, output_root=tmp_path)
    else:
        _scan_retained_text(archive_path, output_root=tmp_path)


@pytest.mark.parametrize(
    "failure_class", ["SUPERVISOR", "CHILD_NONZERO_EXIT", "RPC_EXEC_POLL_TIMEOUT"]
)
@pytest.mark.parametrize("location", ["status.json.partial", "supervisor.log", "nested.json"])
def test_review_r4_incomplete_exemption_is_only_the_validated_receipt(
    tmp_path: Path, failure_class: str, location: str
) -> None:
    from research.sandbox_a_resident.training_resume import (
        _check_output_root_fault_evidence,
        _check_supervisor_log_fault_evidence,
    )

    record = {**receipt(), "failure_class": failure_class}
    path = tmp_path / location
    path.write_text(json.dumps({"record": record} if location == "nested.json" else record))
    with pytest.raises(ResumeRefused, match="unknown execution status"):
        if location == "supervisor.log":
            _check_supervisor_log_fault_evidence(path)
        else:
            _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("failure_class", ["SUPERVISOR", "CHILD_NONZERO_EXIT"])
def test_review_r4_receiptless_enospc_never_excuses_incomplete_failure(
    tmp_path: Path, failure_class: str
) -> None:
    from research.sandbox_a_resident.training_resume import (
        compute_retained_evidence_inventory,
        verify_host_enospc_recovery,
    )
    from research.sandbox_a_resident.units import digest

    fx = _setup_enospc_fixture(tmp_path)
    log = fx["sup_log"]
    lines = log.read_text().splitlines()
    lines.insert(
        -2,
        json.dumps(
            {
                **receipt(),
                "job_id": fx["job_id"],
                "generation": fx["gen"],
                "failure_class": failure_class,
            }
        ),
    )
    log.write_text("\n".join(lines) + "\n")
    cert = fx["base_cert"]
    cert["supervisor_log"].update(
        size=log.stat().st_size, sha256=digest(log), line_count=len(lines)
    )
    cert["retained_evidence"] = compute_retained_evidence_inventory(fx["output_root"])
    cert_path = fx["write_cert"]("unexplained-incomplete.json", cert)
    with pytest.raises(ResumeRefused, match="unknown execution status"):
        verify_host_enospc_recovery(
            cert_path,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=fx["captured_ts"] + 30,
        )


def test_review_r4_receipt_exception_is_exact_and_nonrecursive(tmp_path: Path) -> None:
    rec = receipt()
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(rec))
    verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)
    for mutated in ({**rec, "job_id": "foreign"}, {**rec, "nested": rec}):
        path.write_text(json.dumps(mutated))
        with pytest.raises(ResumeRefused):
            verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)
    # A forged top-level envelope still goes through mechanical-trigger validation.
    forged = {**rec, "failure_class": "CHILD_NONZERO_EXIT"}
    path.write_text(json.dumps(forged))
    with pytest.raises(ResumeRefused, match="lacks a mechanical"):
        verify_mechanical_stop(forged, manifest(), "b" * 64, output_root=tmp_path)


@pytest.mark.parametrize(
    "record",
    [
        {"job": "foreign", "generation": "g-1", "status": "running"},
        {"job": "train-01", "job_id": "conflict", "status": "running"},
        {"status": "failed"},
        {"status": "crashed"},
        {"status": "aborted"},
        {"status": "INCOMPLETE", "failure_class": "CHILD_NONZERO_EXIT", "stop_verified": True},
        {"stop_verified": False},
    ],
)
def test_review_r4_alias_and_identityless_failures_refuse(tmp_path: Path, record: dict) -> None:
    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    (tmp_path / "status.json.part").write_text(json.dumps({"diagnostic": record}))
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("suffix", [".partial", ".part", ".tmp", ".tmp123", ".tmp-a.b"])
@pytest.mark.parametrize(
    "name,magic",
    [
        ("iteration-0128.pt.xz", b"\xfd7zXZ\x00"),
        ("iteration-0128-actor.pth", b"PK\x03\x04"),
        ("iteration-0128.pt", b"PK\x03\x04"),
    ],
)
def test_review_r4_model_partials_require_verified_bytes(
    tmp_path: Path, suffix: str, name: str, magic: bytes
) -> None:
    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    content = magic + b"opaque model state" * 20
    publish(training / name, lambda p: p.write_bytes(content))
    partial = training / (name + suffix)
    partial.write_bytes(content[:50])
    _check_output_root_fault_evidence(tmp_path)
    # Filename + magic alone cannot exempt a renamed fault diagnostic.
    partial.write_bytes(magic + b"FloatingPointError: hidden diagnostic")
    with pytest.raises(ResumeRefused, match="differs from verified payload"):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("tamper", ["none", "truncated", "changed", "unbound", "foreign-dir"])
def test_review_r4_complete_transfer_can_match_poll_hash_without_final_publication(
    tmp_path: Path, tamper: str
) -> None:
    import hashlib

    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    training = tmp_path / ("other" if tamper == "foreign-dir" else "exports/training")
    training.mkdir(parents=True)
    data = b"\xfd7zXZ\x00opaque checkpoint bytes"
    path = training / "iteration-0128.pt.xz.partial"
    path.write_bytes(
        data[:-1] if tamper == "truncated" else data + (b"x" if tamper == "changed" else b"")
    )
    entry = {
        "path": "training/iteration-0128.pt.xz",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (tmp_path / "poll-000001.json").write_text(
        json.dumps(
            {
                "job": "train-01",
                "generation": "g-1",
                "status": "running",
                "exports": [] if tamper == "unbound" else [entry],
            }
        )
    )
    if tamper == "none":
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )
    else:
        with pytest.raises(ResumeRefused):
            _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("sidecar", ["sha256", "complete"])
def test_review_r4_unfinalized_json_sidecars_are_scanned_without_json_parse(
    tmp_path: Path, sidecar: str
) -> None:
    import hashlib

    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    payload = tmp_path / "proxy-0128.json"
    payload.write_text('{"status": "PASS"}\n')
    path = Path(str(payload) + "." + sidecar)
    path.write_text(
        hashlib.sha256(payload.read_bytes()).hexdigest() + "\n" if sidecar == "sha256" else "comp"
    )
    _check_output_root_fault_evidence(tmp_path)
    payload.write_text('{"status": "failed"}\n')
    with pytest.raises(ResumeRefused):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize(
    "defect", ["utf8", "bad-sha", "bad-marker", "symlink-payload", "symlink-sha", "symlink-marker"]
)
def test_review_r4_publication_sidecars_fail_closed(tmp_path: Path, defect: str) -> None:
    from research.sandbox_a_resident.training_resume import _publication_sidecar

    payload = tmp_path / "proxy.json"
    publish(payload, lambda p: p.write_text('{"healthy": true}\n'))
    sha = Path(str(payload) + ".sha256")
    marker = Path(str(payload) + ".complete")
    if defect == "utf8":
        sha.write_bytes(b"\xe2\x82")
    elif defect == "bad-sha":
        sha.write_text("0" * 64)
    elif defect == "bad-marker":
        marker.write_text("FloatingPointError\n")
    else:
        target = {"symlink-payload": payload, "symlink-sha": sha, "symlink-marker": marker}[defect]
        saved = tmp_path / "saved"
        target.rename(saved)
        target.symlink_to(saved)
    with pytest.raises(ResumeRefused):
        _publication_sidecar(marker)


@pytest.mark.parametrize("sidecar", ["sha256", "complete"])
def test_review_r4_unfinalized_poll_sidecars_never_authenticate_payloads(
    tmp_path: Path, sidecar: str
) -> None:
    from research.sandbox_a_resident.training_resume import _check_output_root_fault_evidence

    (tmp_path / ("poll-001.json." + sidecar)).write_text(
        "0" * 64 + "\n" if sidecar == "sha256" else "complete\n"
    )
    # Both complete publication records require their published payload. A full
    # digest is permitted only with a matching staged sibling (covered by c2).
    with pytest.raises(ResumeRefused, match="finalized publication|orphan full digest"):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("prefix", ["spend", "usage"])
@pytest.mark.parametrize("suffix", [".json.partial", ".json.tmp1", ".jsonl"])
def test_review_r4_accounting_variant_inventory_is_bound(
    tmp_path: Path, prefix: str, suffix: str
) -> None:
    from research.sandbox_a_resident.training_resume import compute_retained_evidence_inventory

    path = tmp_path / (prefix + "-0001" + suffix)
    path.write_text('{"spent": 1}\n')
    before = compute_retained_evidence_inventory(tmp_path)
    assert before[prefix + "_count"] == 1
    path.write_text('{"spent": 2}\n')
    assert (
        compute_retained_evidence_inventory(tmp_path)[prefix + "_sha256"]
        != before[prefix + "_sha256"]
    )


@pytest.mark.parametrize("suffix", [".json.partial", ".json.tmp1", ".jsonl"])
def test_review_r4_unbound_recovery_variants_refuse(tmp_path: Path, suffix: str) -> None:
    from research.sandbox_a_resident.training_resume import (
        compute_retained_evidence_inventory,
        verify_host_enospc_recovery,
    )

    fx = _setup_enospc_fixture(tmp_path)
    (fx["output_root"] / ("recovery-other" + suffix)).write_text('{"stop_verified": true}\n')
    cert = fx["base_cert"]
    cert["retained_evidence"] = compute_retained_evidence_inventory(fx["output_root"])
    cert_path = fx["write_cert"]("unbound-recovery-variant.json", cert)
    with pytest.raises(ResumeRefused, match="unbound recovery"):
        verify_host_enospc_recovery(
            cert_path,
            fx["exec_manifest"],
            fx["exec_manifest_sha"],
            output_root=fx["output_root"],
            source_control=fx["ctrl"],
            now_unix_seconds=fx["captured_ts"] + 30,
        )


def test_review_r4_fault_marker_survives_multiple_nul_padded_chunks(tmp_path: Path) -> None:
    from research.sandbox_a_resident.training_resume import _scan_retained_text

    path = tmp_path / "job.log.partial"
    path.write_bytes(b"FloatingPoint" + b"\x00" * (65536 * 3) + b"Error: invalid learner\n")
    with pytest.raises(ResumeRefused, match="explicit scientific/non-mechanical fault"):
        _scan_retained_text(path, output_root=tmp_path)


def test_review_r4_verified_transport_partial_never_enters_resume_ledger(tmp_path: Path) -> None:
    import hashlib

    training = tmp_path / "exports/training"
    snapshot(training, 2)
    data = b"PK\x03\x04checkpoint transfer without final publication"
    (training / "iteration-0003.pt.partial").write_bytes(data)
    (tmp_path / "poll-000001.json").write_text(
        json.dumps(
            {
                "job": "train-01",
                "generation": "g-1",
                "status": "running",
                "exports": [
                    {
                        "path": "training/iteration-0003.pt",
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
        )
    )
    verify_mechanical_stop(receipt(), manifest(), "b" * 64, output_root=tmp_path)
    ledger, uploads, latest = build_ledger(training, control())
    assert latest == 2
    assert [unit["iteration"] for unit in ledger["units"]] == [2]
    assert all("0003" not in item["dest"] for item in uploads)
