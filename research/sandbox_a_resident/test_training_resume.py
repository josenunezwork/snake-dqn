"""Offline checks for TRAIN-01's Mac-side resume-packet builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.sandbox_a_resident.training_resume import (
    ResumeRefused,
    build_ledger,
    completion_packet,
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
