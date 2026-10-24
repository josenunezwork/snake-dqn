"""Prospective observer evidence through the real final-only admission gates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from research.colab_cli.acceptance import FINAL_ARTIFACT_BYTES, review_gates, source_manifest
from research.colab_cli.acceptance_faults import AcceptanceObserver, DirectObserverDownloadEvidence
from research.colab_cli.transfers import record_census_verified_download
from research.rpc_qualification import admission
from tests.test_l4_prerequisite_evidence import (
    _CLI,
    _prospective_real_poll_sequence,
    _prospective_receipt,
    _row,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def final_transfer(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A real hash/size-verified local final artifact, shared across these cases."""
    path = tmp_path_factory.mktemp("c6-final-only") / "final.bin"
    with path.open("wb") as stream:
        stream.truncate(FINAL_ARTIFACT_BYTES)
    return {
        "status": "PASS",
        "profile": "final-only",
        "hardware": "L4",
        "verified_on_mac": True,
        "exported_bytes": FINAL_ARTIFACT_BYTES,
        "elapsed_seconds": 1.0,
        "bytes_per_second_lower_bound": float(FINAL_ARTIFACT_BYTES),
        "path": str(path),
        "sha256": admission.sha(path),
    }


def _write_result(tmp_path: Path, cli: object, injection: dict, transfer: dict) -> Path:
    """Use real transport/admission validation; no classifier or gate is mocked."""
    jobs, execution = [], []
    for arm in ("cpu", "l4"):
        manifest = (
            copy.deepcopy(injection)
            if arm == "l4"
            else {
                "schema": "l4-label-faults-injection-manifest/v1",
                "written_before_dispatch": True,
                "session_job_id": "scope-cpu",
                "entries": [],
            }
        )
        # The isolated operation tests observer reads, not a watchdog unassign.
        manifest["entries"] = [e for e in manifest["entries"] if e["command"] != "terminal"]
        receipt = (
            _prospective_receipt(
                cli,
                job_id=injection["session_job_id"],
                marker_census=getattr(cli, "marker_census", {}),
            )
            if arm == "l4"
            else {
                "job_id": "scope-cpu",
                "status": "PASS",
                "stop_verified": True,
                "cli_calls": [
                    {
                        "command": "version",
                        "status": "completed",
                        "elapsed_seconds": 0.1,
                        "timeout_seconds": 30.0,
                    }
                ],
                "transfers": [],
                "marker_census": {},
            }
        )
        manifest_path = tmp_path / f"{arm}.injection.json"
        receipt_path = tmp_path / f"{arm}.receipt.json"
        manifest_path.write_text(json.dumps(manifest))
        receipt_path.write_text(json.dumps(receipt))
        jobs.append(
            {
                "name": arm,
                "injection_manifest": str(manifest_path),
                "injection_manifest_sha256": admission.sha(manifest_path),
            }
        )
        execution.append(
            {
                "job": arm,
                "status": "PASS",
                "receipt": str(receipt_path),
                "receipt_sha256": admission.sha(receipt_path),
            }
        )
    gates = copy.deepcopy(review_gates("final-only"))
    for gate in gates:
        gate["status"] = "OBSERVED" if gate["required"] else "NOT_REQUIRED"
        if gate["id"] == "final-artifact-download":
            gate["evidence_record"] = transfer
        elif gate["id"] == "cold-kernel-metadata-recovery":
            gate["evidence_record"] = {
                "status": "PASS",
                "arms": [
                    {
                        "arm": arm,
                        "metadata_recovery": {
                            "original_kernel_id": arm,
                            "recovered_kernel_id": arm,
                            "same_kernel_id": True,
                            "tracked_metadata_unchanged": True,
                        },
                        "first_upload_transfer": {
                            "status": "completed",
                            "attempts": 1,
                            "remote": "/content/input",
                            "size_bytes": 1,
                            "elapsed_seconds": 0.1,
                        },
                    }
                    for arm in ("cpu", "L4")
                ],
            }
    sources = source_manifest(REPO)
    result = {
        "schema": "snake-dqn-colab-cli-acceptance-result/v1",
        "qualified": True,
        "return_code": 0,
        "status": "PASS",
        "profile": "final-only",
        "source_manifest": sources,
        "final_artifact_transfer": transfer,
        "final_plan": {
            "commit": admission.git_commit(REPO),
            "status": "PASS",
            "profile": "final-only",
            "source_manifest": sources,
            "jobs": jobs,
            "execution": execution,
            "gates": gates,
        },
    }
    path = tmp_path / "acceptance-result.json"
    path.write_text(json.dumps(result))
    return path


def _make_real_failure(row: dict) -> None:
    row.pop("failure_type", None)
    row.pop("realized_selector_id", None)
    row.update(failure_class="RPC_EXEC_POLL_TIMEOUT", error="download timed out")


@pytest.mark.parametrize("probes", [0, 1, 2, 3, 80, 150, 300])
def test_lawful_probe_counts_preserve_three_attempt_allowance_at_admission(
    tmp_path: Path, final_transfer: dict, probes: int
) -> None:
    remote = "/content/scope-503/output/lease-checkpoint.pkl.sha256"
    cli = _CLI()
    evidence = DirectObserverDownloadEvidence(cli, {remote})
    injection = {
        "schema": "l4-label-faults-injection-manifest/v1",
        "written_before_dispatch": True,
        "session_job_id": "scope-many",
        "entries": [
            {
                "injection_id": "absent",
                "command": "download",
                "type": "absence-probe",
                "failure_type": "NOT_FOUND",
                "max_calls": max(1, probes),
                "realized_event_required": True,
                "selector": {
                    "kind": "planned_absence_probe",
                    "remote_suffix": "/lease-checkpoint.pkl.sha256",
                },
            }
        ],
    }
    # Real attempts 1 and 2 are followed by arbitrary lawful absences, then
    # attempt 3 verifies. Even probes at this edge cannot reset the allowance.
    for index in range(probes + 3):
        completed = index == probes + 2
        row = _row(remote, "completed" if completed else "failed", index + 1)
        if index < 2:
            _make_real_failure(row)
        elif not completed:
            row["error"] = "[colab] Download failed: File or directory not found: " + remote
        cli.call_records.append(row)
        evidence.observe_since(index)
        AcceptanceObserver._tag_observed_absence(
            cli, remote, declared_selectors={"/lease-checkpoint.pkl.sha256": "absent"}
        )
    assert evidence.bind_verified(remote, hashlib.sha256(b"verified checkpoint").hexdigest())
    path = _write_result(tmp_path, cli, injection, final_transfer)
    arm = admission.require_prerequisite(path, REPO, "l4")
    assert arm["max_attempts"] == 3
    assert len(arm["excluded_injections"]) == probes
    assert cli.call_records[-1]["transfer_attempt"] == probes + 3


@pytest.mark.parametrize("real_attempts", [2, 3, 4])
def test_raw_44_82_and_allowance_edge_reach_final_admission(
    tmp_path: Path, final_transfer: dict, real_attempts: int
) -> None:
    cli, injection = _prospective_real_poll_sequence()
    # Probes precede and follow genuine failures, including immediately before
    # the final read. They cannot reset a real count at the allowance edge.
    for index in range(real_attempts - 2):
        _make_real_failure(cli.call_records[index])
    path = _write_result(tmp_path, cli, injection, final_transfer)
    arm = admission.prerequisite_arms(path, REPO)["l4"]
    if real_attempts <= 3:
        assert arm["verdict"] == "PASS", arm
        assert arm["max_attempts"] == real_attempts
        assert admission.require_prerequisite(path, REPO, "l4")["verdict"] == "PASS"
    else:
        assert arm["counted_by_class"]["C6"] > 0
        with pytest.raises(ValueError, match="counted v2 faults"):
            admission.require_prerequisite(path, REPO, "l4")
    assert cli.call_records[43]["transfer_attempt"] == 44
    assert cli.call_records[81]["transfer_attempt"] == 82


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_selector",
        "missing_selector",
        "wrong_remote",
        "occurrence_overflow",
        "missing_final_verification",
        "changed_hash",
        "missing_parent",
        "missing_child",
        "missing_raw_census",
        "missing_timeout_census",
        "stale_verification",
        "earlier_completed_unverified",
        "wrong_parent_direction",
        "wrong_child_direction",
        "wrong_parent_remote",
        "forged_timeout_absence",
        "forged_503_absence",
    ],
)
def test_invalid_observer_evidence_cannot_reach_final_admission(
    tmp_path: Path, final_transfer: dict, defect: str
) -> None:
    cli, injection = _prospective_real_poll_sequence()
    # Three real attempts already use the allowance. Invalid absence evidence
    # consumes the fourth, so even a later verified read cannot launder it.
    _make_real_failure(cli.call_records[0])
    row = cli.call_records[1]
    final = cli.call_records[81]
    parent = next(
        p for p in cli.transfer_records if p["transfer_id"] == final["parent_transfer_id"]
    )
    if defect == "wrong_selector":
        row["realized_selector_id"] = "foreign"
    elif defect == "missing_selector":
        row.pop("realized_selector_id")
    elif defect == "wrong_remote":
        row["remote"] += ".foreign"
    elif defect == "occurrence_overflow":
        for entry in injection["entries"]:
            if entry["injection_id"] == row["realized_selector_id"]:
                entry["max_calls"] = 77
    elif defect == "missing_final_verification":
        final.pop("verified_sha256")
    elif defect == "changed_hash":
        final["verified_sha256"] = "0" * 64
    elif defect == "missing_parent":
        cli.transfer_records.remove(parent)
    elif defect == "missing_child":
        parent["children"] = []
    elif defect == "missing_raw_census":
        cli.call_records.pop(2)
    elif defect == "missing_timeout_census":
        cli.call_records.pop(43)
    elif defect == "stale_verification":
        row.update(status="completed", verified_sha256=final.pop("verified_sha256"))
    elif defect == "earlier_completed_unverified":
        row["status"] = "completed"
    elif defect == "wrong_parent_direction":
        parent["direction"] = "upload"
    elif defect == "wrong_child_direction":
        parent["children"][0]["direction"] = "upload"
    elif defect == "wrong_parent_remote":
        parent["remote"] += ".foreign"
    elif defect == "forged_timeout_absence":
        row.update(failure_class="RPC_EXEC_POLL_TIMEOUT", error="download timed out")
    elif defect == "forged_503_absence":
        row.update(failure_class="HTTP_503", error="HTTP 503 Service Unavailable")
    path = _write_result(tmp_path, cli, injection, final_transfer)
    arm = admission.prerequisite_arms(path, REPO)["l4"]
    assert arm["verdict"] != "PASS", arm
    with pytest.raises(ValueError, match="counted v2 faults|missing v2 evidence"):
        admission.require_prerequisite(path, REPO, "l4")


@pytest.mark.parametrize("defect", ["missing_manifest", "wrong_manifest", "changed_manifest"])
def test_absence_manifest_identity_is_required_at_final_admission(
    tmp_path: Path, final_transfer: dict, defect: str
) -> None:
    cli, injection = _prospective_real_poll_sequence()
    if defect == "wrong_manifest":
        injection["session_job_id"] += "-foreign"
        # Bind the original session to the receipt so manifest/session validation runs.
        receipt_job = injection["session_job_id"].removesuffix("-foreign")
    path = _write_result(tmp_path, cli, injection, final_transfer)
    result = json.loads(path.read_text())
    job = next(j for j in result["final_plan"]["jobs"] if j["name"] == "l4")
    if defect == "missing_manifest":
        job.pop("injection_manifest")
    elif defect == "changed_manifest":
        Path(job["injection_manifest"]).write_text("{}")
    else:
        execution = next(r for r in result["final_plan"]["execution"] if r["job"] == "l4")
        receipt_path = Path(execution["receipt"])
        receipt = json.loads(receipt_path.read_text())
        receipt["job_id"] = receipt_job
        receipt_path.write_text(json.dumps(receipt))
        execution["receipt_sha256"] = admission.sha(receipt_path)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="injection manifest|different session"):
        admission.require_prerequisite(path, REPO, "l4")


@pytest.mark.parametrize("census_state", ["verified", "missing", "changed_hash"])
def test_client_marker_census_is_required_at_final_admission(
    tmp_path: Path, final_transfer: dict, census_state: str
) -> None:
    cli = _CLI()
    remote = "/content/scope/client-start.json"
    evidence = DirectObserverDownloadEvidence(cli, {remote}, verification_kind="census-file-digest")
    cli.call_records.append(_row(remote, "completed", 1))
    evidence.observe_since(0)
    payload = b"verified marker"
    digest = hashlib.sha256(payload).hexdigest()
    census = {"markers": {}, "client_files": {remote: {"sha256": digest, "size": len(payload)}}}
    if census_state == "missing":
        census = {}
    elif census_state == "changed_hash":
        census["client_files"][remote]["sha256"] = "0" * 64
    record_census_verified_download(
        cli,
        first_call=0,
        remote=remote,
        local_size=len(payload),
        local_sha256=digest,
        census=census,
    )
    cli.marker_census = census
    injection = {
        "schema": "l4-label-faults-injection-manifest/v1",
        "written_before_dispatch": True,
        "session_job_id": "census-scope",
        "entries": [],
    }
    path = _write_result(tmp_path, cli, injection, final_transfer)
    arm = admission.prerequisite_arms(path, REPO)["l4"]
    if census_state == "verified":
        assert admission.require_prerequisite(path, REPO, "l4")["verdict"] == "PASS"
    else:
        assert arm["verdict"] != "PASS"
        if census_state == "changed_hash":
            assert arm["counted_by_class"]["C6"] > 0
        else:
            assert arm["evidence_missing"]
        with pytest.raises(ValueError, match="counted v2 faults"):
            admission.require_prerequisite(path, REPO, "l4")
