"""GET-only broker reconciliation for receipt-less HOST-LOSS recovery.

The operator runs capture only after loss/release. Admission is entirely offline.
This module never stops a runtime or edits broker pending/active records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from research.colab_cli.supervisor import Broker
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.units import digest, publish

SCHEMA = "sandbox-a-colab-broker-get-only-reconciliation/v1"
QUALIFIED_TRANSPORT_TREE = "5833e54e9c5396121cbbf99a5cd4e52006e564fd"
PRODUCER = "research.colab_cli.supervisor.Broker.status"


def _bound_json(desc: Any, label: str) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(desc, Mapping) or not isinstance(desc.get("path"), str):
        raise r.ResumeRefused(f"missing {label} descriptor")
    path = Path(desc["path"])
    actual = r._descriptor(path, runtime_path=path.name)
    if any(desc.get(k) != actual[k] for k in ("size", "sha256")):
        raise r.ResumeRefused(f"{label} size or SHA-256 differs")
    value = r._json_loads(path.read_text())
    if not isinstance(value, Mapping):
        raise r.ResumeRefused(f"malformed {label}")
    return path, value


def _absent(status: Any, endpoint: str, session: str) -> None:
    if not isinstance(status, Mapping) or not isinstance(status.get("sessions"), Mapping):
        raise r.ResumeRefused("broker reconciliation lacks server sessions")
    sessions = status["sessions"]
    if endpoint in sessions or session in sessions.values():
        raise r.ResumeRefused("target endpoint/session is still present in server sessions")
    if not isinstance(status.get("active"), list) or not isinstance(status.get("unresolved"), list):
        raise r.ResumeRefused("broker reconciliation lacks pending/active evidence")
    if any(
        not isinstance(row, Mapping)
        or row.get("endpoint") == endpoint
        or row.get("session") == session
        for row in status["active"]
    ) or any(
        not isinstance(row, str) or Path(row).name == f"pending-{session}.json"
        for row in status["unresolved"]
    ):
        raise r.ResumeRefused("target endpoint/session has unresolved broker state")


def _descriptor(path: Path) -> dict[str, Any]:
    return {**r._descriptor(path, runtime_path=path.name), "path": str(path.resolve())}


def capture_prefix_release(
    assignment_path: Path,
    output_root: Path,
    job: str,
    generation: str,
    *,
    state_dir: Path,
    colab: str,
    reconciliation_output: Path,
    capture_output: Path,
) -> None:
    """Create-only record using the qualified broker's GET-only status method.

    NOT Broker.recover: that method calls stop even when the endpoint is absent.
    Pending/active state must already have been reconciled by the operator.
    """
    for path in (reconciliation_output, capture_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        if path.resolve().is_relative_to(output_root.resolve()):
            raise r.ResumeRefused("release records must be outside prior output")
    assignment = r._json_loads(assignment_path.read_text())
    session = f"snake-{job}-{generation[:12]}"
    if (
        not isinstance(assignment, Mapping)
        or assignment.get("output") != str(output_root.resolve())
        or assignment.get("generation") != generation
        or assignment.get("session") != session
        or not isinstance(assignment.get("endpoint"), str)
        or not assignment["endpoint"]
    ):
        raise r.ResumeRefused("release assignment identity differs")
    # Native assignment.json has no publication sidecars: bind its full bytes.
    assignment_desc = {
        "path": str(assignment_path.resolve()),
        "size": assignment_path.stat().st_size,
        "sha256": digest(assignment_path),
    }
    status = Broker(state_dir, colab).status()
    _absent(status, assignment["endpoint"], session)
    record = {
        "schema": SCHEMA,
        "producer": PRODUCER,
        "qualified_transport_tree": QUALIFIED_TRANSPORT_TREE,
        "operation": "GET_ONLY_STATUS_RECONCILIATION",
        "captured_unix_seconds": time.time(),
        "job_id": job,
        "assignment_record": assignment_desc,
        "status": status,
    }
    publish(reconciliation_output, lambda dst: dst.write_text(json.dumps(record) + "\n"))
    capture = {
        "schema": r.ACCOUNT_RELEASE_CAPTURE_SCHEMA,
        "captured_unix_seconds": record["captured_unix_seconds"],
        "status": status,
        "broker_reconciliation": _descriptor(reconciliation_output),
    }
    publish(capture_output, lambda dst: dst.write_text(json.dumps(capture) + "\n"))


def verify_prefix_release(
    capture_desc: Any,
    supervisor_log: Path,
    output_root: Path,
    job: str,
    generation: str,
    *,
    now_unix_seconds: float | None,
    max_capture_age_seconds: float | None,
) -> dict[str, Any]:
    """Reopen the qualified reconciliation and assignment; allow unrelated jobs."""
    _, capture = _bound_json(capture_desc, "account_status_capture")
    _, record = _bound_json(capture.get("broker_reconciliation"), "broker_reconciliation")
    if (
        capture.get("schema") != r.ACCOUNT_RELEASE_CAPTURE_SCHEMA
        or record.get("schema") != SCHEMA
        or record.get("producer") != PRODUCER
        or record.get("qualified_transport_tree") != QUALIFIED_TRANSPORT_TREE
        or record.get("operation") != "GET_ONLY_STATUS_RECONCILIATION"
        or record.get("job_id") != job
        or record.get("status") != capture.get("status")
        or record.get("captured_unix_seconds") != capture.get("captured_unix_seconds")
    ):
        raise r.ResumeRefused(
            "release capture differs from qualified GET-only broker reconciliation"
        )
    desc = record.get("assignment_record")
    if not isinstance(desc, Mapping) or not isinstance(desc.get("path"), str):
        raise r.ResumeRefused("broker reconciliation lacks assignment_record")
    assignment_path = Path(desc["path"])
    r._verified_regular_file(assignment_path, desc, label="release assignment")
    assignment = r._json_loads(assignment_path.read_text())
    session = f"snake-{job}-{generation[:12]}"
    if (
        not isinstance(assignment, Mapping)
        or assignment.get("generation") != generation
        or assignment.get("session") != session
        or assignment.get("output") != str(output_root.resolve())
        or not isinstance(assignment.get("endpoint"), str)
        or not assignment["endpoint"]
    ):
        raise r.ResumeRefused("release assignment identity differs")
    _absent(capture.get("status"), assignment["endpoint"], session)
    # Reuse all existing timestamp/freshness/sidecar checks with a validated
    # target-only view. Account-wide rates/counts belong to dispatch, not release.
    # No evidence bytes or status facts are rewritten or published here.
    r._verify_capture_timestamp(
        capture,
        capture_desc,
        supervisor_log,
        now_unix_seconds=now_unix_seconds,
        max_capture_age_seconds=max_capture_age_seconds,
    )
    return dict(capture)
