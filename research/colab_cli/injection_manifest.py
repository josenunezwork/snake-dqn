"""Pre-dispatch injection manifests for ``l4-label-faults/v2`` (create-only).

Authority: JUDGMENT_L4_FAULT_DEFINITION_20261024.md §2.3. A deliberately
induced fault is excluded from the v2 census only by identity listed here,
written before dispatch. Anything not listed is a real fault.

Exec injections are pinned by a request ID generated at plan time; transfer
probes by remote suffix with a bounded call count (the harness's own wait
window divided by its poll cadence, doubled once); the L4 Mac-loss watchdog's
designed backend release by its terminal class and error text. Call indices
are unknowable before dispatch; the realized indices are reported by the
classifier and checked against these bounds.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "l4-label-faults-injection-manifest/v1"
SILENCE_CODE = "import time\ntime.sleep(65)\nresult={'after_silence':True}"


def acceptance_entries(gpu: str) -> list[dict[str, Any]]:
    """Every deliberate fault of the final-only acceptance arm for ``gpu``."""
    if gpu == "cpu":
        return [
            {
                "injection_id": "silence-timeout",
                "type": "deliberate-65s-kernel-silence",
                "command": "exec",
                "request_id": uuid.uuid4().hex,
                "request_sha256": hashlib.sha256(SILENCE_CODE.encode()).hexdigest(),
                "max_calls": 1,
            },
            {
                # _wait_file: 150 s window, <= 2 s sleep per absent read -> 75; x2.
                "injection_id": "client-loss-start-absent",
                "type": "absence-probe",
                "command": "download",
                "remote_suffix": "/client-start.json",
                "max_calls": 150,
            },
            {
                # One deliberate not-found probe + 300 s _wait_file window -> 151; x2.
                "injection_id": "client-loss-finished-absent",
                "type": "absence-probe",
                "command": "download",
                "remote_suffix": "/client-finished.json",
                "max_calls": 302,
            },
            {
                "injection_id": "corrupt-export-rejection",
                "type": "deliberate-hash-mismatch-export",
                "command": "download",
                "remote_suffix": "-fault/output/acceptance-corrupt.bin",
                "max_calls": 1,
            },
        ]
    if gpu == "L4":
        return [
            {
                # Watchdog checkpoint reads: (180 s lease + 15 s grace) / ~1 s -> 195; x2.
                "injection_id": "watchdog-checkpoint-marker-absent",
                "type": "absence-probe",
                "command": "download",
                "remote_suffix": "/output/lease-checkpoint.pkl.sha256",
                "max_calls": 390,
            },
            {
                "injection_id": "watchdog-checkpoint-absent",
                "type": "absence-probe",
                "command": "download",
                "remote_suffix": "/output/lease-checkpoint.pkl",
                "max_calls": 390,
            },
            {
                "injection_id": "watchdog-designed-backend-release",
                "type": "designed-mac-loss-release",
                "command": "terminal",
                "failure_class": "ACCOUNTING",
                "error_contains": "supervised assignment disappeared",
                "max_calls": 1,
            },
        ]
    raise ValueError(f"unknown acceptance arm: {gpu}")


def write(path: Path, job_id: str, entries: list[dict[str, Any]], *, purpose: str) -> str:
    """Create-only manifest + .sha256 + .complete; returns the sha256."""
    value = {
        "schema": SCHEMA,
        "session_job_id": job_id,
        "purpose": purpose,
        "written_before_dispatch": True,
        "entries": entries,
    }
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for target, data in (
        (path, raw),
        (path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode()),
        (path.with_name(path.name + ".complete"), b"complete\n"),
    ):
        descriptor = os.open(target, flags, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    return digest


def load(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("injection manifest differs from its planned sha256")
    value = json.loads(raw)
    if value.get("schema") != SCHEMA or value.get("written_before_dispatch") is not True:
        raise ValueError("foreign injection manifest")
    return value
