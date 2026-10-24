"""Fail-closed operator evidence for the P1 -05 quiet-host window.

This module deliberately prepares only local, create-only evidence.  It never
starts a supervisor, dispatches Colab, drains a factory, or changes process
priority.  The operator uses the CLI between already-prepared admissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

TIMING_ARMED = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed")
MAC_LEASE = Path("/Users/josenunez/Projects/ml/snake-dqn/exp/mac-lease.json")
SCHEMA = "p1-attempt2-quiet-window/v1"


class WindowError(RuntimeError):
    """A quiet-window assertion could not be established."""


def sha256(path: Path) -> str:
    """Return a file digest without interpreting its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise WindowError(f"JSON evidence is not an object: {path}")
    return value


def publish_create_only(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Publish immutable JSON and its digest sidecar, refusing replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WindowError(f"create-only evidence already exists: {path}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256(encoded).hexdigest()
    sidecar = path.with_name(path.name + ".sha256")
    try:
        sidecar_fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WindowError(f"create-only digest already exists: {sidecar}") from exc
    with os.fdopen(sidecar_fd, "w") as stream:
        stream.write(f"{digest}  {path.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    completion = path.with_name(path.name + ".complete")
    try:
        complete_fd = os.open(completion, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WindowError(f"create-only completion marker already exists: {completion}") from exc
    with os.fdopen(complete_fd, "w") as stream:
        stream.write(digest + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": digest}


def arm(*, arm_path: Path, owner: str, deadline_seconds: float, evidence: Path) -> dict[str, Any]:
    """Create the canonical arm with an unguessable owner token.

    This does not ask the factory to stop; its own timing-aware admission loop
    performs the cooperative drain at completed work boundaries.
    """
    if not owner.strip() or deadline_seconds <= 0:
        raise WindowError("nonempty owner and positive deadline are required")
    token = secrets.token_hex(24)
    payload = {
        "schema": SCHEMA,
        "kind": "arm",
        "owner": owner,
        "owner_token": token,
        "purpose": "P1 -05 exact-input quiet-host maintenance window",
        "deadline_seconds": deadline_seconds,
        "created_epoch": time.time(),
    }
    arm_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(arm_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise WindowError(f"canonical timing arm is already owned: {arm_path}") from exc
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {**payload, "arm_path": str(arm_path.resolve()), "arm_sha256": sha256(arm_path)}
    receipt["evidence"] = publish_create_only(evidence, receipt)
    return receipt


def ps_rows() -> list[dict[str, Any]]:
    """Take one bounded process-table snapshot suitable for evidence."""
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,nice=,stat=,lstart=,command="],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            continue
        try:
            pid, ppid, nice = (int(piece) for piece in fields[:3])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "nice": nice,
                "stat": fields[3],
                "start": " ".join(fields[4:9]),
                "command": fields[9],
            }
        )
    return rows


def descendants(rows: Iterable[dict[str, Any]], root_pid: int) -> list[dict[str, Any]]:
    """Return the root and all live descendants using PID/PPID identity only."""
    remaining = {
        int(row["pid"]): row for row in rows if not str(row.get("stat", "")).startswith("Z")
    }
    found: set[int] = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in remaining.items():
            if int(row["ppid"]) in found and pid not in found:
                found.add(pid)
                changed = True
    return [remaining[pid] for pid in sorted(found) if pid in remaining]


def verify_drain(
    *,
    factory_result: Path,
    factory_root_pid: int,
    lease_path: Path,
    arm_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require a terminal factory result, released lease, and no live factory tree."""
    result = _json(factory_result)
    status = result.get("status")
    if status not in {"COMPLETE", "PREEMPTED", "PAUSED"}:
        raise WindowError(f"factory runtime result is not terminal/drained: {status!r}")
    arm = _json(arm_path)
    if arm.get("schema") != SCHEMA or arm.get("kind") != "arm" or not arm.get("owner_token"):
        raise WindowError("canonical timing arm is not owned by a P1 -05 operation")
    lease = result.get("mac_lease")
    if lease is not None and not isinstance(lease, dict):
        raise WindowError("factory result has invalid Mac-lease context")
    if result.get("cleanup_error") is not None:
        raise WindowError("factory result does not attest successful cleanup")
    if lease_path.exists():
        raise WindowError(f"shared Mac lease remains present: {lease_path}")
    live = descendants(rows, factory_root_pid)
    if live:
        raise WindowError(f"factory process tree remains live: {[row['pid'] for row in live]}")
    active_factory = [
        row
        for row in rows
        if "research.bl_labels.mac_driver" in str(row["command"])
        or "research/bl_labels/mac_driver.py" in str(row["command"])
        or "multiprocessing.spawn" in str(row["command"])
        or "multiprocessing.forkserver" in str(row["command"])
    ]
    if active_factory:
        raise WindowError(
            f"an actual factory mac_driver remains live: {[row['pid'] for row in active_factory]}"
        )
    return {
        "schema": SCHEMA,
        "kind": "factory-drain",
        "factory_result": str(factory_result.resolve()),
        "factory_result_sha256": sha256(factory_result),
        "factory_status": status,
        "arm_path": str(arm_path.resolve()),
        "arm_sha256": sha256(arm_path),
        "lease_absent": True,
        "factory_root_pid": factory_root_pid,
        "live_factory_pids": [],
        "active_factory_mac_driver_pids": [],
        "sample_epoch": time.time(),
        "host_load": list(os.getloadavg()),
    }


def control_sample(*, root_pid: int, rows: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    """Prove one supervisor/CLI tree is entirely at absolute nice 0."""
    if phase not in {"before-dispatch", "during-staging", "export-release"}:
        raise WindowError(f"invalid quiet-window sample phase: {phase}")
    tree = descendants(rows, root_pid)
    if not tree:
        raise WindowError(f"control root PID is absent: {root_pid}")
    nonzero = [row for row in tree if row["nice"] != 0]
    if nonzero:
        raise WindowError(
            f"control tree violates absolute nice 0: {[row['pid'] for row in nonzero]}"
        )
    return {
        "schema": SCHEMA,
        "kind": "control-tree-sample",
        "phase": phase,
        "root_pid": root_pid,
        "members": tree,
        "sample_epoch": time.time(),
        "host_load": list(os.getloadavg()),
    }


def register_control(
    *, arm_path: Path, owner_token: str, root_pid: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind the admitted nice-0 persistent shell to its P1 arm.

    PID reuse is prevented by retaining the BSD ``lstart`` identity and the
    required control marker.  The marker is deliberately checked only at the
    root; descendants are discovered solely from their PPID ancestry.
    """
    arm_value = _json(arm_path)
    if arm_value.get("owner_token") != owner_token:
        raise WindowError("control root is not bound to this operation's arm")
    root = next((row for row in rows if row["pid"] == root_pid), None)
    if root is None:
        raise WindowError(f"control root PID is absent: {root_pid}")
    if root["nice"] != 0:
        raise WindowError(f"control root must have absolute nice 0, got {root['nice']}")
    if "p1_attempt2_control" not in str(root["command"]):
        raise WindowError("control root command lacks required p1_attempt2_control marker")
    if not root.get("start"):
        raise WindowError("control root lacks process-start identity")
    return {
        "schema": SCHEMA,
        "kind": "control-root",
        "root": {key: root[key] for key in ("pid", "ppid", "nice", "start", "command")},
        "arm_path": str(arm_path.resolve()),
        "arm_sha256": sha256(arm_path),
        "registered_epoch": time.time(),
        "phase": "before-dispatch",
        "host_load": list(os.getloadavg()),
    }


def read_registered_control(
    *, root_record: Path, arm_path: Path, rows: list[dict[str, Any]]
) -> int:
    """Revalidate the registered root before every dynamic-tree sample."""
    record = _json(root_record)
    if record.get("schema") != SCHEMA or record.get("kind") != "control-root":
        raise WindowError("invalid control-root record")
    if record.get("arm_path") != str(arm_path.resolve()) or record.get("arm_sha256") != sha256(
        arm_path
    ):
        raise WindowError("control-root record is not bound to current arm bytes")
    root = record.get("root")
    if not isinstance(root, dict):
        raise WindowError("control-root record lacks root identity")
    pid = root.get("pid")
    current = next((row for row in rows if row["pid"] == pid), None)
    if current is None or any(
        current.get(key) != root.get(key) for key in ("ppid", "start", "command")
    ):
        raise WindowError("control root identity changed or disappeared")
    if current["nice"] != 0 or "p1_attempt2_control" not in str(current["command"]):
        raise WindowError("control root no longer has marker and absolute nice 0")
    return int(pid)


def disarm(
    *,
    arm_path: Path,
    owner_token: str,
    release_proof: Path,
    cancellation: bool = False,
    root_records: Iterable[Path] = (),
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Remove only this operation's arm after verified release/cancellation proof."""
    arm_value = _json(arm_path)
    if arm_value.get("owner_token") != owner_token:
        raise WindowError("refusing to remove arm owned by a different operation")
    proof = _json(release_proof)
    actual_job = arm_value.get("owner")
    for record_path in root_records:
        if not record_path.exists():
            continue
        record = _json(record_path)
        if record.get("arm_sha256") != sha256(arm_path):
            raise WindowError("control registration belongs to another owned window")
        if "-allocation-retry-1" in record_path.stem:
            actual_job = arm_value["owner"] + "-retry-1"
        if rows is None or descendants(rows, record["root"]["pid"]):
            raise WindowError("control tree must disappear before disarming")
    valid = (
        proof.get("stop_verified") is True
        and proof.get("cleanup_error") is None
        and isinstance(proof.get("endpoint"), str)
        and bool(proof["endpoint"])
        and proof.get("job_id") == actual_job
        and proof.get("status") in {"PASS", "INCOMPLETE", "FAIL", "VOID", "KILL"}
    )
    if cancellation:
        valid = valid or (
            proof.get("dispatch") == "NOT_DISPATCHED" and proof.get("allocation") == "NONE"
        )
    if not valid:
        raise WindowError(
            "disarm requires verified release or unallocated no-dispatch cancellation"
        )
    digest = sha256(arm_path)
    arm_path.unlink()
    return {
        "schema": SCHEMA,
        "kind": "disarm",
        "arm_sha256": digest,
        "release_proof": str(release_proof.resolve()),
        "release_proof_sha256": sha256(release_proof),
        "cancellation": cancellation,
        "disarmed_epoch": time.time(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    arm_p = sub.add_parser("arm")
    arm_p.add_argument("--owner", required=True)
    arm_p.add_argument("--deadline-seconds", type=float, required=True)
    arm_p.add_argument("--evidence", type=Path, required=True)
    arm_p.add_argument("--arm-path", type=Path, default=TIMING_ARMED)
    drain_p = sub.add_parser("verify-drain")
    drain_p.add_argument("--factory-result", type=Path, required=True)
    drain_p.add_argument("--factory-root-pid", type=int, required=True)
    drain_p.add_argument("--evidence", type=Path, required=True)
    drain_p.add_argument("--arm-path", type=Path, default=TIMING_ARMED)
    root_p = sub.add_parser("register-control")
    root_p.add_argument("--owner-token", required=True)
    root_p.add_argument("--root-pid", type=int, required=True)
    root_p.add_argument("--root-record", type=Path, required=True)
    root_p.add_argument("--evidence", type=Path, required=True)
    root_p.add_argument("--arm-path", type=Path, default=TIMING_ARMED)
    sample_p = sub.add_parser("sample")
    sample_p.add_argument("--root-pid", type=int)
    sample_p.add_argument("--root-record", type=Path)
    sample_p.add_argument("--phase", required=True)
    sample_p.add_argument("--evidence", type=Path, required=True)
    sample_p.add_argument("--arm-path", type=Path, default=TIMING_ARMED)
    disarm_p = sub.add_parser("disarm")
    disarm_p.add_argument("--owner-token", required=True)
    disarm_p.add_argument("--release-proof", type=Path, required=True)
    disarm_p.add_argument("--evidence", type=Path, required=True)
    disarm_p.add_argument("--arm-path", type=Path, default=TIMING_ARMED)
    disarm_p.add_argument("--unallocated-cancellation", action="store_true")
    disarm_p.add_argument("--root-record", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "arm":
        value = arm(
            arm_path=args.arm_path,
            owner=args.owner,
            deadline_seconds=args.deadline_seconds,
            evidence=args.evidence,
        )
    elif args.command == "verify-drain":
        value = verify_drain(
            factory_result=args.factory_result,
            factory_root_pid=args.factory_root_pid,
            lease_path=MAC_LEASE,
            arm_path=args.arm_path,
            rows=ps_rows(),
        )
        value["evidence"] = publish_create_only(args.evidence, value)
    elif args.command == "register-control":
        value = register_control(
            arm_path=args.arm_path,
            owner_token=args.owner_token,
            root_pid=args.root_pid,
            rows=ps_rows(),
        )
        value["root_record"] = publish_create_only(args.root_record, value)
        value["evidence"] = publish_create_only(args.evidence, value)
    elif args.command == "sample":
        rows = ps_rows()
        if (args.root_pid is None) == (args.root_record is None):
            raise WindowError("sample requires exactly one of --root-pid or --root-record")
        root_pid = (
            args.root_pid
            if args.root_record is None
            else read_registered_control(
                root_record=args.root_record, arm_path=args.arm_path, rows=rows
            )
        )
        value = control_sample(root_pid=root_pid, rows=rows, phase=args.phase)
        value["evidence"] = publish_create_only(args.evidence, value)
    else:
        value = disarm(
            arm_path=args.arm_path,
            owner_token=args.owner_token,
            release_proof=args.release_proof,
            cancellation=args.unallocated_cancellation,
            root_records=args.root_record,
            rows=ps_rows(),
        )
        value["evidence"] = publish_create_only(args.evidence, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
