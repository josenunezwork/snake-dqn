#!/usr/bin/env python3
"""Durable FIFO dispatch queue with receipt readiness and zero-export retries."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

if __package__:
    from . import auto_retry
    from .qualification import OPS, save
else:
    import auto_retry
    from qualification import OPS, save

HERE = Path(__file__).resolve().parent
SCHEMA = "snake-dqn-ready-queue/v1"
# Cover the 600s lock wait, two 1200s host admissions, two 600s allocation
# windows for H100/A100 preference, and bounded qualification overhead.
DISPATCH_TIMEOUT_SECONDS = 4500


@contextlib.contextmanager
def locked(ops: Path) -> Iterator[dict[str, Any]]:
    """Serialize queue changes across enqueuers, workers and competing daemons."""
    ops.mkdir(parents=True, exist_ok=True)
    with (ops / "ready-queue.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = ops / "ready-queue.json"
        queue = auto_retry.read(path) if path.exists() else {"schema": SCHEMA, "jobs": []}
        if queue.get("schema") != SCHEMA or not isinstance(queue.get("jobs"), list):
            raise ValueError("invalid ready queue")
        yield queue
        save(path, queue)


def event(ops: Path, kind: str, **fields: Any) -> None:
    """Append each operational decision as one bounded JSON log line."""
    payload = json.dumps({"at": time.time(), "event": kind, **fields}, sort_keys=True) + "\n"
    fd = os.open(ops / "events.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def dispatch_parts(argv: list[str]) -> tuple[str, Path, Path, Path]:
    """Accept explicit dispatch.sh argv only, never a shell expression."""
    if (
        len(argv) < 3
        or not Path(argv[0]).is_absolute()
        or Path(argv[0]).name != "dispatch.sh"
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", argv[1])
        or any(arg in {"--mac", "--dry-run", ";", "&&", "|"} for arg in argv)
    ):
        raise ValueError("expected absolute dispatch.sh NAME MANIFEST --broker ... --cwd ...")

    def option(name: str) -> Path:
        if argv.count(name) != 1 or argv.index(name) + 1 >= len(argv):
            raise ValueError(f"one explicit {name} required")
        result = Path(argv[argv.index(name) + 1])
        if not result.is_absolute():
            raise ValueError(f"{name} must be absolute")
        return result

    manifest = Path(argv[2])
    if not manifest.is_absolute():
        raise ValueError("manifest must be absolute")
    return argv[1], manifest, option("--broker"), option("--cwd")


def enqueue(
    ops: Path,
    identity: str,
    argv: list[str],
    *,
    after: list[str] = (),
    prepare_command: list[str] = (),
    retry_limit: int | None = None,
    track: bool = False,
) -> None:
    """Reserve one queue identity and one dispatch name without executing anything."""
    name, manifest, _, _ = dispatch_parts(argv)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", identity):
        raise ValueError("invalid queue identity")
    if retry_limit is not None and (type(retry_limit) is not int or not 0 <= retry_limit <= 2):
        raise ValueError("retry limit must be 0, 1 or 2")
    with locked(ops) as queue:
        if any(row["id"] == identity or row["argv"][1] == name for row in queue["jobs"]):
            raise ValueError("queue identity or dispatch name already enqueued")
        if (ops / "dispatch" / name).exists() and not track:
            raise ValueError("dispatch name already used")
        if track and not (ops / "logs" / (name + ".log")).is_file():
            raise ValueError("track requires an existing dispatch log")
        known = {row["id"] for row in queue["jobs"]}
        for condition in after:
            key, _, mode = condition.partition(":")
            if key not in known or mode not in {"pass", "finished"}:
                raise ValueError("dependency must be a previous queue ID:pass or ID:finished")
        row = {
            "id": identity,
            "argv": argv,
            "after": list(after),
            "status": "LAUNCHED" if track else "READY",
            "prepare_command": list(prepare_command),
            "retry_limit": retry_limit,
            "enqueued_at": time.time(),
            "manifest": str(manifest),
        }
        # Prepared fit manifests may not exist until their readiness gate seals them.
        if manifest.exists():
            row["manifest_sha256"] = auto_retry.digest(manifest)
            identity_value = auto_retry.read(manifest)
            row["output_dir"] = identity_value["output_dir"]
            row["job_id"] = identity_value["job_id"]
            if any(
                j.get("output_dir") == row["output_dir"] or j.get("job_id") == row["job_id"]
                for j in queue["jobs"]
            ):
                raise ValueError("manifest output or job identity already enqueued")
        elif not prepare_command:
            raise ValueError("manifest absent without a prepare command")
        queue["jobs"].append(row)
        event(ops, "READY_ENQUEUED", id=identity, command=shlex.join(argv))


def latest(queue: dict[str, Any], identity: str) -> dict[str, Any]:
    """Follow a retry chain so dependent work waits for the actual final attempt."""
    seen = set()
    while identity not in seen:
        seen.add(identity)
        row = next(row for row in queue["jobs"] if row["id"] == identity)
        if not row.get("retry_id"):
            return row
        identity = row["retry_id"]
    raise ValueError("cyclic retry chain")


def retry_backoff(attempt: int) -> float:
    """Back off a refused reconciliation without idling independent queue rows."""
    return min(600.0, 30.0 * 2 ** min(max(0, attempt - 1), 5))


def ready(queue: dict[str, Any], row: dict[str, Any]) -> bool:
    """Evaluate dependencies from bound terminal evidence, never a log PASS string."""
    if time.time() < row.get("not_before", 0):
        return False
    for condition in row["after"]:
        identity, mode = condition.split(":")
        prior = latest(queue, identity)
        if mode == "pass" and prior["status"] != "PASS":
            return False
        if mode == "finished" and prior["status"] not in {"PASS", "FAILED", "UNAVAILABLE"}:
            return False
    return True


def no_assignment_proof(ops: Path, name: str) -> bool:
    """Honor dispatch's exact archived no-assignment proof without changing receipts."""
    path = ops / "dispatch" / name / "unavailable.json"
    if not path.exists():
        return False
    value = auto_retry.read(path)
    receipt = value.get("receipt", {})
    return (
        value.get("schema") == "colab-allocation-unavailable/v1"
        and value.get("mode") == "LIVE"
        and value.get("endpoint_ever_assigned") is False
        and value.get("vm_compute_started") is False
        and value.get("recovery") == {"endpoint": None}
        and auto_retry.digest(Path(receipt["path"])) == receipt.get("sha256")
    )


def observations(ops: Path) -> dict[str, dict[str, Any]]:
    """Count local dispatches using immutable manifest snapshots, logs and receipts.

    An exited shell with an unverified stop still occupies a slot. Preference
    fallback and exact accounting retries have their own snapshots and logs.
    Never infer release from a dead PID, elapsed time or an EXIT line alone.
    """
    result = {}
    assignments = {}
    for path in (Path.home() / ".config/colab-cli/snake").glob("active-*.json"):
        assignment = auto_retry.read(path)
        if assignment.get("output") and assignment.get("endpoint"):
            output_key = str(Path(assignment["output"]).resolve())
            if output_key in assignments:
                raise ValueError("ambiguous active assignment output")
            assignments[output_key] = assignment["endpoint"]
    for manifest in sorted((ops / "manifests").glob("*.json")):
        if manifest.name.endswith(".live.json"):
            continue
        name = manifest.stem
        log = ops / "logs" / f"{name}.log"
        if not log.exists():
            continue  # validation-only reservation, no detached submission yet
        value = auto_retry.read(manifest)
        output = Path(value["output_dir"])
        path = output / "receipt.json"
        receipt = auto_retry.read(path) if path.exists() else None
        if receipt is not None:
            executed = output / "manifest.json"
            if (
                receipt.get("manifest_sha256") != auto_retry.digest(executed)
                or receipt.get("job_id") != value.get("job_id")
                or receipt.get("commit") != value.get("commit")
                or auto_retry.read(executed).get("output_dir") != value["output_dir"]
            ):
                raise ValueError(f"unbound dispatch receipt: {path}")
        exited = bool(re.search(r"^EXIT=\d+$", log.read_text(), re.MULTILINE))
        stopped = receipt is not None and receipt.get("stop_verified") is True
        unavailable = no_assignment_proof(ops, name)
        # A run cannot allocate before it creates its output/manifest.json.
        never_started = exited and not output.exists()
        preallocation_blocked = exited and auto_retry.blocked_before_allocation(value, output, log)
        result[name] = {
            "active": not (
                exited and (stopped or unavailable or never_started or preallocation_blocked)
            ),
            "preallocation_blocked": preallocation_blocked,
            "manifest": str(manifest),
            "log": str(log),
            "receipt": receipt,
            "unavailable": unavailable,
            "exited": exited,
            "endpoint": (receipt.get("endpoint") if receipt else None)
            or assignments.get(str(output.resolve())),
        }
    return result


def current_observation(observed: dict[str, Any], name: str) -> tuple[str, Any]:
    """Follow dispatch's own exact accounting retry and pre-admitted GPU fallback."""
    if __package__:
        from .dispatch import sibling_id
    else:
        from dispatch import sibling_id
    fallback = sibling_id(name)
    for candidate in (
        fallback + "-retry-1",
        fallback,
        name + "-retry-1",
        name,
    ):
        if candidate in observed:
            return candidate, observed[candidate]
    return name, None


def reconcile(ops: Path, queue: dict[str, Any], observed: dict[str, Any]) -> None:
    """Close completed submissions and enqueue eligible completion attempts once."""
    for row in list(queue["jobs"]):
        if row["status"] != "LAUNCHED":
            continue
        if time.time() < row.get("reconcile_retry_at", 0):
            continue
        try:
            name = row["argv"][1]
            # dispatch's built-in accounting retry uses a distinct actual output.
            current_name, state = current_observation(observed, name)
            if state and state["active"] and state["exited"]:
                receipt = state["receipt"]
                if (
                    receipt
                    and receipt.get("failure_class") == "SUPERVISOR"
                    and receipt.get("endpoint") is None
                ):
                    retire_unavailable(ops, row, state)
                    state["unavailable"] = no_assignment_proof(ops, current_name)
                    if state["unavailable"]:
                        state["active"] = False
            if state is None or state["active"]:
                continue
            receipt = state["receipt"]
            row["status"] = (
                "UNAVAILABLE"
                if state["unavailable"]
                else "PASS" if receipt and receipt.get("status") == "PASS" else "FAILED"
            )
            event(ops, "READY_FINISHED", id=row["id"], status=row["status"])
            if row["status"] not in {"FAILED", "UNAVAILABLE"} or row.get("retry_limit") == 0:
                continue
            try:
                original = auto_retry.read(Path(state["manifest"]))
                proof = (
                    ops / "dispatch" / current_name / "unavailable.json"
                    if state["unavailable"]
                    else None
                )
                auto_retry.eligible(
                    original, Path(original["output_dir"]), Path(state["log"]), proof
                )
                attempt, maximum = auto_retry.completion_limit(original, row.get("retry_limit"))
                if attempt >= maximum:
                    raise ValueError("package completion limit exhausted")
                identity = row["id"][:60] + "-retry-" + uuid.uuid4().hex[:10]
                # Commit preparation intent before invoking a package helper. A crash
                # leaves a HELD row, never another automatic submission of the same work.
                row["retry_id"] = identity
                retry = {
                    **row,
                    "id": identity,
                    "status": "HELD",
                    "after": [],
                    "prepare_command": [],
                    "parent": row["id"],
                }
                retry.pop("retry_id", None)
                for field in ("claimed", "token", "dispatch_attempted"):
                    retry.pop(field, None)
                queue["jobs"].append(retry)
                save(ops / "ready-queue.json", queue)
                destination = ops / "auto-retry" / identity
                manifest = auto_retry.prepare(
                    Path(state["manifest"]),
                    Path(state["log"]),
                    destination,
                    dispatch_parts(row["argv"])[2],
                    limit=row.get("retry_limit"),
                    python=os.environ.get("SNAKE_DQN_PYTHON", sys.executable),
                    unallocated_proof=proof,
                )
                argv = list(row["argv"])
                argv[1:3] = [identity, str(manifest)]
                for flag in ("--state-dir", "--dispatch-txt"):
                    if flag in argv:
                        index = argv.index(flag)
                        del argv[index : index + 2]
                argv += ["--state-dir", str(destination / "state")]
                retry.update(
                    argv=argv,
                    manifest=str(manifest),
                    manifest_sha256=auto_retry.digest(manifest),
                    status="READY",
                    enqueued_at=time.time(),
                    not_before=time.time()
                    + (
                        retry_backoff(attempt + 1)
                        if proof or state.get("preallocation_blocked")
                        else 0
                    ),
                    output_dir=auto_retry.read(manifest)["output_dir"],
                    job_id=auto_retry.read(manifest)["job_id"],
                )
                event(
                    ops,
                    "AUTO_RETRY_QUEUED",
                    id=identity,
                    parent=row["id"],
                    completion_attempt=attempt + 1,
                    limit=maximum,
                    command=shlex.join(argv),
                )
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row["retry_refusal"] = str(exc)
                # No eligible retry: dependencies must see the original failure.
                if row.get("retry_id"):
                    failed_retry = next(j for j in queue["jobs"] if j["id"] == row["retry_id"])
                    failed_retry["error"] = str(exc)
                    row.pop("retry_id")
                event(ops, "AUTO_RETRY_REFUSED", id=row["id"], reason=str(exc))
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            attempts = row.get("reconcile_failures", 0) + 1
            row.update(
                reconcile_failures=attempts,
                reconcile_error=str(exc),
                reconcile_retry_at=time.time() + retry_backoff(attempts),
            )
            event(
                ops,
                "READY_RECONCILE_PARKED",
                id=row["id"],
                reason=str(exc),
                retry_at=row["reconcile_retry_at"],
            )


def retire_unavailable(ops: Path, row: dict[str, Any], state: dict[str, Any]) -> None:
    """Reuse exact no-assignment retirement for a standalone accelerator canary."""
    if __package__:
        from . import dispatch
    else:
        import dispatch
    manifest = Path(state["manifest"])
    packet = auto_retry.read(manifest)
    output = Path(packet["output_dir"])
    log = Path(state["log"])
    if not dispatch.allocation_unavailable(output, log, packet["requirements"]["gpu"]):
        return
    _, _, broker, _ = dispatch_parts(row["argv"])
    receipt = state["receipt"]
    pending = Path.home() / ".config/colab-cli/snake" / ("pending-" + receipt["session"] + ".json")
    supervisor_state = Path(auto_retry.read(pending)["config"]).parent.parent
    proof = ops / "dispatch" / manifest.stem / "unavailable.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "reconcile_gpu_pending.py"),
            "--live",
            "--manifest",
            str(manifest),
            "--broker",
            str(broker),
            "--state-dir",
            str(supervisor_state),
            "--log",
            str(log),
            "--proof",
            str(proof),
            "--pending",
            str(pending),
        ],
        cwd=HERE.parents[1],
        check=True,
        timeout=660,
        env={**os.environ, "OPS_DIR": str(ops)},
    )
    event(ops, "READY_ALLOCATION_UNAVAILABLE", id=row["id"], proof=str(proof))


def provider_sessions() -> set[str]:
    """Read all account assignments; unknown inventory always blocks submission."""
    result = subprocess.run(
        [
            os.environ.get("OPS_QUEUE_COLAB", str(Path.home() / ".local/bin/colab")),
            "--auth",
            "oauth2",
            "sessions",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    text = result.stdout.strip()
    if text == "No active sessions found on server.":
        return set()
    rows = text.splitlines()
    endpoints = set()
    for line in rows:
        if not line.strip() or line.startswith("[colab]"):
            continue
        match = re.fullmatch(r"\[.*?\]\s+(\S+)\s+\| Hardware: .+", line.strip())
        if match is None or match[1] in endpoints:
            raise ValueError("unrecognized account sessions inventory")
        endpoints.add(match[1])
    if not endpoints:
        raise ValueError("empty account sessions inventory")
    return endpoints


def occupancy(
    queue: dict[str, Any],
    observed: dict[str, Any],
    inventory: set[str] | None,
) -> tuple[int, int]:
    """Combine dispatch evidence with account-wide inventory and durable claims.

    Old unverified receipts never authorize retry. Fresh backend inventory can
    nevertheless establish that they no longer occupy capacity. Unmanaged and
    legacy notebook assignments count too; the supervisor remains the allocator.
    """
    occupied = {name for name, state in observed.items() if state["active"]}
    reserved = 0
    for row in queue["jobs"]:
        pending = row["status"] in {"DISPATCHING", "LAUNCHED"} or (
            row["status"] == "HELD" and row.get("dispatch_attempted")
        )
        if not pending:
            continue
        current_name, state = current_observation(observed, row["argv"][1])
        counted = (
            current_name in occupied
            if inventory is None
            else state is not None and state.get("endpoint") in inventory
        )
        if not counted:
            reserved += 1
    return (len(occupied) if inventory is None else len(inventory)), reserved


def tick(
    ops: Path,
    *,
    dry_run: bool = False,
    inventory: set[str] | None = None,
) -> dict[str, Any]:
    """Reserve up to three occupied or pending runtime slots across all daemons."""
    if dry_run:
        queue = auto_retry.read(ops / "ready-queue.json")
        observed = observations(ops)
        active, pending = occupancy(queue, observed, inventory)
        return {
            "active": active,
            "reserved": pending,
            "would_dispatch": [
                row["id"] for row in queue["jobs"] if row["status"] == "READY" and ready(queue, row)
            ][: max(0, 3 - active - pending)],
        }
    with locked(ops) as queue:
        observed = observations(ops)
        # A previous manual dispatch can consume a name before the daemon sees
        # it. Adopt only its exact snapshot; never submit a used identity again.
        for row in queue["jobs"]:
            state = observed.get(row["argv"][1])
            if row["status"] == "READY" and state is not None:
                if auto_retry.digest(Path(state["manifest"])) != row.get("manifest_sha256"):
                    row.update(status="HELD", error="existing dispatch manifest differs")
                    event(ops, "READY_HELD", id=row["id"], error=row["error"])
                else:
                    row["status"] = "LAUNCHED"
                    event(ops, "READY_ADOPTED", id=row["id"])
        reconcile(ops, queue, observed)
        # A pending intent fences the entire account. Do not burn ready identities
        # while the exact proof is parked/backing off or foreign intent needs help.
        pending_records = sorted((Path.home() / ".config/colab-cli/snake").glob("pending-*.json"))
        if pending_records:
            return {"dispatched": [], "blocked_pending": [str(p) for p in pending_records]}
        active, reserved = occupancy(queue, observed, inventory)
        slots = max(0, 3 - active - reserved)
        selected = []
        # FIFO among ready entries: an unmet dependency doesn't idle other lanes.
        for row in queue["jobs"]:
            if not slots:
                break
            if row["status"] != "READY" or not ready(queue, row):
                continue
            row.update(status="DISPATCHING", token=uuid.uuid4().hex, claimed=False)
            save(ops / "ready-queue.json", queue)
            event(ops, "READY_DISPATCHING", id=row["id"], command=shlex.join(row["argv"]))
            suffix = (
                "-" + row["token"]
                if row.get("lock_refusals") or row.get("capacity_refusals")
                else ""
            )
            worker_log = ops / ("queue-worker-" + row["id"] + suffix + ".log")
            try:
                with worker_log.open("xb") as stream:
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--ops",
                            str(ops),
                            "worker",
                            row["id"],
                            row["token"],
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={**os.environ, "OPS_DIR": str(ops)},
                    )
            except OSError as exc:
                # No worker accepted this claim. Park the row and keep scanning
                # independent work; never abort the whole tick at a failing head.
                row.update(status="HELD", error=str(exc))
                event(ops, "READY_HELD", id=row["id"], error=str(exc))
                continue
            selected.append(row["id"])
            slots -= 1
        return {"active": active, "reserved": reserved, "dispatched": selected}


def worker(ops: Path, identity: str, token: str) -> None:
    """Submit one durable claim exactly once; ambiguous failures remain held."""
    with locked(ops) as queue:
        row = next(row for row in queue["jobs"] if row["id"] == identity)
        if row["status"] != "DISPATCHING" or row["token"] != token or row.get("claimed"):
            raise ValueError("dispatch claim already consumed or invalid")
        row["claimed"] = True
        argv = list(row["argv"])
        prepare_command = [] if row.get("prepared") else row["prepare_command"]
        expected = row.get("manifest_sha256")
    _, manifest, _, cwd = dispatch_parts(argv)
    no_submission = False
    lock_refused = False
    capacity_refused = False
    try:
        if prepare_command:
            subprocess.run(prepare_command, cwd=cwd, check=True, timeout=600)
            # Sealing is create-only too. A lock retry must reuse the successful
            # preparation and its exact bytes, rather than try to seal it again.
            expected = expected or auto_retry.digest(manifest)
            with locked(ops) as queue:
                next(row for row in queue["jobs"] if row["id"] == identity).update(
                    prepared=True, manifest_sha256=expected
                )
        if expected is not None and auto_retry.digest(manifest) != expected:
            raise ValueError("queued manifest bytes changed")
        with locked(ops) as queue:
            next(row for row in queue["jobs"] if row["id"] == identity)["dispatch_attempted"] = True
        # The dispatcher performs its real, refreshed admission and holds the
        # account allocation lock. No direct Colab mutation exists in this daemon.
        # Retain diagnostics on disk, including the exact pre-reservation lock
        # refusal. Avoid treating unrelated/ambiguous failures as retryable.
        diagnostic = ops / f"queue-dispatch-{identity}-{token}.log"
        with diagnostic.open("xb") as stream:
            result = subprocess.run(
                argv,
                cwd=cwd,
                env={**os.environ, "OPS_DIR": str(ops)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=DISPATCH_TIMEOUT_SECONDS,
            )
        output = diagnostic.read_text(errors="replace")
        print(output, end="", flush=True)
        status = (
            "LAUNCHED"
            if result.returncode == 0 or (ops / "logs" / (argv[1] + ".log")).exists()
            else "HELD"
        )
        # start() creates its log before submitting any detached supervisor.
        # A completed refusal without that log never submitted a runtime.
        no_submission = status == "HELD"
        lock_refused = (
            no_submission
            and output.strip() == f"REFUSE: account allocation lock is held: {ops / 'alloc.lock'}"
            and not (ops / "dispatch" / argv[1]).exists()
            and not (ops / "manifests" / (argv[1] + ".json")).exists()
        )
        capacity_refused = (
            no_submission
            and output.strip() == "REFUSE: account-wide assignment cap 3 (including notebooks)"
            and not (ops / "dispatch" / argv[1]).exists()
            and not (ops / "manifests" / (argv[1] + ".json")).exists()
            and not (ops / "manifests" / (argv[1] + ".live.json")).exists()
            and not Path(auto_retry.read(manifest)["output_dir"]).exists()
        )
        if lock_refused or capacity_refused:
            status = "READY"
        error = None if result.returncode == 0 else f"dispatch exit={result.returncode}"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        status, error = "HELD", str(exc)
    with locked(ops) as queue:
        row = next(row for row in queue["jobs"] if row["id"] == identity)
        row.update(status=status, error=error)
        if no_submission:
            row["dispatch_attempted"] = False
        if lock_refused or capacity_refused:
            counter = "lock_refusals" if lock_refused else "capacity_refusals"
            row[counter] = row.get(counter, 0) + 1
            row["not_before"] = time.time() + retry_backoff(row[counter])
            row["claimed"] = False
        event(
            ops,
            (
                "READY_LOCK_BACKOFF"
                if lock_refused
                else (
                    "READY_CAPACITY_BACKOFF"
                    if capacity_refused
                    else ("READY_SUBMITTED" if status == "LAUNCHED" else "READY_HELD")
                )
            ),
            id=identity,
            error=error,
        )


def main() -> int:
    """Operator interface: enqueue, inspect, dry-run, or run the detached loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, default=Path(os.environ.get("OPS_DIR", OPS)))
    actions = parser.add_subparsers(dest="action", required=True)
    add = actions.add_parser("enqueue")
    add.add_argument("id")
    add.add_argument("--after", action="append", default=[])
    add.add_argument("--prepare-command", default="")
    add.add_argument("--retry-limit", type=int)
    add.add_argument("--track", action="store_true", help="monitor an already dispatched job")
    add.add_argument("--command", required=True, help="one quoted dispatch.sh command line")
    run = actions.add_parser("run")
    run.add_argument("--once", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--interval", type=float, default=15)
    actions.add_parser("status")
    work = actions.add_parser("worker")
    work.add_argument("id")
    work.add_argument("token")
    args = parser.parse_args()
    try:
        if args.action == "enqueue":
            enqueue(
                args.ops,
                args.id,
                shlex.split(args.command),
                after=args.after,
                prepare_command=shlex.split(args.prepare_command),
                retry_limit=args.retry_limit,
                track=args.track,
            )
        elif args.action == "worker":
            worker(args.ops, args.id, args.token)
        elif args.action == "status":
            print(json.dumps(auto_retry.read(args.ops / "ready-queue.json"), indent=2))
        else:
            if not 1 <= args.interval <= 60:
                raise ValueError("interval must be between 1 and 60 seconds")
            while True:
                try:
                    print(
                        json.dumps(
                            tick(args.ops, dry_run=args.dry_run, inventory=provider_sessions())
                        ),
                        flush=True,
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    if args.once or args.dry_run:
                        raise
                    event(args.ops, "READY_PROBE_FAILED", reason=str(exc))
                if args.once or args.dry_run:
                    break
                time.sleep(args.interval)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
