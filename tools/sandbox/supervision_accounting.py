"""Future-only child CPU evidence for the sandbox supervisor adapter.

This module deliberately does not patch ``research.colab_cli`` or attach to an
already-running process.  A future sandbox adapter can opt into these helpers
when it launches a compact, declared remote-supervision operation.  The direct
``wait4`` result supplies a final per-child user+system counter even when the
child begins and exits between host sampler observations; it is conservatively
marked descendant-inclusive until a reducer can prove an exclusive allocation.

The journal is create-only and hash linked.  It is an evidence primitive, not a
claim that an arbitrary existing supervisor has been observed or admitted.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.bl_paced_identity.events import canonical_bytes
from research.bl_paced_identity.record import export_json, read_export

POLICY_VERSION = "classified-supervised-v2"
CAPABILITY = "NOT_ADMISSIBLE"
MAX_APPLICATION_PAYLOAD_BYTES = 64 * 1024
COLLECTOR_VERSION = "sandbox-child-collector/v2"
ALLOWED_TEMPLATES = frozenset(
    {"usage", "sessions", "read_state", "renew_lease", "heartbeat_upload"}
)


class AccountingError(RuntimeError):
    """Raised when a future adapter cannot preserve auditable CPU evidence."""


class LaunchAccountingError(AccountingError):
    """A post-spawn journal failure that still exposes the live child to cleanup."""

    def __init__(self, message: str, child: "Wait4Popen", tracked: "TrackedChild") -> None:
        super().__init__(message)
        self.child = child
        self.tracked = tracked


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return canonical_bytes(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AccountingError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise AccountingError(f"{name} must be a finite nonnegative number")
    return result


def _monotonic_ns(value: int | None) -> int:
    if value is None:
        return time.monotonic_ns()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AccountingError("monotonic_ns must be a nonnegative integer")
    return value


def sha256_bytes(value: bytes) -> str:
    """Return the stable digest used in journal bindings without retaining secrets."""
    return hashlib.sha256(value).hexdigest()


def operation_payload(template_id: str, request: bytes, response: bytes = b"") -> dict[str, Any]:
    """Bind compact application payloads while refusing oversized operations.

    Request and response are deliberately accepted as bytes so a caller can
    hash sensitive content before it ever reaches the journal.
    """
    if not isinstance(template_id, str) or not template_id:
        raise AccountingError("operation template_id is required")
    if template_id not in ALLOWED_TEMPLATES:
        raise AccountingError("operation template is not permitted")
    if not isinstance(request, bytes) or not isinstance(response, bytes):
        raise AccountingError("operation payloads must be bytes")
    total = len(request) + len(response)
    if total > MAX_APPLICATION_PAYLOAD_BYTES:
        raise AccountingError("operation application payload exceeds 65536 bytes")
    return {
        "template_id": template_id,
        "request_sha256": sha256_bytes(request),
        "request_bytes": len(request),
        "response_sha256": sha256_bytes(response),
        "response_bytes": len(response),
        "application_bytes": total,
        "application_limit_bytes": MAX_APPLICATION_PAYLOAD_BYTES,
    }


class ImmutableJournal:
    """Create-only, hash-linked JSON records for one prospective attempt.

    A failed journal write raises ``AccountingError`` to the observer.  Callers
    must mark the panel evidence invalid, but must not use this failure to skip
    subprocess timeout/cleanup safety paths.
    """

    def __init__(
        self,
        directory: Path,
        *,
        policy_version: str = POLICY_VERSION,
        context_sha256: str | None = None,
    ) -> None:
        if not policy_version:
            raise AccountingError("policy version is required")
        self.directory = Path(directory)
        self.policy_version = policy_version
        self.context_sha256 = context_sha256
        self.events_dir = self.directory / "events"
        self.events_dir.mkdir(parents=True, exist_ok=False)
        self._sequence = 0
        self._previous_sha256: str | None = None
        self._last_monotonic_ns = -1
        self._sealed = False
        self._failed = False
        self._lock = threading.RLock()

    def append(
        self, kind: str, payload: Mapping[str, Any], *, monotonic_ns: int | None = None
    ) -> dict[str, Any]:
        """Write one immutable record and return its public hash binding."""
        with self._lock:
            if self._sealed:
                raise AccountingError("cannot append after journal seal")
            if self._failed:
                raise AccountingError("cannot append after journal failure")
            try:
                if not isinstance(kind, str) or not kind:
                    raise AccountingError("journal event kind is required")
                if not isinstance(payload, Mapping):
                    raise AccountingError("journal event payload must be an object")
                observed_ns = _monotonic_ns(monotonic_ns)
                if monotonic_ns is None:
                    # A host clock can have a coarser effective resolution than
                    # its integer unit. Preserve strict journal ordering for
                    # concurrent foreground/heartbeat writers without silently
                    # accepting an explicitly supplied stale timestamp.
                    observed_ns = max(observed_ns, self._last_monotonic_ns + 1)
                if observed_ns <= self._last_monotonic_ns:
                    raise AccountingError("journal monotonic_ns must strictly increase")
                sequence = self._sequence + 1
                body: dict[str, Any] = {
                    "policy_version": self.policy_version,
                    "capability": CAPABILITY,
                    "collector_version": COLLECTOR_VERSION,
                    "context_sha256": self.context_sha256,
                    "seq": sequence,
                    "kind": kind,
                    "mono_ns": observed_ns,
                    "previous_sha256": self._previous_sha256,
                    **dict(payload),
                }
                if set(payload) & {
                    "seq",
                    "mono_ns",
                    "kind",
                    "sha256",
                    "previous_sha256",
                    "policy_version",
                    "capability",
                    "collector_version",
                    "context_sha256",
                }:
                    raise AccountingError("reserved journal field")
                digest = sha256_bytes(_canonical_json(body))
                record = {**body, "sha256": digest}
                path = self.events_dir / f"{sequence:08d}-{kind}.json"
            except (TypeError, ValueError) as exc:
                raise AccountingError("invalid journal payload") from exc
            # Validation cannot leave partial evidence. Once writing starts,
            # any failure must poison the journal because a triplet may exist.
            self._failed = True
            try:
                export_json(path, record)
            except BaseException as exc:
                if isinstance(exc, AccountingError):
                    raise
                raise AccountingError("immutable journal write failed") from exc
            self._sequence = sequence
            self._last_monotonic_ns = observed_ns
            self._previous_sha256 = digest
            self._failed = False
            return {"sequence": sequence, "sha256": digest, "path": str(path)}

    def seal(self, *, status_sha256: str | None = None) -> dict[str, Any]:
        """Create a completion record binding the complete event chain."""
        with self._lock:
            if self._sealed:
                raise AccountingError("journal already sealed")
            if self._failed:
                raise AccountingError("cannot seal failed journal")
            result = {
                "policy_version": self.policy_version,
                "capability": CAPABILITY,
                "event_count": self._sequence,
                "collector_version": COLLECTOR_VERSION,
                "context_sha256": self.context_sha256,
                "status_sha256": status_sha256,
                "terminal_event_sha256": self._previous_sha256,
            }
            path = self.directory / "COMPLETE.json"
            try:
                digest = export_json(path, result)
            except BaseException as exc:
                self._failed = True
                raise AccountingError("immutable journal seal failed") from exc
            self._sealed = True
            return {**result, "sha256": digest, "path": str(path)}


class _Timeval(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("microseconds", ctypes.c_long)]


class _Rusage(ctypes.Structure):
    # Darwin and Linux rusage: two timevals followed by fourteen native longs.
    _fields_ = [("user", _Timeval), ("system", _Timeval), ("rest", ctypes.c_long * 14)]


def _native_wait4(pid: int, flags: int) -> tuple[int, int, dict[str, int] | None]:
    """Retain kernel timeval integers instead of rounding Python float seconds."""
    libc = ctypes.CDLL(None, use_errno=True)
    wait4 = libc.wait4
    wait4.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(_Rusage),
    ]
    wait4.restype = ctypes.c_int
    status, usage = ctypes.c_int(), _Rusage()
    while True:
        child = wait4(pid, ctypes.byref(status), flags, ctypes.byref(usage))
        if child >= 0:
            break
        error = ctypes.get_errno()
        if error != errno.EINTR:
            raise OSError(error, os.strerror(error))
    counters = None
    if child:

        def ns(value: _Timeval) -> int:
            if value.seconds < 0 or not 0 <= value.microseconds < 1_000_000:
                raise AccountingError("invalid native final counter")
            return value.seconds * 1_000_000_000 + value.microseconds * 1000

        counters = {"user_cpu_ns": ns(usage.user), "system_cpu_ns": ns(usage.system)}
    return child, status.value, counters


class Wait4Popen(subprocess.Popen[bytes]):
    """Retain exact per-child kernel counters on wait, poll and communicate."""

    final_rusage: Any | None = None
    final_wait_status: int | None = None

    def _wait4(self, wait_flags: int) -> tuple[int, int]:
        try:
            pid, status, usage = _native_wait4(self.pid, wait_flags)
        except ChildProcessError as exc:
            # A different reaper has destroyed both status and final counters.
            # Never feed a synthetic success status to Popen._handle_exitstatus.
            self.returncode = 255
            raise AccountingError("child was already reaped; final CPU is unavailable") from exc
        if pid == self.pid:
            self.final_rusage = usage
            self.final_wait_status = status
        return pid, status

    def _try_wait(self, wait_flags: int) -> tuple[int, int]:
        return self._wait4(wait_flags)

    def _internal_poll(self, _deadstate: int | None = None) -> int | None:
        if self.returncode is None:
            if not self._waitpid_lock.acquire(False):
                return None
            try:
                if self.returncode is None:
                    pid, status = self._wait4(os.WNOHANG)
                    if pid == self.pid:
                        self._handle_exitstatus(status)
            finally:
                self._waitpid_lock.release()
        return self.returncode


@dataclass(frozen=True)
class TrackedChild:
    """A direct child launched by a future adapter under one permitted operation.

    ``birth_ns`` is a local monotonic launch bracket, not an OS process-creation
    identity.  It cannot establish PID-reuse resistance by itself and must be
    paired with a qualified birth-aware sampler before any admission.
    """

    operation_id: str
    pid: int
    uid: int
    argv: tuple[str, ...]
    launch_monotonic_ns: int
    parent_pid: int
    birth_ns: int
    identity: dict[str, Any] = field(default_factory=dict)
    parent_identity: str = ""


def launch_operation(
    journal: ImmutableJournal,
    operation_id: str,
    argv: Sequence[str],
    *,
    payload: Mapping[str, Any],
    parent_identity: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    identity_metadata: Mapping[str, Any] | None = None,
) -> tuple[Wait4Popen, TrackedChild]:
    """Launch and immediately record a direct child for future-only accounting.

    The function never changes signal or timeout semantics.  A calling adapter
    remains responsible for its qualified timeout and cleanup path; if this
    accounting write fails after launch, the live child is returned only through
    the raised exception's surrounding caller and must still be cleaned up.
    """
    if not operation_id:
        raise AccountingError("operation_id is required")
    if not parent_identity:
        raise AccountingError("parent_identity is required")
    normalized = tuple(str(part) for part in argv)
    if not normalized or not normalized[0]:
        raise AccountingError("argv must contain an executable")
    _canonical_json(payload)
    launch_ns = time.monotonic_ns()
    record_operation_start(journal, operation_id, payload, monotonic_ns=launch_ns)
    child = Wait4Popen(normalized, cwd=cwd, env=None if env is None else dict(env))
    import psutil

    tracked = TrackedChild(
        operation_id,
        child.pid,
        os.getuid(),
        normalized,
        launch_ns,
        os.getpid(),
        0,
        {},
        parent_identity,
    )
    try:
        birth_ns = round(psutil.Process(child.pid).create_time() * 1_000_000) * 1000
        identity = {
            **dict(identity_metadata or {}),
            "pid": child.pid,
            "birth_ns": birth_ns,
            "uid": os.getuid(),
            "argv": list(normalized),
            "cwd": str(Path(cwd or Path.cwd()).resolve()),
        }
        tracked = TrackedChild(
            operation_id=operation_id,
            pid=child.pid,
            uid=os.getuid(),
            argv=normalized,
            launch_monotonic_ns=launch_ns,
            parent_pid=os.getpid(),
            birth_ns=birth_ns,
            identity=identity,
            parent_identity=parent_identity,
        )
        journal.append(
            "launch",
            {
                "operation_id": operation_id,
                "identity": identity,
                "parent": parent_identity,
                "launch_start_ns": launch_ns,
                "launch_end_ns": time.monotonic_ns(),
            },
        )
    except BaseException as exc:
        # Do not terminate or otherwise alter the operation here: a qualified
        # adapter's existing timeout/cleanup path owns that safety decision.
        raise LaunchAccountingError(
            "child launched but journal launch record failed", child, tracked
        ) from exc
    return child, tracked


def record_root_identity(
    journal: ImmutableJournal,
    root_id: str,
    identity: Mapping[str, Any],
    *,
    policy_source_hash: str,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Record a registry-verified root identity before any child operation."""
    if not root_id:
        raise AccountingError("root_id is required")
    return journal.append(
        "root_identity",
        {"root_id": root_id, "identity": dict(identity), "policy_source_hash": policy_source_hash},
        monotonic_ns=monotonic_ns,
    )


def record_exec(
    journal: ImmutableJournal,
    tracked: TrackedChild,
    identity: Mapping[str, Any] | None = None,
    *,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Record an exact exec identity; the verifier rejects changed executables."""
    return journal.append(
        "exec",
        {
            "identity": dict(tracked.identity if identity is None else identity),
            "parent": tracked.parent_identity,
            "operation_id": tracked.operation_id,
        },
        monotonic_ns=monotonic_ns,
    )


def record_operation_start(
    journal: ImmutableJournal,
    operation_id: str,
    payload: Mapping[str, Any],
    *,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Start before spawn with the verifier's flat fixed-template metadata."""
    template = payload.get("template", payload.get("template_id"))
    if template not in ALLOWED_TEMPLATES:
        raise AccountingError("operation template is not permitted")
    return journal.append(
        "operation_start",
        {
            **dict(payload),
            "operation_id": operation_id,
            "template": template,
        },
        monotonic_ns=monotonic_ns,
    )


def record_operation_end(
    journal: ImmutableJournal,
    operation_id: str,
    *,
    request: bytes = b"",
    response: bytes = b"",
    response_captured: bool = True,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Complete after every exit with independently captured byte counts/digests."""
    if not isinstance(request, bytes) or not isinstance(response, bytes):
        raise AccountingError("operation captures must be bytes")
    if type(response_captured) is not bool:
        raise AccountingError("response capture completeness must be boolean")
    return journal.append(
        "operation_end",
        {
            "operation_id": operation_id,
            "response_bytes": len(response),
            "response_digest": sha256_bytes(response),
            "request_capture": {
                "bytes": len(request),
                "digest": sha256_bytes(request),
                "independent": True,
            },
            "response_capture": {
                "bytes": len(response),
                "digest": sha256_bytes(response),
                "independent": response_captured,
            },
        },
        monotonic_ns=monotonic_ns,
    )


def wait4_exit(
    journal: ImmutableJournal, child: Wait4Popen, tracked: TrackedChild
) -> dict[str, Any]:
    """Reap exactly one launched child and journal its final CPU counters.

    ``wait4`` is intentionally used instead of ``RUSAGE_CHILDREN``: the latter
    is an aggregate and is unsafe when multiple permitted operations overlap.
    """
    if child.pid != tracked.pid:
        raise AccountingError("tracked child PID does not match Popen")
    child.wait()
    usage = child.final_rusage
    if usage is None:
        raise AccountingError("child was already reaped; final CPU is unavailable")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("user_cpu_ns", "system_cpu_ns")
    ):
        raise AccountingError("final CPU requires exact finite native integer counters")
    user_ns, system_ns = usage["user_cpu_ns"], usage["system_cpu_ns"]
    exit_ns = time.monotonic_ns()
    result = {
        "operation_id": tracked.operation_id,
        "pid": tracked.pid,
        "uid": tracked.uid,
        "parent_pid": tracked.parent_pid,
        "birth_ns": tracked.birth_ns,
        "launch_monotonic_ns": tracked.launch_monotonic_ns,
        "exit_monotonic_ns": exit_ns,
        "wait_status": child.final_wait_status,
        "returncode": child.returncode,
        "user_cpu_ns": user_ns,
        "system_cpu_ns": system_ns,
        "final_cpu_ns": user_ns + system_ns,
        "native_resolution_ns": 1000,
        # A direct wait4 counter may include work of descendants already reaped
        # by this helper.  A reducer must refuse this until launch/exit records
        # establish a non-overlapping exclusive allocation or reconcile it.
        "final_scope": "subtree",
        "descendants_complete": False,
        "final_inclusions": [],
        "accounting_kind": "wait4-direct-child-self",
    }
    journal.append("exit", result, monotonic_ns=exit_ns)
    return result


def record_counter_sample(
    journal: ImmutableJournal,
    *,
    process_id: int,
    counter_seconds: float,
    sample_kind: str,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Record an independently acquired ongoing process CPU counter.

    The monitor owns the process sampler.  This narrow hook records its raw
    counter with a monotonic bracket and refuses nonsensical values; it does not
    convert missing samples into zero CPU.
    """
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        raise AccountingError("invalid sampled process CPU counter")
    observed_counter = _finite_nonnegative(counter_seconds, "sampled process CPU counter")
    if not isinstance(sample_kind, str) or not sample_kind:
        raise AccountingError("sample_kind is required")
    result = {
        "pid": process_id,
        "counter_seconds": observed_counter,
        "sample_kind": sample_kind,
    }
    journal.append("counter_sample", result, monotonic_ns=monotonic_ns)
    return result


def read_journal(directory: Path) -> list[dict[str, Any]]:
    """Verify immutable sidecars, sequence, timestamps, hash chain and final seal."""
    directory = Path(directory)
    seal = read_export(directory / "COMPLETE.json")
    records, previous, last = [], None, -1
    paths = sorted((directory / "events").glob("*.json"))
    if len(paths) != seal["event_count"] or seal["policy_version"] != POLICY_VERSION:
        raise AccountingError("journal count/policy mismatch")
    for index, path in enumerate(paths, 1):
        row = read_export(path)
        body = {key: value for key, value in row.items() if key != "sha256"}
        if (
            row["seq"] != index
            or row["mono_ns"] <= last
            or row["previous_sha256"] != previous
            or row["sha256"] != sha256_bytes(_canonical_json(body))
            or row["policy_version"] != POLICY_VERSION
            or row["collector_version"] != COLLECTOR_VERSION
            or row["context_sha256"] != seal["context_sha256"]
        ):
            raise AccountingError("journal chain/sequence/time mismatch")
        records.append(row)
        previous, last = row["sha256"], row["mono_ns"]
    expected_parts = {
        path.name + suffix for path in paths for suffix in ("", ".sha256", ".complete")
    }
    if {path.name for path in (directory / "events").iterdir()} != expected_parts:
        raise AccountingError("journal contains unexpected or orphan components")
    if previous != seal["terminal_event_sha256"]:
        raise AccountingError("journal terminal hash mismatch")
    return records
