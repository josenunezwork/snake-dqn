"""Outcome-blind ordinary-production process-priority monitor, never ownership.

Only the presence veto is debounced. Native birth ticks identify a process
across exec/reparenting; failed observations latch a stop instead of granting
another grace period. This module never signals or reprioritizes processes.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research.compute.factory_companion_native import _tick_seconds, usage

POLICY_ID = "factory-nice15-persistence30/v1"
RULING_SHA256 = "35295b5b4669bec32a61736c1d5dbb5d9c70015208d6f06db06e5f064bca7c32"
INTERIM_RULING_SHA256 = "ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd"
CADENCE = 5.0
OBSERVATIONS = 7
MIN_SPAN = 30.0
MAX_GAP = 15.0
MIN_NICE = -20
MAX_NICE = 20  # Darwin includes 20; Linux's upper bound of 19 does not apply.
TRUSTED_TMUX = Path("/opt/homebrew/bin/tmux")
_OBSERVATION_DEADLINE: ContextVar[float | None] = ContextVar(
    "factory_observation_deadline", default=None
)


class ProcessMonitorStop(RuntimeError):
    """A latched persistent-priority or process-monitor stop."""


class ObservationFault(ValueError):
    """A sensor error with safe metadata for the implicated process."""

    def __init__(self, reason: str, row: dict[str, Any]) -> None:
        super().__init__(reason)
        self.context = {key: value for key, value in row.items() if key != "argv"}
        self.context["argv_sha256"] = _digest(row.get("argv", ""))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def observation_timeout() -> float:
    """Share one total observation deadline across all bounded OS commands."""
    deadline = _OBSERVATION_DEADLINE.get()
    remaining = 10.0 if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("full process observation deadline exhausted")
    return min(10.0, remaining)


def _native_identity(pid: int) -> dict[str, Any]:
    """Keep raw OS birth precision and verify the executable, not argv text."""
    observation_timeout()
    birth = usage(pid).start_tick
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidpath
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    function.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    observation_timeout()
    if function(pid, buffer, len(buffer)) <= 0:
        raise OSError(ctypes.get_errno(), "proc_pidpath unavailable")
    # An exec changes executable but not birth. Detect reuse during lookup.
    observation_timeout()
    if usage(pid).start_tick != birth or birth <= 0:
        raise RuntimeError("process identity changed during native observation")
    return {
        "birth": str(birth),
        "birth_precision": {"unit": "raw Darwin Mach absolute tick", "seconds": _tick_seconds()},
        "executable": os.fsdecode(buffer.value),
    }


def _confirmed_absent(pid: int) -> bool:
    """One bounded kernel liveness check; permission failure is never absence."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _seed(command: str, executable: str | None = None) -> str | None:
    from .slowdown_resume import _sol_codex_job

    if _sol_codex_job(command):
        return "agent"
    words = command.split(None, 1)
    name = Path(executable or (words[0] if words else "")).name.lower()
    if name.startswith(("python", "pytest")) and re.search(
        r"(?:-m\s+(?:pytest|research\.)|(?:research|src/scripts)/|train_)", command
    ):
        return "research_python_or_pytest"
    return None


def _confirmed_zombie(pid: int, read: Callable[[list[str]], str]) -> bool:
    """Confirm an unreaped exit with OS state, never with spoofable argv text.

    Zombies cannot execute work and may lack native path/rusage metadata.
    A live or unreadable process is not excused by this bounded follow-up.
    """
    try:
        state = read(["/bin/ps", "-p", str(pid), "-o", "pid=,stat="])
    except subprocess.CalledProcessError as error:
        if error.returncode == 1 and _confirmed_absent(pid):
            return True
        raise
    fields = state.split()
    if len(fields) != 2 or fields[0] != str(pid):
        raise ValueError("malformed process state observation")
    return fields[1].startswith("Z")


def _verified_tmux_server(row: dict[str, Any], read: Callable[[list[str]], str]) -> bool:
    """Bind an actual tmux executable to its server PID on its own socket.

    Socket discovery is PID-specific and read-only. This also covers custom
    -L/-S servers, whose retained argv must not seed unrelated pane workloads.
    A real client connected to that socket returns a different server PID.
    """
    try:
        if not os.path.samefile(row["executable"], TRUSTED_TMUX):
            return False
    except OSError:
        return False
    try:
        descriptors = read(["/usr/sbin/lsof", "-a", "-p", str(row["pid"]), "-U", "-Fn"])
    except subprocess.CalledProcessError as error:
        if error.returncode == 1:  # lsof reports no matching descriptors.
            return False
        raise
    sockets = {line[1:] for line in descriptors.splitlines() if line.startswith("n/")}
    if len(sockets) > 8:
        raise ObservationFault("unbounded tmux socket observation", row)
    for socket in sorted(sockets):
        raw = read([str(TRUSTED_TMUX), "-S", socket, "display-message", "-p", "#{pid}"]).strip()
        if not raw.isdigit():
            raise ObservationFault("cannot verify tmux server identity", row)
        if int(raw) == row["pid"]:
            return True
    return False


def process_rows(
    read: Callable[[list[str]], str], known: dict[tuple[int, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    """Read a full table, then native identity for each relevant live process.

    A verified tmux server is excluded before descendant expansion. Every
    pane/workload is still independently classified. No executable-name-only
    supervisor exemption is used.
    """
    # Two -w flags remove the display-width limit, including when stdout is
    # redirected. Only ASCII newlines delimit rows; Unicode separators are args.
    table = read(["/bin/ps", "-A", "-ww", "-o", "pid=,ppid=,nice=,args="])
    rows: dict[int, dict[str, Any]] = {}
    for line in table.removesuffix("\n").split("\n"):
        fields = line.strip().split(None, 3)
        if len(fields) < 3 or not all(x.isascii() and x.isdigit() for x in fields[:2]):
            raise ValueError("malformed full process observation")
        if len(fields) == 3:
            fields.append("")  # A readable process need not expose arguments.
        pid, parent = map(int, fields[:2])
        if pid <= 0 or pid in rows:
            raise ValueError("invalid or duplicate process PID")
        try:
            nice = int(fields[2])
        except ValueError as exc:
            raise ObservationFault(
                "unreadable process nice observation",
                {
                    "pid": pid,
                    "ppid": parent,
                    "argv": fields[3],
                    "nice": None,
                },
            ) from exc
        if not MIN_NICE <= nice <= MAX_NICE:
            raise ObservationFault(
                "invalid process nice observation",
                {
                    "pid": pid,
                    "ppid": parent,
                    "argv": fields[3],
                    "nice": nice,
                },
            )
        rows[pid] = {"pid": pid, "ppid": parent, "nice": nice, "argv": fields[3]}
    if not rows:
        raise ValueError("empty full process observation")
    relevant = {pid for pid, row in rows.items() if _seed(row["argv"])}
    relevant.update(pid for pid, _ in known if pid in rows)
    resolved: set[int] = set()
    excluded: set[int] = set()
    non_roots: set[int] = set()
    while relevant - resolved:
        for pid in sorted(relevant - resolved):
            observation_timeout()
            row = rows[pid]
            resolved.add(pid)
            try:
                metadata = _native_identity(pid)
            except (OSError, RuntimeError) as error:
                try:
                    absent = _confirmed_absent(pid) or _confirmed_zombie(pid, read)
                except (OSError, ValueError, subprocess.SubprocessError):
                    absent = False
                if absent:
                    excluded.add(pid)
                    continue
                raise ObservationFault(
                    "unreadable live process birth or executable", row
                ) from error
            if not metadata.get("birth") or not metadata.get("executable"):
                raise ObservationFault("unreadable required process birth or executable", row)
            row.update(metadata)
            if Path(row["executable"]).name == "tmux":
                if _verified_tmux_server(row, read):
                    excluded.add(pid)
                    non_roots.add(pid)
                    continue
            identity = (pid, row["birth"])
            row["relevance_reason"] = (
                known[identity]["relevance_reason"]
                if identity in known
                else _seed(row["argv"], row["executable"])
            )
            if row["relevance_reason"] is None:
                # A reused PID with generic argv must not inherit relevance.
                if row["ppid"] not in relevant - non_roots:
                    excluded.add(pid)
                    non_roots.add(pid)
                    continue
                row["relevance_reason"] = "relevant_descendant"
        relevant.update(pid for pid, row in rows.items() if row["ppid"] in relevant - non_roots)
    # Keep the full ancestry table for ownership without requiring native
    # reads of unrelated system processes (which may legitimately deny them).
    for pid, row in rows.items():
        row["relevant"] = pid in relevant - excluded
    return list(rows.values())


class PersistenceTracker:
    """One attempt's retained relevance, consecutive streaks and latched stops."""

    def __init__(self, attempt_id: str, telemetry_path: Path | None = None) -> None:
        self.attempt_id = attempt_id
        self.path = telemetry_path
        self.known: dict[tuple[int, str], dict[str, Any]] = {}
        self.pending: dict[tuple[int, str], dict[str, Any]] = {}
        self.history: dict[tuple[int, str], dict[str, Any]] = {}
        self.sequence = 0
        self.first_sample: float | None = None
        self.last_sample: float | None = None
        self.latched: str | None = None
        self.errors: list[str] = []
        self.stops: list[str] = []
        self.records: list[dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x"):
                pass
            self.path.chmod(0o600)

    def _record(self, decision: str, now: float, row: dict[str, Any] | None = None) -> None:
        record = {
            "policy_id": POLICY_ID,
            "ruling_sha256": RULING_SHA256,
            "interim_ruling_sha256": INTERIM_RULING_SHA256,
            "attempt_id": self.attempt_id,
            "utc": datetime.now(timezone.utc).isoformat(),
            "sample_monotonic": now,
            "sample_sequence": self.sequence,
            "sample_age_seconds": None if self.last_sample is None else now - self.last_sample,
            "decision": decision,
            **(row or {}),
        }
        self.records.append(record)
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def fault(self, now: float, reason: str, context: dict[str, Any] | None = None) -> None:
        # Do not put command output or argv/secrets in fault records.
        self.errors.append(reason)
        self.latched = self.latched or "process monitor fault"
        self.stops.append("process monitor fault")
        self._record("monitor_fault", now, {"reason": reason, **(context or {})})
        raise ProcessMonitorStop(self.latched)

    def observe(self, rows: list[dict[str, Any]], now: float, owned_pids: set[int]) -> None:
        if self.latched:
            raise ProcessMonitorStop(self.latched)
        if not math.isfinite(now) or (
            self.last_sample is not None and not 0 <= now - self.last_sample <= MAX_GAP
        ):
            self.fault(now, "stale or nonmonotonic full observation")
        self.sequence += 1
        if self.first_sample is None:
            self.first_sample = now
        # A foreign live identity is never adopted through later ancestry or
        # an ownership-list collision. Relevance and accountability survive
        # reparenting throughout this attempt.
        foreign_pids = {r["pid"] for r in rows if (r["pid"], r.get("birth")) in self.known}
        owned = {os.getpid(), *owned_pids} - foreign_pids
        while True:
            descendants = {
                r["pid"] for r in rows if r["ppid"] in owned and r["pid"] not in foreign_pids
            }
            if descendants <= owned:
                break
            owned.update(descendants)
        current = {}
        try:
            for row in rows:
                if not row.get("relevant") or row["pid"] in owned:
                    continue
                if (
                    not isinstance(row.get("birth"), str)
                    or not row["birth"]
                    or not row.get("birth_precision")
                    or not row.get("executable")
                    or type(row.get("nice")) is not int
                    or not MIN_NICE <= row["nice"] <= MAX_NICE
                ):
                    raise ValueError("unknown relevant metadata")
                identity = row["pid"], row["birth"]
                safe = {key: value for key, value in row.items() if key != "argv"}
                safe.update(ownership_reason="non_owned", argv_sha256=_digest(row.get("argv", "")))
                self.known[identity] = safe
                current[identity] = safe
        except (KeyError, TypeError, ValueError):
            self.fault(now, "unknown relevant metadata", ObservationFault("metadata", row).context)
        for identity, previous in list(self.pending.items()):
            if identity not in current or current[identity]["nice"] >= 15:
                decision = "clear_absent" if identity not in current else "clear_priority"
                cleared = {
                    **previous,
                    **current.get(identity, {}),
                    "elapsed_span": now - previous["first_seen"],
                }
                self._record(decision, now, cleared)
                self.history[identity]["cleared_transients"] += 1
                del self.pending[identity]
        self.known = {key: value for key, value in self.known.items() if key in current}
        for identity, safe in current.items():
            history = self.history.setdefault(
                identity,
                {
                    "pid": identity[0],
                    "birth": identity[1],
                    "first_sighting": now,
                    "last_sighting": now,
                    "maximum_streak": 0,
                    "cleared_transients": 0,
                    "confirmed_trips": 0,
                },
            )
            history["last_sighting"] = now
            if safe["nice"] >= 15:
                if identity not in self.pending:
                    self._record(
                        "clear_priority",
                        now,
                        {**safe, "count": 0, "first_seen": None, "elapsed_span": 0},
                    )
                continue
            prior = self.pending.get(identity)
            streak = {
                **safe,
                "first_seen": now if prior is None else prior["first_seen"],
                "count": 1 if prior is None else prior["count"] + 1,
            }
            streak["elapsed_span"] = now - streak["first_seen"]
            self.pending[identity] = streak
            history["maximum_streak"] = max(history["maximum_streak"], streak["count"])
            trip = streak["count"] >= OBSERVATIONS and streak["elapsed_span"] >= MIN_SPAN
            self._record("preempt_persistent" if trip else "pending", now, streak)
            if trip:
                history["confirmed_trips"] += 1
                self.latched = self.latched or f"persistent concurrent compute pid {identity[0]}"
                self.stops.append(self.latched)
        self._record("observation_complete", now, {"relevant_foreign_count": len(current)})
        self.last_sample = now
        if self.latched:
            raise ProcessMonitorStop(self.latched)

    def sample(self, read: Callable[[list[str]], str], owned_pids: set[int]) -> None:
        if self.latched:
            raise ProcessMonitorStop(self.latched)
        before = time.monotonic()
        if self.last_sample is not None and before - self.last_sample > MAX_GAP:
            self.fault(before, "more than 15 seconds without full observation")
        deadline = before + MAX_GAP if self.last_sample is None else self.last_sample + MAX_GAP
        token = _OBSERVATION_DEADLINE.set(deadline)

        def bounded_read(command: list[str]) -> str:
            observation_timeout()
            result = read(command)
            if time.monotonic() > deadline:
                raise TimeoutError("full process observation deadline exhausted")
            return result

        try:
            rows = process_rows(bounded_read, self.known)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            context = (
                {**error.context, "sensor_reason": str(error)}
                if isinstance(error, ObservationFault)
                else {"sensor_reason": type(error).__name__}
            )
            self.fault(time.monotonic(), "failed or malformed full process observation", context)
        finally:
            _OBSERVATION_DEADLINE.reset(token)
        if time.monotonic() - before > MAX_GAP:
            self.fault(time.monotonic(), "full process observation exceeded 15 seconds")
        self.observe(rows, time.monotonic(), owned_pids)

    def summary(self) -> dict[str, Any]:
        return {
            "policy_id": POLICY_ID,
            "ruling_sha256": RULING_SHA256,
            "interim_ruling_sha256": INTERIM_RULING_SHA256,
            "attempt_id": self.attempt_id,
            "window_start_monotonic": self.first_sample,
            "window_end_monotonic": self.last_sample,
            "observation_count": self.sequence,
            "final_pending": list(self.pending.values()),
            "sightings": list(self.history.values()),
            "observation_errors": self.errors,
            "stop_reasons": self.stops,
            "telemetry_path": str(self.path) if self.path else None,
            "telemetry_sha256": (
                hashlib.sha256(self.path.read_bytes()).hexdigest()
                if self.path
                else _digest(self.records)
            ),
        }


def admission_window(
    tracker: PersistenceTracker,
    sample: Callable[[], None],
    immediate_guard: Callable[[], None],
) -> dict[str, Any]:
    """One bounded seven-sample window; never wait-until-clean or auto-dispatch."""
    for index in range(OBSERVATIONS):
        if index:
            time.sleep(max(0.0, CADENCE - (time.monotonic() - tracker.last_sample)))
        immediate_guard()
        sample()
    # Timer jitter may make the seventh observation slightly early. A later
    # qualifying sample, never an inferred observation, completes the span.
    if tracker.last_sample - tracker.first_sample < MIN_SPAN:
        time.sleep(CADENCE)
        immediate_guard()
        sample()
    if tracker.pending:
        tracker.stops.append("HOLD_PENDING_FOREIGN_PROCESS")
        raise ProcessMonitorStop("HOLD_PENDING_FOREIGN_PROCESS")
    return tracker.summary()
