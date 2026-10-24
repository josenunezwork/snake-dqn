"""Bounded, Mac-wide admission for optional factory companion work.

This module deliberately owns no factory lease.  It only accounts for command
groups it created, under a separate advisory lock, so a factory can shed an
optional companion without ever signalling an unrelated application or tmux
server.  Its on-disk state is operational telemetry, not scientific evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import factory_companion_native
from .factory_companion_argv import classify_argv, native_argv
from .factory_companion_commands import CommandValidationError, validate_command
from .factory_companion_policy import POLICY_SHA256, check_admission, update_factory_state

DEFAULT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/factory-companions")
POLICY_VERSION = "factory-concurrency-20261023/v1"
LIGHT_CPU = 0.25
TOTAL_CPU = 1.25
SAMPLE_CPU = 2.0
TEST_CPU = 90.0
TEST_ELAPSED = 120.0
TEST_RSS = 4 * 1024**3
TOTAL_RSS = 8 * 1024**3
TEST_DUTY = 300.0
TEST_DUTY_WINDOW = 1800.0
WARMUP = 60.0
_START_GATE = (
    "import os,sys; fd=int(sys.argv[1]); os.read(fd,1); "
    "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
)


class AdmissionError(RuntimeError):
    """A companion command was not admitted safely."""


@dataclass(frozen=True)
class ProcessIdentity:
    """A PID plus a kernel-observed start identity prevents PID reuse mistakes."""

    pid: int
    start: str
    pgid: int


@dataclass(frozen=True)
class CompanionAssessment:
    """The small fact set factory guards need; it contains no payload outcomes."""

    healthy: bool
    reason: str | None
    active_test: bool
    light_only_seconds: float
    no_unmanaged_compute: bool
    monitor_heartbeat_age: float | None
    policy_sha256: str | None


def _now() -> float:
    return time.monotonic()


def _epoch() -> float:
    return time.time()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def classify_local_argv(argv_text: str) -> str | None:
    """Recognize actual local compute by executable/argv, never prompt text.

    ChatGPT desktop and computer-use helpers are specifically outside this
    classifier.  A Codex executable is a client requiring registration, not
    proof that its complete process tree may be ignored.
    """
    raw = argv_text.lstrip()
    # Darwin ps does not quote executable paths containing spaces.  This must
    # happen before shlex parsing or a desktop helper becomes fake argv words.
    if (
        raw.startswith("/Applications/ChatGPT.app/")
        or raw.startswith(os.path.expanduser("~/.codex/computer-use/"))
        or raw.startswith(os.path.expanduser("~/Library/Application Support/Codex/"))
    ):
        return None
    try:
        argv = shlex.split(argv_text)
    except ValueError:
        return "unmanaged_compute"
    if not argv:
        return None
    exe = argv[0]
    name = Path(exe).name.lower()
    if name in {"codex", "claude"} or (
        name == "node" and any("codex" in token for token in argv[1:3])
    ):
        return "unregistered_client"
    if name not in {
        "python",
        "python3",
        "pytest",
        "node",
        "npm",
        "make",
        "ruff",
        "black",
        "flake8",
    }:
        return None
    # Tokens are parsed arguments. A remote-client prompt mentioning research
    # cannot reach here because its executable was classified above.
    meaningful = argv[1:]
    if name == "pytest" or "pytest" in meaningful:
        return "unmanaged_compute"
    if "research.compute.factory_companion" in meaningful:
        return None
    if any(
        token.startswith(("research/", "src/scripts/", "research.")) or "train_" in Path(token).name
        for token in meaningful
    ):
        return "unmanaged_compute"
    return None


def unmanaged_local_processes(
    rows: dict[int, dict[str, Any]], registry: dict[str, Any]
) -> list[int]:
    """Return actual unregistered compute PIDs without trusting a client tree."""
    registered = {
        int(item["pid"])
        for item in registry.get("registrations", {}).values()
        if item.get("status") == "running" and isinstance(item.get("pid"), int)
    }
    factory_owned = {
        (int(item["pid"]), str(item["start"]))
        for item in registry.get("factory", {}).get("owned", [])
        if isinstance(item, dict)
        and isinstance(item.get("pid"), int)
        and item.get("start") is not None
    }
    companion_owned = {
        (int(child["pid"]), str(child["start"]))
        for item in registry.get("registrations", {}).values()
        for child in item.get("descendants", [])
        if isinstance(child, dict)
        and isinstance(child.get("pid"), int)
        and child.get("start") is not None
    }
    result = []
    for pid, row in rows.items():
        owned = (
            pid in registered
            or (pid, str(row.get("start"))) in factory_owned
            or (pid, str(row.get("start"))) in companion_owned
        )
        vector = row.get("argv_vector")
        kind = (
            classify_argv(vector)
            if isinstance(vector, tuple)
            else classify_local_argv(str(row.get("argv", "")))
        )
        if row.get("argv_error") and owned:
            result.append(pid)
        elif not owned and kind in {"unmanaged_compute", "unregistered_client"}:
            result.append(pid)
    return sorted(result)


def _unmanaged_kinds(rows: dict[int, dict[str, Any]], registry: dict[str, Any]) -> dict[int, str]:
    """Classify unmanaged PIDs without exposing raw argv in persisted state."""
    known = set(unmanaged_local_processes(rows, registry))
    result: dict[int, str] = {}
    for pid in known:
        row = rows[pid]
        vector = row.get("argv_vector")
        result[pid] = (
            classify_argv(vector)
            if isinstance(vector, tuple)
            else classify_local_argv(str(row.get("argv", "")))
        ) or "unmanaged_compute"
    return result


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class DarwinProcessTable:
    """Native process observations using cumulative CPU time, never ps %%cpu.

    ``ps time`` is the Darwin cumulative user+system CPU counter.  It is read
    alongside lstart and pgid, so the supervisor can reject PID reuse and only
    ever account members of a command group it started.  A wrapped child is
    also reconciled with ``RUSAGE_CHILDREN`` when it exits.
    """

    @staticmethod
    def rows() -> dict[int, dict[str, Any]]:
        command = ["/bin/ps", "-axo", "pid=,ppid=,pgid=,lstart=,time=,rss=,stat=,args="]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=2)
        rows: dict[int, dict[str, Any]] = {}
        for line in result.stdout.splitlines():
            fields = line.split(None, 11)
            if len(fields) != 12 or not all(item.isdigit() for item in fields[:3]):
                continue
            try:
                parts = fields[8].split(":")
                if len(parts) == 2:
                    minutes, seconds = parts
                    cpu = int(minutes) * 60 + float(seconds)
                elif len(parts) == 3:
                    hours, minutes, seconds = parts
                    cpu = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                else:
                    raise ValueError("unexpected ps time")
                rss = int(fields[9]) * 1024
            except (ValueError, IndexError):
                continue
            rows[int(fields[0])] = {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "start": " ".join(fields[3:8]),
                "cpu": cpu,
                "rss": rss,
                "zombie": fields[10].startswith("Z"),
                "argv": fields[11],
            }
        # Replace ps time/RSS/start strings with native cumulative counters
        # when Darwin supplies them.  A failed native observation stays visible
        # to the caller as a later identity/measurement failure; it is never
        # silently converted into zero consumption.
        for pid, row in rows.items():
            try:
                native = factory_companion_native.usage(pid)
            except (ProcessLookupError, RuntimeError):
                row["native_error"] = True
                continue
            row.update(
                {
                    "start": str(native.start_tick),
                    "cpu": native.self_cpu_seconds,
                    "child_cpu": native.child_cpu_seconds,
                    "rss": native.resident_bytes,
                }
            )
            try:
                row["argv_vector"] = native_argv(pid)
            except OSError:
                row["argv_error"] = True
        return rows


class FactoryCompanionSupervisor:
    """Stateful companion admission and group accounting, safe across clients."""

    def __init__(
        self,
        root: Path = DEFAULT_ROOT,
        *,
        clock: Any = _now,
        epoch: Any = _epoch,
        process_table: Any = DarwinProcessTable.rows,
    ) -> None:
        self.root, self.clock, self.epoch, self.process_table = (
            Path(root),
            clock,
            epoch,
            process_table,
        )
        self.state_path = self.root / "registry.json"
        self.lock_path = self.root / "registry.lock"
        self.test_slot_path = self.root / "test-slot.lock"
        self._test_slot: Any | None = None

    def _require_live_root_qualification(self) -> None:
        """Fence direct imports as well as CLI use; temp roots are excluded fixtures."""
        if self.root.resolve() == DEFAULT_ROOT.resolve():
            from research.compute.factory_companion_qualification import require_activation

            require_activation()

    def _empty(self) -> dict[str, Any]:
        now = self.clock()
        return {
            "schema": POLICY_VERSION,
            "registrations": {},
            "duty": [],
            "samples": [],
            "light_only_since": now,
            "observed_since": now,
            "heartbeat": now,
            "history_hold_until": now + TEST_DUTY_WINDOW,
            "suspended_until": 0.0,
        }

    def _read(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._empty()
        try:
            state = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise AdmissionError("invalid companion registry; fail closed") from error
        if not isinstance(state, dict) or state.get("schema") != POLICY_VERSION:
            raise AdmissionError("unknown companion registry schema; fail closed")
        return state

    def _write(self, state: dict[str, Any]) -> None:
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n")
        os.replace(temp, self.state_path)

    def claim_monitor(self) -> None:
        """Bind the persistent monitor's own kernel identity into the registry."""
        self._require_live_root_qualification()
        rows = self.process_table()
        observed = rows.get(os.getpid())
        if observed is None:
            raise AdmissionError("cannot bind companion monitor identity")
        with _locked(self.lock_path):
            state = self._read()
            prior = state.get("monitor")
            if isinstance(prior, dict) and prior.get("pid") != os.getpid():
                prior_row = rows.get(prior.get("pid"))
                if prior_row is not None and prior_row.get("start") == prior.get("start"):
                    raise AdmissionError("another companion monitor is live")
            state["monitor"] = {
                "pid": os.getpid(),
                "start": observed["start"],
                "pgid": observed["pgid"],
            }
            if observed["pgid"] == os.getpid():
                state["registrations"]["__monitor__"] = {
                    "token": "__monitor__",
                    "class": "monitor",
                    "status": "running",
                    "pid": os.getpid(),
                    "pgid": observed["pgid"],
                    "start": observed["start"],
                    "cpu": 0.0,
                    "max_rss": 0,
                    "policy_sha256": POLICY_SHA256,
                    "lease_identity": "monitor",
                    "started_monotonic": self.clock(),
                }
            state["heartbeat"] = self.clock()
            self._write(state)

    def update_factory_state(
        self,
        *,
        identity: str,
        protected: bool,
        streams: dict[str, Any],
        owned: list[dict[str, Any]] | None = None,
    ) -> None:
        """Atomically publish factory control facts and drain tests before protection."""
        self._require_live_root_qualification()
        with _locked(self.lock_path):
            state = self._read()
            update_factory_state(state, identity, protected, streams, self.clock(), owned)
            tokens = []
            if protected:
                tokens = [
                    token
                    for token, record in state["registrations"].items()
                    if record.get("class") == "test" and record.get("status") == "running"
                ]
            self._write(state)
        for token in tokens:
            self.terminate(token)

    def bind_factory_attempt(self, *, identity: str, policy_sha: str) -> None:
        """Rebind surviving registered panels at a lawful factory attempt boundary."""
        self._require_live_root_qualification()
        if policy_sha != POLICY_SHA256 or not identity:
            raise AdmissionError("companion policy or factory identity differs")
        with _locked(self.lock_path):
            state = self._read()
            prior = state.get("factory", {}).get("identity")
            if prior not in {None, identity} and any(
                item.get("class") == "test" and item.get("status") == "running"
                for item in state["registrations"].values()
            ):
                raise AdmissionError("cannot rebind a live factory attempt with active test")
            for item in state["registrations"].values():
                if item.get("status") == "running":
                    item["lease_identity"] = identity
            now = self.clock()
            state["light_only_since"] = now
            state["observed_since"] = now
            update_factory_state(state, identity, True, {}, now)
            self._write(state)

    def _prune(self, state: dict[str, Any], now: float) -> None:
        state["duty"] = [
            item for item in state["duty"] if now - float(item["at"]) <= TEST_DUTY_WINDOW
        ]
        state["samples"] = [item for item in state["samples"] if now - float(item["at"]) <= WARMUP]

    @staticmethod
    def _threads_env() -> dict[str, str]:
        return {
            "SNAKE_DQN_DEVICE": "cpu",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
        }

    def register_client(
        self,
        *,
        policy_sha: str,
        lease_identity: str,
        worktree: str,
        executable: str,
        argv: Sequence[str],
    ) -> str:
        """Register an idle remote-client root; registration grants no test slot."""
        self._require_live_root_qualification()
        if not policy_sha or not lease_identity or not argv:
            raise AdmissionError("policy, lease identity, and argv are required")
        token = uuid.uuid4().hex
        with _locked(self.lock_path):
            state = self._read()
            now = self.clock()
            state["registrations"][token] = {
                "token": token,
                "class": "light",
                "policy_sha256": policy_sha,
                "lease_identity": lease_identity,
                "worktree": worktree,
                "executable": executable,
                "argv_sha256": _json_hash(list(argv)),
                "pid": os.getpid(),
                "started_monotonic": now,
                "started_epoch": self.epoch(),
                "status": "idle",
            }
            self._write(state)
            os.setpriority(os.PRIO_PROCESS, os.getpid(), 15)
            if os.getpriority(os.PRIO_PROCESS, os.getpid()) != 15:
                raise AdmissionError("unable to verify client nice level")
        return token

    @contextmanager
    def control_helper(
        self, *, policy_sha: str, lease_identity: str, argv: Sequence[str]
    ) -> Iterator[None]:
        """Register only this helper PID for a bounded status/host-check operation."""
        self._require_live_root_qualification()
        if policy_sha != POLICY_SHA256 or not lease_identity:
            raise AdmissionError("companion policy or factory identity differs")
        rows = self.process_table()
        observed = rows.get(os.getpid())
        if observed is None:
            raise AdmissionError("cannot bind control helper kernel identity")
        os.setpriority(os.PRIO_PROCESS, os.getpid(), 15)
        if os.getpriority(os.PRIO_PROCESS, os.getpid()) != 15:
            raise AdmissionError("cannot verify control helper nice level")
        token = f"helper-{uuid.uuid4().hex}"
        with _locked(self.lock_path):
            state = self._read()
            state["registrations"][token] = {
                "token": token,
                "class": "light",
                "status": "running",
                "pid": os.getpid(),
                "pgid": observed["pgid"],
                "start": observed["start"],
                "cpu": 0.0,
                "max_rss": 0,
                "policy_sha256": policy_sha,
                "lease_identity": lease_identity,
                "argv_sha256": hashlib.sha256("\0".join(argv).encode()).hexdigest(),
                "started_monotonic": self.clock(),
            }
            self._write(state)
        try:
            yield
        finally:
            with _locked(self.lock_path):
                state = self._read()
                record = state["registrations"].get(token)
                if record is not None:
                    record["status"] = "completed"
                    record["completed_monotonic"] = self.clock()
                    self._write(state)

    def _admit(
        self, state: dict[str, Any], klass: str, policy_sha: str, lease_identity: str
    ) -> None:
        self._require_live_root_qualification()
        now = self.clock()
        self._prune(state, now)
        try:
            check_admission(state, klass, now, policy_sha, lease_identity)
        except ValueError as error:
            raise AdmissionError(str(error)) from error
        if klass == "test" and state.get("client_grace"):
            raise AdmissionError("test admission is closed during unregistered-client grace")
        if klass == "test":
            self.root.mkdir(parents=True, exist_ok=True)
            slot = self.test_slot_path.open("a+")
            try:
                fcntl.flock(slot.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                slot.close()
                raise AdmissionError("one global factory test slot is already admitted") from error
            self._test_slot = slot
            if any(
                item.get("class") == "test" and item.get("status") == "running"
                for item in state["registrations"].values()
            ):
                self._release_test_slot()
                raise AdmissionError("one global factory test slot is already admitted")
        # A registry may not mix factory attempts or policies.
        for item in state["registrations"].values():
            if (
                item.get("class") != "monitor"
                and item.get("status") == "running"
                and (
                    item.get("policy_sha256") != policy_sha
                    or item.get("lease_identity") != lease_identity
                )
            ):
                raise AdmissionError("live companion belongs to another policy or factory attempt")

    def _release_test_slot(self) -> None:
        if self._test_slot is not None:
            fcntl.flock(self._test_slot.fileno(), fcntl.LOCK_UN)
            self._test_slot.close()
            self._test_slot = None

    def _finalize_wait4(self, token: str, waited: int, status: int, waited_usage: Any) -> int:
        """Reconcile the owned root exactly once after wait4 reaps it."""
        if waited <= 0:
            raise AdmissionError("wait4 did not reap owned companion root")
        wait_cpu = float(waited_usage.ru_utime + waited_usage.ru_stime)
        with _locked(self.lock_path):
            state = self._read()
            record = state["registrations"].get(token)
            if record is None:
                raise AdmissionError("owned companion registration disappeared")
            observed = float(record.get("cpu", 0.0))
            # wait4's inclusive result has microsecond precision.  Native
            # rusage has already charged observed tree CPU, so only a positive
            # conservative late correction enters the current bucket.
            charged = max(observed, wait_cpu)
            late = charged - observed
            record.update({"cpu": charged, "late_cpu": late, "wait4_reaped": True})
            if late:
                state["samples"].append(
                    {
                        "at": self.clock(),
                        "total_cpu": late / 5.0,
                        "light_cpu": (
                            late / 5.0 if record["class"] in {"light", "client", "monitor"} else 0.0
                        ),
                        "total_seconds": late,
                        "light_seconds": (
                            late if record["class"] in {"light", "client", "monitor"} else 0.0
                        ),
                        "rss": 0,
                    }
                )
            if record["class"] == "test":
                state["duty"].append({"at": self.clock(), "cpu": charged})
            remaining = [
                row
                for row in self.process_table().values()
                if row.get("pgid") == record.get("pgid")
            ]
            record["descendants_gone"] = not remaining
            if remaining:
                record["status"] = "failed_cleanup"
                state["observation_failure"] = True
            elif record.get("status") == "resource_limit":
                pass
            elif record["class"] == "test":
                if charged > TEST_CPU:
                    record["status"] = "resource_limit"
                else:
                    record["status"] = "completed"
            else:
                record["status"] = "completed"
            self._write(state)
        if record.get("descendants_gone"):
            self._release_test_slot()
        else:
            return 125
        return (
            124 if record.get("status") == "resource_limit" else os.waitstatus_to_exitcode(status)
        )

    def launch(
        self,
        klass: str,
        *,
        policy_sha: str,
        lease_identity: str,
        worktree: str,
        argv: Sequence[str],
        approved: dict[str, str] | None = None,
    ) -> int:
        """Atomically register and run an argv command in an owned process group."""
        self._require_live_root_qualification()
        if (
            klass not in {"light", "test", "client"}
            or not argv
            or any(not isinstance(v, str) for v in argv)
        ):
            raise AdmissionError("class and a nonempty argv vector are required")
        approved_hash = hashlib.sha256("\0".join(argv).encode()).hexdigest()
        if not isinstance(approved, dict) or approved.get("argv_sha256") != approved_hash:
            raise AdmissionError("companion command lacks validator-approved argv metadata")
        token = uuid.uuid4().hex
        with _locked(self.lock_path):
            state = self._read()
            self._admit(state, klass, policy_sha, lease_identity)
            now = self.clock()
            record = {
                "token": token,
                "class": klass,
                "policy_sha256": policy_sha,
                "lease_identity": lease_identity,
                "worktree": worktree,
                "executable": argv[0],
                "argv_sha256": _json_hash(list(argv)),
                "command_metadata": approved or {},
                "status": "starting",
                "started_monotonic": now,
                "started_epoch": self.epoch(),
                "cpu": 0.0,
                "max_rss": 0,
                "priority_verified": False,
            }
            state["registrations"][token] = record
            self._write(state)  # Start barrier: durable registration precedes spawn.
            env = os.environ.copy()
            env.update(self._threads_env())
            # The command itself cannot run until identity and absolute nice
            # are verified.  A tiny exec gate is a process-group member and
            # becomes the requested argv only after the parent releases it.
            read_fd, write_fd = os.pipe()
            child = subprocess.Popen(
                [sys.executable, "-c", _START_GATE, str(read_fd), *argv],
                start_new_session=True,
                env=env,
                pass_fds=(read_fd,),
            )
            os.close(read_fd)
            try:
                os.setpriority(os.PRIO_PROCESS, child.pid, 19 if klass == "test" else 15)
                actual = os.getpriority(os.PRIO_PROCESS, child.pid)
                if actual != (19 if klass == "test" else 15):
                    raise AdmissionError("unable to verify companion nice level")
            except Exception:
                child.terminate()
                child.wait(timeout=2)
                os.close(write_fd)
                del state["registrations"][token]
                self._write(state)
                raise
            rows = self.process_table()
            observed = rows.get(child.pid)
            if observed is None:
                child.terminate()
                child.wait(timeout=2)
                os.close(write_fd)
                del state["registrations"][token]
                self._write(state)
                raise AdmissionError("cannot bind child kernel process identity")
            record.update(
                {
                    "pid": child.pid,
                    "pgid": child.pid,
                    "start": observed["start"],
                    "status": "running",
                    "priority_verified": True,
                }
            )
            self._write(state)
            os.write(write_fd, b"1")
            os.close(write_fd)
        deadline = None if klass == "client" else now + (116.0 if klass == "test" else 30.0)
        while True:
            waited, status, waited_usage = os.wait4(child.pid, os.WNOHANG)
            if waited == child.pid:
                return self._finalize_wait4(token, waited, status, waited_usage)
            if deadline is not None and self.clock() >= deadline:
                self.terminate(token)
                waited, status, waited_usage = os.wait4(child.pid, 0)
                return self._finalize_wait4(token, waited, status, waited_usage)
            # This wrapper loop is independent from the driver's pmset/ioreg
            # probes, which may block its five-second outer guard.
            self.sample()
            time.sleep(1)

    def _group_usage(
        self, record: dict[str, Any], rows: dict[int, dict[str, Any]]
    ) -> tuple[float, int, bool]:
        members = [row for row in rows.values() if row["pgid"] == record.get("pgid")]
        leader = rows.get(int(record.get("pid", -1)))
        # A vanished leader with no final wrapper reconciliation is deliberately
        # not treated as zero CPU: a short child could otherwise disappear
        # between five-second samples.  The owner must finalise it or the
        # factory gets a conservative measurement failure.
        if leader is None or leader["start"] != record.get("start"):
            return 0.0, 0, False
        if any(row.get("native_error") for row in members):
            return 0.0, 0, False
        return (
            # Native child counters transfer CPU of reaped short-lived children
            # to their parent.  Summing self+child over the stable owned tree
            # therefore keeps consumption continuous across a reaping event.
            sum(float(row["cpu"]) + float(row.get("child_cpu", 0.0)) for row in members),
            sum(int(row["rss"]) for row in members if not row["zombie"]),
            True,
        )

    def sample(self, *, monitor: bool = False) -> list[str]:
        """Poll all owned groups once.  Return evicted tokens; never signal foreign PIDs."""
        self._require_live_root_qualification()
        with _locked(self.lock_path):
            state = self._read()
            now, rows, evicted = self.clock(), self.process_table(), []
            self._prune(state, now)
            total_cpu, total_rss, light_cpu, total_seconds, light_seconds = 0.0, 0, 0.0, 0.0, 0.0
            for token, record in state["registrations"].items():
                if record.get("status") != "running":
                    continue
                cpu, rss, identity_ok = self._group_usage(record, rows)
                if not identity_ok:
                    record["status"] = "failed_measurement"
                    evicted.append(token)
                    continue
                prior = float(record.get("cpu", cpu))
                delta = max(0.0, cpu - prior)
                interval = max(0.001, now - float(record.get("last", now - 5.0)))
                record.update(
                    {"cpu": cpu, "max_rss": max(int(record.get("max_rss", 0)), rss), "last": now}
                )
                record["descendants"] = [
                    {"pid": row["pid"], "start": row["start"], "pgid": row["pgid"]}
                    for row in rows.values()
                    if row.get("pgid") == record.get("pgid")
                ]
                total_cpu += delta / interval
                total_seconds += delta
                total_rss += rss
                if record["class"] in {"light", "client", "monitor"}:
                    light_cpu += delta / interval
                    light_seconds += delta
                test_bad = record["class"] == "test" and (
                    cpu > 85 or now - record["started_monotonic"] > 116 or rss > TEST_RSS
                )
                if test_bad:
                    record["status"] = "resource_limit"
                    evicted.append(token)
            bucket_at = math.floor(now / 5.0) * 5.0
            if state["samples"] and state["samples"][-1].get("at") == bucket_at:
                bucket = state["samples"][-1]
                bucket["total_seconds"] += total_seconds
                bucket["light_seconds"] += light_seconds
                bucket["total_cpu"] = max(float(bucket["total_cpu"]), total_cpu)
                bucket["light_cpu"] = max(float(bucket["light_cpu"]), light_cpu)
                bucket["rss"] = max(int(bucket["rss"]), total_rss)
            else:
                state["samples"].append(
                    {
                        "at": bucket_at,
                        "total_cpu": total_cpu,
                        "light_cpu": light_cpu,
                        "total_seconds": total_seconds,
                        "light_seconds": light_seconds,
                        "rss": total_rss,
                    }
                )
            sample_bad = total_cpu > SAMPLE_CPU or total_rss > TOTAL_RSS
            sixty_total = sum(float(item.get("total_seconds", 0.0)) for item in state["samples"])
            sixty_light = sum(float(item.get("light_seconds", 0.0)) for item in state["samples"])
            if sample_bad or sixty_total > TOTAL_CPU * WARMUP or sixty_light > LIGHT_CPU * WARMUP:
                evicted.extend(
                    token
                    for token, item in state["registrations"].items()
                    if item.get("status") == "running" and item.get("class") in {"test", "light"}
                )
                state["suspended_until"] = now + 900
                state["light_only_since"] = now
                state["observed_since"] = now
            if now - float(state.get("heartbeat", now)) > 10:
                state["observed_since"] = now
            if any(
                item.get("class") == "test" and item.get("status") == "running"
                for item in state["registrations"].values()
            ):
                state["light_only_since"] = now
            if monitor:
                state["heartbeat"] = now
            self._write(state)
        for token in sorted(set(evicted)):
            self.terminate(token)
        return sorted(set(evicted))

    def terminate(self, token: str) -> None:
        """TERM/KILL only a process group still proven to match its saved identity."""
        with _locked(self.lock_path):
            state = self._read()
            record = state.get("registrations", {}).get(token)
            if not record or not record.get("pid"):
                return
            rows = self.process_table()
            leader = rows.get(int(record["pid"]))
            if leader is not None and leader.get("start") != record.get("start"):
                record["status"] = "failed_cleanup"
                self._write(state)
                return
            pgid = int(record["pgid"])
        for sig, seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            deadline = self.clock() + seconds
            while self.clock() < deadline:
                if not any(
                    row["pgid"] == pgid and not row["zombie"]
                    for row in self.process_table().values()
                ):
                    break
                time.sleep(0.05)
        with _locked(self.lock_path):
            state = self._read()
            record = state["registrations"].get(token)
            if record:
                cpu, _rss, ok = self._group_usage(record, self.process_table())
                if not ok:
                    record["status"] = "failed_cleanup"
                elif record.get("status") != "resource_limit":
                    record["status"] = "stopped"
                self._write(state)

    def assessment(self, *, policy_sha: str | None = None) -> CompanionAssessment:
        with _locked(self.lock_path):
            exists, state, now = self.state_path.exists(), self._read(), self.clock()
            heartbeat = state.get("heartbeat")
            age = None if heartbeat is None else now - float(heartbeat)
            live = [r for r in state["registrations"].values() if r.get("status") == "running"]
            active_test = any(item.get("class") == "test" for item in live)
            bad = any(
                item.get("status") in {"failed_measurement", "failed_cleanup"}
                for item in state["registrations"].values()
            )
            try:
                rows = self.process_table()
                kinds = _unmanaged_kinds(rows, state)
            except Exception:
                kinds = {-1: "unmanaged_compute"}  # enforcing observation
            grace = state.setdefault("client_grace", {})
            unmanaged = []
            for pid, kind in kinds.items():
                if kind != "unregistered_client":
                    unmanaged.append(pid)
                    continue
                row = rows.get(pid, {}) if pid >= 0 else {}
                identity = f"{pid}:{row.get('start', '?')}"
                since = float(grace.setdefault(identity, now))
                if now - since >= 60:
                    unmanaged.append(pid)
            for identity in list(grace):
                if not any(identity.startswith(f"{pid}:") for pid in kinds):
                    del grace[identity]
            if unmanaged:
                state.setdefault("unmanaged_since", now)
            else:
                state.pop("unmanaged_since", None)
            self._write(state)
            policy = next((item.get("policy_sha256") for item in live), policy_sha)
            monitor = state.get("monitor")
            monitor_live = False
            if isinstance(monitor, dict) and isinstance(monitor.get("pid"), int):
                observed = self.process_table().get(monitor["pid"])
                monitor_live = observed is not None and observed.get("start") == monitor.get(
                    "start"
                )
            good = (
                exists and monitor_live and not bad and not unmanaged and (age is None or age <= 10)
            )
            return CompanionAssessment(
                good,
                (
                    "unregistered_client_grace"
                    if kinds and not unmanaged
                    else (
                        "unmanaged_local_compute"
                        if unmanaged
                        else "companion_monitor_unavailable" if not good else None
                    )
                ),
                active_test,
                max(0.0, now - float(state.get("light_only_since", now))),
                not bad and not unmanaged,
                age,
                policy,
            )

    def finalize_receipt(self, *, attempt_identity: str, policy_sha: str) -> dict[str, Any]:
        """Create one immutable, digest-bound operations receipt for an attempt."""
        if not attempt_identity or not policy_sha:
            raise AdmissionError("attempt identity and policy SHA are required")
        with _locked(self.lock_path):
            state = self._read()
            receipt = {
                "schema": "factory-companion-receipt/v1",
                "attempt_identity": attempt_identity,
                "policy_sha256": policy_sha,
                "created_epoch": self.epoch(),
                "registry_sha256": _json_hash(state),
                "registry": state,
            }
            payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
            digest = hashlib.sha256(payload).hexdigest()
            path = self.root / "receipts" / f"{attempt_identity}-{digest}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                raise AdmissionError("immutable companion receipt already exists") from error
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            for suffix, data in (
                (".sha256", f"{digest}  {path.name}\n"),
                (".complete", "complete\n"),
            ):
                with path.with_name(path.name + suffix).open("x") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            return {"path": str(path), "sha256": digest}

    def revoke_tests(self, reason: str = "factory_early_warning") -> list[str]:
        """Shed only optional tests after a 1.15x factory warning.

        The 15-minute suspension is intentionally recorded without modifying
        factory slow-down state or baselines.
        """
        with _locked(self.lock_path):
            state = self._read()
            tokens = [
                token
                for token, item in state["registrations"].items()
                if item.get("class") == "test" and item.get("status") == "running"
            ]
            state["suspended_until"] = max(
                float(state.get("suspended_until", 0.0)), self.clock() + 900
            )
            state["warning"] = {"revoked_at": self.clock(), "reason": reason}
            for token in tokens:
                state["registrations"][token]["eviction_reason"] = reason
            self._write(state)
        for token in tokens:
            self.terminate(token)
        return tokens

    def prepare_exclusive(self, *, attempt_identity: str, policy_sha: str) -> dict[str, Any]:
        """Close admission and shed locally owned tools before an exclusive arm."""
        if policy_sha != POLICY_SHA256 or not attempt_identity:
            raise AdmissionError("companion policy or factory identity differs")
        with _locked(self.lock_path):
            state = self._read()
            state["admission_closed"] = True
            state["exclusive_arm"] = {"attempt_identity": attempt_identity, "at": self.clock()}
            tokens = [
                token
                for token, item in state["registrations"].items()
                if item.get("status") == "running"
                and item.get("class") in {"test", "light", "client"}
            ]
            self._write(state)
        for token in tokens:
            self.terminate(token)
        return {"phase": "prepared", "tokens": tokens, "at": self.clock()}

    def quiesce_for_exclusive(self, *, attempt_identity: str, policy_sha: str) -> dict[str, Any]:
        """Prove owned local companions are gone after factory terminal cleanup."""
        prepared = self.prepare_exclusive(attempt_identity=attempt_identity, policy_sha=policy_sha)
        state = self._read()
        monitor = state.get("monitor")
        current = (
            self.process_table().get(monitor.get("pid")) if isinstance(monitor, dict) else None
        )
        if (
            isinstance(monitor, dict)
            and current is not None
            and current.get("start") == monitor.get("start")
            and current.get("pgid") == monitor.get("pid")
        ):
            try:
                os.kill(int(monitor["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = self.clock() + 2
            while self.clock() < deadline:
                current = self.process_table().get(monitor["pid"])
                if current is None or current.get("start") != monitor.get("start"):
                    break
                time.sleep(0.05)
        with _locked(self.lock_path):
            state = self._read()
            rows = self.process_table()
            live = []
            for token, item in state["registrations"].items():
                if item.get("status") != "running":
                    continue
                row = rows.get(item.get("pid"))
                if row is not None and row.get("start") == item.get("start"):
                    live.append(token)
            if live:
                state["observation_failure"] = True
                self._write(state)
                raise AdmissionError("owned companion remains live for exclusive window")
            monitor = state.get("monitor")
            if isinstance(monitor, dict) and rows.get(monitor.get("pid"), {}).get(
                "start"
            ) == monitor.get("start"):
                state["observation_failure"] = True
                self._write(state)
                raise AdmissionError("companion monitor remains live for exclusive window")
            receipt = {
                "schema": "factory-companion-exclusive-quiescence/v1",
                "attempt_identity": attempt_identity,
                "policy_sha256": policy_sha,
                "prepared": prepared,
                "quiescent_monotonic": self.clock(),
                "owned_live_tokens": [],
                "registry_sha256": _json_hash(state),
            }
            receipt["sha256"] = _json_hash(receipt)
            self._write(state)
            return receipt


def bind_factory_attempt(*, identity: str, policy_sha: str, root: Path = DEFAULT_ROOT) -> None:
    """Public factory boundary hook; retains duty history but restarts warmup."""
    FactoryCompanionSupervisor(root).bind_factory_attempt(identity=identity, policy_sha=policy_sha)


def control_helper(
    *, policy_sha: str, lease_identity: str, argv: Sequence[str], root: Path = DEFAULT_ROOT
) -> Iterator[None]:
    """Public scoped registration for an exact local control helper invocation."""
    return FactoryCompanionSupervisor(root).control_helper(
        policy_sha=policy_sha, lease_identity=lease_identity, argv=argv
    )


def bound_factory_identity(*, policy_sha: str, root: Path = DEFAULT_ROOT) -> str:
    """Return the persisted factory identity for a scoped control helper."""
    if policy_sha != POLICY_SHA256:
        raise AdmissionError("companion policy SHA differs")
    supervisor = FactoryCompanionSupervisor(root)
    with _locked(supervisor.lock_path):
        state = supervisor._read()
        factory = state.get("factory", {})
        identity = factory.get("identity")
        if (
            factory.get("policy_sha256") != policy_sha
            or not isinstance(identity, str)
            or not identity
        ):
            raise AdmissionError("no bound companion factory identity")
        return identity


def factory_control_update(
    *,
    identity: str,
    policy_sha: str,
    protected: bool,
    streams: dict[str, Any],
    owned: list[dict[str, Any]] | None = None,
    root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Publish factory protected-period facts, draining tests before protection."""
    if policy_sha != POLICY_SHA256:
        raise AdmissionError("companion policy SHA differs")
    supervisor = FactoryCompanionSupervisor(root)
    supervisor.update_factory_state(
        identity=identity, protected=protected, streams=streams, owned=owned
    )
    return factory_runtime_guard(identity, policy_sha=policy_sha, root=root)


def host_check_snapshot(
    *, active_factory_identity: str | None, policy_sha: str, root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    """Return versioned, receipt-ready host facts for slowdown-resume checks."""
    supervisor = FactoryCompanionSupervisor(root)
    assessment = supervisor.assessment(policy_sha=policy_sha)
    with _locked(supervisor.lock_path):
        state = supervisor._read()
        samples = state.get("samples", [])
        monitor = state.get("monitor")
        started = state.get("light_only_since")
        expected_identity = active_factory_identity or state.get("factory", {}).get("identity")
        registry_complete = (
            isinstance(monitor, dict)
            and all(key in monitor for key in ("pid", "start", "pgid"))
            and all(
                isinstance(item, dict)
                and item.get("policy_sha256") == policy_sha
                and (
                    item.get("class") == "monitor"
                    or item.get("lease_identity") == expected_identity
                )
                for item in state.get("registrations", {}).values()
                if item.get("status") == "running"
            )
        )
    return {
        "schema": "bl-labels-f0-slowdown-host-check/v2",
        "concurrency_policy_sha256": policy_sha,
        "factory_attempt_identity": active_factory_identity,
        "no_unmanaged_compute": assessment.no_unmanaged_compute,
        "no_active_test": not assessment.active_test,
        "light_only_observation_seconds": assessment.light_only_seconds,
        "light_observation_required_seconds": WARMUP,
        "observation": {
            "started_monotonic": started,
            "ended_monotonic": supervisor.clock(),
            "sample_count": len(samples),
            "trailing_cpu_core_seconds": sum(
                float(item.get("total_seconds", 0)) for item in samples
            ),
            "trailing_light_cpu_core_seconds": sum(
                float(item.get("light_seconds", 0)) for item in samples
            ),
            "rss_bytes": int(samples[-1].get("rss", 0)) if samples else 0,
        },
        "monitor_identity": monitor,
        "registry_complete": registry_complete,
        "monitor_heartbeat_age_seconds": assessment.monitor_heartbeat_age,
        "registry_monitor_identity": str(Path(root) / "registry.json"),
        "companion_healthy": assessment.healthy,
        "reason": assessment.reason,
    }


def factory_startup_guard(
    active_factory_identity: str | None, *, policy_sha: str, root: Path = DEFAULT_ROOT
) -> dict[str, Any]:
    """Factory-facing admission result. It intentionally contains no name scan."""
    from research.compute.factory_companion_qualification import require_activation

    require_activation()
    receipt = host_check_snapshot(
        active_factory_identity=active_factory_identity, policy_sha=policy_sha, root=root
    )
    # Factory startup must not require a test-lane warmup.  The 60-second
    # observation remains mandatory for a test admission and the resume host
    # predicate that consumes this receipt.
    ready = (
        receipt["companion_healthy"]
        and receipt["no_unmanaged_compute"]
        and receipt["no_active_test"]
    )
    return {
        "ok": ready,
        "reason": None if ready else receipt["reason"] or "companion_warmup",
        "action": "continue" if ready else "preempt",
        "receipt": receipt,
    }


def factory_runtime_guard(
    active_factory_identity: str | None,
    *,
    policy_sha: str,
    root: Path = DEFAULT_ROOT,
    early_warning: bool = False,
) -> dict[str, Any]:
    """Factory runtime guard; early warning sheds tests while hard failures preempt."""
    supervisor = FactoryCompanionSupervisor(root)
    if early_warning:
        supervisor.revoke_tests()
    receipt = host_check_snapshot(
        active_factory_identity=active_factory_identity, policy_sha=policy_sha, root=root
    )
    hard_failure = not receipt["companion_healthy"] or not receipt["no_unmanaged_compute"]
    return {
        "ok": not hard_failure,
        "reason": receipt["reason"],
        "action": "preempt" if hard_failure else "continue",
        "receipt": receipt,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--lease-identity", required=True)
    parser.add_argument("--worktree", default=os.getcwd())
    parser.add_argument("--review-manifest")
    parser.add_argument("--review-sha")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("client")
    tool = sub.add_parser("tool")
    tool.add_argument("--class", dest="klass", choices=("light", "test"), required=True)
    sub.add_parser("status")
    monitor = sub.add_parser("monitor")
    monitor.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, command = _parser().parse_known_args(argv)
    from research.compute.factory_companion_qualification import (
        qualification_status,
        require_activation,
    )

    if args.mode != "status":
        require_activation()
    supervisor = FactoryCompanionSupervisor(args.root)
    if args.mode == "client":
        if command[:1] == ["--"]:
            command = command[1:]
        if command:
            try:
                approved = validate_command(
                    "client", command, args.worktree, args.review_manifest, args.review_sha
                )
            except CommandValidationError as error:
                raise SystemExit(str(error)) from error
            return supervisor.launch(
                "client",
                policy_sha=args.policy_sha,
                lease_identity=args.lease_identity,
                worktree=args.worktree,
                argv=command,
                approved=approved,
            )
        print(
            supervisor.register_client(
                policy_sha=args.policy_sha,
                lease_identity=args.lease_identity,
                worktree=args.worktree,
                executable=sys.executable,
                argv=sys.argv,
            )
        )
        return 0
    if args.mode == "status":
        print(
            json.dumps(
                {
                    **asdict(supervisor.assessment(policy_sha=args.policy_sha)),
                    "qualification": qualification_status(),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "monitor":
        if os.getpgrp() != os.getpid():
            os.setsid()
        os.setpriority(os.PRIO_PROCESS, os.getpid(), 15)
        if os.getpriority(os.PRIO_PROCESS, os.getpid()) != 15:
            raise SystemExit("unable to verify companion monitor nice level")
        supervisor.claim_monitor()
        if args.once:
            supervisor.sample(monitor=True)
            return 0
        # A dedicated monitor process, rather than the factory driver's loop,
        # keeps the heartbeat current while pmset/ioreg checks block.
        while True:
            supervisor.sample(monitor=True)
            time.sleep(5)
        return 0
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("tool requires -- followed by an argv command")
    try:
        approved = validate_command(
            args.klass, command, args.worktree, args.review_manifest, args.review_sha
        )
    except CommandValidationError as error:
        raise SystemExit(str(error)) from error
    return supervisor.launch(
        args.klass,
        policy_sha=args.policy_sha,
        lease_identity=args.lease_identity,
        worktree=args.worktree,
        argv=command,
        approved=approved,
    )


if __name__ == "__main__":
    raise SystemExit(main())
