"""Preemptible, resume-safe Mac production driver for immutable F0 life units.

This is an operational wrapper only.  It delegates all science to the frozen
factory and exports no label outcomes in its telemetry.  It holds the shared
Mac lease while active, but yields at a completed life boundary when an
operator arms ``ops/mac-timing-armed`` for timing, strict, or serving work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from research.mac_lease import DEFAULT_LEASE_PATH, held

from .exports import (
    completed_session_parents,
    load_ledger,
    publish_json,
    publish_life,
    publish_session,
)

DEFAULT_TIMING_ARMED = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed")
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
MAX_WORKERS = 12
DEFAULT_WORKERS = 6
MAX_WORKER_RSS_BYTES = 4 * 1024**3
MAX_TOTAL_RSS_BYTES = 40 * 1024**3
MIN_FREE_DISK_BYTES = 50 * 1024**3
# The coordinator is single-instance under the Mac lease.  This registry lets
# main's outer finally reap owned children even if Process.start/IPC raises.
_OWNED_ACTIVE: dict[int, mp.Process] = {}


class MacGuardError(RuntimeError):
    """Mac host conditions are unsafe for a new F0 life admission."""


@dataclass(frozen=True)
class _Job:
    allocation: dict[str, Any]
    skip_lives: set[int]
    suppress_session: bool


def platform_attestation() -> dict[str, str]:
    """Return the host identity bound into the operational receipt."""
    return {
        "platform": "darwin-arm64",
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "code_sha": os.environ.get("F0_COMMIT", "UNSET"),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _command_output(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout


def _guard_snapshot(output: Path) -> dict[str, Any]:
    """Fail closed before every admission, without inspecting label evidence."""
    if platform.system() != "Darwin":
        raise MacGuardError("F0 Mac production driver requires Darwin")
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise MacGuardError("F0 Mac production driver requires darwin-arm64")
    battery = _command_output(["/usr/bin/pmset", "-g", "batt"])
    lid = _command_output(["/usr/sbin/ioreg", "-r", "-k", "AppleClamshellState"])
    thermal = _command_output(["/usr/bin/pmset", "-g", "therm"])
    if "AC Power" not in battery:
        raise MacGuardError("refusing F0 admission on battery")
    if '"AppleClamshellState" = No' not in lid:
        raise MacGuardError("refusing F0 admission with lid closed")
    from research.compute.thermal_guard import parse_pmset_therm, thermal_reasons

    parsed_thermal = parse_pmset_therm(thermal)
    reasons = thermal_reasons(parsed_thermal)
    if reasons:
        raise MacGuardError(f"refusing F0 admission under thermal guard: {','.join(reasons)}")
    usage = shutil.disk_usage(output)
    if usage.free < MIN_FREE_DISK_BYTES:
        raise MacGuardError("refusing F0 admission: less than 50 GiB free disk")
    return {
        "ac_power": True,
        "lid_open": True,
        "thermal": thermal.strip(),
        "thermal_parsed": parsed_thermal,
        "free_disk_bytes": usage.free,
    }


def _rss_bytes(pid: int) -> int:
    """Read Darwin resident bytes. A failed observation is unsafe, not zero."""
    value = _command_output(["/bin/ps", "-o", "rss=", "-p", str(pid)]).strip()
    if not value or not value.isdigit():
        raise MacGuardError(f"cannot read RSS for worker {pid}")
    return int(value) * 1024


def _rss_many(pids: list[int]) -> dict[int, int]:
    """Read all owned-worker RSS values in one bounded process-table call."""
    if not pids:
        return {}
    text = _command_output(["/bin/ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))])
    values: dict[int, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            values[int(fields[0])] = int(fields[1]) * 1024
    return values


def _stop_workers(active: dict[int, mp.Process]) -> None:
    """Stop and join only the driver-owned worker processes before lease release."""
    for process in active.values():
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 3
    for process in active.values():
        process.join(timeout=max(0, deadline - time.monotonic()))
    survivors = [process for process in active.values() if process.is_alive()]
    for process in survivors:
        os.kill(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + 2
    for process in survivors:
        process.join(timeout=max(0, deadline - time.monotonic()))
    if any(process.is_alive() for process in active.values()):
        raise RuntimeError("owned F0 worker failed to stop before lease release")


def _command_group_live(pgid: int) -> bool:
    """Count executable members of the command's owned process group, not zombies."""
    rows = _command_output(["/bin/ps", "-axo", "pgid=,stat="])
    for line in rows.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == str(pgid) and not fields[1].startswith("Z"):
            return True
    return False


def _stop_command_group(child: subprocess.Popen) -> None:
    """Reap the leader and prove all its descendants stopped before lease release."""
    # The leader can have exited while a multiprocessing child remains alive.
    # Always inspect the owned group, including the successful-command path.
    for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        if _command_group_live(child.pid):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while _command_group_live(child.pid) and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.05)
        if not _command_group_live(child.pid):
            break
    child.wait(timeout=5)
    if _command_group_live(child.pid):
        raise RuntimeError("owned Mac command group remains executable before lease release")


def _timing_armed(path: Path) -> bool:
    return path.is_file()


def _prepare_jobs(
    control: dict[str, Any], ledger: Path
) -> tuple[str, list[dict[str, Any]], list[_Job]]:
    namespace, allocation = control["namespace"], control["allocation"]
    done = load_ledger(ledger, namespace, allocation)
    _verify_resume_metadata(ledger, namespace, allocation, done)
    sessions = completed_session_parents(ledger, done)
    by_parent: dict[str, set[int]] = {}
    for name in done:
        parent, marker, life = name.partition("--life-")
        if marker:
            by_parent.setdefault(parent, set()).add(int(life))
    jobs = [
        _Job(item, by_parent.get(item["allocation_id"], set()), item["allocation_id"] in sessions)
        for item in allocation
        if item["allocation_id"] not in sessions
    ]
    return namespace, allocation, jobs


def _verify_resume_metadata(
    ledger: Path, namespace: str, allocation: list[dict[str, Any]], names: set[str]
) -> None:
    """Refuse a hash-valid but foreign leaf before it can suppress collection.

    This opens only wrapper identity fields.  Rows and outcomes remain sealed.
    """
    by_parent = {item["allocation_id"]: item for item in allocation}
    for name in names:
        payload = ledger.parent / "lives" / f"{name}.json"
        value = json.loads(payload.read_text())
        parent = name.split("--life-", 1)[0].removesuffix("--session")
        expected = by_parent.get(parent)
        if expected is None or value.get("namespace") != namespace:
            raise ValueError("foreign F0 resume leaf")
        if name.endswith("--session"):
            valid = (
                value.get("schema") == "bl-labels-f0-session/v1"
                and value.get("allocation_id") == parent
            )
        elif "--life-" in name:
            suffix = name.rsplit("--life-", 1)[1]
            valid = (
                value.get("schema") == "bl-labels-f0-life/v1"
                and value.get("allocation_id") == parent
                and type(value.get("life")) is int
                and value["life"] >= 0
                and suffix == f"{value['life']:06d}"
            )
        else:
            valid = value.get("schema") == "bl-labels-f0-unit/v1"
        if not valid or value.get("allocation") != expected or value.get("status") != "COMPLETE":
            raise ValueError("foreign F0 resume leaf")


def _worker(
    job: _Job,
    namespace: str,
    allocation: list[dict[str, Any]],
    b8: Path,
    frp3: Path,
    output: Path,
    timing_armed: Path,
    stop_event: Any,
    queue: Any,
) -> None:
    """Collect one parent, preserving every completed life before yielding."""
    os.environ.update(THREAD_ENV)
    try:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
        if os.getpriority(os.PRIO_PROCESS, 0) != 10:
            raise OSError("worker priority is not nice 10")
    except OSError as error:
        queue.put({"kind": "error", "error": f"setting nice 10 failed: {error}"})
        return
    from .factory import FactoryInvalid, FactoryPreempted, collect_unit

    def publish(value: dict[str, Any]) -> None:
        if value.get("schema") == "bl-labels-f0-session/v1":
            if not job.suppress_session:
                publish_session(output, namespace, allocation, value)
        else:
            value = dict(value)
            parent_index = job.allocation.get("allocation_index")
            if type(parent_index) is int:
                # Lexical parent allocation order, then natural life order;
                # completion races cannot insert a later life into a prefix.
                value["allocation_index"] = parent_index * 1_000_000 + value["life"]
                value["life_id"] = f"{namespace}/{job.allocation['allocation_id']}/{value['life']}"
                value["parent_world"] = job.allocation["allocation_id"]
                value["stream"] = "S-Watch6" if job.allocation["stream"] == "watch6" else "S-gate"
                value["rescue_bearing"] = any(
                    bool(row.get("decisive")) for row in value.get("rows", [])
                )
            publish_life(output, namespace, allocation, value)
            queue.put(
                {
                    "kind": "life_boundary",
                    "allocation_id": job.allocation["allocation_id"],
                    # This is coordinator-private operational accounting; it
                    # is stripped before sealed telemetry is published.
                    "_rescue_bearing": any(
                        bool(row.get("decisive")) for row in value.get("rows", [])
                    ),
                }
            )

    def progress(value: dict[str, Any]) -> None:
        # Operational-only: no rows, labels, selected actions, or outcomes.
        queue.put({"kind": "progress", "allocation_id": job.allocation["allocation_id"], **value})

    supplied = dict(job.allocation)
    supplied.update(
        publish_life=publish,
        progress=progress,
        skip_lives=job.skip_lives,
        should_preempt=lambda: stop_event.is_set() or _timing_armed(timing_armed),
    )
    try:
        collect_unit(supplied, b8, frp3, namespace)
        queue.put({"kind": "complete", "allocation_id": job.allocation["allocation_id"]})
    except FactoryPreempted:
        queue.put({"kind": "preempted", "allocation_id": job.allocation["allocation_id"]})
    except (FactoryInvalid, ValueError) as error:
        publish_json(
            output / f"INVALID-{job.allocation['allocation_id']}.json",
            {
                "schema": "bl-labels-f0-invalid/v1",
                "namespace": namespace,
                "allocation_id": job.allocation["allocation_id"],
                "reason": str(error),
            },
        )
        queue.put({"kind": "invalid", "error": str(error)})
    except BaseException as error:
        queue.put({"kind": "error", "error": repr(error)})


def _validate_workers(workers: int, owner_w12_ok: bool, calibration_passed: bool) -> None:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("F0 workers must be in [1, 12]")
    if workers > DEFAULT_WORKERS and not (owner_w12_ok and calibration_passed):
        raise ValueError(
            "W>6 requires explicit owner approval and a passing 30-minute W12 calibration"
        )


def _verify_w12_calibration(path: Path) -> None:
    """Require a sealed calibration receipt before W12 admission."""
    from .exports import _verify

    _verify(path)
    value = json.loads(path.read_text())
    duration, ratio = value.get("duration_seconds"), value.get("w12_w6_wall_ratio")
    code_sha = os.environ.get("F0_COMMIT", "")
    if not (
        value.get("schema") == "bl-labels-f0-w12-calibration/v1"
        and isinstance(duration, (int, float))
        and not isinstance(duration, bool)
        and math.isfinite(duration)
        and duration >= 1800
        and isinstance(ratio, (int, float))
        and not isinstance(ratio, bool)
        and math.isfinite(ratio)
        and 0 < ratio <= 1.30
        and value.get("thermal_not_ok") == 0
        and value.get("slowdown_trips") == 0
        and value.get("platform") == "darwin-arm64"
        and value.get("code_sha") == os.environ.get("F0_COMMIT")
        and len(code_sha) == 40
        and all(c in "0123456789abcdef" for c in code_sha)
    ):
        raise ValueError("W12 calibration receipt does not satisfy the frozen Mac admission rule")


def run(
    control: dict[str, Any],
    ledger: Path,
    b8: Path,
    frp3: Path,
    output: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    timing_armed: Path = DEFAULT_TIMING_ARMED,
    deadline_seconds: float | None = None,
    owner_w12_ok: bool = False,
    calibration_passed: bool = False,
    state_dir: Path | None = None,
    started_monotonic: float | None = None,
) -> dict[str, Any]:
    """Run admitted parents until complete, guard stop, or cooperative preemption."""
    started = time.monotonic() if started_monotonic is None else started_monotonic
    _validate_workers(workers, owner_w12_ok, calibration_passed)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("F0 Mac output directory must be fresh and empty")
    output.mkdir(parents=True, exist_ok=True)
    if "mac_gate" in control:
        from .mac_gates import require_admission

        require_admission(control)
        source_commit = control.get("source_commit")
        if not isinstance(source_commit, str) or len(source_commit) != 40:
            raise ValueError("admitted Mac control needs a full source_commit")
        if os.environ.get("F0_COMMIT") != source_commit:
            raise ValueError("F0_COMMIT must match the admitted Mac control source_commit")
        stage = control["mac_gate"]["stage"]
        pack = Path(control["mac_gate"]["package_root"])
        expected_cap = 10800 if stage == "canary-b" else 57600
        expected_state = pack / (
            "canary-state-mac-01" if stage == "canary-b" else "d256-state-mac-01"
        )
        if (
            deadline_seconds != expected_cap
            or state_dir is None
            or state_dir.resolve() != expected_state.resolve()
        ):
            raise ValueError("Mac registered cap/state cannot be changed or reset")
        if stage == "canary-b" and workers != 6:
            raise ValueError("Canary-B requires exactly six Mac workers")
    if _timing_armed(timing_armed):
        raise MacGuardError("refusing F0 admission while Mac timing is armed")
    namespace, allocation, jobs = _prepare_jobs(control, ledger)
    attempt_id: str | None = None
    prior_elapsed = 0.0
    prior_attempts: list[str] = []
    prior_thermal: list[float] = []
    prior_admission_workers = workers
    prior_baseline: list[float] = []
    prior_rate_events: list[tuple[float, bool]] = []
    expected_control = hashlib.sha256(
        json.dumps(control, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if deadline_seconds is not None and (
        isinstance(deadline_seconds, bool)
        or not math.isfinite(deadline_seconds)
        or deadline_seconds <= 0
    ):
        raise ValueError("deadline_seconds must be finite and positive")
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        # A prior start without a verified terminal receipt is a crash HOLD;
        # never silently reset its wall/accounting envelope on resume.
        from .exports import _verify

        start_paths = sorted(
            state_dir.glob("attempt-*.start.json"), key=lambda p: p.stat().st_mtime_ns
        )
        if control.get("mac_gate", {}).get("stage") == "canary-b" and len(start_paths) >= 2:
            raise MacGuardError("Canary-B original plus one completion already consumed; HOLD")
        for start_receipt in start_paths:
            _verify(start_receipt)
            start_value = json.loads(start_receipt.read_text())
            if (
                start_value.get("namespace") != namespace
                or start_value.get("workers") != workers
                or start_value.get("control_sha256") != expected_control
                or start_value.get("output") == str(output)
                or start_value.get("deadline_seconds") != deadline_seconds
                or start_value.get("source_commit") != control.get("source_commit")
                or start_value.get("state_dir") != str(state_dir.resolve())
            ):
                raise MacGuardError("prior Mac factory start identity differs; HOLD")
            terminal = start_receipt.with_name(
                start_receipt.name.replace(".start.json", ".terminal.json")
            )
            if not terminal.exists():
                raise MacGuardError(
                    "prior Mac factory attempt lacks a verified terminal receipt; HOLD"
                )
            _verify(terminal)
            terminal_value = json.loads(terminal.read_text())
            if (
                terminal_value.get("namespace") != namespace
                or terminal_value.get("workers") != workers
                or terminal_value.get("output") != start_value.get("output")
                or terminal_value.get("start_sha256") != _verify(start_receipt)
            ):
                raise MacGuardError("prior Mac factory terminal identity differs; HOLD")
            elapsed = terminal_value.get("elapsed_seconds")
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed <= 0
            ):
                raise MacGuardError("invalid prior Mac wall charge; HOLD")
            prior_elapsed += elapsed
            prior_thermal.extend(terminal_value.get("thermal_trip_epochs", []))
            prior_admission_workers = terminal_value.get("admission_workers_final", workers)
            prior_baseline = terminal_value.get("first_hour_life_seconds", [])
            sealed_path = terminal_value.get("rate_sealed_path")
            if sealed_path:
                _verify(Path(sealed_path))
                prior_rate_events = [
                    tuple(x) for x in json.loads(Path(sealed_path).read_text())["events"]
                ]
            prior_attempts.append(terminal.name)
        if deadline_seconds is not None and prior_elapsed >= deadline_seconds:
            raise MacGuardError("prior Mac factory attempts exhausted the frozen wall cap; HOLD")
        attempt_id = uuid.uuid4().hex
        publish_json(
            state_dir / f"attempt-{attempt_id}.start.json",
            {
                "schema": "bl-labels-f0-mac-attempt-start/v1",
                "namespace": namespace,
                "allocation_sha256": hashlib.sha256(
                    json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "workers": workers,
                "deadline_seconds": deadline_seconds,
                "original_wall_cap_seconds": deadline_seconds,
                "started_epoch": time.time(),
                "output": str(output.resolve()),
                "state_dir": str(state_dir.resolve()),
                "source_commit": control.get("source_commit"),
                "control_sha256": hashlib.sha256(
                    json.dumps(control, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "platform_attestation": platform_attestation(),
            },
        )
    deadline = None
    if deadline_seconds is not None:
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        if prior_elapsed >= deadline_seconds:
            raise MacGuardError("prior Mac factory attempts exhausted the frozen wall cap; HOLD")
        # Reserve the complete ten-minute life-boundary drain inside the fixed
        # cap, so an operational preemption cannot overrun it.
        deadline = started + max(0.0, deadline_seconds - prior_elapsed - 600)
    ctx = mp.get_context("spawn")
    queue: Any = ctx.Queue()
    stop_event = ctx.Event()
    active = _OWNED_ACTIVE
    active.clear()
    pending = list(jobs)
    completed: list[str] = []
    preempted = False
    telemetry: list[dict[str, Any]] = []
    invalid: str | None = None
    stop_reason: str | None = None
    preempt_started: float | None = None
    life_started: dict[str, float] = {}
    baseline_lives: list[float] = prior_baseline
    sealed_lives: list[tuple[float, bool]] = prior_rate_events
    thermal_trips = sorted({at for at in prior_thermal if time.time() - at <= 86400})
    admission_workers = prior_admission_workers
    thermal_epoch_seen = False
    next_report = time.monotonic() + 110

    def note_thermal() -> None:
        nonlocal admission_workers, thermal_epoch_seen
        if thermal_epoch_seen:
            return
        thermal_epoch_seen = True
        thermal_trips.append(time.time())
        if len(thermal_trips) % 3 == 0 and control.get("mac_gate", {}).get("stage") != "canary-b":
            admission_workers = max(1, admission_workers - 3)
        telemetry.append({"event": "thermal_pause", "workers_next": admission_workers})

    try:
        while pending or active or not queue.empty():
            if time.monotonic() >= next_report:
                print(
                    json.dumps(
                        {
                            "event": "F0_MAC_PROGRESS",
                            "active_workers": len(active),
                            "pending_parents": len(pending),
                            "completed_parents": len(completed),
                            "elapsed_seconds": time.monotonic() - started,
                            "preempting": preempted,
                        }
                    ),
                    flush=True,
                )
                next_report = time.monotonic() + 110
            if deadline is not None and time.monotonic() >= deadline:
                preempted = True
            if _timing_armed(timing_armed):
                preempted = True
            if preempted and preempt_started is None:
                preempt_started = time.monotonic()
                stop_event.set()
            admission_guard: dict[str, Any] | None = None
            while pending and len(active) < admission_workers and not preempted:
                if _timing_armed(timing_armed) or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    preempted = True
                    stop_event.set()
                    break
                try:
                    if admission_guard is None:
                        admission_guard = _guard_snapshot(output)
                except (MacGuardError, subprocess.SubprocessError, OSError) as error:
                    if "thermal" in str(error).lower():
                        note_thermal()
                    stop_reason = f"operational admission guard: {error}"
                    preempted, stop_event = True, stop_event
                    stop_event.set()
                    break
                if _timing_armed(timing_armed):
                    preempted = True
                    stop_event.set()
                    break
                telemetry.append(
                    {"event": "admit", "at_monotonic": time.monotonic(), "guard": admission_guard}
                )
                job = pending.pop(0)
                process = ctx.Process(
                    target=_worker,
                    args=(
                        job,
                        namespace,
                        allocation,
                        b8,
                        frp3,
                        output,
                        timing_armed,
                        stop_event,
                        queue,
                    ),
                )
                process.start()
                active[process.pid] = process
                life_started[job.allocation["allocation_id"]] = time.monotonic()
            try:
                messages = [queue.get(timeout=1.0)]
                while True:
                    try:
                        messages.append(queue.get_nowait())
                    except Empty:
                        break
                for message in messages:
                    kind = message.pop("kind")
                    rescue_bearing = bool(message.pop("_rescue_bearing", False))
                    telemetry.append({"event": kind, "at_monotonic": time.monotonic(), **message})
                    if kind == "complete":
                        completed.append(str(message["allocation_id"]))
                    elif kind == "life_boundary":
                        allocation_id = str(message["allocation_id"])
                        sealed_lives.append((time.time(), rescue_bearing))
                        elapsed = time.monotonic() - life_started.get(allocation_id, started)
                        if prior_elapsed + time.monotonic() - started <= 3600:
                            baseline_lives.append(elapsed)
                        life_started[allocation_id] = time.monotonic()
                        # First-hour observations are the factory's local baseline.
                        if prior_elapsed + time.monotonic() - started > 3600 and baseline_lives:
                            baseline = sum(baseline_lives) / len(baseline_lives)
                            if elapsed > baseline * 1.30:
                                preempted = True
                                telemetry.append(
                                    {
                                        "event": "slowdown_guard",
                                        "elapsed": elapsed,
                                        "baseline": baseline,
                                    }
                                )
                        if prior_elapsed + time.monotonic() - started >= 6 * 3600 and not any(
                            time.time() - at <= 6 * 3600 for at in thermal_trips
                        ):
                            recent = [
                                item for at, item in sealed_lives if time.time() - at <= 6 * 3600
                            ]
                            rate = sum(recent) / 6
                            baseline_rate = control.get("operational_canary_rate")
                            if (
                                isinstance(baseline_rate, (int, float))
                                and rate < float(baseline_rate) * 0.5
                            ):
                                stop_reason = (
                                    "sealed rolling six-hour rescue rate below "
                                    "50 percent Canary-B baseline"
                                )
                                preempted = True
                                stop_event.set()
                    elif kind == "preempted":
                        preempted = True
                    elif kind == "invalid":
                        invalid = str(message.get("error", kind))
                        preempted = True
                    elif kind == "error":
                        stop_reason = str(message.get("error", kind))
                        preempted = True
                        stop_event.set()
            except Empty:
                pass
            # Reap completed processes before sampling the remaining RSS table.
            for pid, process in list(active.items()):
                if not process.is_alive():
                    process.join()
                    del active[pid]
                    if process.exitcode not in (0, None) and invalid is None:
                        stop_reason = f"worker {pid} exited {process.exitcode}"
                        preempted = True
                        stop_event.set()
            try:
                _guard_snapshot(output)
                rss_values = _rss_many(list(active))
                if any(
                    pid not in rss_values and process.is_alive() for pid, process in active.items()
                ):
                    raise MacGuardError("cannot read RSS for a live owned worker")
            except (MacGuardError, subprocess.SubprocessError, OSError) as error:
                if "thermal" in str(error).lower():
                    note_thermal()
                stop_reason = f"operational guard: {error}"
                preempted = True
                stop_event.set()
                rss_values = {}
            for pid, process in list(active.items()):
                if process.is_alive():
                    rss = rss_values.get(pid, 0)
                    total = sum(rss_values.values())
                    if rss > MAX_WORKER_RSS_BYTES or total > MAX_TOTAL_RSS_BYTES:
                        stop_reason = (
                            "RSS guard tripped; stopping new admissions at life boundaries"
                        )
                        preempted = True
                        stop_event.set()
                    continue
                process.join()
                del active[pid]
                if process.exitcode not in (0, None) and invalid is None:
                    invalid = f"worker {pid} exited {process.exitcode}"
                    preempted = True
            if preempted and not active:
                break
            if preempt_started is not None and time.monotonic() - preempt_started > 555:
                # Drain and force-stop only owned workers inside the ten-minute
                # bound. Uncompleted units stay absent from the verified ledger.
                stop_reason = "preemption did not reach a life boundary within 10 minutes"
                stop_event.set()
                _stop_workers(active)
                active.clear()
    finally:
        _stop_workers(active)
        active.clear()
    status = "INVALID" if invalid else ("PREEMPTED" if preempted else "COMPLETE")
    result = {
        "schema": "bl-labels-f0-mac-driver/v1",
        "status": status,
        "namespace": namespace,
        "workers": workers,
        "admission_workers_final": admission_workers,
        "nice": 10,
        "thread_env": THREAD_ENV,
        "completed_parents": completed,
        "pending_parents": [job.allocation["allocation_id"] for job in pending],
        "invalid_reason": invalid,
        "stop_reason": stop_reason,
        "thermal_trip_count_24h": len(thermal_trips),
        "thermal_trip_epochs": thermal_trips,
        "first_hour_life_seconds": baseline_lives,
        "elapsed_seconds": time.monotonic() - started,
        "prior_elapsed_seconds": prior_elapsed,
        "cumulative_elapsed_seconds": prior_elapsed + (time.monotonic() - started),
        "worker_seconds_denominator": workers * (time.monotonic() - started),
        "cumulative_worker_seconds_denominator": workers
        * (prior_elapsed + (time.monotonic() - started)),
        "platform_attestation": platform_attestation(),
        "telemetry_sealed_no_outcomes": telemetry,
    }
    if state_dir is not None:
        result["state_dir"] = str(state_dir.resolve())
        assert attempt_id is not None
        result["attempts_ledger"] = [*prior_attempts, f"attempt-{attempt_id}.terminal.json"]
        result["attempt_id"] = attempt_id
        result["start_receipt"] = f"attempt-{attempt_id}.start.json"
        result["start_sha256"] = hashlib.sha256(
            (state_dir / result["start_receipt"]).read_bytes()
        ).hexdigest()
        result["output"] = str(output)
        sealed_path = state_dir / f"attempt-{attempt_id}.rate-sealed.json"
        publish_json(sealed_path, {"events": sealed_lives})
        os.chmod(sealed_path, 0o600)
        result["rate_sealed_path"] = str(sealed_path.resolve())
        publish_json(state_dir / f"attempt-{attempt_id}.terminal.json", result)
    return result


def main() -> None:
    entry_started = time.monotonic()
    if len(sys.argv) > 1 and sys.argv[1] == "guard-command":
        guard_command()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--b8", required=True, type=Path)
    parser.add_argument("--frp3", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--timing-armed", type=Path, default=DEFAULT_TIMING_ARMED)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lease-owner", default="bl-labels-f0-mac-factory")
    parser.add_argument("--owner-w12-ok", action="store_true")
    parser.add_argument("--w12-calibration-receipt", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    control = json.loads(args.control.read_text())
    if args.check:
        from .mac_gates import require_admission

        require_admission(control)
        _prepare_jobs(control, args.ledger)
        _validate_workers(args.workers, args.owner_w12_ok, args.w12_calibration_receipt is not None)
        _guard_snapshot(args.output.parent)
        print(
            json.dumps({"status": "CHECKED", "workers": args.workers, "output": str(args.output)})
        )
        return
    if args.workers > DEFAULT_WORKERS:
        if args.w12_calibration_receipt is None:
            parser.error("W12 requires --w12-calibration-receipt")
        _verify_w12_calibration(args.w12_calibration_receipt)
    lease_seconds = args.deadline_seconds or 16 * 3600
    with held(
        args.lease_owner,
        "F0 Mac hindsight label factory",
        seconds=lease_seconds,
        path=args.lease_path,
    ) as lease:
        try:
            result = run(
                control,
                args.ledger,
                args.b8,
                args.frp3,
                args.output,
                workers=args.workers,
                timing_armed=args.timing_armed,
                deadline_seconds=args.deadline_seconds,
                owner_w12_ok=args.owner_w12_ok,
                calibration_passed=args.w12_calibration_receipt is not None,
                state_dir=args.state_dir,
                started_monotonic=entry_started,
            )
            result["mac_lease"] = lease.receipt
            publish_json(args.output / "runtime-result.json", result)
        finally:
            _stop_workers(_OWNED_ACTIVE)
            _OWNED_ACTIVE.clear()
    print(
        json.dumps(
            {key: result[key] for key in ("status", "workers", "elapsed_seconds")}, sort_keys=True
        )
    )
    if result["status"] != "COMPLETE":
        raise SystemExit(2)


def guard_command() -> None:
    """Run one Mac-equivalence command under the shared lease and host guards."""
    parser = argparse.ArgumentParser(description="guard one Mac F1/audit command")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--timing-armed", type=Path, default=DEFAULT_TIMING_ARMED)
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument("--lease-owner", default="bl-labels-f0-mac-equivalence")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(sys.argv[2:])
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("positive --seconds and a command after -- are required")
    if _timing_armed(args.timing_armed):
        raise MacGuardError("refusing Mac-equivalence command while timing is armed")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(THREAD_ENV)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    if args.check:
        _guard_snapshot(args.state_dir)
        print(json.dumps({"status": "CHECKED", "command": command}))
        return
    with held(
        args.lease_owner, "F0 Mac equivalence", seconds=args.seconds, path=args.lease_path
    ) as lease:
        _guard_snapshot(args.state_dir)
        child = subprocess.Popen(["/usr/bin/nice", "-n", "10", *command], start_new_session=True)
        deadline = time.monotonic() + args.seconds
        try:
            while child.poll() is None:
                if _timing_armed(args.timing_armed):
                    raise MacGuardError("Mac equivalence preempted by timing arm")
                _guard_snapshot(args.state_dir)
                if time.monotonic() >= deadline:
                    raise MacGuardError("Mac equivalence command exhausted wall cap")
                time.sleep(5)
        finally:
            _stop_command_group(child)
        if child.returncode:
            raise SystemExit(child.returncode)
        publish_json(
            args.state_dir / f"guard-command-{uuid.uuid4().hex}.json",
            {
                "schema": "bl-labels-mac-guard-command/v1",
                "lease": lease.receipt,
                "command": command,
            },
        )


if __name__ == "__main__":
    main()
