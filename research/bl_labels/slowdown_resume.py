"""Outcome-blind escalation policy for sealed F0 slowdown trips.

The slowdown statistic itself lives in :mod:`mac_driver`.  This module only
reads terminal wrappers and host-check receipts to decide whether an operator
may resume at a requested width.  It never opens lives, labels, or rate-sealed
diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exports import _verify, publish_json
from .factory_persistence import INTERIM_RULING_SHA256, POLICY_ID, RULING_SHA256

W6 = 6
W12 = 12
SIX_HOURS = 6 * 60 * 60
HOST_CHECK_SCHEMA = "bl-labels-f0-slowdown-host-check-persistence/v3"
INTERIM_POLICY = "ALLOW_NICE15_INTERIM"
PERSISTENCE_POLICY = "ALLOW_PERSISTENCE_30S"


def _operational_pin() -> str:
    """Bind recovery to this checkout's pin, never the frozen science env pin."""
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if dirty:
        raise ValueError("interim operational checkout is dirty; HOLD")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", result):
        raise ValueError("cannot establish interim operational pin; HOLD")
    return result


def _finite_positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} is not finite; HOLD")
    if value <= 0:
        raise ValueError(f"{what} must be positive; HOLD")
    return float(value)


def _read_triplet(path: Path) -> dict[str, Any]:
    _verify(path)
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid sealed JSON: {path}; HOLD") from error
    if not isinstance(value, dict):
        raise ValueError(f"sealed JSON must be an object: {path}; HOLD")
    return value


def _terminal_records(state_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return exact, ordered start/terminal records after requiring complete pairs."""
    starts = sorted(state_dir.glob("attempt-*.start.json"), key=lambda p: p.stat().st_mtime_ns)
    terminals = sorted(
        state_dir.glob("attempt-*.terminal.json"), key=lambda p: p.stat().st_mtime_ns
    )
    if {p.name.replace(".start.json", "") for p in starts} != {
        p.name.replace(".terminal.json", "") for p in terminals
    }:
        raise ValueError("slowdown resume state has an unverified attempt hole; HOLD")
    values = []
    for start_path in starts:
        terminal_path = start_path.with_name(
            start_path.name.replace(".start.json", ".terminal.json")
        )
        start, terminal = _read_triplet(start_path), _read_triplet(terminal_path)
        if terminal.get("start_sha256") != _verify(start_path):
            raise ValueError("slowdown terminal start binding differs; HOLD")
        if terminal.get("workers") != start.get("workers"):
            raise ValueError("slowdown terminal worker width differs; HOLD")
        values.append((start, terminal))
    return values


def _trip(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("slowdown trip record is malformed; HOLD")
    stream = value.get("stream")
    width = value.get("width")
    if (
        not isinstance(stream, str)
        or not stream
        or isinstance(width, bool)
        or not isinstance(width, int)
        or not 1 <= width <= W12
    ):
        raise ValueError("slowdown trip stream or width is malformed; HOLD")
    return {
        "stream": stream,
        "width": width,
        "wall_seconds": _finite_positive(value.get("wall_seconds"), "slowdown trip wall_seconds"),
        "epoch": _finite_positive(value.get("epoch"), "slowdown trip epoch"),
    }


def _legacy_trips(start: dict[str, Any], terminal: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Translate pre-record-list telemetry without opening outcome payloads."""
    telemetry = terminal.get("telemetry_sealed_no_outcomes", [])
    if not isinstance(telemetry, list):
        raise ValueError("legacy slowdown telemetry is malformed; HOLD")
    events = [
        item
        for item in telemetry
        if isinstance(item, dict) and item.get("event") == "slowdown_guard"
    ]
    if not events:
        return
    width = terminal.get("workers")
    if width not in (W6, W12):
        raise ValueError("legacy slowdown terminal width is malformed; HOLD")
    wall = _finite_positive(terminal.get("cumulative_elapsed_seconds"), "legacy slowdown wall")
    started = _finite_positive(start.get("started_epoch"), "legacy slowdown start epoch")
    elapsed = _finite_positive(terminal.get("elapsed_seconds"), "legacy slowdown elapsed")
    # Legacy guard events are concurrent worker notifications for a single
    # cooperative preemption.  They form one trip, never one trip per worker.
    # Legacy F0 records have no stream identity; Watch6 is the sole lawful
    # reconstruction for that original D256 route.
    yield {"stream": "watch6", "width": width, "wall_seconds": wall, "epoch": started + elapsed}


def slowdown_trip_records(state_dir: Path) -> list[dict[str, Any]]:
    """Read all terminal trip lists, retaining old telemetry as W6 Watch6 facts."""
    records = []
    for start, terminal in _terminal_records(state_dir):
        raw = terminal.get("slowdown_trip_records")
        if raw is None:
            records.extend(_trip(item) for item in _legacy_trips(start, terminal))
            continue
        if not isinstance(raw, list):
            raise ValueError("slowdown trip records are malformed; HOLD")
        records.extend(_trip(item) for item in raw)
    return records


def _qualifying_pairs(
    records: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["stream"], record["width"]), []).append(record)
    pairs = []
    for values in grouped.values():
        values.sort(key=lambda record: record["wall_seconds"])
        for earlier, later in zip(values, values[1:]):
            if later["wall_seconds"] - earlier["wall_seconds"] <= SIX_HOURS:
                pairs.append((earlier, later))
    return pairs


def escalation(state_dir: Path) -> dict[str, float | int | None]:
    """Summarize the latest qualifying trip pair for each width."""
    result: dict[str, float | int | None] = {"w6_epoch": None, "w12_epoch": None}
    for _earlier, later in _qualifying_pairs(slowdown_trip_records(state_dir)):
        # W1--W6 are the same operational host-check class after a W6
        # thermal stepdown.  Preserve exact widths for pair matching above,
        # then classify only after a pair is established.
        if later["width"] <= W6:
            key = "w6_epoch"
        elif later["width"] == W12:
            key = "w12_epoch"
        else:
            # Widths 7--11 are not admitted F0 widths.  Keep accepting an
            # authentic wrapper shape without inventing an escalation class.
            continue
        prior = result[key]
        result[key] = max(float(prior) if prior is not None else 0.0, later["epoch"])
    return result


def _baseline_sanity_widths(state_dir: Path) -> set[int]:
    """Return widths whose frozen baseline refused sanity before any outcome read."""
    widths = set()
    for _start, terminal in _terminal_records(state_dir):
        if terminal.get("stop_reason") == "slowdown baseline sanity":
            width = terminal.get("workers")
            if width not in (W6, W12):
                raise ValueError("baseline sanity terminal width is malformed; HOLD")
            widths.add(width)
    return widths


def _host_check_after(state_dir: Path, trip_epoch: float) -> bool:
    pin = _operational_pin()
    for path in state_dir.glob("slowdown-host-check-*.json"):
        value = _read_triplet(path)
        if (
            value.get("schema") == HOST_CHECK_SCHEMA
            and value.get("policy") == PERSISTENCE_POLICY
            and value.get("policy_id") == POLICY_ID
            and value.get("ruling_sha256") == RULING_SHA256
            and value.get("interim_ruling_sha256") == INTERIM_RULING_SHA256
            and value.get("operational_pin") == pin
            and value.get("for_trip_epoch") == trip_epoch
            and _finite_positive(value.get("checked_epoch"), "host check epoch") >= trip_epoch
            and "no_sol_codex" not in value
            and value.get("no_disallowed_competing_compute") is True
            and value.get("serial_test_lane_stopped") is True
            and isinstance(value.get("host_health"), dict)
            and value["host_health"].get("ac_power") is True
            and value["host_health"].get("lid_open") is True
            and isinstance(value.get("pmset_therm"), str)
            and isinstance(value.get("top_cpu_consumers"), list)
            and _valid_process_window(value.get("process_window"), state_dir)
        ):
            return True
    return False


def _valid_process_window(value: object, state_dir: Path) -> bool:
    """Verify this v3 window and its outcome-blind full-observation journal."""
    if not isinstance(value, dict):
        return False
    try:
        start, end = value["window_start_monotonic"], value["window_end_monotonic"]
        count = value["observation_count"]
        path = Path(value["telemetry_path"])
        if (
            value.get("policy_id") != POLICY_ID
            or value.get("ruling_sha256") != RULING_SHA256
            or value.get("interim_ruling_sha256") != INTERIM_RULING_SHA256
            or type(count) is not int
            or count < 7
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or end - start < 30.0
            or value.get("final_pending") != []
            or value.get("observation_errors") != []
            or value.get("stop_reasons") != []
            or path.parent.resolve() != state_dir.resolve()
        ):
            return False
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != value.get("telemetry_sha256"):
            return False
        rows = [json.loads(line) for line in payload.splitlines()]
        if any(
            row.get("policy_id") != POLICY_ID
            or row.get("ruling_sha256") != RULING_SHA256
            or row.get("interim_ruling_sha256") != INTERIM_RULING_SHA256
            or row.get("attempt_id") != value.get("attempt_id")
            or row.get("decision") in {"monitor_fault", "preempt_persistent"}
            for row in rows
        ):
            return False
        samples = [row for row in rows if row.get("decision") == "observation_complete"]
        times = [row["sample_monotonic"] for row in samples]
        pending = set()
        for row in rows:
            identity = row.get("pid"), row.get("birth")
            if row.get("decision") == "pending":
                pending.add(identity)
            elif row.get("decision") in {"clear_priority", "clear_absent"}:
                pending.discard(identity)
        return (
            len(samples) == count
            and not pending
            and [row["sample_sequence"] for row in samples] == list(range(1, count + 1))
            and times[0] == start
            and times[-1] == end
            and all(0 <= later - earlier <= 15 for earlier, later in zip(times, times[1:]))
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def resume_policy(
    requested_workers: int, state_dir: Path, *, now: float | None = None
) -> tuple[int, list[str]]:
    """Apply the slowdown escalation ruling to a requested W6/W12 continuation.

    W12's 24-hour clean-window proof is intentionally composed by
    ``w12_runtime`` because it owns the common thermal-clean receipt rule.
    ``now`` is accepted for that uniform public API and deterministic callers.
    """
    del now
    if requested_workers not in (W6, W12):
        raise ValueError("slowdown resume accepts only W6 or W12")
    baseline_widths = _baseline_sanity_widths(state_dir)
    if W6 in baseline_widths:
        raise ValueError("W6 slowdown baseline sanity requires a new judge ruling; HOLD")
    if requested_workers == W12 and W12 in baseline_widths:
        return W6, ["operational_width_fallback:slowdown baseline sanity"]
    state = escalation(state_dir)
    # W12 trip cooldown is composed by w12_runtime with the thermal clean
    # interval, which is the only sealed proof authorized to re-admit W12.
    # A W6 pair is a host-wide escalation: changing to W12 cannot evade the
    # required operator host check.  Once a fresh sealed check is logged, the
    # normal W12 admission and clean-window policy still decides the width.
    if state["w6_epoch"] is not None:
        epoch = float(state["w6_epoch"])
        if not _host_check_after(state_dir, epoch):
            raise ValueError("slowdown W6 host check required; HOLD")
    return requested_workers, []


def _sol_codex_job(argv: str) -> bool:
    """Match launched CLI jobs without treating desktop app helpers as jobs."""
    desktop_roots = (
        "/Applications/ChatGPT.app/",
        str(Path.home() / ".codex/computer-use") + "/",
        str(Path.home() / "Library/Application Support/Codex") + "/",
    )
    if argv.lstrip().startswith(desktop_roots):
        return False
    if "codex-handoff.sh" in argv or "CODEX_MODEL=gpt-6.1-sol" in argv:
        return True
    return any(
        not match.group("executable").startswith("/Applications/ChatGPT.app/")
        for match in re.finditer(r"(?:^|\s)(?P<executable>(?:\S*/)?codex)\s+exec(?:\s|$)", argv)
    )


def record_host_check(state_dir: Path, *, checked_epoch: float | None = None) -> Path:
    """Capture the required operator W6 check as a completion-last receipt."""
    from research.compute.factory_interim_launcher import serial_test_slot

    # Hold the same lock as every admitted numerical test until publication.
    # A lock-file's existence alone cannot establish that a test is stopped.
    with serial_test_slot():
        return _record_host_check_locked(state_dir, checked_epoch=checked_epoch)


def _record_host_check_locked(state_dir: Path, *, checked_epoch: float | None = None) -> Path:
    """Observe health/top consumers while the numerical lane is fenced."""
    latest = escalation(state_dir)["w6_epoch"]
    if latest is None:
        raise ValueError("no qualifying W6 slowdown pair requires a host check; HOLD")
    epoch = (
        time.time()
        if checked_epoch is None
        else _finite_positive(checked_epoch, "host check epoch")
    )
    if epoch < float(latest):
        raise ValueError("host check predates qualifying W6 slowdown pair; HOLD")
    pin = _operational_pin()
    import uuid

    from .factory_persistence import PersistenceTracker
    from .mac_driver import (
        DEFAULT_TIMING_ARMED,
        MacGuardError,
        _factory_admission_window,
        _guard_snapshot,
    )

    tracker = PersistenceTracker(
        "host-check-" + uuid.uuid4().hex,
        state_dir / f"slowdown-host-check-{int(epoch)}.process.jsonl",
    )
    try:
        window = _factory_admission_window(tracker, state_dir, DEFAULT_TIMING_ARMED)
        health = _guard_snapshot(state_dir)
    except (MacGuardError, subprocess.SubprocessError, OSError) as exc:
        tracker.stops.append(f"host check refused: {type(exc).__name__}")
        publish_json(
            state_dir / f"slowdown-host-check-{int(epoch)}.refused.json", tracker.summary()
        )
        raise ValueError("host check found unsafe process observation; HOLD: " + str(exc)) from exc
    therm = health["thermal"]
    rows = subprocess.check_output(
        ["/bin/ps", "-axo", "pid=,pcpu=,comm="],
        text=True,
        errors="surrogateescape",
        timeout=10,
    ).splitlines()
    consumers = sorted(
        rows, key=lambda row: float(row.split()[1]) if len(row.split()) >= 2 else 0.0, reverse=True
    )[:20]
    if _operational_pin() != pin:
        raise ValueError("host check operational pin changed during observation; HOLD")
    path = state_dir / f"slowdown-host-check-{int(epoch)}.json"
    publish_json(
        path,
        {
            "schema": HOST_CHECK_SCHEMA,
            "policy": PERSISTENCE_POLICY,
            "policy_id": POLICY_ID,
            "ruling_sha256": RULING_SHA256,
            "interim_ruling_sha256": INTERIM_RULING_SHA256,
            "operational_pin": pin,
            "for_trip_epoch": latest,
            "checked_epoch": time.time() if checked_epoch is None else epoch,
            "process_window": window,
            "pmset_therm": therm,
            "top_cpu_consumers": consumers,
            "host_health": health,
            "no_disallowed_competing_compute": True,
            "serial_test_lane_stopped": True,
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the required W6 slowdown host check.")
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    print(record_host_check(args.state_dir))


if __name__ == "__main__":
    main()
