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
import shlex
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .exports import _verify, publish_json

W6 = 6
W12 = 12
SIX_HOURS = 6 * 60 * 60
HOST_CHECK_SCHEMA_V1 = "bl-labels-f0-slowdown-host-check/v1"
HOST_CHECK_SCHEMA = "bl-labels-f0-slowdown-host-check/v2"
CONCURRENCY_POLICY = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_FACTORY_CONCURRENCY_20261023.md"
)


def _concurrency_policy_sha256() -> str:
    """Return the exact ruling identity used by the Mac companion supervisor."""
    try:
        value = hashlib.sha256(CONCURRENCY_POLICY.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError("factory concurrency policy is unavailable; HOLD") from error
    from research.compute.factory_companion_policy import POLICY_SHA256

    if value != POLICY_SHA256:
        raise ValueError("factory concurrency policy bytes differ from the reviewed ruling; HOLD")
    return value


def _record_host_health(state_dir: Path) -> dict[str, Any]:
    """Reuse the unchanged AC/lid/thermal/disk admission check for a resume."""
    from .mac_driver import _guard_snapshot

    return _guard_snapshot(state_dir)


def _require_companion_activation() -> None:
    """Do not publish a new-policy host receipt before native qualification."""
    from research.compute.factory_companion_qualification import require_activation

    require_activation()


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
    for path in state_dir.glob("slowdown-host-check-*.json"):
        value = _read_triplet(path)
        if (
            value.get("schema") == HOST_CHECK_SCHEMA
            and value.get("for_trip_epoch") == trip_epoch
            and _finite_positive(value.get("checked_epoch"), "host check epoch") >= trip_epoch
            and value.get("no_unmanaged_compute") is True
            and value.get("no_active_test") is True
            and value.get("companion_healthy") is True
            and value.get("registry_complete") is True
            and isinstance(value.get("concurrency_policy_sha256"), str)
            and value["concurrency_policy_sha256"] == _concurrency_policy_sha256()
            and isinstance(value.get("registry_monitor_identity"), str)
            and isinstance(value.get("monitor_identity"), dict)
            and type(value["monitor_identity"].get("pid")) is int
            and type(value["monitor_identity"].get("pgid")) is int
            and isinstance(value["monitor_identity"].get("start"), str)
            and isinstance(value.get("monitor_heartbeat_age_seconds"), (int, float))
            and 0 <= value["monitor_heartbeat_age_seconds"] <= 10
            and isinstance(value.get("host_health"), dict)
            and value["host_health"].get("ac_power") is True
            and value["host_health"].get("lid_open") is True
            and isinstance(value["host_health"].get("thermal"), str)
            and isinstance(value["host_health"].get("thermal_parsed"), dict)
            and isinstance(value["host_health"].get("free_disk_bytes"), int)
            and value["host_health"]["free_disk_bytes"] >= 50 * 1024**3
            and isinstance(value.get("observation"), dict)
            and value["observation"].get("light_only_seconds", 0) >= 60
            and type(value["observation"].get("sample_count")) is int
            and value["observation"]["sample_count"] >= 12
            and all(
                isinstance(value["observation"].get(key), (int, float))
                and not isinstance(value["observation"].get(key), bool)
                and math.isfinite(value["observation"][key])
                for key in (
                    "started_monotonic",
                    "ended_monotonic",
                    "trailing_cpu_core_seconds",
                    "trailing_light_cpu_core_seconds",
                )
            )
            and value["observation"]["ended_monotonic"] - value["observation"]["started_monotonic"]
            >= 60
            and value["observation"]["trailing_light_cpu_core_seconds"] <= 15
            and value["observation"]["trailing_cpu_core_seconds"] <= 75
            and isinstance(value["observation"].get("rss_bytes"), int)
            and 0 <= value["observation"]["rss_bytes"] <= 8 * 1024**3
        ):
            return True
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
    try:
        words = shlex.split(argv)
    except ValueError:
        return False
    if not words:
        return False
    executables = [word for word in words if word.startswith("/")]
    if any(executable.startswith(desktop_roots) for executable in executables):
        return False
    executable = Path(words[0]).name
    if executable == "env":
        for word in words[1:]:
            if "=" not in word:
                executable = Path(word).name
                break
    if (
        executable == "codex-handoff.sh"
        or any(Path(word).name == "codex-handoff.sh" for word in words)
        or "CODEX_MODEL=gpt-6.1-sol" in words
    ):
        return True
    return executable == "codex" and len(words) > 1 and words[1] == "exec"


def record_host_check(state_dir: Path, *, checked_epoch: float | None = None) -> Path:
    """Capture the required operator W6 check as a completion-last receipt."""
    _require_companion_activation()
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
    policy_sha = _concurrency_policy_sha256()
    from research.compute.factory_companion import (
        bound_factory_identity,
        control_helper,
        host_check_snapshot,
    )

    # This Python entry point is measured as a bounded light helper rather
    # than silently exempted from the argv-based unmanaged-work inventory.
    identity = bound_factory_identity(policy_sha=policy_sha)
    with control_helper(policy_sha=policy_sha, lease_identity=identity, argv=sys.argv):
        health = _record_host_health(state_dir)
        companion = host_check_snapshot(active_factory_identity=None, policy_sha=policy_sha)
    if not (
        companion.get("no_unmanaged_compute") is True
        and companion.get("no_active_test") is True
        and companion.get("companion_healthy") is True
        and companion.get("registry_complete") is True
        and isinstance(companion.get("light_only_observation_seconds"), (int, float))
        and companion["light_only_observation_seconds"] >= 60
        and isinstance(companion.get("observation"), dict)
        and isinstance(companion.get("monitor_identity"), dict)
    ):
        raise ValueError("companion host observation is not ready; HOLD")
    path = state_dir / f"slowdown-host-check-{int(epoch)}.json"
    publish_json(
        path,
        {
            "schema": HOST_CHECK_SCHEMA,
            "for_trip_epoch": latest,
            "checked_epoch": epoch,
            "concurrency_policy_sha256": policy_sha,
            "no_unmanaged_compute": True,
            "no_active_test": True,
            "companion_healthy": True,
            "registry_complete": True,
            "registry_monitor_identity": companion["registry_monitor_identity"],
            "monitor_heartbeat_age_seconds": companion["monitor_heartbeat_age_seconds"],
            "monitor_identity": companion["monitor_identity"],
            "host_health": health,
            "observation": {
                **companion["observation"],
                "light_only_seconds": companion["light_only_observation_seconds"],
                "factory_attempt_identity": companion["factory_attempt_identity"],
            },
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
