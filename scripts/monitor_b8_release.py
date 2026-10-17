#!/usr/bin/env python3
"""Evaluate B8 release-monitoring stderr JSONL records.

Usage:
    ./venv/bin/python scripts/monitor_b8_release.py /var/log/snake-web.stderr

It is intentionally conservative: absent B8 Watch records, malformed monitoring
records, unqualified rosters, or incomplete decision/death fields report
``NOT_HEALTHY`` rather than a green result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PREFIX = "b8-release-monitor:"
DEATHS_PER_1K_TRIGGER = 0.848
# T12B check-4 amended qualification, default-speed GameState.update p95.
UPDATE_P95_BASELINE_MS = {
    ("watch", 6): 8.048326522111893,
    ("watch", 12): 15.417010523378849,
    ("play", 5): 5.789862293750048,
    ("play", 11): 11.726538324728608,
}
DEFAULT_SPEED = 12.0


def percentile95(values: list[float]) -> float:
    """Match NumPy's linear percentile interpolation used by qualification."""
    if not values:
        raise ValueError("p95 requires at least one sample")
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def records(lines: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract monitor JSONL records and retain malformed prefixed lines as faults."""
    parsed: list[dict[str, Any]] = []
    faults: list[str] = []
    for line in lines:
        if "ego2s serving: dqn called with an empty dispatch queue; row 0 served" in line:
            parsed.append({"event": "ego2s_fallback", "fallback_reason": line.strip()})
            continue
        if PREFIX not in line:
            continue
        _, raw = line.split(PREFIX, 1)
        try:
            value = json.loads(raw.strip())
        except json.JSONDecodeError:
            faults.append("malformed monitoring JSON")
            continue
        if not isinstance(value, dict) or not isinstance(value.get("event"), str):
            faults.append("monitoring record missing event")
            continue
        parsed.append(value)
    return parsed, faults


def assess(
    items: list[dict[str, Any]], parse_faults: list[str], now: float | None = None
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return health, trigger messages, and the metrics printed by the CLI."""
    faults = list(parse_faults)
    resolutions = [item for item in items if item["event"] == "resolution"]
    if not resolutions:
        faults.append("missing released-default resolution record")
    fallback = [item for item in resolutions if item.get("fallback_reason")]
    fallback.extend(item for item in items if item["event"] == "ego2s_fallback")
    exceptions = [item for item in items if item["event"] == "ego2s_exception"]
    watch_ticks = [item for item in items if item["event"] == "watch_tick"]
    latency_ticks = [item for item in items if item["event"] in ("watch_tick", "play_tick")]
    if not watch_ticks:
        faults.append("missing B8 Watch tick telemetry")

    decisions = deaths = 0
    default_update_ms: dict[tuple[str, int], list[float]] = {}
    timestamps: list[float] = []
    for item in watch_ticks:
        try:
            if item.get("checkpoint") != "b8-s3":
                faults.append("Watch telemetry checkpoint is not b8-s3")
                continue
            decision = int(item["decisions"])
            death = int(item["deaths"])
            roster_size = int(item["roster_size"])
            speed = float(item["speed"])
            update = float(item["update_ms"])
            timestamp = float(item["timestamp"])
            if (
                decision not in (0, 1)
                or death not in (0, 1)
                or death > decision
                or roster_size <= 0
                or not all(math.isfinite(value) for value in (speed, update, timestamp))
                or update < 0
                or type(item.get("latency_eligible")) is not bool
            ):
                raise ValueError("invalid Watch tick values")
            decisions += decision
            deaths += death
            timestamps.append(timestamp)
            if speed == DEFAULT_SPEED and item["latency_eligible"]:
                default_update_ms.setdefault(("watch", roster_size), []).append(update)
        except (KeyError, TypeError, ValueError, OverflowError):
            faults.append("incomplete Watch tick telemetry")
    if decisions <= 0:
        faults.append("no actual Watch hero decisions")
    if not timestamps:
        faults.append("missing valid Watch telemetry timestamp")

    death_rate = None if decisions <= 0 else 1000.0 * deaths / decisions
    for item in latency_ticks:
        if item.get("event") != "play_tick":
            continue
        try:
            if item.get("checkpoint") != "b8-s3":
                raise ValueError("non-B8 Play telemetry")
            roster_size = int(item["driven_ai"])
            speed = float(item["speed"])
            update = float(item["update_ms"])
            if (
                roster_size <= 0
                or update < 0
                or not all(math.isfinite(value) for value in (speed, update))
                or type(item.get("latency_eligible")) is not bool
            ):
                raise ValueError("invalid Play tick values")
            if speed == DEFAULT_SPEED and item["latency_eligible"]:
                default_update_ms.setdefault(("play", roster_size), []).append(update)
        except (KeyError, TypeError, ValueError, OverflowError):
            faults.append("incomplete Play tick telemetry")

    latency: dict[str, dict[str, float | None]] = {}
    for (mode, roster_size), samples in sorted(default_update_ms.items()):
        baseline = UPDATE_P95_BASELINE_MS.get((mode, roster_size))
        observed_p95 = percentile95(samples)
        label = f"{mode}{roster_size}"
        latency[label] = {
            "update_p95_ms": observed_p95,
            "qualification_update_p95_ms": baseline,
            "trigger_ms": None if baseline is None else 1.5 * baseline,
        }
        if baseline is None:
            faults.append(f"no T12B-qualified update baseline for {label}")
    if not latency:
        faults.append("missing default-speed 12 fps Watch update telemetry")

    current = time.time() if now is None else now
    elapsed_days = None if not timestamps else max(0.0, (current - min(timestamps)) / 86400.0)
    window_complete = bool(elapsed_days is not None and elapsed_days >= 7.0 and decisions >= 100000)
    metrics = {
        "decisions": decisions,
        "deaths": deaths,
        "deaths_per_1000": death_rate,
        "latency_by_roster": latency,
        "fallback_records": len(fallback),
        "ego2s_exceptions": len(exceptions),
        "elapsed_days": elapsed_days,
        "window_complete": window_complete,
    }
    triggers: list[str] = []
    if fallback:
        triggers.append("ROLLBACK: recorded served-checkpoint or ego2s fallback reason")
    if exceptions:
        triggers.append("ROLLBACK: recorded ego2s-path exception")
    if death_rate is not None and death_rate > DEATHS_PER_1K_TRIGGER:
        triggers.append(f"ROLLBACK: deaths/1k {death_rate:.6f} exceeds {DEATHS_PER_1K_TRIGGER:.3f}")
    for label, values in latency.items():
        observed_p95 = values["update_p95_ms"]
        latency_trigger = values["trigger_ms"]
        if (
            latency_trigger is not None
            and observed_p95 is not None
            and observed_p95 > latency_trigger
        ):
            triggers.append(
                "ROLLBACK: %s update p95 %.6f ms exceeds %.6f ms"
                % (label, observed_p95, latency_trigger)
            )
    return not faults and not triggers, faults + triggers, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="stderr/server log containing B8 monitor records")
    args = parser.parse_args(argv)
    try:
        with args.log.open(encoding="utf-8", errors="replace") as handle:
            items, parse_faults = records(handle)
    except OSError as exc:
        print(f"NOT_HEALTHY: cannot read log: {exc}")
        return 2
    healthy, messages, metrics = assess(items, parse_faults)
    print("B8 release monitoring")
    print(
        "fallback/ego2s exceptions: "
        f"{metrics['fallback_records']}/{metrics['ego2s_exceptions']} (trigger: any > 0)"
    )
    print(
        "Watch hero deaths/1k: "
        f"{metrics['deaths_per_1000'] if metrics['deaths_per_1000'] is not None else 'missing'} "
        f"({metrics['deaths']}/{metrics['decisions']}; trigger: > {DEATHS_PER_1K_TRIGGER:.3f})"
    )
    for label, values in metrics["latency_by_roster"].items():
        print(
            "%s default-speed update p95: %s ms "
            "(qualification=%s, trigger: > %s ms)"
            % (
                label,
                values["update_p95_ms"],
                values["qualification_update_p95_ms"],
                values["trigger_ms"],
            )
        )
    print(
        "Observation window: %.3f days, %d decisions; %s. Owner check-in: %s."
        % (
            metrics["elapsed_days"] if metrics["elapsed_days"] is not None else 0.0,
            metrics["decisions"],
            "complete" if metrics["window_complete"] else "active",
            (
                "due now"
                if metrics["elapsed_days"] is not None and metrics["elapsed_days"] >= 7
                else "due day 7"
            ),
        )
    )
    for message in messages:
        print(f"NOT_HEALTHY: {message}")
    print("HEALTHY" if healthy else "NOT_HEALTHY")
    return 0 if healthy else 2


if __name__ == "__main__":
    sys.exit(main())
