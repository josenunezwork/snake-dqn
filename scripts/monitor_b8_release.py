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
WATCH_ROSTER_ALL_B8 = "all_b8"
WATCH_ROSTER_MIXED = "mixed"
WATCH_ROSTERS = (WATCH_ROSTER_ALL_B8, WATCH_ROSTER_MIXED)
# Registered Watch death baselines are roster-mode and size specific. They
# must not be inherited from or pooled with another Watch roster.
WATCH_DEATHS_PER_1K_BASELINE = {
    6: 0.424,
    8: 0.496,
    12: 0.768,
}
# Compatibility view of the all-B8 rows from the monitoring erratum.
WATCH_DEATHS_PER_1K_BASELINE_BY_ROSTER = {
    **{(WATCH_ROSTER_ALL_B8, size): value for size, value in WATCH_DEATHS_PER_1K_BASELINE.items()},
    # T15 mixed N6 continuous Watch qualification. Mixed N8/N12 intentionally
    # have no death-rate baseline and therefore fail closed.
    (WATCH_ROSTER_MIXED, 6): 0.272,
}
# T12B check-4 amended qualification, default-speed GameState.update p95.
UPDATE_P95_BASELINE_MS = {
    ("watch", 6): 8.048326522111893,
    ("watch", 12): 15.417010523378849,
    ("play", 5): 5.789862293750048,
    ("play", 11): 11.726538324728608,
}
WATCH_UPDATE_P95_BASELINE_MS = {
    (WATCH_ROSTER_ALL_B8, 6): UPDATE_P95_BASELINE_MS[("watch", 6)],
    (WATCH_ROSTER_ALL_B8, 12): UPDATE_P95_BASELINE_MS[("watch", 12)],
    # T15 retained quiet-machine update p95 values, kept at source precision.
    (WATCH_ROSTER_MIXED, 6): 7.792219356633723,
    (WATCH_ROSTER_MIXED, 8): 9.519156138412654,
    (WATCH_ROSTER_MIXED, 12): 13.141333474777639,
}
DEFAULT_SPEED = 12.0
QUALIFICATION_UPDATE_BOUND_MS = 66.667


def watch_label(roster_mode: str, roster_size: int) -> str:
    """Keep established all-B8 labels while making mixed rows unambiguous."""
    if roster_mode == WATCH_ROSTER_ALL_B8:
        return f"watch{roster_size}"
    return f"{roster_mode}_watch{roster_size}"


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
    fallback.extend(item for item in items if item["event"] == "mixed_roster_fallback")
    exceptions = [item for item in items if item["event"] == "ego2s_exception"]
    watch_ticks = [item for item in items if item["event"] == "watch_tick"]
    latency_ticks = [item for item in items if item["event"] in ("watch_tick", "play_tick")]
    if not watch_ticks:
        faults.append("missing B8 Watch tick telemetry")

    decisions = deaths = 0
    watch_deaths: dict[tuple[str, int], dict[str, int]] = {}
    default_update_ms: dict[tuple[str, str, int], list[float]] = {}
    watch_timestamps: dict[tuple[str, int], list[float]] = {}
    timestamps: list[float] = []
    for item in watch_ticks:
        try:
            if item.get("checkpoint") != "b8-s3":
                faults.append("Watch telemetry checkpoint is not b8-s3")
                continue
            decision = int(item["decisions"])
            death = int(item["deaths"])
            roster_size = int(item["roster_size"])
            # Pre-T15 records did not emit mode. They only ever represented the
            # all-B8 release roster, so retain that interpretation.
            roster_mode = item.get("roster_mode", WATCH_ROSTER_ALL_B8)
            speed = float(item["speed"])
            update = float(item["update_ms"])
            timestamp = float(item["timestamp"])
            if (
                decision not in (0, 1)
                or death not in (0, 1)
                or death > decision
                or roster_size <= 0
                or roster_mode not in WATCH_ROSTERS
                or not all(math.isfinite(value) for value in (speed, update, timestamp))
                or update < 0
                or type(item.get("latency_eligible")) is not bool
            ):
                raise ValueError("invalid Watch tick values")
            decisions += decision
            deaths += death
            roster_key = (roster_mode, roster_size)
            roster_deaths = watch_deaths.setdefault(roster_key, {"decisions": 0, "deaths": 0})
            roster_deaths["decisions"] += decision
            roster_deaths["deaths"] += death
            timestamps.append(timestamp)
            watch_timestamps.setdefault(roster_key, []).append(timestamp)
            if speed == DEFAULT_SPEED and item["latency_eligible"]:
                default_update_ms.setdefault(("watch", roster_mode, roster_size), []).append(update)
        except (KeyError, TypeError, ValueError, OverflowError):
            faults.append("incomplete Watch tick telemetry")
    if decisions <= 0:
        faults.append("no actual Watch hero decisions")
    if not timestamps:
        faults.append("missing valid Watch telemetry timestamp")

    death_rate = None if decisions <= 0 else 1000.0 * deaths / decisions
    deaths_by_roster: dict[str, dict[str, float | int | None]] = {}
    for (roster_mode, roster_size), values in sorted(watch_deaths.items()):
        roster_decisions = values["decisions"]
        roster_death_rate = (
            None if roster_decisions <= 0 else 1000.0 * values["deaths"] / roster_decisions
        )
        baseline = WATCH_DEATHS_PER_1K_BASELINE_BY_ROSTER.get((roster_mode, roster_size))
        label = watch_label(roster_mode, roster_size)
        deaths_by_roster[label] = {
            "decisions": roster_decisions,
            "deaths": values["deaths"],
            "deaths_per_1000": roster_death_rate,
            "baseline_deaths_per_1000": baseline,
            "trigger_per_1000": None if baseline is None else 2.0 * baseline,
            "roster_mode": roster_mode,
        }
        if baseline is None:
            faults.append(f"no registered Watch death baseline for {label}")
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
                default_update_ms.setdefault(("play", "", roster_size), []).append(update)
        except (KeyError, TypeError, ValueError, OverflowError):
            faults.append("incomplete Play tick telemetry")

    latency: dict[str, dict[str, float | None]] = {}
    for (mode, roster_mode, roster_size), samples in sorted(default_update_ms.items()):
        baseline = (
            WATCH_UPDATE_P95_BASELINE_MS.get((roster_mode, roster_size))
            if mode == "watch"
            else UPDATE_P95_BASELINE_MS.get((mode, roster_size))
        )
        observed_p95 = percentile95(samples)
        label = watch_label(roster_mode, roster_size) if mode == "watch" else f"{mode}{roster_size}"
        latency[label] = {
            "update_p95_ms": observed_p95,
            "qualification_update_p95_ms": baseline,
            "trigger_ms": None if baseline is None else 1.5 * baseline,
            "qualification_bound_ms": QUALIFICATION_UPDATE_BOUND_MS,
        }
        if baseline is None:
            qualification = (
                "T12B" if mode == "watch" and roster_mode == WATCH_ROSTER_ALL_B8 else "T15"
            )
            faults.append(f"no {qualification}-qualified update baseline for {label}")
    for roster_mode, roster_size in watch_deaths:
        latency_key = ("watch", roster_mode, roster_size)
        if (
            roster_mode,
            roster_size,
        ) in WATCH_UPDATE_P95_BASELINE_MS and latency_key not in default_update_ms:
            faults.append(
                "missing default-speed eligible Watch update telemetry for "
                f"{watch_label(roster_mode, roster_size)}"
            )
    if not latency:
        faults.append("missing default-speed 12 fps Watch update telemetry")

    current = time.time() if now is None else now
    elapsed_days = None if not timestamps else max(0.0, (current - min(timestamps)) / 86400.0)
    windows_by_roster: dict[str, dict[str, float | int | bool]] = {}
    for roster_key, values in sorted(watch_deaths.items()):
        roster_mode, roster_size = roster_key
        roster_timestamps = watch_timestamps[roster_key]
        roster_elapsed_days = max(0.0, (current - min(roster_timestamps)) / 86400.0)
        roster_decisions = values["decisions"]
        windows_by_roster[watch_label(roster_mode, roster_size)] = {
            "decisions": roster_decisions,
            "elapsed_days": roster_elapsed_days,
            "window_complete": roster_elapsed_days >= 7.0 and roster_decisions >= 100000,
        }
    # A mode/size group must independently satisfy the fixed seven-day and
    # 100k-decision observation window; traffic is never pooled across groups.
    window_complete = bool(windows_by_roster) and all(
        values["window_complete"] for values in windows_by_roster.values()
    )
    metrics = {
        "decisions": decisions,
        "deaths": deaths,
        "deaths_per_1000": death_rate,
        "deaths_by_roster": deaths_by_roster,
        "latency_by_roster": latency,
        "windows_by_roster": windows_by_roster,
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
    for label, values in deaths_by_roster.items():
        observed_death_rate = values["deaths_per_1000"]
        death_trigger = values["trigger_per_1000"]
        if (
            death_trigger is not None
            and observed_death_rate is not None
            and observed_death_rate > death_trigger
        ):
            triggers.append(
                "ROLLBACK: %s deaths/1k %.6f exceeds %.3f"
                % (label, observed_death_rate, death_trigger)
            )
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
        "Watch hero deaths/1k aggregate (informational only): "
        f"{metrics['deaths_per_1000'] if metrics['deaths_per_1000'] is not None else 'missing'} "
        f"({metrics['deaths']}/{metrics['decisions']})"
    )
    for label, values in metrics["deaths_by_roster"].items():
        print(
            "%s hero deaths/1k: %s (%s/%s; baseline=%s, trigger: > %s)"
            % (
                label,
                values["deaths_per_1000"],
                values["deaths"],
                values["decisions"],
                values["baseline_deaths_per_1000"],
                values["trigger_per_1000"],
            )
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
        "Aggregate observation: %.3f days, %d decisions; %s. Owner check-in: %s."
        % (
            metrics["elapsed_days"] if metrics["elapsed_days"] is not None else 0.0,
            metrics["decisions"],
            "all mode/size windows complete" if metrics["window_complete"] else "window active",
            (
                "due now"
                if metrics["elapsed_days"] is not None and metrics["elapsed_days"] >= 7
                else "due day 7"
            ),
        )
    )
    for label, values in metrics["windows_by_roster"].items():
        print(
            "%s observation window: %.3f days, %d decisions; %s"
            % (
                label,
                values["elapsed_days"],
                values["decisions"],
                "complete" if values["window_complete"] else "active",
            )
        )
    for message in messages:
        print(f"NOT_HEALTHY: {message}")
    print("HEALTHY" if healthy else "NOT_HEALTHY")
    return 0 if healthy else 2


if __name__ == "__main__":
    sys.exit(main())
