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
PLAY_ROSTER_ALL_B8 = "all_b8"
PLAY_ROSTER_MIXED = "mixed"
PLAY_ROSTERS = (PLAY_ROSTER_ALL_B8, PLAY_ROSTER_MIXED)
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
REPLAY_WINDOW_DAYS = 7.0
REPLAY_WINDOW_FRAMES = 50_000
REPLAY_SUMMARY_FULL_FRAMES = 600
REPLAY_SUMMARY_MIN_P95_FRAMES = 20
REPLAY_UPDATE_P95_TRIGGER_MS = 41.67
REPLAY_ACTIVE_EVENTS = (
    "watch_replay_active",
    "replay_active",
    "watch_replay_selected",
    "replay_selected",
)
REPLAY_SUMMARY_EVENTS = ("watch_replay_summary", "replay_summary", "watch_replay_tick")
REPLAY_FAILURE_EVENTS = ("replay_step_failure", "watch_replay_failure", "replay_failure")
REPLAY_BANK_EXHAUSTED_EVENTS = (
    "replay_bank_exhausted",
    "watch_replay_bank_exhausted",
    "bank_exhausted",
)


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


def replay_window_p95(
    summary_batches: list[tuple[int, float]], raw_updates: list[float]
) -> float | None:
    """Aggregate replay update p95 without letting tiny partial flushes dominate."""
    if not summary_batches and not raw_updates:
        return None
    if not summary_batches:
        return percentile95(raw_updates)
    all_batches = [*summary_batches, *((1, val) for val in raw_updates)]
    full_batches = [
        (frames, p95) for frames, p95 in all_batches if frames >= REPLAY_SUMMARY_FULL_FRAMES
    ]
    if not full_batches:
        full_batches = [
            (frames, p95) for frames, p95 in all_batches if frames >= REPLAY_SUMMARY_MIN_P95_FRAMES
        ]
    eligible = full_batches if full_batches else all_batches
    ordered = sorted(eligible, key=lambda item: item[1])
    total_frames = sum(frames for frames, _ in ordered)
    if total_frames <= 1 or len(ordered) == 1:
        return ordered[-1][1]
    target = 0.95 * (total_frames - 1)
    cumulative = 0
    for frames, p95 in ordered:
        if cumulative + frames > target:
            return p95
        cumulative += frames
    return ordered[-1][1]


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
    fallback.extend(
        item for item in items if item["event"] in ("replay_fallback", "watch_replay_fallback")
    )
    exceptions = [item for item in items if item["event"] == "ego2s_exception"]
    replay_active = [item for item in items if item["event"] in REPLAY_ACTIVE_EVENTS]
    replay_summaries = [item for item in items if item["event"] in REPLAY_SUMMARY_EVENTS]
    replay_failures = [item for item in items if item["event"] in REPLAY_FAILURE_EVENTS]
    replay_bank_exhausted = [
        item for item in items if item["event"] in REPLAY_BANK_EXHAUSTED_EVENTS
    ]
    replay_mode = bool(
        replay_active or replay_summaries or replay_failures or replay_bank_exhausted
    )
    watch_ticks = [item for item in items if item["event"] == "watch_tick"]
    latency_ticks = [item for item in items if item["event"] in ("watch_tick", "play_tick")]
    if not watch_ticks and not replay_mode:
        faults.append("missing B8 Watch tick telemetry")

    decisions = deaths = 0
    watch_deaths: dict[tuple[str, int], dict[str, int]] = {}
    default_update_ms: dict[tuple[str, str, int], list[float]] = {}
    watch_timestamps: dict[tuple[str, int], list[float]] = {}
    timestamps: list[float] = []
    replay_frames = 0
    replay_over_trigger_frames = 0
    replay_over_trigger_invalid = 0
    replay_summary_batches: list[tuple[int, float]] = []
    replay_raw_updates: list[float] = []
    replay_timestamps: list[float] = []
    replay_step_failures = len(replay_failures)
    replay_bank_exhaustions = len(replay_bank_exhausted)
    for item in replay_active:
        try:
            if item.get("checkpoint", "b8-s3") != "b8-s3":
                faults.append("Watch replay telemetry checkpoint is not b8-s3")
                continue
            roster_size = int(item.get("roster_size", 6))
            roster_mode = item.get("roster_mode", WATCH_ROSTER_MIXED)
            speed = float(item.get("speed", 24.0))
            timestamp = float(item["timestamp"])
            if (
                roster_size <= 0
                or roster_mode not in WATCH_ROSTERS
                or not all(math.isfinite(val) for val in (speed, timestamp))
            ):
                raise ValueError("invalid Watch replay active telemetry")
            replay_timestamps.append(timestamp)
            timestamps.append(timestamp)
        except (KeyError, TypeError, ValueError, OverflowError):
            faults.append("incomplete Watch replay telemetry")
    for item in replay_summaries:
        frames: int | None = None
        p95_val: float | None = None
        update_val: float | None = None
        try:
            if item.get("checkpoint", "b8-s3") != "b8-s3":
                faults.append("Watch replay telemetry checkpoint is not b8-s3")
                continue
            frames = int(item.get("frames", 1))
            roster_size = int(item.get("roster_size", 6))
            roster_mode = item.get("roster_mode", WATCH_ROSTER_MIXED)
            speed = float(item.get("speed", 24.0))
            timestamp = float(item["timestamp"])
            first_ts = float(item.get("first_timestamp", timestamp))
            p95_raw = item.get("update_p95_ms")
            update_raw = item.get("update_ms")
            if p95_raw is None and update_raw is None:
                raise ValueError("missing replay update latency")
            p95_val = None if p95_raw is None else float(p95_raw)
            update_val = None if update_raw is None else float(update_raw)
            if (
                frames <= 0
                or roster_size <= 0
                or roster_mode not in WATCH_ROSTERS
                or not all(math.isfinite(val) for val in (speed, timestamp, first_ts))
                or (p95_val is not None and (not math.isfinite(p95_val) or p95_val < 0))
                or (update_val is not None and (not math.isfinite(update_val) or update_val < 0))
            ):
                raise ValueError("invalid Watch replay telemetry values")
            replay_frames += frames
            replay_timestamps.extend((first_ts, timestamp))
            timestamps.extend((first_ts, timestamp))
            if p95_val is not None:
                replay_summary_batches.append((frames, p95_val))
            if update_val is not None:
                replay_raw_updates.append(update_val)
            if int(item.get("step_failures", 0) or 0) > 0:
                replay_step_failures += int(item["step_failures"])
            if (
                bool(item.get("bank_exhausted", False))
                or int(item.get("bank_exhaustions", 0) or 0) > 0
            ):
                replay_bank_exhaustions += max(1, int(item.get("bank_exhaustions", 0) or 0))
        except (KeyError, TypeError, ValueError, OverflowError):
            faults.append("incomplete Watch replay telemetry")
            continue
        if item.get("event") == "watch_replay_summary":
            if "over_trigger_frames" in item:
                over_trigger_raw = item["over_trigger_frames"]
                if (
                    frames is not None
                    and type(over_trigger_raw) is int
                    and 0 <= over_trigger_raw <= frames
                ):
                    replay_over_trigger_frames += over_trigger_raw
                else:
                    replay_over_trigger_invalid += 1
            elif update_val is not None and update_val > REPLAY_UPDATE_P95_TRIGGER_MS:
                replay_over_trigger_frames += 1
    for item in [*replay_failures, *replay_bank_exhausted]:
        ts = item.get("timestamp")
        if isinstance(ts, (int, float)) and math.isfinite(ts):
            replay_timestamps.append(float(ts))
            timestamps.append(float(ts))
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
    if decisions <= 0 and not replay_mode:
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
    play_ticks_excluded_by_roster = 0
    for item in latency_ticks:
        if item.get("event") != "play_tick":
            continue
        try:
            if item.get("checkpoint") != "b8-s3":
                raise ValueError("non-B8 Play telemetry")
            roster_mode = item.get("roster_mode", PLAY_ROSTER_ALL_B8)
            driven_ai = int(item["driven_ai"])
            roster_size_raw = item.get("roster_size")
            roster_size = driven_ai if roster_size_raw is None else int(roster_size_raw)
            speed = float(item["speed"])
            update = float(item["update_ms"])
            if (
                roster_mode not in PLAY_ROSTERS
                or roster_size <= 0
                or driven_ai < 0
                or (roster_mode == PLAY_ROSTER_ALL_B8 and driven_ai <= 0)
                or update < 0
                or not all(math.isfinite(value) for value in (speed, update))
                or type(item.get("latency_eligible")) is not bool
            ):
                raise ValueError("invalid Play tick values")
            if roster_mode != PLAY_ROSTER_ALL_B8:
                play_ticks_excluded_by_roster += 1
            elif speed == DEFAULT_SPEED and item["latency_eligible"]:
                default_update_ms.setdefault(("play", "", driven_ai), []).append(update)
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
    if not replay_mode:
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
    replay_update_p95_ms = replay_window_p95(replay_summary_batches, replay_raw_updates)
    replay_elapsed_days = (
        None if not replay_timestamps else max(0.0, (current - min(replay_timestamps)) / 86400.0)
    )
    replay_window_complete = bool(
        replay_mode
        and replay_elapsed_days is not None
        and replay_elapsed_days >= REPLAY_WINDOW_DAYS
        and replay_frames >= REPLAY_WINDOW_FRAMES
    )
    if replay_mode and not replay_window_complete:
        faults.append(
            "incomplete Watch replay observation window: "
            "%.3f/%.1f days, %d/%d replay frames"
            % (
                replay_elapsed_days if replay_elapsed_days is not None else 0.0,
                REPLAY_WINDOW_DAYS,
                replay_frames,
                REPLAY_WINDOW_FRAMES,
            )
        )
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
    if replay_mode:
        window_complete = replay_window_complete
    else:
        window_complete = bool(windows_by_roster) and all(
            values["window_complete"] for values in windows_by_roster.values()
        )
    metrics = {
        "decisions": decisions,
        "deaths": deaths,
        "deaths_per_1000": death_rate,
        "deaths_by_roster": deaths_by_roster,
        "latency_by_roster": latency,
        "play_ticks_excluded_by_roster": play_ticks_excluded_by_roster,
        "windows_by_roster": windows_by_roster,
        "fallback_records": len(fallback),
        "ego2s_exceptions": len(exceptions),
        "elapsed_days": elapsed_days,
        "window_complete": window_complete,
        "replay_mode": replay_mode,
        "replay_frames": replay_frames,
        "replay_over_trigger_frames": replay_over_trigger_frames,
        "replay_over_trigger_invalid": replay_over_trigger_invalid,
        "replay_update_p95_ms": replay_update_p95_ms,
        "replay_update_trigger_ms": REPLAY_UPDATE_P95_TRIGGER_MS,
        "replay_step_failures": replay_step_failures,
        "replay_bank_exhaustions": replay_bank_exhaustions,
        "replay_elapsed_days": replay_elapsed_days,
        "replay_window_complete": replay_window_complete,
    }
    triggers: list[str] = []
    if fallback:
        triggers.append("ROLLBACK: recorded served-checkpoint or ego2s fallback reason")
    if replay_step_failures > 0:
        triggers.append("ROLLBACK: recorded Watch replay step failure")
    if exceptions:
        triggers.append("ROLLBACK: recorded ego2s-path exception")
    if replay_update_p95_ms is not None and replay_update_p95_ms > REPLAY_UPDATE_P95_TRIGGER_MS:
        triggers.append(
            "ROLLBACK: Watch replay update p95 %.6f ms exceeds %.2f ms"
            % (replay_update_p95_ms, REPLAY_UPDATE_P95_TRIGGER_MS)
        )
    if replay_bank_exhaustions > 0:
        triggers.append("ROLLBACK: Watch replay bank exhausted")
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
    if metrics["play_ticks_excluded_by_roster"]:
        print(
            "Play ticks excluded by roster (informational only): "
            f"{metrics['play_ticks_excluded_by_roster']}"
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
    if metrics["replay_mode"]:
        print(
            "Watch replay observation window: %.3f/%.1f days, %d/%d frames, "
            "update p95=%s ms (trigger: > %.2f ms), step failures=%d, bank exhaustions=%d; %s"
            % (
                (
                    metrics["replay_elapsed_days"]
                    if metrics["replay_elapsed_days"] is not None
                    else 0.0
                ),
                REPLAY_WINDOW_DAYS,
                metrics["replay_frames"],
                REPLAY_WINDOW_FRAMES,
                (
                    f"{metrics['replay_update_p95_ms']:.4f}"
                    if metrics["replay_update_p95_ms"] is not None
                    else "missing"
                ),
                metrics["replay_update_trigger_ms"],
                metrics["replay_step_failures"],
                metrics["replay_bank_exhaustions"],
                "complete" if metrics["replay_window_complete"] else "incomplete",
            )
        )
    for message in messages:
        print(f"NOT_HEALTHY: {message}")
    print("HEALTHY" if healthy else "NOT_HEALTHY")
    return 0 if healthy else 2


if __name__ == "__main__":
    sys.exit(main())
