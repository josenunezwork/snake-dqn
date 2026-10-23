"""A1 absolute-clock GC-policy qualification probes (Round 25, §d.2).

This is an operator-only Mac measurement.  It uses the real Watch6 constructor,
the M1 absolute clock, and only records evidence; it never changes serving
policy, cadence, or GC thresholds.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from research.bl_gc import common
from research.bl_k1.m1_background import background_summary

SCHEMA = "bl-gc-a1/v1"
WARMUP_TICKS = 200
FPS = 24
POLICIES = frozenset(("default", "freeze_startup"))
HISTOGRAM_BINS = ((0.0, 1.0), (1.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, math.inf))


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * q
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _histogram(values: Iterable[float]) -> dict[str, int]:
    result = {"0-1": 0, "1-5": 0, "5-10": 0, "10-20": 0, "20+": 0}
    labels = tuple(result)
    for value in values:
        for index, (lower, upper) in enumerate(HISTOGRAM_BINS):
            if lower <= value < upper:
                result[labels[index]] += 1
                break
    return result


def _meter() -> Any:
    # The production callback is counters-only.  The probe owns a separate
    # instance so its drain windows are unambiguous and it never retains events.
    from web.backend.gc_policy import GcTickMeter

    meter = GcTickMeter()
    meter.install()
    return meter


def _drain(meter: Any) -> dict[str, Any]:
    row = dict(meter.drain())
    for generation in range(3):
        row.setdefault(f"gc_gen{generation}_count", 0)
        row.setdefault(f"gc_gen{generation}_pause_ms", 0.0)
    row.setdefault("gc_pause_ms", 0.0)
    row.setdefault("gc_freeze_count", gc.get_freeze_count())
    row.setdefault("gc_policy", os.environ.get("SNAKE_SERVE_GC_POLICY", "default"))
    return row


def _timed_tick(session: Any) -> tuple[float, float, Any]:
    """Reuse T15's exact timed tick; wrap only its snapshot observation."""
    from research.t15_mixed_roster_20261017.measurements import _timed_tick as product_timed_tick

    original = session.snapshot
    snapshot_ms = 0.0
    captured = None

    def observed(*args: Any, **kwargs: Any) -> Any:
        nonlocal snapshot_ms, captured
        started = time.perf_counter_ns()
        try:
            captured = original(*args, **kwargs)
            return captured
        finally:
            snapshot_ms = (time.perf_counter_ns() - started) / 1e6

    session.snapshot = observed
    try:
        update_ms, _ = product_timed_tick(session)
    finally:
        session.snapshot = original
    return update_ms, snapshot_ms, captured


def _rss_mb(process: Any) -> float:
    return float(process.memory_info().rss) / (1024 * 1024)


def forbidden_processes() -> list[dict[str, Any]]:
    """Find Codex handoff/exec jobs and local Colab supervisors, excluding app helpers."""
    import psutil

    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = str(process.info["name"] or "").lower()
            argv = process.info["cmdline"] or []
            if (
                any(
                    Path(token).name == "codex" and following == "exec"
                    for token, following in zip(argv, argv[1:])
                )
                or any(
                    "research.colab_cli" in token
                    or "research/colab_cli/" in token
                    or Path(token).name == "codex-handoff.sh"
                    for token in argv
                )
                or (name == "colab" and "run" in argv)
            ):
                found.append({"pid": process.pid, "name": name, "argv": argv})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _observe() -> dict[str, Any]:
    from research.bl_k1.m1 import foreign_load
    from research.bl_mac_latency.host import snapshot

    if forbidden_processes():
        raise RuntimeError("Codex or Colab supervisor present")
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise RuntimeError("A1 requires nice 0")
    return snapshot(process_policy=lambda screen: foreign_load(screen, os.getpid()))


def _timing_admission() -> dict[str, Any]:
    flag = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed")
    if not flag.is_file():
        raise RuntimeError("A1 requires an armed mac-timing-armed waiter")
    idle = _observe()
    if not idle["quiet"]:
        raise RuntimeError("A1 requires a quiet idle census")
    return idle


def _gc_state(process: Any) -> dict[str, Any]:
    return {
        "monotonic_ns": time.monotonic_ns(),
        "rss_mb": _rss_mb(process),
        "tracked_objects": len(gc.get_objects()),
        "gc_freeze_count": gc.get_freeze_count(),
        "enabled": gc.isenabled(),
        "thresholds": list(gc.get_threshold()),
    }


def _tick_row(session: Any, meter: Any, epoch: int, index: int) -> dict[str, Any]:
    # Clock copied from research/bl_k1/m1.py:992-1081. There is no epoch rebase,
    # relative tick sleep, skipped tick, or clock rounding before multiplication.
    target = epoch + index * 1_000_000_000 // 24
    now = time.monotonic_ns()
    if now < target:
        time.sleep((target - now) / 1e9)
    tick_started = time.monotonic_ns()
    prior_dead = {snake.id for snake in session.game.snakes if not snake.is_alive}
    update_ms, snapshot_ms, _snapshot = _timed_tick(session)
    gc_row = _drain(meter)
    respawns = sum(
        snake.id in prior_dead and (snake.is_alive or snake.id in session.game.frame_death_causes)
        for snake in session.game.snakes
    )
    finished = time.monotonic_ns()
    return {
        "index": index,
        "target_ns": target,
        "tick_started_ns": tick_started,
        "finished_ns": finished,
        "update_ms": update_ms,
        "update_seconds": update_ms / 1000,
        "snapshot_ms": snapshot_ms,
        "overrun": finished >= epoch + (index + 1) * 1_000_000_000 // 24,
        "late_ms_target": (finished - target) / 1e6,
        "late_ms_deadline": max(0, finished - (epoch + (index + 1) * 1_000_000_000 // 24)) / 1e6,
        "deaths": len(session.game.frame_death_causes),
        "respawns": respawns,
        "food_count": len(session.game.food_manager.food),
        "hero_length": int(session.game.snakes[0].length),
        "total_segments": sum(len(snake.segments) for snake in session.game.snakes),
        **gc_row,
        "gc_attributed": bool(gc_row["gc_pause_ms"] >= 20 or gc_row["gc_gen2_count"] > 0),
    }


def run(policy: str, seed: int, ticks: int) -> dict[str, Any]:
    """Record one bounded cell; no verdict or automatic retry is produced here."""
    import psutil

    from research.bl_k1.m1 import HostMonitor, new_watch

    if policy not in POLICIES or (seed, ticks) not in {
        (2026102300, 2400),
        (2026102303, 2400),
        (2026102304, 14400),
    }:
        raise ValueError("cell differs from frozen A1 allocation")
    admission = _timing_admission()
    started = time.monotonic()
    deadline = started + ticks / FPS + 180
    receipt, initial = common.initialize(policy, stage="probe")
    session, roster = new_watch(seed)
    process = psutil.Process()
    meter = _meter()
    monitor = HostMonitor(_observe)
    rows: list[dict[str, Any]] = []
    reason = None
    start_state = end_state = None
    final_gc: dict[str, Any] = {}
    epoch = 0
    try:
        for _ in range(WARMUP_TICKS):
            _timed_tick(session)
        start_state = _gc_state(process)  # tick 200, outside the scored clock.
        meter.drain()
        monitor.thread.start()
        epoch = time.monotonic_ns()
        last_rss_ns = epoch - 1_000_000_000
        for index in range(ticks):
            if monitor.error or time.monotonic() >= deadline:
                reason = monitor.error or "mechanical wall cap"
                break
            row = _tick_row(session, meter, epoch, index)
            rows.append(row)
            boundary = epoch + (index + 1) * 1_000_000_000 // 24
            # Census allocation is excluded only when actually in the sleep
            # slack. Its GC is retained by the meter through the next drain.
            if index % 240 == 0 and time.monotonic_ns() < boundary:
                row["tracked_objects"] = len(gc.get_objects())
                row["tracked_sample_outside_scored_interval"] = True
            if (
                time.monotonic_ns() - last_rss_ns >= 1_000_000_000
                and time.monotonic_ns() < boundary
            ):
                row["rss_mb"] = _rss_mb(process)
                row["rss_sample_ns"] = time.monotonic_ns()
                row["rss_sample_outside_scored_interval"] = True
                last_rss_ns = row["rss_sample_ns"]
            if index and index % (FPS * 60) == 0:
                print(json.dumps({"seed": seed, "policy": policy, "ticks": index}), flush=True)
        # Keep the callback live through the final sleep, rather than silently
        # losing snapshot/disclosure/slack GC at the last boundary.
        boundary = epoch + len(rows) * 1_000_000_000 // 24
        now = time.monotonic_ns()
        if now < boundary:
            time.sleep((boundary - now) / 1e9)
        final_gc = meter.drain()
        end_state = _gc_state(process)
        reason = reason or monitor.error
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if monitor.thread.ident is not None:
            monitor.close()
        host_finished_ns = time.monotonic_ns()
        meter.close()
        session.close(wait=True)
    return {
        "schema": SCHEMA,
        "status": "INCOMPLETE" if reason else "COMPLETE",
        "mechanical_reason": reason,
        "policy": policy,
        "seed": seed,
        "ticks_requested": ticks,
        "warmup_ticks": WARMUP_TICKS,
        "fps": FPS,
        "epoch_ns": epoch,
        "clock": "m1.py:992-1081 absolute epoch+i*1e9//24",
        "gc_policy_receipt": receipt,
        "attestation": {**initial, "roster": roster},
        "timing_admission": admission,
        "host_observations": monitor.rows,
        "background_load": background_summary([(monitor.rows, host_finished_ns)]),
        "roster": roster,
        "ticks": rows,
        "final_boundary_gc": final_gc,
        "start_state": start_state,
        "end_state": end_state,
        "gc_enabled_end": gc.isenabled(),
        "thresholds_end": list(gc.get_threshold()),
        "freeze_count_end": gc.get_freeze_count(),
        "retry_rule": "one fresh mechanical rerun only after verified stop; never automatic",
    }


def _block_table(rows: Sequence[dict[str, Any]], block: int = 400) -> list[dict[str, Any]]:
    return [
        {
            "start": index,
            "end": index + len(piece) - 1,
            "update_p50_ms": _percentile([float(row["update_ms"]) for row in piece], 0.5),
            "update_p95_ms": _percentile([float(row["update_ms"]) for row in piece], 0.95),
            "hero_length_p50": _percentile([float(row["hero_length"]) for row in piece], 0.5),
        }
        for index in range(0, len(rows), block)
        if (piece := rows[index : index + block])
    ]


def _validate_cell(cell: dict[str, Any]) -> str | None:
    """Reject incomplete or foreign evidence before applying any scientific bar."""
    try:
        rows, receipt = cell["ticks"], cell["gc_policy_receipt"]
        if (
            cell["schema"] != SCHEMA
            or cell["status"] != "COMPLETE"
            or cell["mechanical_reason"] is not None
            or len(rows) != cell["ticks_requested"]
            or cell["warmup_ticks"] != 200
            or cell["fps"] != 24
            or receipt["policy"] != cell["policy"]
            or receipt["stage"] != "probe"
        ):
            return "partial, foreign, or mismatched cell metadata"
        attest = cell["attestation"]
        if (
            attest["torch_num_threads"] != 1
            or attest["pathb"] != "0"
            or not cell["timing_admission"]["quiet"]
            or not cell["host_observations"]
            or cell["roster"]["watch_roster"] != "mixed"
            or cell["roster"]["served"]["name"] != "b8-s3"
            or len(cell["roster"]["snakes"]) != 6
            or not cell["mac_lease"]
        ):
            return "missing d.0 or exclusive-host evidence"
        if cell["policy"] == "freeze_startup" and receipt["frozen_count"] <= 0:
            return "freeze_startup has no frozen objects"
        final_gc = cell["final_boundary_gc"]
        for row in [*rows, final_gc]:
            if (
                row["gc_policy"] != cell["policy"]
                or type(row["gc_freeze_count"]) is not int
                or row["gc_freeze_count"] < 0
            ):
                return "invalid GC policy/counter"
            total_pause = 0.0
            for generation in range(3):
                count, pause = row[f"gc_gen{generation}_count"], row[f"gc_gen{generation}_pause_ms"]
                hist = row["gc_histograms"][f"gen{generation}"]
                if (
                    type(count) is not int
                    or count < 0
                    or not math.isfinite(pause)
                    or pause < 0
                    or set(hist) != {"0-1", "1-5", "5-10", "10-20", "20+"}
                    or any(type(v) is not int or v < 0 for v in hist.values())
                    or sum(hist.values()) != count
                ):
                    return "invalid final/tick GC counters"
                total_pause += pause
            if (
                not math.isfinite(row["gc_pause_ms"])
                or abs(total_pause - row["gc_pause_ms"]) > 1e-6
            ):
                return "GC total pause does not reconcile"
        epoch = cell["epoch_ns"]
        for index, row in enumerate(rows):
            target = epoch + index * 1_000_000_000 // 24
            deadline = epoch + (index + 1) * 1_000_000_000 // 24
            if (
                row["index"] != index
                or row["target_ns"] != target
                or row["tick_started_ns"] < target
                or row["finished_ns"] < row["tick_started_ns"]
                or type(row["overrun"]) is not bool
                or row["overrun"] != (row["finished_ns"] >= deadline)
                or row["gc_policy"] != cell["policy"]
            ):
                return "noncontiguous or inconsistent absolute clock"
            numeric = ["update_ms", "snapshot_ms", "gc_pause_ms", "late_ms_deadline"]
            numeric += [f"gc_gen{g}_pause_ms" for g in range(3)]
            for key in numeric:
                if (
                    type(row[key]) not in (float, int)
                    or not math.isfinite(row[key])
                    or row[key] < 0
                ):
                    return "non-finite or negative scored metric"
            for g in range(3):
                count = row[f"gc_gen{g}_count"]
                histogram = row["gc_histograms"][f"gen{g}"]
                if (
                    type(count) is not int
                    or count < 0
                    or set(histogram) != {"0-1", "1-5", "5-10", "10-20", "20+"}
                    or any(type(v) is not int or v < 0 for v in histogram.values())
                    or sum(histogram.values()) != count
                ):
                    return "GC histogram/counter mismatch"
        for row in rows:
            if "rss_mb" in row and (
                not math.isfinite(row["rss_mb"])
                or row["rss_mb"] < 0
                or type(row["rss_sample_ns"]) is not int
                or row["rss_sample_outside_scored_interval"] is not True
                or not row["finished_ns"]
                <= row["rss_sample_ns"]
                < epoch + (row["index"] + 1) * 1_000_000_000 // 24
            ):
                return "invalid RSS sample"
        for key in ("start_state", "end_state"):
            state = cell[key]
            for field in ("rss_mb", "tracked_objects", "gc_freeze_count", "monotonic_ns"):
                if not math.isfinite(state[field]) or state[field] < 0:
                    return "invalid memory endpoint"
        return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return "missing or malformed qualification field"


def check_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Apply frozen bars, recomputing attribution from the measured counters."""
    fault = _validate_cell(cell)
    if fault:
        return {
            "verdict": "INCOMPLETE",
            "reason": fault,
            "background_load": cell.get("background_load"),
        }
    rows = cell["ticks"]
    overruns = [row for row in rows if row["overrun"]]
    gc_overruns = [row for row in overruns if row["gc_pause_ms"] >= 20 or row["gc_gen2_count"] > 0]
    non_gc_count = len(overruns) - len(gc_overruns)
    overrun_verdict = "FAIL" if gc_overruns else "GC_FIXED+BAR_FAIL(non-GC)" if overruns else "PASS"
    updates = [float(row["update_ms"]) for row in rows]
    all_gc = [*rows, cell["final_boundary_gc"]]
    pauses = [float(row["gc_pause_ms"]) for row in all_gc]
    violations = []
    if gc_overruns:
        violations.append("GC-attributed overrun")
    p95_failure = _percentile(updates, 0.95) > 11.688
    if p95_failure:
        violations.append("BAR_FAIL(p95, non-GC)" if not gc_overruns else "update p95 > 11.688 ms")
    if any(pause >= 20 for pause in pauses):
        violations.append("gc_pause_ms >= 20")
    receipt = cell["gc_policy_receipt"]
    baseline = receipt["frozen_count"]
    # Chronological observations, including the drain after the final sleep.
    counts = [cell["start_state"]["gc_freeze_count"]]
    counts += [r["gc_freeze_count"] for r in all_gc]
    counts.append(cell["end_state"]["gc_freeze_count"])
    if any(
        current > previous or current > baseline
        for previous, current in zip([baseline, *counts], counts)
    ):
        violations.append("gc_freeze_count is not non-increasing since startup")
    if any(
        not state["enabled"] or state["thresholds"] != [700, 10, 10]
        for state in [receipt, cell["start_state"], cell["end_state"]]
    ):
        violations.append("GC enabled/threshold invariant failed")
    consecutive = max_consecutive = 0
    for row in rows:
        consecutive = consecutive + 1 if row["overrun"] else 0
        max_consecutive = max(max_consecutive, consecutive)
    structural = [v for v in violations if v not in ("BAR_FAIL(p95, non-GC)",)]
    verdict = "FAIL" if structural else "BAR_FAIL(p95, non-GC)" if p95_failure else overrun_verdict
    gen2_ticks = [i for i, row in enumerate(rows) if row["gc_gen2_count"]]
    return {
        "cell": {key: cell[key] for key in ("policy", "seed", "ticks_requested")},
        "verdict": verdict,
        "overrun_verdict": overrun_verdict,
        "background_load": cell.get("background_load"),
        "violations": violations,
        "overruns": len(overruns),
        "gc_attributed_overruns": len(gc_overruns),
        "non_gc_overruns": non_gc_count,
        "update_ms": _summary(updates),
        "gc_pause_histogram_ms": {
            f"gen{g}": {
                label: sum(r["gc_histograms"][f"gen{g}"][label] for r in all_gc)
                for label in ("0-1", "1-5", "5-10", "10-20", "20+")
            }
            for g in range(3)
        },
        "gc_freeze_count_total_decrease": counts[0] - counts[-1],
        "gc_freeze_count_decrease_since_startup": baseline - counts[-1],
        "gc_gen2_count": sum(r["gc_gen2_count"] for r in all_gc),
        "gc_gen2_intervals": [b - a for a, b in zip(gen2_ticks, gen2_ticks[1:])],
        "gc_pause_max_ms": max(pauses),
        "max_consecutive_overruns": max_consecutive,
        "deaths": sum(row["deaths"] for row in rows),
        "respawns": sum(row["respawns"] for row in rows),
        "food_count": _summary([row["food_count"] for row in rows]),
        "lateness_max_ms": max(row["late_ms_deadline"] for row in rows),
        "block_table": _block_table(rows),
    }


def _memory_check(default: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    epoch = frozen["epoch_ns"]
    end_ns = epoch + 14400 * 1_000_000_000 // 24
    samples = [
        (row["rss_sample_ns"], row["rss_mb"])
        for row in frozen["ticks"]
        if "rss_mb" in row and row.get("rss_sample_outside_scored_interval") is True
    ]
    last_five = [
        (float(ns - epoch) / 60e9, float(value))
        for ns, value in samples
        if end_ns - 300_000_000_000 <= ns <= end_ns
    ]
    if len(last_five) < 250 or last_five[-1][0] - last_five[0][0] < 4.8:
        return {
            "pass": False,
            "incomplete": True,
            "reason": "missing last-five-minute 1 Hz RSS coverage",
        }
    mean_x = statistics.fmean(x for x, _ in last_five)
    mean_y = statistics.fmean(y for _, y in last_five)
    denominator = sum((x - mean_x) ** 2 for x, _ in last_five)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in last_five) / denominator
    delta = frozen["end_state"]["rss_mb"] - frozen["start_state"]["rss_mb"]
    default_delta = default["end_state"]["rss_mb"] - default["start_state"]["rss_mb"]
    tracked = frozen["end_state"]["tracked_objects"] - frozen["start_state"]["tracked_objects"]
    return {
        "pass": slope <= 1 and delta <= default_delta + 32 and tracked <= 50000,
        "incomplete": False,
        "rss_slope_mib_per_min": slope,
        "rss_delta_mib": delta,
        "default_rss_delta_mib": default_delta,
        "tracked_objects_delta": tracked,
    }


def check(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Require exactly the six registered paired cells and apply all A1 bars."""
    if len(cells) != 6:
        raise ValueError("A1 requires exactly six cells")
    keys = [
        (str(cell.get("policy")), int(cell.get("seed", -1)), int(cell.get("ticks_requested", -1)))
        for cell in cells
    ]
    if len(set(keys)) != 6 or any(policy not in POLICIES for policy, _, _ in keys):
        raise ValueError("A1 cells must have unique supported policy/seed/tick identities")
    by_ticks = {
        ticks: [cell for cell in cells if int(cell["ticks_requested"]) == ticks]
        for ticks in (2400, 14400)
    }
    if set(by_ticks) != {2400, 14400} or len(by_ticks[2400]) != 4 or len(by_ticks[14400]) != 2:
        raise ValueError("A1 requires four 2400-tick and two 14400-tick cells")
    expected = {2400: [2026102300, 2026102303], 14400: [2026102304]}
    for ticks, group in by_ticks.items():
        policies = {str(cell["policy"]) for cell in group}
        by_policy = {
            policy: sorted(int(cell["seed"]) for cell in group if cell["policy"] == policy)
            for policy in POLICIES
        }
        if policies != POLICIES or by_policy["default"] != by_policy["freeze_startup"]:
            raise ValueError(
                f"{ticks}-tick A1 cells must be paired default/freeze on identical seeds"
            )
        if by_policy["default"] != expected[ticks]:
            raise ValueError(f"{ticks}-tick A1 seeds differ from the frozen allocation")
    bindings = [cell.get("attestation", {}) for cell in cells]
    identity_keys = ("judgment_sha256", "interpreter", "version", "git_commit", "source_hashes")
    for key in identity_keys:
        if not bindings[0].get(key) or any(
            value.get(key) != bindings[0][key] for value in bindings
        ):
            return {
                "schema": SCHEMA + "/verdict",
                "verdict": "INCOMPLETE",
                "reason": f"mismatched or missing d.0 binding: {key}",
            }
    if bindings[0]["judgment_sha256"] != common.JUDGMENT_SHA256:
        return {
            "schema": SCHEMA + "/verdict",
            "verdict": "INCOMPLETE",
            "reason": "foreign judgment",
        }
    summaries = [check_cell(cell) for cell in cells]
    if any(summary.get("verdict") == "INCOMPLETE" for summary in summaries):
        return {"schema": SCHEMA + "/verdict", "verdict": "INCOMPLETE", "cells": summaries}
    long_default = next(cell for cell in by_ticks[14400] if cell["policy"] == "default")
    long_frozen = next(cell for cell in by_ticks[14400] if cell["policy"] == "freeze_startup")
    memory = _memory_check(long_default, long_frozen)
    frozen_summaries = [x for x in summaries if x["cell"]["policy"] == "freeze_startup"]
    if memory.get("incomplete"):
        verdict = "INCOMPLETE"
    elif not memory["pass"] or any(x["verdict"] == "FAIL" for x in frozen_summaries):
        verdict = "FAIL"
    elif any(x["verdict"] == "BAR_FAIL(p95, non-GC)" for x in frozen_summaries):
        verdict = "BAR_FAIL(p95, non-GC)"
    elif any(x["overrun_verdict"] == "GC_FIXED+BAR_FAIL(non-GC)" for x in frozen_summaries):
        verdict = "GC_FIXED+BAR_FAIL(non-GC)"
    else:
        verdict = "PASS"
    return {"schema": SCHEMA + "/verdict", "verdict": verdict, "cells": summaries, "memory": memory}


def _load(path: Path) -> dict[str, Any]:
    from research.bl_served_search.exports import verify_export

    verify_export(path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"not an A1 cell: {path}")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.policy, args.seed, args.ticks)):
            parser.error("--check cannot be combined with a run cell")
        result = check([_load(path) for path in args.check])
    else:
        if args.policy is None or args.seed is None or args.ticks is None:
            parser.error("a cell requires --policy, --seed, and --ticks")
        from research.mac_lease import held

        # Refuse reuse even after a crash, before admitting a session or clock.
        for candidate in (
            args.out,
            Path(str(args.out) + ".sha256"),
            Path(str(args.out) + ".complete"),
            Path(str(args.out) + ".started"),
        ):
            if candidate.exists():
                raise FileExistsError(f"fresh output required: {candidate}")
        _timing_admission()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with held(
            "bl-gc-a1", "Round-25 A1 absolute-clock probe", seconds=args.ticks / FPS + 240
        ) as lease:
            with Path(str(args.out) + ".started").open("x") as marker:
                json.dump(
                    {
                        "policy": args.policy,
                        "seed": args.seed,
                        "ticks": args.ticks,
                        "lease": lease.receipt,
                    },
                    marker,
                )
            result = run(args.policy, args.seed, args.ticks)
            result["mac_lease"] = lease.receipt
    common.write_result(args.out, result)
    print(
        json.dumps(
            {"out": str(args.out), "verdict": result.get("verdict", result.get("status"))},
            sort_keys=True,
        )
    )
    return 0 if result.get("verdict", result.get("status")) in ("PASS", "COMPLETE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
