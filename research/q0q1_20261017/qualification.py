#!/usr/bin/env python3
"""Fail-closed Q0b hardware-packet and Q1 budget admission checks.

This module evaluates reported evidence only.  It does not time workloads, start
jobs, or authorize queue dispatch.  In particular, a DRAFT_PENDING_OWNER_REVIEW
registration remains non-dispatchable even when its packet is qualified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


PACKET_SCHEMA = "q0q1-qualification-packet/v1"
PLAN_SCHEMA = "q0q1-study-budget/v1"
SCHEDULES = {"restart", "mature"}
COLLECTORS = {"n1", "n3"}
SHA256_LEN = 64
ROWS_PER_UPDATE = 256
WINDOW_SECONDS = 180.0
WARMUP_SECONDS = 120.0
PACKET_SECONDS = 3600.0


def _finite_number(value: Any, label: str, errors: list[str], *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{label} must be a finite number")
        return None
    number = float(value)
    if positive and number <= 0:
        errors.append(f"{label} must be > 0")
    return number


def _mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    return value


def _hash_bundle(packet: Mapping[str, Any], errors: list[str]) -> None:
    """Require independently bound source, config, environment, and input hashes."""
    for name in ("source", "config", "environment", "hashes"):
        bundle = _mapping(packet.get(name), name, errors)
        if not bundle:
            errors.append(f"{name} must contain at least one SHA-256 binding")
            continue
        for key, value in bundle.items():
            if not isinstance(key, str) or not key:
                errors.append(f"{name} contains an invalid binding name")
            if not isinstance(value, str) or len(value) != SHA256_LEN or any(
                char not in "0123456789abcdef" for char in value.lower()
            ):
                errors.append(f"{name}.{key} must be a lowercase/uppercase SHA-256 hex digest")


def _check_cell(cell: Mapping[str, Any], errors: list[str], checks: dict[str, bool]) -> dict[str, float]:
    name = f"{cell.get('schedule', '?')}/{cell.get('collector', '?')}"
    result: dict[str, float] = {}
    if cell.get("schedule") not in SCHEDULES or cell.get("collector") not in COLLECTORS:
        errors.append(f"cell {name} must name schedule restart|mature and collector n1|n3")
    if not isinstance(cell.get("actor_count"), int) or cell["actor_count"] <= 0:
        errors.append(f"cell {name}.actor_count must be a positive integer")
    warmup = _finite_number(cell.get("warmup_seconds"), f"cell {name}.warmup_seconds", errors)
    if warmup is not None and warmup < WARMUP_SECONDS:
        errors.append(f"cell {name} warmed for less than {WARMUP_SECONDS:g} seconds")
    if cell.get("compile_ready") is not True or cell.get("replay_ready") is not True:
        errors.append(f"cell {name} lacks completed compilation or replay readiness")
    windows = cell.get("scored_windows")
    if not isinstance(windows, list) or len(windows) != 2:
        errors.append(f"cell {name} requires exactly two scored windows")
        windows = []
    throughputs: list[float] = []
    for index, window in enumerate(windows, 1):
        item = _mapping(window, f"cell {name}.scored_windows[{index}]", errors)
        if item is None:
            continue
        duration = _finite_number(item.get("seconds"), f"cell {name}.window{index}.seconds", errors)
        decisions = _finite_number(item.get("learner_decisions"), f"cell {name}.window{index}.learner_decisions", errors)
        if duration is not None and duration != WINDOW_SECONDS:
            errors.append(f"cell {name}.window{index} must be exactly {WINDOW_SECONDS:g} seconds")
        if duration and decisions is not None and decisions > 0:
            throughputs.append(decisions / duration)
    if len(throughputs) == 2:
        if throughputs[1] < throughputs[0] * 0.9:
            errors.append(f"cell {name} throughput declined by more than 10% between scored windows")
        result["decisions_per_second"] = sum(throughputs) / len(throughputs)
    for field in ("clean_exit", "finite_updates"):
        if cell.get(field) is not True:
            errors.append(f"cell {name}.{field} must be true")
    for field in ("dropped_transitions", "ring_overruns"):
        value = cell.get(field)
        if not isinstance(value, int) or value != 0:
            errors.append(f"cell {name}.{field} must be zero")
    counters = _mapping(cell.get("counters"), f"cell {name}.counters", errors)
    if counters is not None:
        rows = _finite_number(counters.get("emitted_replay_rows"), f"cell {name}.counters.emitted_replay_rows", errors)
        updates = _finite_number(counters.get("completed_updates"), f"cell {name}.counters.completed_updates", errors)
        debt = _finite_number(counters.get("pending_update_debt"), f"cell {name}.counters.pending_update_debt", errors)
        decisions = _finite_number(counters.get("actual_learner_decisions"), f"cell {name}.counters.actual_learner_decisions", errors)
        if rows is not None and updates is not None and debt is not None and rows > 0:
            reconciliation = abs(rows - (ROWS_PER_UPDATE * updates + debt)) / rows
            if reconciliation > 0.02:
                errors.append(f"cell {name} R/U/debt reconciliation exceeds 2%")
            result["cadence_error"] = reconciliation
        if decisions is not None and decisions <= 0:
            errors.append(f"cell {name} must record actual learner decisions")
    lag = _mapping(cell.get("policy_lag"), f"cell {name}.policy_lag", errors)
    if lag is not None:
        control = _mapping(lag.get("matched_control"), f"cell {name}.policy_lag.matched_control", errors)
        values: dict[str, float] = {}
        controls: dict[str, float] = {}
        for field in ("mean", "p95", "age_p95"):
            value = _finite_number(lag.get(field), f"cell {name}.policy_lag.{field}", errors)
            baseline = _finite_number(control.get(field) if control else None, f"cell {name}.policy_lag.matched_control.{field}", errors)
            if value is not None:
                values[field] = value
            if baseline is not None:
                controls[field] = baseline
        if len(values) == len(controls) == 3:
            if values["mean"] > controls["mean"] or values["p95"] > controls["p95"]:
                errors.append(f"cell {name} has an unexplained policy-lag increase over its matched control")
            if values["age_p95"] > controls["age_p95"] * 1.1:
                errors.append(f"cell {name} ingestion age p95 exceeds matched control by more than 10%")
    checks[name] = not any(f"cell {name}" in error for error in errors)
    return result


def validate_packet(packet: Any) -> dict[str, Any]:
    """Validate one bounded L4 or A100 Q0b qualification packet."""
    errors: list[str] = []
    checks: dict[str, bool] = {}
    if not isinstance(packet, Mapping):
        return {"status": "REFUSED", "errors": ["packet root must be an object"], "checks": checks}
    if packet.get("schema") != PACKET_SCHEMA:
        errors.append(f"packet schema must equal {PACKET_SCHEMA}")
    if packet.get("hardware") not in {"L4", "A100"}:
        errors.append("packet hardware must be L4 or A100")
    _hash_bundle(packet, errors)
    elapsed = _finite_number(packet.get("packet_seconds"), "packet_seconds", errors, positive=True)
    if elapsed is not None and elapsed > PACKET_SECONDS:
        errors.append("packet exceeds the 3600-second Q0b cap")
    rate = _finite_number(packet.get("unit_rate_per_hour"), "unit_rate_per_hour", errors, positive=True)
    cells = packet.get("cells")
    results: dict[str, dict[str, float]] = {}
    if not isinstance(cells, list) or len(cells) != 4:
        errors.append("packet requires exactly four restart/mature × n1/n3 cells")
    else:
        seen: set[tuple[str, str]] = set()
        actors: dict[str, set[int]] = {"n1": set(), "n3": set()}
        accounted_seconds = 0.0
        for raw in cells:
            cell = _mapping(raw, "cell", errors)
            if cell is None:
                continue
            schedule, collector = cell.get("schedule"), cell.get("collector")
            if isinstance(schedule, str) and isinstance(collector, str):
                seen.add((schedule, collector))
                results[f"{schedule}/{collector}"] = _check_cell(cell, errors, checks)
                if collector in actors and isinstance(cell.get("actor_count"), int):
                    actors[collector].add(cell["actor_count"])
            warmup = cell.get("warmup_seconds")
            if isinstance(warmup, (int, float)) and not isinstance(warmup, bool) and math.isfinite(warmup):
                accounted_seconds += float(warmup)
            windows = cell.get("scored_windows")
            if isinstance(windows, list):
                for window in windows:
                    if isinstance(window, Mapping):
                        seconds = window.get("seconds")
                        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds):
                            accounted_seconds += float(seconds)
        expected = {(schedule, collector) for schedule in SCHEDULES for collector in COLLECTORS}
        if seen != expected:
            errors.append("packet cells must contain each restart/mature × n1/n3 combination exactly once")
        for collector, counts in actors.items():
            if len(counts) != 1:
                errors.append(f"{collector} must use one predeclared actor count across restart and mature")
        if elapsed is not None and accounted_seconds > elapsed:
            errors.append("packet_seconds cannot cover its declared warm-up and scored-window wall time")
    parity = _mapping(packet.get("parity"), "parity", errors)
    if parity is not None:
        if parity.get("target_runtime") != "cuda":
            errors.append("target-runtime parity must be CUDA, not Mac-only")
        for field in ("short_body", "long_body"):
            if parity.get(field) is not True:
                errors.append(f"{field} fixture parity must pass outside timed windows")
        for field in ("t22", "t24"):
            if parity.get(field) != "PASS":
                errors.append(f"{field} target-CUDA parity must PASS")
    if rate is not None:
        checks["measured_rate"] = True
    return {
        "status": "QUALIFIED" if not errors else "INCOMPLETE",
        "errors": errors,
        "checks": checks,
        "measured_cells": results,
        "unit_rate_per_hour": rate,
    }


def validate_admission(packet: Any, plan: Any) -> dict[str, Any]:
    """Compute a complete Q1 forecast and refuse incomplete or over-cap plans."""
    outcome = validate_packet(packet)
    errors = list(outcome["errors"])
    if not isinstance(plan, Mapping):
        return {**outcome, "status": "REFUSED", "errors": errors + ["study plan root must be an object"]}
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append(f"study plan schema must equal {PLAN_SCHEMA}")
    draft_status = plan.get("owner_review_status", plan.get("status"))
    if draft_status != "DRAFT_PENDING_OWNER_REVIEW":
        errors.append("study plan must explicitly remain DRAFT_PENDING_OWNER_REVIEW")
    if plan.get("dispatch") is not False:
        errors.append("qualification must not authorize dispatch")
    if packet.get("hardware") != "L4" or plan.get("q1_training_hardware") != "L4":
        errors.append("Q1 training admission requires one exact qualified L4 cohort")
    q0_units = _finite_number(plan.get("q0_units"), "q0_units", errors, positive=True)
    if q0_units is not None and q0_units > 15:
        errors.append("Q0 exceeds its 15-unit cap")
    cpu_rate = _finite_number(plan.get("cpu_unit_rate_per_hour"), "cpu_unit_rate_per_hour", errors, positive=True)
    cpu_seconds = _finite_number(plan.get("cpu_seconds_per_episode"), "cpu_seconds_per_episode", errors, positive=True)
    workers = _finite_number(plan.get("cpu_workers"), "cpu_workers", errors, positive=True)
    efficiency = _finite_number(plan.get("cpu_worker_efficiency"), "cpu_worker_efficiency", errors, positive=True)
    if efficiency is not None and efficiency > 1:
        errors.append("cpu_worker_efficiency cannot exceed 1")
    episodes = plan.get("evaluation_episodes")
    if episodes != 3200:
        errors.append("Q1 admission requires all 3,200 evaluation episodes")
    runs_per_arm = plan.get("runs_per_arm")
    decisions_per_run = plan.get("decisions_per_run")
    if runs_per_arm != 5 or decisions_per_run != 8_000_000:
        errors.append("Q1 admission requires 5 fresh seeds per arm and 8,000,000 decisions per run")
    measured = outcome.get("measured_cells", {})
    train_hours = 0.0
    if not errors and isinstance(measured, Mapping):
        arm_cells = {"A": "restart/n3", "B": "mature/n3", "C": "restart/n1", "D": "mature/n1"}
        for cell_name in arm_cells.values():
            speed = measured.get(cell_name, {}).get("decisions_per_second") if isinstance(measured.get(cell_name), Mapping) else None
            if not isinstance(speed, (int, float)) or speed <= 0:
                errors.append(f"missing measured throughput for {cell_name}")
                continue
            train_hours += 5 * 8_000_000 / float(speed) / 3600.0
    rate = outcome.get("unit_rate_per_hour")
    training_units = train_hours * float(rate) if isinstance(rate, (int, float)) else None
    if training_units is not None and training_units > 60:
        errors.append("Q1 training forecast exceeds its 60-unit cap")
    cpu_hours = 0.0
    if None not in (cpu_seconds, workers, efficiency):
        cpu_hours = 3200 * float(cpu_seconds) / (3600 * float(workers) * float(efficiency))
    cpu_units = cpu_hours * float(cpu_rate) if cpu_rate is not None else None
    total_units = (float(q0_units) if q0_units is not None else 0.0) + (training_units or 0.0) + (cpu_units or 0.0)
    if train_hours > 24 * 0.8:
        errors.append("GPU forecast consumes more than the 80% usable 24 GPU-hour envelope")
    if cpu_hours > 40 * 0.8:
        errors.append("CPU forecast consumes more than the 80% usable 40 CPU-hour envelope")
    if total_units > 150 * 0.8:
        errors.append("total forecast consumes more than the 80% usable 150-unit envelope")
    return {
        "status": "ADMITTED_FOR_OWNER_REVIEW" if not errors else "REFUSED",
        "errors": errors,
        "qualification": outcome,
        "forecast": {
            "q1_training_gpu_hours": train_hours,
            "q1_training_units": training_units,
            "q1_evaluation_cpu_hours": cpu_hours,
            "q1_evaluation_units": cpu_units,
            "total_units_including_q0": total_units,
            "reserve_fraction": 0.20,
        },
        "dispatch_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path, help="Q0b packet JSON")
    parser.add_argument("--study-plan", type=Path, help="Q1 budget/admission JSON")
    parser.add_argument("--json-output", type=Path, help="write result JSON in addition to stdout")
    args = parser.parse_args(argv)
    try:
        packet = json.loads(args.receipt.read_text())
        result = validate_admission(packet, json.loads(args.study_plan.read_text())) if args.study_plan else validate_packet(packet)
    except (OSError, json.JSONDecodeError) as exc:
        result = {"status": "REFUSED", "errors": [str(exc)]}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.write_text(rendered)
    sys.stdout.write(rendered)
    return 0 if result["status"] in {"QUALIFIED", "ADMITTED_FOR_OWNER_REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
