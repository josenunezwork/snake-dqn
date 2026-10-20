"""Bounded M2/M3 T34 measurements on the recorded T3 dense ceiling bank.

This module deliberately measures reconstruction and branch-search work.  It
does not play an assisted policy or produce an efficacy estimate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.t1_commitment_20261015.replay import Replay, build_replay, configure_cpu, load_inputs
from research.t3_rollout_labels_20261015.engine import batched_rollout
from research.t34_shadow.common import ARTIFACTS, Budget, provenance, sha, write_once
from research.t34_shadow.lad import search_lad, search_lad_teacher, search_lad_truth

T3_DENSE = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/dense")
T3_CEILING = T3_DENSE / "ceiling"
T3_SUMMARY = T3_DENSE / "ceiling-go-for-bank.json"
T1_ROWS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/t1-commitment-20261015/rows")
HORIZON = 5_000


def percentile(values: list[float], q: float) -> float | None:
    """Return the deterministic linear percentile, or None for no samples."""
    return float(np.percentile(values, q)) if values else None


def timing_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize observed one-thread latency samples without inventing samples."""
    return {
        "count": len(values),
        "p50_seconds": percentile(values, 50),
        "p95_seconds": percentile(values, 95),
        "p99_seconds": percentile(values, 99),
    }


def source_receipt(world: str, ceiling: Path = T3_CEILING) -> dict[str, Any]:
    """Load one immutable T3 ceiling receipt by its ``arm_env`` world name."""
    path = ceiling / f"{world}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing T3 ceiling receipt: {path}")
    row = json.loads(path.read_text())
    arm, env_text = world.rsplit("_", 1)
    if row.get("arm") != arm or int(row.get("env")) != int(env_text):
        raise ValueError(f"receipt identity mismatch: {path}")
    return row


def world_names(ceiling: Path = T3_CEILING) -> list[str]:
    """List the fixed 64 T3 dense worlds in deterministic order."""
    names = sorted(path.stem for path in ceiling.glob("b8_s?_*.json"))
    if len(names) != 64:
        raise RuntimeError(f"T3 dense-bank inventory drift: expected 64 worlds, found {len(names)}")
    return names


def changed_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return T3's historical interventions, never confusing survival with one."""
    return [trigger for trigger in row["trigger_rows"] if trigger["chosen"] != trigger["greedy"]]


def _greedy(replay: Replay) -> tuple[int, np.ndarray, list[int]]:
    q = replay.q()
    legal = [int(a) for a in np.flatnonzero(replay.sim.get_advisory_action_mask()[0, 0])]
    if not legal:
        legal = [0, 1, 2]
    choice = max(legal, key=lambda action: (float(q[action]), -action))
    return choice, q, legal


def reconstruct_sources(
    row: dict[str, Any], wanted: set[int], budget: Budget, context: Any
) -> tuple[dict[int, Replay], int, int]:
    """Rebuild T3's altered timeline and retain pre-action changed snapshots.

    Each saved trigger's raw greedy action and length are checked.  Saved outcome
    fields are intentionally not reused: after an earlier intervention they refer
    to T3's altered timeline and cannot answer a fresh greedy-shadow branch.
    """
    arm, env = str(row["arm"]), int(row["env"])
    _, series = load_inputs(arm)
    replay = build_replay(arm, [env], context)
    triggers = {int(item["decision"]): item for item in row["trigger_rows"]}
    snapshots: dict[int, Replay] = {}
    checks = 0
    stop = max(triggers, default=-1)
    for decision in range(stop + 1):
        budget.check()
        trigger = triggers.get(decision)
        selected: dict[str, int] = {}

        def selector(_sim: Any) -> np.ndarray:
            nonlocal checks
            greedy, _q, legal = _greedy(replay)
            if trigger is not None:
                length = int(replay.sim.length[0, 0]) if replay.sim.alive[0, 0] else 0
                if length != int(trigger["length"]) or greedy != int(trigger["greedy"]):
                    raise AssertionError(
                        (arm, env, decision, length, trigger["length"], greedy, trigger["greedy"])
                    )
                if int(trigger["chosen"]) not in legal:
                    raise AssertionError(("saved chosen action became illegal", arm, env, decision))
                checks += 1
            if decision in wanted:
                snapshots[decision] = replay.clone(prepared=True)
            selected["action"] = int(trigger["chosen"]) if trigger is not None else greedy
            return replay.actions([selected["action"]])

        # Before T3's first intervention this agrees with its recorded baseline;
        # afterward this executes exactly the saved T3 override stream.
        replay.sim.step_with_policy(selector)
        replay.env_frames += 1
        if not bool(replay.sim.alive[0, 0]) and decision < stop:
            raise AssertionError(
                ("T3 source timeline died before a saved trigger", arm, env, decision)
            )
    if checks != len(triggers) or len(snapshots) != len(wanted):
        raise AssertionError(
            ("incomplete saved-trigger reconstruction", checks, len(triggers), len(snapshots))
        )
    return snapshots, checks, replay.env_frames


def greedy_shadow(
    source: Replay, first: int, budget: Budget
) -> tuple[int | None, dict[int, Replay], list[float]]:
    """Advance one exact greedy continuation and retain prepared decision states."""
    shadow = source.clone(prepared=True)
    snapshots: dict[int, Replay] = {first: shadow.clone(prepared=True)}
    step_seconds: list[float] = []
    greedy, _q, _legal = _greedy(shadow)
    started = time.perf_counter()
    shadow.first(greedy)
    step_seconds.append(time.perf_counter() - started)
    if not bool(shadow.sim.alive[0, 0]):
        return first, snapshots, step_seconds
    while bool(shadow.sim.alive[0, 0]) and int(shadow.sim.frame[0]) < min(HORIZON, first + 300):
        budget.check()
        decision = int(shadow.sim.frame[0])
        started = time.perf_counter()

        def capture(current: Replay) -> None:
            # ``frame`` is one-based during selector preparation; the branch
            # decision is its zero-based predecessor.
            snapshots[int(current.sim.frame[0]) - 1] = current.clone(prepared=True)

        shadow.step(observer=capture)
        step_seconds.append(time.perf_counter() - started)
        if not bool(shadow.sim.alive[0, 0]):
            return decision, snapshots, step_seconds
    return None, snapshots, step_seconds


def expand_snapshot(snapshot: Replay, decision: int, doom: int, budget: Budget) -> dict[str, Any]:
    """Expand only alternatives; the confirmed greedy death is a free control."""
    budget.check()
    greedy, q, _legal = _greedy(snapshot)
    advisory = [int(a) for a in np.flatnonzero(snapshot.sim.get_advisory_action_mask()[0, 0])]
    alternatives = sorted(
        (action for action in advisory if action != greedy),
        key=lambda action: (-float(q[action]), action),
    )
    horizon = min(300, HORIZON - decision)
    if alternatives:
        outcomes = batched_rollout(
            snapshot.clone(prepared=True), alternatives, decision, depth=300, record_outcomes=True
        )
        receipt = outcomes.pop("_batch_receipt")
    else:
        outcomes, receipt = {}, {"env_frames": 0, "candidate_count": 0, "wall_seconds": 0.0}
    budget.check()
    normalized = {
        str(action): {**outcomes[action], "final_mass": outcomes[action].get("final_length")}
        for action in alternatives
    }
    executed = min(horizon, doom - decision + 1)
    normalized[str(greedy)] = {
        "action": greedy,
        "survives": False,
        "horizon": horizon,
        "frames_survived": max(0, executed - 1),
        "frames_executed": executed,
        "death_decision": doom,
        "control_reproduced": True,
        "env_frames": 0,
        "final_mass": 0,
        "final_mass_status": "known_greedy_doom",
    }
    return {
        "greedy": greedy,
        "q": [float(value) for value in q],
        "outcomes": normalized,
        "batch": {
            **receipt,
            "actual_env_frames": int(receipt["env_frames"]),
            "original_env_frames": int(receipt["env_frames"]),
            "cache_hit": False,
        },
    }


def measure_world(world: str, budget: Budget, context: Any | None = None) -> dict[str, Any]:
    """Measure historical T3 intervention sources for one immutable bank world."""
    from research.b8_death_census_20261014 import census

    source = source_receipt(world)
    checkpoint_actual_sha = sha(Path(source["checkpoint"]["path"]))
    if checkpoint_actual_sha != source["checkpoint"]["sha256"]:
        raise RuntimeError("T3 source checkpoint bytes no longer match its immutable receipt")
    context = census.load_context() if context is None else context
    started = time.perf_counter()
    historical = changed_rows(source)
    saved_triggers = {int(trigger["decision"]): trigger for trigger in source["trigger_rows"]}
    detection = []
    previous_change = -1
    for change in historical:
        greedy_outcome = change["outcomes"][str(change["greedy"])]
        doom = greedy_outcome.get("death_decision")
        if doom is not None:
            start = max(previous_change + 1, int(doom) - 299)
            detection.append({"change": change, "start": start, "doom": int(doom)})
        previous_change = int(change["decision"])
    snapshots, trigger_checks, reconstruction_steps = reconstruct_sources(
        source, {item["start"] for item in detection}, budget, context
    )
    cases = []
    shadow_times: list[float] = []
    search_times: list[float] = []
    search_steps = shadow_steps = 0
    recovered = 0
    historical_cache_checks = 0
    for item in detection:
        budget.check()
        decision, expected_doom = int(item["start"]), int(item["doom"])
        source_snapshot = snapshots[decision]
        doom, shadow_states, one_step_times = greedy_shadow(source_snapshot, decision, budget)
        shadow_times.extend(one_step_times)
        shadow_steps += len(one_step_times)
        if doom != expected_doom:
            raise AssertionError(("greedy doom mismatch", world, decision, expected_doom, doom))
        cache: dict[int, dict[str, Any]] = {}

        def expand(probe: int) -> dict[str, Any]:
            if probe not in shadow_states:
                raise AssertionError(
                    ("missing retained greedy shadow state", decision, probe, doom)
                )
            if probe not in cache:
                trigger = saved_triggers.get(probe)
                if trigger is not None and probe <= int(item["change"]["decision"]):
                    # T3's historical source closure differs from this engine, so its
                    # outcomes are evidence to validate, never a free current answer.
                    measured = expand_snapshot(shadow_states[probe], probe, doom, budget)
                    for action, saved in trigger["outcomes"].items():
                        current = measured["outcomes"].get(action)
                        if current is None or any(
                            current[key] != saved[key]
                            for key in (
                                "survives",
                                "frames_survived",
                                "frames_executed",
                                "death_decision",
                                "horizon",
                            )
                        ):
                            raise AssertionError(
                                ("historical cache validation failed", probe, action)
                            )
                    measured["historical_cache"] = {
                        "status": "validated_fresh_not_reused",
                        "t3_receipt_sha256": sha(T3_CEILING / f"{world}.json"),
                        "reason": "historical_source_closure_drift",
                    }
                    cache[probe] = measured
                else:
                    cache[probe] = expand_snapshot(shadow_states[probe], probe, doom, budget)
            return cache[probe]

        bisection_started = time.perf_counter()
        bisection = search_lad(decision, doom, expand, refinement_radius=3)
        bisection_seconds = time.perf_counter() - bisection_started
        teacher_started = time.perf_counter()
        teacher = search_lad_teacher(decision, doom, expand)
        teacher_seconds = time.perf_counter() - teacher_started
        truth_started = time.perf_counter()
        truth = search_lad_truth(decision, doom, expand)
        truth_seconds = time.perf_counter() - truth_started
        search_times.append(bisection_seconds)
        enrichment = None
        actual_union_steps = sum(
            int(value["batch"]["actual_env_frames"]) for value in cache.values()
        )
        historical_cache_checks += sum("historical_cache" in value for value in cache.values())
        search_steps += actual_union_steps
        recovered_here = teacher["t_star"] is not None
        recovered += int(recovered_here)
        cases.append(
            {
                "source_decision": decision,
                "historical_action_change_decision": int(item["change"]["decision"]),
                "greedy_doom": doom,
                "bisection": bisection,
                "teacher": teacher,
                "truth": truth,
                "recovered": recovered_here,
                "teacher_truth_lad_error": (
                    None
                    if teacher["t_star"] is None or truth["t_star"] is None
                    else int(teacher["t_star"]) - int(truth["t_star"])
                ),
                "historical_change_truth_lad_error": (
                    None
                    if truth["t_star"] is None
                    else int(item["change"]["decision"]) - int(truth["t_star"])
                ),
                "actual_union_search_env_frames": actual_union_steps,
                "final_label_enrichment": (
                    None
                    if enrichment is None
                    else {
                        "decision": int(truth["t_star"]),
                        "actual_env_frames": int(enrichment["batch"]["actual_env_frames"]),
                        "cached_survival_validated": True,
                    }
                ),
                "hypothetical_charge_env_frames": {
                    name: sum(
                        int(cache[int(probe["decision"])]["batch"]["original_env_frames"])
                        for probe in result["probes"]
                    )
                    for name, result in (
                        ("bisection", bisection),
                        ("teacher", teacher),
                        ("truth_diagnostic_only", truth),
                    )
                },
                "rule_seconds": {
                    "bisection": bisection_seconds,
                    "teacher": teacher_seconds,
                    "truth_diagnostic_only": truth_seconds,
                },
                "timing_note": (
                    "Bisection runs fresh current-engine expansions and historical validation. "
                    "Teacher then truth reuse that cache; their times are not cold."
                ),
            }
        )
    return {
        "schema": "t34-shadow-bank-world/v1",
        "authority": "M2/M3 measurement only; no assisted-policy efficacy estimate",
        "world": world,
        "arm": source["arm"],
        "env": source["env"],
        "world_seed": source["world_seed"],
        "t3_receipt_sha256": sha(T3_CEILING / f"{world}.json"),
        "historical_cache": {
            "status": "unqualified_due_source_drift; fresh validation required",
            "fresh_validated_queries": historical_cache_checks,
        },
        "source_checkpoint": {
            **source["checkpoint"],
            "actual_sha256": checkpoint_actual_sha,
        },
        "saved_trigger_checks": trigger_checks,
        "historical_action_changes": len(historical),
        "recovered_action_changes": recovered,
        "search_row_steps": search_steps,
        "shadow_row_steps": shadow_steps,
        "reconstruction_row_steps": reconstruction_steps,
        "operational_row_steps": search_steps + shadow_steps + reconstruction_steps,
        "shadow_step_latency": timing_summary(shadow_times),
        "catchup_300_latency": {
            **timing_summary([]),
            "status": "not_measured_here; see ShadowTrajectory M3 benchmark",
        },
        "lad_search_latency_cached_diagnostic": timing_summary(search_times),
        "cases": cases,
        "nonmonotonic_limitation": "Bisection plus radius-3 refinement is not a global LAD proof.",
        "wall_seconds": time.perf_counter() - started,
    }


def t1_lad_reachability(search_seconds: list[float], rows_root: Path = T1_ROWS) -> dict[str, Any]:
    """Convert recorded late T1 H_LAD values through observed LAD-search latency."""
    horizons = []
    for path in sorted(rows_root.glob("b8_s*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("stratum") == "late" and row.get("H_LAD") is not None:
                horizons.append(int(row["H_LAD"]))
    all_late = 0
    known = []
    for path in sorted(rows_root.glob("b8_s*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("stratum") == "late":
                all_late += 1
                known.append(row.get("H_LAD"))
    if all_late != 49:
        raise ValueError(f"T1 late-death denominator drift: {all_late}")
    result: dict[str, Any] = {
        "late_deaths": 49,
        "known_h_lad": len(horizons),
        "censored_failures": sum(value is None for value in known),
        "latency_scenarios": {},
    }
    for label, latency in (
        ("p50", percentile(search_seconds, 50)),
        ("p95", percentile(search_seconds, 95)),
        ("p99", percentile(search_seconds, 99)),
    ):
        result["latency_scenarios"][label] = {
            "search_seconds": latency,
            "reachable_fraction": {
                str(fps): (
                    None
                    if latency is None
                    else sum((299 - h) / fps > latency for h in horizons) / 49
                )
                for fps in (12, 24)
            },
            "definition": (
                "(299-H_LAD)/fps > measured LAD search; 3 censored late deaths fail. "
                "Root adds catch-up separately."
            ),
        }
    return result


def summarize(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate complete per-world receipts; missing worlds remain explicit."""
    expected = world_names()
    by_world = {str(row["world"]): row for row in receipts}
    changes = sum(int(row["historical_action_changes"]) for row in receipts)
    recovered = sum(int(row["recovered_action_changes"]) for row in receipts)
    bisection_times = [
        case["rule_seconds"]["bisection"]
        for row in receipts
        for case in row["cases"]
        if "rule_seconds" in case
    ]
    teacher_times = [
        case["rule_seconds"]["teacher"]
        for row in receipts
        for case in row["cases"]
        if "rule_seconds" in case
    ]
    charges = {
        method: sum(
            int(case["hypothetical_charge_env_frames"][method])
            for row in receipts
            for case in row["cases"]
            if "hypothetical_charge_env_frames" in case
        )
        for method in ("bisection", "teacher", "truth_diagnostic_only")
    }
    teacher_errors = [
        abs(int(case["teacher_truth_lad_error"]))
        for row in receipts
        for case in row["cases"]
        if case.get("teacher_truth_lad_error") is not None
    ]
    return {
        "schema": "t34-shadow-bank-summary/v1",
        "completed_worlds": len(by_world),
        "expected_t3_worlds": len(expected),
        "missing_worlds": [world for world in expected if world not in by_world],
        "historical_action_changes": changes,
        "historical_action_change_worlds": sum(
            bool(row["historical_action_changes"]) for row in receipts
        ),
        "recovered_action_changes": recovered,
        "recovery_fraction": recovered / changes if changes else None,
        "search_row_steps": sum(int(row["search_row_steps"]) for row in receipts),
        "operational_row_steps": sum(int(row["operational_row_steps"]) for row in receipts),
        "hypothetical_method_charges_env_frames": charges,
        "teacher_truth_lad_absolute_error": {
            "count": len(teacher_errors),
            "p50_decisions": percentile([float(value) for value in teacher_errors], 50),
            "p95_decisions": percentile([float(value) for value in teacher_errors], 95),
        },
        "mean_operational_row_steps_per_completed_world": (
            sum(int(row["operational_row_steps"]) for row in receipts) / len(receipts)
            if receipts
            else None
        ),
        "v1_reference_row_steps_per_world": 400_000,
        "t3_v1_total_core_rollout_env_frames": 55_002_897,
        "t3_dense_survival": {
            "survivors": 54,
            "worlds": 64,
            "meaning": "survival, not interventions",
        },
        "rule_wall_seconds_shared_cache": {
            "bisection": timing_summary(bisection_times),
            "teacher": timing_summary(teacher_times),
        },
        "t1_h_lad_reachability_cached_diagnostic_not_deployable": t1_lad_reachability(
            bisection_times
        ),
        "nonmonotonic_limitation": (
            "No global LAD proof follows from this registered bisection heuristic."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", help="one T3 world, e.g. b8_s0_0")
    parser.add_argument("--unit-list", action="store_true")
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--time-limit-seconds", type=float, default=900)
    parser.add_argument(
        "--resume", action="store_true", help="allow an existing matching receipt to be read"
    )
    args = parser.parse_args()
    if args.unit_list == (args.world is not None):
        parser.error("choose exactly one of --unit-list or --world")
    if args.world is not None and args.world not in world_names():
        parser.error(f"unknown fixed T3 world: {args.world}")
    if not 0 < args.time_limit_seconds <= 1800:
        parser.error("--time-limit-seconds must be in (0, 1800]")
    return args


def main() -> int:
    args = parse_args()
    if args.unit_list:
        print(json.dumps({"worlds": world_names(), "count": len(world_names())}, indent=2))
        return 0
    receipt_path = args.out / "bank" / f"{args.world}.json"
    if receipt_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"create-only receipt exists: {receipt_path}; pass --resume to inspect it"
            )
        prior = json.loads(receipt_path.read_text())
        expected_sha = sha(T3_CEILING / f"{args.world}.json")
        if (
            prior.get("schema") != "t34-shadow-bank-world/v1"
            or prior.get("status") != "COMPLETE"
            or prior.get("world") != args.world
            or prior.get("t3_receipt_sha256") != expected_sha
        ):
            raise RuntimeError(
                "resume requires a complete receipt matching the immutable T3 source"
            )
        expected_checkpoint = source_receipt(args.world)["checkpoint"]
        if prior.get("source_checkpoint", {}).get("actual_sha256") != sha(
            Path(expected_checkpoint["path"])
        ):
            raise RuntimeError("resume receipt checkpoint bytes differ from the T3 source")
        current = provenance()
        if prior.get("provenance", {}).get("source_sha256") != current["source_sha256"]:
            raise RuntimeError(
                "resume receipt source closure differs from the current implementation"
            )
        print(json.dumps(prior, indent=2, sort_keys=True))
        return 0
    started = time.monotonic()
    meta: dict[str, Any] | None = None
    try:
        configure_cpu()
        meta = provenance()
        result = {
            "status": "COMPLETE",
            "provenance": meta,
            **measure_world(args.world, Budget(args.time_limit_seconds)),
        }
        write_once(receipt_path, result)
        print(json.dumps({"world": args.world, "status": "COMPLETE", "receipt": str(receipt_path)}))
        return 0
    except BaseException as error:
        failure = {
            "schema": "t34-shadow-bank-world/v1",
            "status": "INCOMPLETE",
            "world": args.world,
            "provenance": meta,
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": time.monotonic() - started,
        }
        write_once(receipt_path, failure)
        print(
            json.dumps({"world": args.world, "status": "INCOMPLETE", "receipt": str(receipt_path)})
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
