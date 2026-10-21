"""Real 24 fps development smoke on immutable M3/T3 historical sources.

Each historical action-change interval is an independent counterfactual. Earlier
T3 overrides reconstruct its root, then only bare B8 plus the actual v2 adapter
runs until first delivered change or the historical greedy doom. This measures
would-execute coverage, never episode efficacy or a registered verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT = ART / "t34-pathb-smoke-20261021"
SHADOW = ART / "t34-shadow-20261019"
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def sha(path: Path) -> str:
    """Hash a specific input without scanning any artifact tree."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, data: Any) -> None:
    """Create an immutable JSON receipt."""
    with path.open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load(path: Path) -> Any:
    """Read only explicitly named historical development inputs."""
    return json.loads(path.read_text())


def host() -> dict:
    """Check AC, open lid and thermal state without touching running jobs."""
    from research.t34_pathb.host_guard_v2 import _power_thermal

    ac, lid, thermal = _power_thermal()
    if not all((ac, lid, thermal)):
        raise RuntimeError(f"unsafe host: ac={ac}, lid={lid}, thermal={thermal}")
    return {"ac": ac, "lid_open": lid, "thermal_ok": thermal, "load": os.getloadavg()}


def inventory() -> tuple[list[dict], dict[str, str]]:
    """Bind exactly 22 historical trajectories/75 changes and five M3 dooms."""
    from research.t34_pathb.precheck_v2 import materialize_t3
    from research.t34_shadow.bank_measure import changed_rows

    units, inputs = [], {}
    rows = materialize_t3()
    for row in rows:
        path = Path(row["t3_receipt"]["path"])
        source = load(path)
        changes = changed_rows(source)
        if not changes:
            continue
        bank_path = SHADOW / "bank" / f'{row["t3_world"]}.json'
        bank = load(bank_path)
        if bank["t3_receipt_sha256"] != sha(path) or len(bank["cases"]) != len(changes):
            raise ValueError("historical T3 source identity/count mismatch")
        inputs[str(path)], inputs[str(bank_path)] = sha(path), sha(bank_path)
        inputs[source["checkpoint"]["path"]] = sha(Path(source["checkpoint"]["path"]))
        cases = []
        previous = -1
        for index, (case, change) in enumerate(zip(bank["cases"], changes)):
            if case["historical_action_change_decision"] != change["decision"]:
                raise ValueError("historical change ordering drift")
            cases.append(
                {
                    "id": f'{row["t3_world"]}-c{index:02d}',
                    "group": "T3",
                    "doom": case["greedy_doom"],
                    "start": max(previous + 1, case["greedy_doom"] - 499),
                    "historical_detection": case["source_decision"],
                    "historical_change": change["decision"],
                    "historical_lad_d300": case["truth"]["t_star"],
                    "historical_lad_exact": True,
                    "historical_survival_rescue": bool(
                        change["outcomes"][str(change["chosen"])]["survives"]
                    ),
                }
            )
            previous = change["decision"]
        units.append({"group": "T3", "row": row, "source": source, "cases": cases})
    for mix in ("frozen", "mixed"):
        path = SHADOW / f"lad-{mix}.json"
        inputs[str(path)] = sha(path)
        by_seed: dict[int, list] = {}
        for index, case in enumerate(load(path)["cases"]):
            by_seed.setdefault(case["seed"], []).append(
                {
                    "id": f'M3-{mix}-{case["seed"]}-{index}',
                    "group": "M3",
                    "doom": case["doom"],
                    "start": max(0, case["doom"] - 499),
                    "historical_detection": case["first"],
                    "historical_lad_d300": case["result"]["t_star"],
                    "historical_lad_exact": False,
                }
            )
            inputs.update(case["identity"]["checkpoint_sha256"])
        for seed, cases in by_seed.items():
            units.append({"group": "M3", "mix": mix, "seed": seed, "cases": cases})
    t3 = [u for u in units if u["group"] == "T3"]
    if len(t3) != 22 or sum(len(u["cases"]) for u in t3) != 75:
        raise ValueError("T3 fixed inventory drift")
    if sum(len(u["cases"]) for u in units if u["group"] == "M3") != 5:
        raise ValueError("M3 fixed inventory drift")
    return units, inputs


def reconstruct(unit: dict, check: Any) -> tuple[dict[str, Any], list, str]:
    """Replay the exact saved override stream, retaining only interval roots."""
    import numpy as np

    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t34_pathb.engine_v2 import _prepared_metadata
    from research.t34_shadow.shadow import clone_world_exact
    from src.simd_env.eval_engine import _dispatch_actions

    if unit["group"] == "T3":
        row = unit["row"]
        world = build_world(row["t3_source_arm"], row)
        specs = row["specs"]
        checkpoint = row["t3_checkpoint"]["sha256"]
        triggers = {t["decision"]: t for t in unit["source"]["trigger_rows"]}
    else:
        from research.t34_shadow.identity import _new_pair

        _live, world, specs, identity = _new_pair(unit["seed"], unit["mix"])
        checkpoint = identity["hero"]["sha256"]
        triggers = {}
    wanted: dict[int, list] = {}
    for case in unit["cases"]:
        wanted.setdefault(case["start"], []).append(case)
    snapshots = {}
    life_start = 0
    for frame in range(max(wanted) + 1):
        check()
        if not bool(world.sim.alive[0, 0]):
            life_start = frame
        if frame in wanted:
            for case in wanted[frame]:
                if frame < life_start:
                    raise ValueError("root crosses hero life")
                snapshots[case["id"]] = clone_world_exact(world)
        if frame == max(wanted):
            break

        def choose(sim: Any) -> Any:
            world.runtime.prepare(sim)
            actions = np.ones((1, sim.S), dtype=np.int64)
            trigger = triggers.get(frame)
            if trigger:
                # Exact B8 metadata before dispatch; original trigger action and
                # logical length must reproduce on the altered T3 timeline.
                sim._policy_phase_prepared = True
                try:
                    bare, _q, legal = _prepared_metadata(world)
                finally:
                    sim._policy_phase_prepared = False
                if bare != trigger["greedy"] or int(sim.length[0, 0]) != trigger["length"]:
                    raise ValueError(f"T3 replay trigger mismatch at {frame}")
                if not legal[trigger["chosen"]]:
                    raise ValueError("saved override became illegal")
            _dispatch_actions(
                sim, sim.get_resolved_action_mask(), actions, [world.hero], [world.opponents]
            )
            if trigger:
                actions[0, 0] = trigger["chosen"]
            return actions

        world.sim.step_with_policy(choose)
        world.runtime.observe_step(world.sim)
    return snapshots, specs, checkpoint


class ObservedEngine:
    """Observe real worker calls without altering engine decisions or fences."""

    def __init__(self, checkpoint: str, terminal_horizon: int | None) -> None:
        from research.t34_pathb.engine_v2 import PathBSearchEngine

        self.inner = PathBSearchEngine(checkpoint=checkpoint)
        self.inner.terminal_horizon = terminal_horizon
        self.detections: list[dict] = []
        self.searches: list[dict] = []
        self.live_progress: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def maintain(self, request: Any) -> Any:
        started = time.monotonic()
        value = self.inner.maintain(request)
        if value.doom_frame is not None:
            self.detections.append(
                {
                    "wall": time.monotonic(),
                    "source_frame": value.source_frame,
                    "doom_frame": value.doom_frame,
                    "maintenance_seconds": time.monotonic() - started,
                    "actual_live_frame": self.live_progress()[0] if self.live_progress else None,
                }
            )
        return value

    def run_search(self, request: Any) -> Any:
        started = time.monotonic()
        value = self.inner.run_search(request)
        self.searches.append(
            {
                "start_wall": started,
                "finish_wall": time.monotonic(),
                "status": value.status,
                "target_frame": value.target_frame,
                "action": value.action,
                "kind": value.intervention_kind,
                "outcomes": {str(a): vars(o) for a, o in value.outcomes.items()},
            }
        )
        return value


def measure_lad(case: dict, source: Any, check: Any) -> dict:
    """Find exact last v2-improving H500 decision, outside paced measurement.

    V2 allows both a horizon rescue and a strict survival delay. This LAD
    avoids the original greedy death and is distinct from survival-only LAD.
    """
    from research.t34_pathb.engine_v2 import _capture_and_step, evaluate_prepared
    from research.t34_shadow.shadow import clone_world_exact

    root = clone_world_exact(source)
    decisions = []
    for frame in range(case["start"], case["doom"] + 1):
        check()
        decision = _capture_and_step(root)
        if decision.frame != frame:
            raise ValueError("diagnostic decision/frame convention mismatch")
        decisions.append(decision)
        if not bool(root.sim.alive[0, 0]):
            if frame != case["doom"]:
                raise ValueError("diagnostic greedy doom mismatch")
            break
    if bool(root.sim.alive[0, 0]):
        raise ValueError("historical greedy doom did not reproduce")
    probes = 0
    for decision in reversed(decisions):
        check()
        horizon = min(500, 5000 - decision.frame) if case["group"] == "T3" else 500
        actions = [a for a in range(6) if decision.legal[a] or a == decision.bare_action]
        outcomes = evaluate_prepared(decision, actions, horizon)
        probes += 1
        baseline = outcomes[decision.bare_action]
        improving = [
            a
            for a in actions
            if a != decision.bare_action
            and not baseline.survives
            and (outcomes[a].survives or outcomes[a].alive_steps > baseline.alive_steps)
        ]
        if improving:
            return {
                "lad": decision.frame,
                "rule": "exact_backward_last_v2_improvement_H500",
                "kind": "rescue" if any(outcomes[a].survives for a in improving) else "delay",
                "probes": probes,
                "outcomes": {str(a): vars(o) for a, o in outcomes.items()},
            }
    return {"lad": None, "rule": "exact_backward_last_v2_improvement_H500", "probes": probes}


def delivered_search(
    searches: list[dict], applied: dict | None, start: int, wall: float
) -> dict | None:
    """Match actual execution to its complete result, including later requests."""
    if applied is None:
        return None
    return next(
        (
            s
            for s in searches
            if s["status"] == "IMPROVING"
            and s["target_frame"] == applied["frame"]
            and s["action"] == applied["executed_action"]
            and s["finish_wall"] < wall + (s["target_frame"] - start) / 24
        ),
        None,
    )


def run_case(
    case: dict,
    source: Any,
    specs: list,
    checkpoint: str,
    workers: int,
    check: Any,
    pool: Any,
) -> dict:
    """Pace actual adapter decisions and record actual execution, never wait a tick."""
    from research.t34_pathb import adapter_v2
    from research.t34_shadow.shadow import clone_world_exact

    engines = []

    def engine_factory(*args: Any, **kwargs: Any) -> ObservedEngine:
        engine = ObservedEngine(checkpoint, 5000 if case["group"] == "T3" else None)
        engines.append(engine)
        return engine

    started = time.monotonic()
    events, ticks, overruns = [], 0, 0
    evaluations = []

    def observe(decision: Any, outcomes: dict) -> None:
        baseline = outcomes[decision.bare_action]
        improving = [
            a
            for a, outcome in outcomes.items()
            if a != decision.bare_action
            and decision.legal[a]
            and not baseline.survives
            and (outcome.survives or outcome.alive_steps > baseline.alive_steps)
        ]
        if improving:
            evaluations.append(
                {"wall": time.monotonic(), "frame": decision.frame, "improving_actions": improving}
            )

    pool.observer = observe
    # The constructor's immediate worker submission is included in wall timing.
    with patch.object(adapter_v2, "PathBSearchEngine", engine_factory):
        session = adapter_v2.RunSession(
            clone_world_exact(source),
            "paced",
            specs=specs,
            checkpoint_sha=checkpoint,
            terminal_horizon=5000 if case["group"] == "T3" else None,
        )
    engine = engines[0]
    engine.live_progress = session.controller._live_progress
    applied = None
    max_slip = 0.0
    try:
        for frame in range(case["start"], case["doom"] + 1):
            check()
            deadline = started + ticks / 24
            time.sleep(max(0, deadline - time.monotonic()))
            tick_started = time.monotonic()
            max_slip = max(max_slip, tick_started - deadline)
            session.step()
            ticks += 1
            overruns += int(time.monotonic() > started + ticks / 24)
            event = next(
                (
                    e
                    for e in reversed(session.controller_state()["recent"])
                    if e["frame"] == frame and e.get("executed_action", -1) >= 0
                ),
                None,
            )
            if event and (
                event.get("applied") or event.get("late_discarded") or event.get("stale_discarded")
            ):
                events.append(dict(event))
            if event and event.get("applied"):
                applied = dict(event)
                break
            counts = session.controller_state()["counts"]
            if (
                counts["worker_errors"]
                or counts["invalid_results"]
                or counts["unconditional_faults"]
            ):
                raise RuntimeError(f"worker fault: {session.controller_state()}")
            if not session.hero_alive:
                if session.frame - 1 != case["doom"]:
                    raise ValueError(f'greedy doom drift: {session.frame - 1} != {case["doom"]}')
                break
        state = session.controller_state()
        paced_finished = time.monotonic()
    finally:
        session.close(wait=True)
        pool.observer = None
    first = next((s for s in engine.searches if s["status"] == "IMPROVING"), None)
    detection = engine.detections[0] if engine.detections else None
    target_deadline = started + (first["target_frame"] - case["start"]) / 24 if first else None
    on_time = bool(first and first["finish_wall"] < target_deadline)
    matched = delivered_search(engine.searches, applied, case["start"], started)
    delivered = matched is not None
    historical_lad = case["historical_lad_d300"]
    detection_live_frame = (
        case["start"] + int((detection["wall"] - started) * 24) if detection else None
    )
    return {
        **case,
        "workers": workers,
        "status": "COMPLETE",
        "ticks": ticks,
        "elapsed_seconds": paced_finished - started,
        "wall_including_drain": time.monotonic() - started,
        "fps": 24,
        "achieved_fps": ticks / (paced_finished - started),
        "tick_overruns": overruns,
        "max_tick_start_slip_seconds": max_slip,
        "detections": engine.detections,
        "searches": engine.searches,
        "fully_evaluated_improvements": evaluations,
        "time_to_first_fully_evaluated_improving_action_seconds": (
            min(e["wall"] for e in evaluations) - detection["wall"]
            if evaluations and detection
            else None
        ),
        "evaluated_late_count": sum(
            e["wall"] >= started + (e["frame"] - case["start"]) / 24 for e in evaluations
        ),
        "first_improving_seconds_from_start": first["finish_wall"] - started if first else None,
        "first_improving_seconds_from_detection": (
            first["finish_wall"] - detection["wall"] if first and detection else None
        ),
        "detection_live_frame_at_24fps": detection_live_frame,
        "lead_frames_historical_d300_lad": (
            historical_lad - detection_live_frame
            if historical_lad is not None and detection
            else None
        ),
        "lead_reference_note": "Historical D300 LAD, not D500 LAD; M3 LAD is heuristic, T3 exact.",
        "result_before_its_frame": on_time,
        "delivered_before_its_frame": delivered,
        "applied": applied,
        "events": events,
        "controller": state,
        "late_count": state["counts"]["late_discarded"],
        "stale_count": state["counts"]["stale_discarded"],
    }


def summarize(rows: list[dict]) -> dict:
    """Apply the precommitted fixed denominator without conditioning on success."""
    summary = {}
    for workers in (1, 6):
        selected = [r for r in rows if r["workers"] == workers and r["group"] == "T3"]
        delivered = sum(r["delivered_before_its_frame"] for r in selected)
        rescued = sum(
            r["delivered_before_its_frame"] and r["applied"]["intervention_kind"] == "rescue"
            for r in selected
        )
        summary[str(workers)] = {
            "cases": len(selected),
            "denominator": 75,
            "delivered": delivered,
            "fraction": delivered / 75,
            "delivered_h500_rescues": rescued,
            "complete": len(selected) == 75,
        }
    complete = all(s["complete"] for s in summary.values())
    verdict = "INCOMPLETE"
    if complete:
        if summary["1"]["fraction"] >= 0.5:
            verdict = "N1_DELIVERABLE_DEVELOPMENT_ONLY"
        elif summary["6"]["fraction"] >= 0.5:
            verdict = "STOP_OPERATOR02_AMEND_SECTION4_AND_RETIME"
        else:
            verdict = "STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY"
    return {"workers": summary, "stopping_rule_verdict": verdict}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seconds", type=float, default=6900)
    args = parser.parse_args()
    if not 0 < args.seconds <= 7200:
        parser.error("wall limit must be <= two hours")
    if (
        any(os.environ.get(v) != "1" for v in THREAD_VARS)
        or os.environ.get("SNAKE_DQN_DEVICE") != "cpu"
    ):
        parser.error("CPU device and all numerical thread caps must explicitly equal one")
    from research.t34_pathb import batch_v2
    from research.t34_pathb_smoke.parallel import BranchPool
    from research.t34_shadow.common import provenance

    units, inputs = inventory()
    if args.check:
        print(
            json.dumps(
                {
                    "units": len(units),
                    "T3_cases": 75,
                    "M3_cases": 5,
                    "host": host(),
                    "output_exists": args.out.exists(),
                    "inputs": inputs,
                },
                indent=2,
            )
        )
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    next_host = started

    def check() -> None:
        nonlocal next_host
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("development smoke wall cap reached")
        if time.monotonic() >= next_host:
            host()
            next_host = time.monotonic() + 30

    meta = provenance()
    meta["source_sha256"].update(
        {
            str(p.relative_to(ROOT)): sha(p)
            for directory in ("research/t34_pathb", "research/t34_pathb_smoke")
            for p in (ROOT / directory).glob("*.py")
        }
    )
    write(
        args.out / "start.json",
        {
            "provenance": meta,
            "inputs": inputs,
            "stopping_rule_sha256": sha(Path(__file__).with_name("STOPPING_RULE.md")),
            "max_seconds": args.seconds,
            "host": host(),
        },
    )
    rows = []
    lads = {}
    error = None
    original = batch_v2.evaluate_decisions
    try:
        for workers in (1, 6):
            pool = BranchPool(workers, original)
            try:
                with patch.object(batch_v2, "evaluate_decisions", pool.evaluate):
                    for unit in units:
                        check()
                        snapshots, specs, checkpoint = reconstruct(unit, check)
                        for case in unit["cases"]:
                            check()
                            if workers == 1:
                                lads[case["id"]] = measure_lad(case, snapshots[case["id"]], check)
                                write(args.out / f'lad-{case["id"]}.json', lads[case["id"]])
                            row = run_case(
                                case, snapshots[case["id"]], specs, checkpoint, workers, check, pool
                            )
                            row["v2_lad"] = lads[case["id"]]
                            detection_frame = (
                                row["detections"][0]["actual_live_frame"]
                                if row["detections"]
                                else None
                            )
                            row["lead_frames_to_last_avoidable_decision"] = (
                                lads[case["id"]]["lad"] - detection_frame
                                if lads[case["id"]]["lad"] is not None
                                and detection_frame is not None
                                else None
                            )
                            write(args.out / f'n{workers}-{case["id"]}.json', row)
                            rows.append(row)
                            print(
                                json.dumps(
                                    {
                                        "workers": workers,
                                        "case": case["id"],
                                        "done": len(rows),
                                        "delivered": row["delivered_before_its_frame"],
                                        "elapsed": round(time.monotonic() - started, 1),
                                    }
                                ),
                                flush=True,
                            )
                        del snapshots
            finally:
                pool.close()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, file=sys.stderr, flush=True)
    summary = summarize(rows)
    receipt = {
        "schema": "t34-pathb-development-deliverability/v1",
        "development_only": True,
        "excluded_from_registered_verdicts": True,
        "summary": summary,
        "status": "COMPLETE" if len(rows) == 160 and error is None else "INCOMPLETE",
        "error": error,
        "wall_seconds": time.monotonic() - started,
        "inputs_unchanged": all(sha(Path(p)) == digest for p, digest in inputs.items()),
        "source_unchanged": all(
            sha(ROOT / p) == digest for p, digest in meta["source_sha256"].items()
        ),
        "rows": {p.name: sha(p) for p in args.out.glob("n*.json")},
        "source_commit": meta["commit"],
        "start_sha256": sha(args.out / "start.json"),
    }
    if not receipt["inputs_unchanged"] or not receipt["source_unchanged"]:
        receipt["status"] = "INCOMPLETE"
        receipt["error"] = "source/input identity changed during smoke"
    if receipt["status"] != "COMPLETE":
        receipt["summary"]["stopping_rule_verdict"] = "INCOMPLETE"
    write(args.out / "receipt.json", receipt)
    print(json.dumps(receipt["summary"], indent=2), flush=True)
    return 0 if receipt["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
