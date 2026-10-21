"""CPU-only O2 triple-arm producer using one shared online H500 root tree.

This intentionally uses the retained O1 branch evaluator as a *single* tree at
each prospective online root.  It is an unpaced research controller: it does
not claim Path B's asynchronous paced deployment behaviour.  D's action ranking
is imported from the retained Path B/ExIt implementation; D+O scores metadata
from the exact same all-action continuation and never creates extra branches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.t34_o1_harvest.provenance import ProvenanceObserver
from research.t34_o1_harvest.runner import digest, evaluate_root, step
from research.t34_o2_harvest.controller import (
    HORIZON,
    ControllerChoice,
    HarvestScore,
    OpportunitySchedule,
    choose_shared_tree,
)

SCHEMA = "t34-o2-harvest-world/v1"
RUN_SCHEMA = "t34-o2-harvest-run/v1"
ARMS = ("bare", "d", "do")
CAMPAIGN_FRAMES = 5000
PREFLIGHT_FRAMES = 40
PREFLIGHT_HORIZON = 500


class O2RunError(RuntimeError):
    """Fail-closed producer error; partial efficacy rows are not valid evidence."""


class ScientificFault(RuntimeError):
    """A preregistered model/action fault that scientifically KILLs O2."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        super().__init__(self.payload)

    @classmethod
    def from_exception(
        cls, *, row: Mapping[str, Any], arm: str, decision: int, phase: str, exc: BaseException
    ) -> "ScientificFault":
        return cls(
            {
                "world": dict(row),
                "arm": arm,
                "decision": int(decision),
                "phase": phase,
                "exception_class": type(exc).__name__,
                "detail": str(exc) or type(exc).__name__,
            }
        )


def _scientific_fault(exc: BaseException) -> bool:
    """Only named scientific tripwires become a KILL; all else remains incomplete."""
    from research.t34_exit_b8.faults import IllegalActionFault, NonFiniteQLossFault
    from research.t34_pathb.engine_v2 import NonFiniteQ

    return isinstance(exc, (NonFiniteQ, NonFiniteQLossFault, IllegalActionFault))


def _fault_class_name(name: str) -> str:
    return {
        "NonFiniteQ": "nonfinite_q",
        "NonFiniteQLossFault": "nonfinite_loss",
        "IllegalActionFault": "illegal_action",
    }.get(name, "unknown")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threads() -> None:
    """Set the numerical library contract before torch/numba work is imported."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    import torch

    torch.set_num_threads(1)


def _scores(root: Mapping[str, Any]) -> list[HarvestScore]:
    branches = root["branches"]
    bare = branches[str(root["bare_action"])]
    output: list[HarvestScore] = []
    for text, branch in branches.items():
        action = int(text)
        delta = int(branch["delta_mass_sum"])
        death = int(
            round(
                ProvenanceObserver.death_attributed_delta(
                    branch["ledger"], bare["ledger"], root["horizon"]
                )
                * root["horizon"]
            )
        )
        output.append(HarvestScore(action, delta, death, bool(branch["survives"])))
    return output


def _choice(
    root: Mapping[str, Any],
) -> tuple[ControllerChoice, ControllerChoice, list[HarvestScore]]:
    scores = _scores(root)
    d, do = choose_shared_tree(
        {str(key): value for key, value in root["branches"].items()},
        q=root["q"],
        bare_action=int(root["bare_action"]),
        scores=scores,
    )
    return d, do, scores


def _nearby_death(world: Any, prior_alive: Any) -> bool:
    """Detect an O1-style local opponent death from the just-completed live tick."""
    import numpy as np

    died = prior_alive[0] & ~world.sim.alive[0]
    if not bool(world.sim.alive[0, 0]):
        return False
    heads = world.sim.heads()[0]
    return any(
        slot > 0 and int(np.abs(heads[slot] - heads[0]).sum()) <= 30
        for slot in np.flatnonzero(died)
    )


def _prepare_online_root(world: Any) -> None:
    """Prepare exactly as O1 does, while preserving the process-global RNG."""
    from research.t34_exit_b8.trajectory import prepare_world

    outer = random.getstate()
    random.setstate(world._o1_rng)
    try:
        prepare_world(world)
        world._o1_rng = random.getstate()
    finally:
        random.setstate(outer)


def rollout_arm(
    row: Mapping[str, Any],
    checkpoint: str,
    arm: str,
    *,
    frames: int,
    horizon: int,
    schedule: OpportunitySchedule,
    deadline: float = math.inf,
    role: str = "study",
) -> dict[str, Any]:
    """Run one terminal hero arm; every controller root is complete and exported."""
    if arm not in ARMS or frames < 1 or horizon < 1:
        raise O2RunError("invalid O2 arm or horizon")
    _threads()
    from research.t34_exit_b8.policy import B8Policy
    from research.t34_exit_b8.trajectory import build_world

    world = build_world(dict(row), checkpoint)
    world._o1_rng = random.Random(int(row["world_seed"])).getstate()
    initial_digest = digest(world)
    observer = ProvenanceObserver(world.sim)
    policy = B8Policy(device="cpu")
    actions: list[int] = []
    length_trace: list[int] = []
    nearby_death_decisions: list[int] = []
    roots: list[dict[str, Any]] = []
    event_roots = 0
    last_event = -schedule.event_dedupe
    nearby_death = False
    started = time.monotonic()
    decision = 0
    try:
        for decision in range(frames):
            if time.monotonic() >= deadline:
                raise TimeoutError("O2 per-world wall deadline expired")
            if not bool(world.sim.alive[0, 0]):
                break
            forced: int | None = None
            if arm != "bare" and schedule.due(
                decision, event_roots, nearby_death=nearby_death, last_event=last_event
            ):
                _prepare_online_root(world)
                search_started = time.monotonic()
                root = evaluate_root(world, observer, policy, horizon, deadline)
                d_choice, do_choice, scores = _choice(root)
                search_elapsed_seconds = time.monotonic() - search_started
                choice = d_choice if arm == "d" else do_choice
                forced = choice.action
                event_root = decision not in schedule.time_roots
                if event_root:
                    event_roots += 1
                    last_event = decision
                roots.append(
                    {
                        "decision": decision,
                        "time_root": not event_root,
                        "event_root": event_root,
                        "root": root,
                        "choice": choice.record(),
                        "scores": [asdict(score) for score in scores],
                        "search_elapsed_seconds": search_elapsed_seconds,
                    }
                )
            prior_alive = world.sim.alive.copy()
            delivered = step(world, policy, forced)
            observer.assert_accounting()
            actions.append(int(delivered[0]))
            length_trace.append(int(world.sim.length[0, 0]) if bool(world.sim.alive[0, 0]) else 0)
            nearby_death = _nearby_death(world, prior_alive)
            if nearby_death:
                nearby_death_decisions.append(decision + 1)
        alive = bool(world.sim.alive[0, 0])
        mass_sum = int(sum(length_trace))
        metrics = {
            "mass_integral": mass_sum / frames,
            "mass_sum": mass_sum,
            "survival_fraction": sum(value > 0 for value in length_trace) / frames,
            "no_death": int(alive),
            "deaths": int(not alive),
            "decisions": len(actions),
            "boost_decisions": sum(action >= 3 for action in actions),
            "action_counts": [actions.count(action) for action in range(6)],
            "unconditional_faults": [],
        }
    except BaseException as exc:
        if role == "study" and _scientific_fault(exc):
            raise ScientificFault.from_exception(
                row=row, arm=arm, decision=decision, phase=role, exc=exc
            ) from exc
        raise
    finally:
        observer.assert_accounting()
        observer.uninstall()
    return {
        **metrics,
        "action_trace": actions,
        "mass_trace": length_trace,
        "initial_digest": initial_digest,
        "final_digest": digest(world),
        "controller": {
            "rule_id": (
                "bare_b8"
                if arm == "bare"
                else "rescue_only" if arm == "d" else "rescue_plus_harvest"
            ),
            "roots": roots,
            "time_roots": list(schedule.time_roots),
            "event_dedupe": schedule.event_dedupe,
            "max_event_roots": schedule.max_event_roots,
            "nearby_death_decisions": nearby_death_decisions,
            "branch_budget": {
                "max_roots": len(schedule.time_roots) + schedule.max_event_roots,
                "max_actions": 6,
                "horizon": horizon,
                "harvest_added_branches": 0,
            },
            "shared_tree_branches": sum(len(item["root"]["branches"]) for item in roots),
            "search_rows_total": sum(
                int(branch["row_steps"])
                for item in roots
                for branch in item["root"]["branches"].values()
            ),
            "harvest_added_branches": 0,
        },
        "base_row_steps": len(actions),
        "search_elapsed_seconds": sum(float(item["search_elapsed_seconds"]) for item in roots),
        "branch_row_steps": sum(
            int(branch["row_steps"])
            for item in roots
            for branch in item["root"]["branches"].values()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "executed_cfg": asdict(world.sim.cfg),
    }


def evaluate_world(
    row: Mapping[str, Any],
    checkpoint: str,
    *,
    frames: int,
    horizon: int,
    deadline: float = math.inf,
    schedule: OpportunitySchedule | None = None,
    role: str = "study",
) -> dict[str, Any]:
    """Run the paired bare/D/D+O triple serially on one CPU worker."""
    started = time.monotonic()
    schedule = schedule or OpportunitySchedule(
        time_roots=(10, 20) if frames <= PREFLIGHT_FRAMES else (1000, 2000, 3000, 4000),
        max_event_roots=0 if frames <= PREFLIGHT_FRAMES else 4,
    )
    arms = {
        arm: rollout_arm(
            row,
            checkpoint,
            arm,
            frames=frames,
            horizon=horizon,
            schedule=schedule,
            deadline=deadline,
            role=role,
        )
        for arm in ARMS
    }
    cfg = {"mechanics_version": 2, "terminal_hero": True, "horizon": frames}
    if any(arm_row["executed_cfg"].get("mechanics_version") != 2 for arm_row in arms.values()):
        raise O2RunError("O2 must execute mechanics version 2")
    return canonical_world_row(
        row=row,
        arms=arms,
        checkpoint_sha256=_sha256(Path(checkpoint)),
        cfg=cfg,
        elapsed_seconds=time.monotonic() - started,
    )


def _worker_task(
    payload: tuple[Mapping[str, Any], str, int, int, float, str | None, str | None, str],
) -> dict[str, Any]:
    row, checkpoint, frames, horizon, deadline, catalog, input_root, role = payload
    # Spawn workers require their own checkpoint-path remapping context.
    from research.t34_o1_harvest.runner import runtime_inputs

    with runtime_inputs(catalog, input_root):
        schedule = (
            OpportunitySchedule(time_roots=(10, 20), max_event_roots=0)
            if role == "timing"
            else None
        )
        return evaluate_world(
            row,
            checkpoint,
            frames=frames,
            horizon=horizon,
            deadline=deadline,
            schedule=schedule,
            role=role,
        )


def _parity_task(
    payload: tuple[Mapping[str, Any], str, float, str | None, str | None],
) -> dict[str, Any]:
    """Run O1's instrumented-vs-plain source-preservation parity fixture."""
    row, checkpoint, deadline, catalog, input_root = payload
    from research.t34_o1_harvest.runner import collect, runtime_inputs

    try:
        with runtime_inputs(catalog, input_root):
            return collect(
                dict(row),
                checkpoint,
                frames=500,
                horizon=500,
                time_roots=(),
                event_limit=0,
                deadline=deadline,
                parity=True,
            )
    except BaseException as exc:
        if _scientific_fault(exc):
            raise ScientificFault.from_exception(
                row=row, arm="parity", decision=0, phase="parity", exc=exc
            ) from exc
        raise


def _run_parallel(
    tasks: list[Any],
    worker: Any,
    workers: int,
    *,
    phase: str,
    on_result: Any | None = None,
    executor_factory: Any | None = None,
) -> list[Any]:
    """Merge in order while admitting only one bounded worker wave at a time."""
    if workers == 1:
        values = []
        for index, task in enumerate(tasks):
            value = worker(task)
            if on_result is not None:
                on_result(index, value)
            values.append(value)
        return values
    results: list[Any] = [None] * len(tasks)
    factory = executor_factory or (
        lambda: ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
    )
    pool = factory()
    pending: dict[Any, int] = {}
    next_index = 0
    try:
        while next_index < len(tasks) and len(pending) < workers:
            pending[pool.submit(worker, tasks[next_index])] = next_index
            next_index += 1
        last = time.monotonic()
        while pending:
            done, _ = wait(pending, timeout=120, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if now - last >= 120:
                print(
                    json.dumps(
                        {
                            "event": "t34_o2_progress",
                            "phase": phase,
                            "done": len(tasks) - len(pending),
                            "total": len(tasks),
                        }
                    ),
                    flush=True,
                )
                last = now
            completed: list[tuple[int, Any]] = []
            for future in sorted(done, key=lambda item: pending[item]):
                index = pending.pop(future)
                completed.append((index, future.result()))
            for index, value in completed:
                results[index] = value
                if on_result is not None:
                    on_result(index, value)
            while next_index < len(tasks) and len(pending) < workers:
                pending[pool.submit(worker, tasks[next_index])] = next_index
                next_index += 1
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return results


def canonical_world_row(
    *,
    row: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    checkpoint_sha256: str,
    cfg: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the exact analyzer-facing triple with source/counter evidence."""
    if set(arms) != set(ARMS):
        raise O2RunError("canonical O2 rows require bare, d, and do arms")
    initial = {str(arms[name].get("initial_digest", "")) for name in ARMS}
    if len(initial) != 1 or len(next(iter(initial))) != 64:
        raise O2RunError("O2 paired arms did not start from one exact source digest")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "world": dict(row),
        "checkpoint_sha256": checkpoint_sha256,
        "cfg": dict(cfg),
        "horizon": int(cfg["horizon"]),
        "arms": {name: dict(arms[name]) for name in ARMS},
        "paired": {"source_digest": next(iter(initial))},
        "operational": {"threads": 1, "device": "cpu", "worlds_per_process": 1},
        "elapsed_seconds": float(elapsed_seconds),
    }


def forecast_row_steps(worlds: int = 96, roots: int = 8, actions: int = 6) -> int:
    """Registered upper bound: three base arms plus two shared-tree controllers."""
    if worlds < 1 or roots < 0 or actions < 1:
        raise ValueError("invalid forecast dimensions")
    return worlds * (CAMPAIGN_FRAMES * 3 + roots * actions * HORIZON * 2)


def forecast_admissible(*, forecast_seconds: float, cap_seconds: float, deadline: float) -> bool:
    """Gate study submission before its pool exists; non-finite values fail closed."""
    return (
        math.isfinite(forecast_seconds)
        and math.isfinite(cap_seconds)
        and cap_seconds > 0
        and forecast_seconds <= cap_seconds
        and time.monotonic() + forecast_seconds < deadline
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "campaign"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--input-root", type=Path)
    args = parser.parse_args(argv)
    from research.t34_o2_harvest.registration import CAPS, require_registration

    if args.workers < 1 or args.workers > int(CAPS["workers_per_job"]):
        raise O2RunError("workers exceeds the registered per-job CPU cap")
    if args.mode == "preflight" and args.workers > 2:
        raise O2RunError("Mac preflight permits at most two CPU workers")
    if args.out.exists():
        raise O2RunError("output directory must be fresh")
    registration = require_registration(args.registration)
    bank = registration["bank"]
    rows = list(bank["excluded_preflight_rows" if args.mode == "preflight" else "rows"])
    if not isinstance(rows, list) or not rows:
        raise O2RunError("registration bank has no selected rows")
    timing_rows: list[Mapping[str, Any]] = []
    parity_rows: list[Mapping[str, Any]] = []
    if args.mode == "campaign":
        if args.catalog is None or args.input_root is None:
            raise O2RunError("campaign requires catalog and extracted public input root")
        if args.start < 0 or args.start >= len(rows):
            raise O2RunError("campaign start is outside the registered bank")
        count = len(rows) - args.start if args.count is None else args.count
        if count < 1 or args.start + count > len(rows):
            raise O2RunError("campaign count is outside the registered bank")
        timing_rows = list(bank["excluded_timing_rows"])
        parity_rows = list(bank["excluded_parity_rows"])
        rows = rows[args.start : args.start + count]
    elif args.start or args.count is not None:
        raise O2RunError("preflight always runs exactly its three excluded triples")
    frames = PREFLIGHT_FRAMES if args.mode == "preflight" else CAMPAIGN_FRAMES
    horizon = PREFLIGHT_HORIZON if args.mode == "preflight" else HORIZON
    args.out.mkdir(parents=True)
    started = time.monotonic()
    rows_out = []
    deadline = started + (
        (600.0 if args.workers == 1 else 300.0)
        if args.mode == "preflight"
        else float(CAPS["ordinary_job_seconds"])
    )
    catalog = str(args.catalog) if args.catalog else None
    input_root = str(args.input_root) if args.input_root else None
    timing_rows_out: list[dict[str, str]] = []
    parity_rows_out: list[dict[str, str]] = []
    timing_admission: dict[str, Any] | None = None

    def publish(payload: Mapping[str, Any], name: str, destination: list[dict[str, str]]) -> None:
        path = args.out / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        digest_value = _sha256(path)
        (args.out / f"{name}.sha256").write_text(digest_value + "  " + name + "\n")
        (args.out / f"{name}.complete").write_text("complete\n")
        destination.append({"path": name, "sha256": digest_value})

    def finish_scientific_fault(fault: ScientificFault) -> int:
        details = fault.payload
        fault_payload = {
            "schema": "t34-o2-scientific-fault/v1",
            "world": details["world"],
            "arm": details["arm"],
            "class": _fault_class_name(str(details["exception_class"])),
            "decision": int(details["decision"]),
            "phase": details["phase"],
            "detail": details["detail"],
            "registration_sha256": _sha256(args.registration),
            "source_commit": registration["source_commit"],
            "checkpoint_sha256": _sha256(args.checkpoint),
            "bank_sha256": registration["bank_sha256"],
        }
        fault_path = args.out / "scientific-fault.json"
        publish(fault_payload, fault_path.name, [])
        receipt = {
            "schema": RUN_SCHEMA,
            "status": "SCIENTIFIC_FAULT",
            "mode": args.mode,
            "registration_sha256": _sha256(args.registration),
            "source_commit": registration["source_commit"],
            "bank_sha256": registration["bank_sha256"],
            "checkpoint_sha256": _sha256(args.checkpoint),
            "timing_worlds": sorted(timing_rows_out, key=lambda item: item["path"]),
            "parity_worlds": sorted(parity_rows_out, key=lambda item: item["path"]),
            "parity_pass": len(parity_rows_out) == 3 if args.mode == "campaign" else "NOT_RUN",
            "timing_admission": timing_admission,
            "worlds": sorted(rows_out, key=lambda item: item["path"]),
            "scientific_fault": {"path": fault_path.name, "sha256": _sha256(fault_path)},
            "elapsed_seconds": time.monotonic() - started,
        }
        receipt_path = args.out / "receipt.json"
        with receipt_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
        (args.out / "receipt.json.sha256").write_text(_sha256(receipt_path) + "  receipt.json\n")
        (args.out / "receipt.json.complete").write_text("complete\n")
        print(json.dumps({"receipt": str(receipt_path), "status": "SCIENTIFIC_FAULT"}))
        return 0

    # The three H5000/H500 timing triples are complete before forecast
    # admission.  No study or parity pool is constructed before this returns.
    if args.mode == "campaign":
        timing_payloads = [
            (
                row,
                str(args.checkpoint),
                CAMPAIGN_FRAMES,
                HORIZON,
                deadline,
                catalog,
                input_root,
                "timing",
            )
            for row in timing_rows
        ]
        timing_results = _run_parallel(
            timing_payloads,
            _worker_task,
            args.workers,
            phase="timing",
            on_result=lambda index, payload: publish(
                payload, f"timing-{index:03d}.json", timing_rows_out
            ),
        )
        base_rates, branch_rates = [], []
        for payload in timing_results:
            for arm in payload["arms"].values():
                search_seconds = float(arm["search_elapsed_seconds"])
                base_rates.append(
                    max(0.0, float(arm["elapsed_seconds"]) - search_seconds)
                    / max(1, int(arm["base_row_steps"]))
                )
                if int(arm["branch_row_steps"]):
                    branch_rates.append(search_seconds / int(arm["branch_row_steps"]))
        base_rate = max(base_rates)
        branch_rate = max(branch_rates)
        per_world = CAMPAIGN_FRAMES * 3 * base_rate + 8 * 6 * HORIZON * 2 * branch_rate
        forecast = (
            math.ceil(len(rows) / args.workers) * per_world * 2
            + max(float(item["elapsed_seconds"]) for item in timing_results)
            + 600
        )
        cap = float(os.environ.get("T34_O2_MAX_FORECAST_SECONDS", CAPS["ordinary_job_seconds"]))
        timing_admission = {
            "passed": forecast_admissible(
                forecast_seconds=forecast, cap_seconds=cap, deadline=deadline
            ),
            "forecast_seconds": forecast,
            "cap_seconds": cap,
            "workers": args.workers,
            "study_worlds": len(rows),
            "formula": "max-rates-2x-plus600/v1",
            "base_seconds_per_step": base_rate,
            "branch_seconds_per_step": branch_rate,
            "pre_admission_elapsed_seconds": time.monotonic() - started,
            "ordinary_deadline_seconds": float(CAPS["ordinary_job_seconds"]),
        }
        if not forecast_admissible(forecast_seconds=forecast, cap_seconds=cap, deadline=deadline):
            failure = {
                "schema": RUN_SCHEMA,
                "status": "INCOMPLETE",
                "phase": "forecast",
                "forecast_seconds": forecast,
                "cap_seconds": cap,
                "timing_worlds": timing_rows_out,
                "study_started": False,
            }
            (args.out / "failure.json").write_text(json.dumps(failure, sort_keys=True) + "\n")
            raise O2RunError("timing forecast refuses study before any study world starts")
        parity_payloads = [
            (row, str(args.checkpoint), deadline, catalog, input_root) for row in parity_rows
        ]

        def publish_parity(index: int, payload: Mapping[str, Any]) -> None:
            if payload.get("parity", {}).get("pass") is not True:
                raise O2RunError("instrumented observer parity failed before study")
            publish(payload, f"parity-{index:03d}.json", parity_rows_out)

        _run_parallel(
            parity_payloads, _parity_task, args.workers, phase="parity", on_result=publish_parity
        )
    payloads = [
        (row, str(args.checkpoint), frames, horizon, deadline, catalog, input_root, "study")
        for row in rows
    ]
    try:
        _run_parallel(
            payloads,
            _worker_task,
            args.workers,
            phase="study",
            on_result=lambda index, payload: publish(payload, f"world-{index:03d}.json", rows_out),
        )
    except ScientificFault as fault:
        return finish_scientific_fault(fault)
    receipt = {
        "schema": RUN_SCHEMA,
        "status": "COMPLETE",
        "mode": args.mode,
        "registration_sha256": _sha256(args.registration),
        "source_commit": registration["source_commit"],
        "bank_sha256": registration["bank_sha256"],
        "checkpoint_sha256": _sha256(args.checkpoint),
        "elapsed_seconds": time.monotonic() - started,
        "worlds": sorted(rows_out, key=lambda item: item["path"]),
        "timing_worlds": sorted(timing_rows_out, key=lambda item: item["path"]),
        "parity_worlds": sorted(parity_rows_out, key=lambda item: item["path"]),
        "parity_pass": len(parity_rows_out) == 3 if args.mode == "campaign" else "NOT_RUN",
        "forecast_row_steps": forecast_row_steps(),
        "threads": 1,
        "device": "cpu",
        "workers": args.workers,
        "timing_admission": timing_admission,
    }
    receipt_path = args.out / "receipt.json"
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    (args.out / "receipt.json.sha256").write_text(_sha256(receipt_path) + "  receipt.json\n")
    (args.out / "receipt.json.complete").write_text("complete\n")
    print(json.dumps({"receipt": str(receipt_path), "elapsed_seconds": receipt["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
