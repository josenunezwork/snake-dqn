"""Bounded shadow/LAD teacher, extending the existing ExIt engine and exports.

Worlds never rewind. A prospective label or scheduled action is retained only
while its exact shadow prefix remains valid. The original Amendment-1 helper
selects every probe; continuation work uses ExIt's compacting branch engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.t34_exit_b8.engine import ExitTeacher, ExitTeacherConfig, mix_action
from research.t34_exit_b8.exports import state_hash, write_world
from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world
from research.t34_shadow.lad import search_lad_teacher
from research.t34_shadow.shadow import ShadowTrajectory, clone_world_exact

KERNEL = "shadow-lad-v2"
LABEL_RULE = "teacher-stride10-transition-refine5"
WORK_KEYS = ("root_row_steps", "shadow_row_steps", "reconstruction_row_steps", "search_row_steps")


def _outcome_dict(outcomes: Mapping[int, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(a): {
            "survives": bool(v.survives),
            "alive_steps": int(v.alive_steps),
            "mass": float(v.mass),
            "row_steps": int(v.row_steps),
        }
        for a, v in outcomes.items()
    }


def _step(world: Any, teacher: ExitTeacher, hero_action: int) -> list[int]:
    """Advance the normal prepared six-snake dispatcher once."""
    prepare_world(world)
    grid = teacher.actions(world, hero_action)
    world.sim.step(grid)
    world.runtime.observe_step(world.sim)
    return np.asarray(grid[0], dtype=np.int64).tolist()


def _shadow_stepper(teacher: ExitTeacher):
    def stepper(world: Any) -> tuple[np.ndarray, np.ndarray, None]:
        prepare_world(world)
        evaluation = teacher.evaluate(world)
        grid = np.asarray(_step(world, teacher, evaluation.greedy_action()), dtype=np.int64)[None]
        return grid, np.asarray(evaluation.legal, dtype=bool), None

    return stepper


def _label(
    *,
    decision: int,
    evaluation: Any,
    student_action: int,
    role: str,
    shadow_id: int,
    shadow_frame: int,
    shadow_confirmed: bool,
    searched: bool = False,
    decisive: bool = False,
    kl_anchor_only: bool = False,
    t_star: int | None = None,
    a_star: int | None = None,
    forbidden: list[int] | None = None,
    per_action: Mapping[str, Any] | None = None,
    lad: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "decision": int(decision),
        "label_role": role,
        "searched": bool(searched),
        "decisive": bool(decisive),
        "kl_anchor_only": bool(kl_anchor_only),
        "student_action": int(student_action),
        "action": int(a_star if a_star is not None else student_action),
        "executed_action": int(student_action),
        "q": [float(v) for v in evaluation.q],
        "legal": [bool(v) for v in evaluation.legal],
        "shadow_id": int(shadow_id),
        "shadow_frame": int(shadow_frame),
        "shadow_confirmed": bool(shadow_confirmed),
        "t_star": t_star,
        "a_star": a_star,
        "F": list(forbidden or []),
        "forbidden": list(forbidden or []),
        "outcomes": dict(per_action or {}),
        "per_action": dict(per_action or {}),
        "lad": dict(lad or {}),
        "reason": role,
        "expanded": bool(searched),
    }


def _put_label(labels: dict[int, dict[str, Any]], item: dict[str, Any]) -> None:
    """Canonicalize one row per replay decision: decisive > sparse > anchor."""

    def rank(row: Mapping[str, Any]) -> int:
        return 2 if row.get("decisive") else (0 if row.get("kl_anchor_only") else 1)

    old = labels.get(int(item["decision"]))
    if old is None or rank(item) > rank(old):
        labels[int(item["decision"])] = item


def _search_dooms(
    states: list[dict[str, Any]],
    teacher: ExitTeacher,
    branch_bucket_size: int = 64,
    deadline: float = math.inf,
) -> None:
    """Batch coarse and refinement queries across independent doomed worlds.

    The helper is run once with dummy negative refinement results to discover
    its query set, then again with exact memoized results. Coarse edge flags
    alone determine refinement locations, so no predicate is approximated in
    the final labels. Snapshots are reconstructed lazily in bounded buckets.
    """
    if not 6 <= branch_bucket_size <= 1536:
        raise ValueError("branch bucket size must lie in [6, 1536]")
    memos: list[dict[int, dict[str, Any]]] = [{} for _ in states]
    for state in states:
        state.pop("sparse_decision", None)
    intervals = [
        (int(s["decision"]), int(s["decision"]) + len(s["shadow"].frames) - 1) for s in states
    ]

    def check() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("v2 LAD deadline reached")

    def evaluate(requests: list[tuple[int, int]]) -> None:
        # Every query has at most six legal alternatives, including greedy.
        width = max(1, branch_bucket_size // 6)
        for start in range(0, len(requests), width):
            check()
            batch = requests[start : start + width]
            sources, choices, evaluations = [], [], []
            for index, decision in batch:
                state = states[index]
                stage = time.perf_counter()
                source = clone_world_exact(state["world"])
                ambient = random.getstate()
                try:
                    random.setstate(state["rng"])
                    for frame in state["shadow"].frames[: decision - state["decision"]]:
                        check()
                        _step(source, teacher, frame.action)
                        state["work"]["reconstruction_row_steps"] += 1
                    prepare_world(source)
                    evaluation = teacher.evaluate(source)
                finally:
                    random.setstate(ambient)
                state["stages"]["reconstruction_s"] += time.perf_counter() - stage
                ordered = sorted(
                    np.flatnonzero(evaluation.legal),
                    key=lambda a: (-float(evaluation.q[a]), int(a)),
                )
                sources.append(source)
                choices.append([int(a) for a in ordered])
                evaluations.append(evaluation)
            stage = time.perf_counter()
            if all(hasattr(source.sim, "cfg") for source in sources):
                outcomes = teacher._rollout_merged(sources, choices)
            else:
                outcomes = [
                    teacher._rollout(source, actions) for source, actions in zip(sources, choices)
                ]
            elapsed = time.perf_counter() - stage
            for (index, decision), evaluation, outcome in zip(batch, evaluations, outcomes):
                state = states[index]
                work = sum(int(v.row_steps) for v in outcome.values())
                state["work"]["search_row_steps"] += work
                state["stages"]["branch_s"] += elapsed / len(batch)
                state["search_bucket_peak"] = max(
                    state["search_bucket_peak"], sum(map(len, choices))
                )
                memos[index][decision] = {
                    "greedy": evaluation.greedy_action(),
                    "q": [float(v) for v in evaluation.q],
                    "outcomes": _outcome_dict(outcome),
                    "evaluation": evaluation,
                }

    coarse = [
        (i, t)
        for i, (first, doom) in enumerate(intervals)
        for t in sorted({*range(first, doom + 1, 10), doom})
    ]
    evaluate(coarse)
    refinement: set[tuple[int, int]] = set()
    for index, (first, doom) in enumerate(intervals):

        def discover(t: int, i: int = index) -> dict[str, Any]:
            if t in memos[i]:
                return memos[i][t]
            refinement.add((i, t))
            return {"greedy": 0, "q": [0.0] * 6, "outcomes": {}}

        search_lad_teacher(first, doom, discover)
    evaluate(sorted(refinement))
    results = [
        search_lad_teacher(first, doom, memos[i].__getitem__)
        for i, (first, doom) in enumerate(intervals)
    ]
    sparse_requests = []
    for index, result in enumerate(results):
        first, _ = intervals[index]
        latest = result["t_star"]
        if latest is not None and latest > first:
            token = hashlib.sha256(
                f"v2-sparse:{states[index]['seed']}:{first}:{latest}".encode()
            ).digest()
            sample = first + int.from_bytes(token[:8], "big") % (latest - first)
            states[index]["sparse_decision"] = sample
            if sample not in memos[index]:
                sparse_requests.append((index, sample))
    evaluate(sparse_requests)
    for index, (state, result) in enumerate(zip(states, results)):
        memo, latest = memos[index], result["t_star"]
        decisions = {
            int(p["decision"])
            for p in result["probes"]
            if p["surviving_alternatives"]
            and int(p["decision"]) + 1 in memo
            and not any(
                v["survives"]
                for a, v in memo[int(p["decision"]) + 1]["outcomes"].items()
                if int(a) != memo[int(p["decision"]) + 1]["greedy"]
            )
        }
        if latest is not None:
            decisions.add(int(latest))
        decisive_decisions = set(decisions)
        sample = state.get("sparse_decision")
        if sample is not None:
            decisions.add(sample)
        for decision in sorted(decisions):
            value = memo[decision]
            survivors = sorted(
                (
                    int(a)
                    for a, v in value["outcomes"].items()
                    if v["survives"] and int(a) != value["greedy"]
                ),
                key=lambda a: (-value["q"][a], a),
            )
            decisive = decision in decisive_decisions
            action = survivors[0] if survivors else value["greedy"]
            forbidden = (
                [int(a) for a, v in value["outcomes"].items() if not v["survives"]]
                if survivors
                else []
            )
            item = _label(
                decision=decision,
                evaluation=value["evaluation"],
                student_action=value["greedy"],
                role=(
                    "lad"
                    if decision == latest
                    else ("lad_transition" if decisive else "sparse_before_lad")
                ),
                shadow_id=state["shadow_id"],
                shadow_frame=decision,
                shadow_confirmed=True,
                searched=True,
                decisive=decisive,
                t_star=latest,
                a_star=action,
                forbidden=forbidden,
                per_action=value["outcomes"],
                lad={
                    "rule": LABEL_RULE,
                    "latest_surviving_decision": latest,
                    "transitions": result["transitions"],
                    "expansion_count": result["expansion_count"],
                    "global_last_avoidable_proven": False,
                },
            )
            item["length"] = 0  # Filled from the actual root when this decision is reached.
            state["pending_labels"][decision] = item
        if latest is not None:
            state["pending_actions"][int(latest)] = int(result["a_star"])
        state["searched"] = True
        state["searches"] += 1


def run_worlds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect bounded resident worlds with deterministic, prospective v2 labels."""
    started = time.monotonic()
    depth, frames = int(config.get("shadow_depth", 300)), int(config.get("frames", 5000))
    beta, seconds = float(config.get("beta", 1.0)), float(config.get("seconds", 1500))
    capacity = int(config.get("worlds_per_process", config.get("worlds", 256)))
    bucket = int(config.get("branch_bucket_size", 64))
    if not 1 <= depth <= 500 or not 1 <= frames <= 5000 or beta not in (0.0, 0.5, 1.0):
        raise ValueError("shadow depth, frames or beta exceeds v2 bounds")
    if (
        not math.isfinite(seconds)
        or seconds <= 0
        or not 1 <= capacity <= 256
        or not 6 <= bucket <= 1536
    ):
        raise ValueError("invalid wall, root capacity or branch bucket bound")
    rows = list(config.get("rows") or make_world_rows(config, int(config.get("max_worlds", 2))))
    if not rows or len({r["world_id"] for r in rows}) != len(rows):
        raise ValueError("world ids must be unique and nonempty")
    checkpoint = Path(str(config["checkpoint"]))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rollout_sha = hashlib.sha256(
        Path(str(config.get("student_checkpoint") or checkpoint)).read_bytes()
    ).hexdigest()
    teacher = config.get("teacher") or ExitTeacher(
        ExitTeacherConfig(horizon=int(config.get("horizon", 300)), beta=beta),
        device=str(config.get("device", "cpu")),
        policy=config.get("policy"),
    )
    deadline = started + seconds
    teacher.deadline = deadline
    output = Path(str(config["output_dir"]))
    if output.exists() and any(output.iterdir()):
        raise ValueError("v2 output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    states, all_states = [], []
    iterator = iter(enumerate(rows))
    done = decisions_done = export_bytes = peak = root_batch_peak = 0
    stages: dict[str, float] = defaultdict(float)
    mixes = {str(r["mix"]): {"searched": 0, "expanded": 0, "worlds": 0} for r in rows}
    heartbeat = started
    gpu_samples: list[float] = []

    def check() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("v2 collection deadline reached")

    def progress() -> None:
        nonlocal heartbeat
        now = time.monotonic()
        if now - heartbeat < 110:
            return
        sampler = config.get("gpu_utilization_sampler")
        sample = sampler() if sampler else None
        if sample is not None:
            gpu_samples.append(float(sample))
        print(
            json.dumps(
                {
                    "event": "t34_v2_progress",
                    "done": done,
                    "total": len(rows),
                    "resident_roots": len(states),
                    "elapsed_seconds": now - started,
                    "executed_row_steps": sum(sum(s["work"].values()) for s in all_states),
                }
            ),
            flush=True,
        )
        heartbeat = now

    teacher.progress = progress

    def fork(state: dict[str, Any]) -> None:
        state["shadow_id"] += 1
        state["pending_labels"].clear()
        state["pending_actions"].clear()
        state["searched"] = False
        state.pop("sparse_decision", None)
        state["shadow"].fork(state["world"], global_rng_state=state["rng"])

    def catch_up(state: dict[str, Any]) -> None:
        shadow = state["shadow"]
        before = len(shadow.frames)
        stage = time.perf_counter()
        try:
            shadow.catch_up(deadline=deadline)
        finally:
            state["work"]["shadow_row_steps"] += len(shadow.frames) - before
            state["stages"]["shadow_s"] += time.perf_counter() - stage

    try:
        while True:
            check()
            while len(states) < capacity:
                pair = next(iterator, None)
                if pair is None:
                    break
                check()
                index, row = pair
                stage = time.perf_counter()
                world = config.get("world_builder", build_world)(row, checkpoint)
                stages["build_s"] += time.perf_counter() - stage
                seed = int(row.get("world_seed", row.get("seed")))
                state = {
                    "index": index,
                    "row": row,
                    "world": world,
                    "seed": seed,
                    "decision": 0,
                    "shadow_id": -1,
                    "labels": {},
                    "pending_labels": {},
                    "pending_actions": {},
                    "searched": False,
                    "searches": 0,
                    "search_bucket_peak": 0,
                    "actions": [],
                    "hashes": {"0": state_hash(world.sim)},
                    "work": defaultdict(int),
                    "stages": defaultdict(float),
                    "rng": random.Random(seed).getstate(),
                    "divergences": 0,
                    "interventions": 0,
                }
                state["shadow"] = ShadowTrajectory(
                    depth,
                    checkpoint=(checkpoint_sha, rollout_sha),
                    stepper=_shadow_stepper(teacher),
                )
                fork(state)
                states.append(state)
                all_states.append(state)
            peak = max(peak, len(states))
            if not states:
                break
            for state in states:
                if not state["shadow"].confirmed:
                    fork(state)
                catch_up(state)
                prepare_world(state["world"])
                progress()
            doomed = [
                s
                for s in states
                if not s["searched"]
                and s["shadow"].world is not None
                and not bool(s["shadow"].world.sim.alive[0, 0])
            ]
            if doomed:
                _search_dooms(doomed, teacher, bucket, deadline)
            stage = time.perf_counter()
            evaluator = getattr(teacher.policy, "evaluate_worlds", None)
            evaluations = (
                evaluator([s["world"] for s in states], bucket_size=64)
                if callable(evaluator)
                else [teacher.evaluate(s["world"]) for s in states]
            )
            stages["root_policy_s"] += time.perf_counter() - stage
            root_batch_peak = max(root_batch_peak, min(64, len(states)))
            remaining = []
            for state, evaluation in zip(states, evaluations):
                check()
                world, shadow, decision = state["world"], state["shadow"], state["decision"]
                student = evaluation.greedy_action()
                if not shadow.frames or shadow.frames[0].action != student:
                    shadow.invalidate("student_action_mismatch")
                    state["pending_labels"].clear()
                    state["pending_actions"].clear()
                valid = shadow.confirmed
                proposed = state["pending_actions"].pop(decision, student) if valid else student
                executed = mix_action(
                    student, proposed, seed=state["seed"], decision=decision, beta=beta
                )
                pending = state["pending_labels"].pop(decision, None) if valid else None
                length = int(world.sim.length[0, 0])
                stage = time.perf_counter()
                ambient = random.getstate()
                try:
                    random.setstate(state["rng"])
                    grid = _step(world, teacher, executed)
                    state["rng"] = random.getstate()
                    matched = shadow.confirm(world) if executed == student else False
                finally:
                    random.setstate(ambient)
                state["stages"]["root_step_s"] += time.perf_counter() - stage
                state["work"]["root_row_steps"] += 1
                state["actions"].append(grid)
                state["decision"] += 1
                if valid and (matched or executed != student):
                    item = pending or _label(
                        decision=decision,
                        evaluation=evaluation,
                        student_action=student,
                        role="shadow_confirm",
                        shadow_id=state["shadow_id"],
                        shadow_frame=decision,
                        shadow_confirmed=True,
                        kl_anchor_only=True,
                    )
                    item.update(executed_action=executed, length=length)
                    _put_label(state["labels"], item)
                if executed != student:
                    state["interventions"] += 1
                    shadow.invalidate("teacher_intervention")
                elif not matched:
                    state["divergences"] += 1
                    shadow.invalidate("full_digest_mismatch")
                if not shadow.confirmed:
                    state["pending_labels"].clear()
                    state["pending_actions"].clear()
                count = state["decision"]
                if count % 250 == 0:
                    state["hashes"][str(count)] = state_hash(world.sim)
                if bool(world.sim.alive[0, 0]) and count < frames:
                    remaining.append(state)
                    continue
                state["hashes"][str(count)] = state_hash(world.sim)
                work = {key: int(state["work"][key]) for key in WORK_KEYS}
                total = sum(work.values())
                result = {
                    "schema": "t34-exit-world/v1",
                    "world_id": state["row"]["world_id"],
                    "seed": state["seed"],
                    "mix": state["row"]["mix"],
                    "row": state["row"],
                    "checkpoint_sha256": checkpoint_sha,
                    "rollout_policy_sha256": rollout_sha,
                    "actions": state["actions"],
                    "hashes": state["hashes"],
                    "labels": [state["labels"][key] for key in sorted(state["labels"])],
                    "rewinds": 0,
                    "complete": True,
                    "frames": frames,
                    "horizon": teacher.config.horizon,
                    "beta": beta,
                    "profile": {
                        "decisions": count,
                        "active_row_steps": total,
                        "resident_row_steps": total,
                    },
                    "rng_stream_ids": {
                        "food": state["seed"],
                        "scripted": state["seed"],
                        "python": state["seed"],
                        "sample": "sha256:v2-sparse:seed:first:latest",
                        "beta": "sha256:t34-exit-v1:seed:decision:beta",
                    },
                    "teacher_v2": {
                        "kernel": KERNEL,
                        "shadow_depth": depth,
                        "beta": beta,
                        "label_rule": LABEL_RULE,
                        "no_rewinds": True,
                        "action_trace_width": 6,
                        "hash_interval": 250,
                        "shadow_searches": state["searches"],
                        "divergences": state["divergences"],
                        "interventions": state["interventions"],
                        "work": work,
                    },
                }
                stage = time.perf_counter()
                path = output / f"world-{state['index']:06d}.json"
                write_world(path, result)
                stages["export_s"] += time.perf_counter() - stage
                export_bytes += path.stat().st_size
                done += 1
                decisions_done += count
                mixes[str(state["row"]["mix"])]["worlds"] += 1
                mixes[str(state["row"]["mix"])]["searched"] += state["searches"]
                mixes[str(state["row"]["mix"])]["expanded"] += state["searches"]
                # Retain only telemetry for finished worlds, not simulators,
                # models or full traces from the completed chunk prefix.
                for key in (
                    "world",
                    "shadow",
                    "labels",
                    "pending_labels",
                    "pending_actions",
                    "actions",
                    "hashes",
                    "rng",
                ):
                    state.pop(key, None)
            states = remaining
            progress()
    except TimeoutError:
        pass
    elapsed = time.monotonic() - started
    work = {key: sum(int(s["work"][key]) for s in all_states) for key in WORK_KEYS}
    # The branch engine counts partial timed-out expansions too. Outcomes are
    # available only for finished buckets; use its physical counter for totals.
    work["search_row_steps"] = int(
        teacher.profile().get("active_row_steps", work["search_row_steps"])
    )
    total = sum(work.values())
    for state in all_states:
        for key, value in state["stages"].items():
            stages[key] += value
    import numba

    return {
        "kernel": KERNEL,
        "label_rule": LABEL_RULE,
        "shadow_depth": depth,
        "no_rewinds": True,
        "worlds": done,
        "requested_worlds": len(rows),
        "decisions": decisions_done,
        "complete": done == len(rows),
        "elapsed_seconds": elapsed,
        "decisions_per_second": decisions_done / max(elapsed, 1e-9),
        "mixes": mixes,
        "worlds_per_hour": done * 3600 / max(elapsed, 1e-9),
        "export_bytes": export_bytes,
        "export_bytes_per_world": export_bytes / done if done else None,
        "executed_row_steps": total,
        "executed_row_steps_per_world": total / done if done else None,
        "active_row_steps": total,
        "resident_row_steps": total,
        "active_row_steps_per_second": total / max(elapsed, 1e-9),
        "active_resident_fraction": 1.0,
        "configured_root_capacity": capacity,
        "peak_root_worlds": peak,
        "search_batch_peak": max((s["search_bucket_peak"] for s in all_states), default=0),
        "root_inference_bucket_peak": root_batch_peak,
        "branch_bucket_capacity": bucket,
        "teacher_profile": {
            **teacher.profile(),
            **work,
            "active_row_steps": total,
            "resident_row_steps": total,
        },
        "host_stage_seconds": dict(stages),
        "gpu_utilization_samples_percent": gpu_samples,
        "actual_numba_threads": int(numba.get_num_threads()),
        "actual_policy_device": str(teacher.device),
        "world_rate_scope": "completed worlds; work includes unfinished roots",
        "unfinished_worlds": len(rows) - done,
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
    }
