"""Bounded shadow/LAD teacher, extending the existing ExIt engine and exports.

Worlds never rewind. A prospective label or scheduled action is retained only
while its exact shadow prefix remains valid. Amendment 2's radius-three
bisection selects probes; continuation work uses ExIt's compacting branch engine.
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
from research.t34_exit_b8.lockstep import step_prepared_worlds
from research.t34_exit_b8.trajectory import build_world, make_world_rows, prepare_world
from research.t34_shadow.lad import search_lad
from research.t34_shadow.shadow import ShadowFrame, ShadowTrajectory, clone_world_exact, full_digest

KERNEL = "shadow-lad-v2"
LABEL_RULE = "bisection-radius3-prepared-cache"
WORK_KEYS = ("root_row_steps", "shadow_row_steps", "reconstruction_row_steps", "search_row_steps")


class ShadowControlMismatch(RuntimeError):
    """Batched greedy control did not reproduce the confirmed shadow's doom."""


class _MissingProbe(Exception):
    def __init__(self, decision: int) -> None:
        self.decision = decision


def _outcome_dict(outcomes: Mapping[int, Any], *, horizon: int) -> dict[str, dict[str, Any]]:
    return {
        str(a): {
            "survives": bool(v.survives),
            "alive_steps": int(v.alive_steps),
            "mass": float(v.mass),
            "row_steps": int(v.row_steps),
            "horizon": int(horizon),
            # A surviving branch has reached the fixed H horizon and is
            # therefore right-censored; a death is observed, not censored.
            "censored": bool(v.survives),
            "tested": True,
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


def _shadow_evaluations(worlds: list[Any], teacher: ExitTeacher) -> list[Any]:
    """Return batched shadow Qs, recovering the serial decision around ties.

    The historical shadow path used one-world forwards.  Only its selected
    action is observable, so a batch forward is safe except at a narrow margin
    where a batch-N fp32 reduction can reverse the stable serial tie rule.
    Root and branch paths intentionally keep their pre-existing recipes.
    """
    evaluator = getattr(teacher.policy, "evaluate_worlds", None)
    if not callable(evaluator) or len(worlds) == 1:
        return [teacher.evaluate(world) for world in worlds]
    bucket_size = len(worlds) if str(teacher.device).startswith("cuda") else 64
    evaluations = evaluator(worlds, bucket_size=bucket_size)
    for index, evaluation in enumerate(evaluations):
        select = np.asarray(
            evaluation.legal if evaluation.select is None else evaluation.select, dtype=bool
        )
        values = np.asarray(evaluation.q, dtype=np.float32)[select]
        if len(values) > 1:
            ordered = np.sort(values)
            scale = max(1.0, float(np.max(np.abs(values))))
            if float(ordered[-1] - ordered[-2]) <= scale * 64.0 * np.finfo(np.float32).eps:
                evaluations[index] = teacher.evaluate(worlds[index])
    return evaluations


def _catch_up_lockstep(states: list[dict[str, Any]], teacher: ExitTeacher, deadline: float) -> None:
    """Cold-fill or refill every active shadow in compacted batch rounds."""
    if not all(
        isinstance(state["shadow"], ShadowTrajectory) and hasattr(state["shadow"].world.sim, "cfg")
        for state in states
        if state["shadow"].world is not None
    ):
        for state in states:
            shadow = state["shadow"]
            before, stage = len(shadow.frames), time.perf_counter()
            try:
                shadow.catch_up(deadline=deadline)
            finally:
                state["work"]["shadow_row_steps"] += len(shadow.frames) - before
                state["stages"]["shadow_s"] += time.perf_counter() - stage
        return
    active = [
        state
        for state in states
        if state["shadow"].confirmed
        and state["shadow"].world is not None
        and len(state["shadow"].frames) < state["shadow"].depth
        and bool(state["shadow"].world.sim.alive[0, 0])
    ]
    while active:
        if time.monotonic() >= deadline:
            raise TimeoutError("v2 shadow deadline reached")
        stage = time.perf_counter()
        worlds = [state["shadow"].world for state in active]
        for world in worlds:
            prepare_world(world)
        ambient = random.getstate()
        shadow_rng = active[0]["shadow"]._global_rng_state
        try:
            if shadow_rng is not None:
                random.setstate(shadow_rng)
            evaluations = _shadow_evaluations(worlds, teacher)
            actions = [evaluation.greedy_action() for evaluation in evaluations]
            grids = step_prepared_worlds(worlds, actions, teacher)
            if shadow_rng is not None and random.getstate() != shadow_rng:
                raise ShadowControlMismatch("batch shadow path consumed process-global RNG")
        finally:
            random.setstate(ambient)
        for state, world, action, _grid in zip(active, worlds, actions, grids):
            shadow = state["shadow"]
            shadow.frames.append(
                ShadowFrame(
                    action=int(action),
                    digest=full_digest(
                        world,
                        checkpoint=shadow.checkpoint,
                        global_rng_state=shadow._global_rng_state,
                    ),
                    frame=int(world.sim.frame[0]),
                    hero_alive=bool(world.sim.alive[0, 0]),
                    hero_length=float(world.sim.length[0, 0]),
                )
            )
            state["work"]["shadow_row_steps"] += 1
        elapsed = time.perf_counter() - stage
        for state in active:
            state["stages"]["shadow_s"] += elapsed / len(active)
        progress = getattr(teacher, "progress", None)
        if callable(progress):
            progress()
        active = [
            state
            for state in active
            if len(state["shadow"].frames) < state["shadow"].depth
            and bool(state["shadow"].world.sim.alive[0, 0])
        ]


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
    label = {
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
    kind = {
        "shadow_confirm": "shadow_confirm",
        "lad": "lad",
        "lad_transition": "transition",
        # Searched sparse rows carry Amendment-2 ranked outcomes too.
        "sparse_before_lad": "transition" if searched else None,
    }.get(role)
    if kind is not None:
        label["label_kind"] = kind
    return label


def _put_label(labels: dict[int, dict[str, Any]], item: dict[str, Any]) -> None:
    """Canonicalize one row per replay decision: decisive > sparse > anchor."""

    def rank(row: Mapping[str, Any]) -> int:
        return 2 if row.get("decisive") else (0 if row.get("kl_anchor_only") else 1)

    old = labels.get(int(item["decision"]))
    if old is None or rank(item) > rank(old):
        labels[int(item["decision"])] = item


def _rank_outcomes(
    outcomes: Mapping[str, Mapping[str, Any]], *, q: list[float], bare_action: int
) -> dict[str, Any]:
    """Apply Amendment 2's stable full-action ranking and label semantics."""
    bare = int(bare_action)
    tested = [int(action) for action, value in outcomes.items() if value.get("tested", True)]
    survivors = [action for action in tested if outcomes[str(action)]["survives"]]

    def dead_rank(action: int) -> tuple[int, float, bool, int]:
        return (
            -int(outcomes[str(action)]["alive_steps"]),
            -float(q[action]),
            action != bare,
            action,
        )

    if bare in survivors:
        ranked = [bare] + sorted((a for a in survivors if a != bare), key=lambda a: (-q[a], a))
        ranked += sorted((a for a in tested if a not in survivors), key=dead_rank)
        kind, decisive = "bare_survives", False
    elif survivors:
        ranked = sorted(survivors, key=lambda a: (-q[a], a != bare, a))
        ranked += sorted((a for a in tested if a not in survivors), key=dead_rank)
        kind, decisive = "horizon_rescue", True
    else:
        ranked = sorted(tested, key=dead_rank)
        kind, decisive = "all_die_delay", False
    return {
        "action": ranked[0],
        "ranked_actions": ranked,
        "outcome_kind": kind,
        "delay_only": kind == "all_die_delay",
        "label_weight": 0.0 if kind == "all_die_delay" else 1.0,
        "decisive": decisive,
        "forbidden": ([a for a in tested if not outcomes[str(a)]["survives"]] if survivors else []),
    }


def _build_doom_cache(state: dict[str, Any], teacher: ExitTeacher, deadline: float) -> list[Any]:
    """Capture prepared roots once for a confirmed doom; never replay per query.

    Callers retain at most two such caches.  The capture is deliberately
    released after the bisection group, preventing D*W resident simulator
    clones while preserving the M2 saving at every individual branch query.
    """
    source = clone_world_exact(state["world"])
    cache = []
    ambient = random.getstate()
    try:
        random.setstate(state["rng"])
        prepare_world(source)
        cache.append(clone_world_exact(source))
        for frame in state["shadow"].frames:
            if time.monotonic() >= deadline:
                raise TimeoutError("v2 doom-cache deadline reached")
            _step(source, teacher, int(frame.action))
            state["work"]["reconstruction_row_steps"] += 1
            if (
                full_digest(
                    source,
                    checkpoint=state["shadow"].checkpoint,
                    global_rng_state=random.getstate(),
                )
                != frame.digest
            ):
                raise ShadowControlMismatch("doom-cache capture diverged from confirmed shadow")
            prepare_world(source)
            cache.append(clone_world_exact(source))
    finally:
        random.setstate(ambient)
    state["doom_cache_digest"] = tuple(frame.digest for frame in state["shadow"].frames)
    return cache


def _build_doom_caches(states: list[dict[str, Any]], teacher: ExitTeacher, deadline: float) -> None:
    """Reconstruct confirmed dooms in compacted cross-world batch rounds."""
    sources = [clone_world_exact(state["world"]) for state in states]
    for state, source in zip(states, sources):
        prepare_world(source)
        state["doom_cache"] = [clone_world_exact(source)]
    max_depth = max((len(state["shadow"].frames) for state in states), default=0)
    for offset in range(max_depth):
        if time.monotonic() >= deadline:
            raise TimeoutError("v2 doom-cache deadline reached")
        active = [
            index for index, state in enumerate(states) if offset < len(state["shadow"].frames)
        ]
        if not active:
            break
        worlds = [sources[index] for index in active]
        for world in worlds:
            prepare_world(world)
        grids = step_prepared_worlds(
            worlds, [states[index]["shadow"].frames[offset].action for index in active], teacher
        )
        del grids
        for index in active:
            state, source = states[index], sources[index]
            frame = state["shadow"].frames[offset]
            state["work"]["reconstruction_row_steps"] += 1
            if (
                full_digest(
                    source,
                    checkpoint=state["shadow"].checkpoint,
                    global_rng_state=state["rng"],
                )
                != frame.digest
            ):
                raise ShadowControlMismatch("doom-cache capture diverged from confirmed shadow")
            prepare_world(source)
            state["doom_cache"].append(clone_world_exact(source))
    for state in states:
        state["doom_cache_digest"] = tuple(frame.digest for frame in state["shadow"].frames)


def _search_dooms(
    states: list[dict[str, Any]],
    teacher: ExitTeacher,
    branch_bucket_size: int = 64,
    deadline: float = math.inf,
) -> None:
    """Run adaptive radius-three bisection in compacted cross-world rounds.

    Every query takes a clone of a prepared, once-captured doom-cache root;
    no query replays its prefix or rerolls the free greedy control.
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
                offset = decision - state["decision"]
                source = clone_world_exact(state["doom_cache"][offset])
                prepare_world(source)
                evaluation = teacher.evaluate(source)
                state["stages"]["reconstruction_s"] += time.perf_counter() - stage
                greedy = evaluation.greedy_action()
                ordered = sorted(
                    (a for a in np.flatnonzero(evaluation.legal) if int(a) != greedy),
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
                greedy = evaluation.greedy_action()
                offset = decision - state["decision"]
                expected_steps = intervals[index][1] - decision + 1
                frames = state["shadow"].frames[offset : offset + expected_steps]
                observed = frames[: teacher.config.horizon]
                control = {
                    "survives": len(frames) > teacher.config.horizon
                    and all(getattr(frame, "hero_alive", True) for frame in observed),
                    "alive_steps": sum(
                        int(getattr(frame, "hero_alive", True)) for frame in observed
                    ),
                    "mass": sum(
                        (
                            float(getattr(frame, "hero_length", 0.0))
                            if getattr(frame, "hero_alive", True)
                            else 0.0
                        )
                        for frame in observed
                    ),
                    "row_steps": 0,
                    "cached_control": True,
                    "horizon": teacher.config.horizon,
                    "censored": len(frames) > teacher.config.horizon,
                    "tested": True,
                }
                packed = _outcome_dict(outcome, horizon=teacher.config.horizon)
                packed[str(greedy)] = control
                memos[index][decision] = {
                    "greedy": greedy,
                    "q": [float(v) for v in evaluation.q],
                    "outcomes": packed,
                    "evaluation": evaluation,
                }

    # Bisection probes are discovered from cached roots.  Seed each interval
    # with its endpoints, then let the registered radius-three search request
    # only the adaptive midpoint/refinement positions.
    coarse = list(
        dict.fromkeys(
            (i, interval[endpoint]) for endpoint in (0, 1) for i, interval in enumerate(intervals)
        )
    )
    evaluate(coarse)
    results: list[dict[str, Any] | None] = [None] * len(states)
    # A bisection's next midpoint depends on the real previous answer.  Run
    # missing probes in rounds: every unfinished root contributes one request
    # to the next compacted cross-world batch, then repeat until all searches
    # have complete memoized paths.
    while any(value is None for value in results):
        requested: list[tuple[int, int]] = []
        for index, (first, doom) in enumerate(intervals):
            if results[index] is not None:
                continue

            def lookup(t: int, i: int = index) -> dict[str, Any]:
                if t not in memos[i]:
                    raise _MissingProbe(t)
                return memos[i][t]

            try:
                results[index] = search_lad(first, doom, lookup, refinement_radius=3)
            except _MissingProbe as missing:
                requested.append((index, missing.decision))
        if not requested:
            break
        evaluate(sorted(set(requested)))
    if any(value is None for value in results):
        raise RuntimeError("incomplete bisection probe schedule")
    results = [value for value in results if value is not None]
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
        # Every bisection probe is retained: it describes the rescue island
        # rather than pretending the latest point is the sole decisive row.
        decisions = {int(p["decision"]) for p in result["probes"]}
        sample = state.get("sparse_decision")
        if sample is not None:
            decisions.add(sample)
        for decision in sorted(decisions):
            value = memo[decision]
            outcomes = value["outcomes"]
            ranked = _rank_outcomes(outcomes, q=value["q"], bare_action=value["greedy"])
            action = ranked["action"]
            kind, decisive, forbidden = (
                ranked["outcome_kind"],
                ranked["decisive"],
                ranked["forbidden"],
            )
            item = _label(
                decision=decision,
                evaluation=value["evaluation"],
                student_action=value["greedy"],
                role=(
                    "sparse_before_lad"
                    if decision == sample
                    else ("lad" if decision == latest else "lad_transition")
                ),
                shadow_id=state["shadow_id"],
                shadow_frame=decision,
                shadow_confirmed=True,
                searched=True,
                decisive=decisive,
                t_star=decision,
                a_star=action,
                forbidden=forbidden,
                per_action=value["outcomes"],
                lad={
                    "rule": LABEL_RULE,
                    "latest_surviving_decision": latest,
                    "boundary": result["boundary"],
                    "refinement_radius": result["refinement_radius"],
                    "expansion_count": result["expansion_count"],
                    "global_last_avoidable_proven": False,
                },
            )
            item["length"] = 0  # Filled from the actual root when this decision is reached.
            item.update(
                outcome_kind=kind,
                delay_only=kind == "all_die_delay",
                label_weight=0.0 if kind == "all_die_delay" else 1.0,
                ranked_actions=ranked["ranked_actions"],
                horizon=teacher.config.horizon,
            )
            state["pending_labels"][decision] = item
            # Delay rows remain labels only.  The actual controller keeps bare
            # unless a horizon rescue was found at this exact future frame.
            if kind == "horizon_rescue" and latest is not None and decision == int(latest):
                state["pending_actions"][decision] = action
        state["searched"] = True
        state["searches"] += 1


def run_worlds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Collect bounded resident worlds with deterministic, prospective v2 labels."""
    started = time.monotonic()
    depth, frames = int(config.get("shadow_depth", 300)), int(config.get("frames", 5000))
    beta, seconds = float(config.get("beta", 1.0)), float(config.get("seconds", 1500))
    capacity = int(config.get("worlds_per_process", config.get("worlds", 256)))
    bucket = int(config.get("branch_bucket_size", 64))
    steady_window = float(
        config.get("steady_state_seconds", config.get("steady_state_window_seconds", 0.0))
    )
    if not 1 <= depth <= 500 or not 1 <= frames <= 5000 or beta not in (0.0, 0.5, 1.0):
        raise ValueError("shadow depth, frames or beta exceeds v2 bounds")
    if (
        not math.isfinite(seconds)
        or seconds <= 0
        or not 1 <= capacity <= 256
        or not 6 <= bucket <= 1536
        or not math.isfinite(steady_window)
        or steady_window < 0.0
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
    # This is a v2-only throughput recipe.  The shared ExIt engine keeps its
    # historical native Vector61 dispatch unless a v2 CUDA collection opts in.
    opponents = getattr(teacher, "_opponents", None)
    if str(teacher.device).startswith("cuda") and hasattr(opponents, "batch_device_vectors"):
        opponents.batch_device_vectors = True
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
    doom_cache_peak_roots = 0
    mixes = {str(r["mix"]): {"searched": 0, "expanded": 0, "worlds": 0} for r in rows}
    heartbeat = started
    gpu_samples: list[float] = []
    cold_fill_elapsed: float | None = None
    cold_fill_work: dict[str, int] | None = None
    steady_started: float | None = None
    steady_stage_start: dict[str, float] | None = None
    first_decision_elapsed: float | None = None

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

    def stage_snapshot() -> dict[str, float]:
        snapshot: dict[str, float] = defaultdict(float, stages)
        for state in all_states:
            for key, value in state["stages"].items():
                snapshot[key] += float(value)
        return dict(snapshot)

    def fork(state: dict[str, Any]) -> None:
        state["shadow_id"] += 1
        state["pending_labels"].clear()
        state["pending_actions"].clear()
        state["searched"] = False
        state.pop("sparse_decision", None)
        state["shadow"].fork(state["world"], global_rng_state=state["rng"])

    invalid_reason = None
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
            _catch_up_lockstep(states, teacher, deadline)
            if cold_fill_elapsed is None:
                cold_fill_elapsed = time.monotonic() - started
                cold_fill_work = {
                    key: sum(int(state["work"][key]) for state in all_states) for key in WORK_KEYS
                }
                steady_started = time.monotonic()
                steady_stage_start = stage_snapshot()
                if steady_window:
                    deadline = min(deadline, steady_started + steady_window)
                    teacher.deadline = deadline
            for state in states:
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
                # Full prepared roots are expensive; admit only two doomed
                # worlds at a time, retain their cache through the batched
                # branch search, then release it before the next pair.
                for start in range(0, len(doomed), 2):
                    group = doomed[start : start + 2]
                    doom_cache_peak_roots = max(doom_cache_peak_roots, len(group))
                    cache_started = time.perf_counter()
                    try:
                        try:
                            if all(
                                isinstance(state["shadow"], ShadowTrajectory)
                                and hasattr(state["world"], "clone")
                                for state in group
                            ):
                                _build_doom_caches(group, teacher, deadline)
                            else:
                                for state in group:
                                    state["doom_cache"] = _build_doom_cache(
                                        state, teacher, deadline
                                    )
                        finally:
                            cache_elapsed = time.perf_counter() - cache_started
                            for state in group:
                                state["stages"]["doom_cache_s"] += cache_elapsed / len(group)
                        _search_dooms(group, teacher, bucket, deadline)
                    finally:
                        for state in group:
                            state.pop("doom_cache", None)
            stage = time.perf_counter()
            evaluator = getattr(teacher.policy, "evaluate_worlds", None)
            root_bucket = len(states) if str(teacher.device).startswith("cuda") else 64
            evaluations = (
                evaluator([s["world"] for s in states], bucket_size=root_bucket)
                if callable(evaluator)
                else [teacher.evaluate(s["world"]) for s in states]
            )
            stages["root_policy_s"] += time.perf_counter() - stage
            root_batch_peak = max(root_batch_peak, min(root_bucket, len(states)))
            planned = []
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
                planned.append((state, evaluation, student, executed, pending, length, valid))
            stage = time.perf_counter()
            grids = step_prepared_worlds(
                [item[0]["world"] for item in planned], [item[3] for item in planned], teacher
            )
            root_step_elapsed = time.perf_counter() - stage
            if first_decision_elapsed is None and planned:
                first_decision_elapsed = time.monotonic() - started
            remaining = []
            for (state, evaluation, student, executed, pending, length, valid), grid in zip(
                planned, grids
            ):
                world, shadow, decision = state["world"], state["shadow"], state["decision"]
                ambient = random.getstate()
                confirm_started = time.perf_counter()
                try:
                    random.setstate(state["rng"])
                    matched = shadow.confirm(world) if executed == student else False
                    state["rng"] = random.getstate()
                finally:
                    random.setstate(ambient)
                state["stages"]["root_step_s"] += (
                    root_step_elapsed / len(planned) + time.perf_counter() - confirm_started
                )
                state["work"]["root_row_steps"] += 1
                state["actions"].append(grid)
                on_action = config.get("on_action")
                if callable(on_action):
                    on_action(state["row"], int(executed))
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
                state["complete"] = True
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
    except ShadowControlMismatch as exc:
        invalid_reason = str(exc)
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
    completed_work = sum(sum(s["work"].values()) for s in all_states if s.get("complete", False))
    for state in all_states:
        for key, value in state["stages"].items():
            stages[key] += value
    observed_steady = 0.0 if steady_started is None else max(0.0, time.monotonic() - steady_started)
    cold_work = cold_fill_work or dict(work)
    steady_work = {key: max(0, work[key] - cold_work.get(key, 0)) for key in WORK_KEYS}
    steady_stages = (
        {}
        if steady_stage_start is None
        else {
            key: max(0.0, value - (steady_stage_start or {}).get(key, 0.0))
            for key, value in stages.items()
        }
    )
    import numba

    return {
        "kernel": KERNEL,
        "label_rule": LABEL_RULE,
        "shadow_depth": depth,
        "no_rewinds": True,
        "worlds": done,
        "requested_worlds": len(rows),
        "decisions": decisions_done,
        "complete": done == len(rows) and invalid_reason is None,
        "invalid_reason": invalid_reason,
        "elapsed_seconds": elapsed,
        "cold_fill_elapsed_seconds": cold_fill_elapsed,
        "cold_fill_complete": cold_fill_elapsed is not None,
        "cold_fill_active_row_steps": sum(cold_work.values()),
        "cold_fill_resident_row_steps": sum(cold_work.values()),
        "first_decision_elapsed_seconds": first_decision_elapsed,
        "steady_state_window_seconds": steady_window,
        "steady_state_elapsed_seconds": observed_steady,
        "steady_state_window_complete": bool(
            steady_window == 0.0 or observed_steady >= steady_window
        ),
        "steady_state_active_row_steps": sum(steady_work.values()),
        "steady_state_resident_row_steps": sum(steady_work.values()),
        "steady_state_work": steady_work,
        "steady_state_root_row_steps": steady_work["root_row_steps"],
        "steady_state_shadow_row_steps": steady_work["shadow_row_steps"],
        "steady_state_decisions": steady_work["root_row_steps"],
        "steady_state_host_stage_seconds": steady_stages,
        "decisions_per_second": decisions_done / max(elapsed, 1e-9),
        "mixes": mixes,
        "worlds_per_hour": done * 3600 / max(elapsed, 1e-9),
        "export_bytes": export_bytes,
        "export_bytes_per_world": export_bytes / done if done else None,
        "executed_row_steps": total,
        "executed_row_steps_per_world": completed_work / done if done else None,
        "completed_world_row_steps": completed_work,
        "unfinished_row_steps": total - completed_work,
        "active_row_steps": total,
        "resident_row_steps": total,
        "active_row_steps_per_second": total / max(elapsed, 1e-9),
        "active_resident_fraction": 1.0,
        "configured_root_capacity": capacity,
        "peak_root_worlds": peak,
        "search_batch_peak": max((s["search_bucket_peak"] for s in all_states), default=0),
        "root_inference_bucket_peak": root_batch_peak,
        "branch_bucket_capacity": bucket,
        "doom_cache_peak_roots": doom_cache_peak_roots,
        "teacher_profile": {
            **teacher.profile(),
            **work,
            "active_row_steps": total,
            "resident_row_steps": total,
        },
        "host_stage_seconds": dict(stages),
        "model_forward_seconds": {
            "hero": float(getattr(teacher.policy, "hero_forward_seconds", 0.0)),
            "vector_device": float(
                getattr(getattr(teacher, "_opponents", None), "vector_forward_seconds", 0.0)
            ),
            "scope": (
                "nested within host stages; includes tensor transfer "
                "and synchronized fp32 model forward"
            ),
        },
        "gpu_utilization_samples_percent": gpu_samples,
        "actual_numba_threads": int(numba.get_num_threads()),
        "actual_policy_device": str(teacher.device),
        "world_rate_scope": "completed worlds; work includes unfinished roots",
        "unfinished_worlds": len(rows) - done,
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
    }
