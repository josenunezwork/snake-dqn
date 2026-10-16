"""Deterministic, two-pass T4e return-bank collection.

Pass A discovers *genuinely decisive* death-window states and writes compact
outside candidates.  Pass B selects exact same-stratum outside candidates and
replays only those selected states.  The split is deliberate: it prevents a
large, speculative set of D300 outside branches from consuming disk or CPU.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from research.t3_rollout_labels_20261015.bank_engine import DEPTH, MIX_CYCLE

MIN_DECISIVE = 8000
MIN_LENGTH = 700
HORIZON = 5000
WORKER_LIMIT = 12


def masked_greedy(q: np.ndarray, legal: np.ndarray) -> int:
    """Return B8's deterministic advisory-legal greedy action."""
    values = np.asarray(q, dtype=np.float64)
    choices = np.asarray(legal, dtype=np.int64)
    if values.shape != (6,) or not len(choices) or np.any((choices < 0) | (choices > 5)):
        raise ValueError("expected six finite Q values and at least one legal action")
    if not np.isfinite(values).all() or len(set(choices.tolist())) != len(choices):
        raise ValueError("invalid Q values or advisory action set")
    return int(choices[np.argmax(values[choices])])


def genuinely_decisive(
    q: np.ndarray, legal: np.ndarray, died_by_action: Mapping[int, bool]
) -> bool:
    """Require a fatal masked-greedy B8 action and a surviving legal alternative."""
    greedy = masked_greedy(q, legal)
    actions = {int(action) for action in legal}
    if set(map(int, died_by_action)) != actions:
        raise ValueError("all and only advisory-legal actions require a D300 outcome")
    return bool(died_by_action[greedy]) and any(
        not bool(died_by_action[action]) for action in actions if action != greedy
    )


def _fingerprint(record: Mapping[str, Any]) -> str:
    """Return the canonical spillover fingerprint persisted by a world worker.

    A locally invented hash would not match the historic T4c fingerprint set,
    silently defeating overlap exclusion.  Workers must therefore materialize
    ``observation_fingerprints(local, global, scalars)`` while observations are
    still unpacked and persist that exact string with every state record.
    """
    value = record.get("fingerprint")
    if not isinstance(value, str) or not value:
        raise ValueError("collection record lacks canonical observation fingerprint")
    return value


def fixed_stratum(length: int, q: np.ndarray, cuts: np.ndarray) -> str:
    """Use fixed historic T4c cuts, never its historic quotas."""
    from research.t4c_20261016.data import strata

    return str(strata(np.asarray([length]), np.asarray(q, dtype=np.float32)[None], cuts)[0])


def historical_cuts(reference_bank: Path) -> tuple[np.ndarray, set[str]]:
    """Read only immutable T4c deciles and exclusion fingerprints."""
    from research.t4c_20261016.data import bound_files, target_definition

    bound_files(reference_bank)  # Hash-check before exposing its selection contract.
    cuts, _old_quotas, excluded = target_definition(reference_bank)
    return np.asarray(cuts, dtype=np.float64), set(excluded)


def assign_world_holdout(plans: Sequence[Mapping[str, Any]]) -> list[bool]:
    """Assign exactly 10% of each complete forty-world arm cycle before play.

    Each B8 final has the registered forty-world mix cycle; four worlds in each
    complete arm cycle are hash-ranked from immutable plan identity, so worker
    timing cannot change the split. Whole worlds are
    indivisible, so callers report the realised decisive-state fraction instead
    of claiming an exact 10% state split. Partial continuation cycles are
    training-only until they become complete.
    """
    flags = [False] * len(plans)
    per_arm: dict[str, list[int]] = defaultdict(list)
    for index, plan in enumerate(plans):
        per_arm[str(plan["arm"])].append(index)
    for indexes_for_arm in per_arm.values():
        for start in range(0, len(indexes_for_arm) - 39, 40):
            indexes = indexes_for_arm[start : start + 40]
            ranked = sorted(
                indexes,
                key=lambda index: hashlib.sha256(
                    (
                        f"t4e-holdout/v1:{plans[index]['arm']}:{plans[index]['mix']}:"
                        f"{plans[index]['seed']}:{plans[index].get('world_index', index)}"
                    ).encode()
                ).hexdigest(),
            )
            for index in ranked[:4]:
                flags[index] = True
    return flags


def prepare_jobs(
    plans: Sequence[Mapping[str, Any]], finals: Mapping[str, Path]
) -> list[dict[str, Any]]:
    """Bind every world to one of the five B8 finals and its pre-play split."""
    if set(finals) != {f"B8_s{index}" for index in range(5)}:
        raise ValueError("finals must bind exactly B8_s0 through B8_s4")
    jobs = [dict(plan) for plan in plans]
    for index, job in enumerate(jobs):
        job["arm"] = str(job.get("arm", f"B8_s{index % 5}"))
        if job["arm"] not in finals:
            raise ValueError("unknown B8 final in planned job")
        job["student"] = str(Path(finals[job["arm"]]).resolve())
        job["plan_index"] = index
    holds = assign_world_holdout(jobs)
    for job, holdout in zip(jobs, holds):
        job["holdout"] = holdout
    return jobs


def validate_mix_cycle(jobs: Sequence[Mapping[str, Any]]) -> None:
    """Require complete forty-world cycles with the registered 27/9/4 mix."""
    by_arm: dict[str, list[str]] = defaultdict(list)
    for job in jobs:
        by_arm[str(job["arm"])].append(str(job["mix"]))
    for arm, mixes in by_arm.items():
        if len(mixes) % len(MIX_CYCLE) or any(
            tuple(mixes[offset : offset + len(MIX_CYCLE)]) != MIX_CYCLE
            for offset in range(0, len(mixes), len(MIX_CYCLE))
        ):
            raise ValueError(f"{arm} does not use complete registered MIX_CYCLE blocks")


def _stable_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda row: (int(row["plan_index"]), int(row["source_decision"]), _fingerprint(row)),
    )


def dedupe_decisive(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Globally deduplicate observations, reserving holdout identities first."""
    ordered = _stable_records(records)
    held = {_fingerprint(row) for row in ordered if bool(row.get("holdout", False))}
    used: set[str] = set()
    result = []
    for row in ordered:
        fingerprint = _fingerprint(row)
        if not bool(row.get("holdout", False)) and fingerprint in held:
            continue
        if fingerprint not in used:
            used.add(fingerprint)
            result.append(row)
    return result


def select_outside(
    decisive: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    cuts: np.ndarray,
    excluded: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose exact 1:1, same-stratum outside states without substitution."""
    all_decisive = dedupe_decisive(decisive)
    decisive_rows = [row for row in all_decisive if not bool(row.get("holdout", False))]
    quota = Counter(
        fixed_stratum(int(row["length"]), np.asarray(row["q"]), cuts) for row in decisive_rows
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set(excluded)
    used.update(_fingerprint(row) for row in all_decisive)
    counts: Counter[str] = Counter()
    for candidate in _stable_records(candidates):
        if bool(candidate.get("holdout", False)):
            continue
        fingerprint = _fingerprint(candidate)
        key = fixed_stratum(int(candidate["length"]), np.asarray(candidate["q"]), cuts)
        if fingerprint in used or counts[key] >= quota[key]:
            continue
        candidate["fingerprint"] = fingerprint
        candidate["stratum"] = key
        selected.append(candidate)
        used.add(fingerprint)
        counts[key] += 1
    missing = {key: quota[key] - counts[key] for key in sorted(quota) if counts[key] < quota[key]}
    receipt = {
        "decisive_states": len(decisive_rows),
        "outside_states": len(selected),
        "quota_by_stratum": dict(sorted(quota.items())),
        "selected_by_stratum": dict(sorted(counts.items())),
        "missing_by_stratum": missing,
        "selection": "fixed-plan-order; exact new-decisive stratum quota; no substitution",
    }
    if missing:
        raise RuntimeError(json.dumps({"status": "INCOMPLETE_MATCHING", **receipt}, sort_keys=True))
    return selected, receipt


def completion_receipt(
    decisive: Sequence[Mapping[str, Any]], outside: Sequence[Mapping[str, Any]], wall_seconds: float
) -> dict[str, Any]:
    """Fail closed before publication if T4e's state-level contract is unmet."""
    training_decisive = [row for row in decisive if not bool(row.get("holdout", False))]
    training_outside = [row for row in outside if not bool(row.get("holdout", False))]
    if len(training_decisive) < MIN_DECISIVE:
        raise RuntimeError("fewer than 8000 training decisive states")
    if len(training_decisive) != len(training_outside):
        raise RuntimeError("training decisive/outside state counts are not equal")
    held = len(decisive) - len(training_decisive)
    return {
        "status": "COMPLETE_READY_FOR_BANK_VALIDATION",
        "training_decisive_states": len(training_decisive),
        "training_outside_states": len(training_outside),
        "holdout_decisive_states": held,
        "holdout_decisive_fraction": held / len(decisive) if decisive else 0.0,
        "wall_seconds": float(wall_seconds),
    }


def _candidate(
    snapshot: Mapping[str, Any], job: Mapping[str, Any], cuts: np.ndarray
) -> dict[str, Any]:
    """Persist only replayable state metadata; never a long-lived world clone."""
    from research.redesign_m3c_20261008.obs_pack import pack_ref, train_layout
    from research.t4a_posthoc_20261016.spillover import observation_fingerprints

    obs = snapshot["obs"]
    return {
        "plan_index": int(job["plan_index"]),
        "source_decision": int(snapshot["decision"]),
        "length": int(snapshot["length"]),
        "q": np.asarray(snapshot["q"], dtype=np.float32),
        "legal": np.asarray(snapshot["legal"], dtype=np.int64),
        "fingerprint": str(
            observation_fingerprints(obs["local"], obs["global"], obs["scalars"])[0]
        ),
        "state_packed": pack_ref(obs["local"], obs["global"], train_layout())[0],
        "state_scalars": np.asarray(obs["scalars"][0], dtype=np.float32),
        "stratum": fixed_stratum(int(snapshot["length"]), snapshot["q"], cuts),
        "holdout": bool(job["holdout"]),
    }


def collect_phase_a_world(
    job: Mapping[str, Any], cuts: np.ndarray
) -> tuple[list[dict], list[dict], dict]:
    """Discover one world's decisive branches and mature outside candidates.

    This is the only live-world worker entry point.  It limits pending clones to
    the D/5 cadence, validates that evaluator action selection equals the stored
    masked-greedy B8 action, and rolls a death-window state only after it can be
    classified as genuinely decisive.
    """
    from collections import deque

    from research.t3_rollout_labels_20261015.bank_engine import HORIZON as bank_horizon
    from research.t3_rollout_labels_20261015.bank_engine import (
        build_world,
        choose_and_step,
    )
    from research.t4d_return_regression_20261016.collector import (
        _snapshot,
        rollout_branch,
    )

    if int(job["source_decision"] if "source_decision" in job else 0) < 0:
        raise ValueError("invalid phase-A job")
    world = build_world(str(job["arm"]), dict(job), hero_checkpoint=Path(job["student"]))
    pending: deque[dict[str, Any]] = deque()
    decisive, candidates = [], []
    peak = 0
    for decision in range(min(HORIZON, bank_horizon)):
        captured: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepare(sim: Any) -> None:
            original_prepare(sim)
            if decision % 5 == 0:
                state = _snapshot(world, decision)
                if state is not None:
                    captured["state"] = state

        world.runtime.prepare = prepare
        try:
            actions, _mask, _obs = choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        state = captured.get("state")
        if state is not None:
            expected = masked_greedy(state["q"], state["legal"])
            if int(actions[0, 0]) != expected:
                raise RuntimeError("stored B8 Q/mask does not reproduce selected action")
            pending.append(state)
            peak = max(peak, len(pending))
        while pending and decision - int(pending[0]["decision"]) >= DEPTH:
            candidates.append(_candidate(pending.popleft(), job, cuts))
        if not bool(world.sim.alive[0, 0]):
            break
    death_decision = decision if not bool(world.sim.alive[0, 0]) else None
    if death_decision is not None:
        for state in pending:
            distance = death_decision - int(state["decision"])
            if not 0 <= distance < DEPTH:
                continue
            branches = {
                int(action): rollout_branch(state, int(action), DEPTH) for action in state["legal"]
            }
            died = {action: bool(branch["died"]) for action, branch in branches.items()}
            greedy = masked_greedy(state["q"], state["legal"])
            # The observed fatal action must die on the same transition when replayed.
            if not died[greedy] or int(branches[greedy]["T"]) != distance + 1:
                raise RuntimeError("fatal B8 branch does not reproduce observed death distance")
            if not genuinely_decisive(state["q"], state["legal"], died):
                continue
            decisive.append(
                {
                    **_candidate(state, job, cuts),
                    "death_distance": distance,
                    "b8_action": greedy,
                    "branches": branches,
                }
            )
    return (
        decisive,
        candidates,
        {
            "plan_index": int(job["plan_index"]),
            "seed": int(job["seed"]),
            "decisions": decision + 1,
            "death_decision": death_decision,
            "peak_pending_clones": peak,
        },
    )


def replay_selected_outside(
    job: Mapping[str, Any], candidate: Mapping[str, Any], cuts: np.ndarray
) -> dict:
    """Replay one selected candidate and roll every advisory-legal D300 branch.

    The packed observation, canonical identity, legal mask and Q vector must
    reproduce exactly.  A mismatch is a hard failure, never a best-effort label.
    """
    return replay_selected_outside_world(job, [candidate], cuts)[0]


def replay_selected_outside_world(
    job: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], cuts: np.ndarray
) -> list[dict]:
    """Replay one source world once and roll every selected outside state.

    A source world commonly has many selected candidates.  Reconstructing it
    once preserves the played trajectory and avoids replaying thousands of
    identical prefixes.  Every selected prepared state is still byte-checked
    before its D300 all-action branches are rolled.
    """
    from research.t3_rollout_labels_20261015.bank_engine import (
        build_world,
        choose_and_step,
    )
    from research.t4d_return_regression_20261016.collector import (
        _snapshot,
        rollout_branch,
    )

    by_target = {int(candidate["source_decision"]): dict(candidate) for candidate in candidates}
    if not by_target or len(by_target) != len(candidates):
        raise ValueError("selected outside candidates must have unique source decisions per world")
    world = build_world(str(job["arm"]), dict(job), hero_checkpoint=Path(job["student"]))
    records = []
    for decision in range(max(by_target) + 1):
        candidate = by_target.get(decision)
        found: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepare(sim: Any) -> None:
            original_prepare(sim)
            if candidate is not None:
                snapshot = _snapshot(world, decision)
                if snapshot is not None:
                    found["snapshot"] = snapshot

        world.runtime.prepare = prepare
        try:
            actions, _mask, _obs = choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        if candidate is not None:
            snapshot = found.get("snapshot")
            if snapshot is None:
                raise RuntimeError("selected outside state no longer qualifies on replay")
            replayed = _candidate(snapshot, job, cuts)
            for key in ("fingerprint", "state_packed", "state_scalars", "legal", "q"):
                if not np.array_equal(np.asarray(replayed[key]), np.asarray(candidate[key])):
                    raise RuntimeError(f"outside replay mismatch: {key}")
            if int(actions[0, 0]) != masked_greedy(snapshot["q"], snapshot["legal"]):
                raise RuntimeError("replayed B8 action does not match masked-greedy Q")
            branches = {
                int(action): rollout_branch(snapshot, int(action), DEPTH)
                for action in snapshot["legal"]
            }
            records.append({**replayed, "branches": branches, "family": "outside"})
        if not bool(world.sim.alive[0, 0]) and decision < max(by_target):
            raise RuntimeError("outside replay died before all selected source decisions")
    if len(records) != len(candidates):
        raise RuntimeError("outside replay did not reach every requested decision")
    return records


def phase_b_jobs(
    jobs: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]]
) -> list[tuple[dict, dict]]:
    """Bind selected states to their exact immutable source world, in plan order."""
    by_index = {int(job["plan_index"]): dict(job) for job in jobs}
    result = []
    for candidate in _stable_records(selected):
        job = by_index.get(int(candidate["plan_index"]))
        if job is None or bool(job["holdout"]):
            raise RuntimeError("selected outside candidate is not a training source world")
        result.append((job, candidate))
    return result


def execute_phase_a(
    jobs: Sequence[Mapping[str, Any]], cuts: np.ndarray, *, workers: int = WORKER_LIMIT
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run a bounded CPU discovery batch, merging results in plan order.

    ``executor.map`` deliberately preserves input order even if worlds finish in
    a different order.  That order is subsequently used for all quota decisions.
    Callers may append another full fresh 200-world batch when the returned
    exact strata are incomplete; they must never substitute a different stratum.
    """
    if not 1 <= workers <= WORKER_LIMIT:
        raise ValueError("T4e collection permits from one through twelve workers")
    ordered = sorted((dict(job) for job in jobs), key=lambda job: int(job["plan_index"]))
    validate_mix_cycle(ordered)
    with ProcessPoolExecutor(max_workers=workers, initializer=_cpu_worker_init) as executor:
        results = list(executor.map(_phase_a_call, ((job, cuts) for job in ordered)))
    decisive, candidates, receipts = [], [], []
    for world_decisive, world_candidates, receipt in results:
        decisive.extend(world_decisive)
        candidates.extend(world_candidates)
        receipts.append(receipt)
    return decisive, candidates, receipts


def _phase_a_call(
    args: tuple[Mapping[str, Any], np.ndarray],
) -> tuple[list[dict], list[dict], dict]:
    """Pickle-friendly phase-A process target."""
    return collect_phase_a_world(*args)


def _cpu_worker_init() -> None:
    """Cap native and Torch worker pools before inference world construction."""
    from research.t4d_return_regression_20261016.collector import _configure_device

    _configure_device("cpu")


def execute_phase_b(
    jobs: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    cuts: np.ndarray,
    *,
    workers: int = WORKER_LIMIT,
) -> list[dict]:
    """Run only the selected outside replays, in deterministic selection order."""
    if not 1 <= workers <= WORKER_LIMIT:
        raise ValueError("T4e collection permits from one through twelve workers")
    tasks = phase_b_jobs(jobs, selected)
    with ProcessPoolExecutor(max_workers=workers, initializer=_cpu_worker_init) as executor:
        return list(
            executor.map(_phase_b_call, ((job, candidate, cuts) for job, candidate in tasks))
        )


def _phase_b_call(args: tuple[Mapping[str, Any], Mapping[str, Any], np.ndarray]) -> dict:
    """Pickle-friendly phase-B process target."""
    return replay_selected_outside(*args)
