"""Death-window worker for the T3 dense rollout label bank.

The ordinary trajectory itself proves that B8 dies for snapshots in the final
300 decisions.  Consequently the expensive counterfactual work is postponed
until death and evaluates only alternatives for those snapshots.  A small,
deterministic reservoir of older, prefiltered snapshots provides the retained
``B8 was right`` counterweight examples without charging survivor worlds.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from research.t1_commitment_20261015.replay import configure_cpu
from research.t3_rollout_labels_20261015.bank_engine import (
    DEPTH,
    HORIZON,
    CapturedWorld,
    choose_and_step,
    classify,
    dense_eligible,
    prepared_q,
    rollout_all,
    sha256,
    validate_b8_controls,
    write_world_rows,
)
from research.t3_rollout_labels_20261015.collect import _atomic_json, _key, _row
from research.t3_rollout_labels_20261015.trigger import DenseTrigger

WINDOW_SNAPSHOTS = 60
"""Dense snapshots every fifth decision cover a strict 300-decision window."""

CONTROL_EVERY = 20
"""Re-roll B8 exactly once in every twenty eligible death-window batches."""

COUNTER_RESERVOIR = 96
"""Bounded, deterministic older-snapshot sample per dying world.

The initial 32-snapshot epoch supplied only 142 counters for 1,910 new
decisive rows. A 96-element reservoir increases counter coverage while keeping
survivor worlds play-only and retaining the same decisive-window contract.
"""


def inside_death_window(snapshot_decision: int, death_decision: int, depth: int = DEPTH) -> bool:
    """Return whether a selector snapshot belongs to the strict final window."""
    gap = int(death_decision) - int(snapshot_decision)
    return 0 <= gap < int(depth)


def _snapshot(
    world: CapturedWorld,
    decision: int,
    frame: int,
    length: int,
    b8_action: int,
    advisory: np.ndarray,
    obs: dict[str, np.ndarray],
    q: np.ndarray,
    branch: CapturedWorld,
) -> dict[str, Any]:
    """Capture one prepared selector state after all eligibility checks."""
    legal = [int(value) for value in np.flatnonzero(advisory)]
    if int(b8_action) not in legal:
        raise RuntimeError(f"B8 selected an advisory-illegal action at decision {decision}")
    q_advisory = np.where(np.asarray(advisory, dtype=bool), q, -np.inf)
    if int(b8_action) != int(np.argmax(q_advisory)):
        raise RuntimeError(f"B8 Q/advisory action mismatch at decision {decision}")
    return {
        "decision": int(decision),
        "frame": int(frame),
        "length": int(length),
        "action": int(b8_action),
        "legal": legal,
        "obs": obs,
        "q": q,
        "branch": branch,
    }


def _reservoir_add(
    reservoir: list[dict[str, Any]], snapshot: dict[str, Any], seen: int, rng: np.random.Generator
) -> int:
    """Use fixed-size reservoir sampling, returning the updated observation count."""
    seen += 1
    if len(reservoir) < COUNTER_RESERVOIR:
        reservoir.append(snapshot)
    else:
        index = int(rng.integers(seen))
        if index < COUNTER_RESERVOIR:
            reservoir[index] = snapshot
    return seen


def _alternative_outcomes(
    snapshot: dict[str, Any], *, record_outcomes: bool = False
) -> dict[int, dict[str, Any]]:
    """Evaluate alternatives and add B8's trajectory-proven fatal outcome."""
    alternatives = [action for action in snapshot["legal"] if action != snapshot["action"]]
    if not alternatives:
        outcomes = {}
    elif record_outcomes:
        outcomes = rollout_all(
            snapshot["branch"], alternatives, DEPTH, snapshot["decision"], record_outcomes=True
        )
    else:
        outcomes = rollout_all(snapshot["branch"], alternatives, DEPTH, snapshot["decision"])
    # The live trajectory chose B8 at this prepared state and died within the
    # strict window, so this is equivalent to a B8 continuation for labelling.
    outcomes[int(snapshot["action"])] = {"survives": False, "frames_executed": 0}
    return outcomes


def collect_world_death_window(
    job: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    run_identity: dict[str, Any],
    threshold: float = float("-inf"),
    record_outcomes: bool = False,
    label_policy: str = "surviving-set",
    hero_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Collect one world with post-death decisive rollouts and sampled counters.

    This has the same shard and base receipt schema as ``collect_world``.  It
    never rolls a counterfactual before observing the actual B8 death.
    """
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    configure_cpu()
    if str(plan["mix"]) != str(job["mix"]):
        raise ValueError("job mix and materialized roster mix differ")
    key = _key(job, plan)
    receipt_path, rows_path = out / "worlds" / f"{key}.json", out / "worlds" / f"{key}.npz"
    if receipt_path.exists() or rows_path.exists():
        raise RuntimeError(
            f"parent failed to resolve an existing world before worker launch: {key}"
        )

    world: CapturedWorld = (
        build_world(job["arm"], plan)
        if hero_checkpoint is None
        else build_world(job["arm"], plan, hero_checkpoint=hero_checkpoint)
    )
    trigger, rows = DenseTrigger(min_length=700), []
    window: deque[dict[str, Any]] = deque(maxlen=WINDOW_SNAPSHOTS)
    reservoir: list[dict[str, Any]] = []
    # Key-derived RNG makes the selected counter candidates stable under worker scheduling.
    rng = np.random.default_rng(np.random.SeedSequence([int(plan["seed"]), 20261015, 3]))
    reservoir_seen = 0
    dense_candidates = prefilter_skipped = q_action_mismatches = 0
    window_batches = counter_batches = controls_checked = 0
    controls: list[tuple[int, bool, int]] = []
    control_receipts: list[dict[str, int | bool]] = []
    counter_controls: list[tuple[int, bool, int]] = []
    started = time.perf_counter()

    for decision in range(HORIZON):
        selected: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepared(sim: Any) -> None:
            original_prepare(sim)
            frame, alive = int(sim.frame[0]), bool(sim.alive[0, 0])
            length = int(sim.length[0, 0]) if alive else 0
            selected.update(frame=frame, length=length, alive=alive)
            if dense_eligible(trigger, decision, frame, length, alive) and bool(
                sim.get_advisory_action_mask()[0, 0].any()
            ):
                obs = build_ego_raster(
                    sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
                )
                selected["dense"] = (obs, np.asarray(prepared_q(world, obs), dtype=np.float32))
                # Clone while Watch has prepared the selector cache and before
                # its ordinary B8 action advances the live trajectory.
                selected["branch"] = world.clone()

        world.runtime.prepare = prepared
        try:
            actions, advisory, _ = choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare

        if "dense" in selected:
            dense_candidates += 1
            obs, q = selected["dense"]
            try:
                snapshot = _snapshot(
                    world,
                    decision,
                    selected["frame"],
                    selected["length"],
                    int(actions[0, 0]),
                    advisory,
                    obs,
                    q,
                    selected["branch"],
                )
            except RuntimeError:
                q_action_mismatches += 1
                raise
            # Window admission intentionally has no -max(Q) prefilter.
            window.append(snapshot)
            if float(-np.max(q)) > threshold:
                reservoir_seen = _reservoir_add(reservoir, snapshot, reservoir_seen, rng)
            else:
                prefilter_skipped += 1

        if not bool(world.sim.alive[0, 0]):
            break

    death_decision = None if bool(world.sim.alive[0, 0]) else decision
    if death_decision is not None:
        for snapshot in window:
            if not inside_death_window(snapshot["decision"], death_decision):
                continue
            window_batches += 1
            outcomes = _alternative_outcomes(snapshot, record_outcomes=record_outcomes)
            if record_outcomes:
                # The actual trajectory is an exact continuation for B8 in the
                # strict terminal window.  It avoids a duplicate branch rollout.
                outcomes[snapshot["action"]].update(
                    death_decision=int(death_decision), final_length=0
                )
            # Every twentieth window batch performs the independent bitwise B8 control.
            if window_batches % CONTROL_EVERY == 0:
                control = (
                    rollout_all(
                        snapshot["branch"], [snapshot["action"]], DEPTH, snapshot["decision"]
                    )
                    if not record_outcomes
                    else rollout_all(
                        snapshot["branch"],
                        [snapshot["action"]],
                        DEPTH,
                        snapshot["decision"],
                        record_outcomes=True,
                    )
                )[snapshot["action"]]
                outcomes[snapshot["action"]] = control
                controls.append(
                    (
                        snapshot["decision"],
                        bool(control["survives"]),
                        int(control["frames_executed"]),
                    )
                )
                control_receipts.append(
                    {
                        "decision": int(snapshot["decision"]),
                        "survives": bool(control["survives"]),
                        "frames_executed": int(control["frames_executed"]),
                        "expected_death_decision": int(death_decision),
                    }
                )
                controls_checked += 1
            decisive, surviving = classify(outcomes, snapshot["action"])
            if decisive:
                rows.append(
                    _row(
                        snapshot["obs"],
                        snapshot["action"],
                        surviving,
                        True,
                        snapshot["decision"],
                        snapshot["frame"],
                        snapshot["length"],
                        snapshot["q"],
                        outcomes=outcomes if record_outcomes else None,
                        label_policy=label_policy,
                    )
                )

        # Only snapshots known to be outside the death window are eligible for counters.
        for snapshot in reservoir:
            if inside_death_window(snapshot["decision"], death_decision):
                continue
            outcomes = (
                rollout_all(snapshot["branch"], snapshot["legal"], DEPTH, snapshot["decision"])
                if not record_outcomes
                else rollout_all(
                    snapshot["branch"],
                    snapshot["legal"],
                    DEPTH,
                    snapshot["decision"],
                    record_outcomes=True,
                )
            )
            counter_batches += 1
            b8_outcome = outcomes[snapshot["action"]]
            counter_controls.append(
                (
                    snapshot["decision"],
                    bool(b8_outcome["survives"]),
                    int(b8_outcome["frames_executed"]),
                )
            )
            decisive, surviving = classify(outcomes, snapshot["action"])
            if not decisive and bool(surviving.any()):
                rows.append(
                    _row(
                        snapshot["obs"],
                        snapshot["action"],
                        surviving,
                        False,
                        snapshot["decision"],
                        snapshot["frame"],
                        snapshot["length"],
                        snapshot["q"],
                        outcomes=outcomes if record_outcomes else None,
                        label_policy=label_policy,
                    )
                )

    control_checks = validate_b8_controls(controls, death_decision)
    counter_control_checks = validate_b8_controls(counter_controls, death_decision)
    write_world_rows(rows_path, rows)
    receipt = {
        "schema": "t3-label-world/v1",
        "status": "PASS",
        "key": key,
        "arm": job["arm"],
        "mix": job["mix"],
        "cycle": job["cycle"],
        "seed": int(plan["seed"]),
        "roster": plan,
        "run_identity": run_identity,
        "rows": len(rows),
        "decisive_rows": int(sum(bool(row["decisive"]) for row in rows)),
        "counterweight_candidates": int(sum(not bool(row["decisive"]) for row in rows)),
        "batches": window_batches + counter_batches,
        "eligible_batches": window_batches,
        "dense_candidate_batches": dense_candidates,
        "prefilter_skipped_batches": prefilter_skipped,
        "prefilter_passed_batches": dense_candidates - prefilter_skipped,
        "q_action_mismatches": q_action_mismatches,
        "decisions": decision + 1,
        "death_decision": death_decision,
        "b8_control_checks": control_checks,
        "counter_b8_control_checks": counter_control_checks,
        "b8_control_receipts": control_receipts,
        "death_window": {
            "maxlen": WINDOW_SNAPSHOTS,
            "depth": DEPTH,
            "strict": True,
            "window_batches": window_batches,
            "counter_reservoir_seen": reservoir_seen,
            "counter_reservoir_size": len(reservoir),
            "counter_batches": counter_batches,
            "controls_checked": controls_checked,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    receipt["rows_sha256"] = sha256(rows_path)
    _atomic_json(receipt_path, receipt)
    return receipt
