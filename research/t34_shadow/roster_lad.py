"""Bounded fresh LAD timing for the exact frozen/mixed M1 Watch rosters."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from research.t34_shadow.common import Budget, provenance, write_once
from research.t34_shadow.identity import _new_pair
from research.t34_shadow.lad import search_lad
from research.t34_shadow.latency import quantiles
from research.t34_shadow.shadow import clone_world_exact


def step_capture(world: Any, window: deque, budget: Budget) -> bool:
    """Reuse the T3 prepared dispatch phase and retain a bounded exact snapshot."""
    from src.simd_env.eval_engine import _dispatch_actions

    selected: dict[str, Any] = {}

    def choose(sim: Any) -> np.ndarray:
        budget.check()
        world.runtime.prepare(sim)
        active = bool(sim.alive[0, 0])
        selected["active"] = active
        if active:
            started = time.perf_counter()
            snapshot = clone_world_exact(world)
            # A selector has prepared food/respawns but is still on the source
            # call stack. Detached first-action branches must consume that phase.
            snapshot.sim._policy_phase_prepared = True
            snapshot.sim._policy_step_in_progress = False
            if hasattr(snapshot.sim, "_active_env_mask"):
                del snapshot.sim._active_env_mask
            window.append(
                {
                    "decision": int(sim.frame[0]) - 1,
                    "world": snapshot,
                    "clone_seconds": time.perf_counter() - started,
                }
            )
        actions = np.ones((1, sim.S), dtype=np.int64)
        _dispatch_actions(
            sim, sim.get_resolved_action_mask(), actions, [world.hero], [world.opponents]
        )
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return bool(selected["active"])


def expansion(snapshot: Any, decision: int, doom: int, budget: Budget) -> dict:
    """Exact current-policy alternatives, with all frozen/v8 carry isolated."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step, step_prepared
    from src.simd_env.ego2s_policy import student_q

    budget.check()
    q = student_q(snapshot.hero.net, snapshot.sim, np.array([[0, 0]]), "rowwise").numpy()[0]
    legal = np.flatnonzero(snapshot.sim.get_advisory_action_mask()[0, 0]).tolist()
    allowed = legal if legal else [0, 1, 2]
    greedy = max(allowed, key=lambda a: (float(q[a]), -a))
    alternatives = sorted((a for a in legal if a != greedy), key=lambda a: (-float(q[a]), a))
    horizon = min(300, 5000 - decision)
    outcomes: dict[str, dict] = {}
    actual_steps = 0
    for action in alternatives:
        budget.check()
        branch = clone_world_exact(snapshot)
        step_prepared(branch, action)
        executed = 1
        alive_steps = int(branch.sim.alive[0, 0])
        while executed < horizon and bool(branch.sim.alive[0, 0]):
            budget.check()
            choose_and_step(branch, capture_obs=False)
            executed += 1
            alive_steps += int(branch.sim.alive[0, 0])
        alive = bool(branch.sim.alive[0, 0])
        outcomes[str(action)] = {
            "action": action,
            "survives": alive and executed == horizon,
            "horizon": horizon,
            "frames_survived": alive_steps,
            "frames_executed": executed,
            "env_frames": executed,
            "death_decision": None if alive else decision + executed - 1,
            "final_mass": int(branch.sim.length[0, 0]) if alive else 0,
        }
        actual_steps += executed
    executed = doom - decision + 1
    outcomes[str(greedy)] = {
        "action": greedy,
        "survives": False,
        "horizon": horizon,
        "frames_survived": executed - 1,
        "frames_executed": executed,
        "env_frames": 0,
        "death_decision": doom,
        "final_mass": 0,
        "control_reproduced": True,
        "final_mass_status": "known_greedy_doom",
    }
    return {
        "greedy": greedy,
        "q": [float(x) for x in q],
        "outcomes": outcomes,
        "batch": {"env_frames": actual_steps},
    }


def measure(mix: str, budget: Budget) -> dict:
    """Time at most three dooms on a fixed three-seed diagnostic list."""
    cases, source_rows = [], []
    seeds = [2026101934, 2026101935, 2026101936]
    for seed in seeds:
        budget.check()
        _session, world, specs, identity = _new_pair(seed, mix)
        identity.pop("live_rng")
        window: deque = deque(maxlen=300)
        last_post_alive = True
        life_start = 0
        for tick in range(5000):
            budget.check()
            # Drop the old life's snapshots once respawn preparation starts.
            if not last_post_alive:
                window.clear()
                life_start = tick
            active_before = step_capture(world, window, budget)
            last_post_alive = bool(world.sim.alive[0, 0])
            if active_before and not last_post_alive:
                doom = int(world.sim.frame[0]) - 1
                first = max(life_start, doom - 299)
                snapshots = {row["decision"]: row["world"] for row in window}
                if set(range(first, doom + 1)) - set(snapshots):
                    raise RuntimeError("prepared doom snapshot window is incomplete")
                cache_clone_seconds = sum(row["clone_seconds"] for row in window)
                started = time.perf_counter()
                result = search_lad(first, doom, lambda t: expansion(snapshots[t], t, doom, budget))
                search_seconds = time.perf_counter() - started
                cases.append(
                    {
                        "seed": seed,
                        "mix": mix,
                        "identity": identity,
                        "specs": specs,
                        "doom": doom,
                        "first": first,
                        "search_seconds": search_seconds,
                        "snapshot_retention_clone_seconds": cache_clone_seconds,
                        "snapshot_count": len(window),
                        "result": result,
                        "search_row_steps": sum(p["batch"]["env_frames"] for p in result["probes"]),
                    }
                )
                print(
                    json.dumps(
                        {"mix": mix, "doom_samples": len(cases), "seed": seed, "frame": doom + 1}
                    ),
                    flush=True,
                )
                if len(cases) == 3:
                    break
        source_rows.append({"seed": seed, "frames": int(world.sim.frame[0])})
        if len(cases) == 3:
            break
    return {
        "mix": mix,
        "seeds_fixed_before_probe": seeds,
        "maximum_doom_samples": 3,
        "cases": cases,
        "source_trajectories": source_rows,
        "search_latency": quantiles([row["search_seconds"] for row in cases]),
        "snapshot_retention_clone_latency": quantiles(
            [row["snapshot_retention_clone_seconds"] for row in cases]
        ),
        "interpretation": (
            "Fresh exact Watch-roster search, all probes uncached. Snapshot retention is timed "
            "separately; warm search assumes its prepared window already exists. Cold timing "
            "must additionally pay full catch-up and retention, and no online efficacy is measured."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", choices=("frozen", "mixed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1700)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("create-only receipt already exists")
    budget = Budget(args.seconds)
    result: dict[str, Any] = {
        "schema": "t34-watch-roster-lad-timing/v1",
        "provenance": provenance(),
    }
    try:
        result.update(measure(args.mix, budget))
        result["status"] = "COMPLETE" if result["cases"] else "INCOMPLETE_NO_DOOM"
    except Exception as exc:
        result.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
    result["wall_seconds"] = time.monotonic() - budget.started
    write_once(args.out, result)
    print(
        json.dumps(
            {"mix": args.mix, "status": result["status"], "wall_seconds": result["wall_seconds"]}
        ),
        flush=True,
    )
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
