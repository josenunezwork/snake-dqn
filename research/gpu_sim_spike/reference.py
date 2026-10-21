"""Unmodified CPU reference adapter and the six frozen S0 root fixtures."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def fixture_worlds(checkpoint: Path, cases: list[str] | None = None) -> list[tuple[dict, Any]]:
    """Build S0's actual short/long worlds without reading any study artifacts."""
    from research.speed_s0_baseline.harness import _builder, fixture_cases

    tiny = cases is not None and all(name.startswith("tiny-") for name in cases)
    requested = [name.replace("tiny-", "short-", 1) for name in cases] if tiny else cases
    fixtures = fixture_cases()
    if requested is not None:
        unknown = set(requested) - {case["case"] for case in fixtures}
        if unknown:
            raise ValueError(f"unknown S0 fixtures: {sorted(unknown)}")
        fixtures = [case for case in fixtures if case["case"] in requested]
    built = [(case, _builder(case["body"])(case["row"], checkpoint)) for case in fixtures]
    if tiny:
        from dataclasses import replace

        from research.t34_exit_b8.trajectory import prepare_world
        from src.simd_env.vector61_featurizer import Vector61Params
        from src.simd_env.vector61_policy import Vector61Runtime

        for index, (case, world) in enumerate(built):
            cfg = replace(
                world.sim.cfg,
                game_width=160,
                game_height=120,
                initial_food=3,
                max_food=5,
                max_capacity=64,
                body_storage_capacity=64,
            )
            world.sim = type(world.sim)(
                cfg, seeds=[world.seed], train_mode=False, allow_respawn=True, jit=False
            )
            controlled = world.runtime.controlled.copy()
            world.runtime = Vector61Runtime(Vector61Params.from_game_config(cfg), backend="numba")
            world.runtime.bind(world.sim, controlled)
            for policy in world.opponents:
                if hasattr(policy, "runtime"):
                    policy.runtime = world.runtime
            prepare_world(world)
            built[index] = (
                dict(case, case=case["case"].replace("short-", "tiny-", 1), body="tiny-preflight"),
                world,
            )
    return built


def cpu_decision(world: Any, depth: int = 300, *, trace: bool = False) -> dict:
    """Search all advisory-legal actions; measure full-horizon post-step mass.

    T3 dense ranks alive transitions, preferring bare B8 and then Q/index.
    Mass integral is measured independently; it is not substituted for T3's rank.
    Each branch owns simulator, Python RNG, runtime carry and veto dictionaries.
    """
    from research.t34_exit_b8.engine import ExitTeacher
    from research.t34_exit_b8.trajectory import prepare_world
    from research.t34_shadow.shadow import clone_world_exact

    if depth < 1:
        raise ValueError("depth must be positive")
    started = time.perf_counter()
    teacher = ExitTeacher(device="cpu")
    evaluation = teacher.evaluate(world)
    q = np.asarray(evaluation.q)
    legal = np.asarray(world.sim.get_advisory_action_mask()[0, 0], dtype=bool).copy()
    if not legal.any():
        legal[:3] = True
    greedy = int(np.where(legal, q, -np.inf).argmax())
    outcomes = {}
    for first in np.flatnonzero(legal):
        branch = clone_world_exact(world)
        mass_sum = alive_steps = 0
        events = []
        for tick in range(depth):
            if not branch.sim.alive[0, 0]:
                break
            prepare_world(branch)
            hero_action = int(first) if tick == 0 else teacher.evaluate(branch).greedy_action()
            actions = teacher.actions(branch, hero_action)
            if trace:
                events.append({"tick": tick, "actions": actions.tolist()})
            branch.sim.step(actions)
            branch.runtime.observe_step(branch.sim)
            alive = bool(branch.sim.alive[0, 0])
            alive_steps += int(alive)
            mass_sum += int(branch.sim.length[0, 0]) if alive else 0
        outcomes[int(first)] = {
            "alive_steps": alive_steps,
            "mass_integral": mass_sum / depth,
            "trace": events,
        }
    best = max(outcome["alive_steps"] for outcome in outcomes.values())
    tied = [action for action, outcome in outcomes.items() if outcome["alive_steps"] == best]
    chosen = greedy if greedy in tied else max(tied, key=lambda a: (float(q[a]), -a))
    return {
        "action": chosen,
        "greedy": greedy,
        "q": q.tolist(),
        "legal": legal.tolist(),
        "outcomes": outcomes,
        "depth": depth,
        "wall_seconds": time.perf_counter() - started,
    }


def rescore_label(world: Any, actions: list[list[int]], depth: int) -> dict:
    """CPU re-score of a generated label by replaying its recorded action stream.

    This quality check uses reference event dynamics/RNG without rerunning any
    B8 or opponent network/veto. Exhaustive CPU search remains a separate parity
    and baseline measurement, never part of candidate-generation cost.
    """
    from research.t34_shadow.shadow import clone_world_exact

    started = time.perf_counter()
    branch = clone_world_exact(world)
    mass = alive_steps = 0
    for action in actions:
        if not branch.sim.alive[0, 0]:
            break
        branch.sim.step(np.asarray(action, dtype=np.int64).reshape(1, branch.sim.S))
        alive = bool(branch.sim.alive[0, 0])
        alive_steps += int(alive)
        mass += int(branch.sim.length[0, 0]) if alive else 0
    return {
        "mass_integral": mass / depth,
        "alive_steps": alive_steps,
        "wall_seconds": time.perf_counter() - started,
        "method": "CPU simulator action replay",
    }
