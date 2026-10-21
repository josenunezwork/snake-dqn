"""Unmodified CPU reference adapter and the six frozen S0 root fixtures."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def fixture_worlds(checkpoint: Path, cases: list[str] | None = None) -> list[tuple[dict, Any]]:
    """Build S0's actual short/long worlds without reading any study artifacts."""
    from research.speed_s0_baseline.harness import _builder, fixture_cases

    fixtures = fixture_cases()
    if cases is not None:
        unknown = set(cases) - {case["case"] for case in fixtures}
        if unknown:
            raise ValueError(f"unknown S0 fixtures: {sorted(unknown)}")
        fixtures = [case for case in fixtures if case["case"] in cases]
    return [(case, _builder(case["body"])(case["row"], checkpoint)) for case in fixtures]


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
