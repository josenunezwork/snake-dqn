"""Offline real-GameState trace support for T34 M1 identity measurement.

Live ``GameState`` is never directly loaded into GridBatchSim.  This module
records the real Watch6 chronology, including respawns, action outputs and the
isolated global RNG state, so a later seed+action replay has an auditable input
stream instead of treating two cloned Grid worlds as identity evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@contextlib.contextmanager
def isolated_global_random(seed: int) -> Iterator[None]:
    """Give one offline world its own CPython global RNG and restore the caller's."""
    before = random.getstate()
    random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(before)


def _value_digest(value: Any) -> str:
    """Digest live state through the same stable primitive conventions as M1 receipts."""
    from research.t34_shadow.shadow import _update_value

    digest = hashlib.sha256()
    _update_value(digest, value, set())
    return digest.hexdigest()


def game_state_digest(game: Any, *, checkpoint: Any = None) -> str:
    """Digest all live Watch state needed to detect an unexplained divergence."""
    snakes = []
    for snake in list(game.snakes):
        body = [tuple(map(int, segment)) for segment in getattr(snake, "segments", [])]
        veto = getattr(snake, "safety_veto", None)
        snakes.append(
            {
                "id": int(snake.id),
                "alive": bool(getattr(snake, "is_alive", False)),
                "body": body,
                "length": int(getattr(snake, "length", len(body))),
                "direction": tuple(map(int, getattr(snake, "direction", (0, 0)))),
                "boost": int(getattr(snake, "boost_frames", 0)),
                "hunger": int(getattr(snake, "frames_since_food", 0)),
                "respawn_timer": int(getattr(snake, "respawn_timer", 0)),
                "boosting": bool(getattr(snake, "is_boosting", False)),
                "life": int(getattr(snake, "life", 0)),
                "veto": (
                    vars(veto).copy() if veto is not None and hasattr(veto, "__dict__") else None
                ),
            }
        )
    food_manager = getattr(game, "food_manager", None)
    payload = {
        "frame": int(getattr(game, "frame", -1)),
        "snakes": snakes,
        "food": list(getattr(food_manager, "food", getattr(game, "food", []))),
        "frame_deaths": dict(getattr(game, "frame_death_causes", {})),
        "checkpoint": checkpoint,
        "global_random": random.getstate(),
    }
    return _value_digest(payload)


def _physical(
    game: Any, world: Any, specs: list, lives: list[int], *, live: bool, rng: object | None = None
) -> dict[str, Any]:
    """The same physical state schema on real GameState and GridBatchSim."""
    from src.simd_env.batch_sim import CARDINAL

    if live:
        size = int(game.snakes[0].segment_size)

        def cells(ps: Any) -> list:
            return [[int(x // size), int(y // size)] for x, y in ps]

        snakes = []
        for i, snake in enumerate(game.snakes):
            active = bool(snake.is_alive)
            snakes.append(
                {
                    "id": i,
                    "alive": active,
                    "body": cells(snake.segments),
                    "length": int(snake.length),
                    "direction": list(map(int, snake.direction)),
                    "boost_frames": int(snake.boost_frames),
                    "respawn_timer": int(snake.respawn_timer),
                    "hunger": (
                        int(snake.frames_since_food)
                        if active and (i == 0 or specs[i - 1][0] == "checkpoint")
                        else None
                    ),
                }
            )
        return {
            "frame": int(game.frame),
            "rng": rng,
            "lives": lives.copy(),
            "snakes": snakes,
            "food": cells(game.food_manager.food),
            "corpse": sorted(cells(game.food_manager._corpse_positions)),
        }
    sim = world.sim
    snakes = []
    for i in range(sim.S):
        active = bool(sim.alive[0, i])
        snakes.append(
            {
                "id": i,
                "alive": active,
                "body": [[int(x), int(y)] for x, y in sim.get_bodies(0, i)],
                "length": int(sim.length[0, i]),
                "direction": list(map(int, CARDINAL[int(sim.direction[0, i])])),
                "boost_frames": int(sim.boost_frames[0, i]),
                "respawn_timer": int(sim.respawn_timer[0, i]),
                "hunger": (
                    int(sim.frames_since_food[0, i])
                    if active and (i == 0 or specs[i - 1][0] == "checkpoint")
                    else None
                ),
            }
        )
    return {
        "frame": int(sim.frame[0]),
        "rng": sim._rngs[0]._rng.getstate(),
        "lives": lives.copy(),
        "snakes": snakes,
        "food": [[int(x), int(y)] for x, y in sim.food_cells[0]],
        "corpse": sorted([[int(x), int(y)] for x, y in sim.corpse_cells[0]]),
    }


def _carry(game: Any, world: Any, specs: list, *, live: bool) -> list:
    """Canonical alive Vector61 selection carry, trend memory and exact masks."""
    result = []
    for i, spec in enumerate(specs, 1):
        if spec[0] != "checkpoint":
            continue
        if live:
            snake = game.snakes[i]
            carry = getattr(snake, "_carried_selection", None)
            active = bool(snake.is_alive)
            valid = bool(active and carry is not None)
            state = (
                carry["state"].detach().cpu().numpy().astype(np.float32).reshape(-1)
                if valid
                else None
            )
            mask = (
                np.array([a in carry["safe_actions"] for a in range(6)], dtype=bool)
                if valid
                else None
            )
            trend = (
                (
                    float(snake._prev_nearest_enemy_dist),
                    (
                        -1
                        if snake._prev_nearest_enemy_id is None
                        else int(snake._prev_nearest_enemy_id)
                    ),
                )
                if active
                else None
            )
        else:
            rt, sim = world.runtime, world.sim
            active = bool(sim.alive[0, i])
            valid = bool(active and rt._carry_valid[0, i])
            state = rt._carry_state[0, i].copy().reshape(-1) if valid else None
            mask = rt._carry_mask[0, i].copy() if valid else None
            trend = (
                (float(rt.memory.prev_dist[0, i]), int(rt.memory.prev_id[0, i])) if active else None
            )
        result.append({"slot": i, "valid": valid, "state": state, "mask": mask, "trend": trend})
    return result


def _veto(game: Any, world: Any, specs: list, *, live: bool) -> list:
    from research.redesign_scope_20261007.grid_h5000_identity import strip_timing
    from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

    result = []
    for i, spec in enumerate(specs, 1):
        if spec[0] != "checkpoint":
            continue
        hook = game.snakes[i].safety_veto if live else world.opponents[i - 1].live_vetoes.get(0)
        hook = hook if hook is not None else SpaceAndHeadVeto(8.0)
        result.append(
            {
                "slot": i,
                "state": strip_timing(
                    {
                        "record": hook.record(),
                        "diagnostics": hook.diagnostics_record(),
                        "reference_lambda": hook.reference_lambda,
                    }
                ),
            }
        )
    return result


def _new_pair(seed: int, mix: str) -> tuple[Any, Any, list, dict]:
    """Build real offline Watch and independent continuous Grid from one seed."""
    import copy
    from unittest.mock import patch

    from research.apex_veto_serving_20261001 import serving_run as v2run
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015.bank_engine import build_world
    from research.t15_mixed_roster_20261017.episodes import new_session
    from research.t34_shadow.common import sha
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto
    from src.scripts.tournament_eval import _attach_agent, build_mix_specs
    from src.simd_env import eval_engine as ee
    from src.simd_env.grid_sim import GridBatchSim

    context = gid._context(1)
    frp = context["lookup"][gid.FRP3_S12[1]]
    champion = (
        "checkpoint",
        "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/champion_a5_freespace_20260621.pth",
    )
    specs = build_mix_specs(mix, 5, [frp, champion])
    with v2run.parity_config():
        session = new_session(seed, num_snakes=6)
    session.policy.forward = "rowwise"
    session.game.snakes[0].auto_respawn = True
    before = random.getstate()
    cache: dict[str, Any] = {}
    for slot, spec in enumerate(specs, 1):
        _attach_agent(session.game, slot, spec, seed, cache, profile=context["profile"])
        if spec[0] == "checkpoint":
            snake = session.game.snakes[slot]
            snake.policy.training = False
            snake.policy.epsilon = 0.0
            snake.policy.dqn.eval()
            install_space_and_head_veto(snake, 8.0)
    if random.getstate() != before:
        raise RuntimeError("opponent attachment unexpectedly advanced the live global RNG")
    session.policy._invalidate_cache()
    live_rng = random.getstate()
    hero = session.served_checkpoint
    if hero["sha256"] != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise RuntimeError("M1 hero checkpoint mismatch")
    plan = {"seed": seed, "mix": mix, "specs": specs}
    with patch.object(ee, "_terminal_hero_sim_class", return_value=GridBatchSim):
        world = build_world("b8_s3", plan, hero_checkpoint=Path(hero["path"]))
    for slot, spec in enumerate(specs, 1):
        if spec[0] == "checkpoint":
            policy = copy.copy(world.opponents[slot - 1])
            policy._configure_veto((slot,), "v8", 8.0, None)
            world.opponents[slot - 1] = policy
    identities = {
        str(path): sha(Path(path))
        for kind, path in [*(specs), ("checkpoint", hero["path"])]
        if kind == "checkpoint"
    }
    return (
        session,
        world,
        specs,
        {"hero": hero, "checkpoint_sha256": identities, "live_rng": live_rng},
    )


def run_m1_mix(seed: int, mix: str, *, frames: int = 5000, deadline: float | None = None) -> dict:
    """Compare canonical state and independent policy actions, plus rolling full-state shadow."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step
    from research.t34_shadow.shadow import ShadowTrajectory

    if mix not in {"scripted", "frozen", "mixed"} or not 1 <= frames <= 5000:
        raise ValueError("invalid bounded M1 mix or frame count")
    session, world, specs, identity = _new_pair(seed, mix)
    live_rng = identity.pop("live_rng")
    grid_global = random.getstate()
    shadow = ShadowTrajectory(300, checkpoint=identity)
    shadow.fork(world, global_rng_state=grid_global)
    shadow.catch_up(deadline=deadline)
    lives_live, lives_grid = [0] * 6, [0] * 6
    prev_live, prev_grid = [True] * 6, [True] * 6
    evidence, divergences = [], []
    comparisons = {k: 0 for k in ("physical", "carry", "v8", "actions", "shadow")}
    shadow_confirmed = 0
    for index in range(frames):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("M1 wall cap reached")
        chosen: dict[int, int] = {}
        originals = {id(s): s.update for s in session.game.snakes}
        for snake in session.game.snakes:

            def observed(*args: Any, _snake: Any = snake, **kwargs: Any) -> Any:
                result = originals[id(_snake)](*args, **kwargs)
                action = getattr(
                    _snake, "_pre_collision_action", getattr(_snake, "last_action", None)
                )
                if action is not None:
                    chosen[int(_snake.id)] = int(action)
                return result

            snake.update = observed
        ambient = random.getstate()
        random.setstate(live_rng)
        try:
            session.step()
            live_rng = random.getstate()
        finally:
            random.setstate(ambient)
            for snake in session.game.snakes:
                snake.update = originals[id(snake)]
        if session.last_error:
            raise RuntimeError(session.last_error)
        random.setstate(grid_global)
        try:
            actions, _advisory, _obs = choose_and_step(world, capture_obs=False)
            grid_global = random.getstate()
            had_prediction = bool(shadow.frames)
            confirmed = shadow.confirm(world) if had_prediction else None
            if had_prediction:
                shadow_confirmed += int(bool(confirmed))
                comparisons["shadow"] += int(not confirmed)
            if not shadow.confirmed or (not shadow.frames and bool(world.sim.alive[0, 0])):
                shadow.fork(world, global_rng_state=grid_global)
            shadow.catch_up(deadline=deadline)
        finally:
            random.setstate(ambient)
        active_live = [bool(s.is_alive) for s in session.game.snakes]
        active_grid = world.sim.alive[0].tolist()
        for i in range(6):
            lives_live[i] += int(active_live[i] and not prev_live[i])
            lives_grid[i] += int(active_grid[i] and not prev_grid[i])
        prev_live, prev_grid = active_live, active_grid
        valid = world.sim.get_transition_valid()[0]
        action_live = [chosen.get(i) for i in range(6)]
        action_grid = [int(actions[0, i]) if valid[i] else None for i in range(6)]
        pairs = {
            "physical": (
                _physical(session.game, world, specs, lives_live, live=True, rng=live_rng),
                _physical(None, world, specs, lives_grid, live=False),
            ),
            "carry": (
                _carry(session.game, world, specs, live=True),
                _carry(None, world, specs, live=False),
            ),
            "v8": (
                _veto(session.game, world, specs, live=True),
                _veto(None, world, specs, live=False),
            ),
            "actions": (action_live, action_grid),
        }
        record = {"frame": index + 1, "shadow_confirmed": confirmed}
        for key, (left, right) in pairs.items():
            a, b = _value_digest(left), _value_digest(right)
            record[key] = {"live": a, "grid": b}
            if a != b:
                comparisons[key] += 1
                if sum(item["component"] == key for item in divergences) < 3:
                    # Preserve exact repr incl float32 bytes/tuple-vs-list state.
                    divergences.append(
                        {
                            "frame": index + 1,
                            "component": key,
                            "root_cause": "unexplained",
                            "live": repr(left),
                            "grid": repr(right),
                        }
                    )
        evidence.append(record)
        if (index + 1) % 250 == 0:
            print(
                json.dumps({"mix": mix, "frame": index + 1, "divergences": comparisons}), flush=True
            )
    return {
        "schema": "t34-m1-live-grid-shadow/v2",
        "status": "COMPLETE",
        "mix": mix,
        "seed": seed,
        "frames_requested": frames,
        "frames_compared": len(evidence),
        "specs": specs,
        "identity": identity,
        "divergence_frames": comparisons,
        "divergences": divergences,
        "zero_unexplained": not any(comparisons.values()),
        "shadow_confirmed_frames": shadow_confirmed,
        "evidence": evidence,
        "evidence_chain": _value_digest(evidence),
        "representation_notes": [
            "Script hunger is inactive: scripts skip reward upkeep (ego_live_adapter.py).",
            "Dead hunger and trend carry are inactive until respawn; both reset before use.",
            "Elapsed veto timings/address caches are excluded; decision counters retained.",
            "B8 hero has no Vector61 carry; frozen opponents compare carry bytes/v8 counters.",
            "No live-to-Grid loader: independent seeded engines and live policy chronology.",
        ],
    }


def main() -> int:
    from research.t34_shadow.common import Budget, provenance, write_once

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026101934)
    parser.add_argument("--mix", choices=("scripted", "frozen", "mixed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--seconds", type=float, default=1700)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("create-only receipt already exists")
    budget = Budget(args.seconds)
    result = {"provenance": provenance()}
    try:
        result.update(
            run_m1_mix(
                args.seed, args.mix, frames=args.frames, deadline=budget.started + budget.seconds
            )
        )
    except Exception as exc:
        result.update(status="INCOMPLETE", error=f"{type(exc).__name__}: {exc}")
    result["wall_seconds"] = time.monotonic() - budget.started
    write_once(args.out, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"evidence", "provenance"}}),
        flush=True,
    )
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
