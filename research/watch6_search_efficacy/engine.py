"""Continuous real Watch mirror with the frozen synchronous dense adapter."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from research.bl_k1.native import RowStats, configure
from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.ruler_operator import _clone_root, _root_bytes
from research.t3_rollout_labels_20261015.bank_engine import prepared_q, rollout_all
from research.t3_rollout_labels_20261015.trigger import DenseTrigger


def dense_batch(world: Any, decision: int, q: np.ndarray, legal: np.ndarray) -> dict:
    """Independent full H300 branches; native counters attest every decision."""
    root = _clone_root(world)
    before = _root_bytes(root)
    stats = RowStats()
    with configure("kernel", stats):
        result = exact_stop.run(
            np.flatnonzero(legal).tolist(),
            int(np.where(legal, q.astype(np.float64), -np.inf).argmax()),
            q,
            lambda action: rollout_all(
                _clone_root(root), [action], 300, decision, record_outcomes=True
            )[action],
            ceiling._choose,
        )
    if _root_bytes(root) != before:
        raise ValueError("branch mutated its parent RNG/carry")
    counters = stats.as_dict()
    if (
        counters["kernel_rows"] <= 0
        or counters["stock_rows"] != 0
        or counters["fallback_rows"] != 0
        or counters["noncandidate_rows"] != 0
    ):
        raise ValueError("missing native kernel per-decision coverage")
    return {**result, "native_v8": counters, "root_sha256": ceiling.digest_value(before.hex())}


@contextlib.contextmanager
def step_guard(check: Callable[[], None]) -> Any:
    """Bound even a long all-die branch by the cumulative cap/host stop fence."""
    from src.simd_env.grid_sim import GridBatchSim

    def guard(original: Callable) -> Callable:
        def step(*args: Any, **kwargs: Any) -> Any:
            check()
            return original(*args, **kwargs)

        return step

    with (
        patch.object(GridBatchSim, "step", guard(GridBatchSim.step)),
        patch.object(GridBatchSim, "step_with_policy", guard(GridBatchSim.step_with_policy)),
    ):
        yield


def run_arm(seed: int, dense: bool, check: Callable[[], None], horizon: int = 14400) -> dict:
    """Use the existing continuous Watch/Grid bridge, auditing every real tick."""
    from research.watch6_search_efficacy.product import open_watch
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    if not 1 <= horizon <= 14400:
        raise ValueError("horizon must be 1..14400")
    check()
    session = open_watch({"seed": seed}, 24, "bare")
    world = session.world
    if not world.sim.allow_respawn or any(
        not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
    ):
        raise ValueError("C1 requires continuous respawn in real Watch and mirror")
    if getattr(session.live, "pathb", None) is not None:
        raise ValueError("C1 refuses Path B assistance")
    frames, decisions, batches = [], [], []
    from research.t15_mixed_roster_20261017.episodes import _assert_mixed, roster_identity

    _assert_mixed(session.live, 6)
    roster = roster_identity(session.live)
    life, was_alive = 1, True
    previous_alive = [True] * 6
    respawns = [0] * 6
    main_seconds = 0.0
    search_seconds = 0.0
    trigger = DenseTrigger(min_length=600, interval=5)
    last_progress = time.monotonic()
    try:
        with step_guard(check):
            for frame in range(1, horizon + 1):
                frame_started = time.monotonic()
                frame_search_seconds = 0.0
                check()
                selected: dict = {}

                def choose(sim: Any) -> np.ndarray:
                    nonlocal life, trigger, frame_search_seconds
                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        trigger = DenseTrigger(min_length=600, interval=5)
                    actions = np.ones((1, sim.S), dtype=np.int64)
                    if active:
                        obs = build_ego_raster(
                            sim,
                            EgoRasterConfig(version="bv8"),
                            backend="numba",
                            rows=np.array([[0, 0]]),
                        )
                        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
                        advisory = sim.get_advisory_action_mask()[0, 0]
                        legal, bare = ceiling._legal(q, advisory)
                        chosen = bare
                        fired = dense and trigger.fires(len(decisions), int(sim.length[0, 0]), True)
                        if fired and advisory.any():
                            search_started = time.monotonic()
                            batch = dense_batch(world, frame - 1, q, legal)
                            frame_search_seconds += time.monotonic() - search_started
                            chosen = int(batch["chosen"])
                            batches.append({"frame": frame, "life": life, "bare": bare, **batch})
                        if not np.isfinite(q).all() or not 0 <= chosen < 6:
                            raise ValueError("nonfinite Q or illegal action")
                        selected.update(
                            frame=frame,
                            life=life,
                            length=int(sim.length[0, 0]),
                            bare=bare,
                            chosen=chosen,
                            q=q.astype(float).tolist(),
                            advisory=np.asarray(advisory, dtype=bool).tolist(),
                        )
                    _dispatch_actions(
                        sim,
                        sim.get_resolved_action_mask(),
                        actions,
                        [world.hero],
                        [world.opponents],
                    )
                    if active:
                        if int(actions[0, 0]) != selected["bare"]:
                            raise ValueError("bare dispatch/Q mismatch")
                        actions[0, 0] = selected["chosen"]
                    return actions

                actions_holder: dict = {}

                def choose_record(sim: Any) -> np.ndarray:
                    actions = choose(sim)
                    actions_holder["actions"] = actions.copy()
                    return actions

                world.sim.step_with_policy(choose_record)
                world.runtime.observe_step(world.sim)
                # Existing bridge runs actual Watch and compares physical state,
                # RNG, carried selection/trend memory and every v8 counter.
                actual_decisions = []
                hero = session.live.game.snakes[0]
                original_update = hero.update
                original_q = session.live.policy._next_dispatch_q

                def observed_update(*args: Any, **kwargs: Any) -> Any:
                    actual_decisions.append(int(hero.length))
                    return original_update(*args, **kwargs)

                def observed_q(*args: Any, **kwargs: Any) -> Any:
                    value = original_q(*args, **kwargs)
                    if not selected or not np.array_equal(
                        value.detach().cpu().numpy().reshape(6),
                        np.asarray(selected["q"], dtype=np.float32),
                    ):
                        raise ValueError("live/mirror B8 Q mismatch")
                    return value

                hero.update = observed_update
                session.live.policy._next_dispatch_q = observed_q
                try:
                    session._step_live(actions_holder["actions"])
                finally:
                    hero.update = original_update
                    session.live.policy._next_dispatch_q = original_q
                if actual_decisions != ([selected["length"]] if selected else []):
                    raise ValueError("live/mirror actual hero decision mismatch")
                session.live.snapshot()
                hero = session.live.game.snakes[0]
                death = int(hero.id) in session.live.game.frame_death_causes
                if death != bool(
                    world.sim.get_transition_valid()[0, 0] and not world.sim.alive[0, 0]
                ):
                    raise ValueError("Watch/mirror death event mismatch")
                if selected:
                    if int(hero.last_action) != selected["chosen"]:
                        raise ValueError("live delivered action mismatch")
                    decisions.append(selected)
                was_alive = bool(hero.is_alive)
                alive = [bool(s.is_alive) for s in session.live.game.snakes]
                for slot in range(6):
                    respawns[slot] += int(alive[slot] and not previous_alive[slot])
                previous_alive = alive
                frames.append(
                    {
                        "frame": frame,
                        "life": life,
                        "alive": was_alive,
                        "mass": int(hero.length) if was_alive else 0,
                        "death": death,
                        "death_length": selected.get("length", 0) if death else 0,
                    }
                )
                main_seconds += time.monotonic() - frame_started - frame_search_seconds
                search_seconds += frame_search_seconds
                if time.monotonic() - last_progress >= 60:
                    print(
                        f"Watch6 efficacy seed={seed} arm={'dense' if dense else 'bare'} "
                        f"frames={frame}/{horizon}",
                        flush=True,
                    )
                    last_progress = time.monotonic()
    finally:
        session.close()
    return {
        "frames": frames,
        "decisions": decisions,
        "batches": batches,
        "roster": roster,
        "mirror_identity_frames": horizon,
        "respawns_per_slot": respawns,
        "timing": {"main_seconds": main_seconds, "search_seconds": search_seconds},
    }
