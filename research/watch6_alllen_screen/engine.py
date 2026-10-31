"""Watch6-v2 3-arm exploration engine reusing exact_stop.py and the Watch6 world."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from research.bl_served_search import ceiling
from research.t3_rollout_labels_20261015.bank_engine import prepared_q
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.watch6_alllen_screen.protocol import ARM_SPECS
from research.watch_replay.engine import dense_batch, step_guard
from research.watch_replay.product import open_watch


def run_arm(
    seed: int,
    arm: str,
    check: Callable[[], None],
    *,
    horizon: int = 6000,
    audit_live_mirror: bool = False,
    include_raw: bool = False,
    on_frame: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run one Watch6-v2 session for arm in {'bare', 'l2_prime_600', 'exact_stop_alllen'}."""
    from research.t15_mixed_roster_20261017.episodes import _assert_mixed, roster_identity
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm: {arm!r}")
    if not 1 <= horizon <= 14400:
        raise ValueError("horizon must be 1..14400")
    spec = ARM_SPECS[arm]
    min_length: int | None = spec["min_length"]
    interval: int = int(spec["interval"])

    check()
    started = time.monotonic()
    session = open_watch({"seed": int(seed)}, 24, "bare")
    world = session.world
    if not world.sim.allow_respawn or any(
        not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
    ):
        session.close()
        raise ValueError("Watch6 exploration screen requires continuous respawn")
    if getattr(session.live, "pathb", None) is not None:
        session.close()
        raise ValueError("Watch6 exploration screen refuses Path B runtime assistance")

    _assert_mixed(session.live, 6)
    roster = roster_identity(session.live)

    life = 1
    was_alive = True
    trigger = (
        DenseTrigger(min_length=min_length, interval=interval) if min_length is not None else None
    )
    prev_grid_alive = [bool(x) for x in world.sim.alive[0].tolist()]
    grid_respawns = [0] * int(world.sim.S)

    alive_frames = 0
    mass_sum = 0
    deaths = 0
    decisions_count = 0
    boost_decisions = 0
    max_length = 0
    alive_lengths: list[int] = []
    death_lengths: list[int] = []

    search_roots = 0
    search_roots_lt600 = 0
    search_roots_ge600 = 0
    interventions = 0
    interventions_lt600 = 0
    interventions_ge600 = 0
    branches_evaluated = 0
    branches_skipped = 0
    native_v8_kernel_rows = 0

    raw_frames: list[dict[str, Any]] = []
    raw_decisions: list[dict[str, Any]] = []
    raw_batches: list[dict[str, Any]] = []
    last_progress = time.monotonic()

    try:
        with step_guard(check):
            for frame in range(1, horizon + 1):
                check()
                selected: dict[str, Any] = {}

                def choose(sim: Any) -> np.ndarray:
                    nonlocal life, trigger, search_roots, search_roots_lt600, search_roots_ge600
                    nonlocal interventions, interventions_lt600, interventions_ge600
                    nonlocal branches_evaluated, branches_skipped, native_v8_kernel_rows
                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        if min_length is not None:
                            trigger = DenseTrigger(min_length=min_length, interval=interval)
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
                        cur_length = int(sim.length[0, 0])
                        fired = trigger is not None and trigger.fires(
                            decisions_count, cur_length, True
                        )
                        if fired and advisory.any():
                            batch = dense_batch(world, frame - 1, q, legal)
                            chosen = int(batch["chosen"])
                            overrode = chosen != bare
                            search_roots += 1
                            if cur_length < 600:
                                search_roots_lt600 += 1
                                interventions_lt600 += int(overrode)
                            else:
                                search_roots_ge600 += 1
                                interventions_ge600 += int(overrode)
                            interventions += int(overrode)
                            branches_evaluated += int(batch["branches_evaluated"])
                            branches_skipped += int(batch["branches_skipped"])
                            native_v8_kernel_rows += int(batch["native_v8"]["kernel_rows"])
                            if include_raw:
                                raw_batches.append(
                                    {"frame": frame, "life": life, "bare": bare, **batch}
                                )
                        if not np.isfinite(q).all() or not 0 <= chosen < 6:
                            raise ValueError("nonfinite Q or illegal action")
                        selected.update(
                            frame=frame,
                            life=life,
                            length=cur_length,
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

                actions_holder: dict[str, np.ndarray] = {}

                def choose_record(sim: Any) -> np.ndarray:
                    actions = choose(sim)
                    if audit_live_mirror:
                        actions_holder["actions"] = actions.copy()
                    return actions

                world.sim.step_with_policy(choose_record)
                world.runtime.observe_step(world.sim)

                cur_grid_alive = [bool(x) for x in world.sim.alive[0].tolist()]
                for slot in range(int(world.sim.S)):
                    if cur_grid_alive[slot] and not prev_grid_alive[slot]:
                        grid_respawns[slot] += 1
                prev_grid_alive = cur_grid_alive

                grid_death = bool(
                    world.sim.get_transition_valid()[0, 0] and not world.sim.alive[0, 0]
                )
                grid_alive_now = bool(world.sim.alive[0, 0])
                grid_mass_now = int(world.sim.length[0, 0]) if grid_alive_now else 0

                if audit_live_mirror:
                    actual_decisions: list[int] = []
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
                    hero = session.live.game.snakes[0]
                    live_death = int(hero.id) in session.live.game.frame_death_causes
                    if live_death != grid_death:
                        raise ValueError("Watch/mirror death event mismatch")
                    if selected and int(hero.last_action) != selected["chosen"]:
                        raise ValueError("live delivered action mismatch")
                    if bool(hero.is_alive) != grid_alive_now:
                        raise ValueError("Watch/mirror alive state mismatch")
                    live_mass = int(hero.length) if hero.is_alive else 0
                    if live_mass != grid_mass_now:
                        raise ValueError("Watch/mirror hero length mismatch")

                if selected:
                    decisions_count += 1
                    boost_decisions += int(selected["chosen"] >= 3)
                    if include_raw:
                        raw_decisions.append(selected)

                death_len = int(selected.get("length", 0)) if grid_death else 0
                if grid_death:
                    deaths += 1
                    death_lengths.append(death_len)
                if grid_alive_now:
                    alive_frames += 1
                    mass_sum += grid_mass_now
                    alive_lengths.append(grid_mass_now)
                    if grid_mass_now > max_length:
                        max_length = grid_mass_now

                was_alive = grid_alive_now
                if include_raw:
                    raw_frames.append(
                        {
                            "frame": frame,
                            "life": life,
                            "alive": grid_alive_now,
                            "mass": grid_mass_now,
                            "death": grid_death,
                            "death_length": death_len,
                        }
                    )
                if on_frame is not None:
                    on_frame(frame)
                if time.monotonic() - last_progress >= 60:
                    print(
                        f"watch6-alllen seed={seed} arm={arm} frames={frame}/{horizon} "
                        f"roots={search_roots}",
                        flush=True,
                    )
                    last_progress = time.monotonic()
    finally:
        session.close()

    arr_lengths = np.asarray(alive_lengths, dtype=np.float64)
    summary: dict[str, Any] = {
        "arm": arm,
        "min_length": min_length,
        "interval": interval,
        "frames": horizon,
        "alive_frames": alive_frames,
        "mass_sum": mass_sum,
        "mass_integral": mass_sum / horizon,
        "deaths": deaths,
        "deaths_per_1k": deaths * 1000.0 / horizon,
        "lives": life,
        "decisions": decisions_count,
        "boost_decisions": boost_decisions,
        "max_length": max_length,
        "alive_length_median": float(np.median(arr_lengths)) if alive_lengths else 0.0,
        "alive_length_p95": float(np.percentile(arr_lengths, 95)) if alive_lengths else 0.0,
        "death_lengths": death_lengths,
        "short_body_deaths_lt600": sum(length < 600 for length in death_lengths),
        "long_body_deaths_ge600": sum(length >= 600 for length in death_lengths),
        "late_large_body_deaths_ge800": sum(length >= 800 for length in death_lengths),
        "search_roots": search_roots,
        "search_roots_lt600": search_roots_lt600,
        "search_roots_ge600": search_roots_ge600,
        "interventions": interventions,
        "interventions_lt600": interventions_lt600,
        "interventions_ge600": interventions_ge600,
        "branches_evaluated": branches_evaluated,
        "branches_skipped": branches_skipped,
        "native_v8_kernel_rows": native_v8_kernel_rows,
        "respawns_per_slot": grid_respawns,
        "mirror_identity_frames": horizon if audit_live_mirror else 0,
        "wall_seconds": time.monotonic() - started,
        "roster": roster,
    }
    if include_raw:
        summary["raw"] = {
            "frames": raw_frames,
            "decisions": raw_decisions,
            "batches": raw_batches,
        }
    return summary
