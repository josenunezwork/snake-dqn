"""Watch6-L1-V1 3-arm engine with full-horizon live/GridBatchSim mirror and record=None."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from research.watch6_l1_confirm import protocol as p
from research.watch6_search_efficacy.engine import DenseTrigger, dense_batch, step_guard
from research.watch6_search_efficacy.product import open_watch
from tools.ops.package_spec import digest

COLLAPSE_DECISIONS = 100_000
COLLAPSE_SHARE = 0.90


class ScientificFault(ValueError):
    """Raised on any live/GridBatchSim mirror divergence, nonfinite Q, or action collapse."""


def run_arm(
    seed: int,
    arm: str,
    check: Callable[[], None],
    *,
    horizon: int = p.HORIZON,
    record: Callable[[dict], None] | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> dict:
    """Run one Watch6-v2 arm ({bare, l2_prime, l1}) with per-frame live/GridBatchSim mirror."""
    if arm not in p.ARM_SPECS:
        raise ValueError(f"unknown Watch6-L1-V1 arm: {arm!r} (expected one of {p.ARMS})")
    if type(horizon) is not int or not 1 <= horizon <= p.HORIZON:
        raise ValueError(f"horizon must be an int in 1..{p.HORIZON}, got {horizon!r}")

    from research.bl_served_search.ceiling import _legal
    from research.t3_rollout_labels_20261015.bank_engine import prepared_q
    from research.t15_mixed_roster_20261017.episodes import _assert_mixed, roster_identity
    from src.core.game_config import GameConfig
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.eval_engine import _dispatch_actions

    spec = p.ARM_SPECS[arm]
    min_length = spec["min_length"]
    interval = int(spec["interval"])

    check()
    session = open_watch({"seed": int(seed)}, 24, "bare")
    decisions = 0
    boost = 0
    deaths = 0
    alive_frames = 0
    mass_sum = 0
    batches = 0
    batches_lt600 = 0
    batches_ge600 = 0
    batches_ge1000 = 0
    interventions = 0
    interventions_lt600 = 0
    interventions_ge600 = 0
    evaluated = 0
    branches_lt600 = 0
    branches_600_999 = 0
    branches_ge1000 = 0
    skipped = 0
    kernel_rows = 0
    mirror_frames = 0
    main_seconds = 0.0
    search_seconds = 0.0
    search_seconds_lt600 = 0.0
    search_seconds_600_999 = 0.0
    search_seconds_ge1000 = 0.0
    action_counts = [0] * 6
    recent_actions: list[int] = []
    all_frame_masses: list[int] = []
    alive_lengths: list[int] = []
    death_lengths: list[int] = []

    try:
        world = session.world
        if not world.sim.allow_respawn or any(
            not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
        ):
            raise ScientificFault("Watch6-L1-V1 requires continuous respawn in live and mirror")
        if getattr(session.live, "pathb", None) is not None:
            raise ScientificFault("Watch6-L1-V1 refuses Path B assistance")
        _assert_mixed(session.live, 6)
        live_roster = roster_identity(session.live)
        roster = {
            "hero_checkpoint_sha256": digest(Path(session.live.served_checkpoint["path"])),
            "opponent_checkpoint_sha256": digest(
                Path(session.live.watch_opponent_checkpoint["path"])
            ),
            "opponent_count": 5,
            "veto_variant": "v8",
            "veto_lambda": 8.0,
            "mechanics_version": int(GameConfig.MECHANICS_VERSION),
            "num_snakes": int(GameConfig.NUM_SNAKES),
            "live_roster": live_roster,
        }

        trigger = (
            DenseTrigger(min_length=int(min_length), interval=interval)
            if isinstance(min_length, int)
            else None
        )
        was_alive = bool(world.sim.alive[0, 0])
        previous_alive = [bool(s.is_alive) for s in session.live.game.snakes]
        respawns = [0] * 6

        if record is not None:
            record(session.live.snapshot())

        with step_guard(check):
            for frame_index in range(1, horizon + 1):
                frame_started = time.monotonic()
                step_search = 0.0
                check()
                if on_frame is not None:
                    on_frame(frame_index)
                selected: dict[str, Any] = {}

                def choose(sim: Any) -> np.ndarray:
                    nonlocal batches, batches_lt600, batches_ge600, batches_ge1000
                    nonlocal interventions, interventions_lt600, interventions_ge600
                    nonlocal evaluated, branches_lt600, branches_600_999, branches_ge1000
                    nonlocal skipped, kernel_rows, step_search, trigger
                    nonlocal search_seconds_lt600, search_seconds_600_999, search_seconds_ge1000
                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive and isinstance(min_length, int):
                        trigger = DenseTrigger(min_length=int(min_length), interval=interval)
                    actions = np.ones((1, sim.S), dtype=np.int64)
                    if active:
                        length = int(sim.length[0, 0])
                        obs = build_ego_raster(
                            sim,
                            EgoRasterConfig(version="bv8"),
                            backend="numba",
                            rows=np.array([[0, 0]]),
                        )
                        q_row = np.asarray(prepared_q(world, obs), dtype=np.float32)
                        if not np.isfinite(q_row).all():
                            raise ScientificFault("nonfinite hero Q row")
                        advisory = sim.get_advisory_action_mask()[0, 0]
                        legal, bare_action = _legal(q_row, advisory)
                        chosen = int(bare_action)
                        # Pass alive=True to DenseTrigger.fires so an empty advisory mask on the
                        # first length-eligible decision still anchors the 5-decision clock,
                        # matching V3, watch6_alllen_screen, watch_replay, and ceiling.py:236.
                        fired = trigger is not None and trigger.fires(decisions, length, True)
                        if fired and bool(advisory.any()):
                            t0 = time.monotonic()
                            batch = dense_batch(world, frame_index - 1, q_row, legal)
                            batch_dt = time.monotonic() - t0
                            step_search += batch_dt
                            chosen = int(batch["chosen"])
                            if not 0 <= chosen < 6 or not bool(legal[chosen]):
                                raise ScientificFault("illegal exact-stop action selection")
                            batches += 1
                            intervened = int(chosen != bare_action)
                            interventions += intervened
                            b_eval = int(batch["branches_evaluated"])
                            if length < 600:
                                batches_lt600 += 1
                                interventions_lt600 += intervened
                                branches_lt600 += b_eval
                                search_seconds_lt600 += batch_dt
                            else:
                                batches_ge600 += 1
                                interventions_ge600 += intervened
                                if length < 1000:
                                    branches_600_999 += b_eval
                                    search_seconds_600_999 += batch_dt
                                else:
                                    batches_ge1000 += 1
                                    branches_ge1000 += b_eval
                                    search_seconds_ge1000 += batch_dt
                            evaluated += b_eval
                            skipped += int(batch["branches_skipped"])
                            kernel_rows += int(batch["native_v8"]["kernel_rows"])
                        selected.update(
                            length=length,
                            bare=int(bare_action),
                            chosen=chosen,
                            q=q_row.copy(),
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
                            raise ScientificFault("bare dispatch/Q mismatch")
                        actions[0, 0] = selected["chosen"]
                    return actions

                actions_holder: dict[str, np.ndarray] = {}

                def choose_record(sim: Any) -> np.ndarray:
                    actions = choose(sim)
                    actions_holder["actions"] = actions.copy()
                    return actions

                world.sim.step_with_policy(choose_record)
                world.runtime.observe_step(world.sim)

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
                        raise ScientificFault(f"live/mirror B8 Q mismatch at frame {frame_index}")
                    return value

                hero.update = observed_update
                session.live.policy._next_dispatch_q = observed_q
                try:
                    session._step_live(actions_holder["actions"])
                finally:
                    hero.update = original_update
                    session.live.policy._next_dispatch_q = original_q

                if actual_decisions != ([selected["length"]] if selected else []):
                    raise ScientificFault(
                        f"live/mirror actual hero decision mismatch at frame {frame_index}"
                    )

                if record is not None:
                    record(session.live.snapshot())

                hero = session.live.game.snakes[0]
                death = int(hero.id) in session.live.game.frame_death_causes
                grid_death = bool(
                    world.sim.get_transition_valid()[0, 0] and not world.sim.alive[0, 0]
                )
                if death != grid_death:
                    raise ScientificFault(
                        f"Watch/mirror death event mismatch at frame {frame_index}"
                    )
                if selected:
                    if int(hero.last_action) != selected["chosen"]:
                        raise ScientificFault(
                            f"live delivered action mismatch at frame {frame_index}"
                        )
                    decisions += 1
                    chosen_act = int(selected["chosen"])
                    boost += int(chosen_act >= 3)
                    action_counts[chosen_act] += 1
                    recent_actions.append(chosen_act)
                    if len(recent_actions) >= COLLAPSE_DECISIONS:
                        window = recent_actions[-COLLAPSE_DECISIONS:]
                        if max(window.count(a) for a in range(6)) > int(
                            COLLAPSE_DECISIONS * COLLAPSE_SHARE
                        ):
                            raise ScientificFault("unconditional action-collapse fault (>90%)")

                was_alive = bool(hero.is_alive)
                if was_alive != bool(world.sim.alive[0, 0]) or (
                    was_alive and int(hero.length) != int(world.sim.length[0, 0])
                ):
                    raise ScientificFault(
                        f"live/mirror alive or length mismatch at frame {frame_index}"
                    )

                alive_now = [bool(s.is_alive) for s in session.live.game.snakes]
                for slot in range(6):
                    respawns[slot] += int(alive_now[slot] and not previous_alive[slot])
                previous_alive = alive_now

                if death:
                    deaths += 1
                    death_lengths.append(int(selected.get("length", 0)))

                if was_alive:
                    cur_len = int(hero.length)
                    alive_frames += 1
                    mass_sum += cur_len
                    all_frame_masses.append(cur_len)
                    alive_lengths.append(cur_len)
                else:
                    all_frame_masses.append(0)

                main_seconds += time.monotonic() - frame_started - step_search
                search_seconds += step_search
                mirror_frames += 1

        if mirror_frames != horizon:
            raise ScientificFault(
                f"incomplete mirror frame coverage ({mirror_frames} != {horizon})"
            )

        return {
            "arm": arm,
            "frames": horizon,
            "alive_frames": alive_frames,
            "mass_sum": mass_sum,
            "deaths": deaths,
            "decisions": decisions,
            "boost_decisions": boost,
            "action_counts": action_counts,
            "length_median": float(np.median(all_frame_masses)),
            "alive_length_median": float(np.median(alive_lengths)) if alive_lengths else 0.0,
            "length_p95": float(np.percentile(alive_lengths, 95)) if alive_lengths else 0.0,
            "max_length": max(alive_lengths, default=0),
            "death_lengths": death_lengths,
            "short_body_deaths_lt600": sum(length < 600 for length in death_lengths),
            "long_body_deaths_ge600": sum(length >= 600 for length in death_lengths),
            "late_large_body_deaths": sum(length >= 800 for length in death_lengths),
            "batches": batches,
            "batches_lt600": batches_lt600,
            "batches_ge600": batches_ge600,
            "batches_ge1000": batches_ge1000,
            "interventions": interventions,
            "interventions_lt600": interventions_lt600,
            "interventions_ge600": interventions_ge600,
            "branches_evaluated": evaluated,
            "branches_skipped": skipped,
            "native_v8_kernel_rows": kernel_rows,
            "mirror_identity_frames": mirror_frames,
            "respawns_per_slot": respawns,
            "timing": {
                "main_seconds": main_seconds,
                "search_seconds": search_seconds,
                "search_seconds_lt600": search_seconds_lt600,
                "search_seconds_600_999": search_seconds_600_999,
                "search_seconds_ge1000": search_seconds_ge1000,
                "branches_lt600": branches_lt600,
                "branches_600_999": branches_600_999,
                "branches_ge1000": branches_ge1000,
            },
            "roster": roster,
        }
    finally:
        session.close()
