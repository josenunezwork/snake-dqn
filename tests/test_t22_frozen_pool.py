"""Independent pre-T22 pool oracle, recorded streams, and exact shared-feature checks.

The fixture is the unedited league_pool.py from 4c1e1b2. Each real-policy run
uses separate simulators, runtimes, policies, and RNGs, in the actual prepared
step callback. Checkpoints are optional on CI, but required for the Mac receipt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import league_pool as optimized

FIXTURES = Path(__file__).parent / "fixtures" / "t22"
CHECKPOINTS = Path(
    os.environ.get("SNAKE_CKPT_DIR", "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
)
POLICIES = (
    "b8-s3",
    "frp3-s12",
    "frp3-s12-v8",
    "champion-v8",
    "scripted:greedy_food",
    "scripted:random_safe",
)
HAVE_CHECKPOINTS = all(
    (CHECKPOINTS / name).is_file()
    for name in (
        "b8_s3_t7_20261016.pth",
        "frp3_m3_s12_u60000_20261005.pth",
        "champion_a5_freespace_20260621.pth",
    )
)


def reference_module():
    path = FIXTURES / "league_pool_pre_t22.py"
    spec = importlib.util.spec_from_file_location("_t22_frozen_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_is_unedited_pre_t22_source():
    assert (
        hashlib.sha256((FIXTURES / "league_pool_pre_t22.py").read_bytes()).hexdigest()
        == "18c3b2e7257af0410069d0f344f3bbe37dbee0f08d5a8e53938b87667575919c"
    )


def run_recorded_pair(seed: int = 741, frames: int = 96):
    """Compare real policies on lockstep worlds, including scripted forced mortality.

    After frame four, one live row of each identity is marked dead and its grid
    rebuilt before observe_step. The ordinary respawn mechanism then restores
    it. This deterministically covers every policy's lifetime boundary without
    changing the policy actions or depending on rare natural deaths.
    """
    import torch

    from research.redesign_m3c_20261008 import train_m3c as trainer
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.ego_raster_bv8 import v8_scalars

    torch.set_num_threads(1)
    args = trainer.default_args()
    args.worlds, args.seed, args.learner_seats, args.v61_backend = 12, seed, 3, "numba"
    args.obs_version = "bv8"
    ctx = gid._context(1)
    sims, runtimes, pools = [], [], []
    for module in (reference_module(), optimized):
        sim, runtime, _, _ = trainer.build_segment(args, ctx, 0, 0.99)
        pool = module.build_static_pool(
            module.default_pool_identities(str(CHECKPOINTS)),
            world_seeds=sim.get_world_seeds(),
            runtime=runtime,
            device="cpu",
        )
        sims.append(sim)
        runtimes.append(runtime)
        pools.append(pool)
    learner = trainer.learner_seat_mask(seed, 0, sims[0].E, 3)
    seats = np.full(learner.shape, None, dtype=object)
    # Balanced immutable assignment exercises every identity on nonzero slots.
    for i, row in enumerate(np.argwhere(~learner)):
        seats[tuple(row)] = POLICIES[i % len(POLICIES)]
    vector = np.isin(seats, POLICIES[1:4])
    runtimes[1].controlled[:] = vector
    assert np.count_nonzero(runtimes[0].controlled) > np.count_nonzero(vector)
    recorded, deaths, respawns = [], set(), set()
    previous_alive = sims[0].get_alive()
    for frame in range(frames):
        snapshots = []
        for index, (sim, runtime, pool) in enumerate(zip(sims, runtimes, pools)):

            def choose(prepared):
                alive, masks = prepared.get_alive(), prepared.get_resolved_action_mask()
                runtime.prepare(prepared)
                hero = np.argwhere(alive & learner)
                if index:
                    obs = pool.prime_b8_with_learner_rows(prepared, hero, alive, seats)
                else:
                    obs = None
                if obs is None:
                    obs = build_ego_raster(
                        prepared, EgoRasterConfig(version="b"), backend="numba", rows=hero
                    )
                extras, _ = v8_scalars(prepared, hero, trainer.hero_selection_mask(prepared, hero))
                # Independent deterministic learner actions preserve prepared-world semantics.
                actions = np.ones((sim.E, sim.S), dtype=np.int64)
                actions[hero[:, 0], hero[:, 1]] = masks[hero[:, 0], hero[:, 1]].argmax(axis=1)
                pool.dispatch(prepared, masks, alive, actions, seats)
                snapshots.append((actions.copy(), alive.copy(), obs, extras.copy()))
                return actions

            sim.step_with_policy(choose)
        expected, actual = snapshots
        for left, right in zip(expected[:2], actual[:2]):
            np.testing.assert_array_equal(left, right, err_msg=f"frame {frame}")
        for key in ("local", "global", "scalars"):
            assert expected[2][key].tobytes() == actual[2][key].tobytes(), (frame, key)
        assert expected[3].tobytes() == actual[3].tobytes(), frame
        alive = expected[1]
        deaths.update(seats[e, s] for e, s in np.argwhere(previous_alive & ~alive) if seats[e, s])
        respawns.update(seats[e, s] for e, s in np.argwhere(~previous_alive & alive) if seats[e, s])
        previous_alive = alive
        recorded.append(expected[0])
        if frame == 4:
            for policy_id in POLICIES:
                row = tuple(np.argwhere((seats == policy_id) & sims[0].alive)[0])
                for sim in sims:
                    sim.alive[row] = False
                    sim.respawn_timer[row] = 2
                    sim._last_done[row] = True
                    sim._rebuild_env_grids(row[0])
        for sim, runtime in zip(sims, runtimes):
            runtime.observe_step(sim)
        for name in ("sel_state", "sel_mask", "_carry_state", "_carry_mask", "_carry_valid"):
            assert (
                getattr(runtimes[0], name)[vector].tobytes()
                == getattr(runtimes[1], name)[vector].tobytes()
            ), (frame, name)
        for name in ("prev_id", "prev_dist"):
            assert (
                getattr(runtimes[0].memory, name)[vector].tobytes()
                == getattr(runtimes[1].memory, name)[vector].tobytes()
            ), (frame, name)
        for name in (
            "alive",
            "bodies",
            "head_ptr",
            "seg_count",
            "length",
            "direction",
            "frame",
            "respawn_timer",
        ):
            np.testing.assert_array_equal(
                getattr(sims[0], name), getattr(sims[1], name), err_msg=f"{frame} {name}"
            )
        assert sims[0].food_cells == sims[1].food_cells
        assert [rng._rng.getstate() for rng in sims[0]._rngs] == [
            rng._rng.getstate() for rng in sims[1]._rngs
        ]
    assert deaths == set(POLICIES) and respawns == set(POLICIES)
    assert all(
        not p.requires_grad
        for policy in pools[1].policies.values()
        for p in (
            getattr(policy, "network", getattr(policy, "net", None)).parameters()
            if hasattr(policy, "network") or hasattr(policy, "net")
            else []
        )
    )
    return np.stack(recorded), seats


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires all three pinned pool checkpoints")
def test_all_six_real_policy_streams_match_frozen_source_and_recording():
    actual, seats = run_recorded_pair()
    with np.load(FIXTURES / "actions_seed741.npz", allow_pickle=False) as expected:
        np.testing.assert_array_equal(actual, expected["actions"])
        np.testing.assert_array_equal(seats.astype(str), expected["seats"])


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires pinned B8 checkpoint")
def test_shared_raster_bytes_and_cache_lifetime():
    from research.redesign_m3c_20261008 import train_m3c as trainer
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    args = trainer.default_args()
    args.worlds, args.learner_seats = 2, 3
    sim, _, _, _ = trainer.build_segment(args, gid._context(1), 0, 0.99)
    policy = optimized.LeagueBV8Policy(str(CHECKPOINTS / "b8_s3_t7_20261016.pth"), "cpu")
    hero, frozen = np.array([[0, 0], [1, 1]]), np.array([[0, 2], [1, 3]])
    expected = build_ego_raster(sim, EgoRasterConfig(version="b"), backend="numba", rows=hero)
    got = policy.prime_with_learner_rows(sim, hero, frozen)
    for key in ("local", "global", "scalars"):
        assert got[key].tobytes() == expected[key].tobytes()
    # A stale same-row cache must not be used after a simulator step.
    sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
    masks = sim.get_resolved_action_mask()[frozen[:, 0], frozen[:, 1]]
    cached = policy.actions(masks, sim, frozen)
    uncached = policy.actions(masks, sim, frozen)
    np.testing.assert_array_equal(cached, uncached)
    assert policy._base_cache_slots is None


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires pinned vector checkpoint")
def test_compiled_v8_rebind_and_snapshot_restore_use_new_runtime():
    from research.redesign_m3c_20261008.execution_snapshot import (
        capture_policy_execution,
        restore_policy_execution,
    )
    from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

    inner = Vector61SimdPolicy(
        str(CHECKPOINTS / "frp3_m3_s12_u60000_20261005.pth"),
        Vector61Runtime(),
        veto_slots=range(6),
        veto_variant="v8",
        veto_lambda=8.0,
        forward="batched",
    )
    policy = optimized.LeagueVector61Policy(inner)
    replacement = Vector61Runtime()
    policy.runtime = replacement
    assert inner.runtime is replacement
    policy.last_veto = {"final": np.array([2, 1])}
    state = capture_policy_execution({"v8": policy})
    inner.last_veto = None
    restore_policy_execution({"v8": policy}, state)
    np.testing.assert_array_equal(inner.last_veto["final"], [2, 1])


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires pinned vector checkpoint")
def test_compiled_v8_keeps_tied_q_actions_on_all_slots():
    import torch

    from research.redesign_m3c_20261008 import train_m3c as trainer
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    args = trainer.default_args()
    args.worlds, args.learner_seats, args.v61_backend = 2, 3, "numba"
    sim, runtime, _, _ = trainer.build_segment(args, gid._context(1), 0, 0.99)
    runtime.controlled[:] = True
    reference = Vector61SimdPolicy(
        str(CHECKPOINTS / "frp3_m3_s12_u60000_20261005.pth"),
        runtime,
        veto_slots=range(6),
        veto_variant="v8",
        veto_lambda=8.0,
        forward="batched",
    )
    reference.q_values = lambda states: torch.from_numpy(
        np.tile(np.array([3, 3, 2, 3, 3, 2], np.float32), (len(states), 1))
    )
    compiled = optimized.LeagueVector61Policy(reference)
    for _ in range(12):

        def choose(prepared):
            runtime.prepare(prepared)
            rows = np.argwhere(prepared.alive)
            masks = prepared.get_resolved_action_mask()[rows[:, 0], rows[:, 1]]
            expected = reference.actions(masks, prepared, rows)
            np.testing.assert_array_equal(compiled.actions(masks, prepared, rows), expected)
            actions = np.ones((sim.E, sim.S), dtype=np.int64)
            actions[rows[:, 0], rows[:, 1]] = expected
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)
