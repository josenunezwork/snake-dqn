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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import league_pool as optimized

FIXTURES = Path(__file__).parent / "fixtures" / "t22"
T24_FIXTURES = Path(__file__).parent / "fixtures" / "t24"
PRECHANGE_V8_SHA256 = "72947d97a3d3d293a6f99e76a88236f38c933eda3dbf605e99399e1b58082e32"
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


def prechange_v8_module():
    """Load the T24 baseline kernel without consulting the working-tree module."""
    path = T24_FIXTURES / "v8_nb_pre_t24.py"
    spec = importlib.util.spec_from_file_location("_t24_prechange_v8", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def use_prechange_v8(module):
    """Route dynamic v8 imports in the independent legacy oracle to its frozen kernel."""
    import src.simd_env as simd_env

    name = "src.simd_env.v8_nb"
    current = sys.modules.get(name)
    current_attr = getattr(simd_env, "v8_nb", None)
    sys.modules[name] = module
    setattr(simd_env, "v8_nb", module)
    try:
        yield
    finally:
        if current is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = current
        if current_attr is None:
            delattr(simd_env, "v8_nb")
        else:
            setattr(simd_env, "v8_nb", current_attr)


def assert_v8_result_bytes_equal(expected, actual, frame: int) -> None:
    """Compare every stored V8Result array, including all diagnostic planes."""
    from dataclasses import fields

    assert type(expected).__name__ == type(actual).__name__ == "V8Result"
    for field in fields(expected):
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        assert left.dtype == right.dtype and left.shape == right.shape, (frame, field.name)
        assert left.tobytes() == right.tobytes(), (frame, field.name)


def test_reference_is_unedited_pre_t22_source():
    assert (
        hashlib.sha256((FIXTURES / "league_pool_pre_t22.py").read_bytes()).hexdigest()
        == "18c3b2e7257af0410069d0f344f3bbe37dbee0f08d5a8e53938b87667575919c"
    )


def test_prechange_v8_kernel_is_hash_pinned() -> None:
    assert hashlib.sha256((T24_FIXTURES / "v8_nb_pre_t24.py").read_bytes()).hexdigest() == (
        PRECHANGE_V8_SHA256
    )


def test_bodymap_bfs_visit_order_matches_prechange_kernel() -> None:
    """Pin FIFO discovery order, not merely the capped component size.

    The retained-body optimization may share painted maps, but each v8 action
    still has to observe the reference ``(+x, -x, +y, -y)`` BFS order.  Queue
    contents and the seen stamp expose that ordering without relying on a
    decision-level coincidence.
    """
    import src.simd_env.v8_nb as current

    baseline = prechange_v8_module()
    gw, gh = 9, 8
    n = gw * gh
    blocked_one = np.zeros(n, dtype=np.int64)
    blocked_two = np.zeros(n, dtype=np.int64)
    # A wall with two asymmetric gates makes neighbour order visible in both
    # the queue prefix and the full capped visit set.
    for y in range(gh):
        if y not in (1, 6):
            blocked_one[4 * gh + y] = 17
    blocked_two[2 * gh + 2] = 19
    blocked_two[6 * gh + 5] = 19
    release_stamp = np.zeros(n, dtype=np.int64)
    release_value = np.zeros(n, dtype=np.int64)
    release_stamp[3 * gh + 4] = 23
    release_value[3 * gh + 4] = 5
    release_stamp[5 * gh + 3] = 23
    release_value[5 * gh + 3] = 2

    def visit(module, start, use_release, cap):
        ws = module._workspace(gw, gh)
        seen, qx, qy, qd = ws[4], ws[7], ws[8], ws[9]
        seen.fill(0)
        count = module._bfs(
            start[0],
            start[1],
            gw,
            gh,
            blocked_one,
            17,
            blocked_two,
            19,
            release_stamp,
            release_value,
            23,
            use_release,
            seen,
            31,
            qx,
            qy,
            qd,
            cap,
        )
        return count, seen.copy(), qx[:count].copy(), qy[:count].copy(), qd[:count].copy()

    for start, use_release, cap in (((0, 0), False, 64), ((1, 5), True, 23), ((8, 7), True, 11)):
        expected = visit(baseline, start, use_release, cap)
        actual = visit(current, start, use_release, cap)
        for left, right in zip(expected, actual):
            if isinstance(left, np.ndarray):
                assert left.tobytes() == right.tobytes(), (start, use_release, cap)
            else:
                assert left == right, (start, use_release, cap)


def test_bodymap_more_than_64_slots_uses_exact_general_fallback() -> None:
    """The compact per-world bitset must not truncate slot 64 or above."""
    from types import SimpleNamespace

    import src.simd_env.v8_nb as current

    baseline = prechange_v8_module()
    slots, ring, gw, gh = 65, 4, 9, 9
    bodies = np.zeros((1, slots, ring, 2), dtype=np.int64)
    # Slot 64 closes the last exit for slot 0; a truncated 64-bit membership
    # mask would report a very different v2 flood for the selected rows.
    bodies[0, 0, 0] = (4, 4)
    bodies[0, 1, 0] = (3, 4)
    bodies[0, 2, 0] = (4, 3)
    bodies[0, 3, 0] = (4, 5)
    bodies[0, 64, 0] = (5, 4)
    sim = SimpleNamespace(
        cfg=SimpleNamespace(
            arena_type="rectangular",
            min_boost_length=5,
            boost_length_cost_frames=8,
            mechanics_version=2,
        ),
        grid_w=gw,
        grid_h=gh,
        bodies=bodies,
        head_ptr=np.zeros((1, slots), dtype=np.int64),
        seg_count=np.ones((1, slots), dtype=np.int64),
        length=np.ones((1, slots), dtype=np.int64),
        boost_frames=np.zeros((1, slots), dtype=np.int64),
        direction=np.ones((1, slots), dtype=np.int64),
        alive=np.isin(np.arange(slots)[None, :], [0, 1, 2, 3, 64]),
    )
    rows = np.array([[0, 64], [0, 0], [0, 64]], dtype=np.int64)
    q = np.array([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [1, 1, 1, 1, 1, 1]], np.float32)
    mask = np.ones((len(rows), 6), dtype=bool)
    expected = baseline.v8_decide(sim, rows, q, mask, features=True)
    actual = current.v8_decide(sim, rows, q, mask, features=True)
    assert_v8_result_bytes_equal(expected, actual, frame=0)


def test_bodymap_workspace_thread_isolation_and_stamp_rollover() -> None:
    """Concurrent reads and a near-overflow stamp retain frozen-kernel bytes."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import src.simd_env.v8_nb as current

    baseline = prechange_v8_module()
    gw = gh = 11
    bodies = np.zeros((1, 4, 8, 2), dtype=np.int64)
    for slot, cells in enumerate(
        (((5, 5), (4, 5), (3, 5)), ((7, 5),), ((5, 7), (5, 8)), ((2, 2),))
    ):
        for index, cell in enumerate(cells):
            bodies[0, slot, (-index) % 8] = cell
    sim = SimpleNamespace(
        cfg=SimpleNamespace(
            arena_type="rectangular",
            min_boost_length=3,
            boost_length_cost_frames=3,
            mechanics_version=2,
        ),
        grid_w=gw,
        grid_h=gh,
        bodies=bodies,
        head_ptr=np.zeros((1, 4), dtype=np.int64),
        seg_count=np.array([[3, 1, 2, 1]], dtype=np.int64),
        length=np.array([[4, 1, 2, 1]], dtype=np.int64),  # physical n < logical L growth lag
        boost_frames=np.array([[2, 0, 1, 0]], dtype=np.int64),
        direction=np.array([[1, 3, 0, 2]], dtype=np.int64),
        alive=np.ones((1, 4), dtype=bool),
    )
    rows = np.array([[0, 2], [0, 0], [0, 2], [0, 1]], dtype=np.int64)
    mask = np.array(
        [[True, True, True, True, True, True], [False] * 6, [True] * 6, [True] * 6],
        dtype=bool,
    )
    qs = [
        np.array([[2, 2, 1, 3, 0, 3], [3, 3, 2, 3, 3, 2], [1] * 6, [0, 1, 2, 3, 4, 5]], np.float32),
        np.array([[5, 4, 3, 2, 1, 0], [0] * 6, [2, 2, 2, 2, 2, 2], [5, 4, 3, 2, 1, 0]], np.float32),
    ]
    expected = [baseline.v8_decide(sim, rows, q, mask, features=True) for q in qs]

    def invoke(index):
        return current.v8_decide(sim, rows, qs[index], mask, features=True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        actual = list(workers.map(invoke, (0, 1)))
    for old, new in zip(expected, actual):
        assert_v8_result_bytes_equal(old, new, frame=0)

    workspace = current._workspace(gw, gh)
    workspace[0][:] = np.iinfo(np.int64).max - 2
    try:
        rollover = current.v8_decide(sim, rows, qs[0], mask, features=True)
    finally:
        workspace[0][:] = 1
    assert_v8_result_bytes_equal(expected[0], rollover, frame=1)


def run_recorded_pair(
    seed: int = 741,
    frames: int = 96,
    long_body_length: int = 0,
    force_lifetime_boundary: bool = True,
):
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
    if long_body_length:
        from src.simd_env.grid_scenarios import inject_serpentines

        # Physical ring bodies (not merely logical target length) exercise the
        # retained-body map at L >= 1000 on both lockstep simulator instances.
        inject_serpentines(sims, [long_body_length] * sims[0].S, box_height=8)
    learner = trainer.learner_seat_mask(seed, 0, sims[0].E, 3)
    seats = np.full(learner.shape, None, dtype=object)
    # Balanced immutable assignment exercises every identity on nonzero slots.
    for i, row in enumerate(np.argwhere(~learner)):
        seats[tuple(row)] = POLICIES[i % len(POLICIES)]
    vector = np.isin(seats, POLICIES[1:4])
    runtimes[1].controlled[:] = vector
    assert np.count_nonzero(runtimes[0].controlled) > np.count_nonzero(vector)
    baseline_v8 = prechange_v8_module()
    observed_segments = {"learner": 0, **{policy_id: 0 for policy_id in POLICIES}}
    recorded, deaths, respawns = [], set(), set()
    previous_alive = sims[0].get_alive()
    for frame in range(frames):
        snapshots = []
        for index, (sim, runtime, pool) in enumerate(zip(sims, runtimes, pools)):

            def choose(prepared):
                alive, masks = prepared.get_alive(), prepared.get_resolved_action_mask()
                runtime.prepare(prepared)
                hero = np.argwhere(alive & learner)
                if len(hero):
                    observed_segments["learner"] = max(
                        observed_segments["learner"],
                        int(prepared.seg_count[hero[:, 0], hero[:, 1]].max()),
                    )
                for policy_id in POLICIES:
                    policy_rows = np.argwhere((seats == policy_id) & alive)
                    if len(policy_rows):
                        observed_segments[policy_id] = max(
                            observed_segments[policy_id],
                            int(prepared.seg_count[policy_rows[:, 0], policy_rows[:, 1]].max()),
                        )
                if index:
                    obs = pool.prime_b8_with_learner_rows(prepared, hero, alive, seats)
                    extras, v8res = pool.v8_with_learner_rows(
                        prepared,
                        hero,
                        trainer.hero_selection_mask(prepared, hero),
                        alive,
                        seats,
                    )
                else:
                    obs = None
                    # The legacy pool and learner must not silently share the working-tree
                    # kernel with the optimized side.  Its dynamic imports resolve to the
                    # byte-pinned pre-change v8 source for this entire prepared callback.
                    with use_prechange_v8(baseline_v8):
                        extras, v8res = v8_scalars(
                            prepared, hero, trainer.hero_selection_mask(prepared, hero)
                        )
                if obs is None:
                    obs = build_ego_raster(
                        prepared, EgoRasterConfig(version="b"), backend="numba", rows=hero
                    )
                # Independent deterministic learner actions preserve prepared-world semantics.
                actions = np.ones((sim.E, sim.S), dtype=np.int64)
                actions[hero[:, 0], hero[:, 1]] = masks[hero[:, 0], hero[:, 1]].argmax(axis=1)
                if index:
                    pool.dispatch(prepared, masks, alive, actions, seats)
                else:
                    with use_prechange_v8(baseline_v8):
                        pool.dispatch(prepared, masks, alive, actions, seats)
                snapshots.append((actions.copy(), alive.copy(), obs, extras.copy(), v8res))
                return actions

            sim.step_with_policy(choose)
        expected, actual = snapshots
        for left, right in zip(expected[:2], actual[:2]):
            np.testing.assert_array_equal(left, right, err_msg=f"frame {frame}")
        for key in ("local", "global", "scalars"):
            assert expected[2][key].tobytes() == actual[2][key].tobytes(), (frame, key)
        assert expected[3].tobytes() == actual[3].tobytes(), frame
        assert_v8_result_bytes_equal(expected[4], actual[4], frame)
        alive = expected[1]
        deaths.update(seats[e, s] for e, s in np.argwhere(previous_alive & ~alive) if seats[e, s])
        respawns.update(seats[e, s] for e, s in np.argwhere(~previous_alive & alive) if seats[e, s])
        previous_alive = alive
        recorded.append(expected[0])
        if force_lifetime_boundary and frame == 4:
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
    if force_lifetime_boundary:
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
    return np.stack(recorded), seats, observed_segments


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires all three pinned pool checkpoints")
def test_all_six_real_policy_streams_match_frozen_source_and_recording():
    actual, seats, _ = run_recorded_pair()
    with np.load(FIXTURES / "actions_seed741.npz", allow_pickle=False) as expected:
        np.testing.assert_array_equal(actual, expected["actions"])
        np.testing.assert_array_equal(seats.astype(str), expected["seats"])


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires all three pinned pool checkpoints")
def test_all_six_real_policy_streams_match_prechange_v8_on_physical_long_bodies():
    """The independent six-policy oracle also covers the >=1000 segment body-map path."""
    actual, _, observed_segments = run_recorded_pair(
        seed=743, frames=12, long_body_length=1000, force_lifetime_boundary=False
    )
    assert actual.shape == (12, 12, 6)
    assert all(value >= 1000 for value in observed_segments.values()), observed_segments


@pytest.mark.skipif(not HAVE_CHECKPOINTS, reason="requires pinned B8 checkpoint")
def test_shared_raster_and_v8_bytes_cache_lifetime(monkeypatch):
    import src.simd_env.ego_raster_bv8 as bv8
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

    # A single eager v8 call feeds both learner and B8 rows.  Its B8 slice is
    # consumed once; a second call cannot reuse stale result or raster storage.
    calls = 0
    original_v8_scalars = bv8.v8_scalars

    def counted_v8_scalars(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_v8_scalars(*args, **kwargs)

    monkeypatch.setattr(bv8, "v8_scalars", counted_v8_scalars)
    learner_mask = sim.get_advisory_action_mask()[hero[:, 0], hero[:, 1]]
    learner_extras, learner_result = policy.prime_v8_with_learner_rows(
        sim, hero, learner_mask, frozen
    )
    both = np.concatenate([hero, frozen])
    both_mask = sim.get_advisory_action_mask()[both[:, 0], both[:, 1]]
    expected_extras, expected_result = original_v8_scalars(sim, both, both_mask)
    assert learner_extras.tobytes() == expected_extras[: len(hero)].tobytes()
    for field in (
        "action",
        "base",
        "v5_action",
        "v7_action",
        "outcome",
        "v5_reason",
        "counts",
        "cap",
        "need",
        "area_cap",
        "area",
        "landing",
        "head_risky",
        "head_waived",
        "v5_eligible",
    ):
        assert (
            getattr(learner_result, field).tobytes()
            == getattr(expected_result, field)[: len(hero)].tobytes()
        )
    assert calls == 1
    masks = sim.get_resolved_action_mask()[frozen[:, 0], frozen[:, 1]]
    cached = policy.actions(masks, sim, frozen)
    assert calls == 1
    uncached = policy.actions(masks, sim, frozen)
    assert calls == 2
    np.testing.assert_array_equal(cached, uncached)

    # A changed advisory mask must invalidate the v8 cache even before a step.
    policy.prime_with_learner_rows(sim, hero, frozen)
    policy.prime_v8_with_learner_rows(sim, hero, learner_mask, frozen)
    advisory = sim.get_advisory_action_mask().copy()
    advisory[frozen[0, 0], frozen[0, 1]] ^= np.array([True, False, False, False, False, False])
    monkeypatch.setattr(sim, "get_advisory_action_mask", lambda: advisory)
    policy.actions(masks, sim, frozen)
    assert calls == 4  # prime then mismatch fallback; neither cache survives the dispatch

    # A stale same-row cache must not be used after a simulator step.
    sim.step(np.ones((sim.E, sim.S), dtype=np.int64))
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
