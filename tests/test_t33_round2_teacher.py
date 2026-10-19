"""Round-two teacher contracts: compact rings and lazy epsilon forks."""

from __future__ import annotations

import json

import numpy as np

from research.phase0_search_20261019 import teacher as phase0
from research.t33_batched_teacher.teacher import (
    BatchedTeacher,
    BranchLayout,
    _copy_environment,
    _predrawn_epsilon,
    merge_worlds,
)
from tests.test_t33_teacher_integration import _prepared_world


def _world(mix: str = "scripted"):
    bank = json.loads(
        open("research/phase0_search_20261019/WORLD_BANK.json", encoding="utf-8").read()
    )
    row = next(item for item in bank["fixtures"]["timing"] if item["mix"] == mix)
    return _prepared_world(row, 800)


def test_merge_recaps_int64_ring_and_preserves_grid_step_contract() -> None:
    """A compact ring keeps modulo slot semantics and grid invariants."""
    world = _world()
    original_cap = int(world.sim.cap)
    horizon = 7
    merged = merge_worlds([world], rollouts=2, horizon=horizon)
    expected = int(world.sim.seg_count.max()) + horizon + 1
    assert merged.sim.cap == expected < original_cap
    assert merged.sim.bodies.dtype == np.int64
    assert merged.sim.bodies.flags.c_contiguous
    merged.sim.check_grid_invariants()

    # Both initial rollouts must be independent, compact copies of the source.
    np.testing.assert_array_equal(merged.sim.bodies[0], merged.sim.bodies[1])
    merged.sim.bodies[0, 0, 0, 0] += 1
    assert merged.sim.bodies[0, 0, 0, 0] != merged.sim.bodies[1, 0, 0, 0]


def test_fork_copies_rng_runtime_and_simulator_environment() -> None:
    """A promoted follower receives an isolated leader snapshot."""
    merged = merge_worlds([_world("frozen")], rollouts=2, horizon=8)
    leader, follower = 0, 1
    _copy_environment(merged, leader, follower)
    np.testing.assert_array_equal(merged.sim.frame[leader], merged.sim.frame[follower])
    np.testing.assert_array_equal(
        merged.runtime._alive_post[leader], merged.runtime._alive_post[follower]
    )
    assert merged.sim._rngs[leader]._rng.getstate() == merged.sim._rngs[follower]._rng.getstate()

    merged.sim.frame[leader] += 1
    assert not np.array_equal(merged.sim.frame[leader], merged.sim.frame[follower])
    merged.sim._rngs[leader]._rng.random()
    assert merged.sim._rngs[leader]._rng.getstate() != merged.sim._rngs[follower]._rng.getstate()


def test_late_fork_copies_advanced_frame_and_vector61_carry() -> None:
    """A follower forks from the exact post-step leader snapshot."""
    merged = merge_worlds([_world("frozen")], rollouts=2, horizon=8)
    active = np.zeros(merged.sim.E, dtype=bool)
    active[0] = True
    actions = np.ones((merged.sim.E, merged.sim.S), dtype=np.int64)
    merged.sim.step(actions, active_env_mask=active)
    merged.runtime.observe_step(merged.sim, rows_mask=active)
    _copy_environment(merged, 0, 1)
    assert int(merged.sim.frame[0]) == int(merged.sim.frame[1])
    np.testing.assert_array_equal(merged.runtime._alive_post[0], merged.runtime._alive_post[1])
    np.testing.assert_array_equal(
        merged.runtime.memory.prev_dist[0], merged.runtime.memory.prev_dist[1]
    )


def test_lazy_and_eager_match_under_predrawn_nonzero_epsilon() -> None:
    """Forking preserves the eager result under the documented T33 stream."""
    config = phase0.SearchConfig(rollouts=3, horizon=8, epsilon=0.5, length_gate=0)
    lazy_actions, lazy = BatchedTeacher(config, common_random=False, lazy_fork=True).choose(
        [_world()]
    )
    eager_actions, eager = BatchedTeacher(config, common_random=False, lazy_fork=False).choose(
        [_world()]
    )
    np.testing.assert_array_equal(lazy_actions, eager_actions)
    assert lazy[0]["scores"] == eager[0]["scores"]


def test_fixed_inference_uses_separate_source_and_resident_shapes() -> None:
    """The source Watch Q cannot bind the resident CUDA-style adapter to row one."""
    from research.t33_batched_teacher.inference import FixedInference

    world = _world()
    config = phase0.SearchConfig(rollouts=2, horizon=3, epsilon=0.0, length_gate=0)
    resident = FixedInference(world.hero.net, device="cpu")
    search = BatchedTeacher(config, inference=resident, common_random=False)
    search.choose([world])
    assert resident.rows == 12
    assert search._initial_inference is not None
    assert search._initial_inference.rows == 1


def test_predrawn_streams_are_repeatable_and_rollout_independent() -> None:
    """Each follower owns a stable stream even while it remains a lazy leaf."""
    layout = BranchLayout(
        source=np.array([0, 0, 0, 1, 1, 1]),
        action=np.array([0, 0, 1, 0, 0, 1]),
        rollout=np.array([0, 1, 0, 0, 1, 0]),
        active=np.ones(6, dtype=bool),
    )
    worlds = [type("World", (), {"seed": 11})(), type("World", (), {"seed": 12})()]
    first = _predrawn_epsilon(worlds, layout, [100, 101], 16, common_random=True)
    second = _predrawn_epsilon(worlds, layout, [100, 101], 16, common_random=True)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0][0], first[0][1])


def _epsilon_oracle(worlds, layout, decisions, horizon, common_random):
    """Independent direct construction of the documented T33 draw tensor."""
    u = np.empty((len(layout.source), horizon))
    v = np.empty_like(u)
    draw_horizon = 128 if horizon <= 128 else horizon
    for source, world in enumerate(worlds):
        for action in range(6):
            rows = np.flatnonzero((layout.source == source) & (layout.action == action))
            if not len(rows):
                continue
            stream_action = 0 if common_random else action
            rng = np.random.default_rng(
                np.random.SeedSequence([world.seed, decisions[source], stream_action, 20261019])
            )
            draws = rng.random((len(rows), draw_horizon, 2))
            u[rows], v[rows] = draws[:, :horizon, 0], draws[:, :horizon, 1]
    return u, v


def test_epsilon_oracle_crn_ordering_and_rollout_prefix() -> None:
    """The schedule is source-keyed, CRN-aware, and invariant for the first N."""
    worlds = [type("World", (), {"seed": 71})(), type("World", (), {"seed": 19})()]

    def layout(rollouts):
        return BranchLayout(
            source=np.repeat(np.arange(2), 6 * rollouts),
            action=np.tile(np.repeat(np.arange(6), rollouts), 2),
            rollout=np.tile(np.arange(rollouts), 12),
            active=np.ones(12 * rollouts, dtype=bool),
        )

    small, large = layout(2), layout(4)
    actual = _predrawn_epsilon(worlds, small, [90, 44], 9, True)
    expected = _epsilon_oracle(worlds, small, [90, 44], 9, True)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    prefix = _predrawn_epsilon(worlds, large, [90, 44], 9, True)
    for source in range(2):
        for action in range(6):
            a = np.flatnonzero((small.source == source) & (small.action == action))
            b = np.flatnonzero((large.source == source) & (large.action == action))[:2]
            np.testing.assert_array_equal(actual[0][a], prefix[0][b])
    # CRN shares an action stream; non-CRN binds a distinct action component.
    assert np.array_equal(actual[0][0], actual[0][2])
    non_crn = _predrawn_epsilon(worlds, small, [90, 44], 9, False)
    assert not np.array_equal(non_crn[0][0], non_crn[0][2])
    reordered = _predrawn_epsilon(list(reversed(worlds)), small, [44, 90], 9, True)
    for action in range(6):
        source_zero = np.flatnonzero((small.source == 0) & (small.action == action))
        reordered_one = np.flatnonzero((small.source == 1) & (small.action == action))
        np.testing.assert_array_equal(actual[0][source_zero], reordered[0][reordered_one])


def test_epsilon_horizons_are_prefixes_through_registered_h128() -> None:
    """H32/H64 reads are exact prefixes of the H128 stream for all rollouts."""
    worlds = [type("World", (), {"seed": 701})(), type("World", (), {"seed": 702})()]
    layout = BranchLayout(
        source=np.repeat(np.arange(2), 6 * 3),
        action=np.tile(np.repeat(np.arange(6), 3), 2),
        rollout=np.tile(np.arange(3), 12),
        active=np.ones(36, dtype=bool),
    )
    for common_random in (False, True):
        h32 = _predrawn_epsilon(worlds, layout, [51, 52], 32, common_random)
        h64 = _predrawn_epsilon(worlds, layout, [51, 52], 64, common_random)
        h128 = _predrawn_epsilon(worlds, layout, [51, 52], 128, common_random)
        for index in range(2):
            np.testing.assert_array_equal(h32[index], h128[index][:, :32])
            np.testing.assert_array_equal(h64[index], h128[index][:, :64])


def test_merge_pads_food_mirrors_and_rejects_overlapping_source_body() -> None:
    """Different food capacities gather safely while invalid source grids fail closed."""
    first, second = _world(), _world("frozen")
    second.sim._ensure_food_array_capacity(second.sim._food_array.shape[1] + 17)
    merged = merge_worlds([first, second], rollouts=1, horizon=4)
    assert merged.sim._food_array.shape[1] == second.sim._food_array.shape[1]

    invalid = _world()
    head = int(invalid.sim.head_ptr[0, 0])
    invalid.sim.bodies[0, 0, head] = invalid.sim.bodies[0, 1, invalid.sim.head_ptr[0, 1]]
    with np.testing.assert_raises_regex(ValueError, "overlapping|disagree|drifted"):
        merge_worlds([invalid], rollouts=1, horizon=4)


def test_lazy_epsilon_zero_matches_one_rollout_scores() -> None:
    """Without forks, follower totals are exactly their leader totals."""
    config = phase0.SearchConfig(rollouts=2, horizon=8, epsilon=0.0, length_gate=0)
    lazy = BatchedTeacher(config, common_random=False)
    _, lazy_diag = lazy.choose([_world()])
    eager_config = phase0.SearchConfig(rollouts=1, horizon=8, epsilon=0.0, length_gate=0)
    _, eager_diag = BatchedTeacher(eager_config, common_random=False).choose([_world()])
    assert lazy_diag[0]["scores"] == eager_diag[0]["scores"]


def test_dead_source_does_not_search_empty_legal_domain() -> None:
    world = _world()
    world.sim.alive[0, 0] = False
    world.sim._rebuild_env_grids(0)
    world.sim._refresh_action_masks()
    actions, diagnostics = BatchedTeacher(
        phase0.SearchConfig(rollouts=2, horizon=3, length_gate=0)
    ).choose([world])
    assert actions.shape == (1,)
    assert diagnostics[0]["searched"] is False
    assert diagnostics[0]["scores"] == {}
