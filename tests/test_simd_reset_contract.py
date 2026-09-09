"""Contract tests for BatchSim reset, masks, and active-world stepping."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.parity import _install_v2_config


def _cfg(**overrides: object) -> BatchSimConfig:
    base = dict(
        num_envs=2,
        num_snakes=2,
        game_width=200,
        game_height=160,
        segment_size=10,
        wall_thickness=10,
        initial_food=8,
        max_food=8,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        max_capacity=64,
    )
    base.update(overrides)
    return BatchSimConfig(**base)


def _live_snapshot(game) -> tuple:
    """Read actual GameState reset output without using the parity helper."""
    return (
        tuple(
            tuple((pos[0] // 10, pos[1] // 10) for pos in snake.segments) for snake in game.snakes
        ),
        tuple((pos[0] // 10, pos[1] // 10) for pos in game.food),
    )


def _batch_snapshot(sim: BatchSim, env: int) -> tuple:
    return (
        tuple(tuple(sim.get_bodies(env, snake)) for snake in range(sim.S)),
        tuple(sim.get_food(env)),
    )


def _env_reset_snapshot(sim: BatchSim, env: int) -> tuple:
    """Capture every env-scoped field reset_envs must leave untouched elsewhere."""
    return (
        sim.bodies[env].tobytes(),
        sim.head_ptr[env].tobytes(),
        sim.seg_count[env].tobytes(),
        sim.length[env].tobytes(),
        sim.alive[env].tobytes(),
        sim.direction[env].tobytes(),
        sim.boost_frames[env].tobytes(),
        sim.frames_since_food[env].tobytes(),
        sim.respawn_timer[env].tobytes(),
        sim._reward_prev_length[env].tobytes(),
        sim._boosted_this_step[env].tobytes(),
        sim._trav[env].tobytes(),
        sim._trav_valid[env].tobytes(),
        tuple(sim.food_cells[env]),
        frozenset(sim.corpse_cells[env]),
        frozenset(sim.food_set[env]),
        int(sim.frame[env]),
        sim._last_reward[env].tobytes(),
        sim._last_mask[env].tobytes(),
        sim._last_legal_mask[env].tobytes(),
        sim._last_resolved_mask[env].tobytes(),
        sim._last_done[env].tobytes(),
        sim._last_transition_valid[env].tobytes(),
        sim._last_food_ate[env].tobytes(),
        sim._last_death_cause[env].tobytes(),
        sim._last_kills[env].tobytes(),
        copy.deepcopy(sim._last_kill_victim_len[env].tolist()),
        sim._rngs[env]._rng.getstate(),
        sim._seeds[env],
    )


@pytest.mark.parametrize("seed", [41, 314159])
def test_repeated_resets_match_actual_game_state_rng_order(seed: int) -> None:
    """Actual GameState remains aligned through actions and later soft resets."""
    from src.game.game_state import GameState

    cfg = _cfg(num_envs=1)
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(seed)
        game = GameState(headless=True, num_snakes=cfg.num_snakes)
        sim = BatchSim(cfg, seeds=[seed], train_mode=True)
        # Drive the real GameState with a deterministic physical action, without
        # invoking policy exploration or re-implementing GameState.update().
        for snake in game.snakes:
            snake.update = lambda _others, _food, snake=snake, **_kwargs: snake.move()
        for _ in range(3):
            assert _batch_snapshot(sim, 0) == _live_snapshot(game)
            game.update(train_mode=True, learn=False, allow_respawn=False)
            sim.step(np.ones((1, cfg.num_snakes), dtype=np.int64))
            assert _batch_snapshot(sim, 0) == _live_snapshot(game)
            game.reset()
            sim.reset()
    finally:
        initialize_config(saved)


def test_inactive_environment_is_frozen_and_next_reset_keeps_its_rng_stream() -> None:
    """Advancing env 0 cannot mutate env 1 or consume env 1's future spawn draws."""
    cfg = _cfg()
    sim = BatchSim(cfg, seeds=[41, 314159], train_mode=True)
    control = BatchSim(cfg, seeds=[41, 314159], train_mode=True)
    before = _batch_snapshot(sim, 1)
    before_frame = sim.frame[1].copy()
    before_rng = sim._rngs[1]._rng.getstate()

    actions = np.array([[1, 1], [4, 4]], dtype=np.int64)
    sim.step(actions, active_env_mask=np.array([True, False], dtype=bool))
    control.step(actions, active_env_mask=np.array([False, False], dtype=bool))

    assert _batch_snapshot(sim, 1) == before
    assert sim.frame[1] == before_frame
    assert sim._rngs[1]._rng.getstate() == before_rng
    assert not sim.get_transition_valid()[1].any()
    assert sim.get_transition_valid()[0].all()
    assert not sim.get_step_events()["transition_valid"][1].any()

    sim.reset()
    control.reset()
    assert _batch_snapshot(sim, 1) == _batch_snapshot(control, 1)


def test_masks_keep_legacy_advisory_meaning_and_resolve_against_domain_legality() -> None:
    cfg = _cfg(num_envs=1, num_snakes=1, initial_food=0, max_food=0)
    sim = BatchSim(cfg, seeds=[7], train_mode=True)

    # A living length-one snake may issue all normal actions but no boost.
    legal = sim.get_legal_action_mask()[0, 0]
    assert legal.tolist() == [True, True, True, False, False, False]
    assert np.array_equal(sim.get_action_mask(), sim.get_advisory_action_mask())
    assert not sim.get_resolved_action_mask()[0, 0, 3:].any()

    # A fully boxed-in normal-action row has no advisory choices. C0's
    # row-local fallback returns its legal actions, still excluding boost.
    sim.seg_count[0, 0] = 4
    sim.length[0, 0] = 4
    sim.head_ptr[0, 0] = 3
    # Ring index 3 is the head; after a candidate move the old offset-two
    # segment at (0, 1) remains in the self-collision target.
    for ring, cell in enumerate(((1, 1), (0, 1), (1, 0), (0, 0))):
        sim.bodies[0, 0, ring] = cell
    sim.direction[0, 0] = 3
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()
    assert not sim.get_advisory_action_mask()[0, 0].any()
    assert sim.get_resolved_action_mask()[0, 0].tolist() == [True, True, True, False, False, False]

    # At the left wall straight-left is advisory-fatal, while a turn can remain
    # safe. Resolution preserves that advisory veto and never enables boost.
    sim.seg_count[0, 0] = 1
    sim.length[0, 0] = 1
    sim.head_ptr[0, 0] = 0
    sim.bodies[0, 0, 0] = (0, 3)
    sim.direction[0, 0] = 3
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()
    advisory = sim.get_advisory_action_mask()[0, 0]
    assert not advisory[1]
    assert sim.get_resolved_action_mask()[0, 0, 1] == advisory[1]
    assert not sim.get_resolved_action_mask()[0, 0, 3:].any()

    sim.alive[0, 0] = False
    sim._refresh_action_masks()
    assert not sim.get_legal_action_mask()[0, 0].any()
    assert not sim.get_resolved_action_mask()[0, 0].any()


def test_active_env_mask_rejects_non_boolean_or_wrong_shape() -> None:
    sim = BatchSim(_cfg(), seeds=[1, 2], train_mode=True)
    actions = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="boolean dtype"):
        sim.step(actions, active_env_mask=np.array([1, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        sim.step(actions, active_env_mask=np.array([[True, False]], dtype=bool))
    with pytest.raises(ValueError, match="actions must have shape"):
        sim.step(np.zeros((2, 1), dtype=np.int64))


def test_inactive_full_capacity_row_is_not_moved_or_consulted_for_overflow() -> None:
    """An inactive full ring cannot make an unrelated active environment fail."""
    sim = BatchSim(
        _cfg(num_snakes=1, initial_food=0, max_food=0, max_capacity=2),
        seeds=[41, 314159],
        train_mode=True,
    )
    # Construct an otherwise valid full-capacity inactive body. A previous
    # snapshot-and-restore implementation still tried to push this head first.
    sim.seg_count[1, 0] = 2
    sim.length[1, 0] = 2
    sim.head_ptr[1, 0] = 1
    sim.bodies[1, 0, 1] = (8, 4)
    sim.bodies[1, 0, 0] = (7, 4)
    frozen = (
        sim.bodies[1].copy(),
        sim.head_ptr[1].copy(),
        sim.seg_count[1].copy(),
        sim.length[1].copy(),
        sim.frame[1].copy(),
        copy.deepcopy(sim._rngs[1]._rng.getstate()),
    )

    sim.step(np.array([[1], [1]], dtype=np.int64), active_env_mask=np.array([True, False]))

    assert sim.frame[0] == 1
    assert sim.frame[1] == frozen[4]
    assert np.array_equal(sim.bodies[1], frozen[0])
    assert np.array_equal(sim.head_ptr[1], frozen[1])
    assert np.array_equal(sim.seg_count[1], frozen[2])
    assert np.array_equal(sim.length[1], frozen[3])
    assert sim._rngs[1]._rng.getstate() == frozen[5]
    assert sim.get_transition_valid().tolist() == [[True], [False]]


def test_transition_valid_marks_death_frame_but_not_the_following_dead_step() -> None:
    """A terminal transition is trainable once; its dead successor is not."""
    sim = BatchSim(_cfg(num_envs=1, num_snakes=1, initial_food=0, max_food=0), seeds=[9])
    sim.bodies[0, 0, 0] = (0, 3)
    sim.direction[0, 0] = 3
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()

    sim.step(np.array([[1]], dtype=np.int64))
    assert sim.get_done().tolist() == [[True]]
    assert sim.get_transition_valid().tolist() == [[True]]

    sim.step(np.array([[1]], dtype=np.int64))
    assert sim.get_transition_valid().tolist() == [[False]]


def test_inactive_world_preserves_food_timers_masks_and_population_floor() -> None:
    """Inactive rows retain every public episode field, not only their body ring."""
    sim = BatchSim(_cfg(num_snakes=3), seeds=[4, 5], train_mode=True)
    sim.alive[1, 2] = False
    sim.respawn_timer[1, 2] = 7
    sim.boost_frames[1, 0] = 2
    sim.frames_since_food[1, 1] = 11
    sim._refresh_action_masks()
    frozen = (
        tuple(sim.get_food(1)),
        tuple(sim.get_corpse_food(1)),
        sim.get_alive()[1].copy(),
        sim.get_boost_frames()[1].copy(),
        sim.get_frames_since_food()[1].copy(),
        sim.get_legal_action_mask()[1].copy(),
        sim.get_advisory_action_mask()[1].copy(),
        sim.get_resolved_action_mask()[1].copy(),
        bool(sim.population_floor_reached()[1]),
    )

    sim.step(np.ones((2, 3), dtype=np.int64), active_env_mask=np.array([True, False], dtype=bool))

    assert tuple(sim.get_food(1)) == frozen[0]
    assert tuple(sim.get_corpse_food(1)) == frozen[1]
    assert np.array_equal(sim.get_alive()[1], frozen[2])
    assert np.array_equal(sim.get_boost_frames()[1], frozen[3])
    assert np.array_equal(sim.get_frames_since_food()[1], frozen[4])
    assert np.array_equal(sim.get_legal_action_mask()[1], frozen[5])
    assert np.array_equal(sim.get_advisory_action_mask()[1], frozen[6])
    assert np.array_equal(sim.get_resolved_action_mask()[1], frozen[7])
    assert bool(sim.population_floor_reached()[1]) == frozen[8]


def test_reset_envs_resets_only_selected_lanes_and_preserves_continuing_bytes() -> None:
    """A masked reset cannot mutate another lane's world, outputs, or RNG stream."""
    sim = BatchSim(_cfg(num_envs=3, num_snakes=3), seeds=[4, 5, 6], train_mode=True)
    sim.alive[0, 2] = False
    sim.alive[0, 1] = False  # population floor reached in the selected lane
    sim.frame[0] = 5000  # simultaneous configured frame-cap witness
    sim._last_reward[0] = 7.0
    sim._last_done[0] = True
    sim._last_transition_valid[0] = True
    sim._last_kill_victim_len[0, 0] = [3]
    sim._refresh_action_masks()
    before_one = _env_reset_snapshot(sim, 1)
    before_two = _env_reset_snapshot(sim, 2)

    sim.reset_envs(np.array([True, False, False], dtype=bool))

    assert _env_reset_snapshot(sim, 1) == before_one
    assert _env_reset_snapshot(sim, 2) == before_two
    assert sim.frame[0] == 0
    assert sim.alive[0].all()
    assert not sim._last_done[0].any()
    assert not sim._last_transition_valid[0].any()
    assert sim._last_kill_victim_len[0, 0] == []
    assert all(
        sim._last_kill_victim_len[0, s] is not sim._last_kill_victim_len[0, 0] for s in range(1, 3)
    )
    assert not bool(sim.population_floor_reached()[0])


def test_reset_envs_noncontiguous_seeded_reset_replays_selected_lanes() -> None:
    """Selected lanes consume only their supplied new EnvRng streams in mask order."""
    cfg = _cfg(num_envs=3)
    left = BatchSim(cfg, seeds=[1, 2, 3], train_mode=True)
    right = BatchSim(cfg, seeds=[91, 92, 93], train_mode=True)
    actions = np.ones((3, cfg.num_snakes), dtype=np.int64)
    left.step(actions)
    right.step(actions)
    mask = np.array([True, False, True], dtype=bool)
    seeds = [np.uint64((1 << 64) - 1), 17]

    left.reset_envs(mask, seeds=seeds)
    right.reset_envs(mask, seeds=seeds)

    for env in (0, 2):
        assert _env_reset_snapshot(left, env) == _env_reset_snapshot(right, env)
    assert left._seeds == [int(seeds[0]), 2, int(seeds[1])]
    assert right._seeds == [int(seeds[0]), 92, int(seeds[1])]


def test_reset_envs_all_lanes_matches_legacy_soft_reset() -> None:
    """All-lane masked reset keeps the established no-argument soft-reset trace."""
    cfg = _cfg(num_envs=3)
    legacy = BatchSim(cfg, seeds=[7, 8, 9], train_mode=True)
    masked = BatchSim(cfg, seeds=[7, 8, 9], train_mode=True)
    actions = np.ones((3, cfg.num_snakes), dtype=np.int64)
    legacy.step(actions)
    masked.step(actions)

    legacy.reset()
    masked.reset_envs(np.ones(3, dtype=bool))

    assert tuple(_env_reset_snapshot(legacy, env) for env in range(3)) == tuple(
        _env_reset_snapshot(masked, env) for env in range(3)
    )


@pytest.mark.parametrize(
    "mask,seeds",
    [
        ([True, False], None),
        (np.array([[True, False]], dtype=bool), None),
        (np.array([1, 0], dtype=np.int64), None),
        (np.array([True, False], dtype=bool), [1, 2]),
        (np.array([True, False], dtype=bool), [-1]),
        (np.array([True, False], dtype=bool), [1 << 64]),
        (np.array([True, False], dtype=bool), [True]),
        (np.array([True, False], dtype=bool), [1.0]),
        (np.array([False, False], dtype=bool), [1]),
    ],
)
def test_reset_envs_rejects_invalid_inputs_before_mutating(mask: object, seeds: object) -> None:
    sim = BatchSim(_cfg(), seeds=[11, 12], train_mode=True)
    before = tuple(_env_reset_snapshot(sim, env) for env in range(sim.E))

    with pytest.raises(ValueError):
        sim.reset_envs(mask, seeds=seeds)  # type: ignore[arg-type]

    assert tuple(_env_reset_snapshot(sim, env) for env in range(sim.E)) == before


def test_reset_envs_all_false_empty_seed_sequence_is_a_noop() -> None:
    sim = BatchSim(_cfg(), seeds=[11, 12], train_mode=True)
    before = tuple(_env_reset_snapshot(sim, env) for env in range(sim.E))

    sim.reset_envs(np.array([False, False], dtype=bool), seeds=[])

    assert tuple(_env_reset_snapshot(sim, env) for env in range(sim.E)) == before
