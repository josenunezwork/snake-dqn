"""Regression coverage for exact source labels on BatchSim food events."""

from __future__ import annotations

import numpy as np

from src.simd_env.batch_sim import BatchSim, BatchSimConfig


def _config(*, num_envs: int = 1, width: int = 120) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=num_envs,
        num_snakes=1,
        game_width=width,
        game_height=100,
        segment_size=10,
        wall_thickness=10,
        initial_food=0,
        max_food=0,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        max_capacity=32,
    )


def _install_body(sim: BatchSim, env: int, cells: list[tuple[int, int]]) -> None:
    """Install one live, right-facing body in production ring-buffer order."""
    sim.bodies[env, 0] = 0
    sim.head_ptr[env, 0] = 0
    for offset, cell in enumerate(cells):
        sim.bodies[env, 0, (-offset) % sim.cap] = cell
    sim.seg_count[env, 0] = len(cells)
    sim.length[env, 0] = len(cells)
    sim.alive[env, 0] = True
    sim.direction[env, 0] = 1
    sim.boost_frames[env, 0] = 0
    sim._reward_prev_length[env, 0] = len(cells)


def _set_food(
    sim: BatchSim, env: int, cells: list[tuple[int, int]], corpse: set[tuple[int, int]]
) -> None:
    sim.food_cells[env] = list(cells)
    sim.food_set[env] = set(cells)
    sim.corpse_cells[env] = set(corpse)


def _refresh(sim: BatchSim) -> None:
    sim._rebuild_traversed_from_heads()
    sim._refresh_action_masks()


def _assert_source_partition(events: dict[str, np.ndarray]) -> None:
    food = events["food_ate"]
    ambient = events["ambient_food_ate"]
    corpse = events["corpse_food_ate"]
    valid = events["transition_valid"]
    assert np.array_equal(food[valid], np.logical_xor(ambient[valid], corpse[valid]))
    assert not np.any(ambient & corpse)


def test_ambient_and_corpse_pickups_are_distinguished_before_removal() -> None:
    sim = BatchSim(_config(), seeds=[3], train_mode=True)
    _install_body(sim, 0, [(4, 4)])
    _set_food(sim, 0, [(5, 4), (6, 4)], {(6, 4)})
    _refresh(sim)

    sim.step(np.array([[1]], dtype=np.int64))
    first = sim.get_step_events()
    assert bool(first["food_ate"][0, 0]) is True
    assert bool(first["ambient_food_ate"][0, 0]) is True
    assert bool(first["corpse_food_ate"][0, 0]) is False
    _assert_source_partition(first)

    # The earlier corpse tag has been removed with its pellet, so a classifier
    # that looks after mutation would incorrectly call this ambient.
    sim.step(np.array([[1]], dtype=np.int64))
    second = sim.get_step_events()
    assert bool(second["food_ate"][0, 0]) is True
    assert bool(second["ambient_food_ate"][0, 0]) is False
    assert bool(second["corpse_food_ate"][0, 0]) is True
    _assert_source_partition(second)

    second["ambient_food_ate"][:] = True
    second["corpse_food_ate"][:] = False
    unchanged = sim.get_step_events()
    assert bool(unchanged["ambient_food_ate"][0, 0]) is False
    assert bool(unchanged["corpse_food_ate"][0, 0]) is True


def test_boost_uses_first_traversed_food_and_tags_its_own_trail_as_corpse() -> None:
    sim = BatchSim(_config(), seeds=[5], train_mode=True)
    _install_body(sim, 0, [(4, 4), (3, 4), (2, 4), (1, 4), (0, 4)])
    sim.boost_frames[0, 0] = 2  # this boost frame burns a tail pellet in v2
    # The boost visits (5, 4), then (6, 4). Only the first occupied cell may
    # count, even though the second is ambient.
    _set_food(sim, 0, [(5, 4), (6, 4)], {(5, 4)})
    _refresh(sim)

    sim.step(np.array([[4]], dtype=np.int64))
    events = sim.get_step_events()
    assert bool(events["boosted"][0, 0]) is True
    assert bool(events["food_ate"][0, 0]) is True
    assert bool(events["corpse_food_ate"][0, 0]) is True
    assert bool(events["ambient_food_ate"][0, 0]) is False
    _assert_source_partition(events)

    # The simulator creates a v2 boost trail after movement. Consume that
    # actual trail in a controlled following transition: it remains a corpse
    # source even though this snake generated it.
    trail = set(sim.get_corpse_food(0))
    assert len(trail) == 1
    trail_cell = trail.pop()
    _install_body(sim, 0, [(trail_cell[0] - 1, trail_cell[1])])
    _set_food(sim, 0, [trail_cell], {trail_cell})
    _refresh(sim)
    sim.step(np.array([[1]], dtype=np.int64))
    trail_events = sim.get_step_events()
    assert bool(trail_events["food_ate"][0, 0]) is True
    assert bool(trail_events["corpse_food_ate"][0, 0]) is True
    assert bool(trail_events["ambient_food_ate"][0, 0]) is False
    _assert_source_partition(trail_events)


def test_boost_trail_duplicate_does_not_reclassify_an_existing_ambient_pellet() -> None:
    sim = BatchSim(_config(), seeds=[7], train_mode=True)
    _install_body(sim, 0, [(4, 4), (3, 4), (2, 4), (1, 4), (0, 4)])
    sim.boost_frames[0, 0] = 2
    # A two-cell boost from this body burns the tail at (2, 4). The pellet is
    # already ambient, so _add_food's duplicate rejection must leave its source
    # identity untouched rather than turning it into a corpse/trail pellet.
    ambient_trail_cell = (2, 4)
    _set_food(sim, 0, [ambient_trail_cell], set())
    _refresh(sim)

    sim.step(np.array([[4]], dtype=np.int64))
    assert ambient_trail_cell in sim.get_food(0)
    assert ambient_trail_cell not in sim.get_corpse_food(0)

    _install_body(sim, 0, [(ambient_trail_cell[0] - 1, ambient_trail_cell[1])])
    _refresh(sim)
    sim.step(np.array([[1]], dtype=np.int64))
    events = sim.get_step_events()
    assert bool(events["food_ate"][0, 0]) is True
    assert bool(events["ambient_food_ate"][0, 0]) is True
    assert bool(events["corpse_food_ate"][0, 0]) is False
    _assert_source_partition(events)


def test_pickup_remains_labeled_when_the_same_transition_dies_at_a_wall() -> None:
    sim = BatchSim(_config(width=80), seeds=[11], train_mode=True, allow_respawn=False)
    _install_body(sim, 0, [(6, 4), (5, 4), (4, 4), (3, 4), (2, 4)])
    # The first boost substep reaches the valid last grid cell and eats there;
    # its second substep reaches cell 8, outside this width-80 world. Food
    # resolution deliberately happens before collision resolution.
    _set_food(sim, 0, [(7, 4)], set())
    _refresh(sim)

    sim.step(np.array([[4]], dtype=np.int64))
    events = sim.get_step_events()
    assert bool(events["transition_valid"][0, 0]) is True
    assert bool(events["food_ate"][0, 0]) is True
    assert bool(events["ambient_food_ate"][0, 0]) is True
    assert bool(events["corpse_food_ate"][0, 0]) is False
    assert bool(events["done"][0, 0]) is True
    _assert_source_partition(events)


def test_inactive_and_partially_reset_lanes_keep_or_clear_source_snapshots() -> None:
    sim = BatchSim(_config(num_envs=2), seeds=[17, 19], train_mode=True)
    for env in range(2):
        _install_body(sim, env, [(4, 4)])
        _set_food(sim, env, [(5, 4)], set())
    _refresh(sim)
    sim.step(np.ones((2, 1), dtype=np.int64))
    previous = sim.get_step_events()
    assert previous["ambient_food_ate"].all()

    # A partial reset clears source snapshots only for its new world.
    sim.reset_envs(np.array([True, False], dtype=bool))
    reset_events = sim.get_step_events()
    assert not reset_events["food_ate"][0].any()
    assert not reset_events["ambient_food_ate"][0].any()
    assert not reset_events["corpse_food_ate"][0].any()
    assert bool(reset_events["ambient_food_ate"][1, 0]) is True

    # A frozen lane retains its preceding event snapshot and is marked invalid
    # for this transition, whereas the active lane gets a fresh empty snapshot.
    sim.step(np.ones((2, 1), dtype=np.int64), active_env_mask=np.array([True, False], dtype=bool))
    events = sim.get_step_events()
    assert not events["transition_valid"][1].any()
    assert bool(events["ambient_food_ate"][1, 0]) is True
    assert not events["food_ate"][0].any()
    assert not events["ambient_food_ate"][0].any()
    assert not events["corpse_food_ate"][0].any()
