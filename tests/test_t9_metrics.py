"""Synthetic contracts for the read-only T9 divergence measurements."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research.t9_divergence_20261017.metrics import (
    _arrival,
    _arrival_reference,
    _race_counts,
    measure,
)


def _sim(width, height, snakes, *, length=None, direction=1, food=()):
    """Build the small ring-buffer surface used by T9's metrics helper."""
    cap = max(1, max((len(cells) for cells in snakes), default=1))
    bodies = np.full((1, len(snakes), cap, 2), -99, dtype=np.int64)
    for slot, cells in enumerate(snakes):
        for offset, cell in enumerate(cells):
            bodies[0, slot, (-offset) % cap] = cell
    logical = [len(cells) for cells in snakes] if length is None else length
    return SimpleNamespace(
        grid_w=width,
        grid_h=height,
        cap=cap,
        bodies=bodies,
        head_ptr=np.zeros((1, len(snakes)), dtype=np.int64),
        seg_count=np.array([[len(cells) for cells in snakes]], dtype=np.int64),
        length=np.array([logical], dtype=np.int64),
        alive=np.ones((1, len(snakes)), dtype=bool),
        direction=np.full((1, len(snakes)), direction, dtype=np.int64),
        frame=np.array([0], dtype=np.int64),
        food_cells=[list(food)],
    )


def test_static_geometry_opens_only_ttl_one_and_reports_unreachable_tail() -> None:
    # The tail itself is open, but the TTL-two middle segment is a static wall.
    sim = _sim(5, 1, [[(0, 0), (1, 0), (2, 0)]])

    got = measure(sim, 1)

    assert got["head_region"] == 1
    assert got["largest_region"] == 3
    assert got["head_in_largest"] is False
    assert got["tail_bfs"] is None
    assert got["tail_reachable"] is False
    assert got["tail_manhattan"] == 2
    # The optimistic race waits at the TTL-two segment and eventually reaches
    # every square; static topology must not silently use that future release.
    assert got["race_16"] == 5


def test_components_are_uncapped_and_ties_count_as_largest() -> None:
    sim = _sim(30, 30, [[(15, 15)]])

    got = measure(sim, 1)

    assert got["head_region"] == 900
    assert got["largest_region"] == 900
    assert got["head_in_largest"] is True


def test_arrival_race_uses_all_heads_and_excludes_opponent_ties() -> None:
    # Both opponent sources are blocked body cells except for source seeding.
    # On this line the hero loses/ties the central frontier, hence only 0 and 1
    # are strictly hero-first.
    sim = _sim(7, 1, [[(0, 0)], [(4, 0)], [(6, 0)]])

    got = measure(sim, 1)

    assert got["race_16"] == 2
    assert got["race_256"] == 2
    assert got["opponent_cells_r15"] == 2


def test_perimeter_fraction_radius_and_food_direction_are_deterministic() -> None:
    sim = _sim(5, 1, [[(0, 0)], [(4, 0)]], food=[(1, 0), (3, 0)])

    got = measure(sim, 1)

    assert got["own_cells_r15"] == 1
    assert got["own_cells_r30"] == 1
    assert got["opponent_cells_r30"] == 1
    assert got["boundary_own"] == 0.0
    assert got["boundary_opponent"] == 0.1
    assert got["boundary_wall"] == 0.9
    assert got["boundary_max_chebyshev"] == 4
    assert got["boundary_outside15_fraction"] == 0.0
    assert got["toward_food"] is True
    assert got["food_in_smaller_component"] is False


def test_food_in_smaller_component_and_absent_food_are_explicit() -> None:
    # The opponent wall partitions a 5x3 board.  The nearest food is in the
    # smaller right component despite the action pointing toward it.
    wall = [(3, 0), (3, 1), (3, 2)]
    sim = _sim(6, 3, [[(0, 1)], wall], food=[(5, 1)])
    no_food = _sim(3, 3, [[(1, 1)]])

    got = measure(sim, 1)

    assert got["food_in_smaller_component"] is True
    assert got["toward_food"] is True
    assert measure(no_food, 1)["food_in_smaller_component"] is None
    assert measure(no_food, 1)["toward_food"] is False


def test_measure_is_read_only() -> None:
    sim = _sim(7, 3, [[(1, 1), (0, 1)], [(5, 1)]], food=[(3, 1)])
    before = (sim.bodies.copy(), sim.length.copy(), sim.alive.copy(), list(sim.food_cells[0]))

    measure(sim, 1)

    np.testing.assert_array_equal(sim.bodies, before[0])
    np.testing.assert_array_equal(sim.length, before[1])
    np.testing.assert_array_equal(sim.alive, before[2])
    assert sim.food_cells[0] == before[3]


def test_compiled_arrival_matches_heap_reference_on_random_ttl_worlds() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        blocked = rng.random((7, 11)) < 0.2
        release = rng.integers(0, 24, size=(7, 11), dtype=np.int32)
        starts = [(0, 0), (10, 6), (5, 3)]
        np.testing.assert_array_equal(
            _arrival(starts, blocked, release),
            _arrival_reference(starts, blocked, release),
        )


def test_race_horizons_include_cells_exactly_on_the_boundary() -> None:
    sim = _sim(40, 1, [[(0, 0)]])

    got = measure(sim, 1)

    assert got["race_16"] == 17
    assert got["race_32"] == 33
    assert got["race_64"] == 40


def test_race_releases_other_owner_cells_at_their_ttl() -> None:
    owners = np.array([[0, -1, 1, 1, 1, -1]], dtype=np.int32)
    ttl = np.array([[1, 0, 3, 2, 1, 0]], dtype=np.int32)

    got = _race_counts((0, 0), [], owners, ttl)

    assert got["race_16"] == 6
