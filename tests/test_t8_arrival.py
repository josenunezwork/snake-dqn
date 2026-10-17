"""Hand-grid contracts for T8's diagnostic arrival-race features."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.simd_env.t8_features import compute_features, race_features


def test_race_counts_strict_voronoi_ties_and_nearest_contested_frontier() -> None:
    # On a 5x1 lane, hero at x=0 and opponent at x=4 tie at x=2.  The strict
    # territory is x=0,1; contest is x=2,3,4 and its nearest cell is two frames
    # away from the hero.
    area, fraction, margin = race_features((0, 0), [(4, 0)], 5, 1, set(), {})
    assert area == 2
    assert fraction == 3 / 5
    assert margin == 2


def test_race_obeys_tail_release_for_both_arrival_floods() -> None:
    # The middle cell is unavailable until t=3.  Neither flood can cross it
    # early, so the two isolated sides remain strictly owned by their seed.
    area, fraction, margin = race_features((0, 0), [(4, 0)], 5, 1, set(), {(2, 0): 3})
    assert area == 2
    assert fraction == 0.0
    assert margin == 6


def test_race_admits_a_blocked_head_source_but_not_its_body() -> None:
    # The opponent source at x=4 can leave its own blocked body through x=3,
    # while x=5 is blocked.  Treating all blocked cells as invalid sources would
    # incorrectly remove the enemy's territory altogether.
    area, fraction, margin = race_features((0, 0), [(4, 0)], 7, 1, {(4, 0), (5, 0)}, {})
    assert area == 2
    assert fraction == 2 / 4
    assert margin == 2


def test_race_uses_finite_uncontested_sentinel_and_respects_walls() -> None:
    area, fraction, margin = race_features((0, 0), [], 3, 2, {(1, 0), (1, 1)}, {})
    assert area == 2
    assert fraction == 0.0
    assert margin == 7  # width * height + 1


def _sim() -> SimpleNamespace:
    # Hero faces right at (1, 1), opponent is at (5, 1); straight normal move
    # lands at (2, 1), leaving a valid post-action arrival race.
    bodies = np.zeros((1, 2, 8, 2), np.int16)
    bodies[0, 0, 0] = (1, 1)
    bodies[0, 0, 7] = (0, 1)
    bodies[0, 0, 6] = (0, 0)
    bodies[0, 1, 0] = (5, 1)
    bodies[0, 1, 7] = (5, 0)
    return SimpleNamespace(
        E=1,
        S=2,
        cap=8,
        grid_w=7,
        grid_h=3,
        alive=np.array([[True, True]]),
        seg_count=np.array([[3, 2]], np.int16),
        head_ptr=np.zeros((1, 2), np.int16),
        bodies=bodies,
        length=np.array([[3, 2]], np.int16),
        boost_frames=np.zeros((1, 2), np.int16),
        direction=np.array([[1, 3]], np.int16),
        cfg=SimpleNamespace(min_boost_length=5, boost_length_cost_frames=5),
    )


def test_feature_extraction_is_action_major_and_does_not_mutate_simulator() -> None:
    sim = _sim()
    before = {
        name: np.array(getattr(sim, name), copy=True)
        for name in vars(sim)
        if isinstance(getattr(sim, name), np.ndarray)
    }
    values = compute_features(sim)
    assert values.shape == (1, 2, 18)
    assert values.dtype == np.float32
    # Action one (straight normal) is valid and has a nonempty strict territory.
    assert values[0, 0, 3] > 0.0
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(sim, name), original)


def test_unavailable_boost_matches_its_normal_move_and_wall_move_is_zero() -> None:
    sim = _sim()
    values = compute_features(sim)[0, 0].reshape(6, 3)
    # Length three is below min_boost_length, matching T5's non-strict one-step
    # convention: boost action 4 resolves to straight normal action 1.
    np.testing.assert_array_equal(values[1], values[4])
    sim.bodies[0, 0, 0] = (0, 1)
    sim.bodies[0, 0, 7] = (0, 2)
    sim.bodies[0, 0, 6] = (1, 2)
    sim.direction[0, 0] = 0
    # Relative-left normal points through the west wall from x=0.
    assert np.array_equal(compute_features(sim)[0, 0].reshape(6, 3)[0], np.zeros(3, np.float32))


def test_raw_area_switch_preserves_other_columns_and_real_boost_uses_two_cells() -> None:
    sim = _sim()
    logged = compute_features(sim)[0, 0].reshape(6, 3)
    raw = compute_features(sim, raw_area=True)[0, 0].reshape(6, 3)
    np.testing.assert_allclose(logged[:, 0], np.log2(np.minimum(raw[:, 0], 4096) + 1) / 12)
    np.testing.assert_array_equal(logged[:, 1:], raw[:, 1:])
    # Raise logical length above the boost threshold.  Action four advances two
    # cells, so its post-action head territory cannot equal action one's here.
    sim.length[0, 0] = 5
    boosted = compute_features(sim, raw_area=True)[0, 0].reshape(6, 3)
    assert boosted[4, 0] != boosted[1, 0]
