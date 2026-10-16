"""Hand-grid checks for the plain-NumPy T5 topology feature reference."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.simd_env.t5_features import (_advance, _flood, _release_map,
                                      compute_features)


def _sim(width, height, snakes, *, direction=1, lengths=None, boost_frames=None):
    """Minimal read-only ring surface consumed by ``compute_features``.

    ``snakes`` is head-first; all slots are alive.  Keeping this fixture as a
    hand-built lattice makes the topology assertions independent of simulator
    spawning and RNG.
    """
    slots = len(snakes)
    cap = max(1, max(map(len, snakes)))
    bodies = np.full((1, slots, cap, 2), -99, dtype=np.int64)
    for slot, cells in enumerate(snakes):
        # ``head_ptr == 0`` means body offset k is stored at ``-k mod cap``.
        for offset, cell in enumerate(cells):
            bodies[0, slot, (-offset) % cap] = cell
    if lengths is None:
        lengths = [len(cells) for cells in snakes]
    if boost_frames is None:
        boost_frames = [0] * slots
    return SimpleNamespace(
        E=1,
        S=slots,
        cap=cap,
        grid_w=width,
        grid_h=height,
        bodies=bodies,
        head_ptr=np.zeros((1, slots), dtype=np.int64),
        seg_count=np.array([[len(cells) for cells in snakes]], dtype=np.int64),
        length=np.asarray([lengths], dtype=np.int64),
        alive=np.ones((1, slots), dtype=bool),
        direction=np.full((1, slots), direction, dtype=np.int64),
        boost_frames=np.asarray([boost_frames], dtype=np.int64),
        cfg=SimpleNamespace(min_boost_length=3, boost_length_cost_frames=3),
    )


def _at(features, action, feature):
    return features[0, 0, action * 4 + feature]


def test_area_log_is_uncapped_post_move_area_and_action_major_float32():
    # In an empty 5x5 field, the straight normal move lands in the sole free
    # component.  This hand grid establishes the log normalisation and layout.
    values = compute_features(_sim(5, 5, [[(2, 2)]], direction=1))

    assert values.dtype == np.float32
    assert values.shape == (1, 1, 24)
    assert _at(values, 1, 0) == np.float32(np.log2(26) / 12.0)
    # The feature order is action-major: action 1 starts at offset four.
    assert _at(values, 1, 1) == np.float32(0.0)  # Three boosts hit the right wall.


def test_boost3_area_is_zero_when_any_of_three_boosted_frames_collides():
    # A length-three hero may boost, but three boosted frames straight right
    # traverse six cells and hit this seven-cell-wide wall on the third frame.
    values = compute_features(_sim(7, 3, [[(1, 1), (0, 1), (0, 0)]], direction=1))

    assert _at(values, 1, 1) == 0.0
    # A normal one-cell move is still a valid post-move flood.
    assert _at(values, 1, 0) > 0.0


def test_core_area_prunes_a_one_cell_neck_from_the_post_move_flood():
    # Static opponent segments close every cell in x=3 except (3,2), leaving a
    # one-cell neck between two rooms.  The ordinary flood crosses it; core
    # flood treats degree-two neck cells as impassable.
    walls = [(3, 0), (3, 1), (3, 3), (3, 4)]
    values = compute_features(
        _sim(7, 5, [[(1, 2), (0, 2), (0, 1)], walls], direction=1, lengths=[3, 4])
    )

    assert _at(values, 1, 0) > _at(values, 1, 2) > 0.0


def test_tail_reachable_records_a_ttl_one_own_tail_visited_by_the_flood():
    # After moving right, the physical tail at (1,1) has TTL one.  The flood
    # reaches it by circling through the open top row after its conservative
    # slack-one release distance has elapsed.
    values = compute_features(_sim(7, 3, [[(2, 1), (1, 1), (0, 1)]], direction=1))

    assert _at(values, 1, 3) == 1.0


def test_area_is_uncapped_beyond_the_v8_160_cell_limit():
    values = compute_features(_sim(20, 20, [[(10, 10)]], direction=1))

    assert _at(values, 1, 0) == np.float32(np.log2(401) / 12.0)
    assert _at(values, 1, 0) > np.float32(np.log2(161) / 12.0)


def test_boost3_uses_all_three_frames_and_pays_the_burn_cost():
    # The third boost frame reaches the burn cadence and lowers logical length.
    body = [(1, 2), (0, 2), (0, 1)]
    length, frames = 3, 0
    for _ in range(3):
        moved = _advance(body, length, frames, 1, True, 3, 3, 12, 5, set())
        assert moved is not None
        body, length, frames = moved

    assert length == 2
    assert frames == 0
    assert len(body) == 2


def test_post_food_fill_in_lag_preserves_tail_ttl_until_the_body_catches_up():
    # Logical length can exceed physical segments right after food.  A move
    # inserts without popping; its physical tail is not TTL one yet.
    moved = _advance([(2, 1), (1, 1)], 4, 0, 1, False, 3, 3, 8, 4, set())
    assert moved is not None
    body, length, _ = moved
    release, ttl_one = _release_map(body, length)

    assert body == [(3, 1), (2, 1), (1, 1)]
    assert ttl_one == set()
    assert release[(1, 1)] == 3


def test_boost3_rejects_an_intermediate_static_opponent_collision():
    values = compute_features(
        _sim(12, 3, [[(1, 1), (0, 1), (0, 0)], [(2, 1)]], direction=1, lengths=[3, 1])
    )

    assert _at(values, 1, 1) == 0.0


def test_boost_collision_uses_the_completed_post_frame_body_tail_vacancy():
    # Both boost traversals are old tail cells.  The simulator judges self-hit
    # against the completed post-frame body (offset three onward), so this is
    # legal even though the first substep would see one of those cells occupied.
    body = [
        (2, 2),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (5, 2),
        (5, 3),
        (4, 3),
        (3, 3),
        (3, 2),
        (4, 2),
    ]

    moved = _advance(body, 11, 0, 1, True, 3, 3, 8, 8, set())

    assert moved is not None
    assert moved[0][:3] == [(4, 2), (3, 2), (2, 2)]


def test_unavailable_one_step_boost_uses_the_v8_normal_speed_fallback():
    values = compute_features(_sim(8, 4, [[(2, 2), (1, 2)]], direction=1))

    for feature in (0, 2, 3):
        assert _at(values, 4, feature) == _at(values, 1, feature)
    # The distinct boost-three counterfactual remains strict and unavailable.
    assert _at(values, 4, 1) == 0.0


def test_core_degree_counts_own_cells_only_when_their_release_ttl_allows_it():
    # The two vertical neighbours are own-body cells that do eventually release,
    # but not at the seed's arrival distance.  The one-pass core degree is two
    # (left/right), so it rejects the seed.  Treating own cells as permanently
    # free would incorrectly give it degree four.
    area, _ = _flood(
        (2, 1),
        5,
        3,
        set(),
        {(2, 0): 4, (2, 2): 4},
        set(),
        core=True,
    )

    assert area == 0


def test_feature_extraction_does_not_mutate_the_simulator_surface():
    sim = _sim(10, 5, [[(3, 2), (2, 2), (1, 2)], [(8, 2), (8, 1)]], direction=1)
    before = {
        name: np.array(getattr(sim, name), copy=True)
        for name in (
            "bodies",
            "head_ptr",
            "seg_count",
            "length",
            "alive",
            "direction",
            "boost_frames",
        )
    }

    compute_features(sim)

    for name, value in before.items():
        assert np.array_equal(getattr(sim, name), value), name
