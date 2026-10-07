"""perf-sim: optimized hot paths equal the verbatim pre-optimization code, bit for bit.

Each optimized function is compared with its verbatim main@db2ef7a twin
(``tests/perf_sim_reference.py``) on seeded random worlds that stress the exactness
arguments: long and coiled bodies, overlapping snakes, growing and full snakes, dead snakes,
heads on and outside the walls, integer coordinates OFF the segment lattice (the proofs need
integers, not the lattice), rectangular and circular arenas, several cell sizes, and the
caches under in-place mutation. Floats are compared by ``repr`` (so 0.0 vs -0.0 or a 1-ulp
change fails). Whole-episode identity (records, per-frame world digests, network bytes,
actor transitions) is the separate harness in ``research/perf_sim_20261007``.
"""

from __future__ import annotations

import math
import random
from typing import List

import pytest

from src.core.game_config import AppConfig, GameSettings, initialize_config
from src.game import snake_geometry
from src.game.ai_snake import simulate_relative_action_fatality
from src.game.food_manager import FoodManager
from src.game.game_logic import GameLogic, exact_radius_sq
from src.game.snake import Snake
from src.game.snake_geometry import clear_geometry_cache
from tests import perf_sim_reference as ref

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


@pytest.fixture(autouse=True)
def _default_config():
    initialize_config(AppConfig.from_defaults())
    clear_geometry_cache()
    yield
    initialize_config(AppConfig.from_defaults())
    clear_geometry_cache()


def _init(width: int, height: int, circular: bool = False) -> None:
    kwargs = dict(width=width, height=height, num_snakes=6)
    if circular:
        kwargs.update(
            arena_type="circular",
            arena_radius=min(width, height) // 2 - 5,
            arena_center_x=width // 2,
            arena_center_y=height // 2,
        )
    initialize_config(AppConfig(game=GameSettings(**kwargs)))


def _walk(rng: random.Random, start, n: int, ss: int, width: int, height: int, coil: float):
    """Head-first body of ``n`` points: a lattice walk from ``start`` (any integer origin)."""
    x, y = start
    body = [(x, y)]
    d = rng.choice(DIRS)
    for _ in range(n - 1):
        if rng.random() < coil:
            d = rng.choice(DIRS)
        x, y = x - d[0] * ss, y - d[1] * ss
        if not (-2 * ss <= x <= width + 2 * ss and -2 * ss <= y <= height + 2 * ss):
            d = (-d[0], -d[1])
            x, y = x + 2 * d[0] * ss, y + 2 * d[1] * ss
        body.append((x, y))
    return body


def _world(seed: int, ss: int = 10, width: int = 300, height: int = 220, n_snakes: int = 6):
    rng = random.Random(seed)
    snakes: List[Snake] = []
    for sid in range(n_snakes):
        # Off-lattice integer origins for some snakes (offset in [0, ss)).
        off = (rng.randrange(ss), rng.randrange(ss)) if rng.random() < 0.4 else (0, 0)
        start = (
            rng.randrange(-ss, width + ss) // ss * ss + off[0],
            rng.randrange(-ss, height + ss) // ss * ss + off[1],
        )
        n = rng.choice([1, 2, 3, 4, 5, 8, 20, 60, 150, 400])
        snake = Snake(sid, (255, 0, 0), start, ss, width, height, food_capacity=50)
        snake.segments = _walk(rng, start, n, ss, width, height, coil=rng.choice([0.1, 0.5, 0.9]))
        snake.direction = rng.choice(DIRS)
        snake.length = max(1, n + rng.choice([0, 0, 0, 1, 3]))  # some still growing
        snake.boost_frames = rng.randrange(0, 12)
        snake.is_alive = rng.random() > 0.15
        snake.last_move_positions = (
            [snake.segments[0]] if rng.random() < 0.5 else list(snake.segments[:2])
        )
        snakes.append(snake)
    food = [
        (rng.randrange(0, width) // ss * ss, rng.randrange(0, height) // ss * ss)
        for _ in range(rng.choice([0, 1, 30, 120]))
    ]
    return snakes, food


def _r(values) -> str:
    return repr(list(values))


SEEDS = list(range(40))


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("ss", [10, 7, 1])
def test_state_features_match_reference(circular: bool, ss: int) -> None:
    width, height = 30 * ss + 3, 22 * ss + 1
    _init(width, height, circular)
    for seed in SEEDS:
        snakes, food = _world(seed * 7 + ss, ss=ss, width=width, height=height)
        for me in snakes:
            if not me.is_alive:
                continue
            assert _r(me._get_per_action_danger(snakes)) == _r(
                ref.ref_get_per_action_danger(me, snakes)
            ), (seed, me.id)
            # Also with self absent from the roster (the proximity self-term rule).
            others = [s for s in snakes if s is not me]
            assert _r(me._get_per_action_danger(others)) == _r(
                ref.ref_get_per_action_danger(me, others)
            ), (seed, me.id, "no-self")
            assert _r(me._get_danger_map(snakes)) == _r(ref.ref_get_danger_map(me, snakes))
            hx, hy = me.head
            assert _r(me._get_enhanced_food_state(food, hx, hy)) == _r(
                ref.ref_get_enhanced_food_state(me, food, hx, hy)
            )
            assert _r(me._get_free_space_features(snakes)) == _r(
                ref.ref_get_free_space_features(me, snakes)
            ), (seed, me.id)
            # Second call: served from the result cache, still identical.
            assert _r(me._get_free_space_features(snakes)) == _r(
                ref.ref_get_free_space_features(me, snakes)
            )


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("ss", [10, 3])
def test_action_mask_simulation_matches_reference(circular: bool, ss: int) -> None:
    width, height = 30 * ss, 22 * ss
    _init(width, height, circular)
    from src.game.ai_snake import AISnake

    for seed in SEEDS:
        snakes, _ = _world(1000 + seed, ss=ss, width=width, height=height)
        for me in snakes:
            # Borrow AISnake's simulation helpers exactly as ScriptedSnake does.
            me._in_bounds_position = AISnake._in_bounds_position.__get__(me)
            me._simulate_move_after_action = AISnake._simulate_move_after_action.__get__(me)
            me._segments_collide_after_move = AISnake._segments_collide_after_move.__get__(me)
        for me in snakes:
            if not me.is_alive:
                continue
            got = simulate_relative_action_fatality(me, snakes)
            want = ref.ref_simulate_relative_action_fatality(me, snakes)
            assert got == want, (seed, me.id)
            got_none = simulate_relative_action_fatality(me, None)
            assert got_none == ref.ref_simulate_relative_action_fatality(me, None)


@pytest.mark.parametrize("ss", [10, 4])
def test_collision_checks_match_reference(ss: int) -> None:
    width, height = 30 * ss, 22 * ss
    _init(width, height)
    for seed in SEEDS:
        snakes, _ = _world(2000 + seed, ss=ss, width=width, height=height)
        # Plant some guaranteed hits: a head on another body, a head on its own body.
        if len(snakes[1].segments) > 2:
            snakes[0].last_move_positions = [snakes[1].segments[2]]
        for a in snakes:
            assert GameLogic.check_self_collision(a) == ref.ref_check_self_collision(a)
            for b in snakes:
                if a is not b:
                    assert GameLogic.check_body_collision(a, b) == ref.ref_check_body_collision(
                        a, b
                    ), (seed, a.id, b.id)


def test_food_cell_checks_match_reference() -> None:
    rng = random.Random(5)
    for trial in range(200):
        ss = rng.choice([10, 7])
        fm = FoodManager(300, 200, 50, 0, segment_size=ss)
        fm.food = [(rng.randrange(-15, 300), rng.randrange(-15, 200)) for _ in range(40)]
        twin = FoodManager(300, 200, 50, 0, segment_size=ss)
        twin.food = list(fm.food)
        for _ in range(10):
            pos = (rng.randrange(-15, 300), rng.randrange(-15, 200))
            assert fm._position_overlaps_food(pos) == ref.ref_position_overlaps_food(twin, pos)
            assert fm.consume_at(pos, ss) == ref.ref_consume_at(twin, pos, ss)
            assert fm.food == twin.food


def test_caches_follow_in_place_mutation() -> None:
    """Moving a snake in place (same list, same length) must never serve stale geometry."""
    _init(300, 220)
    snakes, _ = _world(77, ss=10)
    for s in snakes:
        s.is_alive = True
    me, other = snakes[0], snakes[1]
    for step in range(60):
        assert _r(me._get_free_space_features(snakes)) == _r(
            ref.ref_get_free_space_features(me, snakes)
        )
        assert _r(me._get_per_action_danger(snakes)) == _r(
            ref.ref_get_per_action_danger(me, snakes)
        )
        for s in (me, other):
            d = random.Random(step).choice(DIRS)
            hx, hy = s.segments[0]
            s.segments.insert(0, (hx + d[0] * 10, hy + d[1] * 10))
            s.segments.pop()  # same length, same list object
            s.direction = d


def test_exact_radius_sq_contract() -> None:
    for r in (1, 3, 7, 10, 30, 300, 9999):
        r2 = exact_radius_sq(r)
        assert r2 == r * r
        for d2 in range(max(0, r2 - 3 * r), r2 + 3 * r):
            assert (d2**0.5 < r) == (d2 < r2)
            assert (math.sqrt(d2) < r) == (d2 < r2)
            assert (math.sqrt(d2) > r) == (d2 > r2)
    assert exact_radius_sq(10.0) is None  # floats keep the original float path
    assert exact_radius_sq(True) is None
    assert exact_radius_sq(0) is None
    assert exact_radius_sq(10) == 100  # the (type, value) cache key keeps 10 and 10.0 apart


def test_geometry_cache_is_content_validated() -> None:
    a = Snake(1, (0, 0, 0), (50, 50), 10, 300, 200)
    a.segments = [(50, 50), (40, 50), (30, 50)]
    a.length = 3
    g1 = snake_geometry.geometry_for(a, 10)
    assert snake_geometry.geometry_for(a, 10) is g1
    a.segments[1] = (40, 60)  # in-place edit, same length and head
    g2 = snake_geometry.geometry_for(a, 10)
    assert g2 is not g1 and (4, 6) in g2.body_cells
    a.length = 4  # growing: the vacating tail now counts as an obstacle
    g3 = snake_geometry.geometry_for(a, 10)
    assert g3 is not g2
    assert sum(len(v) for v in g3.obstacle_cells.values()) == 3


@pytest.mark.parametrize("make_ss", [lambda: 10.0, lambda: __import__("numpy").int64(10)])
def test_non_int_cell_size_takes_the_original_float_path(make_ss) -> None:
    """Borrowers of the mask simulation (ScriptedSnake, parity's reference snake) must
    reach the float fallback too (review finding: it used to be a bound-method call)."""
    from src.game.scripted_snake import ScriptedSnake
    from src.simd_env.parity import _ref_snake_class

    _init(300, 220)
    for seed in range(10):
        snakes, _ = _world(3000 + seed, ss=10)
        for snake in snakes:
            snake.segment_size = make_ss()
        borrowed = [ScriptedSnake, _ref_snake_class()]
        for cls in borrowed:
            for me in snakes:
                if not me.is_alive:
                    continue
                me._in_bounds_position = cls._in_bounds_position.__get__(me)
                me._simulate_move_after_action = cls._simulate_move_after_action.__get__(me)
                me._segments_collide_after_move = cls._segments_collide_after_move.__get__(me)
                assert simulate_relative_action_fatality(
                    me, snakes
                ) == ref.ref_simulate_relative_action_fatality(me, snakes)
