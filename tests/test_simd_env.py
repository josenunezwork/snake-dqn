"""Tests for the vectorized batch simulator (``src/simd_env``).

Covers: single-env determinism, hand-checked head-on resolution (v2 size +
mutual), food consumption growth, boost 2-step + v2 trail-pellet drop, action
mask vs a brute-force fatality check, reward vs the pure ``compute_reward_v2``,
and RNG spawn parity vs the live ``GameLogic.find_empty_position``.

All tests use tiny arenas / few frames to stay fast and hand-checkable.
"""

import random

import numpy as np
import pytest

from src.core.reward_events import RewardEvents, compute_reward_v2
from src.simd_env import DEATH_BODY, DEATH_HEAD, BatchSim, BatchSimConfig
from src.simd_env.rng import EnvRng


def _small_cfg(**overrides):
    """A small rectangular v2 config for fast hand-checkable tests."""
    base = dict(
        num_envs=1,
        num_snakes=2,
        game_width=200,
        game_height=200,
        segment_size=10,
        wall_thickness=10,
        initial_food=5,
        max_food=5,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
        gamma=0.99,
        max_capacity=64,
    )
    base.update(overrides)
    return BatchSimConfig(**base)


# ---------------------------------------------------------------------------
# Helpers to place snakes deterministically (bypass spawn RNG for scenarios)
# ---------------------------------------------------------------------------
def _place_snake(sim, e, sidx, cells, direction=1, length=None):
    """Place a snake with an explicit body (head first) at cell coordinates."""
    cap = sim.cap
    n = len(cells)
    sim.seg_count[e, sidx] = n
    sim.head_ptr[e, sidx] = n - 1  # head at highest written ring slot
    for k, cell in enumerate(cells):
        ring = (int(sim.head_ptr[e, sidx]) - k) % cap
        sim.bodies[e, sidx, ring] = cell
    sim.length[e, sidx] = length if length is not None else n
    sim.direction[e, sidx] = direction
    sim.alive[e, sidx] = True
    sim._reward_prev_length[e, sidx] = sim.length[e, sidx]


def _clear_food(sim):
    for e in range(sim.E):
        sim.food_cells[e] = []
        sim.corpse_cells[e] = set()
        sim.food_set[e] = set()  # keep the membership index in sync (invariant)


# ===========================================================================
# Determinism
# ===========================================================================
def test_single_env_determinism_same_seed_same_trajectory():
    cfg = _small_cfg(
        num_envs=1,
        num_snakes=3,
        game_width=400,
        game_height=400,
        initial_food=30,
        max_food=30,
        max_capacity=128,
    )
    rng = random.Random(123)
    actions_log = [
        np.array([[rng.randint(0, 5) for _ in range(3)]], dtype=np.int64) for _ in range(40)
    ]

    def run():
        sim = BatchSim(cfg, seeds=[7], train_mode=True)
        traj = []
        for acts in actions_log:
            sim.step(acts)
            traj.append(
                (
                    sim.get_lengths().copy(),
                    sim.get_alive().copy(),
                    sim.get_heads().copy(),
                    [tuple(sim.get_food(0))],
                    sim.get_reward().copy(),
                )
            )
        return traj

    t1 = run()
    t2 = run()
    for (l1, a1, h1, f1, r1), (l2, a2, h2, f2, r2) in zip(t1, t2):
        assert np.array_equal(l1, l2)
        assert np.array_equal(a1, a2)
        assert np.array_equal(h1, h2)
        assert f1 == f2
        assert np.array_equal(r1, r2)


def test_different_seeds_diverge():
    cfg = _small_cfg(
        num_envs=2,
        num_snakes=3,
        game_width=400,
        game_height=400,
        initial_food=30,
        max_food=30,
        max_capacity=128,
    )
    sim = BatchSim(cfg, seeds=[1, 999], train_mode=True)
    # Initial food layouts should differ across the two envs.
    assert sim.get_food(0) != sim.get_food(1)


# ===========================================================================
# Head-on resolution (v2 size + mutual)
# ===========================================================================
def test_headon_v2_bigger_survives_and_gets_credit():
    cfg = _small_cfg()
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Two snakes moving toward each other; heads will land on the same cell.
    # Big snake (length 10) moving right; small snake (length 2) moving left.
    # Place so that after one step both heads reach cell (5,5).
    big = [(4, 5)] + [(3 - i, 5) for i in range(9)]  # head (4,5), body trailing left
    small = [(6, 5), (7, 5)]  # head (6,5), body to the right
    _place_snake(sim, 0, 0, big, direction=1, length=10)  # moving right
    _place_snake(sim, 0, 1, small, direction=3, length=2)  # moving left

    actions = np.array([[1, 1]], dtype=np.int64)  # both straight
    sim.step(actions)

    alive = sim.get_alive()[0]
    cause = sim.get_death_cause()[0]
    kills = sim.get_kill_credit()[0]
    # Big survives, small dies via head-on, big credited with the kill.
    assert alive[0] and not alive[1]
    assert cause[1] == DEATH_HEAD
    assert kills[0] == 1
    assert kills[1] == 0


def test_headon_v2_near_equal_mutual_death():
    cfg = _small_cfg()
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Near-equal lengths (10 vs 9): 10 >= 1.15*9? 10.35 -> no. Mutual.
    a = [(4, 5)] + [(3 - i, 5) for i in range(9)]
    b = [(6, 5)] + [(7 + i, 5) for i in range(8)]
    _place_snake(sim, 0, 0, a, direction=1, length=10)
    _place_snake(sim, 0, 1, b, direction=3, length=9)

    sim.step(np.array([[1, 1]], dtype=np.int64))
    alive = sim.get_alive()[0]
    cause = sim.get_death_cause()[0]
    kills = sim.get_kill_credit()[0]
    assert not alive[0] and not alive[1]
    assert cause[0] == DEATH_HEAD and cause[1] == DEATH_HEAD
    assert kills[0] == 0 and kills[1] == 0


def test_body_collision_kill_attribution():
    cfg = _small_cfg()
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Snake 0 head runs into snake 1's body. Snake 1 is a horizontal wall of
    # cells; snake 0 approaches from below into a body cell.
    victim_head_before = (5, 6)
    _place_snake(
        sim, 0, 0, [victim_head_before, (5, 7), (5, 8)], direction=0, length=3
    )  # moving up
    # snake1 body occupies (4,5),(5,5),(6,5); head at (4,5).
    _place_snake(sim, 0, 1, [(4, 5), (5, 5), (6, 5)], direction=3, length=3)
    # snake0 moving up from (5,6) -> head (5,5), which is snake1 body (segments[1:]).
    sim.step(np.array([[1, 1]], dtype=np.int64))
    alive = sim.get_alive()[0]
    cause = sim.get_death_cause()[0]
    kills = sim.get_kill_credit()[0]
    assert not alive[0]
    assert cause[0] == DEATH_BODY
    assert kills[1] == 1


# ===========================================================================
# Food consumption / growth
# ===========================================================================
def test_food_consumption_grows_snake():
    cfg = _small_cfg(num_snakes=1)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    _place_snake(sim, 0, 0, [(5, 5)], direction=1, length=1)  # moving right
    # Put food at the cell the head will move into: (6,5).
    sim.food_cells[0] = [(6, 5)]
    sim.food_set[0] = {(6, 5)}  # keep the membership index in sync (invariant)
    sim.step(np.array([[1]], dtype=np.int64))
    assert sim.get_lengths()[0, 0] == 2
    # Pellet consumed.
    assert (6, 5) not in sim.get_food(0)


def test_growth_timing_body_fills_next_frame():
    cfg = _small_cfg(num_snakes=1, max_food=0, initial_food=0)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    _place_snake(sim, 0, 0, [(5, 5)], direction=1, length=1)
    sim.food_cells[0] = [(6, 5)]
    sim.food_set[0] = {(6, 5)}  # keep the membership index in sync (invariant)
    sim.step(np.array([[1]], dtype=np.int64))  # eats, length -> 2, body still 1
    assert sim.get_lengths()[0, 0] == 2
    assert len(sim.get_bodies(0, 0)) == 1  # body fills in next frame
    sim.step(np.array([[1]], dtype=np.int64))
    assert len(sim.get_bodies(0, 0)) == 2  # now grown


# ===========================================================================
# Boost 2-step + v2 trail pellet
# ===========================================================================
def test_boost_two_step_moves_two_cells():
    cfg = _small_cfg(num_snakes=1, initial_food=0, max_food=0)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    body = [(5 - i, 5) for i in range(6)]  # length 6, head (5,5) moving right
    _place_snake(sim, 0, 0, body, direction=1, length=6)
    sim.step(np.array([[4]], dtype=np.int64))  # straight + boost
    # Head advanced two cells: (5,5) -> (7,5).
    assert tuple(sim.get_heads()[0, 0]) == (7, 5)


def test_boost_burn_drops_trail_pellet_v2():
    cfg = _small_cfg(
        num_snakes=1, initial_food=0, max_food=5, boost_length_cost_frames=1
    )  # burn every boosting frame; max_food>0 so the corpse cap admits the trail pellet
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    body = [(10 - i, 5) for i in range(6)]  # length 6, head (10,5)
    _place_snake(sim, 0, 0, body, direction=1, length=6)
    assert len(sim.get_food(0)) == 0
    sim.step(np.array([[4]], dtype=np.int64))  # boost straight, burns immediately
    # A trail pellet (corpse-class) should have been dropped at the vacated tail.
    corpse = sim.get_corpse_food(0)
    assert len(corpse) == 1
    # Length paid one segment: 6 -> 5.
    assert sim.get_lengths()[0, 0] == 5


def test_boost_no_trail_at_v1():
    cfg = _small_cfg(
        num_snakes=1, initial_food=0, max_food=0, boost_length_cost_frames=1, mechanics_version=1
    )
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    body = [(10 - i, 5) for i in range(6)]
    _place_snake(sim, 0, 0, body, direction=1, length=6)
    sim.step(np.array([[4]], dtype=np.int64))
    assert len(sim.get_corpse_food(0)) == 0


# ===========================================================================
# Action mask vs brute-force fatality
# ===========================================================================
def _brute_fatality(sim, e, sidx):
    """Independent brute-force 6-bit safe mask for one snake.

    Recomputes fatality from scratch using the same rules: wall (pixel bounds),
    self (own segments[3:]), other snakes' full bodies (head+body).
    """
    from src.simd_env.batch_sim import CARDINAL

    s = sim.s
    W, H = sim.cfg.game_width, sim.cfg.game_height
    head = tuple(int(x) for x in sim.get_heads()[e, sidx])
    cur_dir = int(sim.get_directions()[e, sidx])
    own = set(sim._snake_body_cells(e, sidx, start=3))
    others = set()
    for j in range(sim.S):
        if j == sidx or not sim.get_alive()[e, j]:
            continue
        others |= set(sim._snake_body_cells(e, j, start=0))
    can_boost = int(sim.get_lengths()[e, sidx]) >= sim.cfg.min_boost_length

    def fatal(cells):
        for cx, cy in cells:
            x, y = cx * s, cy * s
            if x < 0 or x >= W or y < 0 or y >= H:
                return True
        for c in cells:
            if c in own or c in others:
                return True
        return False

    mask = [False] * 6
    for rel in range(3):
        delta = -1 if rel == 0 else (1 if rel == 2 else 0)
        dv = CARDINAL[(cur_dir + delta) % 4]
        s1 = (head[0] + int(dv[0]), head[1] + int(dv[1]))
        if not fatal([s1]):
            mask[rel] = True
            if can_boost:
                s2 = (s1[0] + int(dv[0]), s1[1] + int(dv[1]))
                if not fatal([s1, s2]):
                    mask[rel + 3] = True
    return mask


def test_mask_matches_bruteforce_open_field():
    cfg = _small_cfg(num_snakes=2)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    _place_snake(sim, 0, 0, [(10 - i, 10) for i in range(6)], direction=1, length=6)
    _place_snake(sim, 0, 1, [(3, 3)], direction=1, length=1)
    sim._rebuild_traversed_from_heads()
    sim._last_mask = sim._compute_action_masks()
    m = sim.get_action_mask()[0, 0]
    b = _brute_fatality(sim, 0, 0)
    assert list(m) == b


def test_mask_matches_bruteforce_near_wall_and_body():
    cfg = _small_cfg(num_snakes=2, game_width=100, game_height=100)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Snake near the right wall (grid_w = 10, cells 0..9). Head at (9,5) moving right.
    _place_snake(sim, 0, 0, [(9, 5), (8, 5), (7, 5), (6, 5), (5, 5)], direction=1, length=5)
    # Another snake forming an obstacle to the "left turn" (up) direction.
    _place_snake(sim, 0, 1, [(9, 4), (9, 3)], direction=0, length=2)
    sim._rebuild_traversed_from_heads()
    sim._last_mask = sim._compute_action_masks()
    m = sim.get_action_mask()[0, 0]
    b = _brute_fatality(sim, 0, 0)
    assert list(m) == b


# ===========================================================================
# Reward vs pure compute_reward_v2
# ===========================================================================
def test_reward_matches_compute_reward_v2_food_and_kill():
    cfg = _small_cfg()
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Reuse the head-on scenario: big kills small.
    big = [(4, 5)] + [(3 - i, 5) for i in range(9)]
    small = [(6, 5), (7, 5)]
    _place_snake(sim, 0, 0, big, direction=1, length=10)
    _place_snake(sim, 0, 1, small, direction=3, length=2)
    prev_len_big = int(sim.get_lengths()[0, 0])
    prev_len_small = int(sim.get_lengths()[0, 1])

    sim.step(np.array([[1, 1]], dtype=np.int64))
    r = sim.get_reward()[0]

    # Big: survives, gains no length, kills a length-2 victim.
    new_len_big = int(sim.get_lengths()[0, 0])
    exp_big, _ = compute_reward_v2(
        RewardEvents(
            prev_length=prev_len_big, new_length=new_len_big, died=False, gamma=0.99, kills=(2.0,)
        )
    )
    exp_small, _ = compute_reward_v2(
        RewardEvents(prev_length=prev_len_small, new_length=0, died=True, gamma=0.99, kills=())
    )
    assert r[0] == pytest.approx(exp_big)
    assert r[1] == pytest.approx(exp_small)


def test_reward_matches_compute_reward_v2_plain_step():
    cfg = _small_cfg(num_snakes=1, initial_food=0, max_food=0)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    _place_snake(sim, 0, 0, [(5, 5)], direction=1, length=1)
    sim.step(np.array([[1]], dtype=np.int64))
    r = sim.get_reward()[0, 0]
    exp, _ = compute_reward_v2(RewardEvents(prev_length=1, new_length=1, died=False, gamma=0.99))
    assert r == pytest.approx(exp)


# ===========================================================================
# RNG spawn parity vs the live GameLogic.find_empty_position
# ===========================================================================
def test_rng_find_empty_matches_live_gamelogic():
    """EnvRng.find_empty_position must match GameLogic.find_empty_position.

    Both consume the global/instance ``random`` stream in the same draw order
    (x then y, snap), with the same cell-exact snake-overlap rejection, so with
    an identical seed and snake set they return the same position.
    """
    from src.core import game_config
    from src.core.config_loader import load_and_initialize_config
    from src.game.game_logic import GameLogic

    # Ensure a rectangular arena config is active for GameLogic; restore the
    # global config singleton afterward so we don't leak into other tests.
    prev_config = game_config._current_config
    load_and_initialize_config("configs/mechanics_v2.yaml")

    width, height, s, wall = 200, 200, 10, 10

    # A couple of fake snakes occupying some cells (living).
    class _FakeSnake:
        def __init__(self, cells):
            self.is_alive = True
            self.segment_size = s
            self.segments = [(c * s, r * s) for (c, r) in cells]

    snake_cells = [(2, 2), (2, 3), (5, 5)]
    fake = _FakeSnake(snake_cells)

    # Live path.
    random.seed(20260705)
    live = GameLogic.find_empty_position(width, height, [fake])

    # Batched path with the same seed and the same occupied cells.
    rng = EnvRng(20260705, width, height, s, wall)
    occ = np.array(snake_cells, dtype=np.int64)
    mine = rng.find_empty_position(occ)

    game_config._current_config = prev_config
    assert mine == live


# ---------------------------------------------------------------------------
# Regression: reward / corpse-order / ring-buffer parity edge cases
# ---------------------------------------------------------------------------
def test_same_frame_double_kill_reward_bit_exact():
    """A same-frame multi-kill reward must equal ``compute_reward_v2`` to the ULP.

    ``K * sum(victims)`` diverges by one ULP from the per-victim accumulation
    ``sum(K * v ...)`` on victim lengths like (1, 6); the reward must match the
    pure function's left-to-right accumulation exactly (parity gate uses ``!=``).
    """
    cfg = _small_cfg(num_snakes=3, initial_food=0, max_food=0)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Snake 0 (killer): horizontal body length 6, head (10,10) facing right.
    _place_snake(
        sim, 0, 0, [(10, 10), (9, 10), (8, 10), (7, 10), (6, 10), (5, 10)], direction=1, length=6
    )
    # Victim 1: length 1, moves up into (7,10) (snake0 body[1:]).
    _place_snake(sim, 0, 1, [(7, 11)], direction=0, length=1)
    # Victim 2: length 6, moves up into (8,10) (snake0 body[1:]).
    _place_snake(
        sim, 0, 2, [(8, 11), (8, 12), (8, 13), (8, 14), (8, 15), (8, 16)], direction=0, length=6
    )

    sim.step(np.array([[1, 1, 1]], dtype=np.int64))

    assert sim.get_kill_credit()[0][0] == 2
    assert sorted(sim.get_kill_victim_lengths(0, 0)) == [1, 6]
    expected, _ = compute_reward_v2(
        RewardEvents(prev_length=6, new_length=6, died=False, gamma=cfg.gamma, kills=(1, 6))
    )
    # Bit-exact: identical object equality, not approx.
    assert sim.get_reward()[0][0] == expected


def test_corpse_food_dropped_in_resolution_order_not_id_order():
    """Two same-frame deaths drop corpses in collision-resolution order.

    Snake 3 (higher id) loses a head-on and is resolved dead BEFORE snake 2
    (lower id) dies via a body collision, so snake 3's corpse must be appended to
    the food list first. An id-order drop would append snake 2 first, desyncing
    the ordered food list from the live game.
    """
    # max_food>0 so the corpse cap (= max_food) admits both dropped corpses; the
    # assertion below filters to corpse food, so ambient top-up does not intrude.
    cfg = _small_cfg(num_snakes=4, game_width=300, game_height=300, initial_food=0, max_food=5)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Snake 0: body used as the wall snake 2 dies into (row 20).
    _place_snake(sim, 0, 0, [(5, 20), (6, 20), (7, 20)], direction=3, length=3)
    # Snake 2 (lower id): moves up into (6,20) (snake0 body[1:]); length-1 victim.
    _place_snake(sim, 0, 2, [(6, 21)], direction=0, length=1)
    # Head-on: big snake 1 vs small snake 3 meeting at (11,5). Snake 1 wins.
    _place_snake(
        sim, 0, 1, [(10, 5), (9, 5), (8, 5), (7, 5), (6, 5), (5, 5)], direction=1, length=6
    )
    _place_snake(sim, 0, 3, [(12, 5)], direction=3, length=1)

    sim.step(np.array([[3, 1, 1, 1]], dtype=np.int64))

    alive = sim.get_alive()[0]
    assert not alive[2] and not alive[3]  # both die this frame
    # Resolution order is snake3 (head-on loser) then snake2 (body death), so the
    # corpses are appended in that order: (11,5) [snake3 head after move] first.
    # (Filter to corpse food so the ambient maintain top-up is excluded.)
    assert sim.get_corpse_food(0) == [(11, 5), (6, 20)]


def test_corpse_food_cap_evicts_oldest_v2():
    """Corpse food is bounded at ``corpse_food_cap(max_food)``; oldest evicted FIFO.

    Cap-exempt corpse food is exempt from the AMBIENT budget but not unbounded;
    once the per-env corpse count exceeds the cap, the oldest corpse pellet (first
    in board order) is dropped. Uses the shared helper so this bound is identical
    in the live game (validated cross-engine by the parity suite).
    """
    from src.core.mechanics_constants import corpse_food_cap

    cfg = _small_cfg(num_snakes=1, initial_food=0, max_food=6)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    cap = corpse_food_cap(cfg.max_food)
    assert cap >= 2  # need a meaningful cap for the FIFO check
    # Drop cap+3 distinct corpse pellets in a known board order (all in-arena).
    cells = [(c, 1) for c in range(cap + 3)]
    for c in cells:
        sim._add_food(0, c, corpse=True)
    corpse = sim.get_corpse_food(0)
    assert len(corpse) == cap  # bounded at the cap, not cap+3
    assert corpse == cells[-cap:]  # FIFO: the newest `cap` survive, in board order
    # The membership index stays consistent with the ordered list after evictions.
    assert sim.food_set[0] == set(sim.food_cells[0])


def test_evict_oldest_corpse_helper():
    """The shared eviction helper drops the first corpse in list order, or None."""
    from src.core.mechanics_constants import evict_oldest_corpse

    food = [(0, 0), (1, 1), (2, 2), (3, 3)]
    corpse = {(1, 1), (3, 3)}
    assert evict_oldest_corpse(food, corpse) == (1, 1)  # first corpse in order
    assert food == [(0, 0), (2, 2), (3, 3)]
    assert corpse == {(3, 3)}
    assert evict_oldest_corpse([(0, 0)], set()) is None  # no corpse -> None


def test_ring_buffer_overflow_raises_before_corruption():
    """Reaching ``max_capacity`` at a head insert raises instead of corrupting.

    The live game's segment list grows unbounded; the batch ring buffer must not
    silently wrap the new head onto the oldest live tail. A tiny capacity makes
    the guard reachable in a single step.
    """
    cfg = _small_cfg(num_snakes=1, initial_food=0, max_food=0, max_capacity=4)
    sim = BatchSim(cfg, seeds=[0], train_mode=True)
    _clear_food(sim)
    # Fill the ring: seg_count == length == cap. Next head insert would wrap.
    _place_snake(sim, 0, 0, [(10, 10), (9, 10), (8, 10), (7, 10)], direction=1, length=4)
    with pytest.raises(RuntimeError, match="ring buffer overflow"):
        sim.step(np.array([[1]], dtype=np.int64))
