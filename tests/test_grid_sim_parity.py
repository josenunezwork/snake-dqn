"""Bit-exact parity: GridBatchSim (redesign M1) vs BatchSim and vs the live reference.

GridBatchSim replaces BatchSim's per-env Python loops with occupancy grids and
batch gathers. These tests pin that the replacement changes nothing observable:

1. Lockstep vs BatchSim (itself CI-pinned to the live game) on recorded action
   streams, comparing every load-bearing field each frame — ordered bodies,
   lengths, alive, heading, boost/hunger/respawn counters, ordered food list,
   corpse set, per-env RNG state, advisory/legal/resolved masks, rewards, done,
   death causes, kill credit + victim lengths, food-source events — for
   mechanics v1 and v2, train and watch food branches, both respawn arms.
2. The big-body regime (injected serpentines of length 150-900 on the full
   1450x830 arena), where self-traps and long-body masks matter most.
3. Directly against the live-game reference harness (``parity.run_parity``).
4. Grid invariants: the incremental grids equal a from-scratch rebuild.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simd_env import parity
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.grid_scenarios import GreedySafePolicy, inject_serpentines, serpentine_path
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.parity import SurvivorPolicy, build_parity_config, run_parity

_ARRAYS = (
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
)
_ACCESSORS = (
    "get_action_mask",
    "get_legal_action_mask",
    "get_resolved_action_mask",
    "get_reward",
    "get_done",
    "get_death_cause",
    "get_kill_credit",
    "get_transition_valid",
    "get_boosted_this_step",
)


def assert_same_world(a: BatchSim, b: GridBatchSim, frame: int) -> None:
    """Assert every observable quantity of the two simulators is identical."""
    for name in _ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=f"{frame}:{name}")
    for name in _ACCESSORS:
        np.testing.assert_array_equal(
            getattr(a, name)(), getattr(b, name)(), err_msg=f"{frame}:{name}"
        )
    for key, value in a.get_step_events().items():
        np.testing.assert_array_equal(value, b.get_step_events()[key], err_msg=f"{frame}:{key}")
    for e in range(a.E):
        assert a.food_cells[e] == b.food_cells[e], f"{frame}: food order env {e}"
        assert a.corpse_cells[e] == b.corpse_cells[e], f"{frame}: corpse env {e}"
        assert a.food_set[e] == b.food_set[e], f"{frame}: food set env {e}"
        assert a._rngs[e]._rng.getstate() == b._rngs[e]._rng.getstate(), f"{frame}: rng {e}"
        for s in range(a.S):
            assert a.get_bodies(e, s) == b.get_bodies(e, s), f"{frame}: body {e},{s}"
            assert a.get_kill_victim_lengths(e, s) == b.get_kill_victim_lengths(e, s)


def _lockstep(a: BatchSim, b: GridBatchSim, frames: int, policy, check_every: int = 1) -> dict:
    """Step both sims with the same per-frame actions; compare every frame."""
    assert_same_world(a, b, 0)
    stats = {"deaths": 0, "kills": 0, "eats": 0, "self": 0, "boost": 0}
    for f in range(frames):
        actions = policy(a)
        a.step(actions)
        b.step(actions)
        assert_same_world(a, b, f + 1)
        if f % check_every == 0:
            b.check_grid_invariants()
        stats["deaths"] += int(a.get_done().sum())
        stats["kills"] += int(a.get_kill_credit().sum())
        stats["eats"] += int(a.get_step_events()["food_ate"].sum())
        stats["self"] += int((a.get_death_cause() == 2).sum())
        stats["boost"] += int(a.get_boosted_this_step().sum())
    return stats


def _survivor(E: int, S: int, boost: float = 0.3):
    pols = [SurvivorPolicy(1000 + e, S, boost_prob=boost) for e in range(E)]

    def act(sim: BatchSim) -> np.ndarray:
        m = sim.get_action_mask()
        return np.stack([pols[e].actions(m[e]) for e in range(sim.E)])

    return act


def _greedy(seed: int, boost: float = 0.2):
    pol = GreedySafePolicy(seed, boost_prob=boost)
    return lambda sim: pol.actions(sim, sim.get_action_mask())


def _pair(cfg: BatchSimConfig, seeds, train_mode: bool, respawn: bool):
    a = BatchSim(cfg, seeds=seeds, train_mode=train_mode, allow_respawn=respawn)
    b = GridBatchSim(cfg, seeds=seeds, train_mode=train_mode, allow_respawn=respawn)
    return a, b


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("train_mode,respawn", [(True, False), (True, True), (False, True)])
def test_lockstep_small_arena(version, train_mode, respawn):
    """Crowded small arena: growth, boost burns, kills, corpses, respawns."""
    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=6,
        game_width=400,
        game_height=300,
        initial_food=60,
        max_food=80,
        mechanics_version=version,
        max_capacity=1200,
        frame_rate=3,
    )
    a, b = _pair(cfg, [11, 12, 13], train_mode, respawn)
    stats = _lockstep(a, b, 400, _survivor(3, 6))
    assert stats["deaths"] > 0 and stats["eats"] > 0
    if version == 2 and respawn:
        assert stats["kills"] > 0


@pytest.mark.parametrize("version", [1, 2])
def test_lockstep_greedy_growth(version):
    """Food-seeking growth to long bodies with boost burns and trail pellets."""
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=4,
        game_width=300,
        game_height=200,
        initial_food=80,
        max_food=100,
        mechanics_version=version,
        max_capacity=1200,
        frame_rate=2,
    )
    a, b = _pair(cfg, [5, 6], True, True)
    stats = _lockstep(a, b, 500, _greedy(3), check_every=5)
    assert stats["eats"] > 50 and stats["boost"] > 0
    assert a.length.max() > 20


@pytest.mark.parametrize("version", [1, 2])
def test_lockstep_big_bodies_full_arena(version):
    """Injected serpentines (L=150..900) on the 1450x830 gate arena."""
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=6,
        mechanics_version=version,
        max_capacity=1200,
        frame_rate=2,
    )
    a, b = _pair(cfg, [21, 22], True, True)
    inject_serpentines([a, b], [900, 600, 400, 300, 150, 150], box_height=8)
    b.check_grid_invariants()
    stats = _lockstep(a, b, 120, _greedy(9, boost=0.3), check_every=10)
    assert a.length.max() >= 600
    assert stats["eats"] > 0
    assert stats["self"] > 0, "big-body run should produce self-trap deaths"


def test_forced_fatal_moves_match():
    """Long bodies driven straight into a wall die identically in both sims."""
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, mechanics_version=2, max_capacity=600)
    a, b = _pair(cfg, [3], True, False)
    inject_serpentines([a, b], [300, 40], box_height=6)
    straight = lambda sim: np.ones((1, 2), dtype=np.int64)  # noqa: E731
    stats = _lockstep(a, b, 160, straight)
    assert stats["deaths"] >= 1


def test_active_env_mask_and_reset_envs():
    """Frozen rows stay byte-identical and soft resets reseed identically."""
    cfg = BatchSimConfig(
        num_envs=3, num_snakes=4, game_width=300, game_height=200, initial_food=30, max_food=40
    )
    a, b = _pair(cfg, [1, 2, 3], True, False)
    pol = _survivor(3, 4)
    active = np.array([True, False, True])
    for f in range(60):
        acts = pol(a)
        a.step(acts, active_env_mask=active)
        b.step(acts, active_env_mask=active)
        assert_same_world(a, b, f)
    mask = np.array([False, True, True])
    a.reset_envs(mask, seeds=[77, 78])
    b.reset_envs(mask, seeds=[77, 78])
    b.check_grid_invariants()
    _lockstep(a, b, 60, pol)


@pytest.mark.parametrize("respawn", [False, True])
def test_direct_live_reference_parity(monkeypatch, respawn):
    """GridBatchSim through the live-game reference harness (``run_parity``)."""
    monkeypatch.setattr(parity, "BatchSim", GridBatchSim)
    cfg = build_parity_config(
        num_snakes=6, game_width=400, game_height=300, initial_food=40, max_food=50
    )
    result = run_parity(
        [0, 1, 2], num_frames=600, cfg=cfg, policy="survivor", allow_respawn=respawn
    )
    assert result.divergence is None, result.divergence
    assert result.frames_tested == 1800


def test_rejects_non_multiple_arena():
    """The cell-bounds wall test needs an arena on the cell lattice."""
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, game_width=305, game_height=200)
    with pytest.raises(ValueError):
        GridBatchSim(cfg)


def test_serpentine_path_is_connected_and_simple():
    cells = serpentine_path(2, 3, 10, 4, 37)
    assert len(cells) == len(set(cells)) == 37
    for (c0, r0), (c1, r1) in zip(cells, cells[1:]):
        assert abs(c0 - c1) + abs(r0 - r1) == 1


def _place(sims, bodies, direction):
    """Set exact bodies (head first) and headings in env 0 of every sim."""
    for sim in sims:
        occupied = set()
        for sidx, cells in enumerate(bodies):
            n = len(cells)
            sim.bodies[0, sidx] = 0
            sim.bodies[0, sidx, :n] = np.array(cells[::-1], dtype=np.int64)
            sim.head_ptr[0, sidx] = n - 1
            sim.seg_count[0, sidx] = n
            sim.length[0, sidx] = n
            sim.alive[0, sidx] = True
            sim.direction[0, sidx] = direction[sidx]
            sim._reward_prev_length[0, sidx] = n
            occupied.update(cells)
        keep = [c for c in sim.food_cells[0] if c not in occupied]
        sim.food_cells[0][:] = keep
        sim.food_set[0].intersection_update(keep)
        sim.corpse_cells[0].intersection_update(keep)
        if isinstance(sim, GridBatchSim):
            sim._rebuild_env_grids(0)
        sim._rebuild_traversed_from_heads(np.ones(1, dtype=bool))
        sim._refresh_action_masks(np.ones(1, dtype=bool))


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("gap,len_b", [(0, 3), (0, 8), (1, 3), (1, 8), (2, 6)])
@pytest.mark.parametrize("boost", [0, 3])
def test_head_on_and_head_swap(version, gap, len_b, boost):
    """Facing snakes: head swap (gap 0), shared cell (gap 1), boost crossings."""
    cfg = BatchSimConfig(
        num_envs=1, num_snakes=2, game_width=300, game_height=200, mechanics_version=version
    )
    a, b = _pair(cfg, [4], True, False)
    row = 10
    head_a, head_b = 12, 13 + gap
    body_a = [(head_a - k, row) for k in range(6)]  # heading right
    body_b = [(head_b + k, row) for k in range(len_b)]  # heading left
    _place([a, b], [body_a, body_b], [1, 3])
    acts = np.array([[1 + boost, 1]], dtype=np.int64)
    _lockstep(a, b, 1, lambda sim: acts)
    assert a.get_done().any() or (gap == 2 and not boost)


@pytest.mark.parametrize("seed", range(6))
def test_bulk_corpse_drop_equals_sequential_adds(seed):
    """``_drop_corpse_v2`` == per-pellet ``_add_food`` incl. evict-then-re-add."""
    cfg = BatchSimConfig(
        num_envs=1, num_snakes=2, game_width=200, game_height=200, initial_food=30, max_food=24
    )
    a = GridBatchSim(cfg, seeds=[seed])
    b = GridBatchSim(cfg, seeds=[seed])
    gen = np.random.default_rng(seed)
    for _round in range(4):
        cells = [tuple(int(v) for v in gen.integers(0, 20, size=2)) for _ in range(40)]
        cells += cells[:10]  # duplicates / re-adds after eviction
        for c in cells:
            a._add_food(0, c, corpse=True)
        b._drop_corpse_v2(0, cells)
        assert a.food_cells[0] == b.food_cells[0]
        assert a.corpse_cells[0] == b.corpse_cells[0]
        assert a.food_set[0] == b.food_set[0]
        np.testing.assert_array_equal(a._food, b._food)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("respawn", [False, True])
def test_paused_worlds_with_grown_and_dead_snakes(version, respawn):
    """Worlds paused mid-episode (long + dead snakes) resume without a reset."""
    cfg = BatchSimConfig(
        num_envs=3,
        num_snakes=6,
        game_width=400,
        game_height=300,
        initial_food=60,
        max_food=80,
        mechanics_version=version,
        max_capacity=1200,
        frame_rate=3,
    )
    a, b = _pair(cfg, [31, 32, 33], True, respawn)
    pol = _survivor(3, 6)
    _lockstep(a, b, 150, pol, check_every=1)  # grow and kill some snakes
    assert a.length.max() > 3
    for f in range(300):
        active = np.array([True, (f // 7) % 2 == 0, f % 3 != 0])
        acts = pol(a)
        a.step(acts, active_env_mask=active)
        b.step(acts, active_env_mask=active)
        assert_same_world(a, b, f)
        b.check_grid_invariants()


def test_step_with_policy_and_active_mask():
    """The Watch-phase entry point (``step_with_policy``) stays in lockstep."""
    cfg = BatchSimConfig(
        num_envs=2, num_snakes=5, game_width=300, game_height=200, initial_food=40, max_food=50
    )
    a, b = _pair(cfg, [41, 42], False, True)
    pol_a, pol_b = _survivor(2, 5), _survivor(2, 5)
    for f in range(200):
        active = np.array([True, f % 4 != 0])
        a.step_with_policy(lambda s: pol_a(s), active_env_mask=active)
        b.step_with_policy(lambda s: pol_b(s), active_env_mask=active)
        assert_same_world(a, b, f)
        b.check_grid_invariants()


def test_rejects_tiny_boost_length():
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, min_boost_length=2)
    with pytest.raises(ValueError):
        GridBatchSim(cfg)
