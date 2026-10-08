"""Ego raster draft (redesign M1): vectorized grid gathers vs a slow list reference.

The reference never touches the simulator grids: it reads ordered bodies and the
food list through the BatchSim accessors and rotates with a hand-written
heading table, so it independently checks the grid bookkeeping, the gather, and
the rotation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego_raster import (
    EgoGridView,
    EgoRasterConfig,
    _reach_time,
    build_ego_raster,
)
from src.simd_env.grid_scenarios import GreedySafePolicy, inject_serpentines
from src.simd_env.grid_sim import GridBatchSim

# heading -> (ahead vector, right vector) in screen space (x right, y down).
_FRAME = {0: ((0, -1), (1, 0)), 1: ((1, 0), (0, 1)), 2: ((0, 1), (-1, 0)), 3: ((-1, 0), (0, -1))}


def _world(sim, e, s, r, c, head_row, head_col, scale, origin):
    ahead, right = _FRAME[int(sim.direction[e, s])]
    a, lat = (head_row - r) * scale, (c - head_col) * scale
    return (origin[0] + a * ahead[0] + lat * right[0], origin[1] + a * ahead[1] + lat * right[1])


def _reference_local(sim, e, s, cfg):
    occ = {}
    for j in range(sim.S):
        if not sim.alive[e, j]:
            continue
        for k, cell in enumerate(sim.get_bodies(e, j)):
            occ[cell] = (j, k)
    corpse = sim.corpse_cells[e]
    food = set(sim.food_cells[e])
    head = sim.get_bodies(e, s)[0]
    size = cfg.local_size
    out = np.zeros((6, size, size), dtype=np.int64)
    ttl_grid = np.zeros((size, size), dtype=np.int64)
    wall = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            w = _world(sim, e, s, r, c, cfg.local_head_row, cfg.local_head_col, 1, head)
            if not (0 <= w[0] < sim.grid_w and 0 <= w[1] < sim.grid_h):
                out[4, r, c] = 255
                wall[r, c] = True
                continue
            if w in occ:
                j, k = occ[w]
                ttl = min(max(int(sim.length[e, j]) - k, 0), 255)
                ttl_grid[r, c] = ttl
                out[0 if j == s else 1, r, c] = ttl
                if j != s and k == 0:
                    ratio = min(max(sim.length[e, j] / max(sim.length[e, s], 1), 0.0), 2.0) / 2.0
                    out[2, r, c] = int(ratio * 254.0) + 1
            if w in food:
                out[3, r, c] = 255 if w in corpse else 128
    out[5] = _reference_reach(ttl_grid, wall, cfg)
    return out


def _reference_reach(ttl_grid, wall, cfg):
    """Time-aware flood (first arrival), the slow way: every step, every reached cell.

    No early stop: a step at which nothing is gained does not end the flood, because a
    blocked neighbour can still open when its segment vacates later.
    """
    size = cfg.local_size
    out = np.zeros((size, size), dtype=np.int64)
    seen = {(cfg.local_head_row, cfg.local_head_col): 0}
    for t in range(1, cfg.reach_steps + 1):
        nxt = set()
        for r, c in list(seen):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < size and 0 <= cc < size) or (rr, cc) in seen:
                    continue
                if wall[rr, cc] or ttl_grid[rr, cc] > t:
                    continue
                nxt.add((rr, cc))
        for p in nxt:
            seen[p] = t
            out[p[0], p[1]] = t
    return out


def _reference_global(sim, e, s, cfg):
    cell, size = cfg.global_cell, cfg.global_size
    hc, wc = -(-sim.grid_h // cell), -(-sim.grid_w // cell)
    own = np.zeros((hc, wc), dtype=np.int64)
    enemy = np.zeros((hc, wc), dtype=np.int64)
    food = np.zeros((hc, wc), dtype=np.int64)
    for j in range(sim.S):
        if not sim.alive[e, j]:
            continue
        for col, row in sim.get_bodies(e, j):
            (own if j == s else enemy)[row // cell, col // cell] += 1
    for col, row in sim.food_cells[e]:
        food[row // cell, col // cell] += 1
    hx, hy = sim.get_bodies(e, s)[0]
    origin = (hx // cell, hy // cell)
    out = np.zeros((4, size, size), dtype=np.int64)
    for r in range(size):
        for c in range(size):
            w = _world(sim, e, s, r, c, size // 2, size // 2, 1, origin)
            if not (0 <= w[0] < wc and 0 <= w[1] < hc):
                out[3, r, c] = 255
                continue
            out[0, r, c] = min(own[w[1], w[0]], 255)
            out[1, r, c] = min(enemy[w[1], w[0]], 255)
            out[2, r, c] = min(food[w[1], w[0]], 255)
    return out


@pytest.fixture(scope="module")
def played_sim():
    cfg = BatchSimConfig(num_envs=2, num_snakes=5, mechanics_version=2, max_capacity=1200)
    sim = GridBatchSim(cfg, seeds=[8, 9], train_mode=True, allow_respawn=True)
    inject_serpentines([sim], [700, 300, 120, 60, 20], box_height=8)
    pol = GreedySafePolicy(4, boost_prob=0.3)
    for _ in range(80):
        sim.step(pol.actions(sim, sim.get_action_mask()))
    return sim


def test_shapes_and_dtypes(played_sim):
    obs = build_ego_raster(played_sim)
    E, S = played_sim.E, played_sim.S
    assert obs["local"].shape == (E, S, 6, 31, 31) and obs["local"].dtype == np.uint8
    assert obs["global"].shape == (E, S, 4, 37, 37) and obs["global"].dtype == np.uint8
    assert obs["scalars"].shape == (E, S, 12) and obs["scalars"].dtype == np.float32


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_local_planes_match_reference(played_sim, backend):
    cfg = EgoRasterConfig()
    obs = build_ego_raster(played_sim, cfg, backend=backend)
    checked = 0
    for e in range(played_sim.E):
        for s in range(played_sim.S):
            if not played_sim.alive[e, s]:
                assert not obs["local"][e, s].any()
                continue
            ref = _reference_local(played_sim, e, s, cfg)
            np.testing.assert_array_equal(obs["local"][e, s], ref, err_msg=f"env {e} snake {s}")
            checked += 1
    assert checked >= 4


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_global_planes_match_reference(played_sim, backend):
    cfg = EgoRasterConfig()
    obs = build_ego_raster(played_sim, cfg, backend=backend)
    for e in range(played_sim.E):
        for s in range(played_sim.S):
            if played_sim.alive[e, s]:
                ref = _reference_global(played_sim, e, s, cfg)
                np.testing.assert_array_equal(obs["global"][e, s], ref)


def test_big_body_is_visible_beyond_vector61_caps(played_sim):
    """Uncapped length scalar and own mass far outside the local window."""
    obs = build_ego_raster(played_sim)
    lengths = played_sim.length
    e, s = np.unravel_index(np.argmax(np.where(played_sim.alive, lengths, 0)), lengths.shape)
    assert lengths[e, s] > 150
    assert obs["scalars"][e, s, 1] == pytest.approx(lengths[e, s] / 1000.0)
    assert int(obs["global"][e, s, 0].sum()) == int(played_sim.seg_count[e, s])


def test_rotation_is_exact_permutation():
    """Turning the same world 4 ways rotates the local planes by exact rot90."""
    cfg = BatchSimConfig(num_envs=1, num_snakes=1, game_width=400, game_height=400)
    sim = GridBatchSim(cfg, seeds=[1])
    head = (20, 20)
    body = [head, (19, 20), (18, 20), (18, 21), (18, 22)]
    sim.bodies[0, 0] = 0
    sim.bodies[0, 0, : len(body)] = np.array(body[::-1])
    sim.head_ptr[0, 0] = len(body) - 1
    sim.seg_count[0, 0] = sim.length[0, 0] = len(body)
    sim._rebuild_env_grids(0)
    planes = []
    for d in range(4):
        sim.direction[0, 0] = d
        planes.append(build_ego_raster(sim, EgoRasterConfig(reach_steps=0))["local"][0, 0, 0])
    # Heading the other way turns the world: rotations relate the four views
    # around the head pixel (23, 15) — compare the centred 15x15 patch.
    crop = [p[16:31, 8:23] for p in planes]
    for d in range(4):
        assert np.array_equal(crop[(d + 1) % 4], np.rot90(crop[d], k=1)) or np.array_equal(
            crop[(d + 1) % 4], np.rot90(crop[d], k=-1)
        )


@pytest.mark.parametrize("heading", [0, 1, 2, 3])
def test_scalar_wall_distances_and_boost_phase(heading):
    """Ego wall distances rotate with the heading; boost phase uses the config cadence."""
    cfg = BatchSimConfig(num_envs=1, num_snakes=1, game_width=400, game_height=300)
    sim = GridBatchSim(cfg, seeds=[2])
    sim.bodies[0, 0, 0] = (10, 7)  # 29 cells to the right edge, 22 below
    sim.head_ptr[0, 0] = 0
    sim.seg_count[0, 0] = sim.length[0, 0] = 1
    sim.direction[0, 0] = heading
    sim.boost_frames[0, 0] = 2
    sim._rebuild_env_grids(0)
    sc = build_ego_raster(sim)["scalars"][0, 0]
    edge = [7, 29, 22, 10]  # up, right, down, left
    expect = [edge[(heading + j) % 4] / 145.0 for j in range(4)]
    np.testing.assert_allclose(sc[6:10], expect, rtol=1e-6)
    assert sc[4] == pytest.approx(2 / cfg.boost_length_cost_frames)


# ---------------------------------------------------------------------------
# numba backend == NumPy reference, bitwise
# ---------------------------------------------------------------------------
_KEYS = ("local", "global", "scalars", "mask")


def _assert_backends_equal(source, cfg=EgoRasterConfig(), rows=None, where=""):
    a = build_ego_raster(source, cfg, backend="numpy", rows=rows)
    b = build_ego_raster(source, cfg, backend="numba", rows=rows)
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, (where, key)
        assert np.array_equal(a[key], b[key]), (where, key, np.argwhere(a[key] != b[key])[:5])
    return a


@pytest.mark.parametrize("mechanics", [1, 2])
def test_numba_equals_numpy_over_played_big_body_worlds(mechanics):
    """Every frame of a big-body game: deaths, respawns, boosts, corpses, dead rows."""
    cfg = BatchSimConfig(num_envs=3, num_snakes=6, mechanics_version=mechanics, max_capacity=1200)
    sim = GridBatchSim(cfg, seeds=[21, 22, 23], train_mode=True, allow_respawn=True)
    inject_serpentines([sim], [800, 400, 150, 60, 20, 5], box_height=8)
    pol = GreedySafePolicy(7, boost_prob=0.3)
    dead_rows = deaths = 0
    for frame in range(150):
        sim.step(pol.actions(sim, sim.get_action_mask()))
        deaths += int(sim.get_done().sum())
        dead_rows += int((~sim.alive).sum())
        _assert_backends_equal(sim, where=f"frame {frame}")
    assert deaths > 0 and dead_rows > 0


def test_numba_equals_numpy_on_rows_subset_and_paused_worlds():
    """Hero-only rows (repeated envs, a dead row last) and a paused world."""
    cfg = BatchSimConfig(num_envs=4, num_snakes=4, mechanics_version=2, max_capacity=900)
    sim = GridBatchSim(cfg, seeds=[1, 2, 3, 4], train_mode=True, allow_respawn=False)
    inject_serpentines([sim], [600, 200, 50, 10], box_height=8)
    pol = GreedySafePolicy(3, boost_prob=0.2)
    active = np.ones(sim.E, dtype=bool)
    for frame in range(60):
        if frame == 20:
            active = np.array([True, False, True, True])
        sim.step(pol.actions(sim, sim.get_action_mask()), active_env_mask=active)
        rows = np.array([[e, 0] for e in range(sim.E)] + [[1, 2], [1, 3], [3, 3]])
        full = _assert_backends_equal(sim, where=f"frame {frame}")
        sub = _assert_backends_equal(sim, rows=rows, where=f"rows frame {frame}")
        for key in ("local", "global", "scalars", "mask"):
            np.testing.assert_array_equal(sub[key], full[key][rows[:, 0], rows[:, 1]])


def test_numba_equals_numpy_reach_variants():
    """Reach off, short and long floods, and a moved head anchor."""
    cfg = BatchSimConfig(num_envs=2, num_snakes=5, mechanics_version=2, max_capacity=1200)
    sim = GridBatchSim(cfg, seeds=[5, 6], train_mode=True, allow_respawn=True)
    inject_serpentines([sim], [700, 300, 120, 60, 20], box_height=6)
    pol = GreedySafePolicy(9, boost_prob=0.3)
    for _ in range(40):
        sim.step(pol.actions(sim, sim.get_action_mask()))
    for ecfg in (
        EgoRasterConfig(reach_steps=0),
        EgoRasterConfig(reach_steps=1),
        EgoRasterConfig(reach_steps=60),
        EgoRasterConfig(local_head_row=15, reach_steps=24),
    ):
        _assert_backends_equal(sim, ecfg, where=str(ecfg))


def _random_view(rng, E=3, S=5, H=40, W=50, pad=24, cap=64):
    """A synthetic, internally consistent-enough view with dense random occupancy."""
    shape = (E, H + 2 * pad, W + 2 * pad)
    owner = np.full(shape, -1, dtype=np.int16)
    slot = np.zeros(shape, dtype=np.int32)
    food = np.full(shape, -1, dtype=np.int8)
    inner = (slice(None), slice(pad, pad + H), slice(pad, pad + W))
    food[inner] = rng.choice([0, 0, 0, 1, 2], size=(E, H, W)).astype(np.int8)
    occ = rng.random((E, H, W)) < 0.45
    owner[inner] = np.where(occ, rng.integers(0, S, size=(E, H, W)), -1).astype(np.int16)
    slot[inner] = rng.integers(0, cap, size=(E, H, W)).astype(np.int32)
    alive = rng.random((E, S)) < 0.85
    owner[inner] = np.where(
        alive[np.arange(E)[:, None, None], np.clip(owner[inner], 0, None)] | (owner[inner] < 0),
        owner[inner],
        -1,
    )
    heads = np.stack([rng.integers(0, W, (E, S)), rng.integers(0, H, (E, S))], -1)
    return EgoGridView(
        owner_pad=owner,
        slot_pad=slot,
        food_pad=food,
        head_ptr=rng.integers(0, cap, (E, S)).astype(np.int64),
        seg_count=rng.integers(1, 40, (E, S)).astype(np.int64),
        length=rng.integers(1, 300, (E, S)).astype(np.int64),
        alive=alive,
        direction=rng.integers(0, 4, (E, S)).astype(np.int64),
        heads=heads.astype(np.int64),
        boost_frames=rng.integers(0, 3, (E, S)).astype(np.int64),
        frames_since_food=rng.integers(0, 900, (E, S)).astype(np.int64),
        grid_w=W,
        grid_h=H,
        cap=cap,
        pad=pad,
        boost_length_cost_frames=3,
    )


@pytest.mark.parametrize("seed", range(6))
def test_numba_equals_numpy_on_random_views(seed):
    """Dense random occupancy stresses TTL clamps, the size ratio and the flood."""
    view = _random_view(np.random.default_rng(seed))
    _assert_backends_equal(view, where=f"seed {seed}")
    _assert_backends_equal(view, EgoRasterConfig(reach_steps=40), where=f"seed {seed} K40")


def test_reach_matches_slow_reference_on_random_ttl():
    """The vectorized flood equals the slow per-agent reference on random grids."""
    rng = np.random.default_rng(11)
    cfg = EgoRasterConfig()
    ttl = rng.integers(0, 40, size=(4, 3, 31, 31)) * (rng.random((4, 3, 31, 31)) < 0.6)
    wall = rng.random((4, 3, 31, 31)) < 0.05
    wall[:, :, cfg.local_head_row, cfg.local_head_col] = False
    got = _reach_time(ttl, wall, cfg)
    for e in range(4):
        for s in range(3):
            np.testing.assert_array_equal(got[e, s], _reference_reach(ttl[e, s], wall[e, s], cfg))


def test_reach_waits_for_a_vacating_tail_and_is_batch_independent():
    """A head boxed in by segments that vacate at t=5 is reached later, not cut off.

    The first draft stopped the flood for the whole batch at the first step with no new
    cell anywhere, so this agent's plane was empty when featurized alone and non-empty
    when another agent in its batch kept the flood alive.
    """
    cfg = EgoRasterConfig()
    hr, hc = cfg.local_head_row, cfg.local_head_col
    boxed = np.zeros((31, 31), dtype=np.int64)
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        boxed[hr + dr, hc + dc] = 5
    free = np.zeros((31, 31), dtype=np.int64)
    no_wall = np.zeros((1, 2, 31, 31), dtype=bool)
    alone = _reach_time(boxed[None, None], no_wall[:, :1], cfg)[0, 0]
    batched = _reach_time(np.stack([boxed, free])[None], no_wall, cfg)[0, 0]
    np.testing.assert_array_equal(alone, batched)
    assert alone[hr - 1, hc] == 5 and alone[hr - 2, hc] == 6
    np.testing.assert_array_equal(alone, _reference_reach(boxed, no_wall[0, 0], cfg))


# ---------------------------------------------------------------------------
# ego2s-b (M2b): enemy next-cell channel, enemy / region scalars
# ---------------------------------------------------------------------------
_B = EgoRasterConfig(version="b")


def _reference_enemy_next(sim, e, s, cfg):
    """Slow per-pixel reference (world mapping through the hand-written frame table)."""
    from src.simd_env.batch_sim import CARDINAL

    head = sim.get_bodies(e, s)[0]
    boosting = sim.get_boosted_this_step() & (sim.length > 1)  # live clears it on respawn
    marks = {}
    for j in range(sim.S):
        if j == s or not sim.alive[e, j]:
            continue
        hx, hy = sim.get_bodies(e, j)[0]
        dx, dy = (int(v) for v in CARDINAL[int(sim.direction[e, j])])
        steps = (1, 2) if boosting[e, j] else (1,)
        for k in steps:
            c = (hx + k * dx, hy + k * dy)
            if 0 <= c[0] < sim.grid_w and 0 <= c[1] < sim.grid_h:
                marks[c] = max(marks.get(c, 0), 255 if boosting[e, j] else 128)
    size = cfg.local_size
    out = np.zeros((size, size), dtype=np.int64)
    for r in range(size):
        for c in range(size):
            w = _world(sim, e, s, r, c, cfg.local_head_row, cfg.local_head_col, 1, head)
            out[r, c] = marks.get(w, 0)
    return out


@pytest.mark.parametrize("mechanics", [1, 2])
def test_b_numba_equals_numpy_and_reference(mechanics):
    cfg = BatchSimConfig(num_envs=3, num_snakes=6, mechanics_version=mechanics, max_capacity=1200)
    sim = GridBatchSim(cfg, seeds=[31, 32, 33], train_mode=True, allow_respawn=True)
    inject_serpentines([sim], [800, 400, 150, 60, 20, 5], box_height=8)
    pol = GreedySafePolicy(5, boost_prob=0.4)
    marked = boosting_marks = 0
    for frame in range(120):
        sim.step(pol.actions(sim, sim.get_action_mask()))
        a = _assert_backends_equal(sim, _B, where=f"b frame {frame}")
        assert a["local"].shape[-3] == 7 and a["scalars"].shape[-1] == 30
        np.testing.assert_array_equal(a["local"][..., :6, :, :], build_ego_raster(sim)["local"])
        if frame % 20 == 0:
            for e in range(sim.E):
                for s in range(sim.S):
                    if sim.alive[e, s]:
                        ref = _reference_enemy_next(sim, e, s, _B)
                        np.testing.assert_array_equal(a["local"][e, s, 6], ref)
        marked += int((a["local"][:, :, 6] > 0).any(axis=(2, 3)).sum())
        boosting_marks += int((a["local"][:, :, 6] == 255).any(axis=(2, 3)).sum())
    assert marked > 0 and boosting_marks > 0


def test_b_scalars_reference_on_a_hand_built_world():
    """Nearest enemy geometry, heading, kill flag and uncapped tail-aware regions."""
    from src.simd_env.ego_raster_b import B_SCALAR_NAMES

    cfg = BatchSimConfig(num_envs=1, num_snakes=3, game_width=200, game_height=200)
    sim = GridBatchSim(cfg, seeds=[4], train_mode=True)

    def place(s, cells, direction):
        sim.bodies[0, s] = 0
        sim.bodies[0, s, : len(cells)] = np.array(cells[::-1])
        sim.head_ptr[0, s] = len(cells) - 1
        sim.seg_count[0, s] = sim.length[0, s] = len(cells)
        sim.alive[0, s] = True
        sim.direction[0, s] = direction

    # Observer 0 heading right at (5, 10); a wall of snake 1 at column 7 rows 0..19
    # splits the 20x20 arena; snake 2 is two cells ahead-left, heading down.
    place(0, [(5, 10), (4, 10), (3, 10)], 1)
    place(1, [(7, y) for y in range(0, 20)], 2)
    sim.length[0, 1] = 21  # pending growth: its tail (7, 19) has TTL 2, so the wall is closed
    place(2, [(6, 8), (6, 7)], 2)
    sim._rebuild_env_grids(0)
    sim._boosted_this_step[...] = False
    out = build_ego_raster(sim, _B, backend="numpy")
    out_nb = build_ego_raster(sim, _B, backend="numba")
    np.testing.assert_array_equal(out["scalars"], out_nb["scalars"])
    sc = dict(zip(B_SCALAR_NAMES, out["scalars"][0, 0, 12:]))
    # Nearest enemy = snake 2 head (6, 8): delta (1, -2) -> ahead 1, right -2 (right = down).
    assert sc["nearest_enemy_ahead"] == pytest.approx(1 / 145)
    assert sc["nearest_enemy_right"] == pytest.approx(-2 / 145)
    assert sc["nearest_enemy_heading_right"] == 1.0  # heading down == my right
    assert sc["kill_opportunity"] == 1.0  # its next cell (6, 9) is 1.4 cells away
    # Regions: snake 1's column wall leaves columns 0..6 west of it; snake 2 and own
    # body inside. Straight (6, 10) and left (5, 9) / right (5, 11) are all west.
    # Tail-aware: TTL-1 tails are free: snake 2's (6, 7) and the observer's (3, 10).
    blocked = {(7, y) for y in range(20)} | {(6, 8), (5, 10), (4, 10)}
    west = sum(1 for x in range(7) for y in range(20) if (x, y) not in blocked)
    assert sc["region_frac_straight"] == pytest.approx(min(west / 3, 4) / 4)
    assert sc["region_log_left"] == pytest.approx(np.log1p(west) / np.log1p(400), rel=1e-6)
    # Opening the wall's tail (no pending growth: TTL 1) joins the two halves.
    sim.length[0, 1] = 20
    sc2 = build_ego_raster(sim, _B, backend="numpy")["scalars"][0, 0, 12:]
    whole = 400 - (len(blocked) - 1)  # every cell but the blocked ones (tail now free)
    assert sc2[B_SCALAR_NAMES.index("region_log_left")] == pytest.approx(
        np.log1p(whole) / np.log1p(400), rel=1e-6
    )


def test_b_boosting_flag_clears_on_respawn_like_live():
    """A snake that boosted into a wall and respawned is not 'boosting' in ego2s-b."""
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, game_width=200, game_height=200)
    sim = GridBatchSim(cfg, seeds=[9], train_mode=False, allow_respawn=True)
    cells = [(17, 5), (16, 5), (15, 5), (14, 5), (13, 5), (12, 5), (11, 5)]
    sim.bodies[0, 1] = 0
    sim.bodies[0, 1, : len(cells)] = np.array(cells[::-1])
    sim.head_ptr[0, 1] = len(cells) - 1
    sim.seg_count[0, 1] = sim.length[0, 1] = len(cells)
    sim.direction[0, 1] = 1  # heading right, toward the wall at x = 20
    sim._rebuild_env_grids(0)
    flags = []

    def act(prepared):
        # The decision point: respawns are done, the last move's flags are still set.
        view = prepared.ego_view()
        raw = bool(prepared.get_boosted_this_step()[0, 1])
        flags.append((int(prepared.length[0, 1]), raw, bool(view.boosting[0, 1])))
        return np.array([[1, 4]])  # snake 1 boosts straight into the wall

    for _ in range(8):
        sim.step_with_policy(act)
    # The scenario really has a respawned snake whose BatchSim flag is still set ...
    assert any(length == 1 and raw for length, raw, _ in flags)
    # ... and ego2s-b reports it as not boosting, as the live game does.
    assert all(not b for length, _, b in flags if length == 1)


@pytest.mark.parametrize("seed", range(8))
def test_b_region_kernel_equals_reference_on_dense_random_views(seed):
    """Run-length union-find (numba) == cell BFS (Python) on random dense occupancy."""
    import dataclasses

    from src.simd_env.ego_raster_b import region_sizes

    rng = np.random.default_rng(100 + seed)
    view = _random_view(rng)
    view = dataclasses.replace(view, boosting=rng.random(view.alive.shape) < 0.3)
    rows = np.argwhere(view.alive)
    a = region_sizes(view, rows, "numpy")
    b = region_sizes(view, rows, "numba")
    np.testing.assert_array_equal(a, b)
    assert (a > 0).any() and (a == 0).any() and len(np.unique(a)) > 3
    _assert_backends_equal(view, _B, where=f"b random {seed}")


@pytest.mark.parametrize("seed", range(6))
def test_b_region_kernel_word_path_equals_reference(seed):
    """Aligned padded rows (W + 2 * pad multiple of 8) exercise the 4-cell word skips."""
    import dataclasses

    from src.simd_env.ego_raster_b import region_sizes
    from src.simd_env.ego_raster_nb import word_views

    rng = np.random.default_rng(300 + seed)
    view = _random_view(rng, W=48, H=33)
    assert word_views(view.owner_pad, view.food_pad) is not None
    sparse = rng.random(view.owner_pad.shape) < 0.6  # open regions -> long empty runs
    owner = np.where(sparse, -1, view.owner_pad).astype(np.int16)
    view = dataclasses.replace(
        view, owner_pad=owner, boosting=np.zeros(view.alive.shape, dtype=bool)
    )
    rows = np.argwhere(view.alive)
    np.testing.assert_array_equal(
        region_sizes(view, rows, "numpy"), region_sizes(view, rows, "numba")
    )
