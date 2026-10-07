"""Ego raster draft (redesign M1): vectorized grid gathers vs a slow list reference.

The reference never touches the simulator grids: it reads ordered bodies and the
food list through the BatchSim accessors and rotates with a hand-written
heading table, so it independently checks the grid bookkeeping, the gather, and
the rotation.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
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
    # Time-aware BFS (first arrival), the slow way.
    seen = {(cfg.local_head_row, cfg.local_head_col): 0}
    q = deque([(cfg.local_head_row, cfg.local_head_col)])
    frontier = {(cfg.local_head_row, cfg.local_head_col)}
    for t in range(1, cfg.reach_steps + 1):
        nxt = set()
        for r, c in frontier | set(seen):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < size and 0 <= cc < size) or (rr, cc) in seen:
                    continue
                if wall[rr, cc] or ttl_grid[rr, cc] > t:
                    continue
                nxt.add((rr, cc))
        if not nxt:
            break
        for p in nxt:
            seen[p] = t
            out[5, p[0], p[1]] = t
        frontier = nxt
    del q
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


def test_local_planes_match_reference(played_sim):
    cfg = EgoRasterConfig()
    obs = build_ego_raster(played_sim, cfg)
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


def test_global_planes_match_reference(played_sim):
    cfg = EgoRasterConfig()
    obs = build_ego_raster(played_sim, cfg)
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
