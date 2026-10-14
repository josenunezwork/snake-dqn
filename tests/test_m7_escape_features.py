"""M7 study A candidate escape features (research/m7_feature_auc_20261014/escape_nb.py):
the numba kernels against a plain-Python reference (heap Dijkstra time-aware flood, BFS
components, frontier scan) on big-body serpentine worlds, and the static region size
against ego2s-b's ``region_sizes`` (the existing input it extends)."""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import pytest

from research.m7_feature_auc_20261014 import escape_nb as ef
from src.simd_env.batch_sim import CARDINAL, BatchSimConfig
from src.simd_env.ego_raster_b import region_sizes
from src.simd_env.grid_scenarios import GreedySafePolicy, inject_serpentines
from src.simd_env.grid_sim import GridBatchSim


def _world(sim, e):
    gw, gh = sim.grid_w, sim.grid_h
    ttl = np.zeros((gh, gw), np.int64)
    own = np.full((gh, gw), -1, np.int64)
    snake = np.full((gh, gw), -1, np.int64)
    for s in range(sim.S):
        if not sim.alive[e, s]:
            continue
        L = int(sim.length[e, s])
        for k in range(int(sim.seg_count[e, s])):
            x, y = (int(v) for v in sim.bodies[e, s, (sim.head_ptr[e, s] - k) % sim.cap])
            if not (0 <= x < gw and 0 <= y < gh):
                continue
            ttl[y, x] = max(ttl[y, x], L - k)
            if s == 0 and (own[y, x] < 0 or k < own[y, x]):
                own[y, x] = k
            if snake[y, x] < 0 or s == 0:
                snake[y, x] = s
    return ttl, own, snake


def _nbrs(x, y, gw, gh):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= x + dx < gw and 0 <= y + dy < gh:
            yield x + dx, y + dy


def _components(ttl):
    gh, gw = ttl.shape
    lab = np.full((gh, gw), -1, np.int64)
    sizes = []
    for y0 in range(gh):
        for x0 in range(gw):
            if ttl[y0, x0] > 1 or lab[y0, x0] >= 0:
                continue
            lab[y0, x0] = len(sizes)
            q, n = deque([(x0, y0)]), 0
            while q:
                x, y = q.popleft()
                n += 1
                for nx, ny in _nbrs(x, y, gw, gh):
                    if ttl[ny, nx] <= 1 and lab[ny, nx] < 0:
                        lab[ny, nx] = len(sizes)
                        q.append((nx, ny))
            sizes.append(n)
    return lab, sizes


def _arrival(ttl, sx, sy, tmax):
    gh, gw = ttl.shape
    arr = {(sx, sy): 1}
    heap = [(1, sx, sy)]
    done = set()
    while heap:
        a, x, y = heapq.heappop(heap)
        if (x, y) in done:
            continue
        done.add((x, y))
        for nx, ny in _nbrs(x, y, gw, gh):
            c = max(a + 1, int(ttl[ny, nx]))
            if c <= tmax and c < arr.get((nx, ny), 1 << 60):
                arr[(nx, ny)] = c
                heapq.heappush(heap, (c, nx, ny))
    return arr


def reference(sim, e):
    gw, gh = sim.grid_w, sim.grid_h
    ttl, own, snake = _world(sim, e)
    lab, sizes = _components(ttl)
    L = int(sim.length[e, 0])
    free = sum(sizes)
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))
    largest = order[0] if sizes else -1
    lsize = sizes[largest] if sizes else 0
    second = sizes[order[1]] if len(sizes) > 1 else 0
    hx, hy = (int(v) for v in sim.bodies[e, 0, sim.head_ptr[e, 0]])
    cells = [
        tuple(int(v) for v in sim.bodies[e, 0, (sim.head_ptr[e, 0] - k) % sim.cap])
        for k in range(int(sim.seg_count[e, 0]))
    ]
    xs, ys = [c[0] for c in cells], [c[1] for c in cells]
    state = [
        min(lsize / L, 16.0),
        lsize / max(free, 1),
        sum(s >= 32 for s in sizes),
        free / (gw * gh),
        second / max(lsize, 1),
        sum(abs(x - hx) <= 8 and abs(y - hy) <= 8 for x, y in cells) / 289.0,
        sum(abs(x - hx) <= 16 and abs(y - hy) <= 16 for x, y in cells) / 1089.0,
        L / ((max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)),
        int((snake >= 0).sum()) / (gw * gh),
    ]
    acts = []
    n_own = int((own >= 0).sum())
    for a in range(3):
        d = CARDINAL[(int(sim.direction[e, 0]) + a - 1) % 4]
        sx, sy = hx + int(d[0]), hy + int(d[1])
        if not (0 <= sx < gw and 0 <= sy < gh) or ttl[sy, sx] > 1:
            acts.append(list(ef.BLOCKED))
            continue
        la = lab[sy, sx]
        R = sizes[la]
        arr = _arrival(ttl, sx, sy, 2 * L)
        times = np.array(sorted(arr.values()))

        def N(t):
            return int(np.searchsorted(times, t, side="right"))

        tstar = next((t for t in range(1, L + 1) if N(t) < t), L + 1)
        own_arr = [t for (x, y), t in arr.items() if own[y, x] >= 0]
        lg = [t for (x, y), t in arr.items() if lab[y, x] == largest]
        q4, q2 = max(1, L // 4), max(1, L // 2)
        f_own, f_en, ttls = 0, 0, []
        for y in range(gh):
            for x in range(gw):
                if snake[y, x] < 0 or ttl[y, x] <= 1:
                    continue
                if not any(lab[ny, nx] == la for nx, ny in _nbrs(x, y, gw, gh)):
                    continue
                if own[y, x] >= 0:
                    f_own += 1
                    ttls.append(L - int(own[y, x]))
                else:
                    f_en += 1
        ttls = np.array(ttls)
        acts.append(
            [
                min(N(q4) / q4, 8.0),
                min(N(q2) / q2, 8.0),
                min(N(L) / L, 8.0),
                tstar / L,
                min(own_arr) / L if own_arr else 2.0,
                sum(t <= L for t in own_arr) / max(n_own, 1),
                min(lg) / L if lg else 2.0,
                f_own / L,
                float((2 * ttls <= R).mean()) if f_own else 0.0,
                float((ttls <= R).mean()) if f_own else 0.0,
                ttls.min() / L if f_own else 2.0,
                min(max((R - ttls.min()) / L, -2.0), 8.0) if f_own else 8.0,
                f_own / (f_own + f_en) if f_own + f_en else 0.5,
                1.0 if la == largest else 0.0,
            ]
        )
    return np.array(acts), np.array(state)


@pytest.fixture(scope="module")
def worlds():
    cfg = BatchSimConfig(
        num_envs=2, num_snakes=6, game_width=700, game_height=500, mechanics_version=2,
        max_capacity=1500,
    )  # fmt: skip
    sim = GridBatchSim(cfg, seeds=[5, 6], train_mode=False, allow_respawn=True, jit=True)
    inject_serpentines([sim], [300, 150, 60, 20, 0, 0], box_height=6)
    return sim


def test_kernel_matches_reference_on_big_bodies(worlds):
    sim = worlds
    pol = GreedySafePolicy(3, boost_prob=0.2)
    checked = 0
    for frame in range(120):
        if frame % 15 == 0:
            envs = np.flatnonzero(sim.alive[:, 0])
            got_a, got_s = ef.escape_features(sim, envs)
            for i, e in enumerate(envs):
                ref_a, ref_s = reference(sim, int(e))
                np.testing.assert_allclose(got_s[i], ref_s, rtol=0, atol=1e-12)
                np.testing.assert_allclose(got_a[i], ref_a, rtol=0, atol=1e-12)
                checked += 1
        sim.step(pol.actions(sim, sim.get_action_mask()))
    assert checked >= 8


def test_static_region_is_region_sizes(worlds):
    sim = worlds
    envs = np.flatnonzero(sim.alive[:, 0])
    rows = np.stack([envs, np.zeros_like(envs)], axis=1)
    R = region_sizes(sim.ego_view(), rows, "numba")
    for i, e in enumerate(envs):
        ttl, _, _ = _world(sim, int(e))
        lab, sizes = _components(ttl)
        hx, hy = (int(v) for v in sim.bodies[e, 0, sim.head_ptr[e, 0]])
        for a in range(3):
            d = CARDINAL[(int(sim.direction[e, 0]) + a - 1) % 4]
            sx, sy = hx + int(d[0]), hy + int(d[1])
            inside = 0 <= sx < sim.grid_w and 0 <= sy < sim.grid_h
            want = sizes[lab[sy, sx]] if inside and lab[sy, sx] >= 0 else 0
            assert int(R[i, a]) == want


def test_flood_counting_bound_on_a_pocket():
    # 7x5 arena; own body walls off a 4-cell pocket in the corner: the counting bound
    # fires in the pocket, and in the open part only once L exceeds its 26 cells.
    gw, gh = 7, 5
    ttl = np.zeros(gw * gh, np.int64)
    own = np.full(gw * gh, -1, np.int64)
    for k, (x, y) in enumerate([(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]):
        ttl[y * gw + x] = 50 - k
        own[y * gw + x] = k + 10
    labels = np.zeros(gw * gh, np.int64)
    sizes = np.zeros(gw * gh + 1, np.int64)
    queue = np.zeros(gw * gh, np.int64)
    nlab = ef._label(ttl, gw, gh, labels, sizes, queue)
    assert nlab == 2 and sorted(sizes[:2]) == [4, 26]
    arr, nxt = np.zeros(gw * gh, np.int64), np.zeros(gw * gh, np.int64)
    L = 20
    bucket, hist = np.zeros(2 * L + 2, np.int64), np.zeros(2 * L + 2, np.int64)
    out = np.zeros(5)
    largest = int(np.argmax(sizes[:2]))
    ef._flood(ttl, own, labels, largest, gw, gh, 0, 0, L, arr, nxt, bucket, hist, out)
    assert out[3] * L == 5  # 4 pocket cells; the walls stay shut until t >= 46 > 2L
    ef._flood(ttl, own, labels, largest, gw, gh, 5, 3, L, arr, nxt, bucket, hist, out)
    assert out[3] * L == L + 1  # 26 open cells, all within 8 steps: no bound within L
    L = 30
    bucket, hist = np.zeros(2 * L + 2, np.int64), np.zeros(2 * L + 2, np.int64)
    ef._flood(ttl, own, labels, largest, gw, gh, 5, 3, L, arr, nxt, bucket, hist, out)
    assert out[3] * L == 27  # a 30-long snake cannot spend 27 steps in 26 cells
