"""Compiled (numba) kernels for :class:`~src.simd_env.grid_sim.GridBatchSim` hot paths.

Two per-step array computations of ``GridBatchSim`` are re-expressed as single-pass
loops (same integer logic, no float math), selected by ``GridBatchSim(jit=True)``:

- :func:`action_masks` == ``GridBatchSim._compute_action_masks_numpy`` (the 6-bit
  advisory mask from 3 x 2 candidate-cell gathers);
- :func:`detect_collisions` == ``GridBatchSim._detect_collisions_numpy`` (wall / self /
  head-on incl. head swaps / body detection on the post-move world).

Both NumPy versions stay in ``grid_sim.py`` as the reference. Bit-exact equality is
CI-pinned by running the whole ``tests/test_grid_sim_parity.py`` lockstep suite (every
frame vs ``BatchSim``: masks, deaths, kills, bodies, food, RNG) on both settings, plus a
direct kernel-vs-NumPy check on paused and big-body worlds. Single-threaded.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# CARDINAL of batch_sim (up, right, down, left) as (dx, dy).
_DX = np.array([0, 1, 0, -1], dtype=np.int64)
_DY = np.array([-1, 0, 1, 0], dtype=np.int64)


@njit(cache=True, nogil=True)
def _action_masks_kernel(
    owner_pad: np.ndarray,  # (E, Hp, Wp) int16
    slot_pad: np.ndarray,  # (E, Hp, Wp) int32
    pad: int,
    W: int,
    H: int,
    cap: int,
    head_ptr: np.ndarray,
    seg_count: np.ndarray,
    length: np.ndarray,
    alive: np.ndarray,
    direction: np.ndarray,
    heads: np.ndarray,  # (E, S, 2)
    boost_frames: np.ndarray,
    boost_cost: int,
    min_boost: int,
    dxs: np.ndarray,
    dys: np.ndarray,
    out: np.ndarray,  # (E, S, 6) bool
) -> None:
    E, S = alive.shape
    for e in range(E):
        for s in range(S):
            for a in range(6):
                out[e, s, a] = False
            if not alive[e, s]:
                continue
            n = seg_count[e, s]
            L = length[e, s]
            normal_last = min(n + 1, L) - 2
            l1 = min(n + 1, L)
            l2 = min(l1 + 1, L)
            if boost_frames[e, s] + 1 >= boost_cost:
                f_len = min(l2, max(1, L - 1))
            else:
                f_len = l2
            boost_last = f_len - 3
            can_boost = L >= min_boost
            hx = heads[e, s, 0]
            hy = heads[e, s, 1]
            for rel in range(3):
                d = (direction[e, s] + rel - 1) % 4
                fatal_normal = False
                fatal_boost = False
                for t in range(2):
                    cx = hx + dxs[d] * (t + 1)
                    cy = hy + dys[d] * (t + 1)
                    inb = cx >= 0 and cx < W and cy >= 0 and cy < H
                    own_n = False
                    own_b = False
                    other = False
                    if inb:
                        o = np.int64(owner_pad[e, cy + pad, cx + pad])
                        if o >= 0:
                            k = (head_ptr[e, o] - np.int64(slot_pad[e, cy + pad, cx + pad])) % cap
                            if o == s:
                                own_n = k >= 2 and k <= normal_last
                                own_b = k >= 1 and k <= boost_last
                            elif alive[e, o]:
                                on = seg_count[e, o]
                                if on >= 2 and on >= length[e, o]:
                                    other_last = on - 2
                                else:
                                    other_last = on - 1
                                other = k <= other_last
                    if t == 0:
                        fatal_normal = (not inb) or own_n or other
                    if (not inb) or own_b or other:
                        fatal_boost = True
                safe_normal = not fatal_normal
                out[e, s, rel] = safe_normal
                out[e, s, rel + 3] = safe_normal and can_boost and not fatal_boost


def action_masks(sim) -> np.ndarray:
    """``(E, S, 6)`` advisory masks of a :class:`GridBatchSim` (== the NumPy version)."""
    owner_pad, slot_pad, _ = sim.get_padded_grids()
    out = np.empty((sim.E, sim.S, 6), dtype=np.bool_)
    _action_masks_kernel(
        owner_pad,
        slot_pad,
        sim._grid_pad,
        int(sim.grid_w),
        int(sim.grid_h),
        int(sim.cap),
        np.ascontiguousarray(sim.head_ptr, dtype=np.int64),
        np.ascontiguousarray(sim.seg_count, dtype=np.int64),
        np.ascontiguousarray(sim.length, dtype=np.int64),
        np.ascontiguousarray(sim.alive, dtype=np.bool_),
        np.ascontiguousarray(sim.direction, dtype=np.int64),
        np.ascontiguousarray(sim.heads(), dtype=np.int64),
        np.ascontiguousarray(sim.boost_frames, dtype=np.int64),
        int(sim.cfg.boost_length_cost_frames),
        int(sim.cfg.min_boost_length),
        _DX,
        _DY,
        out,
    )
    return out


@njit(cache=True, nogil=True)
def _detect_kernel(
    owner_pad: np.ndarray,
    slot_pad: np.ndarray,
    pad: int,
    W: int,
    H: int,
    cap: int,
    active: np.ndarray,  # (E,)
    alive: np.ndarray,  # (E, S)
    n_trav: np.ndarray,  # (E, S)
    trav: np.ndarray,  # (E, S, 2, 2)
    trav_valid: np.ndarray,  # (E, S, 2)
    head_ptr: np.ndarray,
    seg_count: np.ndarray,
    direction: np.ndarray,
    dxs: np.ndarray,
    dys: np.ndarray,
    wall: np.ndarray,  # (E, S) out
    selfh: np.ndarray,  # (E, S) out
    head_on: np.ndarray,  # (E, S, S) out
    body: np.ndarray,  # (E, S, S) out
    event_env: np.ndarray,  # (E,) out
) -> None:
    E, S = alive.shape
    tv = np.zeros((S, 2), dtype=np.bool_)
    mover = np.zeros(S, dtype=np.bool_)
    sx = np.zeros((S, 2), dtype=np.int64)  # path edge starts (x, y)
    sy = np.zeros((S, 2), dtype=np.int64)
    ex = np.zeros((S, 2), dtype=np.int64)  # path edge ends
    ey = np.zeros((S, 2), dtype=np.int64)
    for e in range(E):
        event_env[e] = False
        for i in range(S):
            wall[e, i] = False
            selfh[e, i] = False
            for j in range(S):
                head_on[e, i, j] = False
                body[e, i, j] = False
        if not active[e]:
            continue
        for i in range(S):
            mover[i] = alive[e, i] and n_trav[e, i] > 0
            for t in range(2):
                tv[i, t] = trav_valid[e, i, t] and mover[i]
            # Edges: e0 = prev -> t1 (always), e1 = t1 -> t2 (boost only).
            d = direction[e, i]
            sx[i, 0] = trav[e, i, 0, 0] - dxs[d]
            sy[i, 0] = trav[e, i, 0, 1] - dys[d]
            ex[i, 0] = trav[e, i, 0, 0]
            ey[i, 0] = trav[e, i, 0, 1]
            sx[i, 1] = trav[e, i, 0, 0]
            sy[i, 1] = trav[e, i, 0, 1]
            ex[i, 1] = trav[e, i, 1, 0]
            ey[i, 1] = trav[e, i, 1, 1]
        for i in range(S):
            w = False
            self_cell = False
            for t in range(2):
                if not tv[i, t]:
                    continue
                cx = trav[e, i, t, 0]
                cy = trav[e, i, t, 1]
                if not (cx >= 0 and cx < W and cy >= 0 and cy < H):
                    w = True
                    continue
                o = np.int64(owner_pad[e, cy + pad, cx + pad])
                if o < 0:
                    continue
                if o == i:
                    k_post = (head_ptr[e, o] - np.int64(slot_pad[e, cy + pad, cx + pad])) % cap
                    if k_post >= 3 and seg_count[e, i] > 3:
                        self_cell = True
                else:
                    body[e, i, o] = True
            wall[e, i] = w
            selfh[e, i] = self_cell and not w
        for i in range(S):
            for j in range(S):
                if i == j:
                    continue
                hit = False
                for a in range(2):
                    if not tv[i, a]:
                        continue
                    for b in range(2):
                        if (
                            tv[j, b]
                            and trav[e, i, a, 0] == trav[e, j, b, 0]
                            and trav[e, i, a, 1] == trav[e, j, b, 1]
                        ):
                            hit = True
                for a in range(2):
                    if not tv[i, a]:
                        continue
                    for b in range(2):
                        if (
                            tv[j, b]
                            and sx[i, a] == ex[j, b]
                            and sy[i, a] == ey[j, b]
                            and sx[j, b] == ex[i, a]
                            and sy[j, b] == ey[i, a]
                        ):
                            hit = True
                ok = mover[i] and alive[e, j] and not (wall[e, i] or selfh[e, i])
                head_on[e, i, j] = hit and ok
                body[e, i, j] = body[e, i, j] and ok and not head_on[e, i, j]
        for i in range(S):
            ev = wall[e, i] or selfh[e, i]
            for j in range(S):
                ev = ev or head_on[e, i, j] or body[e, i, j]
            if ev:
                event_env[e] = True


def detect_collisions(sim, n_trav: np.ndarray) -> dict:
    """``GridBatchSim._detect_collisions_numpy`` as one compiled pass."""
    E, S = sim.E, sim.S
    owner_pad, slot_pad, _ = sim.get_padded_grids()
    out = {
        "wall": np.empty((E, S), dtype=np.bool_),
        "self": np.empty((E, S), dtype=np.bool_),
        "head_on": np.empty((E, S, S), dtype=np.bool_),
        "body": np.empty((E, S, S), dtype=np.bool_),
        "event_env": np.empty(E, dtype=np.bool_),
    }
    _detect_kernel(
        owner_pad,
        slot_pad,
        sim._grid_pad,
        int(sim.grid_w),
        int(sim.grid_h),
        int(sim.cap),
        np.ascontiguousarray(sim._active_envs(), dtype=np.bool_),
        np.ascontiguousarray(sim.alive, dtype=np.bool_),
        np.ascontiguousarray(n_trav, dtype=np.int64),
        np.ascontiguousarray(sim._trav, dtype=np.int64),
        np.ascontiguousarray(sim._trav_valid, dtype=np.bool_),
        np.ascontiguousarray(sim.head_ptr, dtype=np.int64),
        np.ascontiguousarray(sim.seg_count, dtype=np.int64),
        np.ascontiguousarray(sim.direction, dtype=np.int64),
        _DX,
        _DY,
        out["wall"],
        out["self"],
        out["head_on"],
        out["body"],
        out["event_env"],
    )
    return out
