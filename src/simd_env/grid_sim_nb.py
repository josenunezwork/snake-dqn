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


@njit(cache=True, nogil=True, boundscheck=True)
def corpse_queue_compact(
    old, old_n, old_corpse, incoming, incoming_n, cap, out, out_corpse, food, coarse
):
    """Ordered corpse queue with O(1) live-grid membership and exact evictions."""
    limit = old_n + incoming_n
    keep = np.ones(limit, dtype=np.bool_)
    cells = np.empty((limit, 2), dtype=np.int64)
    corpse = np.zeros(limit, dtype=np.bool_)
    for i in range(old_n):
        cells[i] = old[i]
        corpse[i] = old_corpse[i]
    n = old_n
    queue = np.empty(limit, dtype=np.int64)
    qhead = 0
    qtail = 0
    for i in range(old_n):
        if corpse[i]:
            queue[qtail] = i
            qtail += 1
    nc = qtail
    for j in range(incoming_n):
        x, y = incoming[j, 0], incoming[j, 1]
        if food[y, x] > 0:
            continue
        cells[n, 0] = x
        cells[n, 1] = y
        corpse[n] = True
        queue[qtail] = n
        qtail += 1
        n += 1
        nc += 1
        food[y, x] = 2
        coarse[y // 8, x // 8] += 1
        while nc > cap:
            victim = queue[qhead]
            qhead += 1
            keep[victim] = False
            vx, vy = cells[victim, 0], cells[victim, 1]
            food[vy, vx] = 0
            coarse[vy // 8, vx // 8] -= 1
            nc -= 1
    w = 0
    for i in range(n):
        if keep[i]:
            out[w] = cells[i]
            out_corpse[w] = corpse[i]
            w += 1
    return w


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
    active: np.ndarray,
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
        if not active[e]:
            continue
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
    out = sim._last_mask.copy()
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
        np.ascontiguousarray(sim._active_envs(), dtype=np.bool_),
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


@njit(cache=True, nogil=True)
def _write_new_heads_kernel(
    owner: np.ndarray,
    slot: np.ndarray,
    coarse: np.ndarray,
    trav: np.ndarray,
    trav_valid: np.ndarray,
    n_trav: np.ndarray,
    head_ptr: np.ndarray,
    cap: int,
    coarse_cell: int,
    grid_w: int,
    grid_h: int,
) -> None:
    """Ordered scalar equivalent of GridBatchSim's former ``ufunc.at`` writes.

    ``t`` remains outermost, matching the original vectorized traversal. Event
    environments are rebuilt immediately after the move, so only conflict-free
    rows survive; keeping this order still makes the transient bookkeeping
    exactly equivalent for diagnostics and non-event worlds.
    """
    E, S = n_trav.shape
    for t in range(2):
        for e in range(E):
            for s in range(S):
                if not trav_valid[e, s, t] or n_trav[e, s] <= 0:
                    continue
                x = trav[e, s, t, 0]
                y = trav[e, s, t, 1]
                if x < 0 or x >= grid_w or y < 0 or y >= grid_h:
                    continue
                block_r = y // coarse_cell
                block_c = x // coarse_cell
                old = owner[e, y, x]
                if old >= 0:
                    coarse[e, old, block_r, block_c] -= 1
                k = n_trav[e, s] - 1 - t
                ring = (head_ptr[e, s] - k) % cap
                owner[e, y, x] = s
                slot[e, y, x] = ring
                coarse[e, s, block_r, block_c] += 1


def write_new_heads(sim, n_trav: np.ndarray) -> None:
    """Compiled ordered head stamping, bitwise equal to the NumPy path."""
    _write_new_heads_kernel(
        sim._owner,
        sim._slot,
        sim._csnake,
        np.ascontiguousarray(sim._trav, dtype=np.int64),
        np.ascontiguousarray(sim._trav_valid, dtype=np.bool_),
        np.ascontiguousarray(n_trav, dtype=np.int64),
        np.ascontiguousarray(sim.head_ptr, dtype=np.int64),
        int(sim.cap),
        int(sim.COARSE_CELL) if hasattr(sim, "COARSE_CELL") else 8,
        int(sim.grid_w),
        int(sim.grid_h),
    )


@njit(cache=True, nogil=True)
def _pop_tails_kernel(owner, coarse, old_tail, popped, coarse_cell):
    """Ordered scalar equivalent of the three former coarse ``subtract.at`` calls."""
    E, S = popped.shape
    for p in range(3):
        for e in range(E):
            for s in range(S):
                if popped[e, s] <= p:
                    continue
                x = old_tail[p, e, s, 0]
                y = old_tail[p, e, s, 1]
                if owner[e, y, x] == s:
                    owner[e, y, x] = -1
                    coarse[e, s, y // coarse_cell, x // coarse_cell] -= 1


def pop_tails(sim, old_tail: np.ndarray, popped: np.ndarray) -> None:
    """Compiled grid-tail removal preserving the original p-major order."""
    _pop_tails_kernel(
        sim._owner,
        sim._csnake,
        np.ascontiguousarray(old_tail, dtype=np.int64),
        np.ascontiguousarray(popped, dtype=np.int64),
        8,
    )


@njit(cache=True, nogil=True)
def _consume_uncontested_kernel(
    food,
    coarse,
    ambient_count,
    length,
    trav,
    first,
    ate,
    fast_envs,
    ambient_ate,
    corpse_ate,
    food_array,
    food_n,
    food_version,
):
    """Numeric updates for unique train-mode food claims, preserving e/s order."""
    for e in range(ate.shape[0]):
        if not fast_envs[e]:
            continue
        for s in range(ate.shape[1]):
            if not ate[e, s]:
                continue
            t = first[e, s]
            x, y = trav[e, s, t, 0], trav[e, s, t, 1]
            is_corpse = food[e, y, x] == 2
            food[e, y, x] = 0
            coarse[e, y // 8, x // 8] -= 1
            if not is_corpse:
                ambient_count[e] -= 1
            length[e, s] += 1
            ambient_ate[e, s] = not is_corpse
            corpse_ate[e, s] = is_corpse

        write = 0
        for i in range(food_n[e]):
            x, y = food_array[e, i, 0], food_array[e, i, 1]
            if food[e, y, x] > 0:
                food_array[e, write, 0] = x
                food_array[e, write, 1] = y
                write += 1
        food_n[e] = write
        food_version[e] += 1


def consume_uncontested(sim, first, ate, fast_envs, ambient_ate, corpse_ate) -> None:
    """Commit noncontested numeric food updates without consuming RNG."""
    _consume_uncontested_kernel(
        sim._food,
        sim._cfood,
        sim._amb_count,
        sim.length,
        sim._trav,
        first,
        ate,
        fast_envs,
        ambient_ate,
        corpse_ate,
        sim._food_array,
        sim._food_n,
        sim._food_version,
    )
