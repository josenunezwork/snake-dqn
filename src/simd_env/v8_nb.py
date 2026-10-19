"""Compiled (numba) v8 decision-time safety veto for GridBatchSim / BatchSim hero rows.

Decision-identical to the reference Python path
(:meth:`src.simd_env.vector61_policy.Vector61SimdPolicy.actions` with
``veto_variant="v8"``: the live :class:`~src.evaluation.safety_veto_v8.SpaceAndHeadVeto`
run on :class:`~src.simd_env.vector61_policy.SimRowSnake` views), for the same inputs:

* ``q``: the ``(N, 6)`` float32 Q rows the policy computed (raw; masked here exactly as
  ``actions()`` does: ``where(mask, q, INVALID_Q_VALUE)`` in float32, base = the first
  maximum, i.e. ``torch.argmax`` on CPU);
* ``mask``: the hard mask the Python path uses, i.e. the vector61 runtime's selection
  (advisory) mask; the normal-only fallback for an all-False row is applied here, as
  ``actions()`` does.

Every step is re-expressed on the sim's ring buffers (no per-row Python):

1. v2 inputs: ``cap = min(160, max(32, 2L))``, ``need = min(max(1, L), cap)`` and the
   decision-time capped flood counts from the three next cells with every live body
   blocked (``Vector61Runtime.decision_free_space``; ``min(component, cap)`` is
   order-independent);
2. v5 ``boost_aware_choice`` with a lazy, memoized landing flood (post-boost body except
   the new head and other live snakes blocked, v2 cap);
3. v7 re-rank (``rerank_candidates`` + ``space_preference_choice``) with the tail-aware
   post-move ``landing_area`` (``tail_aware_reachable``: FIFO breadth-first search, the
   same neighbour order ``+x, -x, +y, -y`` and the same "seen only when passable" rule,
   so the visited set and the capped count are identical) and v7's ``area_cap``;
4. v8 head layer (v6 ``opponent_head_reach`` / ``head_risky_actions`` /
   ``head_avoid_choice`` then v7's score over the non-risky pool).

Floats: every score is the same IEEE-754 float64 expression in the same operation order
as the Python code (numba without ``fastmath`` never contracts ``a + b * c`` into an
FMA), on float64 values of the float32 masked Q, so ties and comparisons are bitwise
the Python ones. Tie-breaking is the Python one (strict ``>`` scans in ascending action
order; re-rank ties to the anchor, then the lowest index).

Diagnostics (``features=True`` evaluates them for every action; otherwise only what the
decision needed, ``-1`` = not evaluated): per action the capped tail-aware landing area
(and its ``g = min(area, cap) / cap``), the v6 head-risk and hero-wins waiver flags, the
v5 boost landing count / ok flag and v5 eligibility; per row the v2 counts, cap, need
and v7's area cap. They can be used as network inputs.

Not reproduced (bookkeeping only, never decision-relevant): the live hook's per-episode
counters (``veto_record`` / ``veto_diagnostics``) and the optional v8
``reference_lambda`` diagnostic. Rectangular arenas only (as GridBatchSim).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
from numba import get_thread_id, njit, prange

__all__ = [
    "V8Result",
    "v8_decide",
    "v8_feature_planes",
    "v8_redecide",
    "FEATURE_NAMES",
    "OUTCOME_KEPT",
    "OUTCOME_VETOED",
    "OUTCOME_NO_SPACIOUS",
]

FREE_SPACE_BFS_CAP = 160  # src.game.snake_state (asserted in _constants_check)
FREE_SPACE_MIN_CAP = 32
AREA_CAP_LIMIT = 4096
Q_SCALE_EPS = 1e-6
INVALID_Q_VALUE = -1.0e9

OUTCOME_KEPT = 0
OUTCOME_VETOED = 1
OUTCOME_NO_SPACIOUS = 2

# v5 reasons (diagnostics)
_R_KEPT = 0
_R_V2 = 1
_R_SAME_DIR = 2
_R_LANDING_V2 = 3
_R_LANDING_NONE = 4

_LANDING_UNKNOWN = -1
_LANDING_NO_BOOST = -2  # the boost would not fire (live ``landing_count`` -> None)

# CARDINAL of batch_sim (up, right, down, left) as (dx, dy).
_DX = np.array([0, 1, 0, -1], dtype=np.int64)
_DY = np.array([-1, 0, 1, 0], dtype=np.int64)
# tail_aware_reachable neighbour order: (+x), (-x), (+y), (-y).
_NBX = np.array([1, -1, 0, 0], dtype=np.int64)
_NBY = np.array([0, 0, 1, -1], dtype=np.int64)

FEATURE_NAMES = ("area_g", "head_risky", "landing_ok", "v5_eligible", "spacious")


# ----------------------------------------------------------------------------- kernels
@njit(cache=True, nogil=True)
def _bfs(
    sx,
    sy,
    gw,
    gh,
    blk1,
    st1,
    blk2,
    st2,
    rel_st,
    rel_val,
    rel_cur,
    use_rel,
    seen,
    seen_cur,
    qx,
    qy,
    qd,
    cap,
):
    """``safety_veto_v3.tail_aware_reachable`` from ``(sx, sy)`` at distance 0.

    Blocked = ``blk1 == st1`` or ``blk2 == st2``; release (``use_rel``): a cell with
    ``rel_st == rel_cur`` is passable only at distance ``>= rel_val``.
    """
    if sx < 0 or sx >= gw or sy < 0 or sy >= gh:
        return 0
    idx = sx * gh + sy
    if blk1[idx] == st1 or blk2[idx] == st2:
        return 0
    if use_rel and rel_st[idx] == rel_cur and 0 < rel_val[idx]:
        return 0
    seen[idx] = seen_cur
    head = 0
    tail = 1
    qx[0] = sx
    qy[0] = sy
    qd[0] = 0
    count = 0
    while head < tail and count < cap:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        count += 1
        for k in range(4):
            nx = x + _NBX[k]
            ny = y + _NBY[k]
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                continue
            nidx = nx * gh + ny
            if seen[nidx] == seen_cur:
                continue
            if blk1[nidx] == st1 or blk2[nidx] == st2:
                continue
            if use_rel and rel_st[nidx] == rel_cur and d + 1 < rel_val[nidx]:
                continue
            seen[nidx] = seen_cur
            qx[tail] = nx
            qy[tail] = ny
            qd[tail] = d + 1
            tail += 1
    return count


@njit(cache=True, nogil=True)
def _bfs_membership(
    sx, sy, gw, gh, membership, blocked_bits, blk2, st2, rel_st, rel_val, rel_cur,
    use_rel, seen, seen_cur, qx, qy, qd, cap,
):  # fmt: skip
    """The stamped BFS with a world membership mask as its first blocker.

    This intentionally repeats :func:`_bfs`'s control flow.  In particular, a cell is
    marked seen only after it passes every blocker, which is observable at a capped
    frontier.  Keeping the loops textually parallel makes the FIFO discovery order
    explicit rather than relying on an equivalent-looking flood implementation.
    """
    if sx < 0 or sx >= gw or sy < 0 or sy >= gh:
        return 0
    idx = sx * gh + sy
    if (membership[idx] & blocked_bits) != 0 or blk2[idx] == st2:
        return 0
    if use_rel and rel_st[idx] == rel_cur and 0 < rel_val[idx]:
        return 0
    seen[idx] = seen_cur
    head = 0
    tail = 1
    qx[0] = sx
    qy[0] = sy
    qd[0] = 0
    count = 0
    while head < tail and count < cap:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        count += 1
        for k in range(4):
            nx = x + _NBX[k]
            ny = y + _NBY[k]
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                continue
            nidx = nx * gh + ny
            if seen[nidx] == seen_cur:
                continue
            if (membership[nidx] & blocked_bits) != 0 or blk2[nidx] == st2:
                continue
            if use_rel and rel_st[nidx] == rel_cur and d + 1 < rel_val[nidx]:
                continue
            seen[nidx] = seen_cur
            qx[tail] = nx
            qy[tail] = ny
            qd[tail] = d + 1
            tail += 1
    return count


@njit(cache=True, nogil=True)
def _bfs_owner(
    sx, sy, gw, gh, owner_pad, env, owner_offset, observer, block_any, blk2, st2,
    rel_st, rel_val, rel_cur, use_rel, seen, seen_cur, qx, qy, qd, cap,
):  # fmt: skip
    """Stamped BFS reading GridBatchSim occupancy directly.

    ``block_any`` is the legacy ``all_bits`` case; otherwise only other snakes
    block.  The caller supplies post-move/release overlays through the same
    ``blk2`` and release arrays as the painted reference.
    """
    if sx < 0 or sx >= gw or sy < 0 or sy >= gh:
        return 0
    owner = owner_pad[env, sy + owner_offset, sx + owner_offset]
    if (owner >= 0 and (block_any or owner != observer)) or blk2[sx * gh + sy] == st2:
        return 0
    idx = sx * gh + sy
    if use_rel and rel_st[idx] == rel_cur and 0 < rel_val[idx]:
        return 0
    seen[idx] = seen_cur
    head = 0
    tail = 1
    qx[0] = sx
    qy[0] = sy
    qd[0] = 0
    count = 0
    while head < tail and count < cap:
        x = qx[head]
        y = qy[head]
        d = qd[head]
        head += 1
        count += 1
        for k in range(4):
            nx = x + _NBX[k]
            ny = y + _NBY[k]
            if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
                continue
            nidx = nx * gh + ny
            if seen[nidx] == seen_cur:
                continue
            owner = owner_pad[env, ny + owner_offset, nx + owner_offset]
            if (owner >= 0 and (block_any or owner != observer)) or blk2[nidx] == st2:
                continue
            if use_rel and rel_st[nidx] == rel_cur and d + 1 < rel_val[nidx]:
                continue
            seen[nidx] = seen_cur
            qx[tail] = nx
            qy[tail] = ny
            qd[tail] = d + 1
            tail += 1
    return count


@njit(cache=True, nogil=True)
def _post_move(n, L, bf, boost_action, min_boost, boost_cost):
    """``safety_veto_v5.simulate_move`` sizes: ``(boosted, n_segments, post_length)``."""
    ln = n + 1
    if ln > L:
        ln -= 1
    boosted = boost_action and L >= min_boost
    if boosted:
        ln += 1
        if ln > L:
            ln -= 1
        bf += 1
        if bf >= boost_cost:
            if L > 1:
                L -= 1
                if ln > L:
                    ln -= 1
    return boosted, ln, L


@njit(cache=True, nogil=True)
def _seg_cell(bodies, e, s, hp, ring, nh, hx, hy, dx, dy, i):
    """Cell ``i`` (0 = head) of the post-move body (``nh`` new head cells)."""
    if i < nh:
        step = nh - i  # boosted: i=0 -> head+2d, i=1 -> head+d; normal: i=0 -> head+d
        return hx + dx * step, hy + dy * step
    j = (hp - (i - nh)) % ring
    return bodies[e, s, j, 0], bodies[e, s, j, 1]


@njit(cache=True, nogil=True)
def _landing_count(
    bodies, e, s, hp, ring, n, L, bf, heading, gw, gh, min_boost, boost_cost, cap,
    others, ost, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    """``safety_veto_v5.landing_count`` (``-2`` when the boost would not fire)."""
    boosted, ln, _ = _post_move(n, L, bf, True, min_boost, boost_cost)
    if not boosted:
        return _LANDING_NO_BOOST
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    dx = _DX[heading]
    dy = _DY[heading]
    ws[0] += 1
    tst = ws[0]
    for i in range(1, ln):
        cx, cy = _seg_cell(bodies, e, s, hp, ring, 2, hx, hy, dx, dy, i)
        if cx >= 0 and cx < gw and cy >= 0 and cy < gh:
            tmp[cx * gh + cy] = tst
    ws[1] += 1
    return _bfs(
        hx + 2 * dx, hy + 2 * dy, gw, gh, others, ost, tmp, tst,
        rel_st, rel_val, -1, False, seen, ws[1], qx, qy, qd, cap,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def _landing_area(
    bodies, e, s, hp, ring, n, L, bf, heading, action, gw, gh, min_boost, boost_cost, cap,
    others, ost, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    """``safety_veto_v7.landing_area`` (tail release on, slack 1), at most ``cap``."""
    rel = action % 3
    hd = (heading + rel - 1) % 4
    dx = _DX[hd]
    dy = _DY[hd]
    boosted, ln, lpost = _post_move(n, L, bf, action >= 3, min_boost, boost_cost)
    nh = 2 if boosted else 1
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    ws[2] += 1
    rcur = ws[2]
    for i in range(ln):
        cx, cy = _seg_cell(bodies, e, s, hp, ring, nh, hx, hy, dx, dy, i)
        if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
            continue
        idx = cx * gh + cy
        when = lpost - i + 1
        if rel_st[idx] != rcur:
            rel_st[idx] = rcur
            rel_val[idx] = when
        elif when > rel_val[idx]:
            rel_val[idx] = when
    nx = hx + dx * nh
    ny = hy + dy * nh
    if nx >= 0 and nx < gw and ny >= 0 and ny < gh:
        rel_st[nx * gh + ny] = 0  # release.pop(head): the search starts on the new head
    ws[1] += 1
    return _bfs(
        nx, ny, gw, gh, others, ost, tmp, -1, rel_st, rel_val, rcur, True,
        seen, ws[1], qx, qy, qd, cap,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def _landing_count_membership(
    bodies, e, s, hp, ring, n, L, bf, heading, gw, gh, min_boost, boost_cost, cap,
    membership, other_bits, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    """Membership-backed ``landing_count``; post-move blockers retain legacy semantics."""
    boosted, ln, _ = _post_move(n, L, bf, True, min_boost, boost_cost)
    if not boosted:
        return _LANDING_NO_BOOST
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    dx = _DX[heading]
    dy = _DY[heading]
    ws[0] += 1
    tst = ws[0]
    for i in range(1, ln):
        cx, cy = _seg_cell(bodies, e, s, hp, ring, 2, hx, hy, dx, dy, i)
        if cx >= 0 and cx < gw and cy >= 0 and cy < gh:
            tmp[cx * gh + cy] = tst
    ws[1] += 1
    return _bfs_membership(
        hx + 2 * dx, hy + 2 * dy, gw, gh, membership, other_bits, tmp, tst,
        rel_st, rel_val, -1, False, seen, ws[1], qx, qy, qd, cap,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def _landing_area_membership(
    bodies, e, s, hp, ring, n, L, bf, heading, action, gw, gh, min_boost, boost_cost, cap,
    membership, other_bits, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    """Membership-backed ``landing_area`` with the exact retained-body release map."""
    rel = action % 3
    hd = (heading + rel - 1) % 4
    dx = _DX[hd]
    dy = _DY[hd]
    boosted, ln, lpost = _post_move(n, L, bf, action >= 3, min_boost, boost_cost)
    nh = 2 if boosted else 1
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    ws[2] += 1
    rcur = ws[2]
    for i in range(ln):
        cx, cy = _seg_cell(bodies, e, s, hp, ring, nh, hx, hy, dx, dy, i)
        if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
            continue
        idx = cx * gh + cy
        when = lpost - i + 1
        if rel_st[idx] != rcur:
            rel_st[idx] = rcur
            rel_val[idx] = when
        elif when > rel_val[idx]:
            rel_val[idx] = when
    nx = hx + dx * nh
    ny = hy + dy * nh
    if nx >= 0 and nx < gw and ny >= 0 and ny < gh:
        rel_st[nx * gh + ny] = 0
    ws[1] += 1
    return _bfs_membership(
        nx, ny, gw, gh, membership, other_bits, tmp, -1, rel_st, rel_val, rcur, True,
        seen, ws[1], qx, qy, qd, cap,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def _landing_count_owner(
    bodies, e, s, hp, ring, n, L, bf, heading, gw, gh, min_boost, boost_cost, cap,
    owner_pad, owner_offset, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    boosted, ln, _ = _post_move(n, L, bf, True, min_boost, boost_cost)
    if not boosted:
        return _LANDING_NO_BOOST
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    dx = _DX[heading]
    dy = _DY[heading]
    ws[0] += 1
    tst = ws[0]
    for i in range(1, ln):
        cx, cy = _seg_cell(bodies, e, s, hp, ring, 2, hx, hy, dx, dy, i)
        if cx >= 0 and cx < gw and cy >= 0 and cy < gh:
            tmp[cx * gh + cy] = tst
    ws[1] += 1
    return _bfs_owner(
        hx + 2 * dx, hy + 2 * dy, gw, gh, owner_pad, e, owner_offset, s, False, tmp, tst,
        rel_st, rel_val, -1, False, seen, ws[1], qx, qy, qd, cap,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def _landing_area_owner(
    bodies, e, s, hp, ring, n, L, bf, heading, action, gw, gh, min_boost, boost_cost, cap,
    owner_pad, owner_offset, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    rel = action % 3
    hd = (heading + rel - 1) % 4
    dx = _DX[hd]
    dy = _DY[hd]
    boosted, ln, lpost = _post_move(n, L, bf, action >= 3, min_boost, boost_cost)
    nh = 2 if boosted else 1
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    ws[2] += 1
    rcur = ws[2]
    for i in range(ln):
        cx, cy = _seg_cell(bodies, e, s, hp, ring, nh, hx, hy, dx, dy, i)
        if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
            continue
        idx = cx * gh + cy
        when = lpost - i + 1
        if rel_st[idx] != rcur:
            rel_st[idx] = rcur
            rel_val[idx] = when
        elif when > rel_val[idx]:
            rel_val[idx] = when
    nx = hx + dx * nh
    ny = hy + dy * nh
    if nx >= 0 and nx < gw and ny >= 0 and ny < gh:
        rel_st[nx * gh + ny] = 0
    ws[1] += 1
    return _bfs_owner(
        nx, ny, gw, gh, owner_pad, e, owner_offset, s, False, tmp, -1,
        rel_st, rel_val, rcur, True, seen, ws[1], qx, qy, qd, cap,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def _landing_count_route(
    bodies, e, s, hp, ring, n, L, bf, heading, gw, gh, min_boost, boost_cost, cap,
    membership, other_bits, owner_pad, owner_offset, use_owner_pad,
    tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    if use_owner_pad:
        return _landing_count_owner(
            bodies,
            e,
            s,
            hp,
            ring,
            n,
            L,
            bf,
            heading,
            gw,
            gh,
            min_boost,
            boost_cost,
            cap,
            owner_pad,
            owner_offset,
            tmp,
            ws,
            seen,
            rel_st,
            rel_val,
            qx,
            qy,
            qd,
        )
    return _landing_count_membership(
        bodies,
        e,
        s,
        hp,
        ring,
        n,
        L,
        bf,
        heading,
        gw,
        gh,
        min_boost,
        boost_cost,
        cap,
        membership,
        other_bits,
        tmp,
        ws,
        seen,
        rel_st,
        rel_val,
        qx,
        qy,
        qd,
    )


@njit(cache=True, nogil=True)
def _landing_area_route(
    bodies, e, s, hp, ring, n, L, bf, heading, action, gw, gh, min_boost, boost_cost, cap,
    membership, other_bits, owner_pad, owner_offset, use_owner_pad,
    tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
):  # fmt: skip
    if use_owner_pad:
        return _landing_area_owner(
            bodies,
            e,
            s,
            hp,
            ring,
            n,
            L,
            bf,
            heading,
            action,
            gw,
            gh,
            min_boost,
            boost_cost,
            cap,
            owner_pad,
            owner_offset,
            tmp,
            ws,
            seen,
            rel_st,
            rel_val,
            qx,
            qy,
            qd,
        )
    return _landing_area_membership(
        bodies,
        e,
        s,
        hp,
        ring,
        n,
        L,
        bf,
        heading,
        action,
        gw,
        gh,
        min_boost,
        boost_cost,
        cap,
        membership,
        other_bits,
        tmp,
        ws,
        seen,
        rel_st,
        rel_val,
        qx,
        qy,
        qd,
    )


@njit(cache=True, nogil=True)
def _eager_templates(
    bodies, e, s, hp, ring, n, L, bf, heading, gw, gh, min_boost, boost_cost,
    area_cap, landing_cap, membership, other_bits, tmp, ws, seen, rel_st, rel_val,
    qx, qy, qd, area, landing,
):  # fmt: skip
    """Paint retained old cells once per speed; overlay and restore two action cells.

    Static boost blockers retain an old cell at the landing head; tail-aware area
    deliberately removes that cell. Separate stamps keep these contracts distinct.
    Every search still uses the same FIFO, neighbor order and passability rule.
    """
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    for speed in range(2):
        boosted, ln, lpost = _post_move(n, L, bf, speed == 1, min_boost, boost_cost)
        nh = 2 if boosted else 1
        if speed == 1 and not boosted:
            for d in range(3):
                area[3 + d] = area[d]
                landing[d] = _LANDING_NO_BOOST
            continue
        ws[2] += 1
        rcur = ws[2]
        ws[0] += 1
        tst = ws[0]
        for j in range(max(0, ln - nh)):
            k = (hp - j) % ring
            cx = bodies[e, s, k, 0]
            cy = bodies[e, s, k, 1]
            if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
                continue
            idx = cx * gh + cy
            when = lpost - j - (nh - 1)
            if rel_st[idx] != rcur:
                rel_st[idx] = rcur
                rel_val[idx] = when
            elif when > rel_val[idx]:
                rel_val[idx] = when
            if boosted:
                tmp[idx] = tst
        for d in range(3):
            hd = (heading + d - 1) % 4
            dx = _DX[hd]
            dy = _DY[hd]
            ix = hx + dx
            iy = hy + dy
            mid = -1
            mid_st = 0
            mid_val = 0
            mid_tmp = 0
            if boosted and ln > 1 and 0 <= ix < gw and 0 <= iy < gh:
                mid = ix * gh + iy
                mid_st = rel_st[mid]
                mid_val = rel_val[mid]
                mid_tmp = tmp[mid]
                if mid_st != rcur or lpost > mid_val:
                    rel_val[mid] = lpost
                rel_st[mid] = rcur
                tmp[mid] = tst
            nx = hx + nh * dx
            ny = hy + nh * dy
            if boosted:
                ws[1] += 1
                landing[d] = _bfs_membership(
                    nx,
                    ny,
                    gw,
                    gh,
                    membership,
                    other_bits,
                    tmp,
                    tst,
                    rel_st,
                    rel_val,
                    -1,
                    False,
                    seen,
                    ws[1],
                    qx,
                    qy,
                    qd,
                    landing_cap,
                )
            head = -1
            head_st = 0
            if 0 <= nx < gw and 0 <= ny < gh:
                head = nx * gh + ny
                head_st = rel_st[head]
                rel_st[head] = 0  # release.pop(head), including old-body overlaps
            ws[1] += 1
            area[speed * 3 + d] = _bfs_membership(
                nx,
                ny,
                gw,
                gh,
                membership,
                other_bits,
                tmp,
                -1,
                rel_st,
                rel_val,
                rcur,
                True,
                seen,
                ws[1],
                qx,
                qy,
                qd,
                area_cap,
            )
            if head >= 0:
                rel_st[head] = head_st
            if mid >= 0:
                rel_st[mid] = mid_st
                rel_val[mid] = mid_val
                tmp[mid] = mid_tmp


@njit(cache=True, nogil=True)
def _eager_templates_owner(
    bodies, e, s, hp, ring, n, L, bf, heading, gw, gh, min_boost, boost_cost,
    area_cap, landing_cap, owner_pad, owner_offset, tmp, ws, seen, rel_st, rel_val,
    qx, qy, qd, area, landing,
):  # fmt: skip
    """Owner-pad counterpart of :func:`_eager_templates` without a body bitset."""
    hx = bodies[e, s, hp, 0]
    hy = bodies[e, s, hp, 1]
    for speed in range(2):
        boosted, ln, lpost = _post_move(n, L, bf, speed == 1, min_boost, boost_cost)
        nh = 2 if boosted else 1
        if speed == 1 and not boosted:
            for d in range(3):
                area[3 + d] = area[d]
                landing[d] = _LANDING_NO_BOOST
            continue
        ws[2] += 1
        rcur = ws[2]
        ws[0] += 1
        tst = ws[0]
        for j in range(max(0, ln - nh)):
            k = (hp - j) % ring
            cx = bodies[e, s, k, 0]
            cy = bodies[e, s, k, 1]
            if cx < 0 or cx >= gw or cy < 0 or cy >= gh:
                continue
            idx = cx * gh + cy
            when = lpost - j - (nh - 1)
            if rel_st[idx] != rcur:
                rel_st[idx] = rcur
                rel_val[idx] = when
            elif when > rel_val[idx]:
                rel_val[idx] = when
            if boosted:
                tmp[idx] = tst
        for d in range(3):
            hd = (heading + d - 1) % 4
            dx = _DX[hd]
            dy = _DY[hd]
            ix = hx + dx
            iy = hy + dy
            mid = -1
            mid_st = 0
            mid_val = 0
            mid_tmp = 0
            if boosted and ln > 1 and 0 <= ix < gw and 0 <= iy < gh:
                mid = ix * gh + iy
                mid_st = rel_st[mid]
                mid_val = rel_val[mid]
                mid_tmp = tmp[mid]
                if mid_st != rcur or lpost > mid_val:
                    rel_val[mid] = lpost
                rel_st[mid] = rcur
                tmp[mid] = tst
            nx = hx + nh * dx
            ny = hy + nh * dy
            if boosted:
                ws[1] += 1
                landing[d] = _bfs_owner(
                    nx,
                    ny,
                    gw,
                    gh,
                    owner_pad,
                    e,
                    owner_offset,
                    s,
                    False,
                    tmp,
                    tst,
                    rel_st,
                    rel_val,
                    -1,
                    False,
                    seen,
                    ws[1],
                    qx,
                    qy,
                    qd,
                    landing_cap,
                )
            head = -1
            head_st = 0
            if 0 <= nx < gw and 0 <= ny < gh:
                head = nx * gh + ny
                head_st = rel_st[head]
                rel_st[head] = 0
            ws[1] += 1
            area[speed * 3 + d] = _bfs_owner(
                nx,
                ny,
                gw,
                gh,
                owner_pad,
                e,
                owner_offset,
                s,
                False,
                tmp,
                -1,
                rel_st,
                rel_val,
                rcur,
                True,
                seen,
                ws[1],
                qx,
                qy,
                qd,
                area_cap,
            )
            if head >= 0:
                rel_st[head] = head_st
            if mid >= 0:
                rel_st[mid] = mid_st
                rel_val[mid] = mid_val
                tmp[mid] = mid_tmp


@njit(cache=True, nogil=True)
def _veto_choice(qv, elig, base):
    """``safety_veto.veto_choice`` on precomputed eligibility (mask & spacious)."""
    anyel = False
    for a in range(6):
        if elig[a]:
            anyel = True
    if not anyel:
        return base, OUTCOME_NO_SPACIOUS
    if elig[base]:
        return base, OUTCOME_KEPT
    bb = base >= 3
    best = -1
    for a in range(6):
        if elig[a] and (a >= 3) == bb:
            if best < 0 or qv[a] > qv[best]:
                best = a
    if best < 0:
        for a in range(6):
            if elig[a]:
                if best < 0 or qv[a] > qv[best]:
                    best = a
    return best, OUTCOME_VETOED


@njit(cache=True, nogil=True)
def _space_pref(qv, cand, nc, anchor, lam, area, area_cap_row, scratch):
    """``safety_veto_v7.space_preference_choice`` (first pass: which areas are needed).

    ``area[a] < 0`` means not evaluated. When an area is needed and missing, returns
    ``(-1, a)`` so the caller evaluates it and calls again (lazy, exact order).
    """
    if nc < 2:
        return anchor, -1
    top = qv[cand[0]]
    bot = qv[cand[0]]
    for i in range(1, nc):
        v = qv[cand[i]]
        if v > top:
            top = v
        if v < bot:
            bot = v
    scale = abs(top - bot)
    if scale < Q_SCALE_EPS:
        scale = Q_SCALE_EPS
    for i in range(nc):
        scratch[cand[i]] = (qv[cand[i]] - top) / scale
    qa = scratch[anchor]
    nobody = True
    for i in range(nc):
        a = cand[i]
        if a != anchor and not (scratch[a] + lam <= qa):
            nobody = False
    if nobody:
        return anchor, -1
    capf = max(area_cap_row, 1)
    if area[anchor] < 0:
        return -1, anchor
    g = min(max(area[anchor], 0), capf) / capf
    best = anchor
    best_score = qa + lam * g
    for i in range(nc):  # cand is ascending, so this is ``sorted(others)``
        a = cand[i]
        if a == anchor:
            continue
        if scratch[a] + lam <= best_score:
            continue
        if area[a] < 0:
            return -1, a
        g = min(max(area[a], 0), capf) / capf
        value = scratch[a] + lam * g
        if value > best_score:
            best = a
            best_score = value
    return best, -1


@njit(cache=True, nogil=True)
def _v8_rows(
    bodies, head_ptr, seg_count, length, boost_frames, direction, alive,
    envs, slots, q, mask_in, gw, gh, min_boost, boost_cost, mech_v2, headon_ratio, lam,
    owner_pad, owner_pad_width, owner_pad_offset, coarse_snake, use_owner_pad,
    eager, ws, others, hero, tmp, seen, rel_st, rel_val, qx, qy, qd, membership, touched,
    out_action, out_base, out_v5, out_v7, out_outcome, out_reason,
    out_counts, out_cap, out_need, out_acap, out_area, out_landing, out_risky,
    out_waived, out_elig,
):  # fmt: skip
    N = envs.shape[0]
    S = alive.shape[1]
    ring = bodies.shape[2]
    qv = np.empty(6, dtype=np.float64)
    mask = np.empty(6, dtype=np.bool_)
    elig = np.empty(6, dtype=np.bool_)
    spac = np.empty(3, dtype=np.bool_)
    land_ok = np.empty(3, dtype=np.int64)  # -1 unknown, 0/1
    cand = np.empty(6, dtype=np.int64)
    scratch = np.empty(6, dtype=np.float64)
    area = np.empty(6, dtype=np.int64)
    red = np.empty(6, dtype=np.bool_)
    tx = np.empty(7 * S, dtype=np.int64)
    ty = np.empty(7 * S, dtype=np.int64)
    tb = np.empty(7 * S, dtype=np.int64)
    inv = np.float32(INVALID_Q_VALUE)
    last_e = -1
    ntouched = 0
    # The painted reference rebuilds a frame-local membership map.  The owner-pad
    # route deliberately does neither: occupancy is read directly by its BFSes.
    if not use_owner_pad:
        for i in range(membership.shape[0]):
            membership[i] = np.uint64(0)
    for r in range(N):
        e = envs[r]
        s = slots[r]
        hp = head_ptr[e, s]
        n = seg_count[e, s]
        L = length[e, s]
        bf = boost_frames[e, s]
        heading = direction[e, s]
        hx = bodies[e, s, hp, 0]
        hy = bodies[e, s, hp, 1]
        # Build a body-union bitset once per selected world.  Rows are stable-grouped by
        # environment in ``v8_decide`` so every learner and frozen seat shares it.
        # ``membership`` is intentionally frame-local scratch: no cross-frame cache or
        # invalidation rule is introduced here.
        # Six-slot worlds use one bit per snake.  A wider population falls back to
        # observer-vs-other bits rebuilt for that row: exact union/overlap behavior,
        # without truncating a snake id into a machine word.
        wide_population = S > 63
        if (not use_owner_pad) and (wide_population or e != last_e):
            for i in range(ntouched):
                membership[touched[i]] = np.uint64(0)
            ntouched = 0
            for o in range(S):
                if not alive[e, o]:
                    continue
                if wide_population:
                    bit = np.uint64(1) if o == s else np.uint64(2)
                else:
                    bit = np.uint64(1) << np.uint64(o)
                ohp = head_ptr[e, o]
                for k in range(seg_count[e, o]):
                    j = (ohp - k) % ring
                    cx = bodies[e, o, j, 0]
                    cy = bodies[e, o, j, 1]
                    if cx >= 0 and cx < gw and cy >= 0 and cy < gh:
                        idx = cx * gh + cy
                        if membership[idx] == 0:
                            touched[ntouched] = idx
                            ntouched += 1
                        membership[idx] |= bit
            last_e = e
        # ---- masked Q (float32), base = first maximum (torch.argmax)
        anym = False
        for a in range(6):
            mask[a] = mask_in[r, a]
            anym = anym or mask[a]
        if not anym:
            mask[0] = True
            mask[1] = True
            mask[2] = True
        base = 0
        for a in range(6):
            v = q[r, a] if mask[a] else inv
            qv[a] = np.float64(v)
            if qv[a] > qv[base]:
                base = a
        # ---- observer-specific unique coverage derives from the shared union.  A cell
        # occupied by the observer and an opponent remains covered, matching the old
        # "paint opponents first" semantics for synthetic overlapping bodies.
        ws[0] += 1
        covered = 0
        ws[0] += 1
        hst = ws[0]
        if wide_population:
            all_bits = np.uint64(3)
            own_bit = np.uint64(1)
            other_bits = np.uint64(2)
        else:
            all_bits = (np.uint64(1) << np.uint64(S)) - np.uint64(1)
            own_bit = np.uint64(1) << np.uint64(s)
            other_bits = all_bits ^ own_bit
        if use_owner_pad:
            for o in range(S):
                if o == s:
                    continue
                for cy in range(coarse_snake.shape[2]):
                    for cx in range(coarse_snake.shape[3]):
                        covered += coarse_snake[e, o, cy, cx]
        else:
            for i in range(ntouched):
                if (membership[touched[i]] & other_bits) != 0:
                    covered += 1
        # ---- v2 cap / need / counts / spacious
        cap = min(FREE_SPACE_BFS_CAP, max(FREE_SPACE_MIN_CAP, L * 2))
        need = min(max(1, L), cap)
        for d in range(3):
            hd = (heading + d - 1) % 4
            ws[1] += 1
            if use_owner_pad:
                c = _bfs_owner(
                    hx + _DX[hd],
                    hy + _DY[hd],
                    gw,
                    gh,
                    owner_pad,
                    e,
                    owner_pad_offset,
                    s,
                    True,
                    hero,
                    hst,
                    rel_st,
                    rel_val,
                    -1,
                    False,
                    seen,
                    ws[1],
                    qx,
                    qy,
                    qd,
                    cap,
                )
            else:
                c = _bfs_membership(
                    hx + _DX[hd], hy + _DY[hd], gw, gh, membership, all_bits, hero, hst,
                    rel_st, rel_val, -1, False, seen, ws[1], qx, qy, qd, cap,
                )  # fmt: skip
            out_counts[r, d] = c
            spac[d] = c >= need
            land_ok[d] = -1
            out_landing[r, d] = _LANDING_UNKNOWN
        acap = max(1, min(max(FREE_SPACE_MIN_CAP, 2 * L), AREA_CAP_LIMIT, gw * gh - covered))
        out_cap[r] = cap
        out_need[r] = need
        out_acap[r] = acap
        for a in range(6):
            area[a] = -1
        if eager:
            if use_owner_pad:
                _eager_templates_owner(
                    bodies,
                    e,
                    s,
                    hp,
                    ring,
                    n,
                    L,
                    bf,
                    heading,
                    gw,
                    gh,
                    min_boost,
                    boost_cost,
                    acap,
                    cap,
                    owner_pad,
                    owner_pad_offset,
                    tmp,
                    ws,
                    seen,
                    rel_st,
                    rel_val,
                    qx,
                    qy,
                    qd,
                    area,
                    out_landing[r],
                )
            else:
                _eager_templates(
                    bodies,
                    e,
                    s,
                    hp,
                    ring,
                    n,
                    L,
                    bf,
                    heading,
                    gw,
                    gh,
                    min_boost,
                    boost_cost,
                    acap,
                    cap,
                    membership,
                    other_bits,
                    tmp,
                    ws,
                    seen,
                    rel_st,
                    rel_val,
                    qx,
                    qy,
                    qd,
                    area,
                    out_landing[r],
                )
            for d in range(3):
                lc = out_landing[r, d]
                land_ok[d] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
        # ---- v5 boost_aware_choice (landing floods lazily, memoized)
        for a in range(6):
            elig[a] = mask[a] and spac[a % 3]
        v5 = base
        outcome = OUTCOME_KEPT
        reason = _R_KEPT
        need_clear = False
        if elig[base]:
            d0 = base % 3
            ok = True
            if base >= 3:
                if land_ok[d0] < 0:
                    hd = (heading + d0 - 1) % 4
                    lc = _landing_count_route(
                        bodies, e, s, hp, ring, n, L, bf, hd, gw, gh, min_boost, boost_cost,
                        cap, membership, other_bits, owner_pad, owner_pad_offset,
                        use_owner_pad, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
                    )  # fmt: skip
                    out_landing[r, d0] = lc
                    land_ok[d0] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
                ok = land_ok[d0] == 1
            if not ok:
                if mask[d0]:
                    v5 = d0
                    outcome = OUTCOME_VETOED
                    reason = _R_SAME_DIR
                else:
                    need_clear = True
                    reason = _R_LANDING_V2
        else:
            v5, outcome = _veto_choice(qv, elig, base)
            reason = _R_V2
            if outcome == OUTCOME_VETOED and v5 >= 3:
                d0 = v5 % 3
                if land_ok[d0] < 0:
                    hd = (heading + d0 - 1) % 4
                    lc = _landing_count_route(
                        bodies, e, s, hp, ring, n, L, bf, hd, gw, gh, min_boost, boost_cost,
                        cap, membership, other_bits, owner_pad, owner_pad_offset,
                        use_owner_pad, tmp, ws, seen, rel_st, rel_val, qx, qy, qd,
                    )  # fmt: skip
                    out_landing[r, d0] = lc
                    land_ok[d0] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
                if land_ok[d0] == 0:
                    need_clear = True
        if need_clear:
            # _cleared_boosts: every masked-legal spacious boost whose landing fails.
            for d in range(3):
                a = 3 + d
                if elig[a]:
                    if land_ok[d] < 0:
                        hd = (heading + d - 1) % 4
                        lc = _landing_count_route(
                            bodies, e, s, hp, ring, n, L, bf, hd, gw, gh, min_boost,
                            boost_cost, cap, membership, other_bits, owner_pad,
                            owner_pad_offset, use_owner_pad, tmp, ws, seen, rel_st, rel_val,
                            qx, qy, qd,
                        )  # fmt: skip
                        out_landing[r, d] = lc
                        land_ok[d] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
            for a in range(6):
                red[a] = elig[a] and (a < 3 or land_ok[a - 3] == 1)
            a2, o2 = _veto_choice(qv, red, base)
            if reason == _R_LANDING_V2:
                if o2 == OUTCOME_NO_SPACIOUS:
                    v5 = base
                    outcome = OUTCOME_NO_SPACIOUS
                    reason = _R_LANDING_NONE
                else:
                    v5 = a2
                    outcome = o2
            else:
                v5 = a2
                outcome = o2
        # ---- v7 re-rank
        v7 = v5
        v7_outcome = outcome
        if outcome != OUTCOME_NO_SPACIOUS and lam != 0.0 and reason != _R_SAME_DIR:
            off = 3 if v5 >= 3 else 0
            nc = 0
            for d in range(3):
                a = off + d
                if a == v5:
                    cand[nc] = a
                    nc += 1
                elif mask[a] and spac[d]:
                    if off == 0:
                        cand[nc] = a
                        nc += 1
                    else:
                        if land_ok[d] < 0:
                            hd = (heading + d - 1) % 4
                            lc = _landing_count_route(
                                bodies, e, s, hp, ring, n, L, bf, hd, gw, gh, min_boost,
                                boost_cost, cap, membership, other_bits, owner_pad,
                                owner_pad_offset, use_owner_pad, tmp, ws, seen, rel_st,
                                rel_val, qx, qy, qd,
                            )  # fmt: skip
                            out_landing[r, d] = lc
                            land_ok[d] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
                        if land_ok[d] == 1:
                            cand[nc] = a
                            nc += 1
            if nc >= 2:
                while True:
                    best, missing = _space_pref(qv, cand, nc, v5, lam, area, acap, scratch)
                    if missing < 0:
                        break
                    area[missing] = _landing_area_route(
                        bodies, e, s, hp, ring, n, L, bf, heading, missing, gw, gh, min_boost,
                        boost_cost, acap, membership, other_bits, owner_pad, owner_pad_offset,
                        use_owner_pad, tmp, ws, seen, rel_st, rel_val,
                        qx, qy, qd,
                    )  # fmt: skip
                if best != v5:
                    v7 = best
                    v7_outcome = OUTCOME_VETOED
        # ---- v6 head reach / risk (always computed: cheap, decision-independent)
        nt = 0
        for o in range(S):
            if o == s or not alive[e, o] or seg_count[e, o] <= 0:
                continue
            ohp = head_ptr[e, o]
            ox = bodies[e, o, ohp, 0]
            oy = bodies[e, o, ohp, 1]
            bound = max(1, length[e, o]) + 1
            tx[nt] = ox
            ty[nt] = oy
            tb[nt] = bound
            nt += 1
            can_boost = length[e, o] >= min_boost
            for rel in range(3):
                od = (direction[e, o] + rel - 1) % 4
                tx[nt] = ox + _DX[od]
                ty[nt] = oy + _DY[od]
                tb[nt] = bound
                nt += 1
                if can_boost:
                    tx[nt] = ox + 2 * _DX[od]
                    ty[nt] = oy + 2 * _DY[od]
                    tb[nt] = bound
                    nt += 1
        near = False
        for i in range(nt):
            if abs(tx[i] - hx) + abs(ty[i] - hy) <= 2:
                near = True
                break
        for a in range(6):
            out_risky[r, a] = False
            out_waived[r, a] = False
        if near:
            for a in range(6):
                hd = (heading + (a % 3) - 1) % 4
                boosted, _, lpost = _post_move(n, L, bf, a >= 3, min_boost, boost_cost)
                ntrav = 2 if boosted else 1
                risky = False
                waived = False
                for t in range(ntrav):
                    cx = hx + _DX[hd] * (t + 1)
                    cy = hy + _DY[hd] * (t + 1)
                    bnd = -1
                    for i in range(nt):
                        if tx[i] == cx and ty[i] == cy and tb[i] > bnd:
                            bnd = tb[i]
                    if bnd < 0:
                        continue
                    if mech_v2 and max(1, lpost) >= headon_ratio * max(1, bnd):
                        waived = True
                        continue
                    risky = True
                if risky:
                    waived = False
                out_risky[r, a] = risky
                out_waived[r, a] = waived
        # ---- v8 head layer
        final = v7
        final_outcome = v7_outcome
        if v7_outcome != OUTCOME_NO_SPACIOUS and out_risky[r, v7]:
            ne = 0
            for a in range(6):
                el = mask[a] and spac[a % 3] and not out_risky[r, a]
                if el and a >= 3:
                    d = a - 3
                    if land_ok[d] < 0:
                        hd = (heading + d - 1) % 4
                        lc = _landing_count_route(
                            bodies, e, s, hp, ring, n, L, bf, hd, gw, gh, min_boost,
                            boost_cost, cap, membership, other_bits, owner_pad,
                            owner_pad_offset, use_owner_pad, tmp, ws, seen, rel_st, rel_val,
                            qx, qy, qd,
                        )  # fmt: skip
                        out_landing[r, d] = lc
                        land_ok[d] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
                    el = land_ok[d] == 1
                elig[a] = el
                if el and a != v7:
                    ne += 1
            if ne > 0:
                cb = v7 >= 3
                anchor = -1
                for a in range(6):  # same speed mode first
                    if a != v7 and elig[a] and (a >= 3) == cb:
                        if anchor < 0 or qv[a] > qv[anchor]:
                            anchor = a
                if anchor < 0:
                    for a in range(6):
                        if a != v7 and elig[a]:
                            if anchor < 0 or qv[a] > qv[anchor]:
                                anchor = a
                ab = anchor >= 3
                nc = 0
                for a in range(6):
                    if a != v7 and elig[a] and (a >= 3) == ab:
                        cand[nc] = a
                        nc += 1
                while True:
                    best, missing = _space_pref(qv, cand, nc, anchor, lam, area, acap, scratch)
                    if missing < 0:
                        break
                    area[missing] = _landing_area_route(
                        bodies, e, s, hp, ring, n, L, bf, heading, missing, gw, gh, min_boost,
                        boost_cost, acap, membership, other_bits, owner_pad, owner_pad_offset,
                        use_owner_pad, tmp, ws, seen, rel_st, rel_val,
                        qx, qy, qd,
                    )  # fmt: skip
                final = best
                final_outcome = OUTCOME_VETOED
        # ---- outputs
        for a in range(6):
            out_area[r, a] = area[a]
            # v5 eligibility; a boost whose landing was not evaluated (lazy) reads False.
            out_elig[r, a] = mask[a] and spac[a % 3] and (a < 3 or land_ok[a - 3] == 1)
        out_action[r] = final
        out_base[r] = base
        out_v5[r] = v5
        out_v7[r] = v7
        out_outcome[r] = final_outcome
        out_reason[r] = reason


@njit(cache=True, nogil=True, parallel=True)
def _v8_rows_prange(
    bodies, head_ptr, seg_count, length, boost_frames, direction, alive,
    envs, slots, q, mask_in, gw, gh, min_boost, boost_cost, mech_v2, headon_ratio, lam,
    owner_pad, owner_pad_width, owner_pad_offset, coarse_snake, use_owner_pad, eager,
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
    out_action, out_base, out_v5, out_v7, out_outcome, out_reason,
    out_counts, out_cap, out_need, out_acap, out_area, out_landing, out_risky,
    out_waived, out_elig,
):  # fmt: skip
    """Run contiguous row partitions with scratch indexed by worker id.

    The scalar kernel allocates small per-call action arrays, so invoke it
    once per partition rather than once per row.
    """
    parts = w0.shape[0]
    for part in prange(parts):
        begin = part * envs.shape[0] // parts
        end = (part + 1) * envs.shape[0] // parts
        tid = get_thread_id()
        _v8_rows(
            bodies, head_ptr, seg_count, length, boost_frames, direction, alive,
            envs[begin:end], slots[begin:end], q[begin:end], mask_in[begin:end],
            gw, gh, min_boost, boost_cost, mech_v2, headon_ratio, lam,
            owner_pad, owner_pad_width, owner_pad_offset, coarse_snake, use_owner_pad, eager,
            w0[tid], w1[tid], w2[tid], w3[tid], w4[tid], w5[tid], w6[tid], w7[tid],
            w8[tid], w9[tid], w10[tid], w11[tid],
            out_action[begin:end], out_base[begin:end], out_v5[begin:end], out_v7[begin:end],
            out_outcome[begin:end], out_reason[begin:end], out_counts[begin:end],
            out_cap[begin:end], out_need[begin:end], out_acap[begin:end], out_area[begin:end],
            out_landing[begin:end], out_risky[begin:end], out_waived[begin:end],
            out_elig[begin:end],
        )  # fmt: skip


# ----------------------------------------------------------------------------- Python API
@dataclass
class V8Result:
    """Per-row v8 decisions and diagnostics (``-1`` = not evaluated unless ``features``).

    ``action``: the v8 action. ``base``: masked argmax. ``v5_action``/``v7_action``: the
    intermediate choices. ``outcome``: 0 kept / 1 vetoed / 2 no_spacious (v8's).
    ``counts`` ``(N, 3)``: v2 capped flood counts; ``cap``/``need``: v2 thresholds;
    ``area_cap``: v7's cap. ``area`` ``(N, 6)``: tail-aware post-move landing area, capped
    at ``area_cap``. ``landing`` ``(N, 3)``: v5 boost landing count per direction (``-2``:
    the boost would not fire). ``head_risky``/``head_waived`` ``(N, 6)``: v6 flags.
    ``v5_eligible`` ``(N, 6)``: masked-legal, v2-spacious and (boost) landing ok.
    """

    action: np.ndarray
    base: np.ndarray
    v5_action: np.ndarray
    v7_action: np.ndarray
    outcome: np.ndarray
    v5_reason: np.ndarray
    counts: np.ndarray
    cap: np.ndarray
    need: np.ndarray
    area_cap: np.ndarray
    area: np.ndarray
    landing: np.ndarray
    head_risky: np.ndarray
    head_waived: np.ndarray
    v5_eligible: np.ndarray

    @property
    def area_g(self) -> np.ndarray:
        """``min(area, cap) / cap`` (v7's ``g``; NaN where not evaluated)."""
        a = self.area.astype(np.float64)
        cap = np.maximum(self.area_cap, 1).astype(np.float64)[:, None]
        g = np.minimum(np.maximum(a, 0.0), cap) / cap
        return np.where(self.area >= 0, g, np.nan)

    @property
    def landing_ok(self) -> np.ndarray:
        """``(N, 6)`` v5 landing test (normal moves: True; NaN-free only with features)."""
        out = np.ones((len(self.action), 6), dtype=bool)
        lc = self.landing
        out[:, 3:] = (lc == _LANDING_NO_BOOST) | (lc >= self.need[:, None])
        return out

    @property
    def spacious(self) -> np.ndarray:
        """``(N, 3)`` v2 one-step spacious flags."""
        return self.counts >= self.need[:, None]


def v8_feature_planes(res: V8Result) -> np.ndarray:
    """``(N, 6, len(FEATURE_NAMES))`` float32 network inputs (call with ``features=True``)."""
    if np.any(res.area < 0):
        raise ValueError("feature planes need v8_decide(..., features=True)")
    sp = np.repeat(res.spacious, 2, axis=0).reshape(-1, 2, 3)
    sp6 = np.concatenate([sp[:, 0], sp[:, 1]], axis=1)
    return np.stack(
        [res.area_g, res.head_risky, res.landing_ok, res.v5_eligible, sp6], axis=-1
    ).astype(np.float32)


_WORKSPACE = threading.local()


def _workspace(gw: int, gh: int) -> Tuple[np.ndarray, ...]:
    key = (int(gw), int(gh))
    workspaces = getattr(_WORKSPACE, "by_shape", None)
    if workspaces is None:
        workspaces = {}
        _WORKSPACE.by_shape = workspaces
    ws = workspaces.get(key)
    if ws is None:
        n = int(gw) * int(gh)
        ws = (
            np.zeros(4, dtype=np.int64),  # stamp counters: paint, seen, release, (spare)
            np.zeros(n, dtype=np.int64),  # others
            np.zeros(n, dtype=np.int64),  # hero body
            np.zeros(n, dtype=np.int64),  # tmp (landing body)
            np.zeros(n, dtype=np.int64),  # seen
            np.zeros(n, dtype=np.int64),  # release stamp
            np.zeros(n, dtype=np.int64),  # release value
            np.zeros(n, dtype=np.int64),  # queue x
            np.zeros(n, dtype=np.int64),  # queue y
            np.zeros(n, dtype=np.int64),  # queue distance
            np.zeros(n, dtype=np.uint64),  # per-world snake membership bitset
            np.zeros(n, dtype=np.int64),  # sparse touched cells for membership clearing
        )
        ws[0][:] = 1  # stamps start above the zero-initialized arrays
        workspaces[key] = ws
    return ws


def _parallel_workspace(gw: int, gh: int, threads: int) -> Tuple[np.ndarray, ...]:
    """Allocate independent stamped workspaces for each Numba worker."""
    key = ("parallel", int(gw), int(gh), int(threads))
    workspaces = getattr(_WORKSPACE, "by_shape", None)
    if workspaces is None:
        workspaces = {}
        _WORKSPACE.by_shape = workspaces
    ws = workspaces.get(key)
    if ws is None:
        n = int(gw) * int(gh)
        ws = (
            np.ones((threads, 4), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.int64),
            np.zeros((threads, n), dtype=np.uint64),
            np.zeros((threads, n), dtype=np.int64),
        )
        workspaces[key] = ws
    return ws


def _prepare_workspace(ws: Tuple[np.ndarray, ...]) -> None:
    """Clear stamped scratch before an int64 stamp could wrap.

    Scratch remains local to the calling thread and is rebuilt per v8 invocation.  This
    is intentionally a rare full clear, keeping normal hot-path stamp semantics intact.
    """
    if int(ws[0].max()) < np.iinfo(np.int64).max - 100_000:
        return
    for array in ws[:9]:
        array.fill(0)
    ws[0][:] = 1


def _i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def v8_decide(
    sim,
    rows: np.ndarray,
    q: np.ndarray,
    mask: np.ndarray,
    lam: float = 8.0,
    features: bool = False,
    grid_shape: Optional[Tuple[int, int]] = None,
    membership: str = "painted",
) -> V8Result:
    """The v8 veto (``lambda = lam``, head layer on) for ``(N, 2)`` ``(env, slot)`` rows.

    Args:
        sim: The decision-time BatchSim / GridBatchSim (read only).
        rows: ``(N, 2)`` live rows (the hero rows, slot 0 in training).
        q: ``(N, 6)`` Q values (cast to float32, masked here as ``actions()`` does).
        mask: ``(N, 6)`` hard mask of the Python path (vector61 selection mask); the
            normal-only fallback is applied here.
        lam: v7/v8 ``lambda`` (finite, ``>= 0``).
        features: Evaluate every diagnostic for every action (network inputs); the
            decision is identical either way.
        grid_shape: ``(gw, gh)`` override (default ``(sim.grid_w, sim.grid_h)``, the
            vector61 featurizer's free-space grid).
        membership: ``"painted"`` is the overlap-safe reference. ``"owner_pad"``
            reads GridBatchSim's maintained owner map and is valid for normal
            non-overlapping worlds.
    """
    from src.core.mechanics_constants import HEADON_SIZE_RATIO

    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lambda must be finite and >= 0, got {lam!r}")
    if getattr(sim.cfg, "arena_type", "rectangular") == "circular":
        raise ValueError("v8_nb supports rectangular arenas only")
    if membership not in ("painted", "owner_pad"):
        raise ValueError("membership must be 'painted' or 'owner_pad'")
    use_owner_pad = membership == "owner_pad"
    if use_owner_pad and not hasattr(sim, "_owner_pad"):
        raise ValueError("owner_pad membership requires GridBatchSim")
    rows = _i64(rows).reshape(-1, 2)
    N = len(rows)
    q = np.ascontiguousarray(q, dtype=np.float32).reshape(N, 6)
    mask = np.ascontiguousarray(mask, dtype=bool).reshape(N, 6)
    gw, gh = grid_shape if grid_shape is not None else (int(sim.grid_w), int(sim.grid_h))
    if N and not np.all(sim.alive[rows[:, 0], rows[:, 1]]):
        raise ValueError("v8_decide rows must be alive")
    out = V8Result(
        action=np.zeros(N, dtype=np.int64),
        base=np.zeros(N, dtype=np.int64),
        v5_action=np.zeros(N, dtype=np.int64),
        v7_action=np.zeros(N, dtype=np.int64),
        outcome=np.zeros(N, dtype=np.int64),
        v5_reason=np.zeros(N, dtype=np.int64),
        counts=np.zeros((N, 3), dtype=np.int64),
        cap=np.zeros(N, dtype=np.int64),
        need=np.zeros(N, dtype=np.int64),
        area_cap=np.zeros(N, dtype=np.int64),
        area=np.zeros((N, 6), dtype=np.int64),
        landing=np.zeros((N, 3), dtype=np.int64),
        head_risky=np.zeros((N, 6), dtype=bool),
        head_waived=np.zeros((N, 6), dtype=bool),
        v5_eligible=np.zeros((N, 6), dtype=bool),
    )
    if not N:
        return out
    # The kernel materializes a sparse membership map once for each consecutive world.
    # Stable grouping preserves duplicate-row order; scatter restores the caller's order.
    order = np.argsort(rows[:, 0], kind="stable")
    grouped = not np.array_equal(order, np.arange(N))
    if grouped:
        rows_kernel = rows[order]
        q_kernel = q[order]
        mask_kernel = mask[order]
    else:
        rows_kernel = rows
        q_kernel = q
        mask_kernel = mask
    owner_pad = (
        np.ascontiguousarray(sim._owner_pad, dtype=np.int16)
        if use_owner_pad
        else np.full((1, 1, 1), -1, dtype=np.int16)
    )
    owner_offset = int(getattr(sim, "_grid_pad", 0))
    coarse_snake = (
        np.ascontiguousarray(sim._csnake, dtype=np.int32)
        if use_owner_pad
        else np.zeros((1, 1, 1, 1), dtype=np.int32)
    )
    threads = int(__import__("numba").get_num_threads())
    if threads > 1 and N >= threads:
        ws = _parallel_workspace(gw, gh, threads)
        _v8_rows_prange(
            _i64(sim.bodies), _i64(sim.head_ptr), _i64(sim.seg_count), _i64(sim.length),
            _i64(sim.boost_frames), _i64(sim.direction), np.ascontiguousarray(sim.alive, bool),
            rows_kernel[:, 0].copy(), rows_kernel[:, 1].copy(), q_kernel, mask_kernel,
            int(gw), int(gh),
            int(sim.cfg.min_boost_length), int(sim.cfg.boost_length_cost_frames),
            int(sim.cfg.mechanics_version) == 2, float(HEADON_SIZE_RATIO), lam,
            owner_pad, int(owner_pad.shape[2]), owner_offset, coarse_snake, use_owner_pad,
            bool(features),
            *ws,
            out.action, out.base, out.v5_action, out.v7_action, out.outcome, out.v5_reason,
            out.counts, out.cap, out.need, out.area_cap, out.area, out.landing, out.head_risky,
            out.head_waived, out.v5_eligible,
        )  # fmt: skip
    else:
        ws = _workspace(gw, gh)
        _prepare_workspace(ws)
        _v8_rows(
            _i64(sim.bodies), _i64(sim.head_ptr), _i64(sim.seg_count), _i64(sim.length),
            _i64(sim.boost_frames), _i64(sim.direction), np.ascontiguousarray(sim.alive, bool),
            rows_kernel[:, 0].copy(), rows_kernel[:, 1].copy(), q_kernel, mask_kernel,
            int(gw), int(gh), int(sim.cfg.min_boost_length), int(sim.cfg.boost_length_cost_frames),
            int(sim.cfg.mechanics_version) == 2, float(HEADON_SIZE_RATIO), lam,
            owner_pad, int(owner_pad.shape[2]), owner_offset, coarse_snake, use_owner_pad,
            bool(features), *ws,
            out.action, out.base, out.v5_action, out.v7_action, out.outcome, out.v5_reason,
            out.counts, out.cap, out.need, out.area_cap, out.area, out.landing, out.head_risky,
            out.head_waived, out.v5_eligible,
        )  # fmt: skip
    if grouped:
        inverse = np.empty(N, dtype=np.int64)
        inverse[order] = np.arange(N)
        for name in V8Result.__dataclass_fields__:
            value = getattr(out, name)
            setattr(out, name, value[inverse])
    return out


# ----------------------------------------------------------------------------- cached re-decision
@njit(cache=True, nogil=True)
def _redecide_rows(
    q, mask_in, counts, need_in, landing, area_in, acap_in, risky_in, lam,
    out_action, out_base, out_v5, out_v7, out_outcome, out_reason,
):  # fmt: skip
    """The decision part of :func:`_v8_rows` on fully evaluated (eager) diagnostics."""
    N = q.shape[0]
    qv = np.empty(6, dtype=np.float64)
    mask = np.empty(6, dtype=np.bool_)
    elig = np.empty(6, dtype=np.bool_)
    spac = np.empty(3, dtype=np.bool_)
    land_ok = np.empty(3, dtype=np.int64)
    cand = np.empty(6, dtype=np.int64)
    scratch = np.empty(6, dtype=np.float64)
    area = np.empty(6, dtype=np.int64)
    red = np.empty(6, dtype=np.bool_)
    inv = np.float32(INVALID_Q_VALUE)
    for r in range(N):
        anym = False
        for a in range(6):
            mask[a] = mask_in[r, a]
            anym = anym or mask[a]
        if not anym:
            mask[0] = True
            mask[1] = True
            mask[2] = True
        base = 0
        for a in range(6):
            v = q[r, a] if mask[a] else inv
            qv[a] = np.float64(v)
            if qv[a] > qv[base]:
                base = a
        need = need_in[r]
        acap = acap_in[r]
        for d in range(3):
            spac[d] = counts[r, d] >= need
            lc = landing[r, d]
            land_ok[d] = 1 if (lc == _LANDING_NO_BOOST or lc >= need) else 0
        for a in range(6):
            area[a] = area_in[r, a]
        # ---- v5
        for a in range(6):
            elig[a] = mask[a] and spac[a % 3]
        v5 = base
        outcome = OUTCOME_KEPT
        reason = _R_KEPT
        need_clear = False
        if elig[base]:
            d0 = base % 3
            ok = True
            if base >= 3:
                ok = land_ok[d0] == 1
            if not ok:
                if mask[d0]:
                    v5 = d0
                    outcome = OUTCOME_VETOED
                    reason = _R_SAME_DIR
                else:
                    need_clear = True
                    reason = _R_LANDING_V2
        else:
            v5, outcome = _veto_choice(qv, elig, base)
            reason = _R_V2
            if outcome == OUTCOME_VETOED and v5 >= 3:
                if land_ok[v5 % 3] == 0:
                    need_clear = True
        if need_clear:
            for a in range(6):
                red[a] = elig[a] and (a < 3 or land_ok[a - 3] == 1)
            a2, o2 = _veto_choice(qv, red, base)
            if reason == _R_LANDING_V2:
                if o2 == OUTCOME_NO_SPACIOUS:
                    v5 = base
                    outcome = OUTCOME_NO_SPACIOUS
                    reason = _R_LANDING_NONE
                else:
                    v5 = a2
                    outcome = o2
            else:
                v5 = a2
                outcome = o2
        # ---- v7 re-rank
        v7 = v5
        v7_outcome = outcome
        if outcome != OUTCOME_NO_SPACIOUS and lam != 0.0 and reason != _R_SAME_DIR:
            off = 3 if v5 >= 3 else 0
            nc = 0
            for d in range(3):
                a = off + d
                if a == v5:
                    cand[nc] = a
                    nc += 1
                elif mask[a] and spac[d] and (off == 0 or land_ok[d] == 1):
                    cand[nc] = a
                    nc += 1
            if nc >= 2:
                best, _ = _space_pref(qv, cand, nc, v5, lam, area, acap, scratch)
                if best != v5:
                    v7 = best
                    v7_outcome = OUTCOME_VETOED
        # ---- v8 head layer
        final = v7
        final_outcome = v7_outcome
        if v7_outcome != OUTCOME_NO_SPACIOUS and risky_in[r, v7]:
            ne = 0
            for a in range(6):
                el = mask[a] and spac[a % 3] and not risky_in[r, a]
                if el and a >= 3:
                    el = land_ok[a - 3] == 1
                elig[a] = el
                if el and a != v7:
                    ne += 1
            if ne > 0:
                cb = v7 >= 3
                anchor = -1
                for a in range(6):
                    if a != v7 and elig[a] and (a >= 3) == cb:
                        if anchor < 0 or qv[a] > qv[anchor]:
                            anchor = a
                if anchor < 0:
                    for a in range(6):
                        if a != v7 and elig[a]:
                            if anchor < 0 or qv[a] > qv[anchor]:
                                anchor = a
                ab = anchor >= 3
                nc = 0
                for a in range(6):
                    if a != v7 and elig[a] and (a >= 3) == ab:
                        cand[nc] = a
                        nc += 1
                best, _ = _space_pref(qv, cand, nc, anchor, lam, area, acap, scratch)
                final = best
                final_outcome = OUTCOME_VETOED
        out_action[r] = final
        out_base[r] = base
        out_v5[r] = v5
        out_v7[r] = v7
        out_outcome[r] = final_outcome
        out_reason[r] = reason


def v8_redecide(res: V8Result, q: np.ndarray, mask: np.ndarray, lam: float = 8.0) -> V8Result:
    """The v8 decision for new Q, reusing a ``features=True`` result's diagnostics.

    ``res`` must come from :func:`v8_decide` with ``features=True`` on the same sim state,
    rows and ``mask`` (its diagnostics are Q-independent, so any Q, e.g. zeros, works there).
    Decision-identical to ``v8_decide(sim, rows, q, mask, lam)`` (pinned in
    ``tests/test_ego_raster_bv8.py``) without repeating a single flood. The returned result
    shares ``res``'s diagnostic arrays and carries the new decision fields.
    """
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lambda must be finite and >= 0, got {lam!r}")
    N = len(res.action)
    if N and (np.any(res.area < 0) or np.any(res.landing == _LANDING_UNKNOWN)):
        raise ValueError("v8_redecide needs a v8_decide(..., features=True) result")
    q = np.ascontiguousarray(q, dtype=np.float32).reshape(N, 6)
    mask = np.ascontiguousarray(mask, dtype=bool).reshape(N, 6)
    out = replace(
        res,
        action=np.zeros(N, dtype=np.int64),
        base=np.zeros(N, dtype=np.int64),
        v5_action=np.zeros(N, dtype=np.int64),
        v7_action=np.zeros(N, dtype=np.int64),
        outcome=np.zeros(N, dtype=np.int64),
        v5_reason=np.zeros(N, dtype=np.int64),
    )
    if not N:
        return out
    _redecide_rows(
        q, mask, _i64(res.counts), _i64(res.need), _i64(res.landing), _i64(res.area),
        _i64(res.area_cap), np.ascontiguousarray(res.head_risky, dtype=bool), lam,
        out.action, out.base, out.v5_action, out.v7_action, out.outcome, out.v5_reason,
    )  # fmt: skip
    return out
