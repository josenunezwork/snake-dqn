"""Plain-NumPy T5 counterfactual topology features.

This intentionally small reference implementation is for the T5 separability
study.  It has no numba dependency and does not alter simulator state.  Each
action has four values, in action-major order: uncapped tail-aware area, a
three-frame boosted look-ahead area, bottleneck-pruned area, and whether the
tail can be reached.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from src.simd_env.batch_sim import CARDINAL

__all__ = ["FEATURE_NAMES", "compute_features", "compute_slot_features"]

FEATURE_NAMES = ("area_log", "boost3_area_log", "core_area_log", "tail_reachable")
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _body(sim, env: int, slot: int) -> List[Tuple[int, int]]:
    """Return the live ring-buffer body ordered head to tail."""
    n, cap, hp = int(sim.seg_count[env, slot]), int(sim.cap), int(sim.head_ptr[env, slot])
    return [tuple(map(int, sim.bodies[env, slot, (hp - offset) % cap])) for offset in range(n)]


def _log_area(area: int) -> float:
    return float(np.log2(min(area, 4096) + 1) / 12.0)


def _advance(
    body: Sequence[Tuple[int, int]],
    length: int,
    boost_frames: int,
    direction: int,
    boosted: bool,
    min_boost_length: int,
    boost_cost: int,
    width: int,
    height: int,
    opponents: Set[Tuple[int, int]],
    *,
    strict_boost: bool = False,
) -> Tuple[List[Tuple[int, int]], int, int] | None:
    """Advance one collision-free frame, preserving BatchSim's tail cadence.

    Opponents are static by design for these T5 counterfactuals.  Own tails
    vacate after an insertion when the physical body has reached logical
    length, which also preserves the simulator's post-food fill-in lag.
    """
    if boosted and length < min_boost_length:
        # v8's one-step post-move model treats an unavailable boost action as
        # its normal-speed counterpart.  The boost-three feasibility feature
        # opts into strict mode because it requires three actual boost frames.
        if strict_boost:
            return None
        boosted = False
    out = list(body)
    dx, dy = map(int, CARDINAL[direction])
    steps = 2 if boosted else 1
    traversed = []
    for _ in range(steps):
        x, y = out[0]
        nxt = (x + dx, y + dy)
        if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
            return None
        if nxt in opponents:
            return None
        traversed.append(nxt)
        out.insert(0, nxt)
        if len(out) > length:
            out.pop()
    if boosted:
        boost_frames += 1
        if boost_frames >= boost_cost:
            boost_frames = 0
            length = max(1, length - 1)
            if len(out) > length:
                out.pop()
    # BatchSim resolves self collisions against the completed post-frame body:
    # normal traversals may meet offsets >= 2, boosted traversals offsets >= 1,
    # which is exactly the common post-body offset three after inserting one or
    # two heads.  Checking the completed body also admits the two tails that a
    # boost vacates in the same frame.
    if set(traversed) & set(out[3:]):
        return None
    return out, length, boost_frames


def _release_map(
    body: Sequence[Tuple[int, int]], length: int
) -> Tuple[Dict[Tuple[int, int], int], Set[Tuple[int, int]]]:
    """v8/v3 release distances (slack one) and physical TTL-one tail cells."""
    release: Dict[Tuple[int, int], int] = {}
    tail_cells: Set[Tuple[int, int]] = set()
    for index, cell in enumerate(body):
        # v8 _landing_area uses L - index + 1: v3's conservative slack one.
        release[cell] = max(release.get(cell, -1), length - index + 1)
        if length - index <= 1:
            tail_cells.add(cell)
    release.pop(body[0], None)  # The new head is the BFS seed.
    return release, tail_cells


def _flood(
    start: Tuple[int, int],
    width: int,
    height: int,
    permanently_blocked: Set[Tuple[int, int]],
    release: Dict[Tuple[int, int], int],
    tail_cells: Set[Tuple[int, int]],
    core: bool = False,
) -> Tuple[int, bool]:
    """Uncapped v8-order tail-aware BFS, optionally excluding narrow cells."""

    def tail_passable(cell: Tuple[int, int], distance: int) -> bool:
        x, y = cell
        return (
            0 <= x < width
            and 0 <= y < height
            and cell not in permanently_blocked
            and (cell not in release or distance >= release[cell])
        )

    def passable(cell: Tuple[int, int], distance: int) -> bool:
        if not tail_passable(cell, distance):
            return False
        if not core:
            return True
        # The core is a one-pass degree filter, never iterative erosion.  Its
        # degree is evaluated in the same unpruned tail-aware world at the
        # candidate's arrival distance, so own cells still waiting to vacate
        # cannot make a closing neck look open.
        x, y = cell
        free = sum(tail_passable((x + dx, y + dy), distance) for dx, dy in _NEIGHBOURS)
        return free > 2

    if not passable(start, 0):
        return 0, False
    seen = {start}
    queue = deque([(start, 0)])
    touched_tail = start in tail_cells
    while queue:
        cell, distance = queue.popleft()
        x, y = cell
        for dx, dy in _NEIGHBOURS:
            nxt = (x + dx, y + dy)
            if nxt not in seen and passable(nxt, distance + 1):
                seen.add(nxt)
                touched_tail |= nxt in tail_cells
                queue.append((nxt, distance + 1))
    return len(seen), touched_tail


def _post_move_features(
    body: Sequence[Tuple[int, int]],
    length: int,
    width: int,
    height: int,
    opponents: Set[Tuple[int, int]],
) -> Tuple[int, int, bool]:
    release, tail_cells = _release_map(body, length)
    area, tail = _flood(body[0], width, height, opponents, release, tail_cells)
    core, _ = _flood(body[0], width, height, opponents, release, tail_cells, core=True)
    return area, core, tail


def _post_move_area(
    body: Sequence[Tuple[int, int]],
    length: int,
    width: int,
    height: int,
    opponents: Set[Tuple[int, int]],
) -> int:
    """Return only ordinary post-move area for the boost-three look-ahead."""
    release, tail_cells = _release_map(body, length)
    area, _ = _flood(body[0], width, height, opponents, release, tail_cells)
    return area


def compute_slot_features(sim, env: int, slot: int) -> np.ndarray:
    """Compute the 24 T5 values for one live ``(env, slot)`` row.

    Dead rows intentionally return zeros.  ``boost3_area_log`` is direction
    based, so the normal and boost actions for the same turn share it; the
    one-step post-move features retain the actual action speed.
    """
    out = np.zeros((6, 4), dtype=np.float32)
    if not bool(sim.alive[env, slot]):
        return out.reshape(-1)
    width, height = int(sim.grid_w), int(sim.grid_h)
    length = int(sim.length[env, slot])
    boost_frames = int(sim.boost_frames[env, slot])
    heading = int(sim.direction[env, slot])
    mine = _body(sim, env, slot)
    opponents: Set[Tuple[int, int]] = set()
    for other in range(int(sim.S)):
        if other != slot and bool(sim.alive[env, other]):
            opponents.update(_body(sim, env, other))
    min_boost = int(sim.cfg.min_boost_length)
    boost_cost = int(sim.cfg.boost_length_cost_frames)

    # The boost-three counterfactual only depends on the resolved direction,
    # not on whether the one-step action was normal or boosted.
    boost3 = np.zeros(3, dtype=np.float32)
    for relative in range(3):
        direction = (heading + relative - 1) % 4
        future_body, future_length, future_frames = list(mine), length, boost_frames
        feasible = True
        for _ in range(3):
            future = _advance(
                future_body,
                future_length,
                future_frames,
                direction,
                True,
                min_boost,
                boost_cost,
                width,
                height,
                opponents,
                strict_boost=True,
            )
            if future is None:
                feasible = False
                break
            future_body, future_length, future_frames = future
        if feasible:
            boost3[relative] = _log_area(
                _post_move_area(future_body, future_length, width, height, opponents)
            )

    for action in range(6):
        relative = action % 3
        direction = (heading + relative - 1) % 4
        moved = _advance(
            mine,
            length,
            boost_frames,
            direction,
            action >= 3,
            min_boost,
            boost_cost,
            width,
            height,
            opponents,
        )
        if moved is not None:
            post_body, post_length, _ = moved
            area, core, tail = _post_move_features(post_body, post_length, width, height, opponents)
            out[action, 0] = _log_area(area)
            out[action, 2] = _log_area(core)
            out[action, 3] = float(tail)
        out[action, 1] = boost3[relative]
    return out.reshape(-1)


def compute_features(sim) -> np.ndarray:
    """Return T5 features as float32 ``(E, S, 24)`` without mutating ``sim``."""
    out = np.zeros((int(sim.E), int(sim.S), 24), dtype=np.float32)
    for env in range(int(sim.E)):
        for slot in range(int(sim.S)):
            out[env, slot] = compute_slot_features(sim, env, slot)
    return out
