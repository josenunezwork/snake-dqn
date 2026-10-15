"""Tail-aware static topology measurements for the T1 commitment diagnostic.

This module intentionally reads the simulation ring buffers rather than the grid
cache.  It therefore works for both ``BatchSim`` and ``GridBatchSim`` clones and
uses the study's explicit rule: an empty cell, or an *own* body cell that vacates
on the next transition (``ttl <= 1``), is free.  This deliberately differs from
``ego_raster_b``'s all-owner TTL rule; opponent cells always block this diagnostic.

``opponent_head_inside`` is literal: it is true only if a live opponent head is
itself a free member of the component.  Since ordinary live heads have TTL > 1,
the companion ``opponent_head_in_or_on_boundary`` is the useful pocket measure:
it also includes a blocked live head four-adjacent to the component.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # SciPy is a project dependency; retain the reference path for minimal envs.
    from scipy import ndimage
except ImportError:  # pragma: no cover - exercised only in stripped-down environments
    ndimage = None

from src.simd_env.batch_sim import CARDINAL

__all__ = ["component_geometry", "describe_action", "head_region", "sealer_identity"]

_NEIGHBORS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _cell_tuple(cell: np.ndarray) -> Tuple[int, int]:
    return int(cell[0]), int(cell[1])


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _label_components(blocked: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return labels and sizes for four-connected unblocked cells.

    The boards are only a few thousand cells.  This compact queue implementation
    avoids an optional SciPy dependency and is negligible beside a 300-frame
    counterfactual rollout.
    """
    if ndimage is not None:
        raw, count = ndimage.label(~blocked, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        labels = raw.astype(np.int32) - 1
        labels[blocked] = -1
        sizes = np.bincount(raw.ravel(), minlength=count + 1)[1:].astype(np.int64)
        return labels, sizes
    height, width = blocked.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    sizes: List[int] = []
    for y0 in range(height):
        for x0 in range(width):
            if blocked[y0, x0] or labels[y0, x0] >= 0:
                continue
            label = len(sizes)
            labels[y0, x0] = label
            queue = deque([(x0, y0)])
            size = 0
            while queue:
                x, y = queue.popleft()
                size += 1
                for dx, dy in _NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if (
                        _in_bounds(nx, ny, width, height)
                        and not blocked[ny, nx]
                        and labels[ny, nx] < 0
                    ):
                        labels[ny, nx] = label
                        queue.append((nx, ny))
            sizes.append(size)
    return labels, np.asarray(sizes, dtype=np.int64)


def _world_arrays(sim: Any, e: int) -> Dict[str, Any]:
    """Build tail-aware occupancy tables from one simulator world.

    Multiple live segments in a cell are not expected in a prepared world.  If a
    malformed synthetic world supplies them, the segment nearest its head wins;
    ties are deliberately retained as ambiguous metadata rather than guessed.
    """
    width, height = int(sim.grid_w), int(sim.grid_h)
    owners = np.full((height, width), -1, dtype=np.int32)
    offsets = np.full((height, width), -1, dtype=np.int32)
    ttl = np.zeros((height, width), dtype=np.int32)
    ambiguous = np.zeros((height, width), dtype=bool)
    alive = np.asarray(sim.alive[e], dtype=bool)
    length = np.asarray(sim.length[e], dtype=np.int64)
    cap = int(sim.cap)
    for slot in np.flatnonzero(alive):
        n = int(sim.seg_count[e, slot])
        head_ptr = int(sim.head_ptr[e, slot])
        for offset in range(n):
            x, y = _cell_tuple(sim.bodies[e, slot, (head_ptr - offset) % cap])
            if not _in_bounds(x, y, width, height):
                continue
            cell_ttl = int(length[slot]) - offset
            old = int(owners[y, x])
            if old < 0 or offset < int(offsets[y, x]):
                owners[y, x] = int(slot)
                offsets[y, x] = offset
                ttl[y, x] = cell_ttl
            elif old != int(slot) or offset != int(offsets[y, x]):
                ambiguous[y, x] = True
                ttl[y, x] = max(int(ttl[y, x]), cell_ttl)
    # The commitment spec is stricter than ego2s-b: tail release is only usable
    # for the hero's own body.  An opponent segment blocks even at TTL <= 1.
    blocked = (owners >= 0) & ~((owners == 0) & (ttl <= 1))
    labels, sizes = _label_components(blocked)
    return {
        "width": width,
        "height": height,
        "alive": alive,
        "length": length,
        "owners": owners,
        "offsets": offsets,
        "ttl": ttl,
        "blocked": blocked,
        "labels": labels,
        "sizes": sizes,
        "ambiguous": ambiguous,
    }


def component_geometry(sim: Any, e: int = 0) -> Dict[str, Any]:
    """Return opaque, reusable tail-aware component geometry for world ``e``.

    Consumers should treat this as valid only until ``sim`` is stepped.  It holds
    numpy arrays to make repeated action descriptions at a decision inexpensive.
    """
    geometry = _world_arrays(sim, e)
    geometry["env"] = int(e)
    geometry["hero_slot"] = 0
    geometry["head"] = _cell_tuple(sim.bodies[e, 0, int(sim.head_ptr[e, 0])])
    geometry["heads"] = [
        _cell_tuple(sim.bodies[e, slot, int(sim.head_ptr[e, slot])])
        for slot in range(int(np.asarray(sim.alive[e]).shape[0]))
    ]
    geometry["tail"] = _cell_tuple(
        sim.bodies[e, 0, (int(sim.head_ptr[e, 0]) - int(sim.seg_count[e, 0]) + 1) % int(sim.cap)]
    )
    geometry["direction"] = int(sim.direction[e, 0])
    return geometry


def _component_descriptor(geometry: Dict[str, Any], cell: Tuple[int, int]) -> Dict[str, Any]:
    """Describe a component at ``cell``, including the requested pocket fields."""
    width, height = geometry["width"], geometry["height"]
    x, y = cell
    base = {
        "next_cell": [int(x), int(y)],
        "in_bounds": _in_bounds(x, y, width, height),
        "free": False,
        "component_label": None,
        "component_size": 0,
        "tail_in_component": False,
        "tail_on_boundary": False,
        "tail_in_or_on_boundary": False,
        "min_own_boundary_ttl": None,
        "opponent_head_inside": False,
        "opponent_head_on_boundary": False,
        "opponent_head_in_or_on_boundary": False,
        "opponent_head_slots_inside": [],
        "opponent_head_slots_on_boundary": [],
    }
    if not base["in_bounds"] or geometry["blocked"][y, x]:
        return base
    label = int(geometry["labels"][y, x])
    if label < 0:
        return base
    base["free"] = True
    base["component_label"] = label
    base["component_size"] = int(geometry["sizes"][label])
    labels, owners, ttl = geometry["labels"], geometry["owners"], geometry["ttl"]
    tail_x, tail_y = geometry["tail"]
    base["tail_in_component"] = (
        _in_bounds(tail_x, tail_y, width, height) and int(labels[tail_y, tail_x]) == label
    )
    component = labels == label
    adjacent = np.zeros_like(component, dtype=bool)
    adjacent[1:] |= component[:-1]
    adjacent[:-1] |= component[1:]
    adjacent[:, 1:] |= component[:, :-1]
    adjacent[:, :-1] |= component[:, 1:]
    boundary = adjacent & ~component
    if _in_bounds(tail_x, tail_y, width, height):
        base["tail_on_boundary"] = bool(boundary[tail_y, tail_x])
    boundary_own_ttls = ttl[boundary & (owners == 0)]
    base["tail_in_or_on_boundary"] = bool(base["tail_in_component"] or base["tail_on_boundary"])
    if boundary_own_ttls.size:
        base["min_own_boundary_ttl"] = int(boundary_own_ttls.min())

    for slot in np.flatnonzero(geometry["alive"]):
        if int(slot) == 0:
            continue
        # Head coordinates come from the ring buffer: an opponent head normally
        # remains blocked, hence explicit boundary membership below.
        # The caller's simulator is not retained in geometry, so reconstructing
        # heads from owner/offset would be lossy for out-of-bounds heads.  They are
        # stored by component_geometry immediately after world extraction.
        hx, hy = geometry["heads"][int(slot)]
        if not _in_bounds(hx, hy, width, height):
            continue
        if int(labels[hy, hx]) == label:
            base["opponent_head_slots_inside"].append(int(slot))
        else:
            for dx, dy in _NEIGHBORS:
                nx, ny = hx + dx, hy + dy
                if _in_bounds(nx, ny, width, height) and int(labels[ny, nx]) == label:
                    base["opponent_head_slots_on_boundary"].append(int(slot))
                    break
    base["opponent_head_inside"] = bool(base["opponent_head_slots_inside"])
    base["opponent_head_on_boundary"] = bool(base["opponent_head_slots_on_boundary"])
    base["opponent_head_in_or_on_boundary"] = bool(
        base["opponent_head_inside"] or base["opponent_head_on_boundary"]
    )
    return base


def describe_action(
    sim: Any, action: int, e: int = 0, geometry: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Describe the tail-aware component entered by a hero action.

    Actions ``0..2`` and boosts ``3..5`` share their first entered cell by
    ``action % 3``.  The result additionally records the literal action and boost
    status so both action variants can appear independently in study rows.
    """
    if action < 0 or action > 5:
        raise ValueError(f"expected action in [0, 5], got {action}")
    if geometry is None:
        geometry = component_geometry(sim, e)
    if int(geometry["env"]) != int(e):
        raise ValueError("geometry was built for a different environment")
    rel = int(action) % 3
    direction = (int(geometry["direction"]) + rel - 1) % 4
    dx, dy = (int(v) for v in CARDINAL[direction])
    hx, hy = geometry["head"]
    out = _component_descriptor(geometry, (hx + dx, hy + dy))
    out.update(
        {
            "action": int(action),
            "relative_action": ("left", "straight", "right")[rel],
            "boost": bool(action >= 3),
            "hero_length": int(geometry["length"][0]),
            "frame": int(sim.frame[e]),
        }
    )
    return out


def head_region(sim: Any, e: int = 0, geometry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Describe the three components immediately on the hero's current head side.

    For the region series, the currently occupied hero head is temporarily opened
    as the seed (and no other blocked cell is opened).  This makes ``component_size``
    the exact component on the head side at decision time.  ``by_action`` retains
    action-next-cell measurements; boosts share the corresponding normal cell.
    """
    if geometry is None:
        geometry = component_geometry(sim, e)
    by_action = {
        name: describe_action(sim, action, e, geometry)
        for action, name in enumerate(("left", "straight", "right"))
    }
    blocked = geometry["blocked"].copy()
    hx, hy = geometry["head"]
    if _in_bounds(hx, hy, geometry["width"], geometry["height"]):
        blocked[hy, hx] = False
    head_labels, head_sizes = _label_components(blocked)
    head_label = (
        int(head_labels[hy, hx])
        if _in_bounds(hx, hy, geometry["width"], geometry["height"])
        else -1
    )
    head_size = int(head_sizes[head_label]) if head_label >= 0 else 0
    return {
        "frame": int(sim.frame[e]),
        "hero_length": int(geometry["length"][0]),
        "component_size": head_size,
        "component_label": head_label if head_label >= 0 else None,
        "by_action": by_action,
    }


def _head_side_labels(geometry: Dict[str, Any]) -> List[int]:
    """Labels reachable by one normal move from the current hero head."""
    hx, hy = geometry["head"]
    labels = []
    for dx, dy in _NEIGHBORS:
        x, y = hx + dx, hy + dy
        if _in_bounds(x, y, geometry["width"], geometry["height"]):
            label = int(geometry["labels"][y, x])
            if label >= 0:
                labels.append(label)
    return sorted(set(labels))


def _kind_for(kinds: Any, e: int, slot: int) -> Optional[str]:
    """Accept the census's per-env opponent-kind list or a slot-keyed mapping."""
    if kinds is None or slot == 0:
        return None
    try:
        if isinstance(kinds, dict):
            value = kinds.get(slot, kinds.get(str(slot)))
        else:
            # Census Recorder stores all environments as ``kinds[e]`` while the
            # per-death worker deliberately passes its already-selected flat
            # opponent list.  Strings are sequence-like, so test the first item
            # rather than relying on generic indexing.
            first = kinds[0]
            value = kinds[slot - 1] if isinstance(first, str) else kinds[e][slot - 1]
    except (IndexError, KeyError, TypeError):
        return None
    return None if value is None else str(value)


def _connected_cells(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Return four-connected ``(x, y)`` groups for a small boolean board mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups: List[List[Tuple[int, int]]] = []
    for y0, x0 in np.argwhere(mask):
        x0, y0 = int(x0), int(y0)
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        queue = deque([(x0, y0)])
        group: List[Tuple[int, int]] = []
        while queue:
            x, y = queue.popleft()
            group.append((x, y))
            for dx, dy in _NEIGHBORS:
                nx, ny = x + dx, y + dy
                if _in_bounds(nx, ny, width, height) and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        groups.append(sorted(group))
    return groups


def sealer_identity(before: Any, after: Any, kinds: Any, e: int = 0) -> Dict[str, Any]:
    """Conservatively identify a newly occupied cell that seals the head region.

    Newly occupied *and blocked* cells are clustered into four-connected closure
    groups.  A group is causal only if reopening it reconnects distinct post-step
    fragments of a pre-step head-side component.  Exactly one causal singleton
    can be attributed; a joint group or multiple causal groups is unresolved.
    """
    pre, post = component_geometry(before, e), component_geometry(after, e)
    pre_labels = _head_side_labels(pre)
    pre_occ = pre["owners"] >= 0
    post_occ = post["owners"] >= 0
    new_cells = np.argwhere(~pre_occ & post_occ)
    new_blocked = (~pre_occ & post_occ) & post["blocked"]
    causal_groups: List[Dict[str, Any]] = []
    for group in _connected_cells(new_blocked):
        restored_blocked = post["blocked"].copy()
        for x, y in group:
            restored_blocked[y, x] = False
        restored_labels, _ = _label_components(restored_blocked)
        evidence = []
        for old_label in pre_labels:
            members = (pre["labels"] == old_label) & ~post["blocked"]
            post_parts = sorted(set(int(v) for v in post["labels"][members] if int(v) >= 0))
            if len(post_parts) < 2:
                continue
            restored_parts = {}
            for part in post_parts:
                py, px = np.argwhere(post["labels"] == part)[0]
                restored_parts[part] = int(restored_labels[py, px])
            reconnects = {}
            for part, restored in restored_parts.items():
                if restored >= 0:
                    reconnects.setdefault(restored, []).append(part)
            bridge_labels = [label for label, parts in reconnects.items() if len(parts) >= 2]
            if bridge_labels:
                bridge_label = min(bridge_labels)
                evidence.append(
                    {
                        "pre_component_label": old_label,
                        "post_fragment_labels": post_parts,
                        "reconnected_fragment_labels": reconnects[bridge_label],
                        "restored_component_label": bridge_label,
                        "cells": [[x, y] for x, y in group],
                    }
                )
        if evidence:
            causal_groups.append({"cells": group, "evidence": evidence})
    candidates = [entry["cells"][0] for entry in causal_groups if len(entry["cells"]) == 1]
    joint_candidates = [
        cell for entry in causal_groups if len(entry["cells"]) > 1 for cell in entry["cells"]
    ]
    joint_evidence = [item for entry in causal_groups for item in entry["evidence"]]
    result: Dict[str, Any] = {
        "status": "none",
        "owner": "none",
        "slot": None,
        "kind": None,
        "cell": None,
        "candidate_cells": [[x, y] for x, y in candidates],
        "newly_occupied_cells": [[int(x), int(y)] for y, x in new_cells],
        "joint_candidate_cells": [[x, y] for x, y in sorted(set(joint_candidates))],
        "joint_evidence": joint_evidence,
    }
    if len(causal_groups) > 1:
        result.update({"status": "unresolved", "owner": "unresolved"})
        return result
    if len(causal_groups) == 1 and len(causal_groups[0]["cells"]) == 1:
        x, y = candidates[0]
        owner = int(post["owners"][y, x])
        if bool(post["ambiguous"][y, x]) or owner < 0:
            result.update({"status": "unresolved", "owner": "unresolved"})
            return result
        result["cell"] = [x, y]
        result["slot"] = owner
        if owner == 0:
            result.update({"status": "resolved", "owner": "hero"})
            return result
        kind = _kind_for(kinds, e, owner)
        if kind is None:
            result.update({"status": "unresolved", "owner": "unresolved"})
            return result
        result.update({"status": "resolved", "owner": "opponent", "kind": kind})
        return result
    if joint_candidates:
        result.update(
            {
                "status": "unresolved",
                "owner": "unresolved",
            }
        )
        return result
    return result
