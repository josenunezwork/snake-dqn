"""Pure, versioned scripted evaluation anchors shared by live and SIMD adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

SCRIPTED_ANCHOR_VERSION = "scripted-anchor/v1"
SCRIPTED_ANCHOR_KINDS = ("greedy_food", "random_safe")


@dataclass(frozen=True)
class AnchorContext:
    """Canonical state required for one anchor decision.

    The random stream is keyed by world, slot, and frame, rather than the
    batched array row, which makes action traces invariant to batch ordering.
    """

    world_seed: int
    slot: int
    frame: int
    head_cell: tuple[int, int]
    heading: tuple[int, int]
    food_cells: tuple[tuple[int, int], ...]
    allowed_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.allowed_mask)
        if mask.shape != (6,) or mask.dtype != np.bool_:
            raise ValueError("allowed_mask must be a boolean shape-(6,) vector")
        if len(self.head_cell) != 2 or len(self.heading) != 2:
            raise ValueError("head_cell and heading must be coordinate pairs")
        object.__setattr__(self, "allowed_mask", mask.copy())


def _normal_actions(context: AnchorContext) -> list[int]:
    return [action for action in range(3) if context.allowed_mask[action]]


def _next_cell(context: AnchorContext, relative_action: int) -> tuple[int, int]:
    dx, dy = context.heading
    if relative_action == 0:
        dx, dy = dy, -dx
    elif relative_action == 2:
        dx, dy = -dy, dx
    return context.head_cell[0] + dx, context.head_cell[1] + dy


def greedy_food_action(context: AnchorContext) -> int:
    """Choose nearest-food normal action; ties are straight, left, right."""
    safe = _normal_actions(context)
    if not safe:
        return 1
    priority = {1: 0, 0: 1, 2: 2}
    if not context.food_cells:
        return min(safe, key=priority.__getitem__)

    def score(action: int) -> tuple[int, int]:
        head = _next_cell(context, action)
        distance = min((head[0] - x) ** 2 + (head[1] - y) ** 2 for x, y in context.food_cells)
        return distance, priority[action]

    return min(safe, key=score)


def random_safe_action(context: AnchorContext) -> int:
    """Uniform deterministic selection keyed to canonical world/slot/frame."""
    safe = _normal_actions(context)
    if not safe:
        return 1
    key = (
        f"{SCRIPTED_ANCHOR_VERSION}|random_safe|{context.world_seed}|{context.slot}|{context.frame}"
    )
    index = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % len(safe)
    return safe[index]


class ScriptedAnchor:
    """Immutable handle for the common versioned anchor rule."""

    def __init__(self, kind: str) -> None:
        if kind not in SCRIPTED_ANCHOR_KINDS:
            raise ValueError(f"unknown scripted anchor kind {kind!r}")
        self.kind = kind
        self.version = SCRIPTED_ANCHOR_VERSION

    @property
    def descriptor(self) -> dict[str, str]:
        return {"kind": self.kind, "version": self.version}

    def action(self, context: AnchorContext) -> int:
        if self.kind == "greedy_food":
            return greedy_food_action(context)
        return random_safe_action(context)
