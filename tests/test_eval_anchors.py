"""Canonical anchor tests independent of either execution engine."""

import numpy as np

from src.evaluation.anchors import AnchorContext, ScriptedAnchor, greedy_food_action


def _context(world: int, slot: int, frame: int, *, food=()) -> AnchorContext:
    return AnchorContext(
        world_seed=world,
        slot=slot,
        frame=frame,
        head_cell=(5, 5),
        heading=(1, 0),
        food_cells=tuple(food),
        allowed_mask=np.array([True, True, True, False, False, False]),
    )


def test_greedy_food_uses_straight_left_right_ties() -> None:
    # Food directly ahead leaves all three next-cell distances tied only when
    # there is no food; straight is the documented no-food tie winner.
    assert greedy_food_action(_context(1, 1, 0)) == 1
    assert greedy_food_action(_context(1, 1, 0, food=((5, 4),))) == 0


def test_random_safe_is_stable_for_sparse_reordered_world_slot_rows() -> None:
    anchor = ScriptedAnchor("random_safe")
    contexts = [_context(91, 4, 7), _context(3, 1, 8), _context(91, 1, 7)]
    first = {(c.world_seed, c.slot, c.frame): anchor.action(c) for c in contexts}
    second = {(c.world_seed, c.slot, c.frame): anchor.action(c) for c in reversed(contexts)}
    assert first == second
    # Equivalent canonical context is the live/SIMD cross-engine contract.
    assert anchor.action(_context(91, 4, 7)) == first[(91, 4, 7)]
