"""Construct burned long-body Watch6 stress roots without claiming natural history.

The existing simulator scenario injector maintains every body/grid/mask mirror.
All six snakes occupy legal disjoint bands, with the requested length, at a
fixture-assigned frame 2500. These are timing fixtures, never efficacy worlds.
No gameplay rules, checkpoint bytes, RNG streams or policy internals are changed.
"""

from __future__ import annotations

import math
from typing import Any

from src.simd_env.grid_scenarios import inject_serpentines


def install(world: Any, target_length: int) -> dict[str, Any]:
    """Install a deterministic stress layout through the parity-tested injector."""
    sim = world.sim
    if sim.E != 1 or sim.S != 6 or target_length not in (800, 1200, 1500):
        raise ValueError("requires one Watch6 world and L800/L1200/L1500")
    height = math.ceil(target_length / (sim.grid_w - 4))
    inject_serpentines([sim], [target_length] * 6, box_height=height)
    sim.frame[:] = 2500
    sim.check_grid_invariants()
    if not sim.get_advisory_action_mask()[0, 0].any():
        raise ValueError("constructed hero has no legal dense action")
    return {
        "construction": "parity-tested inject_serpentines; six disjoint equal-length bands",
        "natural_history": False,
        "frame_origin": "fixture-assigned 2500, not 2500 simulated decisions",
        "opponent_carry_origin": "fresh evaluator carry, frozen with root",
        "body_lengths": [int(value) for value in sim.length[0]],
        "box_height": height,
    }
