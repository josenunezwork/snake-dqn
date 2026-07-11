"""Human-readable grouping of the 58-/61-D state vector for the inspector panel.

Mirrors src.core.game_config.StateIndices so the frontend can label each feature
without duplicating the index math.
"""

from __future__ import annotations

from typing import Dict, List

# (group name, start index, end index exclusive). Covers the 58-D base; the
# free-space group (58-61) is only present when use_free_space is enabled.
STATE_GROUPS_58: List[Dict[str, object]] = [
    {"name": "Direction (one-hot)", "start": 0, "end": 4},
    {"name": "Length (normalized)", "start": 4, "end": 5},
    {"name": "Food rel x/y + dist", "start": 5, "end": 8},
    {"name": "Food density (16 sectors)", "start": 8, "end": 24},
    {"name": "Danger map (16 sectors)", "start": 24, "end": 40},
    {"name": "Boundary dist (L/R/T/B)", "start": 40, "end": 44},
    {"name": "Nearest enemy (x/y/size)", "start": 44, "end": 47},
    {"name": "Enemy heading + trend", "start": 47, "end": 50},
    {"name": "2nd enemy (x/y/size)", "start": 50, "end": 53},
    {"name": "Kill opportunity", "start": 53, "end": 54},
    {"name": "Per-action danger (L/S/R)", "start": 54, "end": 57},
    {"name": "Boost available", "start": 57, "end": 58},
]

FREE_SPACE_GROUP: Dict[str, object] = {
    "name": "Free-space (L/S/R)",
    "start": 58,
    "end": 61,
}


def state_groups(input_size: int) -> List[Dict[str, object]]:
    """Return the inspector groups for the active state size (58 or 61)."""
    groups = list(STATE_GROUPS_58)
    if input_size >= 61:
        groups.append(dict(FREE_SPACE_GROUP))
    return groups


# Plain-ASCII action labels (the PyQt widget uses unicode arrows; the backend
# stays PyQt-free, so we define our own).
ACTION_LABELS: List[str] = [
    "Left",
    "Straight",
    "Right",
    "Boost L",
    "Boost S",
    "Boost R",
]
