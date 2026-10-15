"""Focused synthetic checks for T1's tail-aware topology helper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research.t1_commitment_20261015.topology import (
    component_geometry,
    describe_action,
    head_region,
    sealer_identity,
)


def _sim(width, height, snakes, direction=1, frame=17):
    """Make the small ring-buffer surface consumed by the topology module.

    ``snakes`` supplies head-first cells; each snake is alive and has logical
    length equal to the supplied segment count.
    """
    cap = max(1, max((len(cells) for cells in snakes), default=1))
    bodies = np.full((1, len(snakes), cap, 2), -99, dtype=np.int64)
    for slot, cells in enumerate(snakes):
        bodies[0, slot, : len(cells)] = cells
    return SimpleNamespace(
        grid_w=width,
        grid_h=height,
        cap=cap,
        bodies=bodies,
        head_ptr=np.zeros((1, len(snakes)), dtype=np.int64),
        seg_count=np.array([[len(cells) for cells in snakes]], dtype=np.int64),
        length=np.array([[len(cells) for cells in snakes]], dtype=np.int64),
        alive=np.ones((1, len(snakes)), dtype=bool),
        direction=np.full((1, len(snakes)), direction, dtype=np.int64),
        frame=np.array([frame], dtype=np.int64),
    )


def test_action_components_use_own_tail_rule_and_share_boost_next_cell():
    # Hero faces right at (2,2); its middle blocks, tail releases next frame.
    # The opponent's length-one head is still blocked by the study's own-only
    # release rule and sits on the large component's boundary.
    sim = _sim(5, 5, [[(2, 2), (2, 3), (2, 4)], [(4, 2)]])
    geometry = component_geometry(sim)
    normal = describe_action(sim, 1, geometry=geometry)
    boost = describe_action(sim, 4, geometry=geometry)

    assert normal["next_cell"] == [3, 2]
    assert normal["component_size"] == 22
    assert normal["tail_in_component"] is True
    assert normal["tail_in_or_on_boundary"] is True
    assert normal["min_own_boundary_ttl"] == 2
    assert normal["opponent_head_inside"] is False
    assert normal["opponent_head_on_boundary"] is True
    assert normal["opponent_head_in_or_on_boundary"] is True
    assert boost["next_cell"] == normal["next_cell"]
    assert boost["component_size"] == normal["component_size"]
    assert boost["boost"] is True


def test_head_region_opens_only_the_current_head_as_seed():
    sim = _sim(3, 1, [[(1, 0), (0, 0)]])
    region = head_region(sim)

    assert region["component_size"] == 3
    assert region["by_action"]["straight"]["component_size"] == 1


def test_sealer_identity_requires_a_proven_split_and_reports_owner_kind():
    # Two horizontal walls form a one-cell-wide corridor.  The newly occupied
    # centre cell separates the two remaining portions of the head-side corridor.
    walls = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    before = _sim(5, 3, [[(0, 1)], walls])
    # (4,1) is an unrelated newly occupied cell in the same old corridor.
    # It must not be credited: restoring it does not reconnect split pieces.
    after = _sim(5, 3, [[(0, 1)], walls + [(2, 1), (4, 1)]])

    # Per-death study workers pass their already-selected flat census kind list.
    sealer = sealer_identity(before, after, ["greedy_food"])

    assert sealer["status"] == "resolved"
    assert sealer["owner"] == "opponent"
    assert sealer["slot"] == 1
    assert sealer["kind"] == "greedy_food"
    assert sealer["cell"] == [2, 1]
    assert sealer["candidate_cells"] == [[2, 1]]


def test_sealer_identity_does_not_guess_when_no_cell_splits_a_region():
    before = _sim(4, 3, [[(0, 1)], [(3, 0)]])
    after = _sim(4, 3, [[(0, 1)], [(3, 0), (3, 2)]])

    assert sealer_identity(before, after, [["random_safe"]])["status"] == "none"
