"""Parity: opt-in v5 boost-aware veto on the SIMD vector61 path vs the live hook.

The reference is the live :class:`src.evaluation.safety_veto_v5.BoostAwareFreeSpaceVeto`
(installed on the hero of the real ``tournament_eval.rollout`` the way the v5 screen
does it: the built-in install runs first, then ``install_boost_aware_veto``). The
candidate is ``run_simd_eval(..., vector61=True, hero_safety_veto="v5")``.

Three layers:

* the BatchSim landing count (:func:`boost_landing_count`) against the live
  ``landing_count`` on constructed trap pockets (every direction, burn phase and
  boost eligibility);
* the SIMD veto hook against the live hook on constructed decisions where the
  landing veto fires (same direction normal speed, v2 rule, nothing eligible);
* whole rollouts: every vector decision and selection state, every hero veto
  decision (v2 counts, ``need``, base, final, the lazily flooded landings and an
  eager landing table for all three directions), the record and the v5 diagnostics.

``hero_safety_veto=True`` must stay the v2 veto exactly; the v2 parity suite
(``tests/test_simd_vector61_policy.py``) still pins that path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from src.simd_env.vector61_policy import (
    boost_landing_count,
    resolve_safety_veto,
    vector61_provenance,
    veto_threshold,
)
from tests.test_simd_vector61_policy import (  # noqa: F401  (fixtures are used by name)
    SIMD_ONLY_KEYS,
    _compare,
    _first_state_divergence,
    _run_live,
    _run_simd,
    _tiny_rosters,
    deployment_world,
    needs_real_pool,
    tiny_world,
)

SS = 10
TIMING_KEYS = {"apply_seconds_total", "apply_seconds_max", "mean_apply_seconds"}
# Not-flooded / boost-would-not-fire codes in a landing tuple (as last_veto["landing"]).
NOT_FLOODED = -1
NO_BOOST = -2
Q_PREFERS_BOOST_STRAIGHT = [0.0, 1.0, 0.5, 0.0, 9.0, 0.2]


def deterministic(diagnostics: Dict[str, object]) -> Dict[str, object]:
    """v5 diagnostics minus the wall-clock fields (they differ run to run)."""
    return {k: v for k, v in diagnostics.items() if k not in TIMING_KEYS}


# ---------------------------------------------------------------------------
# Variant resolution and provenance
# ---------------------------------------------------------------------------
def test_resolve_safety_veto_keeps_true_as_v2_and_names_v5():
    assert resolve_safety_veto(False) is None
    assert resolve_safety_veto(None) is None
    assert resolve_safety_veto(np.bool_(False)) is None
    assert resolve_safety_veto(True) == "v2"
    assert resolve_safety_veto("v2") == "v2"
    assert resolve_safety_veto("v5") == "v5"
    for bad in ("v3", "V5", 1, "true"):
        with pytest.raises(ValueError, match="hero_safety_veto must be"):
            resolve_safety_veto(bad)


def test_provenance_unchanged_for_v2_and_names_the_v5_method():
    from src.evaluation.safety_veto_v5 import VETO_METHOD_V5

    v2 = {
        "engine": "simd",
        "policy": "Vector61SimdPolicy",
        "forward": "rowwise",
        "bit_exact_forward": True,
        "hero_safety_veto": True,
    }
    assert vector61_provenance("rowwise", True) == v2
    assert vector61_provenance("rowwise", "v2") == v2
    assert vector61_provenance("rowwise", False) == {**v2, "hero_safety_veto": False}
    assert vector61_provenance("rowwise", "v5") == {**v2, "safety_veto_method": VETO_METHOD_V5}


# ---------------------------------------------------------------------------
# Constructed worlds: one BatchSim env mirroring live Snake objects
# ---------------------------------------------------------------------------
def body_path(head, n, height):
    """``n`` cells: left from ``head`` to x = 0, down x = 0, then right along the bottom."""
    hx, hy = head
    cells = [(x, hy) for x in range(hx, -1, -1)]
    cells += [(0, y) for y in range(hy + 1, height)]
    cells += [(x, height - 1) for x in range(1, 10_000)]
    return cells[:n]


def trap_cells(length: int, pocket: int) -> Tuple[List[tuple], List[tuple]]:
    """Hero heading right at (60, 40); another snake walls a 1-wide corridor ahead.

    Straight at normal speed lands on (61, 40) with the open board around it; a
    straight boost lands on (62, 40) with (61, 40) behind it, inside a ``pocket``-cell
    pocket (the trap-horizon pattern of ``tests/test_safety_veto_v5.py``).
    """
    me = body_path((60, 40), length, 80)
    walls = [(62 + i, 39) for i in range(pocket)] + [(62 + i, 41) for i in range(pocket)]
    walls.append((62 + pocket, 40))
    return me, walls


def live_snake(cells, direction, length=None, boost_frames=0, sid=0, board=(100, 80)):
    from src.game.snake import Snake

    head = cells[0]
    snake = Snake(sid, (255, 0, 0), (head[0] * SS, head[1] * SS), SS, board[0] * SS, board[1] * SS)
    snake.segments = [(x * SS, y * SS) for x, y in cells]
    snake.length = len(cells) if length is None else int(length)
    snake.direction = direction
    snake.boost_frames = int(boost_frames)
    return snake


def batch_world(bodies: Sequence[dict], board=(100, 80)):
    """A 1-env BatchSim whose snakes are exactly ``bodies`` (cells head first)."""
    from src.simd_env.batch_sim import CARDINAL, BatchSim, BatchSimConfig

    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=max(len(bodies), 2),
        game_width=board[0] * SS,
        game_height=board[1] * SS,
        initial_food=0,
        max_food=0,
    )
    sim = BatchSim(cfg, seeds=[0], train_mode=False)
    sim.alive[0, :] = False
    cardinal = [tuple(int(v) for v in row) for row in CARDINAL.tolist()]
    for slot, body in enumerate(bodies):
        cells = body["cells"]
        n = len(cells)
        sim.head_ptr[0, slot] = n - 1
        for k, cell in enumerate(cells):
            sim.bodies[0, slot, (n - 1 - k) % sim.cap] = cell
        sim.seg_count[0, slot] = n
        sim.length[0, slot] = int(body.get("length", n))
        sim.direction[0, slot] = cardinal.index(tuple(body["direction"]))
        sim.boost_frames[0, slot] = int(body.get("boost_frames", 0))
        sim.alive[0, slot] = True
    return sim


@pytest.mark.parametrize("boost_frames", [0, 1, 2])
@pytest.mark.parametrize(
    "length, pocket, logical",
    [(4, 5, None), (8, 4, None), (30, 6, None), (120, 12, None), (40, 3, 45), (200, 30, None)],
)
def test_landing_count_matches_live_on_trap_pockets(
    setup_config, length, pocket, logical, boost_frames
):
    """Every direction; burn phase via ``boost_frames``; ``length < 5`` cannot boost."""
    from src.evaluation.safety_veto import free_space_threshold
    from src.evaluation.safety_veto_v5 import landing_count

    me_cells, walls = trap_cells(length, pocket)
    me = live_snake(me_cells, (1, 0), length=logical, boost_frames=boost_frames)
    other = live_snake(walls, (1, 0), sid=1)
    sim = batch_world(
        [
            {"cells": me_cells, "direction": (1, 0), "length": me.length, "boost_frames": boost_frames},
            {"cells": walls, "direction": (1, 0)},
        ]
    )
    cap, _need = free_space_threshold(me.length, me._logical_length())
    assert int(veto_threshold(np.array([me.length]))[0][0]) == cap
    seen = []
    for direction in range(3):
        live = landing_count(me, [me, other], direction, cap)
        simd = boost_landing_count(sim, 0, 0, direction, cap, (100, 80))
        assert simd == live, (direction, live, simd)
        seen.append(live)
    if length < 5:
        assert seen == [None, None, None]
    else:
        assert seen[1] < cap  # the straight boost lands inside the pocket


def test_landing_count_ignores_dead_snakes_and_blocks_off_grid_landings(setup_config):
    """A dead snake is not a wall; a landing off the board counts 0 (as live)."""
    from src.evaluation.safety_veto_v5 import landing_count

    me_cells = [(98, 10), (97, 10), (96, 10), (95, 10), (94, 10), (93, 10)]
    me = live_snake(me_cells, (1, 0))
    sim = batch_world([{"cells": me_cells, "direction": (1, 0)}, {"cells": [(5, 5)], "direction": (1, 0)}])
    sim.alive[0, 1] = False
    for direction in range(3):
        assert boost_landing_count(sim, 0, 0, direction, 32, (100, 80)) == landing_count(
            me, [me], direction, 32
        )
    assert boost_landing_count(sim, 0, 0, 1, 32, (100, 80)) == 0


class _LiveCountsRuntime:
    """Runtime stand-in: v2 counts from the live features (the v2 path is pinned elsewhere)."""

    class _Grid:
        gw, gh = 100, 80

    featurizer = _Grid()

    def __init__(self, counts) -> None:
        self.counts = np.asarray([counts], dtype=np.int64)

    def decision_free_space(self, sim, slots):
        return self.counts


def simd_hook(counts):
    """A bare v5 ``Vector61SimdPolicy`` (no checkpoint) on a stand-in runtime."""
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    policy = Vector61SimdPolicy.__new__(Vector61SimdPolicy)
    policy.runtime = _LiveCountsRuntime(counts)
    policy.veto_slots = frozenset({0})
    policy.veto_variant = "v5"
    policy.veto_counters = {}
    policy.boost_counters = {}
    policy.last_veto = None
    return policy


@pytest.mark.parametrize(
    "mask, expected",
    [
        ([True] * 6, 1),  # landing veto -> same direction at normal speed
        ([True, False, True, True, True, True], 5),  # normal straight masked -> v2 rule
        ([False, False, False, False, True, False], 4),  # nothing eligible -> unchanged
    ],
)
@pytest.mark.parametrize("length, pocket", [(50, 1), (50, 24), (160, 5)])
def test_simd_hook_matches_the_live_hook_when_the_landing_veto_fires(
    setup_config, mask, expected, length, pocket
):
    from src.evaluation.safety_veto import free_space_threshold
    from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto

    me_cells, walls = trap_cells(length, pocket)
    me = live_snake(me_cells, (1, 0))
    other = live_snake(walls, (1, 0), sid=1)
    roster = [me, other]
    cap, _need = free_space_threshold(me.length, me._logical_length())
    counts = [round(f * cap) for f in me._get_free_space_features(roster)]
    q = torch.tensor(Q_PREFERS_BOOST_STRAIGHT)
    live = BoostAwareFreeSpaceVeto()
    live_action = live.apply(me, roster, q, torch.tensor(mask), 4)
    assert live_action == expected

    sim = batch_world(
        [{"cells": me_cells, "direction": (1, 0)}, {"cells": walls, "direction": (1, 0)}]
    )
    policy = simd_hook(counts)
    out = policy._apply_veto(
        sim, np.array([[0, 0]]), q.unsqueeze(0), np.array([mask]), np.array([4])
    )
    assert int(out[0]) == live_action
    assert policy.veto_record(0) == live.record()
    simd_diag = policy.veto_diagnostics(0)
    assert deterministic(simd_diag) == deterministic(live.diagnostics_record())
    assert simd_diag["base_landing_failed"] == 1
    landing = policy.last_veto["landing"][0].tolist()
    assert landing[1] == pocket  # the straight boost's two-cell landing
    assert all(c == NOT_FLOODED or c >= _need for c in (landing[0], landing[2]))
    assert policy.last_veto["reason"][0].startswith("landing_")


def test_v2_variant_never_floods_landings_and_keeps_the_boost(setup_config):
    """The same trap through the v2 path: kept (v2's first-cell approximation)."""
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    me_cells, walls = trap_cells(50, 5)
    sim = batch_world(
        [{"cells": me_cells, "direction": (1, 0)}, {"cells": walls, "direction": (1, 0)}]
    )
    policy = simd_hook([100, 100, 100])
    policy.veto_variant = "v2"
    out = policy._apply_veto(
        sim,
        np.array([[0, 0]]),
        torch.tensor([Q_PREFERS_BOOST_STRAIGHT]),
        np.array([[True] * 6]),
        np.array([4]),
    )
    assert int(out[0]) == 4 and "landing" not in policy.last_veto
    assert policy.veto_record(0)["method"] == "free-space-veto/v2-speed-preserving"
    with pytest.raises(ValueError, match="only for the v5 veto"):
        policy.veto_diagnostics(0)
    with pytest.raises(ValueError, match="veto_variant must be one of"):
        Vector61SimdPolicy("unused.pth", None, veto_variant="v4")
