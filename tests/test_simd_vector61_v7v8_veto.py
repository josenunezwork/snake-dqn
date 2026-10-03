"""Parity: opt-in v7 / v8 vetoes on the SIMD vector61 path vs the live hooks.

The reference is the live hook object itself
(:class:`src.evaluation.safety_veto_v7.SpacePreferenceVeto` and
:class:`src.evaluation.safety_veto_v8.SpaceAndHeadVeto`) applied to real live
``Snake`` objects on pixel coordinates. The candidate is
``Vector61SimdPolicy._apply_veto`` with ``veto_variant="v7"``/``"v8"`` on a
BatchSim holding the same snakes on the cell lattice. Every comparison checks the
returned action, the probe record (descriptor plus v2's seven counters) and the
deterministic part of ``diagnostics_record()`` (every field except wall-clock
``*seconds*`` timings), counter for counter.

Constructed states only (no episode is played):

* named scenarios: a pocket where v7's re-rank fires, a head-on where v8's head
  layer fires (and the hero-wins waiver), a boost into a trap pocket (v5's landing
  veto), Q ties, a boost-mode re-rank and a ``no_spacious`` decision, each at
  ``lambda`` in {0, 1, 4, 8, 16} for v7, v8 and v8 with ``reference_lambda=4``;
* a seeded fuzz over crowded random boards (several envs per call, decisions
  accumulated on one hook per env), with coverage floors so it is not vacuous.

Whole-rollout parity (tiny world, real ``tournament_eval.rollout``) is in the
same file but runs only with ``SNAKE_SIMD_ROLLOUT_PARITY=1`` (it plays episodes).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from src.simd_env.vector61_policy import (
    LAMBDA_VETO_VARIANTS,
    SafetyVetoSpec,
    SimRowSnake,
    live_veto_hook,
    resolve_safety_veto,
    resolve_safety_veto_spec,
    sim_row_roster,
    vector61_provenance,
)

REPO = Path(__file__).resolve().parents[1]
DEPLOYMENT_YAML = REPO / "research" / "apex_safety_20260926" / "deployment.yaml"
SS = 10
INVALID_Q = -1.0e9
LAMBDAS = (0.0, 1.0, 4.0, 8.0, 16.0)
SPECS = [SafetyVetoSpec("v7", lam) for lam in LAMBDAS]
SPECS += [SafetyVetoSpec("v8", lam) for lam in LAMBDAS]
SPECS += [SafetyVetoSpec("v8", lam, 4.0) for lam in LAMBDAS]


def spec_id(spec: SafetyVetoSpec) -> str:
    ref = "" if spec.reference_lambda is None else f"-ref{spec.reference_lambda:g}"
    return f"{spec.variant}-l{spec.lam:g}{ref}"


def deterministic(value: Any) -> Any:
    """``value`` without wall-clock fields (any key containing ``seconds``), recursively."""
    if isinstance(value, dict):
        return {k: deterministic(v) for k, v in value.items() if "seconds" not in k}
    return value


# ---------------------------------------------------------------------------
# Constructed worlds: live Snake objects and a BatchSim with the same snakes
# ---------------------------------------------------------------------------
@pytest.fixture
def v2_world(setup_config):
    """The gate's deployment config (mechanics v2: the head-on size rule is live)."""
    from src.core.config_loader import load_and_initialize_config
    from src.core.game_config import GameConfig

    load_and_initialize_config(str(DEPLOYMENT_YAML))
    assert GameConfig.MECHANICS_VERSION == 2 and GameConfig.ARENA_TYPE == "rectangular"
    yield
    random.seed()


def live_snake(cells, direction, length=None, boost_frames=0, sid=0, board=(100, 80), alive=True):
    from src.game.snake import Snake

    head = cells[0]
    snake = Snake(sid, (255, 0, 0), (head[0] * SS, head[1] * SS), SS, board[0] * SS, board[1] * SS)
    snake.segments = [(x * SS, y * SS) for x, y in cells]
    snake.length = len(cells) if length is None else int(length)
    snake.direction = tuple(direction)
    snake.boost_frames = int(boost_frames)
    snake.is_alive = bool(alive)
    return snake


def batch_worlds(worlds: Sequence[Sequence[dict]], board=(100, 80)):
    """An ``E = len(worlds)`` BatchSim; env ``e`` holds exactly ``worlds[e]``'s snakes."""
    from src.core.game_config import GameConfig
    from src.simd_env.batch_sim import CARDINAL, BatchSim, BatchSimConfig

    slots = max(max(len(w) for w in worlds), 2)
    cfg = BatchSimConfig(
        num_envs=len(worlds),
        num_snakes=slots,
        game_width=board[0] * SS,
        game_height=board[1] * SS,
        initial_food=0,
        max_food=0,
        min_boost_length=int(GameConfig.MIN_BOOST_LENGTH),
        boost_length_cost_frames=int(GameConfig.BOOST_LENGTH_COST_FRAMES),
        mechanics_version=int(GameConfig.MECHANICS_VERSION),
    )
    sim = BatchSim(cfg, seeds=list(range(len(worlds))), train_mode=False)
    sim.alive[:, :] = False
    cardinal = [tuple(int(v) for v in row) for row in CARDINAL.tolist()]
    for env, bodies in enumerate(worlds):
        for slot, body in enumerate(bodies):
            cells = body["cells"]
            n = len(cells)
            assert n <= sim.cap
            sim.head_ptr[env, slot] = n - 1
            for k, cell in enumerate(cells):
                sim.bodies[env, slot, (n - 1 - k) % sim.cap] = cell
            sim.seg_count[env, slot] = n
            sim.length[env, slot] = int(body.get("length", n))
            sim.direction[env, slot] = cardinal.index(tuple(body["direction"]))
            sim.boost_frames[env, slot] = int(body.get("boost_frames", 0))
            sim.alive[env, slot] = bool(body.get("alive", True))
    return sim


def live_roster(bodies: Sequence[dict], board=(100, 80)):
    return [
        live_snake(
            b["cells"],
            b["direction"],
            length=b.get("length"),
            boost_frames=b.get("boost_frames", 0),
            sid=slot,
            board=board,
            alive=b.get("alive", True),
        )
        for slot, b in enumerate(bodies)
    ]


class _CountsRuntime:
    """Runtime stand-in returning the live hero's v2 counts per decision row."""

    def __init__(self, board) -> None:
        class _Grid:
            gw, gh = board

        self.featurizer = _Grid()
        self.counts: Optional[np.ndarray] = None

    def decision_free_space(self, sim, slots):
        assert self.counts is not None and len(self.counts) == len(slots)
        return self.counts


def simd_hook(spec: SafetyVetoSpec, board=(100, 80)):
    """A bare v7/v8 ``Vector61SimdPolicy`` (no checkpoint) on a stand-in runtime."""
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    policy = Vector61SimdPolicy.__new__(Vector61SimdPolicy)
    policy.runtime = _CountsRuntime(board)
    policy._configure_veto((0,), spec.variant, spec.lam, spec.reference_lambda)
    return policy


def live_counts(hero, roster) -> List[int]:
    from src.evaluation.safety_veto import free_space_threshold

    cap, _need = free_space_threshold(hero.length, hero._logical_length())
    return [round(float(f) * cap) for f in hero._get_free_space_features(roster)]


def masked_row(q: Sequence[float], mask: Sequence[bool]) -> Tuple[List[float], List[bool], int]:
    """The live ``AISnake`` call: legacy normal-only fallback, masked Q, first argmax."""
    mask = [bool(m) for m in mask]
    if not any(mask):
        mask = [True, True, True, False, False, False]
    masked = [float(np.float32(v)) if m else INVALID_Q for v, m in zip(q, mask)]
    base = int(torch.tensor(masked).argmax().item())
    return masked, mask, base


def run_both(
    spec: SafetyVetoSpec,
    worlds: Sequence[Sequence[dict]],
    decisions: Sequence[Sequence[Tuple[Sequence[float], Sequence[bool]]]],
    board=(100, 80),
):
    """Apply ``decisions[e]`` (in order) to env ``e`` on both sides; compare everything.

    Each env has one live hook (as one live episode) and the SIMD policy one hook per
    env; every SIMD call carries one decision for every env (several rows per call).
    Returns ``(live hooks, simd policy, actions per env)``.
    """
    sim = batch_worlds(worlds, board)
    rosters = [live_roster(w, board) for w in worlds]
    hooks = [live_veto_hook(spec) for _ in worlds]
    policy = simd_hook(spec, board)
    counts = np.array([live_counts(r[0], r) for r in rosters], dtype=np.int64)
    rows = np.array([[e, 0] for e in range(len(worlds))], dtype=np.int64)
    actions: List[List[int]] = [[] for _ in worlds]
    for step in range(max(len(d) for d in decisions)):
        live_rows = [e for e in range(len(worlds)) if step < len(decisions[e])]
        masked_q = np.full((len(live_rows), 6), INVALID_Q, dtype=np.float32)
        masks = np.zeros((len(live_rows), 6), dtype=bool)
        bases = np.zeros(len(live_rows), dtype=np.int64)
        for i, e in enumerate(live_rows):
            q, m, b = masked_row(*decisions[e][step])
            live_action = hooks[e].apply(
                rosters[e][0], rosters[e], torch.tensor(q, dtype=torch.float32), torch.tensor(m), b
            )
            actions[e].append(int(live_action))
            masked_q[i], masks[i], bases[i] = q, m, b
        policy.runtime.counts = counts[live_rows]
        out = policy._apply_veto(sim, rows[live_rows], torch.from_numpy(masked_q), masks, bases)
        simd_actions = [int(a) for a in out]
        assert simd_actions == [actions[e][-1] for e in live_rows], (step, simd_actions)
    for e, hook in enumerate(hooks):
        assert policy.veto_record(e) == hook.record(), e
        assert deterministic(policy.veto_diagnostics(e)) == deterministic(
            hook.diagnostics_record()
        ), e
    return hooks, policy, actions


# ---------------------------------------------------------------------------
# Resolution, provenance and the row adapter
# ---------------------------------------------------------------------------
def test_resolve_names_v7_v8_and_requires_a_lambda():
    assert resolve_safety_veto("v7") == "v7" and resolve_safety_veto("v8") == "v8"
    assert resolve_safety_veto_spec(False) is None
    assert resolve_safety_veto_spec(True) == SafetyVetoSpec("v2")
    assert resolve_safety_veto_spec("v5") == SafetyVetoSpec("v5")
    assert resolve_safety_veto_spec("v7", 4) == SafetyVetoSpec("v7", 4.0)
    assert resolve_safety_veto_spec("v8", 8, 4) == SafetyVetoSpec("v8", 8.0, 4.0)
    assert resolve_safety_veto_spec("v8", 0.0) == SafetyVetoSpec("v8", 0.0, None)
    for bad in (("v7", None, None), ("v8", None, 4.0)):
        with pytest.raises(ValueError, match="requires a lambda"):
            resolve_safety_veto_spec(*bad)
    with pytest.raises(ValueError, match="reference_lambda applies only to the v8"):
        resolve_safety_veto_spec("v7", 4.0, 4.0)
    for variant in (True, "v2", "v5", False):
        with pytest.raises(ValueError, match="a veto lambda applies only to"):
            resolve_safety_veto_spec(variant, 4.0)
    for lam in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            resolve_safety_veto_spec("v7", lam)
    with pytest.raises(ValueError, match="hero_safety_veto must be"):
        resolve_safety_veto_spec("v6", 1.0)


def test_provenance_names_the_live_method_and_leaves_v2_v5_unchanged():
    from src.evaluation import safety_veto_v7, safety_veto_v8
    from src.evaluation.safety_veto_v5 import VETO_METHOD_V5

    base = {
        "engine": "simd",
        "policy": "Vector61SimdPolicy",
        "forward": "rowwise",
        "bit_exact_forward": True,
        "hero_safety_veto": True,
    }
    assert vector61_provenance("rowwise", True) == base
    assert vector61_provenance("rowwise", "v5") == {**base, "safety_veto_method": VETO_METHOD_V5}
    assert vector61_provenance("rowwise", "v7", 4.0) == {
        **base,
        "safety_veto_method": safety_veto_v7.method_for(4.0),
    }
    assert vector61_provenance("rowwise", "v8", 8.0, 4.0) == {
        **base,
        "safety_veto_method": safety_veto_v8.method_for(8.0),
        "safety_veto_reference_lambda": 4.0,
    }
    with pytest.raises(ValueError, match="requires a lambda"):
        vector61_provenance("rowwise", "v8")


def test_live_veto_hook_matches_the_screens_installers():
    from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
    from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

    v7 = live_veto_hook(SafetyVetoSpec("v7", 4.0))
    assert (
        type(v7) is SpacePreferenceVeto and v7.descriptor() == SpacePreferenceVeto(4.0).descriptor()
    )
    v8 = live_veto_hook(SafetyVetoSpec("v8", 8.0, 4.0))
    assert type(v8) is SpaceAndHeadVeto and v8.head_avoidance and v8.reference_lambda == 4.0
    assert v8.descriptor() == SpaceAndHeadVeto(8.0, reference_lambda=4.0).descriptor()
    with pytest.raises(ValueError, match="no live hook"):
        live_veto_hook(SafetyVetoSpec("v5"))


def test_policy_rejects_bad_veto_arguments_before_loading_a_checkpoint():
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    with pytest.raises(ValueError, match="requires a lambda"):
        Vector61SimdPolicy("unused.pth", None, veto_slots=(0,), veto_variant="v7")
    with pytest.raises(ValueError, match="a veto lambda applies only"):
        Vector61SimdPolicy("unused.pth", None, veto_variant="v5", veto_lambda=1.0)
    with pytest.raises(ValueError, match="veto_variant must be one of"):
        Vector61SimdPolicy("unused.pth", None, veto_variant="v6")


def test_counts_over_cap_round_trip_exactly():
    """The adapter hands ``count / cap``; the live ``round(f * cap)`` must give ``count``."""
    from src.evaluation.safety_veto import spacious_directions

    for cap in range(32, 161):
        for count in range(cap + 1):
            assert round((count / cap) * cap) == count
            assert spacious_directions([count / cap] * 3, cap, count) == [True] * 3
            if count:
                assert spacious_directions([(count - 1) / cap] * 3, cap, count) == [False] * 3


def test_row_view_reproduces_the_live_snake_fields_and_moves(v2_world):
    """Geometry, heading, lengths, aliveness, grid and every action's simulated move."""
    from src.evaluation.safety_veto_v3 import grid_for, own_body_release, static_blocked
    from src.evaluation.safety_veto_v5 import simulate_action

    rng = random.Random(7)
    for trial in range(40):
        world = random_world(rng, board=(30, 24))
        sim = batch_worlds([world], board=(30, 24))
        roster = live_roster(world, board=(30, 24))
        hero, views = sim_row_roster(sim, 0, 0, (30, 24), [1, 2, 3], 32)
        assert views[0] is hero and [v.slot for v in views] == list(range(sim.S))
        assert hero._get_free_space_features() == [1 / 32, 2 / 32, 3 / 32]
        live_grid, view_grid = grid_for(roster[0]), grid_for(hero)
        assert (live_grid.width, live_grid.height) == (view_grid.width, view_grid.height)
        assert not live_grid.circular and not view_grid.circular
        for live, view in zip(roster, views):
            assert view.is_alive == live.is_alive
            if not live.is_alive:
                continue
            assert view.segments == [(x // SS, y // SS) for x, y in live.segments]
            assert view.direction == tuple(live.direction)
            assert (view.length, view.boost_frames) == (live.length, live.boost_frames)
            assert view._logical_length() == live._logical_length()
        assert static_blocked(hero, views, view_grid) == static_blocked(
            roster[0], roster, live_grid
        )
        for action in range(6):
            a, b = simulate_action(roster[0], action), simulate_action(hero, action)
            assert [(x // SS, y // SS) for x, y in a.segments] == list(b.segments), (trial, action)
            assert [(x // SS, y // SS) for x, y in a.traversed] == list(b.traversed)
            assert (a.length, a.boost_frames, a.boosted) == (b.length, b.boost_frames, b.boosted)
            assert own_body_release(a.segments, a.length, live_grid, 1) == own_body_release(
                b.segments, b.length, view_grid, 1
            )
    with pytest.raises(RuntimeError, match="decision row"):
        SimRowSnake(sim, 0, 1, (30, 24))._get_free_space_features()


def test_live_rule_config_mismatch_is_refused(v2_world):
    from dataclasses import replace

    from src.simd_env.vector61_policy import check_live_rule_config

    sim = batch_worlds([[{"cells": [(5, 5), (4, 5)], "direction": (1, 0)}]])
    check_live_rule_config(sim)
    sim.cfg = replace(sim.cfg, min_boost_length=sim.cfg.min_boost_length + 1)
    with pytest.raises(ValueError, match="differs from GameConfig"):
        check_live_rule_config(sim)


# ---------------------------------------------------------------------------
# Named scenarios
# ---------------------------------------------------------------------------
def left_body(head, n):
    """``n`` cells from ``head`` going left, then down at x = 0 (heading right)."""
    hx, hy = head
    cells = [(x, hy) for x in range(hx, -1, -1)] + [(0, y) for y in range(hy + 1, 80)]
    return cells[:n]


def pocket_world(length=60):
    """Hero heads right at (50, 40); turning left (up) enters a closed 91-cell pocket.

    The pocket (x 45..54, y 30..38 plus the entrance (50, 39)) is walled by another
    snake whose head is far away; straight and right lead to the open board.
    """
    walls = [(x, 29) for x in range(44, 56)]
    walls += [(44, y) for y in range(30, 40)] + [(55, y) for y in range(30, 40)]
    walls += [(x, 39) for x in range(45, 55) if x != 50]
    return [
        {"cells": left_body((50, 40), length), "direction": (1, 0)},
        {"cells": walls, "direction": (-1, 0)},
    ]


def head_on_world(hero_length=30, opponent_length=30, gap=2):
    """Hero heads right at (50, 40); an opponent head at (50 + gap, 40) heads left.

    With ``gap=2`` a straight boost lands on the opponent's head (v5's landing veto
    fires first); with ``gap=3`` it lands in front of it (a head risk only).
    """
    opponent = [(x, 40) for x in range(50 + gap, 50 + gap + opponent_length)]
    return [
        {"cells": left_body((50, 40), hero_length), "direction": (1, 0)},
        {"cells": opponent, "direction": (-1, 0)},
    ]


def head_on_pocket_world(length=60):
    """:func:`pocket_world` plus an opponent head at (52, 40) heading left.

    Straight is head-risky; the non-risky alternatives are left (the 91-cell pocket)
    and right (open): v8's head replacement trades Q for area there.
    """
    world = pocket_world(length)
    opponent = [(x, 40) for x in range(52, 100)] + [(99, y) for y in range(41, 61)]
    world.append({"cells": opponent, "direction": (-1, 0)})  # 68 cells: the hero loses
    return world


def trap_world(length=50, pocket=5):
    """v5's trap: a straight boost lands inside a ``pocket``-cell corridor."""
    walls = [(62 + i, 39) for i in range(pocket)] + [(62 + i, 41) for i in range(pocket)]
    walls.append((62 + pocket, 40))
    return [
        {"cells": left_body((60, 40), length), "direction": (1, 0)},
        {"cells": walls, "direction": (1, 0)},
    ]


def boxed_world(length=80):
    """Hero heading right inside a 7x7 box (every direction below ``need``)."""
    walls = [(x, 36) for x in range(46, 55)] + [(x, 44) for x in range(46, 55)]
    walls += [(46, y) for y in range(37, 44)] + [(54, y) for y in range(37, 44)]
    hero = [(50, 40), (49, 40), (48, 40), (47, 40), (47, 41), (48, 41), (49, 41)]
    return [
        {"cells": hero, "direction": (1, 0), "length": length},
        {"cells": walls, "direction": (1, 0)},
    ]


ALL = [True] * 6
NORMAL_ONLY = [True, True, True, False, False, False]
SCENARIOS = {
    # name: (world, [(q, mask), ...])
    "pocket-left-preferred": (
        pocket_world(),
        [([1.0, 0.5, 0.0, -1.0, -1.0, -1.0], ALL), ([1.0, 0.9, 0.95, 0, 0, 0], NORMAL_ONLY)],
    ),
    "pocket-boost-mode": (
        pocket_world(),
        [([-1.0, -1.0, -1.0, 1.0, 0.5, 0.0], ALL), ([0, 0, 0, 1.0, 0.99, 0.2], ALL)],
    ),
    "pocket-ties": (
        pocket_world(),
        [([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ALL), ([2.0, 2.0, 2.0, 0, 0, 0], NORMAL_ONLY)],
    ),
    "head-on-straight-preferred": (
        head_on_world(),
        [([0.5, 1.0, 0.2, 0.1, 0.9, 0.0], ALL), ([0.3, 1.0, 0.3, 0, 0, 0], NORMAL_ONLY)],
    ),
    "head-on-boost-preferred": (
        head_on_world(gap=3),
        [([0.1, 0.5, 0.0, 0.3, 1.0, 0.2], ALL), ([0, 0, 0, 0.3, 1.0, 0.3], ALL)],
    ),
    "head-on-landing-then-head": (
        head_on_world(),
        [([0.1, 0.5, 0.0, 0.3, 1.0, 0.2], ALL)],
    ),
    "head-on-pocket": (
        head_on_pocket_world(),
        [([0.9, 1.0, 0.2, -1.0, -1.0, -1.0], ALL), ([0.9, 1.0, 0.2, 0, 0, 0], NORMAL_ONLY)],
    ),
    "head-on-hero-wins": (
        head_on_world(hero_length=100, opponent_length=10),
        [([0.5, 1.0, 0.2, 0.1, 0.9, 0.0], ALL)],
    ),
    "trap-boost-landing": (
        trap_world(),
        [
            ([0.0, 1.0, 0.5, 0.0, 9.0, 0.2], ALL),
            ([0.0, 1.0, 0.5, 0.0, 9.0, 0.2], [True, False, True, True, True, True]),
            ([0.0, 0.0, 0.0, 0.0, 9.0, 0.0], [False, False, False, False, True, False]),
        ],
    ),
    "boxed-no-spacious": (
        boxed_world(),
        [([0.1, 1.0, 0.2, 0.0, 0.5, 0.0], ALL), ([0, 0, 0, 0, 0, 0], [False] * 6)],
    ),
}


@pytest.mark.parametrize("spec", SPECS, ids=spec_id)
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_decisions_records_and_diagnostics_match_live(v2_world, name, spec):
    world, decisions = SCENARIOS[name]
    run_both(spec, [world], [decisions])


@pytest.mark.parametrize(
    "spec", [SafetyVetoSpec("v8", 8.0, 4.0), SafetyVetoSpec("v7", 4.0)], ids=spec_id
)
@pytest.mark.parametrize("name", ["head-on-hero-wins", "head-on-straight-preferred", "pocket-ties"])
def test_mechanics_v1_scenarios_match_live(setup_config, name, spec):
    """At mechanics v1 every head-on is mutual death: no waiver, parity still exact."""
    from src.core.game_config import GameConfig

    assert GameConfig.MECHANICS_VERSION == 1
    world, decisions = SCENARIOS[name]
    hooks, _policy, actions = run_both(spec, [world], [decisions])
    if name == "head-on-hero-wins" and spec.variant == "v8":
        diag = hooks[0].diagnostics_record()
        assert diag["head_risk_waived_hero_wins"] == 0 and diag["head_risky_vetoes"] == 1
        assert actions[0] != 1


def _scenario(spec: SafetyVetoSpec, name: str):
    world, decisions = SCENARIOS[name]
    hooks, _policy, actions = run_both(spec, [world], [decisions])
    return hooks[0], actions[0]


def _v7_diag(hook) -> Dict[str, Any]:
    diag = hook.diagnostics_record()
    return diag["v7"] if "v7" in diag and isinstance(diag["v7"], dict) else diag


def test_scenarios_exercise_the_rules_they_are_named_for(v2_world):
    """Guard against vacuous parity: each scenario's rule actually fires (live side)."""
    # Pocket: v7 re-ranks away from the pocket at lambda >= 4, never at 0 or 1.
    for lam, changed in ((0.0, False), (1.0, False), (4.0, True), (8.0, True), (16.0, True)):
        hook, actions = _scenario(SafetyVetoSpec("v7", lam), "pocket-left-preferred")
        assert (actions[0] != 0) == changed, (lam, actions)
        assert (_v7_diag(hook)["rerank_changes"] > 0) == changed
    hook, actions = _scenario(SafetyVetoSpec("v7", 16.0), "pocket-boost-mode")
    assert actions[0] in (4, 5) and _v7_diag(hook)["rerank_changes_boost_mode"] >= 1
    # Ties: lowest index / anchor rules decide; v7 at lambda 4 leaves the pocket.
    _hook, actions = _scenario(SafetyVetoSpec("v7", 0.0), "pocket-ties")
    assert actions == [0, 0]
    _hook, actions = _scenario(SafetyVetoSpec("v7", 4.0), "pocket-ties")
    assert actions == [1, 1]
    # Head-on: v7 keeps the risky straight; v8 replaces it at every lambda.
    for lam in LAMBDAS:
        _hook, actions = _scenario(SafetyVetoSpec("v7", lam), "head-on-straight-preferred")
        assert actions == [1, 1]
        hook, actions = _scenario(SafetyVetoSpec("v8", lam, 4.0), "head-on-straight-preferred")
        assert actions[0] not in (1, 4) and actions[1] in (0, 2)
        diag = hook.diagnostics_record()
        assert diag["head_risky_vetoes"] == 2 and diag["action_differs_from_reference"] == 2
    hook, actions = _scenario(SafetyVetoSpec("v8", 8.0), "head-on-boost-preferred")
    assert all(a in (3, 5) for a in actions) and hook.diagnostics_record()["head_risky_vetoes"] == 2
    # A straight boost onto the opponent's head: v5's landing veto (-> 1), then v8's head
    # layer (-> a turn); v7 alone keeps the risky normal-speed straight.
    hook, actions = _scenario(SafetyVetoSpec("v7", 8.0), "head-on-landing-then-head")
    assert actions == [1] and _v7_diag(hook)["v5"]["boost_to_normal_same_direction"] == 1
    hook, actions = _scenario(SafetyVetoSpec("v8", 8.0), "head-on-landing-then-head")
    assert actions[0] in (0, 2) and hook.diagnostics_record()["head_risky_vetoes"] == 1
    # Head veto next to a pocket: the highest-Q replacement is the pocket (left); a large
    # lambda trades it for the open side (right), a small one keeps it.
    for lam, expected in ((0.0, 0), (1.0, 0), (4.0, 0), (8.0, 2), (16.0, 2)):
        hook, actions = _scenario(SafetyVetoSpec("v8", lam), "head-on-pocket")
        assert actions == [expected, expected], (lam, actions)
        diag = hook.diagnostics_record()
        assert diag["head_risky_vetoes"] == 2
        assert diag["head_vetoes_space_differs_from_highest_q"] == (2 if expected == 2 else 0)
    hook, actions = _scenario(SafetyVetoSpec("v8", 8.0), "head-on-hero-wins")
    diag = hook.diagnostics_record()
    assert actions == [1] and diag["head_risk_waived_hero_wins"] == 1
    assert diag["head_risky_decisions"] == 0
    # Boost landing: v5's landing veto -> same direction normal speed, then the v2 rule.
    for spec in (SafetyVetoSpec("v7", 8.0), SafetyVetoSpec("v8", 8.0)):
        hook, actions = _scenario(spec, "trap-boost-landing")
        assert actions == [1, 5, 4], (spec, actions)
        assert _v7_diag(hook)["v5"]["base_landing_failed"] == 3
    # No spacious direction: the base action stands (v7 and v8 never act there).
    for spec in (SafetyVetoSpec("v7", 16.0), SafetyVetoSpec("v8", 16.0)):
        hook, actions = _scenario(spec, "boxed-no-spacious")
        assert actions == [1, 0] and hook.counters.to_dict()["fallback_no_spacious"] == 2


# ---------------------------------------------------------------------------
# Fuzz: crowded random boards, several envs per call, decisions accumulated
# ---------------------------------------------------------------------------
def random_walk(rng: random.Random, start, n, occupied, board) -> List[tuple]:
    cells = [start]
    occupied.add(start)
    while len(cells) < n:
        x, y = cells[-1]
        options = [
            (x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= x + dx < board[0]
            and 0 <= y + dy < board[1]
            and (x + dx, y + dy) not in occupied
        ]
        if not options:
            break
        cell = rng.choice(options)
        cells.append(cell)
        occupied.add(cell)
    return cells


def random_world(rng: random.Random, board=(30, 24), snakes=None) -> List[dict]:
    """Hero (slot 0) plus 1-4 opponents as self-avoiding walks; some dead or pending growth."""
    occupied: set = set()
    world: List[dict] = []
    count = snakes or rng.randint(2, 5)
    for slot in range(count):
        for _ in range(50):
            start = (rng.randrange(1, board[0] - 1), rng.randrange(1, board[1] - 1))
            if start not in occupied:
                break
        n = rng.choice([1, 3, 6, 12, 25, 60, 120, 200]) if slot == 0 else rng.randint(1, 40)
        cells = random_walk(rng, start, n, occupied, board)
        if len(cells) >= 2:
            neck = cells[1]
            direction = (cells[0][0] - neck[0], cells[0][1] - neck[1])
        else:
            direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        world.append(
            {
                "cells": cells,
                "direction": direction,
                "length": len(cells) + rng.choice([0, 0, 0, 1, 2]),
                "boost_frames": rng.randint(0, 2),
                "alive": slot == 0 or rng.random() > 0.15,
            }
        )
    return world


def pocket_fuzz_world(rng: random.Random, board=(40, 30)) -> List[dict]:
    """A hero next to a walled rectangle whose one opening touches its head.

    The rectangle (random size, random side, random opening) is one opponent's body;
    a second opponent's head is sometimes placed within two cells of the hero's head.
    Pockets of every size around ``need`` and the v7 area cap, so re-ranks fire.
    """
    gw, gh = board
    hx, hy = rng.randrange(12, gw - 12), rng.randrange(10, gh - 10)
    heading = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    occupied: set = set()
    back = (hx - heading[0], hy - heading[1])
    hero = [(hx, hy)] + random_walk(
        rng, back, rng.choice([8, 15, 25, 40, 60, 90]), occupied | {(hx, hy)}, board
    )
    occupied.update(hero)
    # The pocket sits on a side of the head that is not the neck's side.
    side = rng.choice(
        [d for d in ((1, 0), (-1, 0), (0, 1), (0, -1)) if d != (-heading[0], -heading[1])]
    )
    w, h = rng.randint(2, 12), rng.randint(2, 12)
    ox, oy = hx + side[0], hy + side[1]  # the opening cell
    if side[0]:
        x0 = ox + (1 if side[0] > 0 else -w)
        y0 = oy - rng.randrange(h)
    else:
        y0 = oy + (1 if side[1] > 0 else -h)
        x0 = ox - rng.randrange(w)
    ring = [(x, y0 - 1) for x in range(x0 - 1, x0 + w + 1)] + [
        (x, y0 + h) for x in range(x0 - 1, x0 + w + 1)
    ]
    ring += [(x0 - 1, y) for y in range(y0, y0 + h)] + [(x0 + w, y) for y in range(y0, y0 + h)]
    walls = [
        c
        for c in dict.fromkeys(ring)
        if c != (ox, oy) and c not in occupied and 0 <= c[0] < gw and 0 <= c[1] < gh
    ]
    # The opening cell's own flanks are walls too, so the pocket has one way in.
    for flank in ((ox + side[1], oy + side[0]), (ox - side[1], oy - side[0])):
        if (
            flank not in occupied
            and flank not in walls
            and 0 <= flank[0] < gw
            and 0 <= flank[1] < gh
        ):
            walls.append(flank)
    occupied.update(walls)
    world = [
        {
            "cells": hero,
            "direction": heading,
            "length": len(hero) + rng.choice([0, 0, 1]),
            "boost_frames": rng.randint(0, 2),
        },
        {"cells": walls or [(0, 0)], "direction": (1, 0)},
    ]
    if rng.random() < 0.5:
        for _ in range(20):
            cell = (hx + rng.randint(-2, 2), hy + rng.randint(-2, 2))
            if cell not in occupied and 0 <= cell[0] < gw and 0 <= cell[1] < gh:
                body = random_walk(rng, cell, rng.randint(1, 60), occupied, board)
                direction = (
                    (body[0][0] - body[1][0], body[0][1] - body[1][1])
                    if len(body) > 1
                    else rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                )
                world.append(
                    {"cells": body, "direction": direction, "boost_frames": rng.randint(0, 2)}
                )
                break
    return world


def random_decision(rng: random.Random) -> Tuple[List[float], List[bool]]:
    if rng.random() < 0.3:
        q = [float(rng.choice([0.0, 0.5, 1.0])) for _ in range(6)]  # ties
    else:
        q = [rng.uniform(-2.0, 2.0) for _ in range(6)]
    mask = [rng.random() < 0.75 for _ in range(6)]
    return q, mask


@pytest.mark.parametrize("spec", SPECS, ids=spec_id)
def test_fuzzed_worlds_match_live_decision_for_decision(v2_world, spec):
    rng = random.Random(1000 + SPECS.index(spec))
    totals: Dict[str, int] = {}
    for batch in range(12):
        worlds = [random_world(rng) for _ in range(4)]
        decisions = [[random_decision(rng) for _ in range(rng.randint(1, 4))] for _ in worlds]
        hooks, _policy, _actions = run_both(spec, worlds, decisions, board=(30, 24))
        for hook in hooks:
            diag = hook.diagnostics_record()
            v7 = _v7_diag(hook)
            for key in ("vetoes_applied", "no_spacious", "decisions"):
                totals[key] = totals.get(key, 0) + int(diag[key])
            totals["rerank_scored"] = totals.get("rerank_scored", 0) + int(v7["rerank_scored"])
            totals["head_checks"] = totals.get("head_checks", 0) + int(diag.get("head_checks", 0))
    # Coverage floors (these seeds): vetoes, no_spacious and scored re-ranks occur.
    assert totals["decisions"] >= 100 and totals["no_spacious"] > 0, totals
    if spec.lam > 0:
        assert totals["rerank_scored"] > 0, totals
    if spec.variant == "v8":
        assert totals["head_checks"] > 0, totals


@pytest.mark.parametrize("spec", SPECS, ids=spec_id)
def test_pocket_fuzz_matches_live_and_reaches_reranks_and_head_vetoes(v2_world, spec):
    """Pocket-rich boards: re-ranks, head vetoes and head-veto area trades all fire."""
    rng = random.Random(77 + SPECS.index(spec))
    totals: Dict[str, int] = {}
    for _ in range(25):
        worlds = [pocket_fuzz_world(rng) for _ in range(4)]
        decisions = [[random_decision(rng) for _ in range(3)] for _ in worlds]
        hooks, _policy, _actions = run_both(spec, worlds, decisions, board=(40, 30))
        for hook in hooks:
            diag = hook.diagnostics_record()
            v7 = _v7_diag(hook)
            for key in ("rerank_changes", "rerank_scored", "rerank_pruned"):
                totals[key] = totals.get(key, 0) + int(v7[key])
            for key in ("head_risky_vetoes", "head_vetoes_space_differs_from_highest_q"):
                totals[key] = totals.get(key, 0) + int(diag.get(key, 0))
            totals["landing"] = totals.get("landing", 0) + int(v7["v5"]["base_landing_failed"])
    assert totals["landing"] > 0, totals
    if spec.lam == 0.0:
        assert totals["rerank_changes"] == totals["rerank_scored"] == 0, totals
    elif spec.lam >= 4.0:
        assert totals["rerank_changes"] > 0, totals
    if spec.variant == "v8":
        assert totals["head_risky_vetoes"] > 0, totals


def test_variants_outside_v7_v8_never_build_live_hooks(v2_world):
    assert LAMBDA_VETO_VARIANTS == ("v7", "v8")
    policy = simd_hook(SafetyVetoSpec("v7", 4.0))
    fresh = policy.veto_record(3)  # an env that never decided: a fresh hook's record
    assert fresh == live_veto_hook(SafetyVetoSpec("v7", 4.0)).record()
    assert fresh["counters"]["decisions"] == 0 and not policy.live_vetoes


# ---------------------------------------------------------------------------
# Whole rollouts (plays episodes: opt-in only, SNAKE_SIMD_ROLLOUT_PARITY=1)
# ---------------------------------------------------------------------------
rollout_parity = pytest.mark.skipif(
    os.environ.get("SNAKE_SIMD_ROLLOUT_PARITY") != "1",
    reason="plays episodes; set SNAKE_SIMD_ROLLOUT_PARITY=1 to run",
)


@rollout_parity
@pytest.mark.parametrize(
    "spec",
    [SafetyVetoSpec("v7", 4.0), SafetyVetoSpec("v8", 8.0, 4.0), SafetyVetoSpec("v8", 16.0)],
    ids=spec_id,
)
def test_tiny_world_rollouts_match_live(spec, monkeypatch, setup_config, tmp_path):
    """Real ``tournament_eval.rollout`` + the live installer vs ``run_simd_eval``."""
    from research.apex_safety_20260926 import dev_screen
    from src.core.config_loader import load_and_initialize_config
    from src.model.apex_network import ApexNetwork
    from src.simd_env import eval_engine as ee
    from src.simd_env.vector61_policy import Vector61SimdPolicy
    from tests.test_simd_vector61_policy import (
        SIMD_ONLY_KEYS,
        TINY_YAML,
        _compare,
        _first_state_divergence,
        _profile,
        _run_live,
        _tiny_rosters,
    )

    cfg = tmp_path / "tiny_v61.yaml"
    cfg.write_text(TINY_YAML)
    load_and_initialize_config(str(cfg))

    def make(name: str, seed: int):
        torch.manual_seed(seed)
        net = ApexNetwork(input_size=61, hidden_size=32, output_size=6)
        with torch.no_grad():
            for param in net.parameters():
                param.mul_(3.0)
        path = tmp_path / f"{name}.pth"
        torch.save({"dqn_state_dict": net.state_dict(), "input_size": 61, "hidden_size": 32}, path)
        return ("checkpoint", str(path))

    seeds = (3, 4, 5, 6)
    frames = 300
    hero, rosters = _tiny_rosters(make, seeds)
    with dev_screen.hero_veto_installer(lambda snake: _install(snake, spec)) as installed:
        live_records, live_actions, live_states, trimmed = _run_live(
            monkeypatch, hero, rosters, frames, seeds, "tiny-v61", True
        )
    assert trimmed == 0 and len(installed) == len(seeds)
    simd_actions: Dict[tuple, int] = {}
    original = Vector61SimdPolicy.actions

    def recording(self, masks, sim, slots):
        out = original(self, masks, sim, slots)
        for (env, slot), action in zip(slots, out):
            simd_actions[(int(seeds[int(env)]), int(sim.frame[int(env)]), int(slot))] = int(action)
        return out

    monkeypatch.setattr(Vector61SimdPolicy, "actions", recording)
    simd_records = ee.run_simd_eval(
        hero,
        rosters[seeds[0]],
        frames,
        list(seeds),
        profile=_profile(frames),
        opponent_specs_by_world=rosters,
        mix_id="tiny-v61",
        vector61=True,
        hero_safety_veto=spec.variant,
        hero_safety_veto_lambda=spec.lam,
        hero_safety_veto_reference_lambda=spec.reference_lambda,
    )
    report = _compare(live_actions, simd_actions)
    assert report["first_divergence"] is None, report
    del live_states, _first_state_divergence
    for live, simd, veto in zip(live_records, simd_records, installed):
        assert deterministic(simd["veto_diagnostics"]) == deterministic(veto.diagnostics_record())
        view = {k: v for k, v in simd.items() if k not in SIMD_ONLY_KEYS | {"veto_diagnostics"}}
        assert view == live


def _install(snake, spec: SafetyVetoSpec):
    if spec.variant == "v7":
        from src.evaluation.safety_veto_v7 import install_space_preference_veto

        return install_space_preference_veto(snake, spec.lam)
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto

    return install_space_and_head_veto(snake, spec.lam, reference_lambda=spec.reference_lambda)


@pytest.mark.parametrize("variant", ["v2", "v5"])
def test_v2_v5_paths_unchanged_through_configure_veto(v2_world, variant):
    """The shared ``_configure_veto`` keeps v2/v5 on their own (non-hook) paths."""
    from src.evaluation.safety_veto import FreeSpaceVeto
    from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto

    world, decisions = SCENARIOS["trap-boost-landing"]
    sim = batch_worlds([world])
    roster = live_roster(world)
    live = FreeSpaceVeto() if variant == "v2" else BoostAwareFreeSpaceVeto()
    policy = simd_hook(SafetyVetoSpec(variant))
    policy.runtime.counts = np.array([live_counts(roster[0], roster)], dtype=np.int64)
    for q, mask in decisions:
        masked, m, base = masked_row(q, mask)
        expected = live.apply(roster[0], roster, torch.tensor(masked), torch.tensor(m), base)
        out = policy._apply_veto(
            sim, np.array([[0, 0]]), torch.tensor([masked]), np.array([m]), np.array([base])
        )
        assert int(out[0]) == int(expected)
    assert policy.veto_record(0) == live.record() and not policy.live_vetoes
    if variant == "v5":
        assert deterministic(policy.veto_diagnostics(0)) == deterministic(live.diagnostics_record())
