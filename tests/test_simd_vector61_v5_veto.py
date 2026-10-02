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
            {
                "cells": me_cells,
                "direction": (1, 0),
                "length": me.length,
                "boost_frames": boost_frames,
            },
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
    sim = batch_world(
        [{"cells": me_cells, "direction": (1, 0)}, {"cells": [(5, 5)], "direction": (1, 0)}]
    )
    sim.alive[0, 1] = False
    for direction in range(3):
        assert boost_landing_count(sim, 0, 0, direction, 32, (100, 80)) == landing_count(
            me, [me], direction, 32
        )
    assert boost_landing_count(sim, 0, 0, 1, 32, (100, 80)) == 0


def burn_gate_cells() -> Tuple[List[tuple], List[tuple]]:
    """Hero heading right at (60, 40); the boost burn's popped tail gates the pocket.

    The hero's body loops over row 38 and comes down to (64, 40), the end of a
    two-cell corridor walled by another snake. A straight boost lands on (62, 40).
    With ``boost_frames=2`` and ``len(segments) == length`` the burn pops a third
    tail cell, (64, 40), which opens the corridor onto the board; without the burn
    that cell stays body and the landing pocket holds only 2 cells.
    """
    me = [(60, 40), (60, 39), (60, 38), (61, 38), (62, 38), (63, 38), (64, 38), (64, 39)]
    me += [(64, 40), (65, 40), (66, 40)]
    walls = [(62, 39), (63, 39), (62, 41), (63, 41)]
    return me, walls


def test_landing_count_matches_live_when_the_burned_tail_opens_the_pocket(setup_config):
    """The tail popped by the boost burn decides the landing; SIMD must replay the burn."""
    from src.core.game_config import GameConfig
    from src.evaluation.safety_veto import free_space_threshold
    from src.evaluation.safety_veto_v5 import landing_count, simulate_action

    me_cells, walls = burn_gate_cells()
    gate = (64, 40)
    counts = {}
    for boost_frames in (2, 0):
        me = live_snake(me_cells, (1, 0), boost_frames=boost_frames)
        other = live_snake(walls, (1, 0), sid=1)
        assert len(me.segments) == me.length == len(me_cells)
        sim = batch_world(
            [
                {"cells": me_cells, "direction": (1, 0), "boost_frames": boost_frames},
                {"cells": walls, "direction": (1, 0)},
            ]
        )
        cap, need = free_space_threshold(me.length, me._logical_length())
        move = simulate_action(me, 4)
        burned = (gate[0] * SS, gate[1] * SS) if boost_frames == 2 else None
        assert move.boosted and move.burned_tail == burned
        assert sim.cfg.boost_length_cost_frames == GameConfig.BOOST_LENGTH_COST_FRAMES == 3
        live = landing_count(me, [me, other], 1, cap)
        simd = boost_landing_count(sim, 0, 0, 1, cap, (100, 80))
        assert simd == live, (boost_frames, live, simd)
        counts[boost_frames] = (live, need)
    (burn_count, need), (no_burn_count, _) = counts[2], counts[0]
    assert no_burn_count == 2 < need  # without the burn the gate cell stays body
    assert burn_count >= need  # the burn opens the pocket: the landing passes


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


# ---------------------------------------------------------------------------
# Whole rollouts: live tournament_eval.rollout + v5 installer vs run_simd_eval("v5")
# ---------------------------------------------------------------------------
VetoEntry = Tuple[int, tuple, int, int, int, tuple, tuple]


def _landing_code(count: Optional[int]) -> int:
    return NO_BOOST if count is None else int(count)


def _install_v5_spies(monkeypatch: pytest.MonkeyPatch, seeds: Sequence[int]):
    """Log every hero veto decision on both sides.

    Entry: ``(frame, v2 counts, need, base, final, lazy landings, eager landings)``.
    Lazy landings are the floods the hook actually ran (``NOT_FLOODED`` otherwise);
    eager landings are all three directions, computed outside the hook so they
    never perturb its counters.
    """
    from src.evaluation import safety_veto as sv
    from src.evaluation import safety_veto_v5 as sv5
    from src.simd_env import vector61_policy as vp

    live_logs: Dict[int, List[VetoEntry]] = {}
    keep_alive: list = []
    simd_logs: Dict[int, List[VetoEntry]] = {int(seed): [] for seed in seeds}
    original_apply = sv5.BoostAwareFreeSpaceVeto.apply
    original_landing = sv5.landing_count
    original_apply_veto = vp.Vector61SimdPolicy._apply_veto
    lazy: Dict[str, Optional[list]] = {"live": None}

    def logging_landing(snake, other_snakes, direction, cap):
        count = original_landing(snake, other_snakes, direction, cap)
        if lazy["live"] is not None:
            lazy["live"][int(direction)] = _landing_code(count)
        return count

    def live_apply(self, snake, other_snakes, q_values, action_mask, base_action):
        others = list(other_snakes)
        features = snake._get_free_space_features(others)
        cap, need = sv.free_space_threshold(snake.length, snake._logical_length())
        counts = tuple(round(float(f) * cap) for f in features)
        eager = tuple(_landing_code(original_landing(snake, others, d, cap)) for d in range(3))
        lazy["live"] = [NOT_FLOODED] * 3
        try:
            action = original_apply(self, snake, others, q_values, action_mask, base_action)
            flooded = tuple(lazy["live"])
        finally:
            lazy["live"] = None
        if id(self) not in live_logs:
            keep_alive.append(self)
        live_logs.setdefault(id(self), []).append(
            (
                int(snake._get_frame()),
                counts,
                int(need),
                int(base_action),
                int(action),
                flooded,
                eager,
            )
        )
        return action

    def simd_apply_veto(self, sim, slots, masked_q, mask, actions):
        self.last_veto = None
        out = original_apply_veto(self, sim, slots, masked_q, mask, actions)
        last = self.last_veto
        if last is not None:
            grid = (int(self.runtime.featurizer.gw), int(self.runtime.featurizer.gh))
            caps, _ = vp.veto_threshold(sim.length[last["rows"][:, 0], last["rows"][:, 1]])
            for i, (env, slot) in enumerate(last["rows"].tolist()):
                eager = tuple(
                    _landing_code(vp.boost_landing_count(sim, env, slot, d, int(caps[i]), grid))
                    for d in range(3)
                )
                simd_logs[int(seeds[int(env)])].append(
                    (
                        int(last["frame"][i]),
                        tuple(int(c) for c in last["counts"][i]),
                        int(last["need"][i]),
                        int(last["base"][i]),
                        int(last["final"][i]),
                        tuple(int(c) for c in last["landing"][i]),
                        eager,
                    )
                )
        return out

    monkeypatch.setattr(sv5, "landing_count", logging_landing)
    monkeypatch.setattr(sv5.BoostAwareFreeSpaceVeto, "apply", live_apply)
    monkeypatch.setattr(vp.Vector61SimdPolicy, "_apply_veto", simd_apply_veto)

    def collected():
        ordered = list(live_logs.values())
        assert len(ordered) == len(seeds)
        return {int(seed): log for seed, log in zip(seeds, ordered)}, simd_logs

    return collected


def _assert_v5_parity(
    monkeypatch: pytest.MonkeyPatch,
    hero: Tuple[str, str],
    rosters: Dict[int, List[Tuple[str, str]]],
    frames: int,
    seeds: Sequence[int],
    mix_id: str,
) -> Dict[str, object]:
    """Every vector decision, state, hero veto decision, record and v5 diagnostic."""
    from research.apex_safety_20260926 import dev_screen
    from src.evaluation.safety_veto_v5 import install_boost_aware_veto

    veto_logs = _install_v5_spies(monkeypatch, seeds)
    with dev_screen.hero_veto_installer(install_boost_aware_veto) as installed:
        live_records, live_actions, live_states, trimmed = _run_live(
            monkeypatch, hero, rosters, frames, seeds, mix_id, True
        )
    assert len(installed) == len(seeds)
    simd_records, simd_actions, stats, simd_states = _run_simd(
        monkeypatch, hero, rosters, frames, seeds, mix_id, "v5"
    )
    assert trimmed == 0, f"live trim_ambient removed {trimmed} pellets"
    report = _compare(live_actions, simd_actions)
    same_states, first_state = _first_state_divergence(live_states, simd_states)
    report["bit_identical_states"] = same_states
    assert first_state is None, (first_state, report)
    assert report["first_divergence"] is None, report
    assert report["decisions"] == stats["rows"]

    live_veto, simd_veto = veto_logs()
    for seed in seeds:
        live_log, simd_log = live_veto[int(seed)], simd_veto[int(seed)]
        first = next((i for i, (a, b) in enumerate(zip(live_log, simd_log)) if a != b), None)
        assert first is None and len(live_log) == len(simd_log), (
            seed,
            first,
            None if first is None else (live_log[first], simd_log[first]),
        )

    provenance = vector61_provenance("rowwise", "v5")
    diagnostics = []
    for live, simd, veto in zip(live_records, simd_records, installed):
        assert "vector61_policy" not in live and "veto_diagnostics" not in live
        assert simd["vector61_policy"] == provenance
        live_diag = veto.diagnostics_record()
        assert deterministic(simd["veto_diagnostics"]) == deterministic(live_diag)
        diagnostics.append(deterministic(live_diag))
        simd_view = {
            k: v for k, v in simd.items() if k not in SIMD_ONLY_KEYS | {"veto_diagnostics"}
        }
        assert simd_view == live, (live, simd)

    entries = [e for log in live_veto.values() for e in log]
    hero_decisions = sum(1 for key in live_actions if key[2] == 0)
    assert len(entries) == hero_decisions
    total = {k: sum(int(d[k]) for d in diagnostics) for k in diagnostics[0]}
    assert total["decisions"] == hero_decisions
    report.update(stats)
    report.update(
        {
            "hero_decisions": hero_decisions,
            "veto_decisions": len(entries),
            "lazy_floods_compared": sum(c >= 0 for e in entries for c in e[5]),
            "eager_landings_compared": sum(c >= 0 for e in entries for c in e[6]),
            "landing_failures_eager": sum(0 <= c < e[2] for e in entries for c in e[6]),
            "landing_vetoes_compared": total["base_landing_failed"],
            "boost_landing_vetoes": total["boost_landing_vetoes"],
            "boost_to_normal_same_direction": total["boost_to_normal_same_direction"],
            "action_differs_from_v2": total["action_differs_from_v2"],
            "landing_checks": total["landing_checks"],
            "vetoes_applied": total["vetoes_applied"],
            "deaths": [r["deaths"] for r in live_records],
        }
    )
    return report


def test_tiny_world_live_and_simd_v5_decisions_are_identical(tiny_world, monkeypatch):  # noqa: F811
    """Random 61-D networks on the tiny world: deaths, respawns and lazy landing floods.

    No landing veto fires here (0 in this range); the deployment-world tests cover those.
    """
    seeds = (3, 4, 5, 6)
    hero, rosters = _tiny_rosters(tiny_world, seeds)
    report = _assert_v5_parity(monkeypatch, hero, rosters, 300, seeds, "tiny-v61")
    print({k: v for k, v in report.items() if k != "first_divergence"})
    assert report["matches"] == report["decisions"] > 1000
    assert report["fresh_after_first_frame"] > 0
    assert report["lazy_floods_compared"] > 0


@needs_real_pool
@pytest.mark.parametrize(
    "mix, frames, seeds, landing_vetoes",
    [
        # Seeds 16 (frozen) and 15 (mixed) were found by a SIMD-only search over
        # seeds 11..58 at 300 frames: each has one real landing veto (a boost
        # replaced by the same direction at normal speed). Scripted had none.
        ("frozen", 300, (12, 16), True),
        ("mixed", 300, (13, 15), True),
        ("frozen", 500, (11, 12, 13), False),
        ("scripted", 300, (11, 12), False),
    ],
)
def test_deployment_world_champion_v5_decisions_match_live(
    deployment_world, monkeypatch, mix, frames, seeds, landing_vetoes  # noqa: F811
):
    """Champion hero + v5 vs the strict pilot pool mixes at the deployment profile."""
    from src.scripts import tournament_eval as te

    pool = deployment_world
    specs = te.build_mix_specs(mix, 5, pool)
    rosters = te.materialize_opponent_specs_by_world(specs, seeds)
    report = _assert_v5_parity(monkeypatch, pool[0], rosters, frames, seeds, mix)
    print(mix, {k: v for k, v in report.items() if k != "first_divergence"})
    assert report["matches"] == report["decisions"] >= len(seeds) * 200
    assert report["eager_landings_compared"] > 0 and report["landing_failures_eager"] > 0
    if landing_vetoes:
        assert report["landing_vetoes_compared"] > 0, report
        assert report["boost_landing_vetoes"] > 0 and report["action_differs_from_v2"] > 0
