"""Tests for the opt-in v6 veto (opponent-head avoidance + no-spacious fallback)."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from pathlib import Path

import pytest
import torch

from src.core import game_config
from src.core.game_config import GameConfig, get_config
from src.evaluation.safety_veto import free_space_threshold, spacious_directions
from src.evaluation.safety_veto_v3 import grid_for
from src.evaluation.safety_veto_v4 import BudgetExhausted
from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto, landing_count
from src.evaluation.safety_veto_v6 import (
    FALLBACK_DEPTH,
    FALLBACK_NODE_BUDGET,
    VETO_METHOD_V6,
    HeadAndFallbackVeto,
    fallback_choice,
    head_avoid_choice,
    head_risky_actions,
    hero_wins_head_on,
    install_head_and_fallback_veto,
    opponent_head_reach,
)
from src.game.game_logic import GameLogic
from tests.test_safety_veto_v5 import clone, make_snake, random_states, survives

REPO = Path(__file__).resolve().parents[1]
# Released v2 (strict receipt), released v5 (served Watch hero) and the frozen v3/v4.
FROZEN = {
    "safety_veto.py": "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    "safety_veto_v3.py": "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
    "safety_veto_v4.py": "3f0881afb7ff8ff980c9b425ef40134f127cc90f1f40260aac39f2264ed44578",
    "safety_veto_v5.py": "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c",
}
SS = 10
ALL = [True] * 6
TIMING_KEYS = {
    "apply_seconds_total",
    "apply_seconds_max",
    "mean_apply_seconds",
    "fallback_seconds_total",
    "fallback_seconds_max",
    "mean_fallback_seconds",
}


@pytest.fixture
def mechanics(setup_config, monkeypatch):
    """Set ``GameConfig.MECHANICS_VERSION`` for one test (restored by monkeypatch)."""

    def set_version(version):
        config = get_config()
        game = dataclasses.replace(config.game, mechanics_version=int(version))
        monkeypatch.setattr(game_config, "_current_config", dataclasses.replace(config, game=game))
        assert GameConfig.MECHANICS_VERSION == int(version)

    return set_version


def deterministic(diagnostics):
    out = {k: v for k, v in diagnostics.items() if k not in TIMING_KEYS}
    out["v5"] = {
        k: v
        for k, v in diagnostics["v5"].items()
        if k not in {"apply_seconds_total", "apply_seconds_max", "mean_apply_seconds"}
    }
    return out


def v6_apply(me, roster, q, mask, base, **options):
    veto = HeadAndFallbackVeto(**options)
    action = veto.apply(me, roster, q, mask, base)
    return action, veto


def v5_apply(me, roster, q, mask, base):
    return BoostAwareFreeSpaceVeto().apply(me, roster, q, mask, base)


class TestFrozenSources:
    def test_released_and_retired_veto_bytes_are_unchanged(self):
        for name, sha in FROZEN.items():
            source = REPO / "src/evaluation" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name


# ------------------------------------------------------------------ layer (a): head-on
#
# The hero (length ``hero_length``) heads right with its head at (20, 20) and its body
# straight to the left. The opponent's head is diagonally adjacent at (21, 19), heading
# down, body straight up. The opponent can reach (21, 20) (straight), (21, 21) (boost),
# (22, 19)/(23, 19) (its left) and (20, 19)/(19, 19) (its right) next frame, so the
# hero's straight (21, 20) and its left turn (20, 19) are head-risky and its right turn
# (20, 21) is not. Q prefers straight; the open 60x60 board is spacious everywhere.
HEAD_Q = [0.0, 5.0, 1.0, 0.0, 0.5, 0.2]


def head_on_world(hero_length, opponent_length, board=60):
    hero = make_snake(
        [(20 - i, 20) for i in range(hero_length)], (1, 0), width=board * SS, height=board * SS
    )
    opponent = make_snake(
        [(21, 19 - i) for i in range(opponent_length)],
        (0, 1),
        width=board * SS,
        height=board * SS,
        sid=1,
    )
    return hero, opponent


def live_pair(hero, opponent, hero_action, opponent_action):
    """Both snakes move with the live ``Snake.move``; return the live head-on check."""
    moved = []
    for snake, action in ((hero, hero_action), (opponent, opponent_action)):
        twin = clone(snake)
        twin.direction = GameLogic.relative_to_absolute_direction(twin.direction, action % 3)
        twin.is_boosting = action >= 3 and twin.length >= GameConfig.MIN_BOOST_LENGTH
        twin.move()
        moved.append(twin)
    return GameLogic.check_head_collision(moved[0], moved[1])


class TestHeadOnLayer:
    @pytest.mark.parametrize("version", [1, 2])
    def test_equal_or_larger_opponent_v5_keeps_the_head_on_and_v6_sidesteps(
        self, mechanics, version
    ):
        mechanics(version)
        hero, opponent = head_on_world(12, 15)
        roster = [hero, opponent]
        base = 1
        assert v5_apply(hero, roster, HEAD_Q, ALL, base) == 1  # v5 sees an empty cell
        action, veto = v6_apply(hero, roster, HEAD_Q, ALL, base)
        assert action == 2  # right turn: the only non-risky normal-speed move
        diag = veto.diagnostics_record()
        assert diag["head_risky_vetoes"] == 1 and diag["head_risky_decisions"] == 1
        assert diag["action_differs_from_v5"] == 1
        counters = veto.record()["counters"]
        assert counters["vetoes_applied"] == 1 and counters["kept_base"] == 0
        # The live game agrees: straight can die head-on, the replacement cannot.
        assert live_pair(hero, opponent, 1, 1)
        assert not any(live_pair(hero, opponent, 2, other) for other in range(6))

    def test_hero_longer_than_the_size_rule_is_not_vetoed(self, mechanics):
        mechanics(2)
        hero, opponent = head_on_world(40, 12)  # 40 >= 1.15 * (12 + 1)
        assert hero_wins_head_on(40, 13)
        action, veto = v6_apply(hero, [hero, opponent], HEAD_Q, ALL, 1)
        assert action == 1
        diag = veto.diagnostics_record()
        assert diag["head_risky_decisions"] == 0 and diag["head_risk_waived_hero_wins"] == 1

    def test_near_size_rule_margin_still_vetoes(self, mechanics):
        mechanics(2)
        # 14 >= 1.15 * 12 but not >= 1.15 * (12 + 1): growth this frame could cancel it.
        hero, opponent = head_on_world(14, 12)
        assert v6_apply(hero, [hero, opponent], HEAD_Q, ALL, 1)[0] == 2

    def test_mechanics_v1_never_waives(self, mechanics):
        mechanics(1)
        assert not hero_wins_head_on(1000, 2)
        hero, opponent = head_on_world(40, 12)
        assert v6_apply(hero, [hero, opponent], HEAD_Q, ALL, 1)[0] == 2

    def test_boost_reach_and_two_cell_hero_path(self, mechanics):
        mechanics(2)
        # Opponent head two cells right of the hero's straight boost landing path: (23, 20)
        # heading left reaches (22, 20) and, boosting, (21, 20).
        hero = make_snake([(20 - i, 20) for i in range(12)], (1, 0), width=600, height=600)
        opponent = make_snake(
            [(23 + i, 20) for i in range(15)], (-1, 0), width=600, height=600, sid=1
        )
        q = [0.0, 1.0, 0.5, 0.0, 9.0, 0.3]  # boost straight first, then boost right
        action, veto = v6_apply(hero, [hero, opponent], q, ALL, 4)
        assert action == 5  # same speed mode: the right-turn boost
        assert veto.diagnostics_record()["head_risky_vetoes"] == 1
        assert live_pair(hero, opponent, 4, 4)
        assert not any(live_pair(hero, opponent, 5, other) for other in range(6))

    def test_short_opponent_cannot_boost_so_its_second_cell_is_safe(self, mechanics):
        mechanics(1)
        grid_owner = make_snake([(20, 20)], (1, 0), width=600, height=600)
        opponent = make_snake(
            [(23 + i, 20) for i in range(3)], (-1, 0), width=600, height=600, sid=1
        )
        assert opponent.length < GameConfig.MIN_BOOST_LENGTH
        reach = opponent_head_reach(grid_owner, [opponent], grid_for(grid_owner))
        assert (22, 20) in reach and (21, 20) not in reach and (23, 20) in reach

    def test_no_alternative_keeps_v5_choice(self, mechanics):
        mechanics(1)
        hero, opponent = head_on_world(12, 15)
        mask = [False, True, False, False, True, False]  # only straight is legal
        action, veto = v6_apply(hero, [hero, opponent], HEAD_Q, mask, 1)
        assert action == 1
        diag = veto.diagnostics_record()
        assert diag["head_risky_kept_no_alternative"] == 1 and diag["head_risky_vetoes"] == 0

    def test_dead_opponents_and_the_hero_itself_are_ignored(self, mechanics):
        mechanics(1)
        hero, opponent = head_on_world(12, 15)
        opponent.is_alive = False
        assert opponent_head_reach(hero, [hero, opponent], grid_for(hero)) == {}
        assert v6_apply(hero, [hero, opponent], HEAD_Q, ALL, 1)[0] == 1

    def test_pure_choice_prefers_same_speed_mode(self):
        eligible = [True] * 6
        risky = [False, True, False, False, False, False]
        q = [0.0, 9.0, 1.0, 0.0, 8.0, 7.0]
        assert head_avoid_choice(q, eligible, risky, 1) == (2, "head_veto")
        assert head_avoid_choice(q, eligible, [False] * 6, 1) == (1, "v5")
        only_boost = [False, False, False, True, True, True]
        assert head_avoid_choice(q, only_boost, risky, 1) == (4, "head_veto")
        assert head_avoid_choice(q, [False] * 6, risky, 1) == (1, "head_kept_no_alternative")


# ------------------------------------------------------------- layer (b): no-spacious
#
# Boost into a tiny pocket. The hero (length 50, need 50) heads right at (60, 40), body
# straight left. A wall snake encloses a 12-cell area: the first cell F = (61, 40), a
# 10-cell column x = 61 above and below F, and a 1-cell pocket P = (62, 40). Left and
# right turns hit the wall. Straight at normal speed counts 12 (< need: no_spacious); the
# straight boost lands in P with F (now body) behind it: landing count 1.
def boxed_pocket():
    wall = [(60, y) for y in range(34, 47) if y != 40]
    wall += [(62, y) for y in range(34, 47) if y != 40]
    wall += [(61, 34), (61, 46), (63, 40)]
    hero = make_snake([(60 - i, 40) for i in range(50)], (1, 0), width=1000, height=800)
    other = make_snake(wall, (0, 1), width=1000, height=800, sid=1)
    other.segments = other.segments[::-1]  # head far from the hero: (63, 40) first
    return hero, other


# Tail chase. A 12x10 board whose outer ring of cells is the only open corridor: every
# interior cell is another (static) snake, except a 3-cell dead-end pocket (5, 1..3). The
# hero (length 30, need 30) runs clockwise on the ring with its head at (5, 0) heading
# right and 10 free ring cells ahead before its own tail. Straight (around the ring)
# counts 10 statically but escapes once the tail moves (v4's release model); the right
# turn into the pocket counts 3 and has no escape. Q prefers the pocket.
RING_Q = [0.0, 1.0, 5.0, 0.0, 0.5, 0.2]
RING_MASK = [False, True, True, False, True, True]  # the left turn leaves the board


def ring_cells(width=12, height=10):
    top = [(x, 0) for x in range(width)]
    right = [(width - 1, y) for y in range(1, height)]
    bottom = [(x, height - 1) for x in range(width - 2, -1, -1)]
    left = [(0, y) for y in range(height - 2, 0, -1)]
    return top + right + bottom + left


def tail_chase():
    ring = ring_cells()
    head_index = ring.index((5, 0))
    body = [ring[(head_index - i) % len(ring)] for i in range(30)]
    hero = make_snake(body, (1, 0), width=120, height=100)
    pocket = {(5, 1), (5, 2), (5, 3)}
    interior = [(x, y) for y in range(1, 9) for x in range(1, 11) if (x, y) not in pocket]
    interior.sort(key=lambda c: -(abs(c[0] - 5) + abs(c[1])))  # head far from the hero
    other = make_snake(interior, (0, 1), width=120, height=100, sid=1)
    return hero, other


def flags_and_counts(hero, roster):
    cap, need = free_space_threshold(hero.length, hero._logical_length())
    features = hero._get_free_space_features(list(roster))
    return spacious_directions(features, cap, need), [round(f * cap) for f in features], need


class TestNoSpaciousFallback:
    def test_boost_into_tiny_pocket_becomes_normal_speed(self, setup_config):
        hero, other = boxed_pocket()
        roster = [hero, other]
        spacious, counts, need = flags_and_counts(hero, roster)
        assert not any(spacious) and need == 50 and counts[1] == 12
        cap, _ = free_space_threshold(hero.length, hero._logical_length())
        assert landing_count(hero, roster, 1, cap) == 1
        mask = [False, True, False, False, True, False]
        q = [0.0, 1.0, 0.0, 0.0, 9.0, 0.0]
        assert v5_apply(hero, roster, q, mask, 4) == 4  # v5: no_spacious, base unchanged
        action, veto = v6_apply(hero, roster, q, mask, 4)
        assert action == 1
        diag = veto.diagnostics_record()
        assert diag["fallback_landing_switches"] == 1 and diag["fallback_decisions"] == 1
        assert diag["fallback_escape_switches"] == 0 and diag["fallback_budget_unknown"] == 0
        assert veto.record()["counters"]["fallback_no_spacious"] == 1
        # Live mechanics: the boost dies within two frames; normal speed lives longer.
        assert not survives(boxed_pocket, [4], 2)
        assert survives(boxed_pocket, [1], 6)

    def test_ranking_chooses_the_escaping_move(self, setup_config):
        hero, other = tail_chase()
        roster = [hero, other]
        spacious, counts, need = flags_and_counts(hero, roster)
        assert not any(spacious) and need == 30 and counts[1] == 10 and counts[2] == 3
        assert v5_apply(hero, roster, RING_Q, RING_MASK, 2) == 2
        action, veto = v6_apply(hero, roster, RING_Q, RING_MASK, 2)
        assert action == 1  # around the ring behind the tail; the boost lands short
        diag = veto.diagnostics_record()
        assert diag["fallback_escape_switches"] == 1 and diag["fallback_landing_switches"] == 0
        assert diag["fallback_searches"] == 1 and diag["fallback_search_nodes"] > 0
        assert not survives(tail_chase, [2], 4)
        assert survives(tail_chase, [1], 14)

    def test_budget_exhaustion_changes_nothing(self, setup_config):
        hero, other = tail_chase()
        action, veto = v6_apply(hero, [hero, other], RING_Q, RING_MASK, 2, node_budget=1)
        assert action == 2
        diag = veto.diagnostics_record()
        assert diag["fallback_budget_unknown"] == 1 and diag["fallback_escape_switches"] == 0

    def test_layers_can_be_switched_off(self, setup_config):
        hero, other = tail_chase()
        roster = [hero, other]
        assert v6_apply(hero, roster, RING_Q, RING_MASK, 2, fallback=False)[0] == 2
        hero, other = boxed_pocket()
        mask = [False, True, False, False, True, False]
        q = [0.0, 1.0, 0.0, 0.0, 9.0, 0.0]
        assert v6_apply(hero, [hero, other], q, mask, 4, fallback=False)[0] == 4

    def test_pure_fallback_rule(self):
        q = [0.0, 1.0, 5.0, 0.0, 9.0, 0.2]
        mask = [True] * 6
        never = lambda d: False  # noqa: E731
        # Landing switch only for a firing boost below its normal count.
        assert fallback_choice(q, mask, 4, lambda d: d == 1, None).action == 1
        assert fallback_choice(q, mask, 4, never, None).action == 4
        # Escape ranking: highest Q among escaping directions, boosts included.
        result = fallback_choice(q, mask, 2, never, lambda d: d == 1)
        assert result.action == 4 and result.escape_switch
        # ... but a boost whose landing is short is not a candidate.
        result = fallback_choice(q, mask, 2, lambda d: d == 1, lambda d: d == 1)
        assert result.action == 1
        # Current direction escapes: kept. Nothing escapes: kept.
        assert fallback_choice(q, mask, 2, never, lambda d: True).action == 2
        assert fallback_choice(q, mask, 2, never, lambda d: False).action == 2

        def exhausted(direction):
            raise BudgetExhausted

        result = fallback_choice(q, mask, 4, lambda d: d == 1, exhausted)
        assert result.action == 1 and result.budget_unknown  # step 1 stands


# ------------------------------------------------------------------ rule properties
def random_head_states(seed, count):
    """v5's random coiled worlds, with every other snake given a random live heading."""
    rng = random.Random(seed)
    worlds = random_states(seed, count)
    for _, roster, _, _, _ in worlds:
        for other in roster[1:]:
            other.direction = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    return worlds


class TestRuleProperties:
    @pytest.mark.parametrize("version", [1, 2])
    def test_equals_v5_unless_a_layer_triggers(self, mechanics, version):
        mechanics(version)
        untriggered = differing = 0
        for me, roster, q, mask, base in random_head_states(11 + version, 400):
            action, veto = v6_apply(me, roster, q, mask, base)
            diag = veto.diagnostics_record()
            v5_action = v5_apply(me, roster, q, mask, base)
            assert diag["v5"]["decisions"] == 1
            if diag["head_risky_decisions"] == 0 and diag["fallback_decisions"] == 0:
                untriggered += 1
                assert action == v5_action
            differing += int(action != v5_action)
            assert diag["action_differs_from_v5"] == int(action != v5_action)
            # Both layers off: exactly v5, always.
            off, _ = v6_apply(me, roster, q, mask, base, head_avoidance=False, fallback=False)
            assert off == v5_action
        assert untriggered >= 200 and differing > 0

    def test_never_returns_a_masked_action_and_is_deterministic(self, mechanics):
        mechanics(2)
        for me, roster, q, mask, base in random_head_states(5, 300):
            first, veto_a = v6_apply(me, roster, q, mask, base)
            second, veto_b = v6_apply(me, roster, q, mask, base)
            assert first == second
            assert deterministic(veto_a.diagnostics_record()) == deterministic(
                veto_b.diagnostics_record()
            )
            assert bool(mask[first]), (first, mask.tolist())

    def test_head_reach_covers_every_live_head_on(self, mechanics):
        mechanics(1)  # every head-on is mutual death: nothing is waived
        checked = 0
        for me, roster, q, mask, base in random_head_states(23, 200):
            grid = grid_for(me)
            risky, waived = head_risky_actions(me, opponent_head_reach(me, roster, grid), grid)
            assert not any(waived)
            for action in range(6):
                for other in roster[1:]:
                    if any(live_pair(me, other, action, o) for o in range(6)):
                        checked += 1
                        assert risky[action], (action, other.segments[:2])
        assert checked > 0

    def test_probe_counters_and_diagnostics_identities(self, mechanics):
        mechanics(2)
        veto = HeadAndFallbackVeto()
        for me, roster, q, mask, base in random_head_states(31, 300):
            veto.apply(me, roster, q, mask, base)
        counters = veto.record()["counters"]
        diag = veto.diagnostics_record()
        assert counters["decisions"] == diag["decisions"] == 300
        assert counters["kept_base"] == diag["kept"]
        assert counters["vetoes_applied"] == diag["vetoes_applied"]
        assert counters["fallback_no_spacious"] == diag["no_spacious"] == diag["fallback_decisions"]
        assert diag["head_risky_decisions"] == (
            diag["head_risky_vetoes"] + diag["head_risky_kept_no_alternative"]
        )
        assert diag["fallback_searches"] + diag["fallback_skipped_no_search"] == (
            diag["fallback_decisions"]
        )
        assert diag["v5"]["no_spacious"] == diag["no_spacious"]
        # A head veto of a v5-kept base turns kept into vetoed; of a v5 veto, stays one.
        assert diag["head_vetoes_of_v5_kept"] <= diag["head_risky_vetoes"]
        assert diag["kept"] == diag["v5"]["kept"] - diag["head_vetoes_of_v5_kept"]
        assert diag["vetoes_applied"] == (
            diag["v5"]["vetoes_applied"] + diag["head_vetoes_of_v5_kept"]
        )


# ------------------------------------------------------------------ hook and record
class TestHookAndRecord:
    def test_descriptor_and_record_shape(self):
        veto = HeadAndFallbackVeto()
        descriptor = veto.descriptor()
        assert descriptor["method"] == VETO_METHOD_V6 == "free-space-veto/v6-head-and-fallback"
        assert descriptor["head_avoidance"] is True and descriptor["no_spacious_fallback"] is True
        assert descriptor["fallback_depth"] == FALLBACK_DEPTH == 8
        assert descriptor["fallback_node_budget"] == FALLBACK_NODE_BUDGET == 2000
        v5 = BoostAwareFreeSpaceVeto().descriptor()
        for key in ("free_space_bfs_cap", "free_space_min_cap", "boost_approximation"):
            assert descriptor[key] == v5[key]
        assert HeadAndFallbackVeto(fallback=False).descriptor() != descriptor
        record = veto.record()
        assert set(record) == set(descriptor) | {"counters"}
        assert set(record["counters"]) == set(BoostAwareFreeSpaceVeto().record()["counters"])

    def test_record_passes_the_strict_wrapper_probe_validator(self, mechanics):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        mechanics(2)
        veto = HeadAndFallbackVeto()
        hero, opponent = head_on_world(12, 15)
        veto.apply(hero, [hero, opponent], torch.tensor(HEAD_Q), torch.tensor(ALL), 1)
        hero, other = tail_chase()
        veto.apply(hero, [hero, other], RING_Q, RING_MASK, 2)
        _validate_candidate_wrapper_probe(veto.record(), veto.descriptor())
        diag = veto.diagnostics_record()
        assert diag["decisions"] == 2 and diag["head_risky_vetoes"] == 1
        assert diag["fallback_escape_switches"] == 1
        assert _screen().v6_identities_hold(diag, veto.counters.to_dict())
        veto.reset()
        assert veto.counters.decisions == 0 and veto.diagnostics_record()["decisions"] == 0

    def test_install_requires_an_ai_snake(self):
        with pytest.raises(TypeError):
            install_head_and_fallback_veto(object())
        with pytest.raises(ValueError):
            HeadAndFallbackVeto(node_budget=0)

    def test_ai_snake_default_off_and_installed_v6(self, setup_config):
        from src.game.ai_snake import AISnake

        class DqnPolicyStub:
            epsilon = 0.0

            def dqn(self, state):
                return torch.tensor([HEAD_Q])

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=DqnPolicyStub())
        snake.direction = (1, 0)
        assert snake.safety_veto is None
        veto = install_head_and_fallback_veto(snake)
        assert snake.safety_veto is veto
        snake.update([snake], [(100, 100)])
        assert veto.counters.decisions == 1 == veto.diagnostics_record()["decisions"]


def _screen():
    from research.apex_veto_v6_screen_20261002 import screen

    return screen


@pytest.fixture
def tiny_live(setup_config, tmp_path):
    """Tiny live world plus a random 58-D vector checkpoint hero (as test_safety_veto_v5)."""
    from src.core.config_loader import load_and_initialize_config
    from src.model.apex_network import ApexNetwork

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(
        "game:\n  width: 300\n  height: 200\n  num_snakes: 3\n"
        "  initial_food: 12\n  max_food: 12\n"
    )
    load_and_initialize_config(str(cfg))
    torch.manual_seed(0)
    net = ApexNetwork(input_size=58, hidden_size=16, output_size=6)
    path = tmp_path / "vector.pth"
    torch.save({"dqn_state_dict": net.state_dict(), "input_size": 58, "hidden_size": 16}, path)
    return ("checkpoint", str(path))


@pytest.fixture
def screen_runner_fatal(monkeypatch):
    """Direct ``rollout`` tests may never reach the screen's episode runner."""
    from research.apex_safety_20260926 import dev_screen

    def fatal(*args, **kwargs):
        raise AssertionError("this test must not run screen episodes")

    monkeypatch.setattr(dev_screen, "run_episode", fatal)


class TestLiveRollout:
    """Direct ``rollout`` calls on a tiny synthetic world (never the screen harness)."""

    def _play(self, tiny_live, install):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts.tournament_eval import rollout

        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        with dev_screen.hero_veto_installer(install) as installed:
            record = rollout(tiny_live, opponents, 60, 3, hero_safety_veto=True)
        return record, installed[-1]

    def test_v6_probe_record_and_counters_from_a_live_rollout(self, tiny_live, screen_runner_fatal):
        screen = _screen()
        record, veto = self._play(tiny_live, screen.install_v6)
        probe = record["probes"]["safety_veto"]
        kwargs = dict(world_seed=3, mix="scripted", roster_hashes=[], smoke=True)
        assert probe["method"] == screen.V6_METHOD
        assert screen.check_record(record, "B", **kwargs) == []
        assert screen.check_record(record, "A", **kwargs)
        diag = veto.diagnostics_record()
        assert diag["decisions"] == probe["counters"]["decisions"] > 0
        assert diag["head_checks"] > 0
        assert screen.v6_identities_hold(diag, probe["counters"])
        again, veto_again = self._play(tiny_live, screen.install_v6)
        assert again == record  # deterministic replay (the D arm's premise)
        assert deterministic(veto_again.diagnostics_record()) == deterministic(diag)

    def test_v5_arm_installer_matches_the_released_veto(self, tiny_live, screen_runner_fatal):
        screen = _screen()
        record, veto = self._play(tiny_live, screen.install_v5)
        kwargs = dict(world_seed=3, mix="scripted", roster_hashes=[], smoke=True)
        assert record["probes"]["safety_veto"]["method"] == screen.V5_METHOD
        assert screen.check_record(record, "A", **kwargs) == []
        assert isinstance(veto, BoostAwareFreeSpaceVeto)


class TestForcedLayersInALiveRollout:
    """Short live episodes rarely meet a head-on or a no-spacious state, so each layer is
    forced to trigger to pin its live path (counters, self-check, replay identity)."""

    def _play_twice(self, tiny_live):
        screen = _screen()
        play = TestLiveRollout()._play

        def long_hero_v6(hero):  # a length-1 hero cannot boost (the mask forbids it)
            hero.grow(11)
            return screen.install_v6(hero)

        first = play(tiny_live, long_hero_v6)
        second = play(tiny_live, long_hero_v6)
        assert second[0] == first[0]
        assert deterministic(second[1].diagnostics_record()) == deterministic(
            first[1].diagnostics_record()
        )
        record, veto = first
        diag = veto.diagnostics_record()
        counters = record["probes"]["safety_veto"]["counters"]
        kwargs = dict(world_seed=3, mix="scripted", roster_hashes=[], smoke=True)
        assert screen.check_record(record, "B", **kwargs) == []
        assert screen.v6_identities_hold(diag, counters)
        return diag, counters

    def test_forced_head_risk(self, tiny_live, screen_runner_fatal, monkeypatch):
        from src.evaluation import safety_veto_v6

        def straight_is_risky(snake, threats, grid):
            return [False, True, False, False, True, False], [False] * 6

        monkeypatch.setattr(safety_veto_v6, "head_risky_actions", straight_is_risky)
        diag, counters = self._play_twice(tiny_live)
        assert diag["head_risky_decisions"] > 0 and diag["head_risky_vetoes"] > 0
        assert counters["vetoes_applied"] >= diag["head_risky_vetoes"]

    def test_forced_no_spacious_fallback(self, tiny_live, screen_runner_fatal, monkeypatch):
        from src.evaluation import safety_veto_v6

        monkeypatch.setattr(
            safety_veto_v6, "spacious_directions", lambda features, cap, need: [False] * 3
        )
        diag, counters = self._play_twice(tiny_live)
        assert diag["fallback_decisions"] == diag["decisions"] == counters["fallback_no_spacious"]
        assert diag["fallback_searches"] > 0 and diag["fallback_search_nodes"] > 0
        assert diag["head_checks"] == 0  # the layers never overlap


# ---------------------------------------------------------------- Tier-1 screen wrapper
FIXTURES = REPO / "research/apex_veto_v3_screen_20261001/fixtures"
REAL_CANDIDATE = FIXTURES / "final-candidate-frozen-1010319811.json"  # run-v3, v2 veto
V2_COUNTERS = {
    "decisions": 10,
    "kept_base": 7,
    "vetoes_applied": 2,
    "fallback_no_spacious": 1,
    "vetoes_to_boost": 0,
    "vetoed_base_boost": 1,
    "vetoes_speed_switched": 1,
}


@pytest.fixture
def no_episodes(monkeypatch):
    """Make every episode runner fatal (v3 revision 4: a guard test ran a real screen)."""
    from research.apex_safety_20260926 import dev_screen
    from src.scripts import tournament_eval

    def fatal(*args, **kwargs):
        raise AssertionError("a guard test must never play an episode")

    monkeypatch.setattr(tournament_eval, "rollout", fatal)
    monkeypatch.setattr(dev_screen, "run_episode", fatal)


def _real():
    return json.loads(REAL_CANDIDATE.read_text())


def _entry(arm, mix, seed, mass):
    record = {
        "mass_integral": mass,
        "survival_fraction": 0.5,
        "probes": {"death_cause": "head_on", "safety_veto": {"counters": dict(V2_COUNTERS)}},
        "denominators": {"decision_frames": 10},
    }
    return {"arm": arm, "mix": mix, "world_seed": seed, "wall_seconds": 1.0, "record": record}


def _v6_diag(counters, **extra):
    """v6 diagnostics consistent with ``counters`` (one head veto of a v5-kept base)."""
    diag = {
        "decisions": counters["decisions"],
        "kept": counters["kept_base"],
        "vetoes_applied": counters["vetoes_applied"],
        "no_spacious": counters["fallback_no_spacious"],
        "head_risky_decisions": 2,
        "head_risky_vetoes": 1,
        "head_risky_kept_no_alternative": 1,
        "head_vetoes_of_v5_kept": 1,
        "fallback_decisions": counters["fallback_no_spacious"],
        "fallback_searches": counters["fallback_no_spacious"],
        "fallback_skipped_no_search": 0,
        "v5": {
            "decisions": counters["decisions"],
            "kept": counters["kept_base"] + 1,
            "vetoes_applied": counters["vetoes_applied"] - 1,
            "no_spacious": counters["fallback_no_spacious"],
        },
    }
    diag.update(extra)
    return diag


def _v5_diag(counters):
    return {
        "decisions": counters["decisions"],
        "kept": counters["kept_base"],
        "vetoes_applied": counters["vetoes_applied"],
        "no_spacious": counters["fallback_no_spacious"],
        "boost_landing_vetoes": 0,
        "boost_to_normal_same_direction": 0,
    }


class TestScreenSpecAndNamespaces:
    def test_worlds_are_fresh_and_cover_every_earlier_bank(self):
        from research.apex_safety_20260926 import dev_screen
        from research.apex_veto_v5_screen_20261001 import screen as v5screen
        from research.apex_veto_v5_strict_20261001 import strict_run as v5strict
        from research.trap_horizon_20261001 import diagnose_live_v5 as census

        screen = _screen()
        earlier = set(screen.EARLIER_DOMAINS)
        assert set(v5screen.EARLIER_DOMAINS) <= earlier
        assert set(v5strict.EARLIER_DOMAINS) <= earlier
        v5_banks = {domain for domain, _ in v5strict.NAMESPACES.values()}
        assert v5_banks <= earlier and v5strict.SMOKE_DOMAIN in earlier
        assert set(census.V5_DOMAINS) <= earlier
        assert {census.DOMAIN, census.SMOKE_DOMAIN, v5screen.DOMAIN, v5screen.SMOKE_DOMAIN} <= (
            earlier
        )
        assert dev_screen.SCREEN_DOMAIN in earlier  # also the SIMD H5000 parity worlds
        for name, purposes in census.V5_DOMAINS.items():
            assert set(purposes) <= set(screen.EARLIER_DOMAINS[name])
        assert screen.DOMAIN == "apex-veto-v6-screen-v1"
        assert not any(k.startswith(f"{screen.DOMAIN}/") for k in screen.SPEC.extra_namespaces)
        assert any(k.startswith(f"{screen.SMOKE_DOMAIN}/") for k in screen.SPEC.extra_namespaces)
        assert not any(
            k.startswith(f"{screen.SMOKE_DOMAIN}/") for k in screen.SMOKE_SPEC.extra_namespaces
        )
        seeds = dev_screen.screen_seeds(40, screen.DOMAIN, screen.NAMESPACE)
        assert len(set(seeds)) == 40
        report = dev_screen.disjointness_report(
            seeds, None, screen_domain=screen.DOMAIN, extra_namespaces=screen.SPEC.extra_namespaces
        )
        assert report["disjoint"] is True
        assert len(report["extra_namespaces"]) == len(screen.SPEC.extra_namespaces) >= 38
        smoke = dev_screen.screen_seeds(2, screen.SMOKE_DOMAIN, screen.NAMESPACE)
        assert not set(smoke) & set(seeds)

    def test_specs(self):
        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        spec = screen.SPEC
        assert spec.arm_vetoes["A"] is spec.arm_vetoes["C"] is screen.install_v5
        assert spec.arm_vetoes["B"] is spec.arm_vetoes["D"] is screen.install_v6
        assert spec.replay_worlds == screen.REPLAY_WORLDS == 4
        assert spec.veto_arms() == ("A", "B")
        assert spec.require_slot_locks and spec.require_ac_power
        assert spec.max_wall_seconds == screen.MAX_WALL_SECONDS == 4 * 3600
        assert screen.SMOKE_SPEC.domain == screen.SMOKE_DOMAIN != spec.domain
        assert dev_screen.DEFAULT_SPEC.replay_worlds == 0  # harness defaults untouched
        assert screen.expected_descriptor("D") == screen.expected_descriptor("B")
        assert screen.expected_descriptor("B") == HeadAndFallbackVeto().descriptor()
        assert screen.expected_descriptor("C") == BoostAwareFreeSpaceVeto().descriptor()

    def test_intent_fields_bind_protocol_rule_and_cap(self):
        from datetime import datetime, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        deadline = datetime(2026, 10, 3, tzinfo=timezone.utc)
        fields = dev_screen.spec_intent_fields(screen.SPEC, 276, deadline)
        protocol = REPO / "research/apex_veto_v6_screen_20261002/protocol.md"
        assert fields["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
        assert fields["decision_rule"].startswith(dev_screen.decision_rule_text())
        assert "B/D replay control on 4 worlds" in fields["decision_rule"]
        assert fields["compute_cap"] == {
            "planned_episodes": 276,
            "deadline_utc": deadline.isoformat(),
            "max_wall_seconds": 4 * 3600,
        }
        assert fields["owner"] == "Apex safety lane" and "v5" in fields["hypothesis"]
        for key in ("decision_informs", "primary_metric", "estimator"):
            assert fields[key]
        text = protocol.read_text()
        for phrase in (
            "4 h (14400 s)",
            "276 episodes",
            "trap-horizon-v5-dev-v1",
            "death_census_v5_2026-10-02.md",
            "40 worlds",
            "apex-veto-v6-screen-smoke-v1",
        ):
            assert phrase in text


class TestScreenSelfCheck:
    def _check(self, entry, arm, record):
        return _screen().check_record(
            record,
            arm,
            world_seed=entry["world_seed"],
            mix=entry["mix"],
            roster_hashes=entry["roster_member_sha256s"],
            smoke=False,
        )

    def _record(self, entry, veto):
        record = copy.deepcopy(entry["record"])
        counters = record["probes"]["safety_veto"]["counters"]
        record["probes"]["safety_veto"] = {**veto.descriptor(), "counters": counters}
        return record

    def test_v5_and_v6_probes_on_the_real_record(self):
        entry = _real()
        v5 = self._record(entry, BoostAwareFreeSpaceVeto())
        v6 = self._record(entry, HeadAndFallbackVeto())
        assert self._check(entry, "A", v5) == [] and self._check(entry, "C", v5) == []
        assert self._check(entry, "B", v6) == [] and self._check(entry, "D", v6) == []
        assert any("arm B" in f for f in self._check(entry, "B", v5))
        assert any("arm A" in f for f in self._check(entry, "A", v6))
        assert self._check(entry, "B", entry["record"])  # the v2 probe is neither arm's
        off = self._record(entry, HeadAndFallbackVeto(head_avoidance=False))
        assert self._check(entry, "B", off)  # a layer switched off is another identity
        broken = copy.deepcopy(v6)
        broken["denominators"]["scored_frames"] = 4999
        assert any("H5000" in f for f in self._check(entry, "B", broken))
        assert any("seed" in f for f in self._check(dict(entry, world_seed=1), "B", v6))

    def test_entry_rules_and_diagnostics_warnings(self):
        screen = _screen()
        real = _real()
        record = self._record(real, HeadAndFallbackVeto())
        counters = record["probes"]["safety_veto"]["counters"]
        entry = {
            "schema_version": screen.SCHEMA,
            "authority": screen.AUTHORITY,
            "screen": screen.SCREEN_ID,
            "arm": "B",
            "mix": real["mix"],
            "world_seed": real["world_seed"],
            "roster_member_sha256s": real["roster_member_sha256s"],
            "hero_sha256": real["hero_sha256"],
            "safety_veto": True,
            "safety_veto_method": screen.V6_METHOD,
            "veto_diagnostics": _v6_diag(counters),
            "record": record,
        }
        assert screen.check_entry(entry, smoke=False) == {"failures": [], "warnings": []}
        assert screen.check_entry(dict(entry, arm="D"), smoke=False)["failures"] == []
        for bad in (
            _v6_diag(counters, decisions=counters["decisions"] + 1),
            _v6_diag(counters, head_risky_vetoes=2),
            _v6_diag(counters, fallback_searches=0),
            _v6_diag(counters, head_vetoes_of_v5_kept=0),
            None,
        ):
            assert screen.check_entry(dict(entry, veto_diagnostics=bad), smoke=False)["warnings"]
        a_record = self._record(real, BoostAwareFreeSpaceVeto())
        a_entry = dict(
            entry,
            arm="A",
            safety_veto_method=screen.V5_METHOD,
            record=a_record,
            veto_diagnostics=_v5_diag(counters),
        )
        assert screen.check_entry(a_entry, smoke=False) == {"failures": [], "warnings": []}
        assert screen.check_entry(dict(a_entry, arm="C"), smoke=False)["failures"] == []
        v6_on_a = dict(a_entry, veto_diagnostics=_v6_diag(counters, boost_landing_vetoes=0))
        assert screen.check_entry(v6_on_a, smoke=False)["warnings"]
        assert screen.check_entry(dict(entry, arm="A"), smoke=False)["failures"]
        assert screen.check_entry(dict(entry, schema_version="x"), smoke=False)["failures"]
        assert screen.check_entry(dict(entry, arm="Z"), smoke=False)["failures"]


class TestScreenPlan:
    SEEDS = [11, 12, 13]

    def test_plan_runs_d_after_c_and_never_plays_a_real_episode(
        self, no_episodes, monkeypatch, tmp_path
    ):
        import argparse
        from datetime import datetime, timedelta, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        played = []

        def fake_episode(arm, row, world_index, lookup, profile, records_dir, **kwargs):
            assert kwargs["spec"] is screen.SPEC
            played.append((arm, row["mix"], row["world_seed"]))
            return _entry(arm, row["mix"], row["world_seed"], 30.0)

        monkeypatch.setattr(dev_screen, "run_episode", fake_episode)
        monkeypatch.setattr(dev_screen, "snapshot_checkpoints", lambda *a: {})
        monkeypatch.setattr(dev_screen, "agent_lookup", lambda *a: {})
        args = argparse.Namespace(
            mixes=dev_screen.MIXES,
            worlds_per_mix=3,
            determinism_worlds=1,
            deadline_utc=datetime.now(timezone.utc) + timedelta(hours=1),
            min_episode_budget_seconds=0.0,
            smoke_frames=None,
            config=dev_screen.DEFAULT_CONFIG,
            checkpoint_dir=tmp_path / "no-checkpoints",
            slot_lock_root=tmp_path,
            slots=1,
            require_ac_power=False,
        )
        out = tmp_path / "o"
        code = dev_screen._run_screen(
            args, [], screen.SPEC, out, self.SEEDS, {}, {}, None, False, False
        )
        assert code == 0
        arms = [arm for arm, _, _ in played]
        assert arms.count("A") == arms.count("B") == 9 and arms.count("C") == 3
        assert arms.count("D") == 9  # min(4, 3 worlds) per mix
        assert arms.index("D") > max(i for i, arm in enumerate(arms) if arm == "C")
        intent = json.loads((out / "intent.json").read_text())
        assert intent["screen"] == screen.SCREEN_ID and intent["schema_version"] == screen.SCHEMA
        assert intent["compute_cap"]["max_wall_seconds"] == 4 * 3600
        assert intent["worlds"]["domain"] == screen.DOMAIN
        for key in ("owner", "hypothesis", "decision_informs", "primary_metric", "estimator"):
            assert intent[key] == getattr(screen.SPEC, key)
        summary = json.loads((out / "summary.json").read_text())
        assert summary["replay_control"]["passes"] is True
        assert summary["planned_episodes"] == 30
        assert summary["decision"] == "NON_PREREGISTERED_DESIGN"


class TestScreenGuards:
    def _argv(self, tmp_path, *extra):
        return ["--out", str(tmp_path / "o"), "--deadline-utc", "2099-01-01T00:00:00+00:00", *extra]

    @pytest.mark.parametrize(
        "extra",
        [
            ("--worlds-per-mix", "5", "--determinism-worlds", "2"),
            ("--determinism-worlds", "0"),
            ("--worlds-per-mix", "41", "--determinism-worlds", "8"),
        ],
    )
    def test_non_preregistered_sizes_are_refused(self, no_episodes, tmp_path, extra):
        assert _screen().main(self._argv(tmp_path, *extra)) == 2
        assert not (tmp_path / "o").exists()

    def test_far_deadline_hits_the_wall_cap(self, no_episodes, tmp_path):
        from datetime import datetime, timedelta, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        assert screen.main(self._argv(tmp_path)) == 2
        late = (datetime.now(timezone.utc) + timedelta(hours=4, minutes=5)).isoformat()
        argv = ["--out", str(tmp_path / "o"), "--deadline-utc", late]
        assert screen.main(argv) == 2
        assert dev_screen.main(argv, spec=screen.SPEC) == 2
        assert not (tmp_path / "o").exists()

    def test_spec_with_mismatched_control_arms_is_refused(self, no_episodes, tmp_path):
        from datetime import datetime, timedelta, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        soon = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        argv = ["--out", str(tmp_path / "o"), "--deadline-utc", soon]
        for arm, other in (("D", screen.install_v5), ("C", screen.install_v6)):
            arms = dict(screen.SPEC.arm_vetoes, **{arm: other})
            bad = dataclasses.replace(screen.SPEC, arm_vetoes=arms)
            assert dev_screen.main(argv, spec=bad) == 2
        assert not (tmp_path / "o").exists()

    def test_smoke_size_guard(self, no_episodes, tmp_path):
        argv = self._argv(tmp_path, "--smoke-frames", "501", "--worlds-per-mix", "1")
        with pytest.raises(SystemExit):
            _screen().main(argv + ["--determinism-worlds", "0"])
        argv = self._argv(tmp_path, "--smoke-frames", "500", "--worlds-per-mix", "2")
        with pytest.raises(SystemExit):  # 4 episodes > 2
            _screen().main(argv + ["--determinism-worlds", "0"])
        assert not (tmp_path / "o").exists()


class TestScreenReports:
    def test_v6_layer_report_rates_and_totals(self):
        screen = _screen()
        diag = {key: 0 for key in screen.SUMMED_V6_FIELDS}
        diag.update(decisions=100, head_risky_vetoes=4, fallback_decisions=10)
        diag.update(fallback_landing_switches=1, fallback_escape_switches=2, fallback_searches=8)
        diag.update(fallback_budget_unknown=2, action_differs_from_v5=7, apply_seconds_total=0.5)
        diag.update(fallback_seconds_total=0.2, head_risky_decisions=5)
        diag["apply_seconds_max"] = 0.01
        diag["fallback_seconds_max"] = 0.02
        entries = [
            dict(_entry("B", "frozen", 1, 1.0), veto_diagnostics=dict(diag), wall_seconds=4.0),
            dict(_entry("B", "mixed", 1, 1.0), veto_diagnostics=dict(diag), wall_seconds=6.0),
            dict(_entry("B", "mixed", 2, 1.0), veto_diagnostics=None, wall_seconds=6.0),
            dict(_entry("A", "mixed", 1, 1.0), wall_seconds=2.0),
        ]
        report = screen.v6_layer_report(entries)
        total = report["total"]
        assert total["episodes"] == 3 and total["missing_diagnostics"] == 1
        assert total["decisions"] == 200 and total["head_risky_vetoes"] == 8
        assert total["head_veto_rate_per_decision"] == pytest.approx(0.04)
        assert total["fallback_switch_rate_per_fallback"] == pytest.approx(0.3)
        assert total["budget_unknown_rate_per_search"] == pytest.approx(0.25)
        assert total["mean_apply_seconds_per_decision"] == pytest.approx(0.005)
        assert total["mean_fallback_seconds_per_fallback"] == pytest.approx(0.02)
        assert total["apply_seconds_max"] == 0.01 and total["fallback_seconds_max"] == 0.02
        assert report["per_mix"]["mixed"]["mean_episode_wall_seconds_arm_A"] == 2.0
        assert report["per_mix"]["frozen"]["mean_episode_wall_seconds"] == 4.0
        assert report["gating"] is False and screen.v6_layer_report([])["total"] == {}

    def test_death_cause_report(self):
        screen = _screen()
        entries = [_entry("A", "scripted", 1, 0.0), _entry("B", "scripted", 1, 0.0)]
        entries[1]["record"]["probes"]["death_cause"] = None
        table = screen.death_cause_report(entries)["by_arm_and_mix"]
        assert table == {"A": {"scripted": {"head_on": 1}}, "B": {"scripted": {"survived": 1}}}
