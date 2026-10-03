"""Tests for the opt-in v8 veto (v7's space preference, then v6's head avoidance).

Pure and constructed-state tests only: nothing here plays a game or a rollout.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch

from src.core import game_config
from src.core.game_config import GameConfig, get_config
from src.evaluation.safety_veto_v3 import grid_for, static_blocked
from src.evaluation.safety_veto_v5 import REASON_LANDING_SAME_DIRECTION
from src.evaluation.safety_veto_v6 import (
    head_avoid_choice,
    head_risky_actions,
    opponent_head_reach,
)
from src.evaluation.safety_veto_v7 import (
    SpacePreferenceVeto,
    area_cap,
    area_score,
    eager_space_preference_choice,
    landing_area,
)
from src.evaluation.safety_veto_v8 import (
    SpaceAndHeadVeto,
    head_replacement_choice,
    install_space_and_head_veto,
    method_for,
)
from tests.test_safety_veto_v5 import random_states
from tests.test_safety_veto_v6 import HEAD_Q, head_on_world, live_pair
from tests.test_safety_veto_v7 import v5_reference

REPO = Path(__file__).resolve().parents[1]
# Released v2 (strict receipt) and v5 (served Watch hero); retired v3/v4; screened v6/v7.
FROZEN = {
    "safety_veto.py": "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    "safety_veto_v3.py": "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
    "safety_veto_v4.py": "3f0881afb7ff8ff980c9b425ef40134f127cc90f1f40260aac39f2264ed44578",
    "safety_veto_v5.py": "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c",
    "safety_veto_v6.py": "a6117cb98383f9f28bf8ebcf22d2ef63d7a03995734a36fb96bb1bf8bb1757d5",
    "safety_veto_v7.py": "56ff7009ce2e0c4c93b6570336b35d48341757fc53b386d309b00126f67e2980",
}
LAMBDAS = (0.0, 1.0, 4.0, 8.0, 16.0)
ALL = [True] * 6


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if "seconds" not in k}
    return value


@pytest.fixture
def mechanics(setup_config, monkeypatch):
    """Set ``GameConfig.MECHANICS_VERSION`` for one test (restored by monkeypatch)."""

    def set_version(version):
        config = get_config()
        game = dataclasses.replace(config.game, mechanics_version=int(version))
        monkeypatch.setattr(game_config, "_current_config", dataclasses.replace(config, game=game))
        assert GameConfig.MECHANICS_VERSION == int(version)

    return set_version


class TestFrozenSources:
    def test_released_retired_and_screened_veto_bytes_are_unchanged(self):
        for name, sha in FROZEN.items():
            source = REPO / "src/evaluation" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name


# ---------------------------------------------------------------- equality with v7


class TestEqualsV7WhenHeadLayerIdle:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_head_layer_off_at_lambda_4_is_v7_exactly(self, setup_config, seed):
        v7, v8 = SpacePreferenceVeto(4.0), SpaceAndHeadVeto(4.0, head_avoidance=False)
        for me, roster, q, mask, base in random_states(seed, 150):
            assert v8.apply(me, roster, q, mask, base) == v7.apply(me, roster, q, mask, base)
        assert v8.counters.to_dict() == v7.counters.to_dict()
        diag = _strip_timing(v8.diagnostics_record())
        assert diag["v7"] == _strip_timing(v7.diagnostics_record())
        assert diag["v7_probe_counters"] == v7.counters.to_dict()
        assert diag["head_checks"] == diag["action_differs_from_v7"] == 0

    @pytest.mark.parametrize("lam", [1.0, 16.0])
    def test_head_layer_off_is_v7_at_any_lambda(self, setup_config, lam):
        v7, v8 = SpacePreferenceVeto(lam), SpaceAndHeadVeto(lam, head_avoidance=False)
        for me, roster, q, mask, base in random_states(2, 100):
            assert v8.apply(me, roster, q, mask, base) == v7.apply(me, roster, q, mask, base)
        assert v8.counters.to_dict() == v7.counters.to_dict()

    @pytest.mark.parametrize("seed", [3, 4])
    def test_head_layer_on_differs_from_v7_only_on_head_vetoes(self, setup_config, seed):
        v7, v8 = SpacePreferenceVeto(4.0), SpaceAndHeadVeto(4.0)
        differ = 0
        for me, roster, q, mask, base in random_states(seed, 200):
            before = v8.v8.head_risky_vetoes
            action, reference = v8.apply(me, roster, q, mask, base), v7.apply(
                me, roster, q, mask, base
            )
            if v8.v8.head_risky_vetoes == before:
                assert action == reference
            else:
                assert action != reference
                differ += 1
        diag = v8.diagnostics_record()
        assert differ == diag["head_risky_vetoes"] == diag["action_differs_from_v7"]
        assert _strip_timing(diag["v7"]) == _strip_timing(v7.diagnostics_record())
        assert diag["head_checks"] == diag["v7"]["decisions"] - diag["v7"]["no_spacious"]


# ---------------------------------------------------------------- head-on geometry


class TestHeadOnGeometry:
    @pytest.mark.parametrize("version", [1, 2])
    @pytest.mark.parametrize("lam", [0.0, 4.0, 16.0])
    def test_equal_or_larger_opponent_v7_keeps_and_v8_sidesteps(self, mechanics, version, lam):
        mechanics(version)
        hero, opponent = head_on_world(12, 15)
        roster = [hero, opponent]
        assert SpacePreferenceVeto(lam).apply(hero, roster, HEAD_Q, ALL, 1) == 1
        veto = SpaceAndHeadVeto(lam)
        action = veto.apply(hero, roster, HEAD_Q, ALL, 1)
        assert action == 2  # right turn: the only non-risky normal-speed move
        diag = veto.diagnostics_record()
        assert diag["head_risky_vetoes"] == 1 and diag["head_vetoes_of_v7_kept"] == 1
        assert diag["head_vetoes_speed_switched"] == 0
        counters = veto.record()["counters"]
        assert counters["vetoes_applied"] == 1 and counters["kept_base"] == 0
        assert live_pair(hero, opponent, 1, 1)  # the live game: straight can die head-on
        assert not any(live_pair(hero, opponent, 2, other) for other in range(6))

    def test_boost_mode_is_preserved_when_possible(self, mechanics):
        mechanics(2)
        from tests.test_safety_veto_v5 import make_snake

        hero = make_snake([(20 - i, 20) for i in range(12)], (1, 0), width=600, height=600)
        opponent = make_snake(
            [(23 + i, 20) for i in range(15)], (-1, 0), width=600, height=600, sid=1
        )
        q = [0.0, 1.0, 0.5, 0.0, 9.0, 0.3]  # boost straight first, then boost right
        veto = SpaceAndHeadVeto(4.0)
        assert veto.apply(hero, [hero, opponent], q, ALL, 4) == 5
        assert veto.diagnostics_record()["head_risky_vetoes"] == 1
        assert not any(live_pair(hero, opponent, 5, other) for other in range(6))

    def test_hero_wins_rule_is_respected(self, mechanics):
        mechanics(2)
        hero, opponent = head_on_world(40, 12)  # 40 >= 1.15 * (12 + 1): the hero survives
        veto = SpaceAndHeadVeto(4.0)
        assert veto.apply(hero, [hero, opponent], HEAD_Q, ALL, 1) == 1
        diag = veto.diagnostics_record()
        assert diag["head_risky_decisions"] == 0 and diag["head_risk_waived_hero_wins"] == 1
        # Near the margin (growth this frame could cancel it) and at mechanics v1: vetoed.
        hero, opponent = head_on_world(14, 12)
        assert SpaceAndHeadVeto(4.0).apply(hero, [hero, opponent], HEAD_Q, ALL, 1) == 2
        mechanics(1)
        hero, opponent = head_on_world(40, 12)
        assert SpaceAndHeadVeto(4.0).apply(hero, [hero, opponent], HEAD_Q, ALL, 1) == 2

    def test_no_alternative_keeps_v7_choice(self, mechanics):
        mechanics(1)
        hero, opponent = head_on_world(12, 15)
        mask = [False, True, False, False, True, False]  # only straight is legal
        veto = SpaceAndHeadVeto(4.0)
        assert veto.apply(hero, [hero, opponent], HEAD_Q, mask, 1) == 1
        diag = veto.diagnostics_record()
        assert diag["head_risky_kept_no_alternative"] == 1 and diag["head_risky_vetoes"] == 0
        assert veto.record()["counters"]["vetoes_applied"] == 0


# ---------------------------------------------------------------- replacement rule


class TestReplacementRule:
    Q = [0.0, 9.0, 1.0, 0.0, 8.0, 0.5]
    RISKY = [False, True, False, False, False, False]

    def test_lambda_zero_is_v6_highest_q(self):
        score = {0: 1.0, 2: 0.2}.get
        for eligible in ([True] * 6, [False, False, False, True, True, True]):
            v6_action, _ = head_avoid_choice(self.Q, eligible, self.RISKY, 1)
            action, reason, anchor = head_replacement_choice(
                self.Q, eligible, self.RISKY, 1, 0.0, lambda a: score(a, 1.0)
            )
            assert action == anchor == v6_action and reason == "head_veto"

    def test_space_score_picks_the_larger_region_among_non_risky(self):
        # Normal-mode options {0, 2}: Qn(2) = 0, Qn(0) = -1. g(0) = 1, g(2) = 0.2.
        def score(action):
            return {0: 1.0, 2: 0.2}[action]

        for lam, expected in ((1.0, 2), (1.25, 2), (2.0, 0), (4.0, 0), (16.0, 0)):
            action, reason, anchor = head_replacement_choice(
                self.Q, [True] * 6, self.RISKY, 1, lam, score
            )
            assert (action, reason, anchor) == (expected, "head_veto", 2), lam

    def test_never_risky_never_ineligible_and_same_mode_first(self):
        calls = []

        def score(action):
            calls.append(action)
            return 1.0 if action == 0 else 0.0

        risky = [True, True, False, False, False, False]
        action, _, _ = head_replacement_choice(self.Q, [True] * 6, risky, 1, 16.0, score)
        assert action == 2 and set(calls) <= {2}  # 0 is risky; R = {2}: nothing to score
        eligible = [False, True, False, True, True, True]  # no normal-mode alternative
        action, _, anchor = head_replacement_choice(self.Q, eligible, self.RISKY, 1, 16.0, score)
        assert anchor == 4 and action in (3, 4, 5) and action >= 3

    def test_not_risky_and_kept_paths(self):
        assert head_replacement_choice(self.Q, ALL, [False] * 6, 1, 4.0, None) == (1, "v7", None)
        only = [False] * 6
        assert head_replacement_choice(self.Q, only, self.RISKY, 1, 4.0, None) == (
            1,
            "head_kept_no_alternative",
            None,
        )
        with pytest.raises(ValueError):
            head_replacement_choice(self.Q, ALL, self.RISKY, 1, -1.0, None)


# ---------------------------------------------------------------- safety properties


def _head_reference(me, roster, q, mask, current, lam):
    """The head layer recomputed eagerly from v6/v7 pieces (full g table)."""
    _, _, _, eligible_set, (qs, ms, _, _) = v5_reference(me, roster, q, mask, 0)
    grid = grid_for(me)
    risky, _ = head_risky_actions(me, opponent_head_reach(me, roster, grid), grid)
    eligible = [a in eligible_set and not risky[a] for a in range(6)]
    anchor, reason = head_avoid_choice(qs, eligible, risky, current)
    if reason != "head_veto":
        return current
    pool = [
        a
        for a in range(6)
        if a != current and eligible[a] and not risky[a] and (a >= 3) == (anchor >= 3)
    ]
    blocked = static_blocked(me, roster, grid)
    cap = area_cap(me.length, grid, blocked)
    scores = {a: area_score(landing_area(me, a, blocked, grid, cap), cap) for a in pool}
    return eager_space_preference_choice(qs, pool, anchor, lam, scores)


class TestSafetyProperties:
    @pytest.mark.parametrize("seed", [5, 6])
    def test_never_masked_never_ineligible_never_risky_and_deterministic(self, setup_config, seed):
        states = random_states(seed, 200)
        vetoes = 0
        for lam in LAMBDAS:
            first, second, v7 = (
                SpaceAndHeadVeto(lam),
                SpaceAndHeadVeto(lam),
                SpacePreferenceVeto(lam),
            )
            for me, roster, q, mask, base in states:
                v5_action, outcome, reason, eligible, _ = v5_reference(me, roster, q, mask, base)
                v7_action = v7.apply(me, roster, q, mask, base)
                action = first.apply(me, roster, q, mask, base)
                assert action == second.apply(me, roster, q, mask, base)
                assert bool(mask[action])
                if outcome == "no_spacious":
                    assert action == v7_action == v5_action
                    continue
                expected = _head_reference(me, roster, q, mask, v7_action, lam)
                assert action == expected
                if action != v7_action:
                    vetoes += 1
                    assert action in eligible
                    grid = grid_for(me)
                    risky, _ = head_risky_actions(me, opponent_head_reach(me, roster, grid), grid)
                    assert risky[v7_action] and not risky[action]
                elif reason != REASON_LANDING_SAME_DIRECTION:
                    assert action in eligible
            assert first.counters.to_dict() == second.counters.to_dict()
            assert _strip_timing(first.diagnostics_record()) == _strip_timing(
                second.diagnostics_record()
            )
        assert vetoes > 0  # the head layer does act on these states

    def test_counter_identities(self, setup_config):
        veto = SpaceAndHeadVeto(8.0)
        for me, roster, q, mask, base in random_states(5, 200):
            veto.apply(me, roster, q, mask, base)
        c, d = veto.counters.to_dict(), veto.diagnostics_record()
        v7 = d["v7"]
        assert c["decisions"] == d["decisions"] == v7["decisions"] == 200
        assert c["kept_base"] == d["kept"] == v7["kept"] - d["head_vetoes_of_v7_kept"]
        assert c["vetoes_applied"] == d["vetoes_applied"]
        assert d["vetoes_applied"] == v7["vetoes_applied"] + d["head_vetoes_of_v7_kept"]
        assert c["fallback_no_spacious"] == d["no_spacious"] == v7["no_spacious"]
        assert d["head_risky_decisions"] == (
            d["head_risky_vetoes"] + d["head_risky_kept_no_alternative"]
        )
        assert d["action_differs_from_v7"] == d["head_risky_vetoes"] > 0
        assert d["v7_probe_counters"]["decisions"] == 200
        assert d["reference_decisions"] == 0 and d["reference_lambda"] is None


# ---------------------------------------------------------------- reference diagnostic


class TestReferenceDiagnostic:
    def test_reference_counts_differences_from_v7_at_the_reference_lambda(self, setup_config):
        states = random_states(8, 150)
        same = SpaceAndHeadVeto(4.0, reference_lambda=4.0)
        other = SpaceAndHeadVeto(16.0, reference_lambda=4.0)
        plain, arm_a = SpaceAndHeadVeto(16.0), SpacePreferenceVeto(4.0)
        differ = 0
        for me, roster, q, mask, base in states:
            same.apply(me, roster, q, mask, base)
            action = other.apply(me, roster, q, mask, base)
            assert action == plain.apply(me, roster, q, mask, base)  # never decides
            differ += int(action != arm_a.apply(me, roster, q, mask, base))
        d_same, d_other = same.diagnostics_record(), other.diagnostics_record()
        assert d_same["action_differs_from_reference"] == d_same["head_risky_vetoes"]
        assert d_other["action_differs_from_reference"] == differ > 0
        assert d_other["reference_decisions"] == 150 and d_other["reference_lambda"] == 4.0
        assert other.descriptor() == plain.descriptor()  # not part of the identity
        assert other.counters.to_dict() == plain.counters.to_dict()


# ---------------------------------------------------------------- hook and record


class TestHookAndRecord:
    def test_descriptor_and_record_shape(self, mechanics):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        mechanics(1)
        veto = SpaceAndHeadVeto(4.0)
        descriptor = veto.descriptor()
        assert descriptor["method"] == "free-space-veto/v8-space-and-head(lambda=4.0)"
        assert method_for(16) == "free-space-veto/v8-space-and-head(lambda=16.0)"
        v7 = SpacePreferenceVeto(4.0).descriptor()
        for key in ("space_preference_lambda", "area_rule", "area_cap_rule", "landing_rule"):
            assert descriptor[key] == v7[key]
        assert descriptor["head_avoidance"] is True
        assert SpaceAndHeadVeto(4.0, head_avoidance=False).descriptor() != descriptor
        hero, opponent = head_on_world(12, 15)
        veto.apply(hero, [hero, opponent], HEAD_Q, ALL, 1)
        record = veto.record()
        assert set(record) == set(descriptor) | {"counters"}
        _validate_candidate_wrapper_probe(record, descriptor)
        veto.reset()
        assert veto.counters.decisions == 0 and veto.diagnostics_record()["decisions"] == 0
        assert veto.diagnostics_record()["v7"]["decisions"] == 0

    def test_install_requires_an_ai_snake_and_default_is_off(self, setup_config):
        from src.game.ai_snake import AISnake

        with pytest.raises(TypeError):
            install_space_and_head_veto(object(), 4.0)

        class DqnPolicyStub:
            epsilon = 0.0

            def dqn(self, state):
                return torch.tensor([[0.0, 1.0, 0.5, 0.0, 0.2, 0.1]])

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=DqnPolicyStub())
        snake.direction = (1, 0)
        assert snake.safety_veto is None
        veto = install_space_and_head_veto(snake, 4.0, reference_lambda=4.0)
        assert snake.safety_veto is veto
        snake.update([snake], [(100, 100)])  # one decision on a constructed state, no game
        assert veto.counters.decisions == 1 == veto.diagnostics_record()["decisions"]

    def test_invalid_inputs(self, setup_config):
        with pytest.raises(ValueError):
            SpaceAndHeadVeto(float("nan"))
        with pytest.raises(ValueError):
            SpaceAndHeadVeto(1.0, reference_lambda=-2.0)
        hero, opponent = head_on_world(12, 15)
        with pytest.raises(ValueError):
            SpaceAndHeadVeto(1.0).apply(hero, [hero, opponent], [0.0] * 5, ALL, 1)


# ---------------------------------------------------------------- pocket + head-on
#
# v7's pocket world (tests/test_safety_veto_v7.py): the hero (length 100) heads north at
# (40, 30); left (west) enters a closed 168-cell pocket, straight (north) and right (east)
# reach the open board (area cap 200). An opponent of the same length has its head at
# (40, 27) heading south (body north up x = 40, then west along the top row, then down
# x = 0); boosting, it reaches (40, 29), the hero's straight cell, but not the hero's left
# (39, 30) or right (41, 30). Q prefers straight, then left (the pocket), then right.
POCKET_HEAD_Q = [1.0, 2.0, 0.95, -5.0, -5.0, -5.0]
NORMAL_ONLY = [True, True, True, False, False, False]


def pocket_head_world():
    from tests.test_safety_veto_v5 import make_snake
    from tests.test_safety_veto_v7 import pocket_world

    me, roster = pocket_world()
    cells = [(40, 27 - i) for i in range(28)] + [(x, 0) for x in range(39, -1, -1)]
    cells += [(0, y) for y in range(1, 33)]
    opponent = make_snake(cells[:100], (0, 1), width=me.game_width, height=me.game_height, sid=2)
    return me, [*roster, opponent], opponent


class TestPocketHeadOn:
    def test_geometry(self, mechanics):
        mechanics(2)
        me, roster, opponent = pocket_head_world()
        grid = grid_for(me)
        reach = opponent_head_reach(me, roster, grid)
        assert (40, 29) in reach and (39, 30) not in reach and (41, 30) not in reach
        risky, waived = head_risky_actions(me, reach, grid)
        assert risky[:3] == [False, True, False] and not any(waived)
        blocked = static_blocked(me, roster, grid)
        cap = area_cap(me.length, grid, blocked)
        assert cap == 200
        assert landing_area(me, 0, blocked, grid, cap) == 168
        assert landing_area(me, 2, blocked, grid, cap) == cap

    @pytest.mark.parametrize("lam, expected", [(0.0, 0), (4.0, 0), (8.0, 2), (16.0, 2)])
    def test_replacement_follows_the_v7_score(self, mechanics, lam, expected):
        # R = {left, right}: Qn(left) = 0, Qn(right) = -1, g(left) = 0.84, g(right) = 1, so
        # right wins iff 0.16 * lam > 1 (lam > 6.25). v7 alone keeps straight at every lam.
        mechanics(2)
        me, roster, _ = pocket_head_world()
        assert SpacePreferenceVeto(lam).apply(me, roster, POCKET_HEAD_Q, NORMAL_ONLY, 1) == 1
        veto = SpaceAndHeadVeto(lam)
        assert veto.apply(me, roster, POCKET_HEAD_Q, NORMAL_ONLY, 1) == expected
        diag = veto.diagnostics_record()
        assert diag["head_risky_vetoes"] == 1
        assert diag["head_vetoes_space_differs_from_highest_q"] == int(expected != 0)
        # Lazy as v7: at lam = 4 right's best case (-1 + 4) cannot beat left's 3.36.
        assert diag["head_area_evaluations"] == {0.0: 0, 4.0: 1}.get(lam, 2)
