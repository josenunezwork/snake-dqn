"""Tests for the opt-in v7 veto (space preference among v5's eligible moves)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch

from src.core.game_config import GameConfig
from src.evaluation.safety_veto import free_space_threshold, spacious_directions
from src.evaluation.safety_veto_v3 import grid_for, static_blocked, tail_aware_reachable
from src.evaluation.safety_veto_v5 import (
    REASON_LANDING_SAME_DIRECTION,
    BoostAwareFreeSpaceVeto,
    boost_aware_choice,
    landing_count,
)
from src.evaluation.safety_veto_v7 import (
    AREA_CAP_LIMIT,
    SpacePreferenceVeto,
    area_cap,
    area_score,
    eager_space_preference_choice,
    install_space_preference_veto,
    landing_area,
    method_for,
    rerank_candidates,
    space_preference_choice,
)
from src.game.game_logic import GameLogic
from tests.test_safety_veto_v5 import clone, make_snake, random_states

REPO = Path(__file__).resolve().parents[1]
# Released v2 (strict receipt) and v5 (served Watch hero); retired v3/v4; screened v6.
FROZEN = {
    "safety_veto.py": "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    "safety_veto_v3.py": "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
    "safety_veto_v4.py": "3f0881afb7ff8ff980c9b425ef40134f127cc90f1f40260aac39f2264ed44578",
    "safety_veto_v5.py": "d86d084e7778fc514c4932b27f3750f5f11543e3c7afa44571869407870ec86c",
    "safety_veto_v6.py": "a6117cb98383f9f28bf8ebcf22d2ef63d7a03995734a36fb96bb1bf8bb1757d5",
}
SS = 10
LAMBDAS = (0.25, 0.5, 1.0, 10.0)
TIMING = {
    "apply_seconds_total",
    "apply_seconds_max",
    "mean_apply_seconds",
    "area_eval_seconds_total",
    "area_eval_seconds_max",
    "mean_area_eval_seconds_per_evaluation",
}


def deterministic(diagnostics):
    out = {k: v for k, v in diagnostics.items() if k not in TIMING}
    out["v5"] = {k: v for k, v in diagnostics["v5"].items() if k not in TIMING}
    return out


def v5_reference(me, roster, q, mask, base):
    """v5's ``(action, outcome, reason)`` and its eligible set, from the released code."""
    others = list(roster)
    cap, need = free_space_threshold(me.length, me._logical_length())
    spacious = spacious_directions(me._get_free_space_features(others), cap, need)

    def landing_ok(direction):
        count = landing_count(me, others, direction, cap)
        return count is None or count >= need

    qs, ms = [float(x) for x in q], [bool(x) for x in mask]
    action, outcome, reason = boost_aware_choice(qs, ms, spacious, int(base), landing_ok)
    eligible = {a for a in range(6) if ms[a] and spacious[a % 3] and (a < 3 or landing_ok(a % 3))}
    return action, outcome, reason, eligible, (qs, ms, spacious, landing_ok)


def g_table(me, roster, candidates):
    grid = grid_for(me)
    blocked = static_blocked(me, roster, grid)
    cap = area_cap(me.length, grid, blocked)
    return {a: area_score(landing_area(me, a, blocked, grid, cap), cap) for a in candidates}


class TestFrozenSources:
    def test_released_retired_and_screened_veto_bytes_are_unchanged(self):
        for name, sha in FROZEN.items():
            source = REPO / "src/evaluation" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name


# ---------------------------------------------------------------- lambda = 0 is v5


class TestLambdaZeroIsV5:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_apply_equals_v5_exactly_on_random_states(self, setup_config, seed):
        v5, v7 = BoostAwareFreeSpaceVeto(), SpacePreferenceVeto(0.0)
        for me, roster, q, mask, base in random_states(seed, 150):
            assert v7.apply(me, roster, q, mask, base) == v5.apply(me, roster, q, mask, base)
        assert v7.counters.to_dict() == v5.counters.to_dict()
        diag = deterministic(v7.diagnostics_record())
        v5_diag = {k: v for k, v in v5.diagnostics_record().items() if k not in TIMING}
        assert diag["v5"] == v5_diag
        assert diag["rerank_decisions"] == diag["area_evaluations"] == 0

    @pytest.mark.parametrize("seed", [3, 4])
    def test_pure_rerank_at_zero_returns_v5_choice(self, setup_config, seed):
        checked = 0
        for me, roster, q, mask, base in random_states(seed, 150):
            action, outcome, reason, _, (qs, ms, sp, ok) = v5_reference(me, roster, q, mask, base)
            if outcome == "no_spacious":
                continue
            candidates = rerank_candidates(action, reason, ms, sp, ok)
            assert action in candidates
            scores = g_table(me, roster, candidates)
            assert eager_space_preference_choice(qs, candidates, action, 0.0, scores) == action
            assert space_preference_choice(qs, candidates, action, 0.0, scores.get)[0] == action
            checked += len(candidates) >= 2
        assert checked > 20


# ---------------------------------------------------------------- pure rule properties


class TestPureRule:
    @pytest.mark.parametrize("seed", [5, 6])
    def test_lazy_equals_eager_and_skips_only_hopeless_actions(self, setup_config, seed):
        for me, roster, q, mask, base in random_states(seed, 120):
            action, outcome, reason, _, (qs, ms, sp, ok) = v5_reference(me, roster, q, mask, base)
            if outcome == "no_spacious":
                continue
            candidates = rerank_candidates(action, reason, ms, sp, ok)
            scores = g_table(me, roster, candidates)
            for lam in (0.0, *LAMBDAS):
                lazy, used = space_preference_choice(qs, candidates, action, lam, scores.get)
                assert lazy == eager_space_preference_choice(qs, candidates, action, lam, scores)
                assert set(used) <= set(candidates)

    def test_ties_go_to_the_anchor_then_the_lowest_index(self):
        q = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        flat = {0: 0.5, 1: 0.5, 2: 0.5}
        assert space_preference_choice(q, [0, 1, 2], 2, 1.0, flat.get)[0] == 2
        assert eager_space_preference_choice(q, [0, 1, 2], 2, 1.0, flat) == 2
        better = {0: 0.9, 1: 0.9, 2: 0.5}
        assert space_preference_choice(q, [0, 1, 2], 2, 1.0, better.get)[0] == 0
        assert eager_space_preference_choice(q, [0, 1, 2], 2, 1.0, better) == 0

    def test_lambda_is_unit_free(self):
        scores = {0: 0.2, 1: 1.0}
        for scale in (1e-3, 1.0, 1e3):
            q = [scale * 1.0, scale * 0.9, 0.0, 0.0, 0.0, 0.0]
            # Qn(1) = -1 with two candidates: 1 wins iff lam * 0.8 > 1.
            assert space_preference_choice(q, [0, 1], 0, 1.0, scores.get)[0] == 0
            assert space_preference_choice(q, [0, 1], 0, 1.5, scores.get)[0] == 1

    def test_single_candidate_and_invalid_inputs(self):
        q = [0.0] * 6
        assert space_preference_choice(q, [4], 4, 9.0, lambda a: 0.0) == (4, {})
        with pytest.raises(ValueError):
            space_preference_choice(q, [0, 1], 2, 1.0, lambda a: 0.0)
        for bad in (-0.1, math.inf, math.nan, True):
            with pytest.raises(ValueError):
                SpacePreferenceVeto(bad)

    def test_landing_same_direction_and_mode_restriction(self):
        ok = lambda d: d != 1  # noqa: E731
        mask, spacious = [True] * 6, [True] * 3
        assert rerank_candidates(1, REASON_LANDING_SAME_DIRECTION, mask, spacious, ok) == [1]
        assert rerank_candidates(1, "kept", mask, spacious, ok) == [0, 1, 2]
        assert rerank_candidates(3, "kept", mask, spacious, ok) == [3, 5]
        assert rerank_candidates(4, "v2_rule", mask, [True, True, False], ok) == [3, 4]
        assert rerank_candidates(0, "kept", [True, False] + [True] * 4, spacious, ok) == [0, 2]

    def test_method_string(self):
        assert method_for(0.5) == "free-space-veto/v7-space-preference(lambda=0.5)"
        assert method_for(1) == "free-space-veto/v7-space-preference(lambda=1.0)"
        assert area_score(0, 10) == 0.0 and area_score(10, 10) == 1.0


# ---------------------------------------------------------------- constructed regions
#
# Board 80 x 60. The hero (length 100, heading up) has its head at (40, 30) and its body
# straight down x = 40 to the bottom row, right along it and up x = 79. A static other
# snake walls a 12 x 14 pocket (x 28..39, y 30..43) west of the head: north wall y = 29,
# south wall y = 44 and west wall x = 27; its east side is the hero's own body, which
# releases far too late to open it. Left (west) enters the pocket (168 cells, spacious
# for v2/v5 since need = 100); straight (north) and right (east) reach the open board.
POCKET = 12 * 14


def pocket_world():
    width, height = 80, 60
    cells = [(40, y) for y in range(30, height)]
    cells += [(x, height - 1) for x in range(41, width)]
    cells += [(width - 1, y) for y in range(height - 2, 0, -1)]
    me = make_snake(cells[:100], (0, -1), width=width * SS, height=height * SS)
    walls = [(x, 29) for x in range(27, 40)] + [(x, 44) for x in range(27, 40)]
    walls += [(27, y) for y in range(30, 44)]
    other = make_snake(walls, (1, 0), width=width * SS, height=height * SS, sid=1)
    return me, [me, other]


# Q prefers left (into the pocket), right is a close second, straight far behind.
POCKET_Q = [1.0, 0.0, 0.95]


def pocket_q(boost):
    low = [-5.0, -5.0, -5.0]
    q = low + POCKET_Q if boost else POCKET_Q + low
    mask = [not boost] * 3 + [boost] * 3
    return torch.tensor(q), torch.tensor(mask), 3 if boost else 0


class TestConstructedRegions:
    def test_geometry_is_what_the_comment_says(self, setup_config):
        me, roster = pocket_world()
        grid = grid_for(me)
        blocked = static_blocked(me, roster, grid)
        cap = area_cap(me.length, grid, blocked)
        assert cap == 400 == 4 * me.length
        assert landing_area(me, 0, blocked, grid, cap) == POCKET
        assert landing_area(me, 1, blocked, grid, cap) == cap
        assert landing_area(me, 2, blocked, grid, cap) == cap
        assert landing_area(me, 3, blocked, grid, cap) == POCKET - 1  # first cell = neck
        # Static check: the pocket really is closed off when the whole body stays.
        body = {grid.to_cell(x, y) for x, y in me.segments}
        assert tail_aware_reachable((39, 30), 0, blocked | body, {}, grid.in_bounds, 10**6) == (
            POCKET
        )
        cap2, need = free_space_threshold(me.length, me._logical_length())
        assert spacious_directions(me._get_free_space_features(roster), cap2, need) == [True] * 3

    @pytest.mark.parametrize("boost", [False, True])
    def test_larger_lambda_prefers_the_larger_region(self, setup_config, boost):
        me, roster = pocket_world()
        q, mask, base = pocket_q(boost)
        assert BoostAwareFreeSpaceVeto().apply(me, roster, q, mask, base) == base  # v5 keeps
        # Pocket g = log1p(168)/log1p(400) ~ 0.856; right wins iff lam * 0.144 > 0.05.
        for lam, expected in ((0.0, base), (0.25, base), (0.5, base + 2), (1.0, base + 2)):
            veto = SpacePreferenceVeto(lam)
            assert veto.apply(me, roster, q, mask, base) == expected, lam
            counters, diag = veto.counters.to_dict(), veto.diagnostics_record()
            changed = int(expected != base)
            assert counters["vetoes_applied"] == changed == diag["rerank_changes"]
            assert counters["kept_base"] == 1 - changed
            assert diag["rerank_changes_of_v5_kept"] == changed
            assert diag["rerank_changes_boost_mode"] == changed * int(boost)
            assert diag["rerank_decisions"] == int(lam > 0)
            if lam > 0:
                # At 0.25 right's best case (-0.05 + 0.25) cannot beat the pocket's score.
                assert diag["area_evaluations"] == (1 if lam == 0.25 else 2)
                assert diag["area_cap_hits"] == diag["area_evaluations"] - 1
                assert diag["area_cap_max"] == 400

    def test_open_anchor_needs_no_other_area(self, setup_config):
        me, roster = pocket_world()
        veto = SpacePreferenceVeto(1.0)
        q = torch.tensor([0.0, 1.0, 0.95, -5.0, -5.0, -5.0])  # straight first: open board
        mask = torch.tensor([True] * 3 + [False] * 3)
        assert veto.apply(me, roster, q, mask, 1) == 1
        diag = veto.diagnostics_record()
        # Anchor at the cap scores 0 + lam; no alternative can exceed it.
        assert diag["area_evaluations"] == 1 and diag["area_cap_hits"] == 1
        assert diag["rerank_scored"] == 1 and diag["rerank_changes"] == 0

    def test_area_cap_is_bounded(self, setup_config):
        me, roster = pocket_world()
        grid = grid_for(me)
        me.length = 5000
        assert area_cap(me.length, grid, set()) == AREA_CAP_LIMIT
        assert area_cap(me.length, grid, {(x, y) for x in range(80) for y in range(59)}) == 80
        assert area_cap(1, grid, set()) == 32

    def test_move_into_the_vacating_tail_cell_counts(self, setup_config):
        # A 2 x 2 loop: the head at (5, 5) heading up, tail at (4, 5); turning left enters
        # the tail cell, which the tail leaves this frame (v3's pre-move model blocks it).
        me = make_snake([(5, 5), (5, 6), (4, 6), (4, 5)], (0, -1))
        grid = grid_for(me)
        assert landing_area(me, 0, set(), grid, 50) == 50


# ---------------------------------------------------------------- safety properties


class TestSafetyProperties:
    @pytest.mark.parametrize("seed", [9, 10])
    def test_never_masked_never_v5_ineligible_and_deterministic(self, setup_config, seed):
        states = random_states(seed, 200)
        changes = 0
        for lam in LAMBDAS:
            first, second = SpacePreferenceVeto(lam), SpacePreferenceVeto(lam)
            for me, roster, q, mask, base in states:
                v5_action, outcome, reason, eligible, _ = v5_reference(me, roster, q, mask, base)
                action = first.apply(me, roster, q, mask, base)
                assert action == second.apply(me, roster, q, mask, base)
                if outcome == "no_spacious" or reason == REASON_LANDING_SAME_DIRECTION:
                    assert action == v5_action
                    continue
                assert bool(mask[action]) and action in eligible
                assert (action >= 3) == (v5_action >= 3)
                changes += int(action != v5_action)
            assert first.counters.to_dict() == second.counters.to_dict()
            assert deterministic(first.diagnostics_record()) == deterministic(
                second.diagnostics_record()
            )
        assert changes > 0  # the re-rank does act on these states

    def test_counter_identities(self, setup_config):
        veto = SpacePreferenceVeto(10.0)
        for me, roster, q, mask, base in random_states(9, 200):
            veto.apply(me, roster, q, mask, base)
        c, d = veto.counters.to_dict(), veto.diagnostics_record()
        v5 = d["v5"]
        assert c["decisions"] == d["decisions"] == v5["decisions"] == 200
        assert c["kept_base"] == d["kept"] == v5["kept"] - d["rerank_changes_of_v5_kept"]
        assert d["vetoes_applied"] == v5["vetoes_applied"] + d["rerank_changes_of_v5_kept"]
        assert c["vetoes_applied"] == d["vetoes_applied"]
        assert c["fallback_no_spacious"] == d["no_spacious"] == v5["no_spacious"]
        assert d["rerank_decisions"] == d["rerank_pruned"] + d["rerank_scored"]
        assert d["rerank_changes"] <= d["rerank_scored"]
        assert d["area_cap_hits"] <= d["area_evaluations"]
        assert d["rerank_changes"] > 0 and d["mean_apply_seconds"] > 0


# ---------------------------------------------------------------- hook, record, live


class TestHookAndRecord:
    def test_descriptor_and_record_shape(self, setup_config):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        veto = SpacePreferenceVeto(0.5)
        v5 = BoostAwareFreeSpaceVeto().descriptor()
        descriptor = veto.descriptor()
        assert descriptor["method"] == "free-space-veto/v7-space-preference(lambda=0.5)"
        for key in ("free_space_bfs_cap", "free_space_min_cap", "boost_approximation"):
            assert descriptor[key] == v5[key]
        assert descriptor["landing_rule"] == v5["landing_rule"]
        assert descriptor["space_preference_lambda"] == 0.5
        me, roster = pocket_world()
        q, mask, base = pocket_q(False)
        veto.apply(me, roster, q, mask, base)
        record = veto.record()
        assert set(record) == set(descriptor) | {"counters"}
        _validate_candidate_wrapper_probe(record, descriptor)
        assert veto.counters.vetoes_applied == 1
        veto.reset()
        assert veto.counters.decisions == 0 and veto.diagnostics_record()["decisions"] == 0

    def test_install_requires_an_ai_snake(self):
        with pytest.raises(TypeError):
            install_space_preference_veto(object(), 0.5)

    def test_ai_snake_default_off_and_installed_v7(self, setup_config):
        from src.game.ai_snake import AISnake

        class DqnPolicyStub:
            epsilon = 0.0

            def dqn(self, state):
                return torch.tensor([[0.0, 1.0, 0.5, 0.0, 0.2, 0.1]])

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=DqnPolicyStub())
        snake.direction = (1, 0)
        assert snake.safety_veto is None
        veto = install_space_preference_veto(snake, 1.0)
        assert snake.safety_veto is veto
        snake.update([snake], [(100, 100)])
        assert veto.counters.decisions == 1 == veto.diagnostics_record()["decisions"]

    @pytest.mark.parametrize("boost_frames", [0, 2])
    def test_area_uses_the_live_post_move_body(self, setup_config, boost_frames):
        from src.evaluation.safety_veto_v3 import own_body_release

        me, roster = pocket_world()
        me.boost_frames = boost_frames  # 2: a boost burns one tail cell this frame
        grid = grid_for(me)
        blocked = static_blocked(me, roster, grid)
        for action in range(6):
            twin = clone(me)
            twin.direction = GameLogic.relative_to_absolute_direction(twin.direction, action % 3)
            twin.is_boosting = action >= 3 and twin.length >= GameConfig.MIN_BOOST_LENGTH
            twin.move()
            head = grid.to_cell(*twin.segments[0])
            release = own_body_release(twin.segments, twin.length, grid, 1)
            release.pop(head)
            live = tail_aware_reachable(head, 0, blocked, release, grid.in_bounds, 400)
            assert landing_area(me, action, blocked, grid, 400) == live, action


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
def runners_fatal(monkeypatch):
    """Direct ``rollout`` tests may never reach a screen's or the sweep's episode runner."""
    from research.apex_safety_20260926 import dev_screen

    def fatal(*args, **kwargs):
        raise AssertionError("this test must not run screen episodes")

    monkeypatch.setattr(dev_screen, "run_episode", fatal)


class TestLiveRollout:
    def _play(self, tiny_live, lam):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts.tournament_eval import rollout

        def install(hero):
            hero.grow(30)  # long enough to boost and to give the re-rank real choices
            return install_space_preference_veto(hero, lam)

        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        with dev_screen.hero_veto_installer(install) as installed:
            record = rollout(tiny_live, opponents, 80, 3, hero_safety_veto=True)
        return record, installed[-1]

    def test_live_probe_identities_and_replay(self, tiny_live, runners_fatal):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        record, veto = self._play(tiny_live, 10.0)
        probe = record["probes"]["safety_veto"]
        assert probe["method"] == method_for(10.0)
        _validate_candidate_wrapper_probe(probe, veto.descriptor())
        diag = veto.diagnostics_record()
        assert diag["decisions"] == probe["counters"]["decisions"] > 0
        assert diag["rerank_decisions"] > 0
        again, veto_again = self._play(tiny_live, 10.0)
        assert again == record
        assert deterministic(veto_again.diagnostics_record()) == deterministic(diag)
