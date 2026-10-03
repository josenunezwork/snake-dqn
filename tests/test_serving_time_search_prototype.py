"""Tests for the research-only hero rollout prototype (serving-time search feasibility)."""

from __future__ import annotations

import random

import pytest

from research.serving_time_search_20261003.rollout_search import (
    HeroCells,
    RolloutParams,
    RolloutScore,
    hero_cells,
    rollout,
    rollout_choice,
    score_actions,
    step_hero,
    world_model,
)
from src.core.game_config import GameConfig
from src.evaluation.safety_veto_v3 import grid_for
from src.game.game_logic import GameLogic
from tests.test_safety_veto_v5 import SS, live_move, make_snake, random_states, trap_pocket

ALL = [True] * 6


def params(horizon: int = 20) -> RolloutParams:
    return RolloutParams(
        horizon=horizon,
        min_boost_length=int(GameConfig.MIN_BOOST_LENGTH),
        boost_cost_frames=int(GameConfig.BOOST_LENGTH_COST_FRAMES),
    )


def score(passes: bool, frames: int = 20) -> RolloutScore:
    return RolloutScore(passes, passes, frames, 100 if passes else 0, 1)


class TestHeroModel:
    def test_step_hero_matches_live_snake_move(self, setup_config):
        for me, _roster, _q, _mask, _base in random_states(11, 60):
            grid = grid_for(me)
            hero = hero_cells(me, grid)
            for action in range(6):
                boost = action >= 3 and me.length >= GameConfig.MIN_BOOST_LENGTH
                after, traversed = step_hero(hero, action % 3, boost, params())
                live = live_move(me, action)
                assert list(after.body) == [grid.to_cell(x, y) for x, y in live.segments]
                assert list(traversed) == [grid.to_cell(x, y) for x, y in live.last_move_positions]
                assert after.length == live.length
                assert after.boost_frames == live.boost_frames

    def test_one_frame_death_matches_live_collisions_without_threats(self, setup_config):
        for me, roster, _q, _mask, _base in random_states(12, 60):
            grid, blocked, _fatal, need = world_model(me, roster)
            hero = hero_cells(me, grid)
            for action in range(6):
                got = rollout(hero, action, blocked, set(), grid.in_bounds, need, params(1))
                live = live_move(me, action)
                cells = [grid.to_cell(x, y) for x, y in live.last_move_positions]
                dead = (
                    any(not grid.in_bounds(c) for c in cells)
                    or GameLogic.check_self_collision(live)
                    or any(c in blocked for c in cells)
                )
                assert got.frames_survived == (0 if dead else 1)


class TestTrapDetection:
    @pytest.mark.parametrize("pocket", [1, 5, 24])
    def test_boost_into_pocket_fails_and_normal_straight_passes(self, setup_config, pocket):
        me, other = trap_pocket(50, pocket)
        scores = score_actions(me, [me, other], range(6), params(20))
        assert not scores[4].passes  # straight boost lands inside the pocket
        assert scores[1].passes  # straight at normal speed keeps the open board
        assert scores[0].passes or scores[2].passes

    def test_head_threat_on_first_frame_is_fatal(self, setup_config):
        width, height = 40 * SS, 40 * SS
        me = make_snake([(21 - i, 20) for i in range(10)], (1, 0), width=width, height=height)
        # A longer opponent heading down whose next head cell is the hero's straight cell.
        other = make_snake(
            [(22, 19)] + [(22, 19 - i) for i in range(1, 15)],
            (0, 1),
            width=width,
            height=height,
            sid=1,
        )
        scores = score_actions(me, [me, other], [0, 1, 2], params(5))
        assert scores[1].frames_survived == 0
        assert scores[2].frames_survived > 0  # turning away (down) is not in its reach


class TestChoiceRule:
    def test_keeps_a_passing_current_choice(self):
        scores = {a: score(True) for a in range(6)}
        assert rollout_choice([0.0] * 6, ALL, 4, scores) == (4, "kept_passes")

    def test_switches_to_highest_q_passing_same_speed_mode(self):
        q = [0.1, 0.9, 0.5, 0.95, 0.2, 0.3]
        scores = {0: score(True), 1: score(False), 2: score(True), 3: score(True)}
        scores.update({4: score(False), 5: score(False)})
        assert rollout_choice(q, ALL, 1, scores) == (2, "switched")
        # Boost current: only boost 3 passes in its mode.
        assert rollout_choice(q, ALL, 4, scores) == (3, "switched")

    def test_falls_back_to_other_mode_and_never_picks_masked(self):
        q = [0.1, 0.9, 0.5, 0.95, 0.2, 0.3]
        scores = {a: score(a == 3 or a == 0) for a in range(6)}
        mask = [False, True, True, True, True, True]
        assert rollout_choice(q, mask, 1, scores) == (3, "switched")
        assert rollout_choice(q, [False, True, True, False, True, True], 1, scores) == (
            1,
            "kept_no_alternative",
        )


class TestDeterminism:
    def test_repeatable_and_does_not_touch_global_rng(self, setup_config):
        worlds = random_states(13, 25)
        random.seed(99)
        before = random.getstate()
        first = [score_actions(me, roster, range(6), params(15)) for me, roster, *_ in worlds]
        second = [score_actions(me, roster, range(6), params(15)) for me, roster, *_ in worlds]
        assert first == second
        assert random.getstate() == before

    def test_does_not_mutate_the_snakes(self, setup_config):
        for me, roster, *_ in random_states(14, 20):
            snap = [(list(s.segments), tuple(s.direction), s.length) for s in roster]
            score_actions(me, roster, range(6), params(10))
            assert snap == [(list(s.segments), tuple(s.direction), s.length) for s in roster]

    def test_hero_cells_is_a_value_copy(self, setup_config):
        me, _ = trap_pocket(20, 3)
        hero = hero_cells(me, grid_for(me))
        assert isinstance(hero, HeroCells) and hero.body[0] == (60, 40)
