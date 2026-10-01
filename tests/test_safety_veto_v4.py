"""Tests for the opt-in look-ahead free-space veto (src/evaluation/safety_veto_v4.py)."""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import pytest
import torch

from src.evaluation.safety_veto import FreeSpaceVeto, free_space_threshold, spacious_directions
from src.evaluation.safety_veto_v3 import TailAwareFreeSpaceVeto, reachable_counts
from src.evaluation.safety_veto_v4 import (
    DEFAULT_DEPTH,
    DEFAULT_NODE_BUDGET,
    VETO_METHOD_V4,
    EscapeSearch,
    LookaheadFreeSpaceVeto,
    install_lookahead_veto,
    turn,
    world_for,
)
from src.game.game_logic import GameLogic
from src.game.snake import Snake

REPO = Path(__file__).resolve().parents[1]
# The v2 strict receipt binds this exact source sha256; v3 is retired but frozen too.
V2_SOURCE_SHA256 = "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
V3_SOURCE_SHA256 = "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1"
SS = 10
ALL = [True] * 6
Q_PREFERS_STRAIGHT = [0.0, 9.0, 1.0, 0.0, 0.5, 0.2]
TIMING_KEYS = {
    "apply_seconds_total",
    "search_seconds_total",
    "search_seconds_max",
    "mean_apply_seconds",
    "mean_search_seconds",
}


def make_snake(cells_head_first, direction, length=None, width=100, height=100, sid=0):
    """A real Snake on a ``width x height`` px board whose body is ``cells`` (grid units)."""
    head = cells_head_first[0]
    snake = Snake(sid, (255, 0, 0), (head[0] * SS, head[1] * SS), SS, width, height)
    snake.segments = [(x * SS, y * SS) for x, y in cells_head_first]
    snake.length = len(cells_head_first) if length is None else int(length)
    snake.direction = direction
    return snake


def edge_corridor():
    """A 7-cell 1-wide dead end along the top wall, walled by another (static) snake.

    Board 20x20. Own snake (length 5, need 5) heads right at (5,0); straight enters the
    corridor (6..12, 0), whose floor (6..12, 1) and end (13, 0) are the other snake. The
    right turn (5,1) leads into open space; the left turn leaves the board.
    """
    me = make_snake([(5, 0), (4, 0), (3, 0), (2, 0), (1, 0)], (1, 0), width=200, height=200)
    walls = [(13, 0), (13, 1)] + [(x, 1) for x in range(12, 5, -1)]
    other = make_snake(walls, (0, -1), width=200, height=200, sid=1)
    return me, other


SPIRAL = [
    "....#######....",
    "...H......#....",
    "....#####.#....",
    "....#...#.#....",
    "....#.#...#....",
    "....#.#####....",
    "....###........",
    "...............",
    "...............",
    "...............",
    "...............",
    "...............",
]


def spiral():
    """A 16-cell 1-wide spiral dead end walled by another snake (head far away).

    Board 15x12. Own snake (length 12, need 12) heads right at H = (3,1); its body runs
    left then down the x = 0 column. Straight enters the spiral (4..9,1) -> (9,2..4) ->
    (8,4) (7,4) -> (7,3) -> (6,3) (5,3) -> (5,4) (5,5), a dead end. Right turn (3,2) is open.
    """
    own = [(3, 1), (2, 1), (1, 1), (0, 1)] + [(0, y) for y in range(2, 10)]
    me = make_snake(own, (1, 0), width=150, height=120)
    walls = [(x, y) for y, row in enumerate(SPIRAL) for x, c in enumerate(row) if c == "#"]
    walls = [(10, 5)] + [cell for cell in walls if cell != (10, 5)]  # head far from H
    other = make_snake(walls, (0, 1), width=150, height=120, sid=1)
    return me, other


def _alive_after(factory, moves):
    """Replay relative ``moves`` with ``Snake.move``; other snake cells are static walls."""
    snake, other = factory()
    walls = {(x // SS, y // SS) for x, y in other.segments}
    for relative in moves:
        snake.direction = GameLogic.relative_to_absolute_direction(snake.direction, relative)
        snake.move()
        head = (snake.head[0] // SS, snake.head[1] // SS)
        if (
            GameLogic.check_wall_collision(snake)
            or GameLogic.check_self_collision(snake)
            or head in walls
        ):
            return False
    return True


def some_continuation_survives(factory, prefix, horizon):
    """Depth-first: does any move sequence extending ``prefix`` survive ``horizon`` moves?"""
    if not _alive_after(factory, prefix):
        return False
    if len(prefix) >= horizon:
        return True
    return any(some_continuation_survives(factory, prefix + [r], horizon) for r in (1, 0, 2))


def v2_flags(snake, others=()):
    roster = [snake, *others]
    cap, need = free_space_threshold(snake.length, snake._logical_length())
    return spacious_directions(snake._get_free_space_features(roster), cap, need)


def deterministic(diagnostics):
    return {k: v for k, v in diagnostics.items() if k not in TIMING_KEYS}


class TestFrozenSources:
    def test_v2_and_v3_source_bytes_are_unchanged(self):
        for name, sha in (
            ("safety_veto.py", V2_SOURCE_SHA256),
            ("safety_veto_v3.py", V3_SOURCE_SHA256),
        ):
            source = REPO / "src/evaluation" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name

    def test_turn_matches_game_logic(self):
        for heading in ((0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)):
            for relative in range(3):
                expected = GameLogic.relative_to_absolute_direction(heading, relative)
                assert turn(heading, relative) == expected


@pytest.mark.parametrize("factory, corridor", [(edge_corridor, 7), (spiral, 16)])
class TestTrapGeometries:
    def test_v2_and_v3_allow_the_dead_end_and_v4_vetoes_it(self, setup_config, factory, corridor):
        me, other = factory()
        roster = [me, other]
        cap, need = free_space_threshold(me.length, me._logical_length())
        assert need <= corridor < need + DEFAULT_DEPTH  # spacious by count, too short to use
        assert v2_flags(me, [other])[1] is True
        assert reachable_counts(me, roster).one_step[1] == corridor  # v3 agrees with v2
        q, mask = torch.tensor(Q_PREFERS_STRAIGHT), torch.tensor(ALL)
        v2, v3, v4 = FreeSpaceVeto(), TailAwareFreeSpaceVeto(), LookaheadFreeSpaceVeto()
        assert v2.apply(me, roster, q, mask, 1) == 1 and v2.counters.kept_base == 1
        assert v3.apply(me, roster, q, mask, 1) == 1 and v3.counters.kept_base == 1
        assert v4.apply(me, roster, q, mask, 1) == 2  # best escaping action: turn right
        assert v4.counters.vetoes_applied == 1
        diag = v4.diagnostics_record()
        assert diag["searches"] == 1 and diag["vetoes_applied"] == 1
        assert diag["budget_exhausted"] == 0 and diag["nodes_max"] <= DEFAULT_NODE_BUDGET

    def test_the_vetoed_move_is_fatal_and_the_replacement_is_not(
        self, setup_config, factory, corridor
    ):
        # Every continuation of "straight" dies before it could leave the corridor.
        assert not some_continuation_survives(factory, [1], corridor + 2)
        # The replacement (right) has a continuation that survives well past the search.
        assert some_continuation_survives(factory, [2], 2 * DEFAULT_DEPTH)

    def test_boost_base_is_replaced_in_the_same_speed_mode(self, setup_config, factory, corridor):
        me, other = factory()
        veto = LookaheadFreeSpaceVeto()
        assert (
            veto.apply(me, [me, other], torch.tensor(Q_PREFERS_STRAIGHT), torch.tensor(ALL), 4) == 5
        )
        assert veto.counters.vetoes_applied == 1 and veto.counters.vetoes_to_boost == 1
        assert veto.counters.vetoes_speed_switched == 0

    def test_no_escaping_legal_action_falls_back_to_v2(self, setup_config, factory, corridor):
        me, other = factory()
        veto = LookaheadFreeSpaceVeto()
        mask = torch.tensor([False, True, False, False, True, False])  # only straight legal
        assert veto.apply(me, [me, other], torch.tensor(Q_PREFERS_STRAIGHT), mask, 1) == 1
        # Straight is v2-spacious, so the v2 fallback keeps it (as the released v2 does).
        assert veto.counters.kept_base == 1 and veto.counters.fallback_no_spacious == 0
        diag = veto.diagnostics_record()
        assert diag["fallback_no_escape"] == 1 and diag["fallback_v2_kept"] == 1

    def test_budget_fallback_is_v2_and_deterministic(self, setup_config, factory, corridor):
        outcomes = []
        for _ in range(2):
            me, other = factory()
            veto = LookaheadFreeSpaceVeto(node_budget=3)
            q, mask = torch.tensor(Q_PREFERS_STRAIGHT), torch.tensor(ALL)
            outcomes.append((veto.apply(me, [me, other], q, mask, 1), veto.record()))
            diag = veto.diagnostics_record()
            assert diag["budget_exhausted"] == 1 and diag["nodes_max"] == 4
            assert diag["fallback_v2_kept"] == 1  # v2 decides; straight is v2-spacious
            assert veto.counters.kept_base == 1
        assert outcomes[0] == outcomes[1] and outcomes[0][0] == 1
        assert outcomes[0][1]["node_budget"] == 3


def head_beside_turn():
    """Probe A (review 15): v2 vetoes into a turn that v4's escape test rejects.

    Board 20x20, own snake (length 5, need 5) heads right at (5,0) along the top wall.
    Straight is a 3-cell dead end (6..8, 0), closed by another snake whose head (6,1) is
    4-adjacent to the right-turn cell (5,1); left leaves the board. v2: straight count 3
    < need, right is open (spacious) -> veto to right. v4: straight has no escape and
    right is blocked at step 1 (beside another head), so no direction escapes.
    """
    me = make_snake([(5, 0), (4, 0), (3, 0), (2, 0), (1, 0)], (1, 0), width=200, height=200)
    other = make_snake(
        [(6, 1), (7, 1), (8, 1), (9, 1), (9, 0)], (-1, 0), width=200, height=200, sid=1
    )
    return me, other


class TestV2Fallback:
    def test_no_escape_falls_back_to_the_v2_veto(self, setup_config):
        me, other = head_beside_turn()
        roster, q, mask = [me, other], torch.tensor(Q_PREFERS_STRAIGHT), torch.tensor(ALL)
        assert v2_flags(me, [other]) == [False, False, True]
        v2, v4 = FreeSpaceVeto(), LookaheadFreeSpaceVeto()
        assert v2.apply(me, roster, q, mask, 1) == 2 and v2.counters.vetoes_applied == 1
        assert v4.apply(me, roster, q, mask, 1) == 2  # was 1 (the dead end) before the fix
        assert v4.record()["counters"] == v2.record()["counters"]
        diag = v4.diagnostics_record()
        assert diag["fallback_no_escape"] == 1 and diag["fallback_v2_vetoes"] == 1
        assert diag["vetoes_applied"] == 0  # the escape search vetoed nothing itself
        assert not some_continuation_survives(head_beside_turn, [1], 5)  # straight is fatal

    def test_budget_exhaustion_falls_back_to_the_v2_veto(self, setup_config):
        # Same 3-cell dead end, but the other snake's head (9,0) is far from both step-1
        # cells: the search proves straight has no escape (1 node), then runs out of
        # budget on right. Before the fix this kept the proven dead end.
        def short_corridor():
            me = make_snake([(5, 0), (4, 0), (3, 0), (2, 0), (1, 0)], (1, 0), 5, 200, 200)
            walls = [(9, 0), (9, 1), (8, 1), (7, 1), (6, 1)]
            return me, make_snake(walls, (0, -1), width=200, height=200, sid=1)

        me, other = short_corridor()
        roster, q, mask = [me, other], torch.tensor(Q_PREFERS_STRAIGHT), torch.tensor(ALL)
        v2, v4 = FreeSpaceVeto(), LookaheadFreeSpaceVeto(node_budget=1)
        assert v2.apply(me, roster, q, mask, 1) == 2
        assert v4.apply(me, roster, q, mask, 1) == 2
        diag = v4.diagnostics_record()
        assert diag["budget_exhausted"] == 1 and diag["fallback_v2_vetoes"] == 1
        assert diag["nodes_max"] == 2 and v4.record()["counters"] == v2.record()["counters"]
        full = LookaheadFreeSpaceVeto()  # with its full budget v4 vetoes on its own
        assert full.apply(me, roster, q, mask, 1) == 2
        assert full.diagnostics_record()["vetoes_applied"] == 1
        assert not some_continuation_survives(short_corridor, [1], 5)

    def test_untriggered_and_fallback_decisions_equal_v2_in_random_worlds(self, setup_config):
        fallbacks = 0
        for budget in (DEFAULT_NODE_BUDGET, 5):
            for me, roster, q, mask, base in random_worlds(101, 200):
                v2, v4 = FreeSpaceVeto(), LookaheadFreeSpaceVeto(node_budget=budget)
                action = v4.apply(me, roster, q, mask, base)
                diag = v4.diagnostics_record()
                if diag["kept_escape"] or diag["vetoes_applied"]:
                    continue
                fallbacks += diag["budget_exhausted"] + diag["fallback_no_escape"]
                assert action == v2.apply(me, roster, q, mask, base)
                assert v4.record()["counters"] == v2.record()["counters"]
        assert fallbacks > 0


class TestSearchModel:
    def test_other_heads_adjacent_cells_are_blocked_only_at_step_one(self, setup_config):
        me = make_snake([(10, 10), (9, 10), (8, 10)], (1, 0), width=300, height=300)
        # Other head two cells ahead: the straight cell (11,10) is adjacent to it.
        other = make_snake([(12, 10), (13, 10), (14, 10)], (-1, 0), width=300, height=300, sid=1)
        search = EscapeSearch(world_for(me, [me, other]), DEFAULT_DEPTH, DEFAULT_NODE_BUDGET)
        assert search.escape(1) is False
        assert search.escape(0) is True and search.escape(2) is True
        assert search.passable((11, 10), 2) is True  # later steps: only the body blocks

    def test_own_tail_releases_by_step(self, setup_config):
        # Head (1,1) moving left; the tail (1,2) sits just below. Slack 0: the tail cell
        # is free from step 1 (index 0 from the tail), so "left turn" (down) can enter.
        me = make_snake([(1, 1), (2, 1), (2, 2), (1, 2)], (-1, 0), width=300, height=300)
        strict = EscapeSearch(world_for(me, [me], slack=0), 1, 100)
        assert strict.passable((1, 2), 1) is True
        late = EscapeSearch(world_for(me, [me], slack=1), 1, 100)
        assert late.passable((1, 2), 1) is False and late.passable((1, 2), 2) is True

    def test_open_board_never_triggers_and_equals_v2(self, setup_config):
        rng = random.Random(7)
        for _ in range(60):
            length = rng.randrange(3, 25)
            heading = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            hx, hy = rng.randrange(25, 35), rng.randrange(25, 35)
            cells = [(hx - heading[0] * i, hy - heading[1] * i) for i in range(length)]
            me = make_snake(cells, heading, width=600, height=600)
            q = torch.tensor([rng.random() for _ in range(6)])
            mask = torch.tensor([rng.random() < 0.8 for _ in range(6)])
            base = int(torch.argmax(torch.where(mask, q, torch.tensor(-1e9))))
            v2, v4 = FreeSpaceVeto(), LookaheadFreeSpaceVeto()
            assert v4.apply(me, [me], q, mask, base) == v2.apply(me, [me], q, mask, base)
            assert v4.diagnostics_record()["kept_untriggered"] == 1


def _random_walk(rng, width, height, n, occupied):
    for _ in range(200):
        start = (rng.randrange(width), rng.randrange(height))
        if start in occupied:
            continue
        cells, seen = [start], {start}
        while len(cells) < n:
            x, y = cells[-1]
            options = [
                c
                for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                if 0 <= c[0] < width and 0 <= c[1] < height and c not in seen | occupied
            ]
            if not options:
                break
            cells.append(rng.choice(options))
            seen.add(cells[-1])
        if len(cells) == n:
            return cells
    return None


def random_worlds(seed, count):
    """Random coiled heroes (tail-first walks, so the head ends the walk) plus others."""
    rng = random.Random(seed)
    worlds = []
    while len(worlds) < count:
        width, height = rng.choice([(20, 15), (30, 20), (40, 30)])
        occupied, others = set(), []
        for sid in range(1, rng.randrange(1, 4)):
            cells = _random_walk(rng, width, height, rng.choice([8, 30, 60]), occupied)
            if cells:
                occupied |= set(cells)
                others.append(
                    make_snake(cells, (1, 0), width=width * SS, height=height * SS, sid=sid)
                )
        walk = _random_walk(rng, width, height, rng.choice([12, 40, 90, 160]), occupied)
        if walk is None:
            continue
        head_first = list(reversed(walk))
        heading = (head_first[0][0] - head_first[1][0], head_first[0][1] - head_first[1][1])
        me = make_snake(head_first, heading, width=width * SS, height=height * SS)
        q = torch.tensor([rng.uniform(-1, 1) for _ in range(6)])
        mask = torch.tensor([rng.random() < 0.7 for _ in range(6)])
        if not bool(mask.any()):
            continue
        base = int(torch.argmax(torch.where(mask, q, torch.tensor(-1e9))))
        worlds.append((me, [me, *others], q, mask, base))
    return worlds


class TestProperties:
    def test_masked_actions_never_selected_and_untriggered_equals_v2(self, setup_config):
        searched = vetoed = 0
        for me, roster, q, mask, base in random_worlds(11, 150):
            v2, v4 = FreeSpaceVeto(), LookaheadFreeSpaceVeto()
            action = v4.apply(me, roster, q, mask, base)
            assert action == base or bool(mask[action])
            diag = v4.diagnostics_record()
            assert diag["decisions"] == 1
            assert diag["nodes_max"] <= DEFAULT_NODE_BUDGET + 1
            if diag["kept_untriggered"]:
                assert v2.apply(me, roster, q, mask, base) == base == action
            searched += diag["searches"]
            vetoed += diag["vetoes_applied"]
            if diag["vetoes_applied"]:
                assert bool(mask[action]) and action != base
        assert searched > 20 and vetoed > 0  # the random worlds exercise the search

    def test_decisions_are_deterministic(self, setup_config):
        first = [(LookaheadFreeSpaceVeto(), w) for w in random_worlds(23, 60)]
        second = [(LookaheadFreeSpaceVeto(), w) for w in random_worlds(23, 60)]
        for (va, wa), (vb, wb) in zip(first, second):
            assert va.apply(*wa) == vb.apply(*wb)
            assert va.record() == vb.record()
            assert deterministic(va.diagnostics_record()) == deterministic(vb.diagnostics_record())

    def test_bounded_cost_on_a_long_coiled_snake(self, setup_config):
        # 400-segment serpentine on a 40x40 board: every head direction is cramped.
        cells = []
        for row in range(10):
            xs = range(40) if row % 2 == 0 else range(39, -1, -1)
            cells.extend((x, row) for x in xs)
        snake = make_snake(list(reversed(cells)), (0, 1), width=400, height=400)
        veto = LookaheadFreeSpaceVeto()
        mask = torch.tensor(ALL)
        started = time.perf_counter()
        for i in range(30):
            veto.apply(snake, [snake], torch.tensor(Q_PREFERS_STRAIGHT), mask, i % 3)
        assert (time.perf_counter() - started) / 30 < 0.05
        diag = veto.diagnostics_record()
        assert diag["decisions"] == 30 and diag["nodes_max"] <= DEFAULT_NODE_BUDGET

    def test_random_worlds_cost_bound(self, setup_config):
        worst = 0.0
        for world in random_worlds(31, 120):
            veto = LookaheadFreeSpaceVeto()
            started = time.perf_counter()
            veto.apply(*world)
            worst = max(worst, time.perf_counter() - started)
        assert worst < 0.5


class TestHookAndRecord:
    def test_descriptor_and_record_shape(self):
        veto = LookaheadFreeSpaceVeto()
        descriptor = veto.descriptor()
        assert descriptor["method"] == VETO_METHOD_V4 == "free-space-veto/v4-lookahead"
        assert descriptor["replacement_rule"] == FreeSpaceVeto().descriptor()["replacement_rule"]
        assert descriptor["lookahead_depth"] == 8 and descriptor["node_budget"] == 4000
        assert descriptor["trigger_factor"] == 2 and descriptor["tail_release_slack"] == 1
        assert descriptor["budget_fallback"] == "v2-veto" == descriptor["no_escape_fallback"]
        record = veto.record()
        assert set(record) == set(descriptor) | {"counters"}
        assert set(record["counters"]) == set(FreeSpaceVeto().record()["counters"])
        assert LookaheadFreeSpaceVeto(depth=5).descriptor()["lookahead_depth"] == 5
        for bad in ({"depth": 0}, {"node_budget": 0}, {"trigger_factor": 0}, {"slack": -1}):
            with pytest.raises(ValueError):
                LookaheadFreeSpaceVeto(**bad)

    def test_record_passes_the_strict_wrapper_probe_validator(self, setup_config):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        veto = LookaheadFreeSpaceVeto()
        me, other = spiral()
        mask = torch.tensor(ALL)
        for base in (1, 4, 2):
            veto.apply(me, [me, other], torch.tensor(Q_PREFERS_STRAIGHT), mask, base)
        _validate_candidate_wrapper_probe(veto.record(), veto.descriptor())
        diag = veto.diagnostics_record()
        assert diag["decisions"] == 3 == veto.counters.decisions
        parts = ("kept_untriggered", "kept_escape", "vetoes_applied")
        parts += ("budget_exhausted", "fallback_no_escape")
        assert sum(diag[key] for key in parts) == diag["decisions"]
        assert diag["searches"] == diag["decisions"] - diag["kept_untriggered"]
        split = ("fallback_v2_kept", "fallback_v2_vetoes", "fallback_v2_no_spacious")
        assert sum(diag[k] for k in split) == diag["budget_exhausted"] + diag["fallback_no_escape"]
        assert _screen().probe_identities_hold(diag, veto.counters.to_dict())
        veto.reset()
        assert veto.counters.decisions == 0 and veto.diagnostics_record()["decisions"] == 0

    def test_install_requires_an_ai_snake(self):
        with pytest.raises(TypeError):
            install_lookahead_veto(object())

    def test_ai_snake_default_off_and_installed_v4(self, setup_config):
        from src.game.ai_snake import AISnake

        class DqnPolicyStub:
            epsilon = 0.0

            def dqn(self, state):
                return torch.tensor([Q_PREFERS_STRAIGHT])

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=DqnPolicyStub())
        snake.direction = (1, 0)
        assert snake.safety_veto is None
        veto = install_lookahead_veto(snake, depth=4)
        assert snake.safety_veto is veto and veto.depth == 4
        snake.update([snake], [(100, 100)])
        assert veto.counters.decisions == 1
        assert veto.diagnostics_record()["kept_untriggered"] == 1  # open board


@pytest.fixture
def tiny_live(setup_config, tmp_path):
    """Tiny live world plus a random 58-D vector checkpoint hero (as test_safety_veto_v3)."""
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


def _screen():
    from research.apex_veto_v4_screen_20261001 import screen

    return screen


class TestLiveRollout:
    """Direct ``rollout`` calls on a tiny synthetic world (never the screen harness)."""

    def _play(self, tiny_live, install):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts.tournament_eval import rollout

        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        with dev_screen.hero_veto_installer(install) as installed:
            record = rollout(tiny_live, opponents, 60, 3, hero_safety_veto=True)
        return record, installed[-1]

    def test_v4_probe_record_and_counters_from_a_live_rollout(self, tiny_live):
        screen = _screen()
        record, veto = self._play(tiny_live, screen.install_v4)
        probe = record["probes"]["safety_veto"]
        assert probe["method"] == screen.V4_METHOD
        assert (
            screen.check_record(
                record, "B", world_seed=3, mix="scripted", roster_hashes=[], smoke=True
            )
            == []
        )
        assert screen.check_record(
            record, "A", world_seed=3, mix="scripted", roster_hashes=[], smoke=True
        )
        diag = veto.diagnostics_record()
        assert diag["decisions"] == probe["counters"]["decisions"] > 0
        again, veto_again = self._play(tiny_live, screen.install_v4)
        assert again == record  # deterministic replay (the D arm's premise)
        assert deterministic(veto_again.diagnostics_record()) == deterministic(diag)

    def test_forced_searches_in_a_live_rollout_are_deterministic(self, tiny_live, monkeypatch):
        # The default trigger never fires on this tiny world, so force every decision
        # through the escape search (and its v2 fallback) to pin the D arm's premise on the
        # search path itself. Only rollout is real here; the screen runner is fatal.
        from research.apex_safety_20260926 import dev_screen

        def fatal(*args, **kwargs):
            raise AssertionError("this test must not run screen episodes")

        monkeypatch.setattr(dev_screen, "run_episode", fatal)
        monkeypatch.setattr(
            LookaheadFreeSpaceVeto, "trigger_threshold", lambda self, cap, need: int(cap) + 1
        )
        screen = _screen()
        record, veto = self._play(tiny_live, screen.install_v4)
        diag = veto.diagnostics_record()
        counters = record["probes"]["safety_veto"]["counters"]
        assert diag["decisions"] == counters["decisions"] > 0
        assert diag["searches"] == diag["decisions"] and diag["kept_untriggered"] == 0
        assert diag["nodes_total"] > 0 and diag["nodes_max"] <= DEFAULT_NODE_BUDGET + 1
        assert screen.probe_identities_hold(diag, counters)
        assert (
            screen.check_record(
                record, "B", world_seed=3, mix="scripted", roster_hashes=[], smoke=True
            )
            == []
        )
        again, veto_again = self._play(tiny_live, screen.install_v4)
        assert again == record
        assert deterministic(veto_again.diagnostics_record()) == deterministic(diag)

    def test_installer_keeps_the_vector61_guard(self, tiny_live):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts.tournament_eval import rollout

        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        with dev_screen.hero_veto_installer(_screen().install_v4):
            with pytest.raises(ValueError, match="vector61 checkpoint hero"):
                rollout(("scripted", "greedy_food"), opponents, 5, 0, hero_safety_veto=True)


# ---------------------------------------------------------------- Tier-1 screen wrapper

FIXTURES = REPO / "research/apex_veto_v3_screen_20261001/fixtures"
REAL_CANDIDATE = FIXTURES / "final-candidate-frozen-1010319811.json"  # run-v3, v2 veto
V2_COUNTERS = {
    "decisions": 10,
    "kept_base": 8,
    "vetoes_applied": 2,
    "fallback_no_spacious": 0,
    "vetoes_to_boost": 0,
    "vetoed_base_boost": 1,
    "vetoes_speed_switched": 0,
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
    import json

    return json.loads(REAL_CANDIDATE.read_text())


def _entry(arm, mix, seed, mass, record_extra=None):
    record = {
        "mass_integral": mass,
        "survival_fraction": 0.5,
        "probes": {"death_cause": "self", "safety_veto": {"counters": dict(V2_COUNTERS)}},
        "denominators": {"decision_frames": 10},
        **(record_extra or {}),
    }
    return {"arm": arm, "mix": mix, "world_seed": seed, "wall_seconds": 1.0, "record": record}


class TestScreenSpecAndNamespaces:
    def test_worlds_are_fresh_and_cover_every_earlier_bank(self):
        from research.apex_safety_20260926 import dev_screen
        from research.apex_veto_strict_20260927 import strict_run
        from research.apex_veto_v3_screen_20261001 import screen as v3

        screen = _screen()
        strict_domains = {domain for domain, _ in strict_run.NAMESPACES.values()}
        strict_domains |= set(strict_run.RUN_V1_NAMESPACES) | {strict_run.SMOKE_DOMAIN}
        assert strict_domains <= set(screen.EARLIER_DOMAINS)
        assert set(v3.EARLIER_DOMAINS) <= set(screen.EARLIER_DOMAINS) | {v3.SMOKE_DOMAIN}
        for consumed in ("apex-veto-v3-screen-v1", "apex-veto-v3-screen-v2", v3.SMOKE_DOMAIN):
            assert consumed in screen.EARLIER_DOMAINS
        assert screen.DOMAIN == "apex-veto-v4-screen-v1" not in screen.EARLIER_DOMAINS
        seeds = dev_screen.screen_seeds(40, screen.DOMAIN, screen.NAMESPACE)
        assert len(set(seeds)) == 40
        report = dev_screen.disjointness_report(
            seeds, None, screen_domain=screen.DOMAIN, extra_namespaces=screen.SPEC.extra_namespaces
        )
        assert report["disjoint"] is True
        assert len(report["extra_namespaces"]) == len(screen.SPEC.extra_namespaces) >= 18
        smoke = dev_screen.screen_seeds(2, screen.SMOKE_DOMAIN, screen.NAMESPACE)
        assert not set(smoke) & set(seeds)

    def test_specs(self):
        from research.apex_safety_20260926 import dev_screen
        from research.apex_veto_v3_screen_20261001 import screen as v3

        screen = _screen()
        spec = screen.SPEC
        assert spec.arm_vetoes["A"] is spec.arm_vetoes["C"] == dev_screen.BUILTIN_VETO
        assert spec.arm_vetoes["B"] is spec.arm_vetoes["D"] is screen.install_v4
        assert spec.replay_worlds == screen.REPLAY_WORLDS == 4
        assert spec.veto_arms() == ("A", "B")
        assert spec.require_slot_locks and spec.require_ac_power
        assert spec.max_wall_seconds == 4 * 3600
        assert screen.SMOKE_SPEC.domain == screen.SMOKE_DOMAIN != spec.domain
        # Opt-in additions are off for every earlier spec.
        assert dev_screen.DEFAULT_SPEC.replay_worlds == 0 == v3.SPEC.replay_worlds
        assert screen.expected_descriptor("D") == screen.expected_descriptor("B")
        assert screen.expected_descriptor("B")["method"] == screen.V4_METHOD

    def test_intent_fields_bind_protocol_rule_and_cap(self):
        from datetime import datetime, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        deadline = datetime(2026, 10, 2, tzinfo=timezone.utc)
        fields = dev_screen.spec_intent_fields(screen.SPEC, 276, deadline)
        protocol = REPO / "research/apex_veto_v4_screen_20261001/protocol.md"
        assert fields["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
        assert fields["decision_rule"].startswith(dev_screen.decision_rule_text())
        assert "B/D replay control on 4 worlds" in fields["decision_rule"]
        assert fields["compute_cap"]["max_wall_seconds"] == 4 * 3600
        assert fields["owner"] == "Apex safety lane" and "v4" in fields["hypothesis"]
        assert dev_screen.decision_rule_text(0) == dev_screen.decision_rule_text()


class TestScreenSelfCheck:
    def _check(self, entry, arm, record=None):
        return _screen().check_record(
            entry["record"] if record is None else record,
            arm,
            world_seed=entry["world_seed"],
            mix=entry["mix"],
            roster_hashes=entry["roster_member_sha256s"],
            smoke=False,
        )

    def test_real_v2_record_passes_as_a_and_fails_as_b(self):
        entry = _real()
        assert self._check(entry, "A") == [] and self._check(entry, "C") == []
        assert any("arm B" in f for f in self._check(entry, "B"))
        assert any("arm D" in f for f in self._check(entry, "D"))

    def test_a_v4_probe_on_the_real_record_passes_as_b_and_d(self):
        import copy

        entry = _real()
        record = copy.deepcopy(entry["record"])
        counters = record["probes"]["safety_veto"]["counters"]
        record["probes"]["safety_veto"] = {
            **LookaheadFreeSpaceVeto().descriptor(),
            "counters": counters,
        }
        assert self._check(entry, "B", record) == [] and self._check(entry, "D", record) == []
        assert self._check(entry, "A", record)
        record["probes"]["safety_veto"]["lookahead_depth"] = 9
        assert self._check(entry, "B", record)

    def test_entry_rules_and_diagnostics_warnings(self):
        import copy

        screen = _screen()
        real = _real()
        record = copy.deepcopy(real["record"])
        counters = record["probes"]["safety_veto"]["counters"]
        record["probes"]["safety_veto"] = {
            **LookaheadFreeSpaceVeto().descriptor(),
            "counters": counters,
        }
        diag = {
            "decisions": counters["decisions"],
            "kept_untriggered": counters["kept_base"],
            "vetoes_applied": counters["vetoes_applied"],
            "fallback_no_escape": counters["fallback_no_spacious"],
            "fallback_v2_no_spacious": counters["fallback_no_spacious"],
            "searches": 3,
        }
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
            "safety_veto_method": screen.V4_METHOD,
            "veto_diagnostics": diag,
            "record": record,
        }
        assert screen.check_entry(entry, smoke=False) == {"failures": [], "warnings": []}
        assert screen.check_entry(dict(entry, arm="D"), smoke=False)["failures"] == []
        stale = dict(entry, veto_diagnostics=dict(diag, decisions=diag["decisions"] + 1))
        assert screen.check_entry(stale, smoke=False)["warnings"]
        split = dict(diag, vetoes_applied=0, fallback_v2_vetoes=counters["vetoes_applied"])
        assert not screen.check_entry(dict(entry, veto_diagnostics=split), smoke=False)["warnings"]
        moved = dict(diag, kept_untriggered=diag["kept_untriggered"] - 1, fallback_v2_kept=0)
        assert screen.check_entry(dict(entry, veto_diagnostics=moved), smoke=False)["warnings"]
        assert screen.check_entry(dict(entry, arm="A"), smoke=False)["failures"]
        assert screen.check_entry(dict(entry, schema_version="x"), smoke=False)["failures"]
        assert screen.check_entry(dict(entry, arm="Z"), smoke=False)["failures"]


class TestReplayControl:
    SEEDS = [11, 12, 13]

    def _entries(self, d_mass=None, d_worlds=2):
        entries = []
        for mix in ("frozen", "scripted", "mixed"):
            for index, seed in enumerate(self.SEEDS):
                entries.append(_entry("A", mix, seed, 30.0 + index))
                entries.append(_entry("B", mix, seed, 31.0 + index))
            entries.append(_entry("C", mix, self.SEEDS[0], 30.0))
            for index, seed in enumerate(self.SEEDS[:d_worlds]):
                mass = 31.0 + index if d_mass is None else d_mass
                entries.append(_entry("D", mix, seed, mass))
        return entries

    def _summarize(self, entries, replay=2):
        from research.apex_safety_20260926.dev_screen import summarize

        return summarize(entries, self.SEEDS, 1, preregistered=(3, 1), replay_worlds=replay)

    def test_identical_replays_pass_and_are_reported(self):
        summary = self._summarize(self._entries())
        assert summary["replay_control"]["passes"] is True
        assert summary["replay_control"]["compared"] == 6 == summary["replay_control"]["planned"]
        assert summary["decision"] in {"RECOMMEND_STRICT_GATE", "NOT_ADVANCED"}
        assert summary["wall_seconds"]["D"]["episodes"] == 6
        assert "B/D replay" in summary["decision_rule"]

    def test_a_differing_replay_is_nondeterministic_and_a_missing_one_incomplete(self):
        assert self._summarize(self._entries(d_mass=0.0))["decision"] == "INVALID_NONDETERMINISTIC"
        assert self._summarize(self._entries(d_worlds=1))["decision"] == "INCOMPLETE"

    def test_default_summary_has_no_replay_block(self):
        from research.apex_safety_20260926.dev_screen import decision_rule_text, summarize

        entries = [e for e in self._entries() if e["arm"] != "D"]
        summary = summarize(entries, self.SEEDS, 1, preregistered=(3, 1))
        assert "replay_control" not in summary and "D" not in summary["wall_seconds"]
        assert summary["decision_rule"] == decision_rule_text()

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
        import json

        summary = json.loads((out / "summary.json").read_text())
        assert summary["replay_control"]["passes"] is True
        assert summary["planned_episodes"] == 30


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
        late = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        argv = ["--out", str(tmp_path / "o"), "--deadline-utc", late]
        assert dev_screen.main(argv, spec=screen.SPEC) == 2
        assert not (tmp_path / "o").exists()

    def test_spec_with_mismatched_replay_arm_is_refused(self, no_episodes, tmp_path):
        import dataclasses
        from datetime import datetime, timedelta, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        arms = dict(screen.SPEC.arm_vetoes, D=dev_screen.BUILTIN_VETO)
        bad = dataclasses.replace(screen.SPEC, arm_vetoes=arms)
        soon = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        argv = ["--out", str(tmp_path / "o"), "--deadline-utc", soon]
        assert dev_screen.main(argv, spec=bad) == 2
        assert not (tmp_path / "o").exists()


class TestCostReport:
    def test_per_decision_cost_and_totals(self):
        screen = _screen()
        diag = {key: 0 for key in screen.SUMMED_V4_FIELDS}
        diag.update(decisions=100, searches=10, vetoes_applied=2, nodes_total=50, nodes_max=9)
        diag.update(apply_seconds_total=0.5, search_seconds_total=0.4, search_seconds_max=0.1)
        entries = [
            dict(_entry("B", "frozen", 1, 1.0), veto_diagnostics=dict(diag), wall_seconds=4.0),
            dict(_entry("B", "mixed", 1, 1.0), veto_diagnostics=dict(diag), wall_seconds=6.0),
            dict(_entry("B", "mixed", 2, 1.0), veto_diagnostics=None, wall_seconds=6.0),
            dict(_entry("A", "mixed", 1, 1.0), wall_seconds=2.0),
        ]
        report = screen.v4_cost_report(entries)
        total = report["total"]
        assert total["episodes"] == 3 and total["missing_diagnostics"] == 1
        assert total["decisions"] == 200 and total["searches"] == 20
        assert total["mean_apply_seconds_per_decision"] == pytest.approx(0.005)
        assert total["mean_search_seconds_per_search"] == pytest.approx(0.04)
        assert total["nodes_max"] == 9 and total["mean_nodes_per_search"] == 5
        assert report["per_mix"]["mixed"]["mean_episode_wall_seconds_arm_A"] == 2.0
        assert report["gating"] is False and screen.v4_cost_report([])["total"] == {}
