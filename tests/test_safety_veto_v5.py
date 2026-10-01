"""Tests for the opt-in boost-aware free-space veto (src/evaluation/safety_veto_v5.py)."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import deque
from pathlib import Path

import pytest
import torch

from src.core.game_config import GameConfig
from src.evaluation.safety_veto import FreeSpaceVeto, free_space_threshold, spacious_directions
from src.evaluation.safety_veto_v5 import (
    VETO_METHOD_V5,
    BoostAwareFreeSpaceVeto,
    boost_aware_choice,
    eager_boost_aware_choice,
    install_boost_aware_veto,
    landing_count,
    simulate_action,
)
from src.game.game_logic import GameLogic
from src.game.snake import Snake

REPO = Path(__file__).resolve().parents[1]
# The v2 strict receipt binds this exact source sha256; v3 and v4 are retired but frozen.
FROZEN = {
    "safety_veto.py": "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428",
    "safety_veto_v3.py": "ed3a6d860b09afd982bc6c87ea0a86566455dfcf9562133d1772e595b4bb5be1",
    "safety_veto_v4.py": "3f0881afb7ff8ff980c9b425ef40134f127cc90f1f40260aac39f2264ed44578",
}
SS = 10
ALL = [True] * 6
Q_PREFERS_BOOST_STRAIGHT = [0.0, 1.0, 0.5, 0.0, 9.0, 0.2]
TIMING_KEYS = {"apply_seconds_total", "apply_seconds_max", "mean_apply_seconds"}


def make_snake(cells, direction, length=None, width=200, height=200, sid=0):
    """A real Snake whose body is ``cells`` (grid units, head first)."""
    head = cells[0]
    snake = Snake(sid, (255, 0, 0), (head[0] * SS, head[1] * SS), SS, width, height)
    snake.segments = [(x * SS, y * SS) for x, y in cells]
    snake.length = len(cells) if length is None else int(length)
    snake.direction = direction
    return snake


def body_path(head, n, height):
    """``n`` cells: left from ``head`` to x = 0, down x = 0, then right along the bottom."""
    hx, hy = head
    cells = [(x, hy) for x in range(hx, -1, -1)]
    cells += [(0, y) for y in range(hy + 1, height)]
    cells += [(x, height - 1) for x in range(1, 10_000)]
    return cells[:n]


def trap_pocket(length, pocket, board=(100, 80)):
    """The trap-horizon pattern: first cell open, two-cell landing in a ``pocket``-cell pocket.

    The hero (``length`` cells) heads right at (60, 40). Another (static) snake walls a
    1-wide corridor (62 .. 61 + pocket, 40) above, below and at its far end. Straight at
    normal speed lands on (61, 40) with the open board around it; a straight boost lands
    on (62, 40) with (61, 40) behind it as the body, inside the pocket.
    """
    width, height = board
    head = (60, 40)
    me = make_snake(body_path(head, length, height), (1, 0), width=width * SS, height=height * SS)
    assert me.length == length
    walls = [(62 + i, 39) for i in range(pocket)] + [(62 + i, 41) for i in range(pocket)]
    walls.append((62 + pocket, 40))
    other = make_snake(walls, (1, 0), width=width * SS, height=height * SS, sid=1)
    return me, other


def v2_flags(snake, roster):
    cap, need = free_space_threshold(snake.length, snake._logical_length())
    return spacious_directions(snake._get_free_space_features(list(roster)), cap, need)


def deterministic(diagnostics):
    return {k: v for k, v in diagnostics.items() if k not in TIMING_KEYS}


def alive_after(factory, actions):
    """Replay ``actions`` (0..5) with the live mechanics; the other snake is a static wall."""
    snake, other = factory()
    walls = {(x // SS, y // SS) for x, y in other.segments}
    for action in actions:
        snake.direction = GameLogic.relative_to_absolute_direction(snake.direction, action % 3)
        snake.is_boosting = action >= 3 and snake.length >= GameConfig.MIN_BOOST_LENGTH
        snake.move()
        cells = [(x // SS, y // SS) for x, y in snake.last_move_positions]
        if (
            GameLogic.check_wall_collision(snake)
            or GameLogic.check_self_collision(snake)
            or any(cell in walls for cell in cells)
        ):
            return False
    return True


def survives(factory, prefix, horizon):
    """Depth-first over normal-speed moves: does any extension of ``prefix`` survive?"""
    if not alive_after(factory, prefix):
        return False
    if len(prefix) >= horizon:
        return True
    return any(survives(factory, prefix + [r], horizon) for r in (1, 0, 2))


class TestFrozenSources:
    def test_v2_v3_v4_source_bytes_are_unchanged(self):
        for name, sha in FROZEN.items():
            source = REPO / "src/evaluation" / name
            assert hashlib.sha256(source.read_bytes()).hexdigest() == sha, name


@pytest.mark.parametrize("length, pocket", [(50, 1), (50, 5), (50, 24), (160, 1), (160, 24)])
class TestTrapHorizonPattern:
    def test_v2_keeps_the_boost_and_v5_takes_the_same_direction_at_normal_speed(
        self, setup_config, length, pocket
    ):
        me, other = trap_pocket(length, pocket)
        roster = [me, other]
        cap, need = free_space_threshold(me.length, me._logical_length())
        assert need == min(length, 160) and pocket < need
        assert v2_flags(me, roster) == [True, True, True]  # first cells all open
        assert landing_count(me, roster, 1, cap) == pocket  # the two-cell landing
        assert landing_count(me, roster, 0, cap) >= need  # turning boosts land in the open
        q, mask = torch.tensor(Q_PREFERS_BOOST_STRAIGHT), torch.tensor(ALL)
        v2, v5 = FreeSpaceVeto(), BoostAwareFreeSpaceVeto()
        assert v2.apply(me, roster, q, mask, 4) == 4 and v2.counters.kept_base == 1
        assert v5.apply(me, roster, q, mask, 4) == 1
        counters = v5.counters.to_dict()
        assert counters["vetoes_applied"] == 1 and counters["vetoed_base_boost"] == 1
        assert counters["vetoes_speed_switched"] == 1 and counters["vetoes_to_boost"] == 0
        diag = v5.diagnostics_record()
        assert diag["boost_landing_vetoes"] == 1 == diag["boost_to_normal_same_direction"]
        assert diag["landing_checks"] == 1 == diag["landing_failures"]
        assert diag["action_differs_from_v2"] == 1

    def test_masked_normal_straight_falls_back_to_v2_rule_over_eligible(
        self, setup_config, length, pocket
    ):
        me, other = trap_pocket(length, pocket)
        mask = torch.tensor([True, False, True, True, True, True])
        veto = BoostAwareFreeSpaceVeto()
        q = torch.tensor(Q_PREFERS_BOOST_STRAIGHT)
        # Same speed mode first: the best eligible boost (right, 0.2 > left, 0.0).
        assert veto.apply(me, [me, other], q, mask, 4) == 5
        diag = veto.diagnostics_record()
        assert diag["boost_landing_v2_rule"] == 1 and diag["boost_to_normal_same_direction"] == 0
        assert veto.counters.vetoes_to_boost == 1 and veto.counters.vetoes_speed_switched == 0

    def test_nothing_eligible_leaves_the_base_unchanged(self, setup_config, length, pocket):
        me, other = trap_pocket(length, pocket)
        mask = torch.tensor([False, False, False, False, True, False])
        veto = BoostAwareFreeSpaceVeto()
        assert veto.apply(me, [me, other], torch.tensor(Q_PREFERS_BOOST_STRAIGHT), mask, 4) == 4
        assert veto.counters.fallback_no_spacious == 1
        diag = veto.diagnostics_record()
        assert diag["boost_landing_no_eligible"] == 1 == diag["base_landing_failed"]
        assert diag["boost_landing_vetoes"] == 0


class TestTrapIsReal:
    """On a short hero the vetoed boost is fatal and the replacement is not."""

    @staticmethod
    def factory():
        return _short_trap()

    def test_boost_dies_next_frame_and_normal_speed_survives(self, setup_config):
        assert alive_after(self.factory, [4])  # the boost itself is legal
        assert not any(alive_after(self.factory, [4, r]) for r in range(6))
        assert survives(self.factory, [1], 12)
        me, other = self.factory()
        q, mask = torch.tensor(Q_PREFERS_BOOST_STRAIGHT), torch.tensor(ALL)
        assert FreeSpaceVeto().apply(me, [me, other], q, mask, 4) == 4
        assert BoostAwareFreeSpaceVeto().apply(me, [me, other], q, mask, 4) == 1


def _short_trap():
    """Length-6 hero (need 6) heading right at (5, 10) on 20x20; a 1-cell pocket at (7, 10)."""
    me = make_snake([(5 - i, 10) for i in range(6)], (1, 0))
    other = make_snake([(8, 10), (7, 9), (7, 11)], (0, 1), sid=1)
    return me, other


# ---------------------------------------------------------------- random states


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


def random_states(seed, count):
    """Random coiled heroes plus other snakes, Q-values, masks and boost counters.

    Heroes are tail-first walks (the head ends the walk); some carry pending growth
    (``length > len(segments)``) and every ``boost_frames`` value, so burns happen.
    """
    rng = random.Random(seed)
    worlds = []
    while len(worlds) < count:
        width, height = rng.choice([(12, 10), (20, 15), (30, 20)])
        occupied, others = set(), []
        for sid in range(1, rng.randrange(1, 4)):
            cells = _random_walk(rng, width, height, rng.choice([4, 12, 30]), occupied)
            if cells:
                occupied |= set(cells)
                others.append(make_snake(cells, (1, 0), width=width * SS, height=height * SS))
                others[-1].id = sid
        walk = _random_walk(rng, width, height, rng.choice([3, 4, 5, 6, 12, 40, 90]), occupied)
        if walk is None:
            continue
        head_first = list(reversed(walk))
        heading = (head_first[0][0] - head_first[1][0], head_first[0][1] - head_first[1][1])
        length = len(head_first) + rng.choice([0, 0, 0, 1, 2])
        me = make_snake(head_first, heading, length, width=width * SS, height=height * SS)
        me.boost_frames = rng.randrange(GameConfig.BOOST_LENGTH_COST_FRAMES)
        q = torch.tensor([rng.uniform(-1, 1) for _ in range(6)])
        mask = torch.tensor([rng.random() < 0.75 for _ in range(6)])
        if not bool(mask.any()):
            continue
        base = int(torch.argmax(torch.where(mask, q, torch.tensor(-1e9))))
        worlds.append((me, [me, *others], q, mask, base))
    return worlds


def clone(snake):
    twin = make_snake([(0, 0)], tuple(snake.direction), width=snake.game_width)
    twin.game_height = snake.game_height
    twin.segments = list(snake.segments)
    twin.length = int(snake.length)
    twin.boost_frames = int(snake.boost_frames)
    return twin


def live_move(snake, action):
    """``AISnake.update``'s decode (ai_snake.py:512-525) on a copy, then ``Snake.move``."""
    twin = clone(snake)
    twin.direction = GameLogic.relative_to_absolute_direction(twin.direction, action % 3)
    twin.is_boosting = action >= 3 and twin.length >= GameConfig.MIN_BOOST_LENGTH
    twin.move()
    return twin


def brute_landing(snake, roster, direction, cap):
    """Independent count: BFS from the live post-boost head, body behind it and others blocked."""
    moved = live_move(snake, 3 + direction)
    to_cell = lambda p: (int(p[0] // SS), int(p[1] // SS))  # noqa: E731
    blocked = {to_cell(p) for p in moved.segments[1:]}
    for other in roster:
        if other is not snake and other.is_alive:
            blocked |= {to_cell(p) for p in other.segments}
    width, height = snake.game_width // SS, snake.game_height // SS
    start = to_cell(moved.segments[0])
    inside = lambda c: 0 <= c[0] < width and 0 <= c[1] < height  # noqa: E731
    if start in blocked or not inside(start):
        return 0
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb not in seen and nb not in blocked and inside(nb):
                seen.add(nb)
                queue.append(nb)
    return min(len(seen), cap)


class TestMovementModel:
    def test_simulated_boost_matches_snake_move_on_random_live_states(self, setup_config):
        burns = boosts = blocked_boosts = 0
        for me, _roster, _q, _mask, _base in random_states(5, 200):
            for action in range(6):
                twin = live_move(me, action)
                sim = simulate_action(me, action)
                assert list(sim.segments) == twin.segments
                assert list(sim.traversed) == twin.last_move_positions
                assert sim.length == twin.length and sim.boost_frames == twin.boost_frames
                pellets = twin.pending_trail_pellets  # v2 mechanics: the burned tail
                assert pellets in ([], [sim.burned_tail]) and (pellets == [] or sim.burned_tail)
                assert sim.boosted == (len(twin.last_move_positions) == 2)
                if action >= 3:
                    boosts += sim.boosted
                    blocked_boosts += not sim.boosted
                    burns += sim.burned_tail is not None
                    if sim.boosted:  # the cell between becomes the body behind the head
                        assert sim.segments[1] == sim.traversed[0]
        assert boosts > 100 and burns > 20 and blocked_boosts > 20

    def test_ai_snake_update_lands_where_the_model_says(self, setup_config):
        from src.game.ai_snake import AISnake

        checked = set()
        for index, (me, roster, q, _mask, _base) in enumerate(random_states(9, 80)):

            class Stub:
                epsilon = 0.0

                def dqn(self, state, _q=q):
                    return _q.unsqueeze(0)

            hero = AISnake(
                0, (255, 0, 0), me.segments[0], SS, me.game_width, me.game_height, policy=Stub()
            )
            hero.segments, hero.length = list(me.segments), me.length
            hero.direction, hero.boost_frames = me.direction, me.boost_frames
            before = clone(hero)
            hero.update([hero, *roster[1:]], [])
            action = int(hero._pre_collision_action)
            sim = simulate_action(before, action)
            assert list(sim.segments) == hero.segments and sim.length == hero.length
            assert sim.boost_frames == hero.boost_frames and sim.boosted == hero.is_boosting
            checked.add(action >= 3)
        assert checked == {True, False}

    def test_landing_count_matches_an_independent_bfs(self, setup_config):
        compared = 0
        for me, roster, _q, _mask, _base in random_states(13, 150):
            cap, _need = free_space_threshold(me.length, me._logical_length())
            for direction in range(3):
                count = landing_count(me, roster, direction, cap)
                if me.length < GameConfig.MIN_BOOST_LENGTH:
                    assert count is None
                    continue
                assert count == brute_landing(me, roster, direction, cap)
                compared += 1
        assert compared > 200


def landing_table(me, roster):
    cap, need = free_space_threshold(me.length, me._logical_length())
    counts = [landing_count(me, roster, d, cap) for d in range(3)]
    return [count is None or count >= need for count in counts]


class TestRuleProperties:
    def test_lazy_rule_equals_the_eager_reference(self, setup_config):
        differ = landing = 0
        for me, roster, q, mask, base in random_states(17, 400):
            flags = v2_flags(me, roster)
            table = landing_table(me, roster)
            calls = []

            def landing_ok(d, table=table, calls=calls):
                calls.append(d)
                return table[d]

            qs, ms = q.tolist(), mask.tolist()
            action, outcome, reason = boost_aware_choice(qs, ms, flags, base, landing_ok)
            assert (action, outcome) == eager_boost_aware_choice(qs, ms, flags, base, table)
            assert all(ms[3 + d] and flags[d] for d in calls)  # floods only where it matters
            veto = BoostAwareFreeSpaceVeto()
            assert veto.apply(me, roster, q, mask, base) == action
            differ += veto.diagnostics_record()["action_differs_from_v2"]
            landing += reason.startswith("landing")
        assert differ > 10 and landing > 5  # the random states exercise the landing path

    def test_masked_actions_are_never_chosen(self, setup_config):
        for me, roster, q, mask, base in random_states(19, 300):
            veto = BoostAwareFreeSpaceVeto()
            action = veto.apply(me, roster, q, mask, base)
            assert bool(mask[action]) or (action == base and veto.counters.vetoes_applied == 0)
            if veto.counters.vetoes_applied:
                assert bool(mask[action]) and action != base
            if action >= 3 and action != base:  # a chosen boost replacement lands spaciously
                assert landing_table(me, roster)[action % 3]

    def test_non_boost_decisions_are_identical_to_v2(self, setup_config):
        compared = exact = 0
        for me, roster, q, mask, base in random_states(29, 400):
            v2_action = FreeSpaceVeto().apply(me, roster, q, mask, base)
            v5_action = BoostAwareFreeSpaceVeto().apply(me, roster, q, mask, base)
            no_boosts = mask.clone()
            no_boosts[3:] = False
            if base < 3:
                compared += 1
                if v2_action < 3 or landing_table(me, roster)[v2_action % 3]:
                    assert v5_action == v2_action
                else:  # v2's fallback is a landing-failing boost: v5 drops only that
                    assert v5_action != v2_action
            if base < 3 and bool(no_boosts.any()):
                nb_base = base
                v2_nb = FreeSpaceVeto().apply(me, roster, q, no_boosts, nb_base)
                assert BoostAwareFreeSpaceVeto().apply(me, roster, q, no_boosts, nb_base) == v2_nb
                exact += 1
        assert compared > 100 and exact > 100

    def test_boost_with_spacious_landing_is_kept_like_v2(self, setup_config):
        me = make_snake([(10 - i, 10) for i in range(8)], (1, 0))  # open 20x20 board
        q, mask = torch.tensor(Q_PREFERS_BOOST_STRAIGHT), torch.tensor(ALL)
        veto = BoostAwareFreeSpaceVeto()
        assert veto.apply(me, [me], q, mask, 4) == 4 == FreeSpaceVeto().apply(me, [me], q, mask, 4)
        diag = veto.diagnostics_record()
        assert diag["kept"] == 1 and diag["landing_checks"] == 1 and diag["landing_failures"] == 0

    def test_decisions_are_deterministic(self, setup_config):
        first = [(BoostAwareFreeSpaceVeto(), w) for w in random_states(23, 120)]
        second = [(BoostAwareFreeSpaceVeto(), w) for w in random_states(23, 120)]
        for (va, wa), (vb, wb) in zip(first, second):
            assert va.apply(*wa) == vb.apply(*wb)
            assert va.record() == vb.record()
            assert deterministic(va.diagnostics_record()) == deterministic(vb.diagnostics_record())

    def test_counter_identities_hold_over_an_episode_of_decisions(self, setup_config):
        veto = BoostAwareFreeSpaceVeto()
        for world in random_states(31, 200):
            me, roster, q, mask, base = world
            veto.apply(me, roster, q, mask, base)
        diag, counters = veto.diagnostics_record(), veto.counters.to_dict()
        assert diag["decisions"] == counters["decisions"] == 200
        assert diag["kept"] + diag["vetoes_applied"] + diag["no_spacious"] == 200
        assert diag["base_landing_failed"] == (
            diag["boost_landing_vetoes"] + diag["boost_landing_no_eligible"]
        )
        assert diag["boost_landing_vetoes"] == (
            diag["boost_to_normal_same_direction"] + diag["boost_landing_v2_rule"]
        )
        assert _screen().probe_identities_hold(diag, counters)


class TestHookAndRecord:
    def test_descriptor_and_record_shape(self):
        veto = BoostAwareFreeSpaceVeto()
        descriptor = veto.descriptor()
        assert descriptor["method"] == VETO_METHOD_V5 == "free-space-veto/v5-boost-aware"
        v2 = FreeSpaceVeto().descriptor()
        for key in ("free_space_bfs_cap", "free_space_min_cap"):
            assert descriptor[key] == v2[key]
        assert descriptor["boost_approximation"] != v2["boost_approximation"]
        record = veto.record()
        assert set(record) == set(descriptor) | {"counters"}
        assert set(record["counters"]) == set(FreeSpaceVeto().record()["counters"])

    def test_record_passes_the_strict_wrapper_probe_validator(self, setup_config):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        veto = BoostAwareFreeSpaceVeto()
        me, other = trap_pocket(50, 5)
        for base, mask in ((4, ALL), (1, ALL), (4, [False] * 4 + [True, False])):
            q = torch.tensor(Q_PREFERS_BOOST_STRAIGHT)
            veto.apply(me, [me, other], q, torch.tensor(mask), base)
        _validate_candidate_wrapper_probe(veto.record(), veto.descriptor())
        diag = veto.diagnostics_record()
        assert diag["decisions"] == 3 and diag["base_landing_failed"] == 2
        assert _screen().probe_identities_hold(diag, veto.counters.to_dict())
        veto.reset()
        assert veto.counters.decisions == 0 and veto.diagnostics_record()["decisions"] == 0

    def test_install_requires_an_ai_snake(self):
        with pytest.raises(TypeError):
            install_boost_aware_veto(object())

    def test_ai_snake_default_off_and_installed_v5(self, setup_config):
        from src.game.ai_snake import AISnake

        class DqnPolicyStub:
            epsilon = 0.0

            def dqn(self, state):
                return torch.tensor([Q_PREFERS_BOOST_STRAIGHT])

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=DqnPolicyStub())
        snake.direction = (1, 0)
        assert snake.safety_veto is None
        veto = install_boost_aware_veto(snake)
        assert snake.safety_veto is veto
        snake.update([snake], [(100, 100)])
        assert veto.counters.decisions == 1 == veto.diagnostics_record()["decisions"]


@pytest.fixture
def tiny_live(setup_config, tmp_path):
    """Tiny live world plus a random 58-D vector checkpoint hero (as test_safety_veto_v4)."""
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
    from research.apex_veto_v5_screen_20261001 import screen

    return screen


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

    def _check_b(self, record):
        return _screen().check_record(
            record, "B", world_seed=3, mix="scripted", roster_hashes=[], smoke=True
        )

    def test_v5_probe_record_and_counters_from_a_live_rollout(self, tiny_live, screen_runner_fatal):
        screen = _screen()
        record, veto = self._play(tiny_live, screen.install_v5)
        probe = record["probes"]["safety_veto"]
        assert probe["method"] == screen.V5_METHOD and self._check_b(record) == []
        assert screen.check_record(
            record, "A", world_seed=3, mix="scripted", roster_hashes=[], smoke=True
        )
        diag = veto.diagnostics_record()
        assert diag["decisions"] == probe["counters"]["decisions"] > 0
        assert screen.probe_identities_hold(diag, probe["counters"])
        again, veto_again = self._play(tiny_live, screen.install_v5)
        assert again == record  # deterministic replay (the D arm's premise)
        assert deterministic(veto_again.diagnostics_record()) == deterministic(diag)

    def test_forced_landing_failures_in_a_live_rollout_are_deterministic(
        self, tiny_live, screen_runner_fatal, monkeypatch
    ):
        # Short live episodes rarely meet a real pocket, so every landing is forced to fail
        # to pin the landing-veto path itself (counters, self-check, replay identity).
        from src.evaluation import safety_veto_v5

        monkeypatch.setattr(safety_veto_v5, "landing_count", lambda *a, **k: 0)
        screen = _screen()

        def long_hero_v5(hero):  # a length-1 hero cannot boost (the mask forbids it)
            hero.grow(11)
            return screen.install_v5(hero)

        record, veto = self._play(tiny_live, long_hero_v5)
        diag = veto.diagnostics_record()
        counters = record["probes"]["safety_veto"]["counters"]
        assert diag["decisions"] == counters["decisions"] > 0
        assert diag["base_landing_failed"] > 0 and diag["boost_landing_vetoes"] > 0
        assert screen.probe_identities_hold(diag, counters) and self._check_b(record) == []
        again, veto_again = self._play(tiny_live, long_hero_v5)
        assert again == record
        assert deterministic(veto_again.diagnostics_record()) == deterministic(diag)

    def test_installer_keeps_the_vector61_guard(self, tiny_live, screen_runner_fatal):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts.tournament_eval import rollout

        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        with dev_screen.hero_veto_installer(_screen().install_v5):
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
        "probes": {"death_cause": "self", "safety_veto": {"counters": dict(V2_COUNTERS)}},
        "denominators": {"decision_frames": 10},
    }
    return {"arm": arm, "mix": mix, "world_seed": seed, "wall_seconds": 1.0, "record": record}


def _diag(counters, **extra):
    diag = {
        "decisions": counters["decisions"],
        "kept": counters["kept_base"],
        "vetoes_applied": counters["vetoes_applied"],
        "no_spacious": counters["fallback_no_spacious"],
        "boost_landing_vetoes": 0,
        "boost_to_normal_same_direction": 0,
    }
    diag.update(extra)
    return diag


class TestScreenSpecAndNamespaces:
    def test_worlds_are_fresh_and_cover_every_earlier_bank(self):
        from research.apex_safety_20260926 import dev_screen
        from research.apex_veto_strict_20260927 import strict_run
        from research.apex_veto_v4_screen_20261001 import screen as v4
        from research.trap_horizon_20261001 import diagnose

        screen = _screen()
        strict_domains = {domain for domain, _ in strict_run.NAMESPACES.values()}
        strict_domains |= set(strict_run.RUN_V1_NAMESPACES) | {strict_run.SMOKE_DOMAIN}
        assert strict_domains <= set(screen.EARLIER_DOMAINS)
        assert set(v4.EARLIER_DOMAINS) <= set(screen.EARLIER_DOMAINS) | {v4.SMOKE_DOMAIN}
        for consumed in (
            "apex-veto-v3-screen-v1",
            "apex-veto-v3-screen-v2",
            "apex-veto-v3-screen-smoke-v1",
            v4.DOMAIN,
            v4.SMOKE_DOMAIN,
            diagnose.DOMAIN,
            diagnose.SMOKE_DOMAIN,
            dev_screen.SCREEN_DOMAIN,  # also the SIMD H5000 parity check's worlds
        ):
            assert consumed in screen.EARLIER_DOMAINS
        assert screen.DOMAIN == "apex-veto-v5-screen-v1" not in screen.EARLIER_DOMAINS
        seeds = dev_screen.screen_seeds(40, screen.DOMAIN, screen.NAMESPACE)
        assert len(set(seeds)) == 40
        report = dev_screen.disjointness_report(
            seeds, None, screen_domain=screen.DOMAIN, extra_namespaces=screen.SPEC.extra_namespaces
        )
        assert report["disjoint"] is True
        assert len(report["extra_namespaces"]) == len(screen.SPEC.extra_namespaces) >= 23
        smoke = dev_screen.screen_seeds(2, screen.SMOKE_DOMAIN, screen.NAMESPACE)
        assert not set(smoke) & set(seeds)

    def test_specs(self):
        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        spec = screen.SPEC
        assert spec.arm_vetoes["A"] is spec.arm_vetoes["C"] == dev_screen.BUILTIN_VETO
        assert spec.arm_vetoes["B"] is spec.arm_vetoes["D"] is screen.install_v5
        assert spec.replay_worlds == screen.REPLAY_WORLDS == 4
        assert spec.veto_arms() == ("A", "B")
        assert spec.require_slot_locks and spec.require_ac_power
        assert spec.max_wall_seconds == screen.MAX_WALL_SECONDS == 4 * 3600
        assert screen.SMOKE_SPEC.domain == screen.SMOKE_DOMAIN != spec.domain
        assert dev_screen.DEFAULT_SPEC.replay_worlds == 0  # harness defaults untouched
        assert screen.expected_descriptor("D") == screen.expected_descriptor("B")
        assert screen.expected_descriptor("B")["method"] == screen.V5_METHOD
        assert screen.expected_descriptor("C") == FreeSpaceVeto().descriptor()

    def test_intent_fields_bind_protocol_rule_and_cap(self):
        from datetime import datetime, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        deadline = datetime(2026, 10, 2, tzinfo=timezone.utc)
        fields = dev_screen.spec_intent_fields(screen.SPEC, 276, deadline)
        protocol = REPO / "research/apex_veto_v5_screen_20261001/protocol.md"
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
        for phrase in ("4 h (14400 s)", "276 episodes", "trap-horizon-dev-v1", "40 worlds"):
            assert phrase in text


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

    def _v5_record(self, entry):
        record = copy.deepcopy(entry["record"])
        counters = record["probes"]["safety_veto"]["counters"]
        record["probes"]["safety_veto"] = {
            **BoostAwareFreeSpaceVeto().descriptor(),
            "counters": counters,
        }
        return record

    def test_real_v2_record_passes_as_a_and_fails_as_b(self):
        entry = _real()
        assert self._check(entry, "A") == [] and self._check(entry, "C") == []
        assert any("arm B" in f for f in self._check(entry, "B"))
        assert any("arm D" in f for f in self._check(entry, "D"))

    def test_a_v5_probe_on_the_real_record_passes_as_b_and_d(self):
        entry = _real()
        record = self._v5_record(entry)
        assert self._check(entry, "B", record) == [] and self._check(entry, "D", record) == []
        assert self._check(entry, "A", record)
        record["probes"]["safety_veto"]["landing_rule"] = "other"
        assert self._check(entry, "B", record)
        broken = self._v5_record(entry)
        broken["denominators"]["scored_frames"] = 4999
        assert any("H5000" in f for f in self._check(entry, "B", broken))
        assert any("seed" in f for f in self._check(dict(entry, world_seed=1), "B", record))

    def test_entry_rules_and_diagnostics_warnings(self):
        screen = _screen()
        real = _real()
        record = self._v5_record(real)
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
            "safety_veto_method": screen.V5_METHOD,
            "veto_diagnostics": _diag(counters),
            "record": record,
        }
        assert screen.check_entry(entry, smoke=False) == {"failures": [], "warnings": []}
        assert screen.check_entry(dict(entry, arm="D"), smoke=False)["failures"] == []
        stale = _diag(counters, decisions=counters["decisions"] + 1)
        assert screen.check_entry(dict(entry, veto_diagnostics=stale), smoke=False)["warnings"]
        too_many = _diag(counters, boost_landing_vetoes=counters["vetoed_base_boost"] + 1)
        assert screen.check_entry(dict(entry, veto_diagnostics=too_many), smoke=False)["warnings"]
        assert screen.check_entry(dict(entry, veto_diagnostics=None), smoke=False)["warnings"]
        a_entry = dict(entry, arm="A", safety_veto_method=screen.V2_METHOD)
        assert screen.check_entry(a_entry, smoke=False)["failures"]  # v5 probe on arm A
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

    def test_smoke_size_guard(self, no_episodes, tmp_path, capsys):
        argv = self._argv(tmp_path, "--smoke-frames", "501", "--worlds-per-mix", "1")
        with pytest.raises(SystemExit):
            _screen().main(argv + ["--determinism-worlds", "0"])
        assert not (tmp_path / "o").exists()


class TestLandingReport:
    def test_per_decision_rates_and_totals(self):
        screen = _screen()
        diag = {key: 0 for key in screen.SUMMED_V5_FIELDS}
        diag.update(decisions=100, boost_base_decisions=40, boost_landing_vetoes=4)
        diag.update(landing_checks=30, action_differs_from_v2=5, apply_seconds_total=0.5)
        diag["apply_seconds_max"] = 0.01
        entries = [
            dict(_entry("B", "frozen", 1, 1.0), veto_diagnostics=dict(diag), wall_seconds=4.0),
            dict(_entry("B", "mixed", 1, 1.0), veto_diagnostics=dict(diag), wall_seconds=6.0),
            dict(_entry("B", "mixed", 2, 1.0), veto_diagnostics=None, wall_seconds=6.0),
            dict(_entry("A", "mixed", 1, 1.0), wall_seconds=2.0),
        ]
        report = screen.v5_landing_report(entries)
        total = report["total"]
        assert total["episodes"] == 3 and total["missing_diagnostics"] == 1
        assert total["decisions"] == 200 and total["boost_landing_vetoes"] == 8
        assert total["landing_veto_rate_per_boost_base"] == pytest.approx(0.1)
        assert total["mean_apply_seconds_per_decision"] == pytest.approx(0.005)
        assert total["apply_seconds_max"] == 0.01
        assert report["per_mix"]["mixed"]["mean_episode_wall_seconds_arm_A"] == 2.0
        assert report["per_mix"]["frozen"]["mean_episode_wall_seconds"] == 4.0
        assert report["gating"] is False and screen.v5_landing_report([])["total"] == {}
