"""Tests for the opt-in tail-aware free-space veto (src/evaluation/safety_veto_v3.py)."""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import pytest
import torch

from src.evaluation.safety_veto import (
    FreeSpaceVeto,
    free_space_threshold,
    spacious_directions,
)
from src.evaluation.safety_veto_v3 import (
    VETO_METHOD_V3,
    TailAwareFreeSpaceVeto,
    grid_for,
    install_tail_aware_veto,
    own_body_release,
    reachable_counts,
    tail_aware_reachable,
)
from src.game.game_logic import GameLogic
from src.game.snake import Snake

REPO = Path(__file__).resolve().parents[1]
# The v2 strict receipt (run-v3) binds this exact source sha256.
V2_SOURCE_SHA256 = "1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428"
SS = 10
ALL = [True] * 6
Q_PREFERS_LEFT = [9.0, 1.0, 2.0, 0.0, 0.0, 0.0]


def make_snake(cells_head_first, direction, length=None, width=100, height=100, sid=0):
    """A real Snake on a ``width x height`` px board whose body is ``cells`` (grid units)."""
    head = cells_head_first[0]
    snake = Snake(sid, (255, 0, 0), (head[0] * SS, head[1] * SS), SS, width, height)
    snake.segments = [(x * SS, y * SS) for x, y in cells_head_first]
    snake.length = len(cells_head_first) if length is None else int(length)
    snake.direction = direction
    return snake


def hairpin_with_tail_exit():
    """Head boxed in at the bottom-left; the only exits run through its own tail.

    Grid 10x10. Body (tail -> head): (0,8) (0,7) (1..8,7) (8,8) (8,9) (7..1,9);
    head (1,9) moving left. Left turn = down = off the board; straight (0,9) is a
    1-cell nook beside the tail; right turn (1,8) enters the 7-cell corridor y=8.
    Statically every direction is a pocket smaller than the 19-cell body, but the
    tail (0,8) vacates on the first move, so both straight and right lead out.
    """
    tail_to_head = [(0, 8), (0, 7)] + [(x, 7) for x in range(1, 9)]
    tail_to_head += [(8, 8), (8, 9)] + [(x, 9) for x in range(7, 0, -1)]
    return make_snake(list(reversed(tail_to_head)), (-1, 0))


def sealed_pocket_beside_neck():
    """A 2-cell pocket walled by body cells near the HEAD (they release far too late).

    Grid 10x10. Head (2,2) moving up; body (head -> tail): (2,2) (2,3) (2,4) (3,4)
    (4,4) (4,3) (4,2) (4,1) (3,1) (2,1) (1,1) (1,0) (2..9,0). Right turn (3,2) enters
    the pocket {(3,2),(3,3)}; straight (2,1) is body; left turn (1,2) is open space.
    """
    head_to_tail = [(2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)]
    head_to_tail += [(3, 1), (2, 1), (1, 1), (1, 0)] + [(x, 0) for x in range(2, 10)]
    return make_snake(head_to_tail, (0, -1))


def boost_into_other_snakes_pocket():
    """One-step cell open, two-step cell a dead end walled by ANOTHER snake.

    Head (2,5) moving right; own body (2,5) (1,5) (1,4) (1,3) (1,2) (1,1) (2..9,1).
    Other snake occupies (4,4) (5,4) (5,5) (5,6) (4,6): straight's first cell (3,5)
    opens up and down, but a straight BOOST lands on (4,5) with (3,5) as its neck.
    """
    own = [(2, 5), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1)] + [(x, 1) for x in range(2, 10)]
    snake = make_snake(own, (1, 0))
    other = make_snake([(4, 4), (5, 4), (5, 5), (5, 6), (4, 6)], (0, -1), sid=1)
    return snake, other


def simulate_survives(snake, relative_moves):
    """Play relative moves with the real ``Snake.move`` + self-collision rule."""
    for relative in relative_moves:
        snake.direction = GameLogic.relative_to_absolute_direction(snake.direction, relative)
        snake.move()
        if GameLogic.check_wall_collision(snake) or GameLogic.check_self_collision(snake):
            return False
    return True


class TestV2Untouched:
    def test_v2_source_bytes_are_unchanged(self):
        source = REPO / "src/evaluation/safety_veto.py"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == V2_SOURCE_SHA256

    def test_v2_descriptor_is_unchanged(self):
        assert FreeSpaceVeto().descriptor() == {
            "method": "free-space-veto/v2-speed-preserving",
            "free_space_bfs_cap": 160,
            "free_space_min_cap": 32,
            "boost_approximation": "one-step-direction-feature",
            "replacement_rule": "highest-q-eligible-same-speed-mode-then-other",
        }


def v2_flags(snake, others=()):
    roster = [snake, *others]
    cap, need = free_space_threshold(snake.length, snake._logical_length())
    return spacious_directions(snake._get_free_space_features(roster), cap, need)


class TestReleaseModel:
    def test_release_distance_is_index_from_tail_plus_pending_growth_plus_slack(self):
        cells = [(5, 5), (4, 5), (3, 5), (2, 5)]  # head first
        snake = make_snake(cells, (1, 0))
        release = own_body_release(snake.segments, 4, grid_for(snake), slack=0)
        # Tail (index 0 from the tail) vacates on the first move, the head cell last.
        assert [release[c] for c in reversed(cells)] == [1, 2, 3, 4]
        # Pending growth of 2 (length 6 > 4 filled segments) delays every cell by 2.
        delayed = own_body_release(snake.segments, 6, grid_for(snake), slack=1)
        assert [delayed[c] for c in reversed(cells)] == [4, 5, 6, 7]

    def test_release_model_matches_snake_move(self, setup_config):
        cells = [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5)]
        snake = make_snake(cells, (0, -1), length=6)  # one pending growth
        release = own_body_release(snake.segments, snake.length, grid_for(snake), slack=0)
        for k in range(1, 8):
            snake.move()
            occupied = {(x // SS, y // SS) for x, y in snake.segments}
            # The head moves up and never revisits an old cell, so occupancy is
            # exactly "fewer moves than the release distance".
            assert [cell in occupied for cell in cells] == [k < release[c] for c in cells]

    def test_body_cell_rejected_early_is_entered_from_a_farther_neighbour(self):
        # 2x3 grid; start (0,0) at distance 1; own-body cell (0,1) releases at 4.
        # Direct entry (distance 2) is too early; the detour (1,0)->(1,1)->(0,1)
        # arrives at distance 4 and continues to (0,2). (1,2) is another snake.
        def in_bounds(cell):
            return 0 <= cell[0] < 2 and 0 <= cell[1] < 3

        args = ((0, 0), 1, {(1, 2)}, {(0, 1): 4}, in_bounds, 32)
        assert tail_aware_reachable(*args) == 5
        assert tail_aware_reachable(*args, tail_release=False) == 3
        assert tail_aware_reachable(*args[:5], 4) == 4  # the cap bounds the count
        blocked_start = ((0, 1), 1, set(), {(0, 1): 2}, in_bounds, 32)
        assert tail_aware_reachable(*blocked_start) == 0


class TestGeometries:
    def test_tail_exit_v2_gives_up_v3_finds_the_tail(self, setup_config):
        snake = hairpin_with_tail_exit()
        assert snake.length == 19
        assert v2_flags(snake) == [False, False, False]  # v2: no spacious move at all
        counts = reachable_counts(snake, [snake])
        _, need = free_space_threshold(snake.length, snake._logical_length())
        assert counts.one_step[0] == 0  # down is off the board
        assert counts.one_step[1] >= need and counts.one_step[2] >= need
        q = torch.tensor(Q_PREFERS_LEFT)
        mask = torch.tensor([True, True, True, False, False, False])
        v2, v3 = FreeSpaceVeto(), TailAwareFreeSpaceVeto()
        assert v2.apply(snake, [snake], q, mask, 0) == 0  # fallback: base (into the wall)
        assert v2.counters.fallback_no_spacious == 1
        assert v3.apply(snake, [snake], q, mask, 0) == 2  # best eligible: corridor
        assert v3.counters.vetoes_applied == 1

    def test_tail_exit_paths_really_survive_and_the_v2_fallback_dies(self, setup_config):
        # Corridor then out through the vacated tail cells.
        assert simulate_survives(hairpin_with_tail_exit(), [2, 0, 2, 1, 1, 1])
        # Nook then up through the tail.
        assert simulate_survives(hairpin_with_tail_exit(), [1, 2, 1, 1, 1])
        # v2's fallback (the policy's own left turn) leaves the board.
        assert not simulate_survives(hairpin_with_tail_exit(), [0])

    def test_late_release_pocket_is_still_vetoed(self, setup_config):
        snake = sealed_pocket_beside_neck()
        assert v2_flags(snake) == [True, False, False]
        counts = reachable_counts(snake, [snake])
        assert counts.one_step[2] == 2  # neck-side walls do not release in time
        q = torch.tensor([1.0, 0.0, 9.0, 0.0, 0.0, 0.0])  # policy prefers the pocket
        veto = TailAwareFreeSpaceVeto()
        assert veto.apply(snake, [snake], q, torch.tensor(ALL), 2) == 0
        assert veto.counters.vetoes_applied == 1
        # Entering the pocket really is fatal: every 2-move continuation dies.
        for first in range(3):
            for second in range(3):
                pocket = sealed_pocket_beside_neck()
                assert not simulate_survives(pocket, [2, first, second])

    def test_two_step_boost_refinement_vetoes_a_boost_into_a_dead_end(self, setup_config):
        snake, other = boost_into_other_snakes_pocket()
        roster = [snake, other]
        assert v2_flags(snake, [other]) == [True, True, True]
        q = torch.tensor([0.0, 0.0, 0.0, 1.0, 9.0, 2.0])  # boost straight is best
        mask = torch.tensor(ALL)
        assert FreeSpaceVeto().apply(snake, roster, q, mask, 4) == 4  # v2 keeps it
        assert TailAwareFreeSpaceVeto().apply(snake, roster, q, mask, 4) == 4  # default mode
        refined = TailAwareFreeSpaceVeto(boost_two_step=True)
        assert refined.apply(snake, roster, q, mask, 4) == 5  # same-mode: boost right
        assert refined.counters.vetoed_base_boost == 1
        assert refined.counters.vetoes_speed_switched == 0
        counts = reachable_counts(snake, roster, boost_two_step=True)
        assert counts.two_step[1] == 1
        # The boosted head lands on (4,5) with neck (3,5): every next cell is a body.
        landing = (4, 5)
        walls = {(x // SS, y // SS) for x, y in other.segments} | {(3, 5)}
        for dx, dy in ((1, 0), (0, 1), (0, -1)):
            assert (landing[0] + dx, landing[1] + dy) in walls

    def test_open_board_is_identical_to_v2(self, setup_config):
        snake = make_snake([(5, 5), (4, 5), (3, 5)], (1, 0), width=200, height=200)
        q = torch.tensor(Q_PREFERS_LEFT)
        veto = TailAwareFreeSpaceVeto()
        assert veto.apply(snake, [snake], q, torch.tensor(ALL), 0) == 0
        assert veto.counters.kept_base == 1


def random_walk(rng, start, steps, occupied, grid_w, grid_h):
    cells = [start]
    taken = set(occupied) | {start}
    for _ in range(steps - 1):
        x, y = cells[-1]
        options = [
            (x + dx, y + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if 0 <= x + dx < grid_w and 0 <= y + dy < grid_h and (x + dx, y + dy) not in taken
        ]
        if not options:
            break
        nxt = rng.choice(options)
        cells.append(nxt)
        taken.add(nxt)
    return cells


def random_world(rng, grid=12):
    size = grid * SS
    own = random_walk(rng, (rng.randrange(grid), rng.randrange(grid)), 60, (), grid, grid)
    head, neck = own[0], own[1] if len(own) > 1 else own[0]
    direction = (head[0] - neck[0], head[1] - neck[1]) if len(own) > 1 else (1, 0)
    pending = rng.choice([0, 0, 2])
    snake = make_snake(own, direction, length=len(own) + pending, width=size, height=size)
    others = []
    for sid in (1, 2):
        start = (rng.randrange(grid), rng.randrange(grid))
        if start in own:
            continue
        cells = random_walk(rng, start, 15, own, grid, grid)
        other = make_snake(cells, (1, 0), width=size, height=size, sid=sid)
        other.is_alive = sid == 1  # dead snakes are ignored by both versions
        others.append(other)
    return snake, others


class TestEquivalenceAndMonotonicity:
    @pytest.mark.parametrize("arena", ["rectangular", "circular"])
    def test_static_counts_equal_v2_and_tail_release_only_adds(
        self, setup_config, monkeypatch, arena
    ):
        from src.core.game_config import GameConfig

        # Both versions read GameConfig.ARENA_TYPE, so patching it covers the
        # circular-arena bounds test of both at once.
        monkeypatch.setattr(type(GameConfig), "ARENA_TYPE", property(lambda self: arena))
        rng = random.Random(20261001)
        checked = 0
        for _ in range(300):
            snake, others = random_world(rng)
            if len(snake.segments) < 2:
                continue
            roster = [snake, *others]
            cap, need = free_space_threshold(snake.length, snake._logical_length())
            features = snake._get_free_space_features(roster)
            static = reachable_counts(snake, roster, tail_release=False)
            assert [round(f * cap) for f in features] == list(static.one_step)
            aware = reachable_counts(snake, roster)
            assert all(a >= s for a, s in zip(aware.one_step, static.one_step))
            assert all(value <= cap for value in aware.one_step)
            # v3 never vetoes a direction v2 calls spacious.
            v2 = spacious_directions(features, cap, need)
            assert all(a >= need for a, ok in zip(aware.one_step, v2) if ok)
            checked += 1
        assert checked >= 250


class TestHookAndRecord:
    def test_descriptor_and_record_shape(self):
        veto = TailAwareFreeSpaceVeto()
        descriptor = veto.descriptor()
        assert descriptor["method"] == VETO_METHOD_V3 == "free-space-veto/v3-tail-aware"
        assert descriptor["boost_approximation"] == "one-step-direction-feature"
        assert descriptor["replacement_rule"] == FreeSpaceVeto().descriptor()["replacement_rule"]
        assert descriptor["tail_release_slack"] == 1
        record = veto.record()
        assert set(record) == set(descriptor) | {"counters"}
        assert set(record["counters"]) == set(FreeSpaceVeto().record()["counters"])
        refined = TailAwareFreeSpaceVeto(boost_two_step=True).descriptor()
        assert refined["boost_approximation"] == "two-step-path-neck-blocked"
        with pytest.raises(ValueError):
            TailAwareFreeSpaceVeto(slack=-1)

    def test_record_passes_the_strict_wrapper_probe_validator(self, setup_config):
        from src.evaluation.strict_promotion import _validate_candidate_wrapper_probe

        veto = TailAwareFreeSpaceVeto()
        snake = hairpin_with_tail_exit()
        veto.apply(snake, [snake], torch.tensor(Q_PREFERS_LEFT), torch.tensor(ALL), 0)
        _validate_candidate_wrapper_probe(veto.record(), veto.descriptor())

    def test_diagnostics_compare_against_v2(self, setup_config):
        veto = TailAwareFreeSpaceVeto(diagnostics=True)
        snake = hairpin_with_tail_exit()
        q, mask = torch.tensor(Q_PREFERS_LEFT), torch.tensor(ALL)
        assert veto.apply(snake, [snake], q, mask, 0) == 2
        assert veto.diagnostics.to_dict() == {
            "decisions": 1,
            "action_differs_from_v2": 1,
            "v2_no_spacious": 1,
            "v2_no_spacious_rescued": 1,
            "v2_vetoed_v3_kept": 0,
            "directions_released_by_tail": 2,
            # The corridor (right) is spacious only via the tail: tail-admitted.
            "tail_admitted": 1,
            "tail_admitted_last_decision": 1,
            "tail_admitted_in_final_window": 1,
        }
        assert "diagnostics" not in veto.record()  # probe shape stays strict-compatible
        veto.reset()
        assert veto.diagnostics.decisions == 0 and veto.counters.decisions == 0

    def test_install_requires_an_ai_snake(self):
        with pytest.raises(TypeError):
            install_tail_aware_veto(object())

    def test_ai_snake_default_off_and_installed_v3(self, setup_config):
        from src.game.ai_snake import AISnake

        class DqnPolicyStub:
            epsilon = 0.0

            def dqn(self, state):
                return torch.tensor([Q_PREFERS_LEFT])

        snake = AISnake(0, (255, 0, 0), (400, 300), 10, 800, 600, policy=DqnPolicyStub())
        snake.direction = (1, 0)
        assert snake.safety_veto is None
        veto = install_tail_aware_veto(snake)
        assert snake.safety_veto is veto
        snake.update([snake], [(100, 100)])
        assert snake._pre_collision_action == 0
        assert veto.counters.kept_base == 1

    def test_bounded_cost_on_a_long_coiled_snake(self, setup_config):
        # 400-segment serpentine filling a 40x40 board region: the cap (160) bounds work.
        cells = []
        for row in range(10):
            xs = range(40) if row % 2 == 0 else range(39, -1, -1)
            cells.extend((x, row) for x in xs)
        snake = make_snake(list(reversed(cells)), (0, 1), width=400, height=400)
        veto = TailAwareFreeSpaceVeto(boost_two_step=True)
        q, mask = torch.tensor(Q_PREFERS_LEFT), torch.tensor(ALL)
        started = time.perf_counter()
        for _ in range(20):
            veto.apply(snake, [snake], q, mask, 0)
        assert (time.perf_counter() - started) / 20 < 0.05
        assert veto.counters.decisions == 20


# ---------------------------------------------------------------- Tier-1 screen wrapper

FIXTURES = REPO / "research/apex_veto_v3_screen_20261001/fixtures"
REAL_CANDIDATE = FIXTURES / "final-candidate-frozen-1010319811.json"  # run-v3, v2 veto
REAL_INCUMBENT = FIXTURES / "final-incumbent-frozen-1010319811.json"  # run-v3, no veto


def _screen():
    from research.apex_veto_v3_screen_20261001 import screen

    return screen


def _real(path):
    import json

    return json.loads(path.read_text())


def _check_real(entry, arm, record=None):
    return _screen().check_record(
        entry["record"] if record is None else record,
        arm,
        world_seed=entry["world_seed"],
        mix=entry["mix"],
        roster_hashes=entry["roster_member_sha256s"],
        smoke=False,
    )


class TestScreenNamespaces:
    def test_worlds_are_fresh_and_cover_every_strict_bank(self):
        from research.apex_safety_20260926 import dev_screen
        from research.apex_veto_strict_20260927 import strict_run

        screen = _screen()
        strict_domains = {domain for domain, _ in strict_run.NAMESPACES.values()}
        strict_domains |= set(strict_run.RUN_V1_NAMESPACES) | {strict_run.SMOKE_DOMAIN}
        assert strict_domains <= set(screen.EARLIER_DOMAINS)
        assert dev_screen.SCREEN_DOMAIN in screen.EARLIER_DOMAINS
        seeds = dev_screen.screen_seeds(40, screen.DOMAIN, screen.NAMESPACE)
        assert len(set(seeds)) == 40
        report = dev_screen.disjointness_report(
            seeds,
            None,
            screen_domain=screen.DOMAIN,
            extra_namespaces=screen.SPEC.extra_namespaces,
        )
        assert report["disjoint"] is True and report["screen_domain"] == screen.DOMAIN
        assert len(report["extra_namespaces"]) == len(screen.SPEC.extra_namespaces) >= 15
        smoke = dev_screen.screen_seeds(2, screen.SMOKE_DOMAIN, screen.NAMESPACE)
        assert not set(smoke) & set(seeds)
        # A planted overlap is caught.
        planted = {"planted": [seeds[7]]}
        bad = dev_screen.disjointness_report(seeds, None, extra_namespaces=planted)
        assert bad["disjoint"] is False

    def test_specs(self):
        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        assert screen.SPEC.veto_arms() == ("A", "B")
        assert screen.SPEC.arm_vetoes["A"] is screen.SPEC.arm_vetoes["C"]
        assert screen.SPEC.arm_vetoes["A"] == dev_screen.BUILTIN_VETO
        assert screen.SPEC.require_slot_locks and screen.SPEC.require_ac_power
        assert screen.SMOKE_SPEC.domain == screen.SMOKE_DOMAIN != screen.SPEC.domain
        assert dev_screen.DEFAULT_SPEC.veto_arms() == ("B",)
        from research.apex_veto_strict_20260927 import strict_run

        assert dev_screen.DEFAULT_SLOT_LOCK_ROOT == strict_run.SLOT_LOCK_ROOT
        assert not dev_screen.DEFAULT_SPEC.require_slot_locks


class TestScreenSelfCheckOnRealRecords:
    def test_real_v2_candidate_record_passes_as_arm_a(self):
        entry = _real(REAL_CANDIDATE)
        assert _check_real(entry, "A") == []
        assert _check_real(entry, "C") == []

    def test_mutations_fail(self):
        import copy

        entry = _real(REAL_CANDIDATE)
        assert any("arm B" in f for f in _check_real(entry, "B"))  # v2 probe is not v3
        cases = {
            "decisions": lambda r: r["probes"]["safety_veto"]["counters"].update(
                decisions=r["probes"]["safety_veto"]["counters"]["decisions"] + 1
            ),
            "digest": lambda r: r.update(evaluation_profile_digest="0" * 64),
            "identity": lambda r: r["world_identity"].update(seed=1),
            "seed": lambda r: r.update(seed=r["seed"] + 1),
            "frames": lambda r: r["denominators"].update(scored_frames=4999),
            "method": lambda r: r["probes"]["safety_veto"].update(method="other"),
            "counter-keys": lambda r: r["probes"]["safety_veto"]["counters"].pop("kept_base"),
        }
        for name, mutate in cases.items():
            record = copy.deepcopy(entry["record"])
            mutate(record)
            assert _check_real(entry, "A", record), name

    def test_real_incumbent_without_a_probe_fails(self):
        entry = _real(REAL_INCUMBENT)
        assert "safety_veto" not in entry["record"]["probes"]
        assert any("probe" in f for f in _check_real(entry, "A"))

    def test_entry_rules_on_a_wrapped_real_record(self):
        screen = _screen()
        real = _real(REAL_CANDIDATE)
        entry = {
            "schema_version": screen.SCHEMA,
            "authority": screen.AUTHORITY,
            "screen": screen.SCREEN_ID,
            "arm": "A",
            "mix": real["mix"],
            "world_seed": real["world_seed"],
            "roster_member_sha256s": real["roster_member_sha256s"],
            "hero_sha256": real["hero_sha256"],
            "safety_veto": True,
            "safety_veto_method": screen.V2_METHOD,
            "veto_diagnostics": None,
            "record": real["record"],
        }
        assert screen.check_entry(entry, smoke=False) == {"failures": [], "warnings": []}
        wrong = dict(entry, safety_veto_method=screen.V3_METHOD)
        assert screen.check_entry(wrong, smoke=False)["failures"]
        assert screen.check_entry(dict(entry, arm="Z"), smoke=False)["failures"]


@pytest.fixture
def tiny_live(setup_config, tmp_path):
    """Tiny live world plus a random 58-D vector checkpoint hero (as test_safety_veto)."""
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


def _tiny_episode(tiny_live, tmp_path, arm, spec, name):
    from research.apex_safety_20260926 import dev_screen

    lookup = {
        dev_screen.CHAMPION[1]: tiny_live,
        "x": ("scripted", "random_safe"),
        "y": ("scripted", "greedy_food"),
    }
    row = {"mix": "scripted", "world_seed": 3, "slots": [{"member_sha256": s} for s in "xy"]}
    records = tmp_path / name
    records.mkdir()
    return dev_screen.run_episode(arm, row, 0, lookup, None, records, smoke_frames=60, spec=spec)


class TestScreenLivePlumbing:
    def test_default_spec_entry_shape_is_unchanged(self, tiny_live, tmp_path):
        from research.apex_safety_20260926 import dev_screen

        entry = _tiny_episode(tiny_live, tmp_path, "B", dev_screen.DEFAULT_SPEC, "d")
        assert set(entry) == {
            "schema_version", "authority", "arm", "mix", "world_index", "world_seed",
            "roster_member_sha256s", "hero_sha256", "safety_veto", "wall_seconds", "record",
        }  # fmt: skip
        assert entry["schema_version"] == dev_screen.SCHEMA and entry["safety_veto"] is True
        assert entry["record"]["probes"]["safety_veto"]["method"] == screen_v2_method()
        a = _tiny_episode(tiny_live, tmp_path, "A", dev_screen.DEFAULT_SPEC, "a")
        assert "safety_veto" not in a["record"]["probes"]

    def test_v3_screen_arms_and_self_check(self, tiny_live, tmp_path):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts import tournament_eval

        screen = _screen()
        original = tournament_eval._install_hero_safety_veto
        b = _tiny_episode(tiny_live, tmp_path, "B", screen.SPEC, "b")
        assert tournament_eval._install_hero_safety_veto is original  # patch restored
        a = _tiny_episode(tiny_live, tmp_path, "A", screen.SPEC, "a")
        c = _tiny_episode(tiny_live, tmp_path, "C", screen.SPEC, "c")
        assert b["safety_veto_method"] == screen.V3_METHOD
        assert b["record"]["probes"]["safety_veto"]["method"] == screen.V3_METHOD
        assert b["veto_diagnostics"]["decisions"] == (
            b["record"]["probes"]["safety_veto"]["counters"]["decisions"]
        )
        assert a["safety_veto_method"] == screen.V2_METHOD and a["veto_diagnostics"] is None
        assert dev_screen.canonical_json(a["record"]) == dev_screen.canonical_json(c["record"])
        for entry in (a, b, c):
            assert screen.check_entry(entry, smoke=True) == {"failures": [], "warnings": []}
        assert screen.check_entry(dict(b, arm="A"), smoke=True)["failures"]

    def test_installer_keeps_the_vector61_guard(self, tiny_live):
        from research.apex_safety_20260926 import dev_screen
        from src.scripts.tournament_eval import rollout

        opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
        with dev_screen.hero_veto_installer(_screen().install_v3):
            with pytest.raises(ValueError, match="vector61 checkpoint hero"):
                rollout(("scripted", "greedy_food"), opponents, 5, 0, hero_safety_veto=True)


def screen_v2_method():
    return FreeSpaceVeto().descriptor()["method"]


class TestSlotLocksAndPower:
    def _root(self, tmp_path):
        from research.apex_safety_20260926 import dev_screen

        for name in dev_screen.SLOT_LOCK_FILES:
            (tmp_path / name).touch()
        return tmp_path

    def test_acquire_release_and_busy(self, tmp_path):
        from research.apex_safety_20260926 import dev_screen

        root = self._root(tmp_path)
        first = dev_screen.acquire_cpu_slots(root, 1, 1.0)
        second = dev_screen.acquire_cpu_slots(root, 1, 1.0)  # takes the other slot
        assert {Path(h.name).name for h in first + second} == set(dev_screen.SLOT_LOCK_FILES)
        with pytest.raises(TimeoutError):
            dev_screen.acquire_cpu_slots(root, 1, 0.2)
        dev_screen.release_cpu_slots(first)
        again = dev_screen.acquire_cpu_slots(root, 1, 1.0)
        dev_screen.release_cpu_slots(again + second)
        with pytest.raises(ValueError):
            dev_screen.acquire_cpu_slots(root, 3, 0.1)

    def test_missing_lock_files_are_never_created(self, tmp_path):
        from research.apex_safety_20260926 import dev_screen

        with pytest.raises(FileNotFoundError):
            dev_screen.acquire_cpu_slots(tmp_path, 1, 0.1)
        assert list(tmp_path.iterdir()) == []

    def _argv(self, tmp_path, out_name, root):
        return [
            "--out", str(tmp_path / out_name), "--deadline-utc", "2099-01-01T00:00:00+00:00",
            "--smoke-frames", "50", "--worlds-per-mix", "1", "--determinism-worlds", "0",
            "--slot-lock-root", str(root), "--slot-timeout-seconds", "0.2",
        ]  # fmt: skip

    def test_screen_refuses_on_battery_and_when_slots_are_busy(
        self, setup_config, tmp_path, monkeypatch
    ):
        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        locks = tmp_path / "locks"
        locks.mkdir()
        root = self._root(locks)
        monkeypatch.setattr(dev_screen, "on_ac_power", lambda: False)
        assert dev_screen.main(self._argv(tmp_path, "o1", root), spec=screen.SMOKE_SPEC) == 2
        assert not (tmp_path / "o1").exists()
        monkeypatch.setattr(dev_screen, "on_ac_power", lambda: True)
        held = dev_screen.acquire_cpu_slots(root, 2, 1.0)
        try:
            assert dev_screen.main(self._argv(tmp_path, "o2", root), spec=screen.SMOKE_SPEC) == 2
        finally:
            dev_screen.release_cpu_slots(held)
        assert not (tmp_path / "o2").exists()
        mismatched = dev_screen.ScreenSpec(arm_vetoes={"A": None, "B": None, "C": "x"})
        assert dev_screen.main(self._argv(tmp_path, "o3", root), spec=mismatched) == 2


class TestSummaryVetoArms:
    def test_both_arms_totalled(self):
        from research.apex_safety_20260926.dev_screen import summarize

        counters = dict(FreeSpaceVeto().record()["counters"], decisions=10, kept_base=10)
        entries = []
        for mix in ("frozen", "scripted", "mixed"):
            for i, seed in enumerate((1, 2, 3)):
                for arm, mass in (("A", 10.0 + i), ("B", 12.0 + i * 1.5)):
                    record = {
                        "mass_integral": mass,
                        "survival_fraction": 0.5,
                        "probes": {"death_cause": "self", "safety_veto": {"counters": counters}},
                        "denominators": {"decision_frames": 10},
                    }
                    entries.append(
                        {"arm": arm, "mix": mix, "world_seed": seed, "wall_seconds": 1.0,
                         "record": record}
                    )  # fmt: skip
        out = summarize(entries, [1, 2, 3], 0, preregistered=(3, 0), veto_arms=("A", "B"))
        row = out["per_mix"]["frozen"]
        assert row["safety_veto_A"]["totals"]["decisions"] == 30
        assert row["safety_veto_B"]["episodes_decisions_match_decision_frames"] == 3


# ---------------------------------------------------------------- review fixes (2026-10-01)


class TestTailAdmissionDiagnostics:
    def test_window_counts_only_the_last_decisions(self):
        from src.evaluation.safety_veto_v3 import CLOSING_POCKET_WINDOW, TailAwareDiagnostics

        diag = TailAwareDiagnostics()
        diag.decisions = 1
        diag.admit()
        for _ in range(CLOSING_POCKET_WINDOW - 1):
            diag.decisions += 1
        assert diag.to_dict()["tail_admitted_in_final_window"] == 1  # 50th-from-last
        diag.decisions += 1
        out = diag.to_dict()
        assert out["tail_admitted_in_final_window"] == 0
        assert out["tail_admitted"] == 1 and out["tail_admitted_last_decision"] == 1
        assert all(isinstance(v, int) for v in out.values()) and "_recent" not in out

    def test_not_admitted_when_v2_also_finds_the_move_spacious(self, setup_config):
        veto = TailAwareFreeSpaceVeto(diagnostics=True)
        snake = sealed_pocket_beside_neck()  # left is statically spacious
        q = torch.tensor([1.0, 0.0, 9.0, 0.0, 0.0, 0.0])
        assert veto.apply(snake, [snake], q, torch.tensor(ALL), 2) == 0
        assert veto.diagnostics.to_dict()["tail_admitted"] == 0


def _b_entry(real, *, death_cause, tail_admitted, in_window, mix=None):
    """A real run-v3 record wrapped as an arm-B entry with given diagnostics."""
    import copy

    record = copy.deepcopy(real["record"])
    record["probes"]["death_cause"] = death_cause
    return {
        "arm": "B",
        "mix": mix or real["mix"],
        "world_seed": real["world_seed"],
        "record": record,
        "veto_diagnostics": {
            "decisions": 10,
            "tail_admitted": tail_admitted,
            "tail_admitted_last_decision": 9 if tail_admitted else 0,
            "tail_admitted_in_final_window": in_window,
        },
    }


class TestClosingPocketReport:
    def test_classifies_real_record_shapes(self):
        screen = _screen()
        real = _real(REAL_CANDIDATE)
        assert real["record"]["probes"]["death_cause"] == screen.SELF_DEATH_CAUSE
        entries = [
            _b_entry(real, death_cause="self", tail_admitted=3, in_window=1),
            _b_entry(real, death_cause="self", tail_admitted=2, in_window=0),
            _b_entry(real, death_cause=None, tail_admitted=4, in_window=2),
            _b_entry(real, death_cause="self", tail_admitted=0, in_window=0),
            _b_entry(real, death_cause="head_on", tail_admitted=1, in_window=1, mix="mixed"),
            dict(real, arm="A", veto_diagnostics=None),  # arm A is ignored
            dict(_b_entry(real, death_cause="self", tail_admitted=1, in_window=1),
                 veto_diagnostics=None),
        ]  # fmt: skip
        out = screen.closing_pocket_report(entries)
        frozen = out["per_mix"]["frozen"]
        assert frozen == {
            "episodes": 5,
            "with_tail_admission": 3,
            "admitted_closing_pocket": 1,
            "rescued": 2,
            "self_collision_deaths": 3,
            "missing_diagnostics": 1,
        }
        assert out["per_mix"]["mixed"]["rescued"] == 1
        assert out["total"]["episodes"] == 6 and out["gating"] is False
        assert out["window_decisions"] == 50
        assert screen.closing_pocket_report([])["total"] == {}


class TestScreenPreregistrationGuards:
    def _argv(self, tmp_path, *extra):
        return ["--out", str(tmp_path / "o"), "--deadline-utc", "2099-01-01T00:00:00+00:00"] + [
            *extra
        ]

    @pytest.mark.parametrize(
        "extra",
        [
            ("--worlds-per-mix", "5", "--determinism-worlds", "2"),
            ("--determinism-worlds", "0"),
            ("--worlds-per-mix", "41", "--determinism-worlds", "8"),
        ],
    )
    def test_non_preregistered_sizes_are_refused_before_any_world(
        self, tmp_path, monkeypatch, extra
    ):
        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        calls = []
        monkeypatch.setattr(dev_screen, "main", lambda *a, **k: calls.append(a) or 0)
        assert screen.main(self._argv(tmp_path, *extra)) == 2
        assert calls == [] and not (tmp_path / "o").exists()

    def test_preregistered_size_with_a_far_deadline_hits_the_wall_cap(self, tmp_path):
        assert _screen().main(self._argv(tmp_path)) == 2  # 2099 is beyond 3 h
        assert not (tmp_path / "o").exists()

    def test_dev_screen_wall_cap_only_for_capped_non_smoke_specs(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        assert screen.SPEC.max_wall_seconds == screen.MAX_WALL_SECONDS == 4 * 3600
        assert dev_screen.DEFAULT_SPEC.max_wall_seconds is None
        # Well past the cap, and rollouts are made fatal: this test once launched the real
        # 264-episode screen when the cap was raised to equal its deadline (revision 4).
        import src.scripts.tournament_eval as tournament_eval

        def no_rollouts(*args, **kwargs):
            raise AssertionError("a guard test must never play an episode")

        monkeypatch.setattr(tournament_eval, "rollout", no_rollouts)
        late = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        argv = ["--out", str(tmp_path / "o"), "--deadline-utc", late]
        assert dev_screen.main(argv, spec=screen.SPEC) == 2
        assert not (tmp_path / "o").exists()


class TestTier1IntentFields:
    def test_spec_fields_bind_protocol_rule_and_cap(self):
        from datetime import datetime, timezone

        from research.apex_safety_20260926 import dev_screen

        screen = _screen()
        deadline = datetime(2026, 10, 2, 6, tzinfo=timezone.utc)
        fields = dev_screen.spec_intent_fields(screen.SPEC, 264, deadline)
        protocol = REPO / "research/apex_veto_v3_screen_20261001/protocol.md"
        assert fields["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
        summary = dev_screen.summarize([], [1, 2], 0, preregistered=(2, 0))
        assert fields["decision_rule"] == summary["decision_rule"]
        assert fields["owner"] == "Apex safety lane"
        assert fields["primary_metric"] == "paired mass_integral B-A per world per mix"
        assert "Holm" in fields["estimator"] and "hypothesis" in fields
        assert "decision_informs" in fields
        assert fields["compute_cap"] == {
            "planned_episodes": 264,
            "deadline_utc": deadline.isoformat(),
            "max_wall_seconds": 14400,
        }

    def test_default_spec_decision_rule_text_unchanged(self):
        from research.apex_safety_20260926 import dev_screen

        assert dev_screen.decision_rule_text() == (
            "NON_PREREGISTERED_DESIGN unless worlds_per_mix and determinism_worlds equal the "
            "pre-registered sizes; RECOMMEND_STRICT_GATE iff complete, A/C deterministic, and "
            "one-sided Holm (alpha 0.05) rejects mass-integral H0 in >= 2 of 3 mixes"
        )
