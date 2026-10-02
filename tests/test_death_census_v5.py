"""research/trap_horizon_20261001/diagnose_live_v5.py: live v5 death census.

Every test that calls the harness ``main`` makes both episode runners (``rollout`` and
``run_simd_eval``) fatal. One integration test calls ``tournament_eval.rollout``
directly on a tiny synthetic world (never the harness) to check the live capture.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from research.trap_horizon_20261001 import diagnose as base
from research.trap_horizon_20261001 import diagnose_live_v5 as d

META = {
    "grid_width": 20,
    "grid_height": 12,
    "min_boost_length": 5,
    "boost_cost_frames": 3,
    "trail_food": True,
    "segment_size": 10,
    "mechanics_version": 2,
}


@pytest.fixture
def runners_fatal(monkeypatch):
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    def boom(*_args, **_kwargs):
        raise AssertionError("an episode runner was called in a census unit test")

    monkeypatch.setattr(ee, "run_simd_eval", boom)
    monkeypatch.setattr(te, "rollout", boom)


def test_runners_fatal_fixture(runners_fatal):
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    with pytest.raises(AssertionError):
        te.rollout()
    with pytest.raises(AssertionError):
        ee.run_simd_eval()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hero(cells, length=None, direction=1, boost_frames=0):
    cells = tuple(tuple(c) for c in cells)
    return base.HeroState(cells, len(cells) if length is None else length, direction, boost_frames)


def other(cells, traversed, direction, length=None, alive=True):
    return {
        "cells": [tuple(c) for c in cells],
        "traversed": [tuple(c) for c in traversed],
        "direction": direction,
        "logical_length": len(cells) if length is None else length,
        "alive": alive,
        "is_hero": False,
    }


def outcome(state, action, others, food=frozenset(), ratio=1.15):
    return d.one_frame_outcome(META, state, action, others, food, ratio)["cause"]


ROW = [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5)]  # heading right (direction 1)


# ---------------------------------------------------------------------------
# One-frame counterfactual
# ---------------------------------------------------------------------------
def test_head_on_same_cell_and_size_resolution():
    # The other snake's head moved down into (6, 5): same cell as the hero's straight move.
    o = other([(6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0)], [(6, 5)], direction=2)
    assert outcome(hero(ROW), 1, [o]) == "head_on"
    assert outcome(hero(ROW), 0, [o]) == "alive"  # left turn goes up to (5, 4)
    big = hero(ROW, length=7)  # 7 >= 1.15 * 6: the hero wins
    assert outcome(big, 1, [o]) == "alive"
    small = dict(o, logical_length=7)
    assert outcome(hero(ROW), 1, [small]) == "head_on"


def test_head_paths_crossed_is_head_on():
    # The other head moves left from (6, 5) to (5, 5) while the hero moves (5, 5) -> (6, 5).
    o = other([(5, 5), (6, 5), (7, 5), (8, 5), (9, 5), (10, 5)], [(5, 5)], direction=3)
    assert d._paths_crossed([(5, 5), (6, 5)], [(6, 5), (5, 5)])
    assert outcome(hero(ROW), 1, [o]) == "head_on"


def test_enemy_body_and_dead_snakes_are_ignored():
    o = other([(7, 3), (7, 4), (6, 4), (5, 4)], [(7, 3)], direction=0)
    assert outcome(hero(ROW), 0, [o]) == "enemy_body"  # up into (5, 4)
    assert outcome(hero(ROW), 1, [o]) == "alive"
    assert outcome(hero(ROW), 0, [dict(o, alive=False)]) == "alive"


def test_boost_second_cell_and_wall_and_self():
    o = other([(9, 9), (9, 8), (7, 5), (7, 6)], [(9, 9)], direction=2)
    assert outcome(hero(ROW), 1, [o]) == "alive"
    assert outcome(hero(ROW), 4, [o]) == "enemy_body"  # boost: (6, 5) then (7, 5)
    edge = hero([(19, 5), (18, 5), (17, 5), (16, 5), (15, 5)])
    assert outcome(edge, 1, []) == "wall"
    curl = hero([(5, 5), (5, 6), (4, 6), (4, 5), (4, 4), (5, 4), (6, 4)], direction=0)
    assert outcome(curl, 1, []) == "self"  # up into (5, 4), its own body


def _live_hero_fate(snakes):
    """The hero's (slot 0) fate from the live detector plus the v2 resolution rule."""
    from src.core.mechanics_constants import HEADON_SIZE_RATIO
    from src.game.game_logic import GameLogic

    for snake, other_snake, kind in GameLogic.check_collisions(snakes):
        if snake is not snakes[0]:
            continue
        if kind in ("wall", "self"):
            return kind
        if kind == "head":
            if snakes[0]._logical_length() >= HEADON_SIZE_RATIO * other_snake._logical_length():
                continue
            return "head_on"
        return "enemy_body"
    return "alive"


def _random_body(rng, start, length, width, height, taken):
    body = [start]
    for _ in range(length - 1):
        x, y = body[-1]
        options = [
            (x + dx, y + dy)
            for dx, dy in base.CARDINAL
            if 0 <= x + dx < width
            and 0 <= y + dy < height
            and (x + dx, y + dy) not in taken
            and (x + dx, y + dy) not in body
        ]
        if not options:
            break
        body.append(rng.choice(options))
    return body


def test_one_frame_outcome_matches_live_collision_detection(setup_config):
    """Random small worlds: live ``Snake.move`` + ``check_collisions`` vs the cell model."""
    from src.core.game_config import GameConfig
    from src.game.game_logic import GameLogic
    from src.game.snake import Snake

    ss, width, height = 10, 12, 10
    meta = dict(META, grid_width=width, grid_height=height)
    rng = random.Random(7)
    seen = {}
    for trial in range(1500):
        taken: set = set()
        cells_by = []
        for sid in range(3):
            free = [(x, y) for x in range(width) for y in range(height) if (x, y) not in taken]
            body = _random_body(rng, rng.choice(free), rng.randint(3, 12), width, height, taken)
            taken |= set(body)
            cells_by.append(body)
        snakes = []
        for sid, body in enumerate(cells_by):
            s = Snake(
                sid, (255, 0, 0), (body[0][0] * ss, body[0][1] * ss), ss, width * ss, height * ss
            )
            s.segments = [(x * ss, y * ss) for x, y in body]
            s.length = len(body) + rng.choice([0, 0, 1])
            if len(body) > 1:
                hx, hy = body[0]
                nx, ny = body[1]
                s.direction = (hx - nx, hy - ny)
            else:
                s.direction = rng.choice(base.CARDINAL)
            snakes.append(s)
        pre = hero(
            cells_by[0], length=snakes[0].length, direction=base.CARDINAL.index(snakes[0].direction)
        )
        hero_action = rng.randrange(6)
        for idx, s in enumerate(snakes):
            a = hero_action if idx == 0 else rng.randrange(6)
            s.direction = GameLogic.relative_to_absolute_direction(s.direction, a % 3)
            s.is_boosting = a >= 3 and s.length >= GameConfig.MIN_BOOST_LENGTH
            s.move()
        live = _live_hero_fate(snakes)
        rows = [
            other(
                [(int(x // ss), int(y // ss)) for x, y in s.segments],
                [(int(x // ss), int(y // ss)) for x, y in s.last_move_positions],
                base.CARDINAL.index(s.direction),
                length=s._logical_length(),
            )
            for s in snakes[1:]
        ]
        model = d.one_frame_outcome(meta, pre, hero_action, rows, frozenset(), 1.15)["cause"]
        assert model == live, (trial, cells_by, hero_action)
        seen[live] = seen.get(live, 0) + 1
    assert {"alive", "self", "wall", "head_on", "enemy_body"} <= set(seen), seen


# ---------------------------------------------------------------------------
# Fatal-frame analysis
# ---------------------------------------------------------------------------
def snapshot(body, others=(), food=(), final=1, mask=None, length=None, direction=1):
    return {
        "frame": 10,
        "body": np.asarray(body, dtype=np.int16),
        "length": len(body) if length is None else length,
        "direction": direction,
        "boost_frames": 0,
        "others": np.asarray(list(others), dtype=np.int16).reshape(-1, 2),
        "food": np.asarray(list(food), dtype=np.int16).reshape(-1, 2),
        "v2_counts": np.asarray([10, 10, 10]),
        "v2_need": 5,
        "base": final,
        "final": final,
        "q": np.zeros(6, np.float32),
        "mask": np.asarray([True] * 6 if mask is None else mask),
        "v5_reason": 0,
    }


def test_fatal_frame_alternatives_and_static_flag():
    pre_other = [(6, 6), (6, 7), (6, 8)]  # heading up; its head moves to (6, 5)
    post_other = other([(6, 5), (6, 6), (6, 7)], [(6, 5)], direction=0, length=6)
    snap = snapshot(ROW, others=pre_other, final=1)
    post = {"frame": 10, "food": [], "snakes": [{"is_hero": True}, post_other]}
    out = d.fatal_frame_analysis(snap, post, META, "head_on", depth=10, budget=5000)
    assert out["taken_model_cause"] == "head_on" and out["model_reproduces_cause"]
    assert out["actions"]["1"]["cause"] == "head_on"
    assert out["taken_hit_static_pre_move"] is False  # (6, 5) was empty before the move
    assert 0 in out["legal_alternatives_avoiding"] and 2 in out["legal_alternatives_avoiding"]
    assert out["any_legal_alternative_avoids_with_escape"] is True
    assert out["actions"]["0"]["escape"] == base.ESCAPE
    # Masking the alternatives removes them from the legal list.
    masked = snapshot(ROW, others=pre_other, final=1, mask=[False, True, False, False, True, False])
    out = d.fatal_frame_analysis(masked, post, META, "head_on", depth=10, budget=5000)
    assert out["any_legal_alternative_avoids"] is False


def test_fatal_frame_static_body_hit_and_cause_mismatch_is_reported():
    wall = [(6, 5), (6, 4), (6, 3)]
    post_other = other([(6, 2), (6, 5), (6, 4), (6, 3)], [(6, 2)], direction=0)
    snap = snapshot(ROW, others=wall, final=1)
    post = {"frame": 10, "food": [], "snakes": [{"is_hero": True}, post_other]}
    out = d.fatal_frame_analysis(snap, post, META, "enemy_body", depth=5, budget=1000)
    assert out["taken_hit_static_pre_move"] is True and out["model_reproduces_cause"]
    out = d.fatal_frame_analysis(snap, post, META, "self", depth=5, budget=1000)
    assert out["model_reproduces_cause"] is False


# ---------------------------------------------------------------------------
# Self-death extras
# ---------------------------------------------------------------------------
def test_static_landing_count_equals_v5_landing_count(setup_config):
    from src.evaluation.safety_veto import free_space_threshold
    from src.evaluation.safety_veto_v5 import landing_count
    from src.game.snake import Snake

    ss = 10
    rng = random.Random(3)
    for _ in range(200):
        taken: set = set()
        hero_body = _random_body(
            rng, (rng.randrange(20), rng.randrange(12)), rng.randint(5, 30), 20, 12, taken
        )
        if len(hero_body) < 5:
            continue
        taken |= set(hero_body)
        free = [(x, y) for x in range(20) for y in range(12) if (x, y) not in taken]
        wall = _random_body(rng, rng.choice(free), rng.randint(2, 20), 20, 12, taken)
        me = Snake(0, (1, 1, 1), (hero_body[0][0] * ss, hero_body[0][1] * ss), ss, 200, 120)
        me.segments = [(x * ss, y * ss) for x, y in hero_body]
        me.length = len(hero_body)
        hx, hy = hero_body[0]
        nx, ny = hero_body[1]
        me.direction = (hx - nx, hy - ny)
        it = Snake(1, (2, 2, 2), (wall[0][0] * ss, wall[0][1] * ss), ss, 200, 120)
        it.segments = [(x * ss, y * ss) for x, y in wall]
        it.length = len(wall)
        snap = snapshot(hero_body, others=wall, direction=base.CARDINAL.index(me.direction))
        world, state = base.world_and_state(snap, META)
        cap, _ = free_space_threshold(me.length, me.length)
        for direction in range(3):
            assert d.static_landing_count(world, state, direction, cap) == landing_count(
                me, [me, it], direction, cap
            )


def walk(status="exact", pnr=5, fatal=10, taken=1, taken_status=base.NO_ESCAPE, escaping=(0,)):
    return {
        "status": status,
        "pnr_index": pnr,
        "fatal_index": fatal,
        "frames_before_death": None if pnr is None else fatal - pnr,
        "frames_before_lower": None if pnr is None else fatal - pnr,
        "taken_action": taken,
        "taken_status": taken_status,
        "escaping_actions": list(escaping),
        "fatal_choice": taken_status == base.NO_ESCAPE and bool(escaping),
    }


def death(primary, count_only):
    return {"walk": primary, "sensitivity": {base.COUNT_ONLY: {"walk": count_only}}}


def test_classification_and_enclosed_early():
    assert d.classify_self_death(death(walk(taken=4), walk())) == "boost_fatal_choice"
    assert d.classify_self_death(death(walk(taken=1), walk())) == "normal_fatal_choice"
    escaped = walk(taken_status=base.ESCAPE)
    assert d.classify_self_death(death(escaped, walk())) == "taken_escaped"
    unknown = walk(taken_status=base.UNKNOWN)
    assert d.classify_self_death(death(unknown, walk())) == "unresolved"
    assert d.classify_self_death(death(walk(pnr=None), walk())) == "unresolved"
    assert d.enclosed_early(death(walk(), walk(pnr=0, fatal=30))) is True
    assert d.enclosed_early(death(walk(), walk(pnr=10, fatal=30))) is False
    beyond = dict(walk(pnr=None), status="beyond_window", frames_before_lower=120)
    assert d.enclosed_early(death(walk(), beyond)) is True
    low = dict(walk(), status=base.UNKNOWN, frames_before_lower=3)
    assert d.enclosed_early(death(walk(), low)) is None


def test_pack_unpack_keeps_v5_reason():
    snaps = [snapshot(ROW, final=1), dict(snapshot(ROW, final=4), v5_reason=2)]
    back = d.unpack_live_window(d.pack_live_window(snaps))
    assert [s["v5_reason"] for s in back] == [0, 2]
    assert [s["final"] for s in back] == [1, 4]


def test_summary_counts():
    fatal = {
        "model_reproduces_cause": True,
        "any_legal_alternative_avoids": True,
        "any_legal_alternative_avoids_with_escape": False,
        "taken_hit_static_pre_move": False,
        "taken": 4,
    }
    v5 = {"v5_reason": "kept", "outcome": "kept"}
    episodes = [
        {
            "mix": "frozen",
            "deaths": 1,
            "death_cause": "head_on",
            "mass_integral": 10.0,
            "v5_diagnostics": {"decisions": 5, "boost_landing_vetoes": 1, "apply_seconds_max": 9},
        },
        {
            "mix": "frozen",
            "deaths": 0,
            "death_cause": None,
            "mass_integral": 30.0,
            "v5_diagnostics": {"decisions": 7, "boost_landing_vetoes": 0},
        },
    ]
    deaths = [
        {
            "mix": "frozen",
            "cause": "head_on",
            "fatal": fatal,
            "fatal_v5": v5,
            "fatal_length": 50,
            "model_check": {
                "transitions_compared": 3,
                "mismatches": 0,
                "fatal_action_model_outcome": "alive",
            },
        },
    ]
    s = d.summarize_census(episodes, deaths)
    assert s["episode_outcomes_by_mix"] == {"frozen": {"head_on": 1, "survived": 1}}
    assert s["mean_mass_integral_by_mix"] == {"frozen": 20.0}
    assert s["v5_counters_total"] == {"decisions": 12, "boost_landing_vetoes": 1}
    row = s["fatal_frame_by_cause"]["head_on"]
    assert row["deaths"] == 1 and row["any_legal_alternative_avoids"] == 1
    assert row["any_legal_alternative_avoids_with_escape"] == 0 and row["taken_boost"] == 1
    assert "self" not in s


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------
def test_worlds_are_fresh_and_cover_every_v5_namespace():
    from research.apex_veto_v5_serving_20261001 import serving_run
    from research.apex_veto_v5_strict_20261001 import strict_run

    earlier = d.earlier_domains(d.DOMAIN)
    for domain, _ in strict_run.NAMESPACES.values():
        assert earlier[domain] == ("worlds",)
    for domain in (serving_run.SEED_DOMAIN, serving_run.SMOKE_SEED_DOMAIN):
        assert set(earlier[domain]) == {"watch", "play", "parity"}
    for domain in (strict_run.SMOKE_DOMAIN, "apex-veto-v5-screen-v1", "trap-horizon-dev-v1"):
        assert domain in earlier
    assert d.SMOKE_DOMAIN in earlier and d.DOMAIN not in earlier
    seeds = base.world_seeds(d.DOMAIN, d.WORLDS_PER_MIX)
    report = d.disjointness(seeds, d.DOMAIN)
    assert report["disjoint"] is True and len(set(seeds)) == d.WORLDS_PER_MIX
    assert any(
        n.startswith("apex-veto-v5-web-serving-v1/parity") for n in report["extra_namespaces"]
    )


# ---------------------------------------------------------------------------
# main guards (runners fatal)
# ---------------------------------------------------------------------------
def test_main_refuses_existing_out_overrides_env_and_battery(runners_fatal, tmp_path, monkeypatch):
    assert d.main(["--out", str(tmp_path)]) == 2  # exists
    assert d.main(["--out", str(tmp_path / "a"), "--frames", "10"]) == 2  # smoke-only
    assert d.main(["--out", str(tmp_path / "a"), "--worlds-per-mix", "2"]) == 2
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert d.main(["--out", str(tmp_path / "a")]) == 2
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    monkeypatch.setattr(base, "on_ac_power", lambda: False)
    assert d.main(["--out", str(tmp_path / "a")]) == 2
    assert not (tmp_path / "a").exists()


# ---------------------------------------------------------------------------
# Live capture on a tiny world (direct rollout; never the harness)
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_live(setup_config, tmp_path):
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


def test_live_capture_matches_veto_and_model(tiny_live):
    from src.evaluation.safety_veto_v5 import VETO_METHOD_V5
    from src.game.game_logic import GameLogic
    from src.scripts import tournament_eval as te

    originals = (
        GameLogic.__dict__["check_collisions"],
        te.create_training_game_state,
        te._install_hero_safety_veto,
    )
    opponents = [("scripted", "random_safe"), ("scripted", "greedy_food")]
    checked = 0
    for seed in range(6):
        recorder = d.LiveRecorder(window=50)
        with d.live_capture(recorder):
            record = te.rollout(tiny_live, opponents, 150, seed, hero_safety_veto=True)
        probe = record["probes"]["safety_veto"]
        assert probe["method"] == VETO_METHOD_V5
        assert recorder.decisions == probe["counters"]["decisions"] > 0
        snaps = list(recorder.buffer)
        frames = [s["frame"] for s in snaps]
        assert frames == list(range(frames[0], frames[0] + len(frames)))
        check = base.model_replay_check(snaps, recorder.world_meta)
        assert check["mismatches"] == 0
        if record["deaths"] >= 1:
            cause = record["probes"]["death_cause"]
            assert recorder.post["frame"] == snaps[-1]["frame"]
            fatal = d.fatal_frame_analysis(
                snaps[-1], recorder.post, recorder.world_meta, cause, depth=5, budget=500
            )
            assert fatal["model_reproduces_cause"], (seed, cause, fatal["taken_model_cause"])
            checked += 1
    assert checked >= 1
    assert originals == (
        GameLogic.__dict__["check_collisions"],
        te.create_training_game_state,
        te._install_hero_safety_veto,
    )
