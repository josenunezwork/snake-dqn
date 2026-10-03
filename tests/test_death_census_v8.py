"""research/death_census_v8_20261003/census.py: Tier-0 v8 death census (SIMD engine).

Every test that calls the harness ``main`` makes both episode runners (``rollout`` and
``run_simd_eval``) fatal. The capture wrappers are tested on constructed BatchSim worlds
(one ``_apply_veto`` call or one ``BatchSim.step``; no episode). One tiny-world
``run_simd_eval`` integration test plays episodes and runs only with
``SNAKE_CENSUS_TINY_ROLLOUT=1``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from research.death_census_v8_20261003 import census as c
from research.trap_horizon_20261001 import diagnose as base
from tests.test_simd_vector61_v7v8_veto import (
    ALL,
    SafetyVetoSpec,
    batch_worlds,
    boxed_world,
    head_on_world,
    live_counts,
    live_roster,
    masked_row,
    pocket_world,
    simd_hook,
    trap_world,
)

META = {
    "grid_width": 20,
    "grid_height": 12,
    "min_boost_length": 5,
    "boost_cost_frames": 3,
    "trail_food": True,
    "segment_size": 1,
    "mechanics_version": 2,
}
ROW = [(5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5)]  # heading right (direction 1)


@pytest.fixture
def runners_fatal(monkeypatch):
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    def boom(*_args, **_kwargs):
        raise AssertionError("an episode runner was called in a census unit test")

    monkeypatch.setattr(ee, "run_simd_eval", boom)
    monkeypatch.setattr(te, "rollout", boom)


@pytest.fixture
def v2_world(setup_config):
    from src.core.config_loader import load_and_initialize_config
    from tests.test_simd_vector61_v7v8_veto import DEPLOYMENT_YAML

    load_and_initialize_config(str(DEPLOYMENT_YAML))
    yield


def test_runners_fatal_fixture(runners_fatal):
    from src.scripts import tournament_eval as te
    from src.simd_env import eval_engine as ee

    with pytest.raises(AssertionError):
        te.rollout()
    with pytest.raises(AssertionError):
        ee.run_simd_eval()


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def test_parse_clamshell_fails_closed():
    assert c.parse_clamshell('  |   "AppleClamshellState" = No\n') is True
    assert c.parse_clamshell('"AppleClamshellState" = Yes') is False
    assert c.parse_clamshell('"AppleClamshellState" = No\n"AppleClamshellState" = Yes') is False
    assert c.parse_clamshell("") is None
    assert c.parse_clamshell('"AppleClamshellState" = maybe') is None


def test_safety_monitor_pauses_resumes_and_aborts():
    readings = iter([(False, True), (True, False), (True, True)])
    state = {"v": (True, True)}

    def ac():
        return state["v"][0]

    def lid():
        return state["v"][1]

    events = []
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        state["v"] = next(readings)

    m = c.SafetyMonitor(ac, lid, poll=1.0, max_pause=100.0, log=events.append, sleep=sleep)
    assert m.safe
    m.wait_if_unsafe()  # safe: no pause
    assert m.pauses == 0
    state["v"] = (False, True)
    m.state = m.read()
    assert not m.safe
    m.wait_if_unsafe()  # polls synchronously (no thread) until safe
    assert m.safe and m.pauses == 1 and len(sleeps) == 3
    kinds = [e["event"] for e in events]
    assert kinds == ["safety_pause", "safety_resume"]

    bad = c.SafetyMonitor(
        lambda: False, lambda: True, poll=0.01, max_pause=0.0, sleep=lambda s: None
    )
    with pytest.raises(RuntimeError, match="safety"):
        bad.wait_if_unsafe()
    broken = c.SafetyMonitor(lambda: 1 / 0, lambda: True)
    assert broken.safe is False


# ---------------------------------------------------------------------------
# Reasons from counter deltas
# ---------------------------------------------------------------------------
def snap0(**over):
    out = c.hook_snapshot(None)
    out.update(over)
    return out


@pytest.mark.parametrize(
    "delta, reason",
    [
        ({}, "kept"),
        ({"no_spacious": 1, "head_risky_vetoes": 1}, "no_spacious"),
        (
            {"no_spacious": 1, "v5_base_landing_failed": 1, "v5_boost_landing_no_eligible": 1},
            "landing_no_eligible",
        ),
        ({"head_risky_vetoes": 1, "v7_rerank_changes": 1}, "head_veto"),
        ({"v7_rerank_changes": 1, "v5_vetoes_applied": 1}, "v7_rerank"),
        ({"v5_base_landing_failed": 1, "v5_boost_landing_no_eligible": 1}, "landing_no_eligible"),
        (
            {"v5_base_landing_failed": 1, "v5_boost_to_normal_same_direction": 1},
            "landing_same_direction_normal",
        ),
        ({"v5_base_landing_failed": 1, "v5_vetoes_applied": 1}, "landing_v2_rule"),
        ({"v5_vetoes_applied": 1}, "v2_rule"),
    ],
)
def test_reason_from_deltas(delta, reason):
    before = snap0(decisions=3, apply_seconds_total=1.0)
    after = dict(before, decisions=4, apply_seconds_total=1.25)
    for key, value in delta.items():
        after[key] = before[key] + value
    out = c.reason_from_deltas(before, after)
    assert c.REASONS[out["reason"]] == reason
    assert out["apply_seconds"] == pytest.approx(0.25)


def test_reason_requires_exactly_one_decision():
    with pytest.raises(RuntimeError):
        c.reason_from_deltas(snap0(), snap0())
    with pytest.raises(RuntimeError):
        c.reason_from_deltas(snap0(), snap0(decisions=2))


# ---------------------------------------------------------------------------
# Decision capture on constructed worlds (one _apply_veto call each)
# ---------------------------------------------------------------------------
def _decide(world, q, mask, capture: bool):
    spec = SafetyVetoSpec("v8", 8.0)
    sim = batch_worlds([world])
    policy = simd_hook(spec)
    roster = live_roster(world)
    policy.runtime.counts = np.array([live_counts(roster[0], roster)], dtype=np.int64)
    masked, m, b = masked_row(q, mask)
    rows = np.array([[0, 0]], dtype=np.int64)
    args = (sim, rows, torch.tensor([masked], dtype=torch.float32), np.array([m]), np.array([b]))
    if not capture:
        return policy._apply_veto(*args), None, b
    recorder = c.CensusRecorder(1, 10)
    with c.census_capture(recorder):
        out = policy._apply_veto(*args)
    return out, recorder, b


@pytest.mark.parametrize(
    "name, world, q, mask, reason",
    [
        ("head", head_on_world(gap=3), [0.5, 1.0, 0.2, 0.1, 0.9, 0.0], ALL, "head_veto"),
        ("pocket", pocket_world(), [1.0, 0.5, 0.0, -1.0, -1.0, -1.0], ALL, "v7_rerank"),
        ("boxed", boxed_world(), [0.1, 1.0, 0.2, 0.0, 0.5, 0.0], ALL, "no_spacious"),
        ("trap", trap_world(), [0.0, 1.0, 0.5, 0.0, 9.0, 0.2], ALL, None),
        ("open", head_on_world(gap=30), [0.5, 1.0, 0.2, 0.1, 0.2, 0.0], ALL, "kept"),
    ],
)
def test_capture_is_read_only_and_labels_the_layer(v2_world, name, world, q, mask, reason):
    plain, _, _ = _decide(world, q, mask, capture=False)
    out, recorder, b = _decide(world, q, mask, capture=True)
    assert out.tolist() == plain.tolist()
    assert recorder.decision_count(0) == 1
    snap = recorder.buffers[0][-1]
    assert snap["base"] == b and snap["final"] == int(out[0])
    got = c.REASONS[snap["v8_reason"]]
    if reason is None:  # trap: v5's landing check fires
        assert got.startswith("landing_"), got
    else:
        assert got == reason
    assert recorder.decisions[0]["changed"][-1] == (int(out[0]) != b)
    assert recorder.world_meta["grid_width"] == 100
    assert snap["body"][0].tolist() == list(world[0]["cells"][0])
    if name == "head":
        assert snap["head_risky"] is True


def test_capture_restores_the_originals(v2_world):
    from src.simd_env.batch_sim import BatchSim
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    apply, resolve = Vector61SimdPolicy._apply_veto, BatchSim.__dict__["_resolve_collisions"]
    with pytest.raises(ValueError):
        with c.census_capture(c.CensusRecorder(1, 5)):
            raise ValueError("boom")
    assert Vector61SimdPolicy._apply_veto is apply
    assert BatchSim.__dict__["_resolve_collisions"] is resolve
    assert not c._ACTIVE


def test_post_capture_keeps_only_hero_death_frames(v2_world):
    """One BatchSim.step: env 0's hero hits the wall, env 1's hero survives."""
    from src.simd_env.batch_sim import DEATH_NONE

    wall = {"cells": [(99, 40), (98, 40), (97, 40)], "direction": (1, 0)}
    free = {"cells": [(50, 40), (49, 40), (48, 40)], "direction": (1, 0)}
    opp = {"cells": [(10, 10), (9, 10), (8, 10)], "direction": (1, 0)}
    worlds = [[wall, opp], [free, opp]]

    def step(capture):
        sim = batch_worlds(worlds)
        actions = np.ones((2, sim.S), dtype=np.int64)
        recorder = c.CensusRecorder(2, 5)
        if capture:
            recorder.sim = sim
            with c.census_capture(recorder):
                sim.step(actions)
        else:
            sim.step(actions)
        return sim, recorder

    plain, _ = step(False)
    sim, recorder = step(True)
    assert sim.get_death_cause().tolist() == plain.get_death_cause().tolist()
    assert sim.get_death_cause()[0, 0] != DEATH_NONE and sim.get_death_cause()[1, 0] == DEATH_NONE
    assert set(recorder.posts) == {0}
    post = recorder.posts[0]
    hero = post["snakes"][0]
    assert hero["is_hero"] and hero["traversed"] == [(100, 40)]
    assert post["frame"] == int(sim.frame[0])
    # The counterfactual reproduces the wall death from the pre-move hero.
    state = base.HeroState(((99, 40), (98, 40), (97, 40)), 3, 1, 0)
    meta = dict(META, grid_width=100, grid_height=80)
    others = [r for r in post["snakes"] if not r["is_hero"]]
    from research.trap_horizon_20261001 import diagnose_live_v5 as d5

    assert d5.one_frame_outcome(meta, state, 1, others, frozenset(), 1.15)["cause"] == "wall"
    assert d5.one_frame_outcome(meta, state, 0, others, frozenset(), 1.15)["cause"] == "alive"


# ---------------------------------------------------------------------------
# Trajectory and near-death metrics
# ---------------------------------------------------------------------------
def traj(horizon=10, death=None, masses=None):
    tr = {
        "alive_pre": np.ones(horizon, bool),
        "alive_post": np.ones(horizon, bool),
        "mass_post": np.asarray(masses if masses is not None else range(10, 10 + horizon)),
        "boosted": np.zeros(horizon, bool),
        "ate": np.zeros(horizon, np.int16),
        "kills": np.zeros(horizon, np.int16),
        "frames_since_food": np.arange(horizon, dtype=np.int32),
        "nearest_food": np.full(horizon, 5, np.int32),
        "food_count": np.full(horizon, 40, np.int32),
    }
    if death is not None:
        tr["alive_post"][death:] = False
        tr["alive_pre"][death + 1 :] = False
        tr["mass_post"] = np.where(tr["alive_post"], tr["mass_post"], 0)
        tr["nearest_food"][death + 1 :] = -1
    return tr


def test_trajectory_metrics_survivor_and_death():
    t = traj(10)
    t["ate"][[2, 5]] = 1
    t["boosted"][[1, 2, 3]] = True
    t["kills"][4] = 1
    m = c.trajectory_metrics(t, 10)
    assert m["mass_integral"] == pytest.approx(sum(range(10, 20)) / 10)
    assert not m["died"] and m["frames_alive"] == 10 and m["final_alive_mass"] == 19
    assert m["food_eaten"] == 2 and m["boost_frames"] == 3 and m["kills"] == 1
    assert m["late_growth_per_frame"] == pytest.approx((19 - 14) / 5)
    d = c.trajectory_metrics(traj(10, death=6), 10)
    assert d["died"] and d["death_frame_index"] == 6 and d["frames_alive"] == 6
    assert d["final_alive_mass"] == 15 and d["late_growth_per_frame"] is None
    assert d["mass_integral"] == pytest.approx(sum(range(10, 16)) / 10)
    shrink = traj(4, masses=[10, 9, 9, 12])
    assert c.trajectory_metrics(shrink, 4)["mass_lost_while_alive"] == 1


def snapshot(final=1, base_action=1, reason="kept", head_risky=False, boost_frames=0):
    return {
        "frame": 10,
        "body": np.asarray(ROW, dtype=np.int16),
        "length": len(ROW),
        "direction": 1,
        "boost_frames": boost_frames,
        "others": np.zeros((0, 2), np.int16),
        "food": np.zeros((0, 2), np.int16),
        "v2_counts": np.asarray([10, 10, 10]),
        "v2_need": 5,
        "base": base_action,
        "final": final,
        "q": np.zeros(6, np.float32),
        "mask": np.asarray([True] * 6),
        "v8_reason": c.REASONS.index(reason),
        "head_risky": head_risky,
        "head_kept_no_alternative": False,
        "head_waived": False,
        "apply_seconds": 0.001,
    }


def test_near_death_veto_and_window_round_trip():
    snaps = [snapshot() for _ in range(50)]
    snaps[20] = snapshot(final=0, base_action=1, reason="v7_rerank")
    snaps[-3] = snapshot(final=2, base_action=1, reason="head_veto", head_risky=True)
    snaps[-1] = snapshot(final=4, base_action=4, reason="no_spacious")
    out = c.near_death_veto(snaps)
    assert out["last_1"]["changed"] == 0 and out["last_1"]["no_spacious"] == 1
    assert out["last_10"]["changed"] == 1 and out["last_10"]["head_risky"] == 1
    assert out["last_10"]["changed_reasons"] == {"head_veto": 1}
    assert out["last_40"]["changed"] == 2 and out["decisions_since_last_change"] == 2
    assert out["fatal_boost"] and out["fatal_reason"] == "no_spacious"
    back = c.unpack_census_window(c.pack_census_window(snaps))
    assert [s["v8_reason"] for s in back] == [s["v8_reason"] for s in snaps]
    assert [s["head_risky"] for s in back] == [s["head_risky"] for s in snaps]
    assert back[-1]["final"] == 4 and back[0]["apply_seconds"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Death analysis
# ---------------------------------------------------------------------------
def test_cpu_deadline_turns_searches_unknown_and_restores():
    original = base.EscapeSearch
    world = base.StaticWorld(20, 12, frozenset(), frozenset())
    state = base.HeroState(tuple(ROW), 6, 1, 0)
    with c.cpu_deadline(-1.0):
        assert base.EscapeSearch is c.DeadlineEscapeSearch
        res = base.analyze_frame(
            world, state, actions=(1,), depth=40, budget=50000, criterion=base.DEPTH_ONLY
        )
        assert res[1].status == base.UNKNOWN
        assert c.DeadlineEscapeSearch.hits >= 1
    assert base.EscapeSearch is original
    with c.cpu_deadline(None):
        res = base.analyze_frame(
            world, state, actions=(1,), depth=10, budget=50000, criterion=base.DEPTH_ONLY
        )
        assert res[1].status == base.ESCAPE


def _self_death_window():
    """Hero in a 3x3 spiral: the window ends with a fatal move into its own body."""
    body = [(2, 1), (1, 1), (1, 2), (2, 2), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0)]
    first = snapshot()
    first.update(body=np.asarray(body, np.int16), length=len(body), direction=1, final=1, base=1)
    first["v2_counts"] = np.asarray([0, 0, 0])
    first["v8_reason"] = c.REASONS.index("no_spacious")
    first["frame"] = 10
    return [first]


def test_analyze_census_death_self_and_v8_view():
    meta = dict(META, grid_width=6, grid_height=4)
    snaps = _self_death_window()
    post = {
        "frame": 10,
        "food": [],
        "snakes": [{"is_hero": True}],
    }
    out = c.analyze_census_death(snaps, post, meta, "self", depth=6, budget=2000, cpu_seconds=30)
    assert out["fatal"]["taken_model_cause"] == "self"
    assert out["fatal_v8"]["v8_reason"] == "no_spacious"
    assert out["near_death"]["fatal_reason"] == "no_spacious"
    assert "walk" in out and out["category"] in (
        "boost_fatal_choice",
        "normal_fatal_choice",
        "taken_escaped",
        "unresolved",
    )
    assert out["trap_class"] in (
        "beyond_lookahead",
        "inside_lookahead",
        "no_count_pnr",
        "beyond_window",
        "unresolved",
    )
    assert "evaluated_frames" not in out and out["deadline_hit"] is False
    assert base.EscapeSearch is not c.DeadlineEscapeSearch


def test_trap_class():
    def death(status="exact", taken=base.NO_ESCAPE, view=None):
        walk = {"status": status, "taken_status": taken}
        return {"sensitivity": {base.COUNT_ONLY: {"walk": walk}}, "count_pnr_v8": view}

    spacious = {"taken_direction_spacious": True}
    tight = {"taken_direction_spacious": False}
    assert c.trap_class(death(view=spacious)) == "beyond_lookahead"
    assert c.trap_class(death(view=tight)) == "inside_lookahead"
    assert c.trap_class(death(view=None)) == "no_count_pnr"
    assert c.trap_class(death("beyond_window")) == "beyond_window"
    assert c.trap_class(death("unknown", view=spacious)) == "unresolved"
    assert c.trap_class(death(taken=base.UNKNOWN, view=spacious)) == "unresolved"


def test_analysis_errors_are_recorded_not_raised(tmp_path):
    job = (str(tmp_path / "missing.npz"), str(tmp_path / "p.json"), META, "self", 5, 100, 1.0)
    out = c.analyze_death_file(job)
    assert "error" in out and "traceback" in out
    ep = _episode("frozen", 1, True, "self", mass=7.0, death_t=6)
    s = c.summarize_census([ep], [{"mix": "frozen", "world_seed": 1, **out}])
    assert s["analysis"]["errors"][0]["world_seed"] == 1
    assert s["headroom_by_mode"]["self:unanalyzed"]["episodes"] == 1


# ---------------------------------------------------------------------------
# Headroom and summary
# ---------------------------------------------------------------------------
def test_death_headroom_and_mode():
    h = c.death_headroom(1000, 100, 5000, 0.02)
    assert h["remaining_frames"] == 4000
    assert h["hold"] == pytest.approx(80.0)
    assert h["grow"] == pytest.approx(80.0 + 0.02 * 4000 * 4000 / 10000)
    assert c.death_headroom(5000, 100, 5000, 1.0)["grow"] == 0.0
    ep = {"died": True, "death_cause": "head_on"}
    avoid = {"fatal": {"any_legal_alternative_avoids_with_escape": True}}
    assert c.death_mode(ep, avoid) == "head_on:avoidable"
    assert c.death_mode({"died": False}, None) == "survived"
    assert c.death_mode({"died": True, "death_cause": "self"}, {"category": "x"}) == "self:x"
    assert c.death_mode({"died": True, "death_cause": "wall"}, None) == "wall"


def _episode(mix, seed, died, cause=None, mass=100.0, death_t=None, growth=0.01):
    tr = c.trajectory_metrics(traj(10, death=death_t), 10)
    tr["late_growth_per_frame"] = None if died else growth
    return {
        "mix": mix,
        "world_seed": seed,
        "died": died,
        "death_cause": cause,
        "mass_integral": mass,
        "survival_fraction": 1.0 if not died else 0.5,
        "mass_at_death": tr["final_alive_mass"] if died else None,
        "trajectory": tr,
        "decision_metrics": c.decision_metrics(
            {
                "reason": [0, 7, 1],
                "changed": [False, True, False],
                "seconds": [0.001] * 3,
                "head_risky": [False, True, False],
                "boost": [False] * 3,
            }
        ),
    }


def _death(mix, seed, cause="head_on"):
    return {
        "mix": mix,
        "world_seed": seed,
        "cause": cause,
        "fatal": {
            "model_reproduces_cause": True,
            "any_legal_alternative_avoids": True,
            "any_legal_alternative_avoids_with_escape": True,
            "taken_hit_static_pre_move": False,
            "taken": 1,
        },
        "fatal_v8": {"v8_reason": "kept", "outcome": "kept", "head_risky": False},
        "fatal_length": 30,
        "fatal_frame": 6,
        "near_death": c.near_death_veto([snapshot()]),
        "model_check": {
            "transitions_compared": 3,
            "mismatches": 0,
            "fatal_action_model_outcome": "alive",
        },
        "deadline_hit": False,
        "cpu_seconds": 0.1,
    }


def test_summarize_census():
    episodes = [
        _episode("frozen", 1, True, "head_on", mass=7.0, death_t=6),
        _episode("frozen", 2, False, mass=14.5),
        _episode("scripted", 1, False, mass=14.5),
    ]
    deaths = [_death("frozen", 1)]
    lat = {
        "seconds": np.array([0.001, 0.002, 0.02]),
        "reason": np.array([0, 7, 7]),
        "head_risky": np.array([False, True, True]),
    }
    s = c.summarize_census(episodes, deaths, lat)
    assert s["episode_outcomes_by_mix"] == {
        "frozen": {"head_on": 1, "survived": 1},
        "scripted": {"survived": 1},
    }
    assert s["mass_integral"]["frozen"]["mean"] == pytest.approx(10.75)
    hr = s["headroom_by_mode"]["head_on:avoidable"]
    assert hr["episodes"] == 1
    # death at index 6 of 10 frames (frames 6..9 dead), mass 15: hold = 15 * 4 / 10 / 3
    assert hr["pooled_gain_hold"] == pytest.approx(15 * 4 / 10 / 3)
    assert s["fatal_frame_by_cause"]["head_on"]["any_legal_alternative_avoids_with_escape"] == 1
    assert s["decisions"]["by_reason"]["head_veto"] == 3
    assert s["veto_latency"]["over_16ms"] == 1
    assert s["veto_latency"]["by_reason"]["head_veto"]["n"] == 2
    assert "self" not in s


# ---------------------------------------------------------------------------
# Seeds, sharding, main guards, merge
# ---------------------------------------------------------------------------
def test_worlds_are_fresh_and_disjoint():
    seeds = c.world_seeds(c.DOMAIN, c.WORLDS_PER_MIX)
    assert len(set(seeds)) == c.WORLDS_PER_MIX
    report = c.disjointness(seeds, c.DOMAIN)
    assert report["disjoint"] is True, report["extra_overlaps"]
    checked = report["extra_checked"]
    assert any(n.startswith("apex-veto-v8-web-serving-v1/watch") for n in checked)
    assert any(n.startswith("trap-horizon-v5-dev-v1/worlds") for n in checked)
    assert any(n.startswith(f"{c.SMOKE_DOMAIN}/") for n in checked)
    assert not any(n.startswith(f"{c.DOMAIN}/") for n in checked)
    assert any(n.startswith("observed:") for n in checked)
    # A seed from an earlier bank is caught.
    from research.apex_safety_20260926 import dev_screen as ds

    stale = ds.uint32_seed("trap-horizon-v5-dev-v1", "worlds", 3)
    assert c.disjointness(seeds[:3] + [stale], c.DOMAIN)["disjoint"] is False


def test_shard_rows_partition():
    rows = [{"i": i} for i in range(10)]
    parts = [c.shard_rows(rows, k, 3) for k in (1, 2, 3)]
    assert sorted(r["i"] for p in parts for r in p) == list(range(10))
    assert [r["i"] for r in parts[1]] == [1, 4, 7]


def test_main_refusals(runners_fatal, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "on_ac_power", lambda: True)
    monkeypatch.setattr(c, "lid_open", lambda: True)
    root = ["--root", str(tmp_path / "root")]
    assert c.main(["run", "--shard", "4"] + root) == 2
    assert c.main(["run", "--shard", "1", "--frames", "10"] + root) == 2
    assert c.main(["run", "--shard", "1", "--worlds-per-mix", "2"] + root) == 2
    assert c.main(["run", "--shard", "1", "--out", str(tmp_path / "x")] + root) == 2
    assert c.main(["run", "--shard", "1", "--smoke"] + root) == 2  # smoke needs --out
    (tmp_path / "root" / "shard-1").mkdir(parents=True)
    assert c.main(["run", "--shard", "1"] + root) == 2  # exists
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert c.main(["run", "--shard", "2"] + root) == 2
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    monkeypatch.setattr(c, "on_ac_power", lambda: False)
    assert c.main(["run", "--shard", "2"] + root) == 2
    monkeypatch.setattr(c, "on_ac_power", lambda: True)
    monkeypatch.setattr(c, "lid_open", lambda: False)
    assert c.main(["run", "--shard", "2"] + root) == 2
    assert not (tmp_path / "root" / "shard-2").exists()


def _write_shard(root, k, episodes, deaths, intent_over=None):
    d = root / f"shard-{k}"
    (d / "deaths").mkdir(parents=True)
    (d / "trajectories").mkdir()
    intent = {
        "shard": k,
        "smoke": True,
        "worlds": {"domain": c.SMOKE_DOMAIN, "seeds": [1, 2], "shared_by_mixes": ["frozen"]},
        "git": {"commit": "abc", "dirty_paths": ""},
        "profile": {"h": 10},
        "veto": {"v": 8},
        "engine": {"e": "simd"},
        "search": {"depth": 40},
        "capture": {"w": 120},
        "config": {"sha256": "x"},
        "hero": {"sha256": "h"},
    }
    intent.update(intent_over or {})
    (d / "intent.json").write_text(json.dumps(intent))
    (d / "episodes.json").write_text(json.dumps(episodes, default=float))
    for death in deaths:
        (d / "deaths" / f"{death['mix']}-{death['world_seed']}.json").write_text(json.dumps(death))
    np.savez(
        d / "trajectories" / f"frozen-{k}.npz",
        dec_seconds=np.array([0.001]),
        dec_reason=np.array([0]),
        dec_head_risky=np.array([False]),
    )
    (d / "summary.json").write_text(json.dumps({"timing": {"t": 1}, "compute": {"slot": k}}))


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, "item") else v))


def test_merge(tmp_path):
    root = tmp_path / "run"
    e1 = _jsonable(_episode("frozen", 1, True, "head_on", mass=7.0, death_t=6))
    e2 = _jsonable(_episode("frozen", 2, False, mass=14.5))
    _write_shard(root, 1, [e1], [_jsonable(_death("frozen", 1))])
    _write_shard(root, 2, [e2], [])
    assert c.main(["merge", "--root", str(root), "--shards", "2"]) == 2  # smoke flag mismatch
    assert c.main(["merge", "--root", str(root), "--shards", "2", "--smoke"]) == 0
    s = json.loads((root / "merged" / "summary.json").read_text())
    assert s["episodes"] == 2 and s["veto_latency"]["all"]["n"] == 2
    assert c.main(["merge", "--root", str(root), "--shards", "2", "--smoke"]) == 2  # exists
    bad = tmp_path / "bad"
    _write_shard(bad, 1, [e1], [])  # death file missing
    _write_shard(bad, 2, [e2], [])
    assert c.main(["merge", "--root", str(bad), "--shards", "2", "--smoke"]) == 2
    other = tmp_path / "other"
    _write_shard(other, 1, [e1], [_jsonable(_death("frozen", 1))])
    _write_shard(other, 2, [e2], [], {"git": {"commit": "zzz", "dirty_paths": ""}})
    assert c.main(["merge", "--root", str(other), "--shards", "2", "--smoke"]) == 2
    knob = tmp_path / "knob"
    _write_shard(knob, 1, [e1], [_jsonable(_death("frozen", 1))])
    _write_shard(knob, 2, [e2], [], {"search": {"depth": 10}})
    assert c.main(["merge", "--root", str(knob), "--shards", "2", "--smoke"]) == 2


# ---------------------------------------------------------------------------
# Tiny-world integration (plays episodes; opt-in)
# ---------------------------------------------------------------------------
tiny = pytest.mark.skipif(
    os.environ.get("SNAKE_CENSUS_TINY_ROLLOUT") != "1",
    reason="plays tiny SIMD episodes; set SNAKE_CENSUS_TINY_ROLLOUT=1",
)


@tiny
def test_tiny_world_capture_matches_records_and_model(setup_config, tmp_path):
    from src.core.config_loader import load_and_initialize_config
    from src.model.apex_network import ApexNetwork
    from src.simd_env import eval_engine as ee
    from tests.test_simd_vector61_policy import TINY_YAML, _profile, _tiny_rosters

    cfg = tmp_path / "tiny_v61.yaml"
    cfg.write_text(TINY_YAML)
    load_and_initialize_config(str(cfg))

    def make(name, seed):
        torch.manual_seed(seed)
        net = ApexNetwork(input_size=61, hidden_size=32, output_size=6)
        with torch.no_grad():
            for param in net.parameters():
                param.mul_(3.0)
        path = tmp_path / f"{name}.pth"
        torch.save({"dqn_state_dict": net.state_dict(), "input_size": 61, "hidden_size": 32}, path)
        return ("checkpoint", str(path))

    seeds = tuple(range(3, 15))
    frames = 1500
    hero, rosters = _tiny_rosters(make, seeds)
    kwargs = dict(
        profile=_profile(frames),
        opponent_specs_by_world=rosters,
        mix_id="tiny",
        vector61=True,
        hero_safety_veto="v8",
        hero_safety_veto_lambda=8.0,
    )
    plain = ee.run_simd_eval(hero, rosters[seeds[0]], frames, list(seeds), **kwargs)
    recorder = c.CensusRecorder(len(seeds), frames)
    with c.census_capture(recorder):
        records = ee.run_simd_eval(
            hero,
            rosters[seeds[0]],
            frames,
            list(seeds),
            frame_observer=recorder.observe_frame,
            **kwargs,
        )
    deaths = 0
    for env, (a, b) in enumerate(zip(plain, records)):
        assert {k: v for k, v in a.items() if k != "veto_diagnostics"} == {
            k: v for k, v in b.items() if k != "veto_diagnostics"
        }
        assert recorder.decision_count(env) == b["probes"]["safety_veto"]["counters"]["decisions"]
        tr = {k: v[env] for k, v in recorder.traj.items()}
        m = c.trajectory_metrics(tr, frames)
        assert m["mass_integral"] == pytest.approx(b["mass_integral"], abs=1e-9)
        if float(b["deaths"]) >= 1:
            deaths += 1
            snaps = list(recorder.buffers[env])
            post = recorder.posts[env]
            assert post["frame"] == snaps[-1]["frame"]
            cause = b["probes"]["death_cause"]
            meta = recorder.world_meta
            fatal = c.d5.fatal_frame_analysis(snaps[-1], post, meta, cause, 10, 2000)
            assert fatal["model_reproduces_cause"], (env, cause, fatal["taken_model_cause"])
            assert base.model_replay_check(snaps, meta)["mismatches"] == 0
        else:
            assert env not in recorder.posts
    assert deaths >= 1
