from __future__ import annotations

import json

import numpy as np

from research.bl_density.runner import (
    _inject_p1_long,
    _root_geometry,
    cheap_prefilter,
    effective_q_features,
    legal_alternatives,
    retrospective_doom,
)


def test_cheap_prefilter_boundaries() -> None:
    assert cheap_prefilter(np.array([159, 500, 500]), np.ones(6, dtype=bool))
    assert not cheap_prefilter(
        np.array([160, 500, 500]), np.array([True, True, True, False, False, False])
    )
    assert cheap_prefilter(
        np.array([160, 500, 500]), np.array([True, True, False, True, True, True])
    )


def test_retrospective_doom_window_is_transition_exclusive_at_300() -> None:
    assert retrospective_doom(20, 20)
    assert retrospective_doom(319, 20)
    assert not retrospective_doom(320, 20)
    assert not retrospective_doom(19, 20)


def test_q_fallback_uses_normal_actions_and_single_action_gap_zero() -> None:
    action, values = effective_q_features(np.array([1, 4, 3, 99, 98, 97]), np.zeros(6, dtype=bool))
    assert action == 1 and values[0] == 1.0
    action, values = effective_q_features(
        np.array([1, 4, 3, 99, 98, 97]), np.array([False, True, False, False, False, False])
    )
    assert action == 1 and values[0] == 0.0


def test_counterfactuals_use_full_legal_domain_not_the_advisory_mask() -> None:
    legal = np.array([True, True, True, False, False, False])
    advisory = np.array([True, False, False, False, False, False])
    assert legal_alternatives(legal, 0) == (1, 2)
    # The two advisory-rejected moves remain eligible rescue alternatives.
    assert not advisory[1] and not advisory[2]


def test_frozen_pool_rotates_across_the_whole_cell() -> None:
    from pathlib import Path

    from research.apex_safety_20260926 import dev_screen as ds
    from research.bl_density.runner import planned_jobs
    from research.t3_rollout_labels_20261015.bank_engine import planned_rosters

    context = {"ds": ds, "lookup": ds.agent_lookup({sha: name for name, sha in ds.POOL})}
    jobs = planned_jobs(
        context,
        b8=Path("b8.pth"),
        arms=["A"],
        mixes=["frozen"],
        episodes=6,
        horizon=5000,
        tiny=False,
        deadline=None,
    )
    seeds = [job["plan"]["seed"] for job in jobs]
    expected = planned_rosters(context, seeds, ["frozen"] * len(seeds))
    assert [job["plan"] for job in jobs] == expected
    assert len({tuple(tuple(spec) for spec in job["plan"]["specs"]) for job in jobs}) > 1


def test_density_two_production_allocation_is_492_and_a_is_shared_reference() -> None:
    from pathlib import Path

    from research.apex_safety_20260926 import dev_screen as ds
    from research.bl_density.runner import ARMS, DENSITY2_ALLOCATION, MIXES, planned_jobs

    context = {"ds": ds, "lookup": ds.agent_lookup({sha: name for name, sha in ds.POOL})}
    jobs = planned_jobs(
        context,
        b8=Path("b8.pth"),
        arms=ARMS,
        mixes=MIXES,
        episodes=6,
        horizon=5000,
        tiny=False,
        deadline=None,
    )
    counts = {
        (arm, mix): sum(j["arm"] == arm and j["mix"] == mix for j in jobs)
        for arm in ARMS
        for mix in MIXES
    }
    assert len(jobs) == 492
    assert counts["A", "frozen"] == 96
    assert counts["A", "mixed"] == counts["A", "scripted"] == 72
    assert all(counts["B70m", mix] == 24 for mix in MIXES)
    assert all(counts["B70p", mix] == 36 for mix in MIXES)
    assert all(counts["C-P1", mix] == 24 for mix in MIXES)
    assert DENSITY2_ALLOCATION["A"]["frozen"] == 96
    assert {job["arm"] for job in jobs} == {"A", "B70m", "B70p", "C-P1"}


def test_density_two_namespaces_are_disjoint_from_all_pilot_worlds() -> None:
    from research.bl_density.runner import (
        CANARY_NAMESPACE,
        DENSITY2_ALLOCATION,
        DENSITY2_NAMESPACE,
        LOCAL_PREFLIGHT_NAMESPACE,
        MIXES,
        PILOT_ALLOCATION,
        PILOT_NAMESPACE,
        allocation_seeds,
        assert_density2_seed_separation,
    )

    pilot = allocation_seeds(PILOT_NAMESPACE, PILOT_ALLOCATION)
    production = allocation_seeds(DENSITY2_NAMESPACE, DENSITY2_ALLOCATION)
    canary = allocation_seeds(
        CANARY_NAMESPACE, {arm: {mix: 1 for mix in MIXES} for arm in DENSITY2_ALLOCATION}
    )
    preflight = allocation_seeds(
        LOCAL_PREFLIGHT_NAMESPACE, {arm: {mix: 1 for mix in MIXES} for arm in DENSITY2_ALLOCATION}
    )
    assert len(pilot) == 90
    assert len(production) == 492
    assert len(canary) == 12
    assert len(preflight) == 12
    assert not (pilot & production)
    assert not (pilot & canary)
    assert not (production & canary)
    assert not (pilot & preflight)
    assert not (production & preflight)
    assert not (canary & preflight)
    assert_density2_seed_separation()


def test_verified_skip_ledger_removes_only_frozen_allocation_ids(tmp_path: object) -> None:
    from pathlib import Path

    from research.bl_density.ledger import SCHEMA
    from research.bl_density.runner import NAMESPACE, _load_skip_ledger, _seed

    path = Path(str(tmp_path)) / "ledger.json"
    jobs = [
        {
            "namespace": NAMESPACE,
            "arm": "A",
            "mix": "scripted",
            "index": 0,
            "plan": {"seed": _seed("A", "scripted", 0)},
        },
        {
            "namespace": NAMESPACE,
            "arm": "A",
            "mix": "scripted",
            "index": 1,
            "plan": {"seed": _seed("A", "scripted", 1)},
        },
    ]
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "namespace": NAMESPACE,
                "entries": [
                    {
                        "episode_id": f"{NAMESPACE}/A/scripted/000",
                        "sha256": "a" * 64,
                        "size": 1,
                    }
                ],
            }
        )
    )
    skipped, _ = _load_skip_ledger(path, namespace=NAMESPACE, jobs=jobs)
    assert skipped == {f"{NAMESPACE}/A/scripted/000"}
    assert jobs[0]["plan"]["seed"] == _seed("A", "scripted", 0)


def test_skip_ledger_rejects_foreign_or_duplicate_ids(tmp_path: object) -> None:
    from pathlib import Path

    import pytest

    from research.bl_density.ledger import SCHEMA
    from research.bl_density.runner import NAMESPACE, _load_skip_ledger

    path = Path(str(tmp_path)) / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "namespace": NAMESPACE,
                "entries": [
                    {
                        "episode_id": f"{NAMESPACE}/A/scripted/999",
                        "sha256": "a" * 64,
                        "size": 1,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="out-of-allocation"):
        _load_skip_ledger(path, namespace=NAMESPACE, jobs=[])


def test_allocation_minus_408_entry_ledger_is_exactly_84_and_preserves_seeds(
    tmp_path: object,
) -> None:
    from pathlib import Path

    from research.bl_density.ledger import SCHEMA
    from research.bl_density.runner import DENSITY2_ALLOCATION, NAMESPACE, _load_skip_ledger, _seed

    jobs = [
        {
            "namespace": NAMESPACE,
            "arm": arm,
            "mix": mix,
            "index": index,
            "plan": {"seed": _seed(arm, mix, index)},
        }
        for arm, by_mix in DENSITY2_ALLOCATION.items()
        for mix, count in by_mix.items()
        for index in range(count)
    ]
    missing = {
        f"{NAMESPACE}/C-P1/{mix}/{index:03d}"
        for mix in ("frozen", "mixed", "scripted")
        for index in range(24)
    } | {f"{NAMESPACE}/B70p/mixed/{index:03d}" for index in (22, *range(25, 36))}
    ledger = {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "entries": [
            {"episode_id": _id, "sha256": "a" * 64, "size": 1}
            for _id in sorted(
                {f"{job['namespace']}/{job['arm']}/{job['mix']}/{job['index']:03d}" for job in jobs}
                - missing
            )
        ],
    }
    path = Path(str(tmp_path)) / "real-shape-ledger.json"
    path.write_text(json.dumps(ledger))
    skipped, _ = _load_skip_ledger(path, namespace=NAMESPACE, jobs=jobs)
    remaining = [
        job
        for job in jobs
        if f"{job['namespace']}/{job['arm']}/{job['mix']}/{job['index']:03d}" not in skipped
    ]
    assert len(skipped) == 408
    assert len(remaining) == 84
    assert {
        f"{job['namespace']}/{job['arm']}/{job['mix']}/{job['index']:03d}" for job in remaining
    } == missing
    assert _seed("B70p", "mixed", 22) == remaining[0]["plan"]["seed"]


def test_partial_records_are_never_exported_to_ledger_eligible_directories() -> None:
    from research.bl_density.runner import _export_group

    assert _export_group("episodes", {"complete": True}) == "episodes"
    assert _export_group("episodes", {"complete": False}) == "episodes-partial"
    assert _export_group("episodes-replay", {"complete": False}) == "episodes-replay-partial"


def test_terminal_a_floor_closes_rows_9_and_9a_without_extending_incomplete_runs() -> None:
    from research.bl_density.runner import _densification_closure

    analysis = {
        "cells": {
            "A": {
                "scripted": {"distinct_doom_bearing_episodes": 3},
                "frozen": {"distinct_doom_bearing_episodes": 4},
                "mixed": {"distinct_doom_bearing_episodes": 1},
            }
        }
    }
    terminal = _densification_closure(analysis, "COMPLETE", production=True)
    assert terminal["scripted"] == {
        "status": "CLOSED",
        "rows": ["9", "9a"],
        "reason": "A_fewer_than_4_doom_bearing_episodes",
        "action": "NO_THIRD_ENLARGEMENT",
    }
    assert terminal["frozen"]["status"] == "OPEN"
    incomplete = _densification_closure(analysis, "INCOMPLETE", production=True)
    assert all(
        row["action"] == "PRESERVE_COLLECTED_EPISODES_NO_EXTENSION" for row in incomplete.values()
    )
    excluded = _densification_closure(analysis, "COMPLETE", production=False)
    assert all(row["status"] == "NOT_APPLICABLE_EXCLUDED_PREFLIGHT" for row in excluded.values())


class _RootProbe:
    S = 6
    E = 1
    grid_w = 121
    grid_h = 69
    cap = 3000

    def __init__(self) -> None:
        self.direction = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int64)
        self.length = np.array([[1200, 600, 400, 300, 150, 150]], dtype=np.int64)
        self.seg_count = self.length.copy()
        self.alive = np.ones((1, self.S), dtype=bool)
        self.head_ptr = self.seg_count - 1
        self.bodies = np.zeros((1, self.S, self.cap, 2), dtype=np.int64)
        for slot, count in enumerate(self.length[0]):
            for offset in range(int(count)):
                self.bodies[0, slot, int(count) - 1 - offset] = [offset + 2000 * slot, 0]

    def heads(self) -> np.ndarray:
        return np.stack([self.bodies[0, slot, self.head_ptr[0, slot]] for slot in range(self.S)])[
            None
        ]


def test_minimum_map_diagnostic_rejects_known_8l_violation_without_losing_root_geometry() -> None:
    row = _root_geometry(_RootProbe(), "B70m")
    assert row["all_snake_lengths"] == [1200, 600, 400, 300, 150, 150]
    assert row["map_cells"] == 8349
    assert row["total_occupied_cells"] == 2800
    assert row["minimum_map_assessment"]["known_floor_cells"] == 9600
    assert not row["minimum_map_assessment"]["source_eligible"]
    assert row["minimum_map_assessment"]["transferability_status"] == "DIAGNOSTIC_RHO95_UNMEASURED"


def test_p1_fixture_delegates_900_and_preserves_the_registered_other_lengths(
    monkeypatch: object,
) -> None:
    import research.bl_ppo.collector as p1_collector
    import src.simd_env.grid_scenarios as scenarios

    calls: list[tuple[str, object]] = []

    class _Sentinel:
        grid_w = 145

    sentinel = _Sentinel()

    def shared(sim: object) -> None:
        calls.append(("shared", sim))

    def inject(sims: object, lengths: object, box_height: int = 6) -> None:
        calls.append(("primitive", (sims, lengths, box_height)))

    monkeypatch.setattr(p1_collector, "_inject_long", shared)
    monkeypatch.setattr(scenarios, "inject_serpentines", inject)
    _inject_p1_long(sentinel, 900)
    _inject_p1_long(sentinel, 1200)
    assert calls == [
        ("shared", sentinel),
        ("primitive", ([sentinel], (1200, 600, 400, 300, 150, 150), 9)),
    ]


def test_p1_fixture_all_registered_strata_construct_without_a_simulation_step() -> None:
    from research.bl_ppo.collector import _inject_long, _sim_config
    from src.simd_env.grid_sim import GridBatchSim

    def fresh() -> GridBatchSim:
        return GridBatchSim(_sim_config(1), seeds=[17], train_mode=False, allow_respawn=True)

    shared, local = fresh(), fresh()
    _inject_long(shared)
    _inject_p1_long(local, 900)
    for name in ("bodies", "alive", "seg_count", "length", "head_ptr", "direction"):
        assert np.array_equal(getattr(local, name), getattr(shared, name))
    for hero, expected_height in ((900, 8), (1200, 9), (1500, 11)):
        sim = fresh()
        _inject_p1_long(sim, hero)
        assert sim.length[0].tolist() == [hero, 600, 400, 300, 150, 150]
        assert int(sim.head_ptr[0, 0]) == hero - 1
        assert max(8, int(np.ceil(hero / (sim.grid_w - 4)))) == expected_height


def test_dead_opponent_geometry_and_rounded_b70_floor() -> None:
    probe = _RootProbe()
    probe.length[0, 0] = 1000
    probe.seg_count[0, 0] = 1000
    probe.head_ptr[0, 0] = 999
    probe.alive[0, 1:] = False
    row = _root_geometry(probe, "B70m")
    assert row["total_occupied_cells"] == 1000
    assert row["minimum_map_assessment"]["known_floor_cells"] == 8349
    assert row["minimum_map_assessment"]["source_eligible"]
    assert row["nearest_opponent_head_manhattan"] is None
    assert row["nearest_opponent_body_manhattan"] is None
    assert row["all_snake_alive"] == [True, False, False, False, False, False]


def test_continuation_diagnostics_do_not_alias_original_or_sibling(monkeypatch) -> None:
    import copy
    from types import SimpleNamespace

    from research.bl_density import runner
    from research.t3_rollout_labels_20261015 import bank_engine

    class World:
        def __init__(self):
            self.hero = SimpleNamespace(live_vetoes={"n": 0})
            self.opponents = [SimpleNamespace(last_veto={"n": 0})]
            self.sim = SimpleNamespace(alive=np.ones((1, 1), dtype=bool))

        def clone(self):
            branch = copy.copy(self)
            branch.hero = copy.copy(self.hero)
            branch.opponents = [copy.copy(self.opponents[0])]
            branch.sim = SimpleNamespace(alive=np.ones((1, 1), dtype=bool))
            return branch

    observed = []

    def step(branch, action):
        branch.hero.live_vetoes["n"] += 1
        branch.opponents[0].last_veto["n"] += 1
        observed.append(branch.hero.live_vetoes["n"])
        branch.sim.alive[0, 0] = False

    monkeypatch.setattr(bank_engine, "step_prepared", step)
    monkeypatch.setattr(runner, "_death_cause_label", lambda sim: "DEATH_SELF")
    world = World()
    runner._rollout_continuation(world, 0, 0, 0)
    runner._rollout_continuation(world, 1, 0, 0)
    assert observed == [1, 1]
    assert world.hero.live_vetoes == {"n": 0}
    assert world.opponents[0].last_veto == {"n": 0}


def test_semantic_parity_excludes_stage_cpu_measurements() -> None:
    from research.bl_density.verify import _canonical

    left = {"arm": "A", "cpu_seconds_by_stage": {"setup": 1.0, "screen": 2.0}}
    right = {"arm": "A", "cpu_seconds_by_stage": {"setup": 4.0, "screen": 8.0}}
    assert _canonical(left) == _canonical(right)
