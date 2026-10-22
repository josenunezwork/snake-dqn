from __future__ import annotations

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


def test_amendment_three_production_allocation_is_90_and_a_is_shared_reference() -> None:
    from pathlib import Path

    from research.apex_safety_20260926 import dev_screen as ds
    from research.bl_density.runner import ARMS, MIXES, planned_jobs

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
    assert len(jobs) == 90
    assert all(counts["A", mix] == 12 for mix in MIXES)
    assert all(counts[arm, mix] == 6 for arm in ARMS if arm != "A" for mix in MIXES)
    assert {job["arm"] for job in jobs} == {"A", "B70m", "B70p", "C-P1"}


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
