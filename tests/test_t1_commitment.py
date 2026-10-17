"""Exact replay, full-state clone isolation and intervention endpoint checks."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.t1_commitment_20261015.parity import compare_world
from research.t1_commitment_20261015.replay import (
    Replay,
    _hash_value,
    assert_same_world,
    build_replay,
    configure_cpu,
    load_inputs,
)
from research.t1_commitment_20261015.study import coarse_decisions, rollout
from tests.t14_fixtures import evaluator_context


@pytest.fixture
def synthetic_census(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Provide a recorded evaluator trace without historical Mac artifacts."""
    from research.b8_death_census_20261014 import census

    context, student = evaluator_context(tmp_path)
    monkeypatch.setattr(census, "b8_path", lambda _arm: student)
    seed = 100
    replay = build_replay(
        "b8_s0",
        [0],
        (context, [seed], {seed: [("scripted", "greedy_food")] * 5}),
    )
    initial_body = replay.sim.bodies[0, 0].copy()
    actions = np.zeros((1, 320), dtype=np.int64)
    lengths = np.zeros((1, 320), dtype=np.int64)
    for step in range(actions.shape[1]):
        lengths[0, step] = int(replay.sim.length[0, 0]) if replay.sim.alive[0, 0] else 0
        actions[0, step] = replay.actions()[0, 0]
        replay.step(actions[:, step])
    assert np.all(
        lengths[0, :300] > 0
    ), "fixture must exercise live hero replay through clone parity"
    assert not np.array_equal(initial_body, replay.sim.bodies[0, 0])
    return context, {"action": actions, "length": lengths}


@pytest.fixture
def census_replay(synthetic_census) -> Replay:
    configure_cpu()
    context, _ = synthetic_census
    return build_replay("b8_s0", [0], (context, [100], {100: [("scripted", "greedy_food")] * 5}))


def test_short_recorded_replay_clone_parity(
    monkeypatch: pytest.MonkeyPatch, synthetic_census
) -> None:
    """One real census world, 200 full-state clone comparisons plus lengths."""
    configure_cpu()
    from research.t1_commitment_20261015 import parity

    context, series = synthetic_census
    monkeypatch.setattr(parity, "load_inputs", lambda _arm: ({}, series))
    receipt = compare_world(
        0,
        warmup=100,
        frames=200,
        context=(context, [100], {100: [("scripted", "greedy_food")] * 5}),
    )
    assert receipt["identical"]
    assert receipt["full_digest_comparisons"] == 201
    assert receipt["length_values_compared"] == 500


def test_prepared_clone_executes_exactly_one_transition(
    census_replay: Replay, synthetic_census
) -> None:
    original = census_replay.clone()
    _, series = synthetic_census
    captured = []

    def capture(r: Replay) -> None:
        captured.append(r.clone(prepared=True))

    action = int(series["action"][0, 0])
    original.step([action], capture)
    branch = captured[0]
    assert int(branch.sim.frame[0]) == 1
    branch.first(action)
    assert int(branch.sim.frame[0]) == 1
    assert_same_world(original, branch)
    original.step([int(series["action"][0, 1])])
    branch.step([int(series["action"][0, 1])])
    assert_same_world(original, branch)


@pytest.mark.parametrize("field", ["bodies", "_owner_pad", "respawn_timer"])
def test_full_digest_catches_raw_field_changes(census_replay: Replay, field: str) -> None:
    left, right = census_replay.clone(), census_replay.clone()
    getattr(right.sim, field).flat[-1] += 1
    with pytest.raises(AssertionError, match=field):
        assert_same_world(left, right)


def test_full_digest_catches_rng_and_opponent_state(census_replay: Replay) -> None:
    left, right = census_replay.clone(), census_replay.clone()
    right.sim._rngs[0]._rng.random()
    with pytest.raises(AssertionError, match="_rngs"):
        assert_same_world(left, right)
    # Do not mutate shared policies: substitute an independent configuration.
    right = census_replay.clone()
    right.opponents = [[SimpleNamespace(kind="changed")]]
    with pytest.raises(AssertionError, match="opponent_policies"):
        assert_same_world(left, right)


def test_object_array_digest_hashes_values_not_pointers() -> None:
    def digest(value: np.ndarray) -> str:
        h = hashlib.sha256()
        _hash_value(h, value)
        return h.hexdigest()

    a = np.empty(2, object)
    b = np.empty(2, object)
    a[:] = [[1, 2], [3]]
    b[:] = [[1, 2], [3]]
    assert digest(a) == digest(b)
    b[0].append(9)
    assert digest(a) != digest(b)


class FakePrepared:
    """Deterministic endpoint fixture with death after a fixed transition count."""

    def __init__(self, dies_after: int) -> None:
        self.dies_after = dies_after
        self.env_frames = 0
        self.sim = SimpleNamespace(alive=np.ones((1, 1), bool))

    def clone(self, prepared: bool = False) -> "FakePrepared":
        return FakePrepared(self.dies_after)

    def first(self, action: int) -> None:
        self.step()

    def step(self) -> None:
        self.env_frames += 1
        self.sim.alive[0, 0] = self.env_frames < self.dies_after


def test_control_extends_one_transition_without_changing_300_endpoint() -> None:
    result = rollout(FakePrepared(301), action=1, decision=0, fatal=300)
    assert result["survives"]
    assert result["frames_executed"] == 300
    assert result["frames_survived"] == 300
    assert result["death_decision"] == 300
    assert result["env_frames"] == 301
    assert result["control_reproduced"]


def test_bad_recorded_control_fails_closed() -> None:
    with pytest.raises(AssertionError, match="control death mismatch"):
        rollout(FakePrepared(20), action=1, decision=0, fatal=25)


def test_endpoint_survival_requires_alive_after_last_transition() -> None:
    assert not rollout(FakePrepared(300), 1, decision=0)["survives"]
    assert rollout(FakePrepared(301), 1, decision=0)["survives"]
    truncated = rollout(FakePrepared(301), 4, decision=4900)
    assert truncated["horizon"] == 100
    assert truncated["frames_executed"] == 100
    assert truncated["survives"]


@pytest.mark.parametrize("fatal", [255, 295, 699, 4362])
def test_scan_contains_start_and_last_coarse_decision(fatal: int) -> None:
    decisions = coarse_decisions(fatal)
    assert decisions[0] == max(0, fatal - 300)
    assert decisions[-1] == fatal - 10
    assert len(decisions) == len(set(decisions))
    assert all(0 <= d < fatal for d in decisions)


@pytest.mark.parametrize("coarse_positive", [False, True])
def test_onset_positive_can_be_latest_lad_and_region_includes_fatal(
    monkeypatch: pytest.MonkeyPatch, coarse_positive: bool
) -> None:
    from research.t1_commitment_20261015 import study, topology

    fatal = 80

    class RecordedWorld:
        def __init__(self) -> None:
            self.sim = SimpleNamespace(
                frame=np.array([0]), alive=np.ones((1, 1), bool), length=np.ones((1, 1), int)
            )
            self.env_frames = 0
            self.kinds = [["scripted:greedy_food"]]

        def clone(self, prepared: bool = False) -> "RecordedWorld":
            return copy.deepcopy(self)

        def first(self, action: int) -> None:
            self.env_frames += 1
            self.sim.alive[0, 0] = int(self.sim.frame[0]) <= fatal

        def step(self, forced=None, observer=None) -> None:
            self.sim.frame[0] += 1
            if observer is not None:
                observer(self)
            self.first(1)

    def measured(replay, chosen, decision, fatal):
        positive = decision in (76, 77) or (coarse_positive and decision == 10)
        return {
            "decision": decision,
            "chosen_action": chosen,
            "surviving_actions": [2] if positive else [],
            "outcomes": {str(chosen): {"env_frames": 1, "control_reproduced": True}},
        }

    monkeypatch.setattr(study, "build_replay", lambda *a: RecordedWorld())
    monkeypatch.setattr(study, "evaluate_decision", measured)
    monkeypatch.setattr(study, "add_geometry", lambda replay, row: dict(row))
    monkeypatch.setattr(topology, "head_region", lambda sim: {"component_size": 1})
    monkeypatch.setattr(topology, "sealer_identity", lambda *a: {"owner": "none"})
    death = {
        "env": 0,
        "fatal_t": fatal,
        "frame": fatal + 1,
        "length": 1,
        "trap_onset": {"t0": 76},
        "world_seed": 123,
    }
    series = {"action": np.ones((1, fatal + 1), int), "length": np.ones((1, fatal + 1), int)}
    row = study.study_death("b8_s0", death, series, None)
    assert row["lad"]["decision"] == 77
    assert row["H_LAD"] == 3
    assert len(row["region_series"]) == 60
    assert row["region_series"][0]["decision"] == 21
    assert row["region_series"][-1]["decision"] == fatal


@pytest.mark.parametrize(
    "horizons,eligible,off",
    [
        ([10] * 20 + [130] * 10 + [None] * 19, False, False),
        ([10] * 20 + [None] * 29, False, True),
        ([10] * 30 + [None] * 19, True, False),
    ],
)
def test_routing_keeps_censored_deaths_in_late_median(horizons, eligible, off) -> None:
    from research.t1_commitment_20261015.report import censored_quartiles, fraction, route

    late = {
        "H_LAD_quartiles_censored": censored_quartiles([{"H_LAD": h} for h in horizons]),
        "survivor_within": {
            str(h): fraction(sum(x is not None and x <= h for x in horizons), 49)
            for h in (150, 300)
        },
        "boost_survives_at_LAD": fraction(0, 49),
    }
    overall = {
        "Q_gap_below_2": fraction(0, 1),
        "Q_gap_above_5": fraction(0, 1),
        "tail_or_boundary_TTL_diff": fraction(0, 1),
    }
    rules = route({"late": late, "all": overall})
    assert rules["R1"]["fires"] is eligible
    assert rules["R1_prime"]["fires"] is off


def test_sealer_backups_reject_other_runs_and_unknown_rows(tmp_path: Path) -> None:
    from research.t1_commitment_20261015.sealer_refresh import bind_original_rows

    (tmp_path / "rows").mkdir()
    canonical = tmp_path / "rows/b8_s0.jsonl"
    canonical.write_text('{"outcome":"first run"}\n')
    run = tmp_path / "run.json"
    run.write_text('{"run":"first"}\n')
    backup, binding = bind_original_rows("b8_s0", tmp_path)
    original = backup.read_bytes()
    assert canonical.read_bytes() == original
    assert bind_original_rows("b8_s0", tmp_path)[1] == binding
    canonical.write_text('{"outcome":"unrelated rows"}\n')
    with pytest.raises(AssertionError, match="canonical rows differ"):
        bind_original_rows("b8_s0", tmp_path)
    canonical.write_bytes(original)
    run.write_text('{"run":"second"}\n')
    with pytest.raises(AssertionError, match="different run"):
        bind_original_rows("b8_s0", tmp_path)
    assert backup.read_bytes() == original


def test_sealer_backup_allows_known_refresh_without_replacing_original(tmp_path: Path) -> None:
    from research.b8_death_census_20261014.census import sha256_file
    from research.t1_commitment_20261015.sealer_refresh import bind_original_rows

    (tmp_path / "rows").mkdir()
    (tmp_path / "sealer-review").mkdir()
    canonical = tmp_path / "rows/b8_s0.jsonl"
    canonical.write_text('{"outcome":"immutable intervention"}\n')
    (tmp_path / "run.json").write_text('{"run":"first"}\n')
    backup, binding = bind_original_rows("b8_s0", tmp_path)
    original = backup.read_bytes()
    canonical.write_text('{"outcome":"immutable intervention","sealer":"reviewed"}\n')
    (tmp_path / "sealer-review/b8_s0.json").write_text(
        json.dumps(
            {
                **binding,
                "reviewed_output_sha256": sha256_file(canonical),
            }
        )
    )
    assert bind_original_rows("b8_s0", tmp_path)[0].read_bytes() == original
    backup.write_text("tampered backup\n")
    with pytest.raises(AssertionError, match="backup changed"):
        bind_original_rows("b8_s0", tmp_path)
