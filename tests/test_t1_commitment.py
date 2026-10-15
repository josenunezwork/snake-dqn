"""Exact replay, full-state clone isolation and intervention endpoint checks."""

import hashlib
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


@pytest.fixture(scope="module")
def census_replay() -> Replay:
    configure_cpu()
    return build_replay("b8_s0", [3])


def test_short_recorded_replay_clone_parity() -> None:
    """One real census world, 200 full-state clone comparisons plus lengths."""
    configure_cpu()
    receipt = compare_world(3, warmup=100, frames=200)
    assert receipt["identical"]
    assert receipt["full_digest_comparisons"] == 201
    assert receipt["length_values_compared"] == 500


def test_prepared_clone_executes_exactly_one_transition(census_replay: Replay) -> None:
    original = census_replay.clone()
    _, series = load_inputs("b8_s0")
    captured = []

    def capture(r: Replay) -> None:
        captured.append(r.clone(prepared=True))

    action = int(series["action"][3, 0])
    original.step([action], capture)
    branch = captured[0]
    assert int(branch.sim.frame[0]) == 1
    branch.first(action)
    assert int(branch.sim.frame[0]) == 1
    assert_same_world(original, branch)
    original.step([int(series["action"][3, 1])])
    branch.step([int(series["action"][3, 1])])
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
