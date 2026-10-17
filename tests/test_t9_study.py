"""Trace timing and recorded-versus-greedy branch contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from research.t9_divergence_20261017 import study


class FakeReplay:
    def __init__(self, dies_after: int) -> None:
        self.dies_after = dies_after
        self.count = 0
        self.forced = []
        self.sim = SimpleNamespace(
            alive=np.ones((1, 1), dtype=bool),
            frame=np.array([101]),
            length=np.array([[10]]),
            get_advisory_action_mask=lambda: np.ones((1, 1, 6), bool),
        )

    def clone(self, prepared: bool = False) -> "FakeReplay":
        assert prepared
        return self

    def q(self) -> np.ndarray:
        return np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])

    def first(self, action: int) -> None:
        self.forced.append(action)
        self.count += 1
        self.sim.alive[0, 0] = self.count < self.dies_after

    def step(self, forced=None, observer=None) -> None:
        self.sim.frame[0] += 1
        observer(self)
        self.forced.append(forced)
        self.count += 1
        self.sim.alive[0, 0] = self.count < self.dies_after


@pytest.mark.parametrize("transitions", [1, 3, 101, 102])
def test_fatal_trace_stops_at_last_live_decision(monkeypatch, transitions: int) -> None:
    monkeypatch.setattr(study, "assert_same_world", lambda *args: None)
    monkeypatch.setattr(study, "measure", lambda sim, action: {"observed_frame": int(sim.frame[0])})
    source = FakeReplay(transitions)
    row = {"env": 0, "fatal_t": 100 + transitions - 1, "lad": {"decision": 100, "chosen_action": 2}}
    series = {"action": np.full((1, 300), 2), "length": np.full((1, 300), 10)}
    result = study.trace_branch(source, row, series, True)
    count = min(101, transitions)
    assert [r["offset"] for r in result["trace"]] == list(range(count))
    assert [r["observed_frame"] for r in result["trace"]] == list(range(101, 101 + count))
    assert result["death_decision"] == 100 + transitions - 1
    assert result["length_values_checked"] == transitions
    assert source.forced == [2] + [[2]] * (transitions - 1)


def test_survivor_is_forced_once_then_greedy_and_records_offset100(monkeypatch) -> None:
    monkeypatch.setattr(study, "assert_same_world", lambda *args: None)
    monkeypatch.setattr(study, "measure", lambda sim, action: {})
    source = FakeReplay(1000)
    row = {
        "env": 0,
        "lad": {"decision": 100, "best_surviving_action": 4, "outcomes": {"4": {"horizon": 300}}},
    }
    result = study.trace_branch(source, row, {}, False)
    assert len(result["trace"]) == 101
    assert result["trace"][-1]["offset"] == 100
    assert result["trace"][0]["action"] == 4
    assert all(t["action"] == 1 for t in result["trace"][1:])
    assert source.forced == [4] + [None] * 299
    assert result["validated_transitions"] == 300
    assert result["endpoint_alive"]
