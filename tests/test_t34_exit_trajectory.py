"""Failure-path coverage for the bounded death-window trajectory controller."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_exit_b8 import trajectory


class _Sim:
    S = 6

    def __init__(self) -> None:
        self.frame = np.array([0])
        self.alive = np.ones((1, 6), dtype=bool)
        self.length = np.full((1, 6), 10)
        self._policy_phase_prepared = False

    def step_with_policy(self, selector) -> None:
        self.frame += 1
        selector(self)

    def step(self, grid: np.ndarray) -> None:
        if self.frame[0] == 3 and grid[0, 0] == 1:
            self.alive[0, 0] = False
        self._policy_phase_prepared = False


class _World:
    def __init__(self) -> None:
        self.sim = _Sim()
        self.runtime = SimpleNamespace(prepare=lambda sim: None, observe_step=lambda sim: None)

    def clone(self) -> _World:
        return copy.deepcopy(self)


class _Teacher:
    def __init__(self, action: int) -> None:
        self.action = action
        self.queries = 0

    def evaluate(self, world) -> SimpleNamespace:
        return SimpleNamespace(greedy_action=lambda: 1)

    def choose(self, worlds) -> tuple[np.ndarray, list[dict]]:
        self.queries += 1
        return np.array([self.action]), [{"expanded": True}]

    def actions(self, world, action: int) -> np.ndarray:
        return np.array([[action, 1, 1, 1, 1, 1]])

    def profile(self) -> dict:
        return {"decisions": self.queries}


@pytest.mark.parametrize(
    "beta,rescue,expected_rewinds,expected_decisions",
    [
        (1.0, 2, 1, 6),
        (0.0, 2, 2, 3),
        (1.0, 1, 2, 3),
    ],
)
def test_rewinds_replace_abandoned_suffix_and_enforce_cap(
    monkeypatch, tmp_path, beta, rescue, expected_rewinds, expected_decisions
) -> None:
    from research.t34_exit_b8 import exports

    monkeypatch.setattr(exports, "state_hash", lambda sim: f"frame-{int(sim.frame[0])}")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    result = trajectory.collect_world(
        {
            "checkpoint": checkpoint,
            "row": {
                "world_id": "unit",
                "world_seed": 7,
                "mix": "scripted",
                "opponents": [["scripted", "greedy_food"]] * 5,
            },
            "world_builder": lambda row, path: _World(),
            "teacher": _Teacher(rescue),
            "frames": 6,
            "seconds": 20,
            "beta": beta,
            "triggers": {"death_window": 3, "max_rewinds": 2, "low_length_probability": 0.0},
        }
    )
    assert result["rewinds"] == expected_rewinds
    assert result["decisions"] == expected_decisions
    assert len(result["actions"]) == expected_decisions
    assert [row["decision"] for row in result["labels"]] == [0, 1, 2]
    assert all(row["reason"] == "death_window" for row in result["labels"])
    assert result["complete"]
    assert result["survives"] == (expected_decisions == 6)


def test_namespace_derivation_is_portable_and_mix_specific() -> None:
    config = {"namespace": "t34-unit-round1-shard2-attempt3", "mixes": "scripted,frozen,mixed"}
    rows = trajectory.make_world_rows(config, 6)
    assert rows == trajectory.make_world_rows(config, 6)
    assert len({row["world_seed"] for row in rows}) == 6
    assert [row["mix"] for row in rows] == ["scripted", "frozen", "mixed"] * 2
    assert all(row["opponent_veto"] == "v8" for row in rows)
    assert "/Users/" not in str(rows)
    assert rows != trajectory.make_world_rows({**config, "namespace": "t34-other"}, 6)
