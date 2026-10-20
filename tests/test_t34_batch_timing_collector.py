"""Deterministic clocks exercise the real collector's cold/steady boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_exit_b8 import teacher_v2
from research.t34_exit_b8.policy import PolicyEvaluation


def _run_clocked(tmp_path, monkeypatch, *, partial_cold: bool):
    clock = [0.0]

    class Sim:
        def __init__(self):
            self.frame = np.array([0])
            self.alive = np.ones((1, 6), dtype=bool)
            self.length = np.full((1, 6), 10)

        def step(self, _grid):
            self.frame += 1
            clock[0] += 1.0

    class Shadow:
        def __init__(self, *_args, **_kwargs):
            self.frames = []
            self.world = SimpleNamespace(sim=SimpleNamespace(alive=np.ones((1, 6), dtype=bool)))
            self.confirmed = True
            self.first_fill = True

        def fork(self, _world, **_kwargs):
            self.frames = []
            self.confirmed = True

        def catch_up(self, *, deadline):
            if not self.frames:
                self.frames.append(SimpleNamespace(action=1))
            if self.first_fill:
                self.first_fill = False
                clock[0] += 2.0 if partial_cold else 1.0
            if clock[0] >= deadline:
                raise TimeoutError("clocked partial fill")

        def confirm(self, _world):
            self.frames.pop(0)
            return True

        def invalidate(self, _reason):
            self.confirmed = False

    class Teacher:
        config = SimpleNamespace(horizon=300)
        device = "cpu"
        policy = None

        def evaluate(self, _world):
            return PolicyEvaluation(
                np.array([0, 1, 0, 0, 0, 0], dtype=np.float32), np.ones(6, dtype=bool)
            )

        def actions(self, _world, action):
            return np.full((1, 6), action, dtype=np.int64)

        def profile(self):
            return {"active_row_steps": 0}

    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"clock fixture")
    monkeypatch.setattr(teacher_v2.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(teacher_v2, "ShadowTrajectory", Shadow)
    monkeypatch.setattr(teacher_v2, "prepare_world", lambda world: world)
    monkeypatch.setattr(teacher_v2, "state_hash", lambda sim: str(int(sim.frame[0])))
    return teacher_v2.run_worlds(
        {
            "checkpoint": checkpoint,
            "output_dir": tmp_path / "worlds",
            "rows": [{"world_id": "clocked", "world_seed": 7, "mix": "scripted"}],
            "world_builder": lambda *_args: SimpleNamespace(
                sim=Sim(), runtime=SimpleNamespace(observe_step=lambda _sim: None)
            ),
            "teacher": Teacher(),
            "worlds_per_process": 1,
            "frames": 100,
            "seconds": 1.0 if partial_cold else 100.0,
            "steady_state_seconds": 2.0,
        }
    )


def test_collector_partial_cold_work_never_becomes_steady_work(tmp_path, monkeypatch):
    result = _run_clocked(tmp_path, monkeypatch, partial_cold=True)
    assert not result["complete"]
    assert not result["cold_fill_complete"]
    assert result["cold_fill_active_row_steps"] == 1
    assert result["steady_state_active_row_steps"] == 0
    assert result["steady_state_host_stage_seconds"] == {}
    assert result["steady_state_elapsed_seconds"] == 0
    assert not result["steady_state_window_complete"]
    assert result["first_decision_elapsed_seconds"] is None


def test_collector_stops_at_requested_post_fill_window(tmp_path, monkeypatch):
    result = _run_clocked(tmp_path, monkeypatch, partial_cold=False)
    assert not result["complete"]
    assert result["cold_fill_complete"]
    assert result["cold_fill_elapsed_seconds"] == pytest.approx(1.0)
    assert result["first_decision_elapsed_seconds"] == pytest.approx(2.0)
    assert result["elapsed_seconds"] == pytest.approx(3.0)
    assert result["steady_state_elapsed_seconds"] == pytest.approx(2.0)
    assert result["steady_state_window_complete"]
    assert result["steady_state_root_row_steps"] == result["steady_state_decisions"] == 2
    assert result["steady_state_shadow_row_steps"] == 1
    assert result["steady_state_host_stage_seconds"].get("build_s", 0.0) == 0
