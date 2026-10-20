"""Focused contracts for the portable T34 world export and replay seam."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from research.t34_exit_b8 import replay
from research.t34_exit_b8.exports import load_world, normalize_world, write_world


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "student.pth"
    path.write_bytes(b"t34-test-checkpoint")
    return path


def _result(checkpoint: Path) -> dict:
    return {
        "schema": "t34-exit-world/v1",
        "world_id": "fixture/scripted/000000",
        "seed": 17,
        "mix": "scripted",
        "row": {"world_seed": 17, "mix": "scripted", "opponents": [["scripted", "greedy_food"]]},
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "actions": [[1, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1]],
        "hashes": {"0": "h0", "2": "h2"},
        "labels": [
            {
                "decision": 0,
                "action": 2,
                "student_action": 1,
                "forbidden": (0, 3),
                "outcomes": {
                    1: {"survives": True, "alive_steps": 300, "mass": 1234.0},
                    2: {"survives": True, "alive_steps": 300, "mass": 1299.0},
                },
            }
        ],
        "rewinds": 0,
        "complete": True,
        "frames": 2,
        "horizon": 300,
    }


def test_write_world_is_create_only_portable_and_fp16_mean_mass(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    path = tmp_path / "world.json"
    write_world(path, _result(checkpoint))
    world = load_world(path)
    assert world["labels"][0]["forbidden"] == [True, False, False, True, False, False]
    assert world["labels"][0]["outcomes"][2]["mean_mass"] == float(np.float16(1299 / 300))
    assert path.with_name("world.json.sha256").is_file()
    assert path.with_name("world.json.complete").is_file()
    try:
        write_world(path, _result(checkpoint))
    except FileExistsError:
        pass
    else:
        raise AssertionError("world writer replaced a create-only export")


class _FakeRuntime:
    def __init__(self) -> None:
        self.prepared = 0
        self.observed = 0

    def prepare(self, sim) -> None:
        self.prepared += 1

    def observe_step(self, sim) -> None:
        self.observed += 1

    def selection(self, slots):
        return np.full((1, 61), self.prepared, dtype=np.float32), np.ones((1, 6), dtype=bool)


class _FakeSim:
    E, S = 1, 6

    def __init__(self) -> None:
        self.frame = 0
        self._policy_phase_prepared = False

    def step_with_policy(self, selector) -> None:
        selector(self)

    def step(self, actions) -> None:
        assert self._policy_phase_prepared
        assert actions.shape == (1, 6)
        self._policy_phase_prepared = False
        self.frame += 1


class _FakeWorld:
    def __init__(self) -> None:
        self.sim = _FakeSim()
        self.runtime = _FakeRuntime()


def test_replay_rejects_entire_world_before_returning_observations(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    exported = normalize_world(_result(checkpoint))
    monkeypatch.setattr(replay, "state_hash", lambda sim: f"h{sim.frame}")

    def observe(world, label):
        del world
        return {"decision": int(label["decision"])}

    result = replay.replay_world(
        exported, checkpoint, build_world=lambda row, ckpt: _FakeWorld(), observe=observe
    )
    assert result.accepted
    assert len(result.observations) == 1
    exported["hashes"]["2"] = "wrong"
    rejected = replay.replay_world(
        exported, checkpoint, build_world=lambda row, ckpt: _FakeWorld(), observe=observe
    )
    assert not rejected.accepted
    assert rejected.observations == ()
    assert "frame 2" in str(rejected.mismatch)
