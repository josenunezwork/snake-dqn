"""Pure contract coverage for the v2 teacher export boundary."""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_exit_b8 import teacher_v2
from research.t34_exit_b8.engine import mix_action
from research.t34_exit_b8.teacher_v2 import _label, _put_label
from research.t34_shadow.lad import search_lad_teacher


def _evaluation() -> SimpleNamespace:
    return SimpleNamespace(q=np.arange(6, dtype=np.float32), legal=np.ones(6, dtype=bool))


def test_decisive_label_promotes_a_same_decision_shadow_anchor() -> None:
    labels = {}
    anchor = _label(
        decision=12,
        evaluation=_evaluation(),
        student_action=5,
        role="shadow_confirm",
        shadow_id=1,
        shadow_frame=12,
        shadow_confirmed=True,
        kl_anchor_only=True,
    )
    decisive = _label(
        decision=12,
        evaluation=_evaluation(),
        student_action=5,
        role="lad_transition",
        shadow_id=1,
        shadow_frame=12,
        shadow_confirmed=True,
        searched=True,
        decisive=True,
        t_star=12,
        a_star=4,
        forbidden=[0, 5],
        per_action={"4": {"survives": True}, "5": {"survives": False}},
    )
    _put_label(labels, anchor)
    _put_label(labels, decisive)
    assert list(labels) == [12]
    assert labels[12]["label_role"] == "lad_transition"
    assert labels[12]["F"] == [0, 5]
    assert labels[12]["a_star"] not in labels[12]["F"]
    assert labels[12]["decisive"] and not labels[12]["kl_anchor_only"]


def test_sparse_label_is_neither_search_result_nor_kl_anchor() -> None:
    sparse = _label(
        decision=7,
        evaluation=_evaluation(),
        student_action=5,
        role="sparse_before_lad",
        shadow_id=2,
        shadow_frame=7,
        shadow_confirmed=True,
    )
    assert sparse["searched"] is False
    assert sparse["decisive"] is False
    assert sparse["kl_anchor_only"] is False


def test_stride_teacher_discovers_two_nonmonotone_transition_edges() -> None:
    """Two separate survive-to-die edges must remain distinct teacher labels."""
    surviving = {*range(0, 5), *range(20, 25)}

    def expand(decision: int) -> dict:
        return {
            "greedy": 0,
            "q": [0.0, 1.0],
            "outcomes": {
                "0": {"survives": False},
                "1": {"survives": decision in surviving},
            },
        }

    result = search_lad_teacher(0, 30, expand)
    probes = {int(probe["decision"]): probe for probe in result["probes"]}
    edges = [
        decision
        for decision, probe in probes.items()
        if probe["surviving_alternatives"]
        and decision + 1 in probes
        and not probes[decision + 1]["surviving_alternatives"]
    ]
    assert edges == [4, 24]
    assert result["t_star"] == 24


class _Outcome:
    def __init__(self, survives: bool, row_steps: int = 1) -> None:
        self.survives, self.alive_steps, self.mass, self.row_steps = survives, 1, 1.0, row_steps


class _SearchTeacher:
    """Small branch adapter whose survival pattern has two separate edges."""

    def evaluate(self, world):
        return SimpleNamespace(
            q=np.array([0.0, 2.0, 1.0, -1.0, -2.0, -3.0], dtype=np.float32),
            legal=np.array([True, True, False, False, False, False]),
            greedy_action=lambda: 0,
        )

    def _rollout(self, world, actions):
        survives = world.sim.t in {*range(0, 5), *range(20, 25)}
        return {
            int(action): _Outcome(
                survives if int(action) == 1 else False,
                1 if int(action) == 1 else 31 - world.sim.t,
            )
            for action in actions
        }


class _NoRescueTeacher(_SearchTeacher):
    def _rollout(self, world, actions):
        return {int(action): _Outcome(False, 31 - world.sim.t) for action in actions}


class _BadControlTeacher(_NoRescueTeacher):
    def _rollout(self, world, actions):
        return {int(action): _Outcome(False, 1) for action in actions}


def test_batched_doom_search_emits_edges_and_forbidden_greedy(monkeypatch) -> None:
    """The v2 batch helper keeps both edges and defines F as dying actions."""
    monkeypatch.setattr(teacher_v2, "prepare_world", lambda world: world)
    monkeypatch.setattr(teacher_v2, "clone_world_exact", copy.deepcopy)

    def step(world, _teacher, _action):
        world.sim.t += 1
        return [0] * 6

    monkeypatch.setattr(teacher_v2, "_step", step)
    world = SimpleNamespace(sim=SimpleNamespace(t=0))
    state = {
        "world": world,
        "shadow": SimpleNamespace(frames=[SimpleNamespace(action=0) for _ in range(31)]),
        "decision": 0,
        "shadow_id": 3,
        "seed": 44,
        "rng": random.Random(44).getstate(),
        "work": defaultdict(int),
        "stages": defaultdict(float),
        "pending_labels": {},
        "pending_actions": {},
        "search_bucket_peak": 0,
        "searches": 0,
    }
    second = copy.deepcopy(state)
    second["seed"] = 45
    second["rng"] = random.Random(45).getstate()
    second["shadow_id"] = 4
    teacher_v2._search_dooms([state, second], _SearchTeacher(), branch_bucket_size=12)
    for searched in (state, second):
        decisive = [row for row in searched["pending_labels"].values() if row["decisive"]]
        assert {row["decision"] for row in decisive} == {4, 24}
        assert all(row["a_star"] == 1 for row in decisive)
        assert all(row["F"] == [0] for row in decisive)
        assert searched["pending_actions"] == {24: 1}
        assert searched["work"]["search_row_steps"] > 0


def test_beta_schedule_is_deterministic_and_keeps_governed_endpoints() -> None:
    assert mix_action(2, 5, seed=8, decision=19, beta=1.0) == 5
    assert mix_action(2, 5, seed=8, decision=19, beta=0.0) == 2
    half_first = [mix_action(2, 5, seed=8, decision=d, beta=0.5) for d in range(32)]
    half_second = [mix_action(2, 5, seed=8, decision=d, beta=0.5) for d in range(32)]
    assert half_first == half_second
    assert set(half_first) == {2, 5}


def test_second_doom_without_lad_clears_a_previous_sparse_sample(monkeypatch) -> None:
    monkeypatch.setattr(teacher_v2, "prepare_world", lambda world: world)
    monkeypatch.setattr(teacher_v2, "clone_world_exact", copy.deepcopy)
    monkeypatch.setattr(
        teacher_v2,
        "_step",
        lambda world, _teacher, _action: (setattr(world.sim, "t", world.sim.t + 1), [0] * 6)[1],
    )
    state = {
        "world": SimpleNamespace(sim=SimpleNamespace(t=0)),
        "shadow": SimpleNamespace(frames=[SimpleNamespace(action=0) for _ in range(31)]),
        "decision": 0,
        "shadow_id": 7,
        "seed": 88,
        "rng": random.Random(88).getstate(),
        "work": defaultdict(int),
        "stages": defaultdict(float),
        "pending_labels": {},
        "pending_actions": {},
        "search_bucket_peak": 0,
        "searches": 0,
    }
    teacher_v2._search_dooms([state], _SearchTeacher(), branch_bucket_size=12)
    assert "sparse_decision" in state
    state["pending_labels"].clear()
    state["pending_actions"].clear()
    teacher_v2._search_dooms([state], _NoRescueTeacher(), branch_bucket_size=12)
    assert "sparse_decision" not in state
    assert state["pending_labels"] == {}
    assert state["pending_actions"] == {}


def test_doom_search_rejects_a_control_that_misses_the_shadow_death(monkeypatch) -> None:
    monkeypatch.setattr(teacher_v2, "prepare_world", lambda world: world)
    monkeypatch.setattr(teacher_v2, "clone_world_exact", copy.deepcopy)
    monkeypatch.setattr(
        teacher_v2,
        "_step",
        lambda world, _teacher, _action: (setattr(world.sim, "t", world.sim.t + 1), [0] * 6)[1],
    )
    state = {
        "world": SimpleNamespace(sim=SimpleNamespace(t=0)),
        "shadow": SimpleNamespace(frames=[SimpleNamespace(action=0) for _ in range(31)]),
        "decision": 0,
        "shadow_id": 9,
        "seed": 89,
        "rng": random.Random(89).getstate(),
        "work": defaultdict(int),
        "stages": defaultdict(float),
        "pending_labels": {},
        "pending_actions": {},
        "search_bucket_peak": 0,
        "searches": 0,
    }
    with pytest.raises(teacher_v2.ShadowControlMismatch):
        teacher_v2._search_dooms([state], _BadControlTeacher(), branch_bucket_size=12)


class _RunSim:
    def __init__(self) -> None:
        self.alive = np.array([[True]], dtype=bool)
        self.length = np.array([[10]], dtype=np.int64)

    def step(self, _grid) -> None:
        pass


class _RunWorld:
    def __init__(self) -> None:
        self.sim = _RunSim()
        self.runtime = SimpleNamespace(observe_step=lambda _sim: None)


class _MismatchShadow:
    def __init__(self, *_args, **_kwargs) -> None:
        self.frames = [SimpleNamespace(action=0)]
        self.world = SimpleNamespace(sim=SimpleNamespace(alive=np.array([[False]], dtype=bool)))
        self._confirmed = True

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def fork(self, _world, **_kwargs) -> None:
        self._confirmed = True

    def catch_up(self, **_kwargs) -> int:
        return 0

    def confirm(self, _world) -> bool:
        return False

    def invalidate(self, _reason) -> None:
        self._confirmed = False


class _RunTeacher:
    config = SimpleNamespace(horizon=3)
    policy = None
    device = "cpu"

    def evaluate(self, _world):
        return SimpleNamespace(
            q=np.zeros(6, dtype=np.float32),
            legal=np.ones(6, dtype=bool),
            greedy_action=lambda: 0,
        )

    def actions(self, _world, action):
        return np.array([[action, 0, 0, 0, 0, 0]], dtype=np.int64)

    def profile(self):
        return {"active_row_steps": 0, "resident_row_steps": 0}


def test_run_discards_future_labels_after_an_unexpected_shadow_mismatch(
    monkeypatch, tmp_path
) -> None:
    """A prospective label must never survive the mismatch that invalidates it."""
    captured = []
    monkeypatch.setattr(teacher_v2, "ShadowTrajectory", _MismatchShadow)
    monkeypatch.setattr(teacher_v2, "prepare_world", lambda world: world)
    monkeypatch.setattr(teacher_v2, "state_hash", lambda _sim: "state")

    def write(path, value):
        path.write_text("{}", encoding="utf-8")
        captured.append(value)
        return path

    monkeypatch.setattr(teacher_v2, "write_world", write)

    injected = False

    def search(states, _teacher, *_args):
        nonlocal injected
        if injected:
            return
        injected = True
        for state in states:
            state["pending_labels"][1] = {"decision": 1, "decisive": True}
            state["pending_actions"][1] = 1
            state["searched"] = True

    monkeypatch.setattr(teacher_v2, "_search_dooms", search)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    result = teacher_v2.run_worlds(
        {
            "checkpoint": checkpoint,
            "output_dir": tmp_path / "out",
            "rows": [{"world_id": "w", "world_seed": 1, "mix": "scripted"}],
            "world_builder": lambda _row, _path: _RunWorld(),
            "teacher": _RunTeacher(),
            "frames": 2,
            "seconds": 5,
        }
    )
    assert result["complete"]
    assert captured[0]["labels"] == []


def test_deadline_returns_incomplete_without_any_complete_world_file(monkeypatch, tmp_path) -> None:
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 2.0

    monkeypatch.setattr(teacher_v2.time, "monotonic", clock)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    out = tmp_path / "out"
    result = teacher_v2.run_worlds(
        {
            "checkpoint": checkpoint,
            "output_dir": out,
            "rows": [{"world_id": "w", "world_seed": 1, "mix": "scripted"}],
            "frames": 2,
            "seconds": 1,
        }
    )
    assert result["complete"] is False
    assert result["worlds"] == 0
    assert not list(out.glob("*.complete"))
