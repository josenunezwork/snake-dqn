"""Independent-root bucketing and exact carried-veto dispatch contracts."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from research.t34_exit_b8.policy import B8Policy


def test_world_forward_buckets_are_bounded_and_scattered(monkeypatch) -> None:
    from src.simd_env import ego_raster

    model = object()
    calls = []
    worlds = []
    for index in range(5):
        sim = SimpleNamespace(index=index)
        sim.get_legal_action_mask = lambda: np.ones((1, 6, 6), dtype=bool)
        sim.get_advisory_action_mask = lambda: np.tile(
            np.array([True, True, True, False, False, False]), (1, 6, 1)
        )
        worlds.append(SimpleNamespace(sim=sim, hero=SimpleNamespace(net=model)))

    def raster(sim, *args, **kwargs):
        return {"local": np.full((1, 1), sim.index, dtype=np.float32)}

    def forward(net, obs):
        assert net is model
        calls.append(len(obs["local"]))
        return np.repeat(obs["local"], 6, axis=1)

    monkeypatch.setattr(ego_raster, "build_ego_raster", raster)
    policy = B8Policy("cpu")
    monkeypatch.setattr(policy, "_forward", forward)
    values = policy.evaluate_worlds(worlds, bucket_size=2)
    assert calls == [2, 2, 1]
    assert [float(value.q[0]) for value in values] == [0, 1, 2, 3, 4]
    assert all(value.greedy_action() == 0 for value in values)
    with pytest.raises(ValueError, match="bucket"):
        policy.evaluate_worlds(worlds, bucket_size=0)
    worlds[1].hero.net = object()
    with pytest.raises(ValueError, match="immutable hero model"):
        policy.evaluate_worlds(worlds, bucket_size=2)


class _DispatchSim:
    S = 3

    def __init__(self) -> None:
        self._alive = np.ones((2, 3), dtype=bool)
        self._resolved = np.ones((2, 3, 6), dtype=bool)

    def get_alive(self) -> np.ndarray:
        return self._alive

    def get_resolved_action_mask(self) -> np.ndarray:
        return self._resolved


class _CarriedVetoPolicy:
    """Minimal Vector61-shaped policy whose action call represents v8 carry."""

    def __init__(self, veto_slots: tuple[int, ...]) -> None:
        self.runtime = object()
        self.q_values = lambda _states: None
        self.veto_slots = frozenset(veto_slots)
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def actions(self, masks, sim, slots) -> np.ndarray:
        del sim
        self.calls.append((np.asarray(masks).copy(), np.asarray(slots).copy()))
        # This deliberately differs from a masked-Q argmax.  It stands in for
        # an existing live v8 hook selecting an action from carried state.
        return np.asarray([5 - int(slot) for _, slot in slots], dtype=np.int64)


class _ShallowCloneWorld:
    """T3-shaped clone that shares policy dictionaries until T34 repairs them."""

    def __init__(self) -> None:
        self.sim = SimpleNamespace(_policy_phase_prepared=True)
        self.runtime = SimpleNamespace()
        self.hero = SimpleNamespace()
        self.opponents = [
            SimpleNamespace(
                veto_slots=frozenset({1}),
                veto_counters={0: {"calls": 2}},
                boost_counters={0: {"calls": 3}},
                live_vetoes={0: {"carry": 4}},
                last_veto={"final": np.asarray([2], dtype=np.int64)},
            )
        ]

    def clone(self):
        clone = copy.copy(self)
        clone.sim = copy.copy(self.sim)
        clone.runtime = copy.copy(self.runtime)
        clone.hero = copy.copy(self.hero)
        clone.opponents = [copy.copy(policy) for policy in self.opponents]
        return clone


def test_vetoed_opponents_use_native_actions_with_all_grouped_slots() -> None:
    """A carried veto never takes the accelerated Q-only dispatch path."""
    sim = _DispatchSim()
    policy = _CarriedVetoPolicy((1, 2))
    grid = np.zeros((2, 3), dtype=np.int64)

    B8Policy("cpu").dispatch(sim, object(), [[policy, policy], [policy, policy]], grid)

    assert len(policy.calls) == 1
    masks, slots = policy.calls[0]
    assert slots.tolist() == [[0, 1], [1, 1], [0, 2], [1, 2]]
    assert masks.shape == (4, 6)
    assert grid[:, 1:].tolist() == [[4, 3], [4, 3]]


def test_engine_branch_clone_retains_isolated_veto_carry() -> None:
    """Search branches begin with, but cannot mutate, their source v8 carry."""
    from research.t34_exit_b8.engine import ExitTeacher

    source = _ShallowCloneWorld()
    clone = ExitTeacher()._clone(source)
    original, copied = source.opponents[0], clone.opponents[0]

    assert copied.veto_slots == frozenset({1})
    assert copied.live_vetoes == {0: {"carry": 4}}
    assert copied.veto_counters == {0: {"calls": 2}}
    assert copied.boost_counters == {0: {"calls": 3}}
    copied.live_vetoes[0]["carry"] = 99
    copied.last_veto["final"][0] = 5
    assert original.live_vetoes[0]["carry"] == 4
    assert int(original.last_veto["final"][0]) == 2


def test_merged_veto_policy_keeps_source_carry_and_slot_identity() -> None:
    """Same checkpoint policies remain distinct when their veto slots differ."""
    from research.t34_exit_b8.engine import _clone_merged_policy, _merged_policy_key

    first = SimpleNamespace(
        checkpoint_path="frozen.pth",
        veto_spec="v8",
        veto_slots=frozenset({1}),
        forward="rowwise",
        veto_counters={0: {"calls": 2}},
        boost_counters={0: {"calls": 3}},
        live_vetoes={0: {"carry": 4}},
    )
    second = copy.copy(first)
    second.veto_slots = frozenset({2})
    assert _merged_policy_key(first, 0, 1) != _merged_policy_key(second, 0, 1)

    clone = _clone_merged_policy(first, runtime=object(), branch_envs=np.asarray([3, 7]))
    assert clone.forward == "rowwise"
    assert clone.veto_slots == frozenset({1})
    assert clone.live_vetoes == {3: {"carry": 4}, 7: {"carry": 4}}
    clone.live_vetoes[3]["carry"] = 99
    assert clone.live_vetoes[7]["carry"] == 4
    assert first.live_vetoes[0]["carry"] == 4
