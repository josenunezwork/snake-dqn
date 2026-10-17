"""Tests for the continuation state around compiled learner and actor callables.

The compiled functions themselves are intentionally rebuilt after a resume.  These tests
pin the numerical/configuration state that must survive that rebuild: AMP scaler state,
compile fallback decisions, graph counters, fixed-shape actor staging buffers, and mutable
policy wrapper fields.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from research.redesign_m3c_20261008 import execution_snapshot as snapshot
from research.redesign_m3c_20261008 import train_m3c
from research.redesign_m3c_20261008.compile_m3c import GraphCount


class _MockGradScaler:
    """Small pickle-free stand-in for torch's GradScaler state protocol."""

    def __init__(self, scale: float) -> None:
        self.state = {"scale": float(scale), "growth": 17}

    def state_dict(self) -> dict:
        return copy.deepcopy(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = copy.deepcopy(state)


class _Hook:
    """A module-level callable object, so pickle round-tripping is meaningful."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, value: int) -> tuple[str, int]:
        return self.name, value


class _Policy:
    def __init__(self, suffix: str) -> None:
        self.veto_counters = {"safe": [1, 2], "death": {"count": 3}}
        self.boost_counters = np.array([4, 5], dtype=np.int64)
        self.live_vetoes = {"rows": {7, 9}}
        self.last_veto = {"hook": _Hook(f"hook-{suffix}"), "action": 2}


def _learner_for_execution() -> SimpleNamespace:
    learner = SimpleNamespace(
        scaler=_MockGradScaler(8.0),
        amp_skipped=4,
        _which=1,
        _cstat={"fallback": "RuntimeError: graph break", "mode": "reduce-overhead"},
        _verified={"shape": [32, 60], "probe": True},
        _gcount=GraphCount(),
        _loss_c=object(),
        _fwd_c=object(),
    )
    learner._gcount.graphs = 3
    learner._gcount.graph_breaks = 2
    return learner


def test_learner_execution_roundtrip_restores_scaler_fallback_and_graph_state():
    source = _learner_for_execution()
    state = snapshot.capture_learner_execution(source)

    # Mutate every source field after capture to prove the detached snapshot is independent.
    source.scaler.state["scale"] = 0.125
    source.amp_skipped = 99
    source._which = 7
    source._cstat["mode"] = "changed"
    source._verified["shape"][0] = 999
    source._gcount.graphs = 100
    source._gcount.graph_breaks = 101

    restored = _learner_for_execution()
    restored.scaler.state["scale"] = 0.25
    restored._cstat = {"fallback": "RuntimeError: graph break", "mode": "other"}
    restored._loss_c = object()
    restored._fwd_c = object()
    snapshot.restore_learner_execution(restored, state)

    assert restored.scaler.state_dict() == {"scale": 8.0, "growth": 17}
    assert restored.amp_skipped == 4
    assert restored._which == 1
    assert restored._cstat == {"fallback": "RuntimeError: graph break", "mode": "reduce-overhead"}
    assert restored._verified == {"shape": [32, 60], "probe": True}
    assert (restored._gcount.graphs, restored._gcount.graph_breaks) == (3, 2)
    # A fallback snapshot must force recompilable callables to be rebuilt lazily.
    assert restored._loss_c is None
    assert restored._fwd_c is None
    assert state["verified"]["shape"] == [32, 60]


def test_learner_execution_rejects_amp_or_compile_configuration_mismatch():
    state = snapshot.capture_learner_execution(_learner_for_execution())

    no_scaler = _learner_for_execution()
    no_scaler.scaler = None
    with pytest.raises(ValueError, match="AMP scaler configuration"):
        snapshot.restore_learner_execution(no_scaler, state)

    no_compile = _learner_for_execution()
    no_compile._cstat = None
    with pytest.raises(ValueError, match="compile configuration"):
        snapshot.restore_learner_execution(no_compile, state)


def test_actor_execution_roundtrip_restores_padded_stage_and_fallback():
    stage = SimpleNamespace(
        v={
            "local": np.arange(2 * 3, dtype=np.uint8).reshape(2, 3),
            "global": np.arange(2 * 4, dtype=np.uint8).reshape(2, 4),
            "scalars": np.arange(2 * 5, dtype=np.float32).reshape(2, 5),
        }
    )
    actor = SimpleNamespace(
        fallback="RuntimeError: static shape",
        calls=11,
        count=GraphCount(),
        stage=stage,
        fn=object(),
    )
    actor.count.graphs = 4
    actor.count.graph_breaks = 1
    state = snapshot.capture_actor_execution(actor)

    for array in actor.stage.v.values():
        array[...] = -7
    actor.calls = 0
    actor.count.graphs = 0
    actor.count.graph_breaks = 0
    actor.fallback = "changed"
    actor.fn = object()

    snapshot.restore_actor_execution(actor, state)
    assert actor.fallback == "RuntimeError: static shape"
    assert actor.fn is None
    assert actor.calls == 11
    assert (actor.count.graphs, actor.count.graph_breaks) == (4, 1)
    np.testing.assert_array_equal(
        actor.stage.v["local"], np.arange(6, dtype=np.uint8).reshape(2, 3)
    )
    np.testing.assert_array_equal(
        actor.stage.v["global"], np.arange(8, dtype=np.uint8).reshape(2, 4)
    )
    np.testing.assert_array_equal(
        actor.stage.v["scalars"], np.arange(10, dtype=np.float32).reshape(2, 5)
    )
    # Captured arrays remain detached from later writes to the restored staging area.
    actor.stage.v["local"][0, 0] = 255
    assert state["stage"]["local"][0, 0] == 0


def test_policy_execution_pickles_mutable_counters_and_hooks_in_isolation():
    source = {"hero": _Policy("source")}
    state = snapshot.capture_policy_execution(source)

    source["hero"].veto_counters["safe"].append(99)
    source["hero"].boost_counters[0] = -1
    source["hero"].live_vetoes["rows"].add(123)
    source["hero"].last_veto["action"] = 5

    restored = {"hero": _Policy("restored")}
    snapshot.restore_policy_execution(restored, state)
    policy = restored["hero"]
    assert policy.veto_counters == {"safe": [1, 2], "death": {"count": 3}}
    np.testing.assert_array_equal(policy.boost_counters, np.array([4, 5], dtype=np.int64))
    assert policy.live_vetoes == {"rows": {7, 9}}
    assert policy.last_veto["action"] == 2
    assert policy.last_veto["hook"](6) == ("hook-source", 6)

    # The restored mutable values are independent of both the source and serialized bytes.
    policy.veto_counters["safe"].append(88)
    assert source["hero"].veto_counters["safe"] == [1, 2, 99]
    with pytest.raises(ValueError, match="opponent roster"):
        snapshot.restore_policy_execution({"other": _Policy("other")}, state)


@pytest.mark.parametrize("field", ["seed", "actors"])
def test_full_state_recipe_changes_when_training_math_changes(field):
    base = train_m3c.default_args()
    baseline = train_m3c.full_state_recipe(base)
    changed = copy.copy(base)
    setattr(changed, field, 12345 if field == "seed" else 8)
    assert train_m3c.full_state_recipe(changed) != baseline


def test_full_state_recipe_allows_output_and_device_migration_changes():
    base = train_m3c.default_args()
    baseline = train_m3c.full_state_recipe(base)
    for field, value in (("out", "/new/output"), ("device", "cuda")):
        changed = copy.copy(base)
        setattr(changed, field, value)
        assert train_m3c.full_state_recipe(changed) == baseline, field
