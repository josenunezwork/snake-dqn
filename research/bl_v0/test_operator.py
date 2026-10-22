"""Focused no-world contracts for V0's exact-stop and dummy-leaf helpers."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from research.bl_k1.native import RowStats
from research.bl_served_search import ceiling
from research.bl_v0.operator import (
    CELL_MODES,
    PREFIX,
    V0Operator,
    _attestation,
    _BatchTrace,
    _FixedDummyLeaf,
    _ordered_actions,
    _project_leaf,
)


def test_registered_cells_and_t3_stop_order_are_fixed() -> None:
    assert CELL_MODES == ("all", "stop", "prefix64_oracle", "prefix64_oracle_stop")
    assert PREFIX == 64
    q = np.array([0.2, 0.8, 0.8, -1.0, 0.0, 0.0], dtype=np.float32)
    # Bare action is supplied by the unchanged selector.  Equal-Q alternatives
    # use action index only after the bare-first branch.
    assert _ordered_actions([0, 1, 2, 4], 1, q) == [1, 2, 0, 4]


def test_oracle_stop_preserves_first_h300_survivor_and_all_die_order() -> None:
    order = [2, 1, 0, 4]
    survivor = {
        2: {"survives": False, "frames_executed": 10},
        1: {"survives": True, "frames_executed": 300},
        0: {"survives": True, "frames_executed": 300},
        4: {"survives": False, "frames_executed": 299},
    }
    assert V0Operator._stop_order_from_oracle(order, survivor) == [2, 1]
    all_die = {action: {"survives": False, "frames_executed": 12} for action in order}
    assert V0Operator._stop_order_from_oracle(order, all_die) == order


def test_h300_stop_uses_t3_tie_order_and_keeps_all_dead_outcomes() -> None:
    operator = V0Operator()
    trace = _BatchTrace(q=np.array([0.7, 0.9, 0.9, 0.0, 0.0, 0.0], dtype=np.float32))
    seen: list[int] = []
    responses = {
        1: {"survives": False, "frames_executed": 20},
        2: {"survives": False, "frames_executed": 25},
        0: {"survives": False, "frames_executed": 30},
    }

    def rollout(_world, actions, *_args, **_kwargs):
        action = actions[0]
        seen.append(action)
        return {action: responses[action]}

    outcomes = operator._stopped_h300(rollout, object(), [0, 1, 2], 300, 0, trace)
    # Bare is Q argmax action 1; tied alternative 2 follows by index, then 0.
    assert seen == [1, 2, 0]
    assert trace.skipped == []
    assert ceiling._choose(outcomes, 1, trace.q) == 0


def test_public_batch_aliases_share_one_instance_cache_without_a_simulator(monkeypatch) -> None:
    operator = V0Operator()
    seen = []

    def fake_batch(_fixture):
        seen.append(operator.mode)
        return {"semantic": {"chosen": 1}}

    monkeypatch.setattr(operator, "_batch", fake_batch)
    for cell, expected in zip(("C1", "C2", "C3", "C4"), CELL_MODES):
        assert operator.batch({}, cell)["semantic"]["chosen"] == 1
        assert seen[-1] == expected
    assert operator.mode == "all"


def test_prefix_runs_real_forced_then_63_steps_with_kernel_scope_and_isolated_carry(
    monkeypatch,
) -> None:
    """Exercise the H64 leaf seam without constructing an actual simulator."""
    from research.bl_k1 import native
    from research.t3_rollout_labels_20261015 import bank_engine
    from src.simd_env import ego_raster

    entered: list[str] = []
    captured = []

    class Policy:
        def __init__(self, carry):
            self.live_vetoes = carry

    class Sim:
        def __init__(self):
            self.alive = np.array([[True]])
            self.calls = 0

    class World:
        mix = "frozen"

        def __init__(self, shared):
            self.sim = Sim()
            self.hero = Policy(shared)
            self.opponents = [Policy(shared)]

        def clone(self):
            # Deliberately retain the same nested carry object so V0 must copy it.
            clone = World.__new__(World)
            clone.sim = Sim()
            clone.hero = Policy(self.hero.live_vetoes)
            clone.opponents = [Policy(self.opponents[0].live_vetoes)]
            return clone

    @contextmanager
    def scope(mix):
        entered.append(mix)
        yield

    def forced(branch, action):
        assert action == 3
        branch.sim.calls += 1
        captured.append(branch)

    def ordinary(branch, *, capture_obs):
        assert capture_obs is False
        branch.sim.calls += 1
        return None

    def raster(sim, *_args, **_kwargs):
        assert sim.calls == 64
        return {
            "local": np.arange(128, dtype=np.float32).reshape(1, 1, 128),
            "global": np.arange(128, 256, dtype=np.float32).reshape(1, 1, 128),
            "scalars": np.arange(256, 384, dtype=np.float32).reshape(1, 128),
        }

    monkeypatch.setattr(native, "branch_scope", scope)
    monkeypatch.setattr(bank_engine, "step_prepared", forced)
    monkeypatch.setattr(bank_engine, "choose_and_step", ordinary)
    monkeypatch.setattr(ego_raster, "build_ego_raster", raster)
    root = World({"counter": [1]})
    outcome, feature = V0Operator()._prefix(root, 3, 99)
    assert entered == ["frozen"]
    assert outcome["frames_executed"] == 64 and outcome["survives"] is True
    assert feature.shape == (128,)
    branch = captured[0]
    assert branch.hero.live_vetoes is not root.hero.live_vetoes
    branch.hero.live_vetoes["counter"].append(2)
    assert root.hero.live_vetoes == {"counter": [1]}


@pytest.mark.parametrize("fatal_at", (1, 64))
def test_prefix_counts_fatal_transition_at_first_or_last_prefix_frame(
    monkeypatch, fatal_at
) -> None:
    from research.bl_k1 import native
    from research.t3_rollout_labels_20261015 import bank_engine
    from src.simd_env import ego_raster

    class World:
        mix = "frozen"

        def __init__(self):
            self.sim = SimpleNamespace(alive=np.array([[True]]), calls=0)
            self.hero = SimpleNamespace(live_vetoes={})
            self.opponents = []

        def clone(self):
            return World()

    @contextmanager
    def scope(_mix):
        yield

    ordinary_calls = 0

    def fatal(branch, _action):
        branch.sim.calls += 1
        if fatal_at == 1:
            branch.sim.alive[0, 0] = False

    def ordinary(branch, **_kwargs):
        nonlocal ordinary_calls
        ordinary_calls += 1
        branch.sim.calls += 1
        if branch.sim.calls == fatal_at:
            branch.sim.alive[0, 0] = False

    monkeypatch.setattr(native, "branch_scope", scope)
    monkeypatch.setattr(bank_engine, "step_prepared", fatal)
    monkeypatch.setattr(bank_engine, "choose_and_step", ordinary)
    monkeypatch.setattr(
        ego_raster,
        "build_ego_raster",
        lambda *_args, **_kwargs: {
            "local": np.zeros((1, 1, 128), np.float32),
            "global": np.zeros((1, 1, 128), np.float32),
            "scalars": np.zeros((1, 128), np.float32),
        },
    )
    outcome, _ = V0Operator()._prefix(World(), 1, 0)
    assert outcome["frames_executed"] == fatal_at
    assert outcome["survives"] is False
    assert ordinary_calls == fatal_at - 1


def test_prefix_oracle_retains_h300_decision_and_charges_one_legal_batch(monkeypatch) -> None:
    fixture = {"root_sha256": "a" * 64}
    q = np.array([0.2, 0.9, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)
    outcomes = {
        0: {"survives": False, "frames_executed": 300},
        1: {"survives": False, "frames_executed": 11},
        2: {"survives": True, "frames_executed": 300},
    }
    calls = []

    def prefix(_world, action, _decision):
        calls.append(action)
        # Deliberately disagree with H300: the output must remain C1's oracle.
        return {"survives": False, "frames_executed": 64}, np.full(128, action, np.float32)

    cache = {fixture["root_sha256"]: {"q": q.tolist(), "outcomes": outcomes}}
    c3 = V0Operator("prefix64_oracle", oracle_cache=cache)
    monkeypatch.setattr(c3, "_prefix", prefix)
    trace = _BatchTrace(q=q.copy())
    result = c3._prefix_oracle(object(), [0, 1, 2], 0, trace, fixture["root_sha256"])
    assert calls == [1, 2, 0]
    assert result[2]["survives"] is True
    assert trace.dummy_leaf_calls == 1 and trace.dummy_leaf_batch_size == 3
    assert c3._dummy_leaf is not None and c3._dummy_leaf.batch_sizes[-1] == 3

    calls.clear()
    c4 = V0Operator("prefix64_oracle_stop", oracle_cache=cache)
    monkeypatch.setattr(c4, "_prefix", prefix)
    trace = _BatchTrace(q=q.copy())
    result = c4._prefix_oracle(object(), [0, 1, 2], 0, trace, fixture["root_sha256"])
    assert calls == [1, 2]
    assert trace.skipped == [0]
    assert result[2]["survives"] is True
    assert trace.dummy_leaf_calls == 1 and trace.dummy_leaf_batch_size == 3
    assert c4._dummy_leaf is not None and c4._dummy_leaf.batch_sizes[-1] == 3

    trace = _BatchTrace(q=q + 1)
    with pytest.raises(RuntimeError, match="B8 Q differs"):
        c3._prefix_oracle(object(), [0, 1, 2], 0, trace, fixture["root_sha256"])


def test_leaf_projection_is_real_input_deterministic_and_fixed_width() -> None:
    obs = {
        "local": np.arange(64, dtype=np.uint8).reshape(1, 8, 8),
        "global": np.arange(64, 128, dtype=np.uint8).reshape(1, 8, 8),
        "scalars": np.arange(128, 192, dtype=np.float32).reshape(1, 64),
    }
    first = _project_leaf(obs)
    second = _project_leaf(obs)
    assert first.shape == (128,)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert first[0] == 0.0
    assert first[-1] == 191.0


def test_dummy_leaf_has_registered_shape_dtype_and_parameter_count() -> None:
    leaf = _FixedDummyLeaf()
    assert sum(parameter.numel() for parameter in leaf.model.parameters()) == 12_806
    assert next(leaf.model.parameters()).dtype == leaf.torch.float32
    leaf(np.zeros((4, 128), dtype=np.float32))


def test_attestation_never_invents_scripted_kernel_rows() -> None:
    stats = RowStats(kernel_rows=4, rows=4, stock_rows=0, noncandidate_rows=0)
    assert _attestation(stats, "frozen")["status"] == "ATTESTED"
    assert _attestation(RowStats(), "scripted")["status"] == "NOT_APPLICABLE_SCRIPTED"
    assert _attestation(RowStats(stock_rows=1), "mixed")["status"] == "INVALID"
