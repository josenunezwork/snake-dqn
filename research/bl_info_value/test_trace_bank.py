"""Focused checks for action-tape replay boundaries and state bindings."""

from __future__ import annotations

import numpy as np
import pytest

from research.bl_info_value.trace_bank import state_binding, trace_step


class _Sim:
    S = 6

    def __init__(self) -> None:
        self.seen = None
        self.frame = np.array([0], dtype=np.int64)

    def get_advisory_action_mask(self):
        return np.ones((1, 6), dtype=bool)

    def get_resolved_action_mask(self):
        return np.ones((1, 6), dtype=bool)

    def step_with_policy(self, selector):
        self.frame += 1
        self.seen = selector(self)


class _Runtime:
    def __init__(self) -> None:
        self.prepared = 0
        self.observed = 0

    def prepare(self, _sim):
        self.prepared += 1

    def observe_step(self, _sim):
        self.observed += 1


def test_trace_step_uses_supplied_actions_without_policy_dispatch(monkeypatch) -> None:
    world = type("World", (), {"sim": _Sim(), "runtime": _Runtime()})()
    import src.simd_env.ego_raster as raster

    monkeypatch.setattr(raster, "build_ego_raster", lambda *_a, **_k: {"local": np.zeros((1,))})
    actions, advisory, obs = trace_step(world, np.arange(6), capture_obs=True)
    np.testing.assert_array_equal(world.sim.seen, np.arange(6, dtype=np.int64).reshape(1, 6))
    np.testing.assert_array_equal(actions, world.sim.seen)
    assert advisory.dtype == bool and obs is not None
    assert world.runtime.prepared == world.runtime.observed == 1


def test_trace_step_rejects_bad_action_shape_or_code() -> None:
    world = type("World", (), {"sim": _Sim(), "runtime": _Runtime()})()
    with pytest.raises(ValueError, match="shape"):
        trace_step(world, np.zeros((2, 6), dtype=np.int64), capture_obs=False)
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        trace_step(world, np.full(6, 6), capture_obs=False)


def test_state_binding_includes_runtime_carry_and_memory(monkeypatch) -> None:
    import research.redesign_scope_20261007.grid_h5000_identity as gid

    monkeypatch.setattr(gid, "env_digest", lambda _sim, _env: "sim-state")
    runtime = type("Runtime", (), {})()
    runtime._carry_state = np.zeros((1, 6, 2), dtype=np.float32)
    runtime._carry_valid = np.ones((1, 6), dtype=bool)
    runtime.memory = type(
        "Memory", (), {"prev_dist": np.zeros((1, 6)), "prev_id": np.zeros((1, 6), dtype=np.int64)}
    )()
    world = type("World", (), {"sim": object(), "runtime": runtime})()
    first = state_binding(world)
    runtime._carry_state[0, 0, 0] = 1.0
    assert state_binding(world) != first
