"""Focused state-boundary tests for the T21 actor snapshot helper."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008.actor_snapshot import (
    capture_actor_state,
    restore_actor_state,
)


class _EnvRng:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self.seed = seed

    def draw(self) -> int:
        return self._rng.randrange(1_000_000)


class _Sim:
    def __init__(self) -> None:
        self.frame = np.array([7, 9], dtype=np.int64)
        self.mass = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._seeds = [101, 202]
        self._rngs = [_EnvRng(101), _EnvRng(202)]
        self._det = object()
        self._kernels = object()


class _Memory:
    def __init__(self) -> None:
        self.prev_dist = np.full((2, 3), np.inf)
        self.prev_id = np.full((2, 3), -1, dtype=np.int64)


class _Runtime:
    def __init__(self) -> None:
        self.memory = _Memory()
        self.controlled = np.ones((2, 3), dtype=bool)
        self._carry_state = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        self.fresh_rows = 3
        self.carried_rows = 4
        self.trace = lambda _: None
        self.featurizer = object()
        self.free_space_featurizer = object()


class _NSB:
    def __init__(self) -> None:
        self.ids = np.arange(4, dtype=np.int64).reshape(2, 2)
        self.cnt = np.array([1, 2], dtype=np.int64)
        self._chunks = [(np.array([3, 4]), np.array([1, 2]))]


def _objects():
    sim, runtime, nsb = _Sim(), _Runtime(), _NSB()
    rng = np.random.default_rng(77)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.arange(p.numel(), dtype=p.dtype).reshape_as(p) / 10)
    return sim, runtime, nsb, rng, net


def test_actor_state_round_trip_restores_sim_runtime_nstep_rng_and_network():
    sim, runtime, nsb, rng, net = _objects()
    live = np.array([[True, False, True], [False, True, True]])
    pool = np.array([["a", None, "b"], ["c", "d", None]], dtype=object)
    saved = capture_actor_state(
        sim,
        runtime,
        nsb,
        rng,
        net,
        segment=11,
        frame=8,
        last=42,
        live_seats=live,
        pool_seats=pool,
        extra={"pool_rng": np.random.default_rng(9)},
    )
    expected_rng = rng.integers(1 << 30, size=5)
    expected_env = [r.draw() for r in sim._rngs]
    expected_out = net(torch.ones((2, 3))).detach().clone()

    # Mutation after capture proves this is a detached boundary, not aliases.
    sim.frame[:] = 99
    runtime.memory.prev_id[:] = 123
    nsb._chunks[0][0][:] = 88
    _ = rng.integers(1 << 30, size=7)
    with torch.no_grad():
        next(net.parameters()).add_(100)

    restored = _objects()
    result = restore_actor_state(saved, *restored)
    got_sim, got_runtime, got_nsb, got_rng, got_net = restored
    np.testing.assert_array_equal(got_sim.frame, [7, 9])
    np.testing.assert_array_equal(got_runtime.memory.prev_id, np.full((2, 3), -1))
    np.testing.assert_array_equal(got_nsb._chunks[0][0], [3, 4])
    np.testing.assert_array_equal(got_rng.integers(1 << 30, size=5), expected_rng)
    assert [r.draw() for r in got_sim._rngs] == expected_env
    torch.testing.assert_close(got_net(torch.ones((2, 3))), expected_out, rtol=0, atol=0)
    assert result["segment"] == 11 and result["frame"] == 8 and result["last"] == 42
    np.testing.assert_array_equal(result["live_seats"], live)
    assert result["pool_seats"].tolist() == pool.tolist()
    assert isinstance(result["extra"]["pool_rng"], np.random.Generator)
    assert result["extra"]["pool_rng"].integers(1000) == np.random.default_rng(9).integers(1000)


def test_actor_state_rejects_incompatible_live_shape_and_schema():
    original = _objects()
    saved = capture_actor_state(*original, segment=1, frame=2, last=3)
    bad = _objects()
    bad[0].mass = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape/dtype"):
        restore_actor_state(saved, *bad)
    saved["schema"] = "wrong"
    with pytest.raises(ValueError, match="unsupported"):
        restore_actor_state(saved, *_objects())
