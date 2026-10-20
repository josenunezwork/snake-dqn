"""Independent-root bucketing preserves row order and private action masks."""

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
