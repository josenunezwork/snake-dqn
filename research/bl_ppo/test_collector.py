"""Focused collector contracts (kept with the development package)."""

import numpy as np

from research.bl_ppo.collector import (
    CollectorConfig,
    PersistentCollector,
    _world_seed,
    make_eval_driver,
)


def test_preflight_and_partition_stable_seeds() -> None:
    assert CollectorConfig(worlds=2, processes=1, preflight=True).worlds == 2
    base = CollectorConfig(seed=91)
    roots = [_world_seed(base, world, 0) for world in range(128)]
    assert len(set(roots)) == 128
    assert (
        _world_seed(CollectorConfig(seed=91, excluded_seeds=frozenset({roots[7]})), 7, 0)
        != roots[7]
    )


def test_terminal_driver_bv8_poststep_mass() -> None:
    driver = make_eval_driver(17, [("scripted", "greedy_food")] * 5, horizon=2)
    obs = driver.observe_hero()
    assert obs["local"].shape == (1, 7, 31, 31)
    assert obs["scalars"].shape == (1, 60)
    assert obs["mask"].dtype == np.bool_
    assert driver.step(1)["frame"] == 1


def test_persistent_preflight_peek_is_fresh_and_world_ordered() -> None:
    config = CollectorConfig(worlds=2, processes=1, preflight=True, rollout_steps=1)
    with PersistentCollector(config) as collector:
        batch = collector.collect(lambda obs, reset: np.ones(len(obs["world_id"]), dtype=np.int64))
        nxt = collector.peek()
    assert batch.world_id.tolist() == [0, 1]
    assert nxt["world_id"].tolist() == [0, 1]
    assert nxt["scalars"].shape == (2, 60)
    assert batch.survived_horizon.shape == (2,)
    assert batch.survived_horizon.dtype == np.bool_
    assert not batch.survived_horizon.any()
    assert batch.length.shape == (2,)
    assert set(batch.mix.tolist()) <= {"scripted", "frozen", "mixed"}
    assert batch.episode.tolist() == [0, 0]


def test_real_native_snapshot_replays_next_transition() -> None:
    import pickle

    from research.bl_ppo.collector import _GridShard

    config = CollectorConfig(worlds=2, processes=1, preflight=True, native_v8=True)
    shard = _GridShard(config, 0, np.arange(2))
    shard.observe()
    shard.step(np.ones(2, dtype=np.int64))
    shard.observe()
    saved = pickle.dumps(shard.snapshot())
    expected = []
    for _ in range(3):
        expected.append(
            (
                shard.step(np.ones(2, dtype=np.int64)),
                shard.observe(),
                shard.sim.bodies.copy(),
            )
        )
    shard.restore(pickle.loads(saved))
    for name in ("_owner", "_slot", "_food"):
        assert np.shares_memory(getattr(shard.sim, name), getattr(shard.sim, name + "_pad"))
    for transition, observation, bodies in expected:
        observed = shard.step(np.ones(2, dtype=np.int64))
        for key in transition:
            assert np.array_equal(transition[key], observed[key]), key
        current = shard.observe()
        for key in observation:
            assert np.array_equal(observation[key], current[key]), key
        assert np.array_equal(bodies, shard.sim.bodies)
    assert shard.frozen.native_v8_actual == "s2-native-v8"


def test_real_worlds_are_partition_independent() -> None:
    from research.bl_ppo.collector import _GridShard

    cfg = CollectorConfig(worlds=2, processes=1, preflight=True)
    together = _GridShard(cfg, 0, np.arange(2))
    apart = [_GridShard(cfg, index, np.array([index])) for index in range(2)]
    for _ in range(3):
        together.observe()
        for shard in apart:
            shard.observe()
        whole = together.step(np.ones(2, dtype=np.int64))
        pieces = [shard.step(np.ones(1, dtype=np.int64)) for shard in apart]
        for key in whole:
            assert np.array_equal(whole[key], np.concatenate([v[key] for v in pieces])), key
