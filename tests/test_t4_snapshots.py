"""Data-only snapshots preserve state, opponent memory, RNG and batch neighbors."""

import json
from dataclasses import replace

import numpy as np
import pytest

from research.t4_snapshot_restart_20261016 import snapshots as S
from research.t4_snapshot_restart_20261016.build import assert_state_equal
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.vector61_policy import Vector61Runtime


def world(envs=1, seeds=None):
    cfg = BatchSimConfig(
        envs,
        6,
        game_width=300,
        game_height=240,
        initial_food=20,
        max_food=30,
        body_storage_capacity=800,
    )
    sim = GridBatchSim(cfg, seeds=seeds or list(range(envs)), train_mode=False)
    rt = Vector61Runtime(backend="numba")
    controlled = np.ones((envs, 6), bool)
    rt.bind(sim, controlled)
    return sim, rt


def store(tmp_path, snap):
    entry = S.save_snapshot(tmp_path / "state.npz", snap)
    (tmp_path / "manifest.json").write_text(json.dumps(dict(schema=S.SCHEMA, snapshots=[entry])))
    return S.SnapshotStore(tmp_path)


def test_disk_roundtrip_and_300_frame_parity(tmp_path):
    sim, rt = world(seeds=[321])
    for _ in range(7):
        S.step_recorded(sim, rt, np.ones((1, 6), np.int64))
    # Ensure snapshot hero survives even if this fixture changes its initial placement.
    assert sim.alive[0, 0]
    snap = S.capture(sim, rt, [("scripted", "random_safe")] * 5)
    S.save_snapshot(tmp_path / "s.npz", snap)
    other, ort, _, _ = S.restore(S.load_snapshot(tmp_path / "s.npz"))
    assert np.shares_memory(other._owner, other._owner_pad)
    assert_state_equal(sim, rt, other, ort)
    rng = np.random.default_rng(34)
    for _ in range(300):
        actions = rng.integers(0, 6, size=(1, 6))
        S.step_recorded(sim, rt, actions)
        S.step_recorded(other, ort, actions)
        assert_state_equal(sim, rt, other, ort)


@pytest.mark.parametrize("target_capacity", [None, 128])
@pytest.mark.parametrize("legacy_mirror", [False, True])
def test_graft_nonzero_row_preserves_neighbors_and_seeds(tmp_path, target_capacity, legacy_mirror):
    source, sr = world(seeds=[321])
    for _ in range(7):
        S.step_recorded(source, sr, np.ones((1, 6), np.int64))
    snap = S.capture(source, sr, [("scripted", "random_safe")] * 5)
    if legacy_mirror:
        for name in S.FOOD_MIRROR_FIELDS:
            snap["sim"].pop(name)
    bank = store(tmp_path, snap)
    sim, rt = world(5)
    if target_capacity is not None:
        sim._ensure_food_array_capacity(target_capacity)
    assert sim._food_array.shape[1] != source._food_array.shape[1]
    rosters = [[("scripted", "greedy_food")] * 5 for _ in range(5)]
    policies, groups = S.make_policies(sim, rt, rosters)
    before = {k: v.copy() for k, v in vars(sim).items() if isinstance(v, np.ndarray)}
    before_rng = [rng._rng.getstate() for rng in sim._rngs]
    before_runtime = {k: v.copy() for k, v in vars(rt).items() if isinstance(v, np.ndarray)}
    before_memory = {k: v.copy() for k, v in vars(rt.memory).items() if isinstance(v, np.ndarray)}
    policies, groups, restarted = bank.apply(sim, rt, policies, groups, 0.2, 5, 0)
    assert restarted.sum() == 1
    e = int(np.flatnonzero(restarted)[0])
    assert e != 0
    assert sim.frame[e] == 7
    assert sim.get_world_seeds()[e] == 321
    assert policies[("scripted", "random_safe")]._world_seeds[e] == 321
    count = len(sim.food_cells[e])
    assert sim._food_n[e] == count
    np.testing.assert_array_equal(sim._food_array[e, :count], sim.food_cells[e])
    for name, value in before.items():
        if value.shape and value.shape[0] == sim.E:
            actual = getattr(sim, name)[~restarted]
            if name == "_food_array":
                np.testing.assert_array_equal(actual[:, : value.shape[1]], value[~restarted])
                assert not actual[:, value.shape[1] :].any()
            else:
                np.testing.assert_array_equal(actual, value[~restarted])
    for row in np.flatnonzero(~restarted):
        assert sim._rngs[row]._rng.getstate() == before_rng[row]
    for name, value in before_runtime.items():
        np.testing.assert_array_equal(getattr(rt, name)[~restarted], value[~restarted])
    for name, value in before_memory.items():
        np.testing.assert_array_equal(getattr(rt.memory, name)[~restarted], value[~restarted])
    for name, value in vars(sr).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(rt, name)[e], value[0])
    for _ in range(30):
        actions = np.ones((sim.E, 6), np.int64)
        S.step_recorded(source, sr, actions[:1])
        S.step_recorded(sim, rt, actions)
        for name in ("bodies", "length", "frame", "_owner_pad", "_food_pad", "_last_reward"):
            np.testing.assert_array_equal(getattr(sim, name)[e], getattr(source, name)[0])
        assert sim._rngs[e]._rng.getstate() == source._rngs[0]._rng.getstate()
        assert sim.food_cells[e] == source.food_cells[0]
        for name, value in vars(sr).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(rt, name)[e], value[0])
        for name, value in vars(sr.memory).items():
            np.testing.assert_array_equal(getattr(rt.memory, name)[e], value[0])


def test_legacy_snapshot_rebuilds_food_mirror_before_continuation():
    source, runtime = world(seeds=[321])
    actions = np.ones((1, 6), np.int64)
    for _ in range(7):
        S.step_recorded(source, runtime, actions)
    snapshot = S.capture(source, runtime, [("scripted", "random_safe")] * 5)
    for name in S.FOOD_MIRROR_FIELDS:
        snapshot["sim"].pop(name)
    restored, other_runtime, _, _ = S.restore(snapshot)
    np.testing.assert_array_equal(
        restored._food_array[0, : restored._food_n[0]], restored.food_cells[0]
    )
    for _ in range(30):
        S.step_recorded(source, runtime, actions)
        S.step_recorded(restored, other_runtime, actions)
        for name in ("bodies", "length", "frame", "_owner_pad", "_food_pad", "_last_reward"):
            np.testing.assert_array_equal(getattr(restored, name), getattr(source, name))
        assert restored._rngs[0]._rng.getstate() == source._rngs[0]._rng.getstate()
        assert restored.food_cells == source.food_cells
        for name, value in vars(runtime).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(other_runtime, name), value)
        for name, value in vars(runtime.memory).items():
            np.testing.assert_array_equal(getattr(other_runtime.memory, name), value)


def test_store_rejects_corruption_and_deadlines(tmp_path):
    sim, rt = world()
    bank = store(tmp_path, S.capture(sim, rt, [("scripted", "random_safe")] * 5))
    with pytest.raises(ValueError, match="segment end"):
        bank.validate_segment(0)
    with (tmp_path / "state.npz").open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError, match="hash"):
        S.SnapshotStore(tmp_path)


def test_graft_rejects_incompatible_arena():
    source, rt = world()
    snap = S.capture(source, rt, [("scripted", "random_safe")] * 5)
    target = GridBatchSim(replace(source.cfg, mechanics_version=1), seeds=[1])
    with pytest.raises(ValueError, match="mechanics_version"):
        S._graft(target, rt, 0, snap)


def test_restore_pins_opponent_featurizer_parameters(tmp_path, monkeypatch):
    from src.simd_env.vector61_featurizer import Vector61Params

    sim, rt = world()
    snap = S.capture(sim, rt, [("scripted", "random_safe")] * 5)
    S.save_snapshot(tmp_path / "s.npz", snap)

    # A restore must not consult a different process's global GameConfig.
    def forbidden(*args, **kwargs):
        raise AssertionError("global featurizer configuration consulted during restore")

    monkeypatch.setattr(Vector61Params, "from_game_config", forbidden)
    restored, runtime, _, _ = S.restore(S.load_snapshot(tmp_path / "s.npz"))
    assert runtime.featurizer.p == rt.featurizer.p
    assert_state_equal(sim, rt, restored, runtime)
    snap["runtime_params"]["max_length"] += 1
    with pytest.raises(ValueError, match="featurizer parameters"):
        S._graft(sim, rt, 0, snap)
