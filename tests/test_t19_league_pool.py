"""Focused invariants for the static mixed-seat frozen pool."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008.league_pool import (
    LeagueBV8Policy,
    PoolIdentity,
    StaticPool,
    _freeze_policy,
    _normal_fallback_mask,
)


class _Policy:
    def __init__(self, action: int) -> None:
        self.action = action

    def actions(self, masks, prepared, rows):
        return np.full(len(rows), self.action, dtype=np.int64)


class _FrozenNetPolicy(_Policy):
    def __init__(self, action: int) -> None:
        super().__init__(action)
        self.net = torch.nn.Linear(2, 2)
        _freeze_policy(self)


def _identity(policy_id: str) -> PoolIdentity:
    return PoolIdentity(policy_id, "scripted", None, None, "scripted", "resolved")


def test_identity_validates_checkpoint_and_fails_closed_on_hash_mismatch(tmp_path):
    path = tmp_path / "frozen.pth"
    path.write_bytes(b"frozen bytes")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = PoolIdentity("x", "bv8", str(path), digest, "ego2s-bv8", "resolved")
    identity.validate()
    path.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        identity.validate()


def test_partition_is_policy_ordered_and_ignores_learner_or_dead_rows():
    pool = StaticPool([_identity("a"), _identity("b")], {"a": _Policy(2), "b": _Policy(4)})
    seats = np.array([[None, "b", "a"], ["a", None, "b"]], dtype=object)
    alive = np.array([[True, True, False], [True, True, True]])
    groups = pool.groups_for(seats, alive)
    assert list(groups) == ["a", "b"]
    assert groups["a"].tolist() == [[1, 0]]
    assert groups["b"].tolist() == [[0, 1], [1, 2]]


def test_dispatch_scatter_has_stable_inverse_mapping_and_policy_counts():
    pool = StaticPool([_identity("a"), _identity("b")], {"a": _Policy(2), "b": _Policy(4)})
    seats = np.array([[None, "b", "a"], ["a", None, "b"]], dtype=object)
    alive = np.ones((2, 3), dtype=bool)
    actions = np.full((2, 3), 1, dtype=np.int64)
    timing = pool.dispatch(object(), np.ones((2, 3, 6), dtype=bool), alive, actions, seats)
    assert actions.tolist() == [[1, 4, 2], [2, 1, 4]]
    assert timing["frozen_live_rows"] == {"a": 2, "b": 2}
    assert timing["frozen_forward_gpu_s"] == 0.0


def test_bad_seat_id_never_silently_falls_back():
    pool = StaticPool([_identity("a")], {"a": _Policy(0)})
    with pytest.raises(ValueError, match="unknown frozen"):
        pool.groups_for(np.array([["missing"]], dtype=object), np.ones((1, 1), dtype=bool))


def test_dispatch_does_not_mutate_or_enable_gradients_on_actual_frozen_policy_net():
    policy = _FrozenNetPolicy(3)
    before = {name: value.detach().clone() for name, value in policy.net.state_dict().items()}
    pool = StaticPool([_identity("a")], {"a": policy})
    actions = np.ones((1, 1), dtype=np.int64)
    pool.dispatch(
        object(),
        np.ones((1, 1, 6), dtype=bool),
        np.ones((1, 1), dtype=bool),
        actions,
        np.array([["a"]], dtype=object),
    )
    assert not policy.net.training
    assert all(not param.requires_grad for param in policy.net.parameters())
    assert all(torch.equal(value, policy.net.state_dict()[name]) for name, value in before.items())


def test_empty_groups_are_valid_and_do_not_write_actions():
    pool = StaticPool([_identity("a")], {"a": _Policy(2)})
    actions = np.full((1, 2), 1, dtype=np.int64)
    timing = pool.dispatch(
        object(),
        np.ones((1, 2, 6), dtype=bool),
        np.ones((1, 2), dtype=bool),
        actions,
        np.array([[None, None]], dtype=object),
    )
    assert actions.tolist() == [[1, 1]] and timing["frozen_live_rows"] == {}


def test_duplicate_and_over_capacity_pools_fail_before_dispatch():
    with pytest.raises(ValueError, match="unique"):
        StaticPool([_identity("a"), _identity("a")], {"a": _Policy(1)})
    identities = [_identity(f"p{i}") for i in range(17)]
    with pytest.raises(ValueError, match="1..16"):
        StaticPool(identities, {item.policy_id: _Policy(1) for item in identities})


def test_no_valid_mask_fallback_selects_from_normal_actions_only():
    masks = _normal_fallback_mask(torch.zeros((2, 6), dtype=torch.bool))
    assert torch.all(masks[:, :3]) and not torch.any(masks[:, 3:])


def test_rebind_preserves_resident_policy_objects_and_resets_segment_state():
    vector = SimpleNamespace(
        runtime=None,
        veto_counters={1: object()},
        boost_counters={1: object()},
        live_vetoes={1: object()},
        last_veto={"old": True},
    )
    scripted = SimpleNamespace(_world_seeds=(0,))
    identities = [
        PoolIdentity("v", "vector61", "x", "y", "vector61", "advisory"),
        _identity("scripted:greedy_food"),
    ]
    pool = StaticPool(identities, {"v": vector, "scripted:greedy_food": scripted})
    runtime = object()
    pool.rebind(runtime, [5, 7])
    assert pool.policies["v"] is vector and vector.runtime is runtime
    assert not vector.veto_counters and not vector.boost_counters and not vector.live_vetoes
    assert vector.last_veto is None and scripted._world_seeds == (5, 7)


def test_league_bv8_dispatch_keeps_real_loaded_net_frozen(monkeypatch, tmp_path):
    import src.simd_env.ego_raster as ego_raster
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    checkpoint = tmp_path / "b8.pth"
    save_ego2s_checkpoint(checkpoint, Ego2sNet(local_channels=7, n_scalars=30))

    def fake_raster(sim, config, backend, rows):
        n = len(rows)
        return {
            "local": np.zeros((n, 7, 31, 31), dtype=np.uint8),
            "global": np.zeros((n, 4, 37, 37), dtype=np.uint8),
            "scalars": np.zeros((n, 30), dtype=np.float32),
        }

    monkeypatch.setattr(ego_raster, "build_ego_raster", fake_raster)
    policy = LeagueBV8Policy(str(checkpoint), "cpu")
    before = {name: value.detach().clone() for name, value in policy.net.state_dict().items()}
    got = policy.actions(np.zeros((1, 6), dtype=bool), object(), np.array([[0, 0]]))
    assert got.shape == (1,) and got[0] in (0, 1, 2)
    assert not policy.net.training
    assert all(not param.requires_grad for param in policy.net.parameters())
    assert all(torch.equal(value, policy.net.state_dict()[name]) for name, value in before.items())
