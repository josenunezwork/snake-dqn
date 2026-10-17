"""Focused invariants for the static mixed-seat frozen pool."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008.league_pool import PoolIdentity, StaticPool


class _Policy:
    def __init__(self, action: int) -> None:
        self.action = action

    def actions(self, masks, prepared, rows):
        return np.full(len(rows), self.action, dtype=np.int64)


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


def test_freeze_is_not_mutated_by_static_pool_dispatch():
    # This guards the key collection invariant: dispatch has no optimizer/update path.
    tensor = torch.tensor([1.0], requires_grad=True)
    before = tensor.detach().clone()
    pool = StaticPool([_identity("a")], {"a": _Policy(3)})
    actions = np.ones((1, 1), dtype=np.int64)
    pool.dispatch(
        object(),
        np.ones((1, 1, 6), dtype=bool),
        np.ones((1, 1), dtype=bool),
        actions,
        np.array([["a"]], dtype=object),
    )
    assert torch.equal(tensor.detach(), before) and tensor.requires_grad
