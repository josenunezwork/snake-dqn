"""Tests for the fixed, immutable rollout policy interface."""

import numpy as np
import pytest

from src.training.rollout_policies import FixedPolicySource, fixed_policy_actions


class _SparsePolicy:
    identity = "scripted:row-and-slot-v1"

    def actions(self, masks, sim, slots):
        del masks, sim
        return (slots[:, 0] + slots[:, 1]) % 6


class _BadShapePolicy(_SparsePolicy):
    def actions(self, masks, sim, slots):
        del masks, sim, slots
        return np.array([[1]])


class _RawActionPolicy(_SparsePolicy):
    def __init__(self, actions):
        self._actions = actions

    def actions(self, masks, sim, slots):
        del masks, sim, slots
        return self._actions


def test_fixed_source_retains_sparse_slot_order_for_scatter():
    source = FixedPolicySource(_SparsePolicy(), "scripted:row-and-slot-v1")
    slots = np.array([[2, 4], [0, 3], [5, 0]], dtype=np.int64)
    actions = source.actions(np.ones((3, 6), dtype=bool), sim=None, slots=slots)
    full = np.full((6, 6), -1, dtype=np.int64)
    full[slots[:, 0], slots[:, 1]] = actions
    assert actions.tolist() == [0, 3, 5]
    assert full[2, 4] == 0
    assert full[0, 3] == 3
    assert full[5, 0] == 5


def test_fixed_policy_identity_and_output_contract_are_validated():
    with pytest.raises(ValueError, match="identity"):
        FixedPolicySource(_SparsePolicy(), "other")
    with pytest.raises(ValueError, match="shape"):
        fixed_policy_actions(
            _BadShapePolicy(), np.ones((1, 6), dtype=bool), sim=None, slots=np.array([[0, 0]])
        )
    with pytest.raises(ValueError, match="shape"):
        fixed_policy_actions(
            _SparsePolicy(), np.ones((1, 5), dtype=bool), sim=None, slots=np.array([[0, 0]])
        )


@pytest.mark.parametrize("raw", [np.array([1.9]), np.array([True])])
def test_fixed_policy_rejects_fractional_and_boolean_actions_before_cast(raw):
    with pytest.raises(ValueError, match="integral|boolean"):
        fixed_policy_actions(
            _RawActionPolicy(raw), np.ones((1, 6), dtype=bool), sim=None, slots=np.array([[0, 0]])
        )


def test_fixed_source_fails_closed_if_wrapped_identity_changes():
    policy = _SparsePolicy()
    source = FixedPolicySource(policy, policy.identity)
    policy.identity = "scripted:mutated"
    with pytest.raises(ValueError, match="changed"):
        source.actions(np.ones((1, 6), dtype=bool), sim=None, slots=np.array([[0, 0]]))
