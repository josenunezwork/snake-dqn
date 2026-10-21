"""Focused contract tests for the E4/E2 command boundary."""

import pytest

from research.bl_ppo.exports import export_json
from research.bl_ppo.run import E4_SCHEMA, e4_target_transitions


@pytest.mark.parametrize(
    "rate, expected",
    [(0, 4_000_000), (999.9, 4_000_000), (1000, 8_000_000), (3999, 8_000_000), (4000, 8_000_000)],
)
def test_e4_receipt_binds_exact_e2_budget(tmp_path, rate, expected):
    receipt = tmp_path / "e4.json"
    export_json(
        receipt,
        {
            "schema": E4_SCHEMA,
            "status": "COMPLETE",
            "scientific": True,
            "worlds": 128,
            "long_bodies": True,
            "selected_processes": 6,
            "configs": [
                {"processes": 6, "allocated_hero_transitions_per_second": rate},
                {"processes": 12, "allocated_hero_transitions_per_second": rate},
            ],
        },
    )
    assert e4_target_transitions(receipt) == expected


def test_e4_refuses_incomplete_or_changed_receipt(tmp_path):
    receipt = tmp_path / "e4.json"
    export_json(
        receipt,
        {
            "schema": E4_SCHEMA,
            "status": "INCOMPLETE",
            "selected_processes": 6,
            "configs": [
                {"processes": 6, "allocated_hero_transitions_per_second": 4000},
                {"processes": 12, "allocated_hero_transitions_per_second": 4000},
            ],
        },
    )
    with pytest.raises(ValueError, match="complete"):
        e4_target_transitions(receipt)


def test_e4_rejects_request_that_does_not_match_frozen_switch(tmp_path):
    receipt = tmp_path / "e4.json"
    export_json(
        receipt,
        {
            "schema": E4_SCHEMA,
            "status": "COMPLETE",
            "scientific": True,
            "worlds": 128,
            "long_bodies": True,
            "selected_processes": 6,
            "configs": [
                {"processes": 6, "allocated_hero_transitions_per_second": 4000},
                {"processes": 12, "allocated_hero_transitions_per_second": 4000},
            ],
        },
    )
    with pytest.raises(ValueError, match="exactly"):
        e4_target_transitions(receipt, 20_000_000)


def test_recurrent_refresh_preserves_next_rollout_log_probabilities() -> None:
    from types import SimpleNamespace

    import numpy as np
    import torch
    from torch.distributions import Categorical

    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PPOPolicy
    from research.bl_ppo.run import PersistentPolicyAdapter
    from src.model.ego2s_network import Ego2sNet

    torch.manual_seed(51)
    policy = PPOPolicy("R", None, 0.7, b8_model=Ego2sNet(7, 60))
    adapter = PersistentPolicyAdapter(policy, "cpu")
    obs = {
        "local": np.zeros((1, 7, 31, 31), dtype=np.uint8),
        "global": np.zeros((1, 4, 37, 37), dtype=np.uint8),
        "scalars": np.zeros((1, 60), dtype=np.float32),
        "mask": np.ones((1, 6), dtype=bool),
        "reset": np.array([False]),
        "frame": np.array([0], dtype=np.int32),
    }

    def collect(steps):
        for _ in range(steps):
            adapter(obs, obs["reset"])
        raw = SimpleNamespace(
            active_transitions=steps,
            local=np.repeat(obs["local"], steps, axis=0),
            global_=np.repeat(obs["global"], steps, axis=0),
            scalars=np.repeat(obs["scalars"], steps, axis=0),
            reward=np.ones(steps, dtype=np.float32),
            done=np.zeros(steps, dtype=bool),
            frame=np.arange(steps, dtype=np.int32),
            world_id=np.zeros(steps, dtype=np.int64),
        )
        return adapter.take_rollout(raw, obs)

    collect(34)
    with torch.no_grad():
        policy.core.bias_ih_l0.add_(0.5)
        policy.residual_policy.weight.normal_(std=0.05)
    adapter.refresh_hidden()
    rollout = collect(2)
    assert rollout.actions.shape == (34, 1)
    assert rollout.mask().sum() == 2
    logits, _ = PPOLearner(policy)._forward_rollout(rollout)
    likelihood = Categorical(logits=logits).log_prob(rollout.actions)
    assert torch.equal(likelihood[32:], rollout.logp[32:])


def test_recovery_requires_fresh_output_before_loading_or_collecting(tmp_path) -> None:
    from types import SimpleNamespace

    from research.bl_ppo.run import run_train

    (tmp_path / "monitor-0001.json").write_text("preserved")
    with pytest.raises(ValueError, match="fresh output directory"):
        run_train(SimpleNamespace(output=tmp_path))
    assert (tmp_path / "monitor-0001.json").read_text() == "preserved"
