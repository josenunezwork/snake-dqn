"""Focused E8 complete-life critic-refit evidence tests."""

from __future__ import annotations

import pytest
import torch

from research.bl_ppo.refit import CompleteLifeRefitBank


@pytest.mark.parametrize("prediction", [0.0, 1e30])
def test_failed_e8_retains_measured_metrics_and_support_without_admitting_actor(prediction) -> None:
    from types import SimpleNamespace

    from research.bl_ppo.learner import PPOLearner

    bank = CompleteLifeRefitBank()
    bank.add_batch(
        torch.tensor([1, 0, 1, 0, 1, 0]),
        torch.zeros(6, 257),
        torch.tensor([800, 800, 900, 900, 1000, 1000]),
        torch.tensor([1.0, 1.0, 0.8, 0.8, 0.6, 0.6]),
        torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        torch.tensor([False, False, False, False, True, True]),
    )

    def reject_admission(**kwargs):
        raise AssertionError("failed gate must never admit actor updates")

    learner = SimpleNamespace(
        policy=SimpleNamespace(
            critic=torch.nn.Linear(257, 1),
            value_from_critic_features=lambda features: torch.full((len(features),), prediction),
        ),
        critic_quality=PPOLearner.critic_quality,
        qualify_critic=reject_admission,
    )
    if prediction == 1e30:
        with pytest.raises(FloatingPointError, match="non-finite E8 qualification metrics"):
            bank.qualification(learner)
        return
    result = bank.qualification(learner)
    assert not result["passed"]
    assert result["late_body_rows"] == result["fit_support"] == result["holdout_support"] == 3
    assert result["mse"] > result["baseline_mse"]
    assert result["explained_variance"] < 0.1


def test_bank_uses_complete_gamma_one_lives_and_world_disjoint_baseline() -> None:
    bank = CompleteLifeRefitBank()
    # World 1 is fit; its terminal suffixes are 6, 5, 3.  World 0 is held out.
    bank.add_batch(
        torch.tensor([1, 0, 1, 0, 1, 0]),
        torch.zeros((6, 257)),
        torch.tensor([800, 800, 900, 900, 1000, 1000]),
        torch.tensor([1.0, 1.0, 0.8, 0.8, 0.6, 0.6]),
        torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        torch.tensor([False, False, False, False, True, True]),
    )
    assert bank.counts() == {"fit": 3, "holdout": 3}
    assert sorted(value for _, _, value in bank._reservoir["fit"]) == [3.0, 5.0, 6.0]
    assert sorted(value for _, _, value in bank._reservoir["holdout"]) == [3.0, 5.0, 6.0]


def test_bank_excludes_open_life_from_counts() -> None:
    bank = CompleteLifeRefitBank()
    bank.add_batch(
        torch.tensor([0, 0]),
        torch.zeros((2, 257)),
        torch.tensor([800, 900]),
        torch.tensor([1.0, 0.8]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([False, False]),
    )
    assert bank.counts() == {"fit": 0, "holdout": 0}
    assert bank.support_counts() == {"fit": 0, "holdout": 0}


def test_pinned_late_fixture_survives_a_full_reservoir() -> None:
    """The prospective fixed/supplemental root cannot be sampled away."""
    bank = CompleteLifeRefitBank(reservoir_size=1)
    bank.add_batch(
        torch.tensor([0, 0]),
        torch.zeros((2, 257)),
        torch.tensor([900, 900]),
        torch.tensor([1.0, 0.9]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([False, True]),
        split_override="holdout",
        pin_first_late=True,
        source="fixed_injected_long_body",
    )
    assert bank.late_support_by_source() == {"fixed_injected_long_body": 1}
