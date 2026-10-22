"""Focused E8 complete-life critic-refit evidence tests."""

from __future__ import annotations

import pytest
import torch

from research.bl_ppo.refit import CompleteLifeRefitBank


def _add_complete_lives(
    bank: CompleteLifeRefitBank,
    worlds: list[int],
    mixes: list[str],
    *,
    source: str = "natural",
) -> None:
    """Add four retained L>=800 rows for each completed synthetic test life."""
    assert len(worlds) == len(mixes)
    world_ids = torch.tensor([world for world in worlds for _ in range(4)])
    feature_values = torch.arange(len(world_ids), dtype=torch.float32)
    features = torch.zeros(len(world_ids), 257)
    features[:, 0] = feature_values
    bank.add_batch(
        world_ids,
        features,
        torch.full((len(world_ids),), 900.0),
        torch.tensor([0.3, 0.4, 0.5, 0.6] * len(worlds)),
        torch.ones(len(world_ids)),
        torch.tensor([False, False, False, True] * len(worlds)),
        mixes=[mix for mix in mixes for _ in range(4)],
        source=source,
    )


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
    result = bank.qualification(learner, require_support_minimum=True)
    assert not result["passed"]
    assert result["late_body_rows"] == result["fit_support"] == result["holdout_support"] == 3
    assert result["mse"] > result["baseline_mse"]
    assert result["explained_variance"] < 0.1
    assert result["status"] == "INSUFFICIENT_LATE_BODY_SUPPORT"
    assert result["passed"] is False


def test_p1_support_requires_every_registered_holdout_mix_and_quarantines_fixture() -> None:
    bank = CompleteLifeRefitBank(samples_per_life=4)
    # A constructed fixture has many late rows but cannot count toward natural
    # P1 support or leak into the natural diagnostic export.
    bank.add_batch(
        torch.tensor([0, 0, 0, 0]),
        torch.zeros(4, 257),
        torch.tensor([900, 900, 900, 900]),
        torch.tensor([0.3, 0.4, 0.5, 0.6]),
        torch.ones(4),
        torch.tensor([False, False, False, True]),
        mixes=["scripted"] * 4,
        split_override="holdout",
        source="fixed_injected_long_body",
        pin_first_late=True,
    )
    support = bank.natural_support("holdout")
    assert support["late_rows"] == 0
    assert support["lives_by_mix"] == {"scripted": 0, "frozen": 0, "mixed": 0}
    assert support["meets_minimum"] is False
    assert bank.diagnostic_rows()["features257"].shape == (0, 257)
    table = bank.support_table("holdout")
    assert table[0]["provenance"] == "fixed_injected_long_body"
    assert table[0]["remaining_horizon_bucket"] == "<=1500"


def test_p1_support_exact_thresholds_count_distinct_natural_lives() -> None:
    bank = CompleteLifeRefitBank(samples_per_life=4)
    fit_worlds = [1 + 5 * index for index in range(160)]
    _add_complete_lives(bank, fit_worlds, ["scripted"] * len(fit_worlds))
    holdout_worlds = [5 * (index + 1) for index in range(40)]
    holdout_mixes = ["scripted"] * 14 + ["frozen"] * 13 + ["mixed"] * 13
    _add_complete_lives(bank, holdout_worlds, holdout_mixes)

    fit, holdout = bank.natural_support("fit"), bank.natural_support("holdout")
    assert (fit["completed_late_lives"], fit["late_rows"], fit["meets_minimum"]) == (
        160,
        640,
        True,
    )
    assert (holdout["completed_late_lives"], holdout["late_rows"], holdout["meets_minimum"]) == (
        40,
        160,
        True,
    )
    assert holdout["lives_by_mix"] == {"scripted": 14, "frozen": 13, "mixed": 13}


def test_p1_support_refuses_missing_mix_and_row_duplicates() -> None:
    bank = CompleteLifeRefitBank(samples_per_life=8)
    # Eight late rows from a single completed world remain one life, while 40
    # lives without mixed opponents still fail the registered per-mix floor.
    _add_complete_lives(bank, [1], ["scripted"])
    _add_complete_lives(
        bank,
        [5 * (index + 1) for index in range(40)],
        ["scripted"] * 20 + ["frozen"] * 20,
    )
    fit, holdout = bank.natural_support("fit"), bank.natural_support("holdout")
    assert fit["completed_late_lives"] == 1
    assert fit["late_rows"] == 4
    assert fit["meets_minimum"] is False
    assert holdout["completed_late_lives"] == 40
    assert holdout["late_rows"] == 160
    assert holdout["lives_by_mix"]["mixed"] == 0
    assert holdout["meets_minimum"] is False


def test_quality_pass_below_p1_support_never_admits_actor() -> None:
    from types import SimpleNamespace

    from research.bl_ppo.learner import PPOLearner

    bank = CompleteLifeRefitBank(samples_per_life=4)
    # Different full-life returns at identical h/L leave a nonzero (1,h,L)
    # baseline residual; the frozen feature value is the exact suffix return.
    for world, reward in zip((1, 2, 3, 4), (1.0, 4.0, 9.0, 16.0)):
        bank.add_batch(
            torch.tensor([world]),
            torch.tensor([[reward] + [0.0] * 256]),
            torch.tensor([900.0]),
            torch.tensor([0.5]),
            torch.tensor([reward]),
            torch.tensor([True]),
            mixes=["scripted"],
        )
    bank.add_batch(
        torch.tensor([0]),
        torch.tensor([[7.0] + [0.0] * 256]),
        torch.tensor([900.0]),
        torch.tensor([0.5]),
        torch.tensor([7.0]),
        torch.tensor([True]),
        mixes=["scripted"],
    )
    admitted = False

    def qualify(**_kwargs):
        nonlocal admitted
        admitted = True

    learner = SimpleNamespace(
        policy=SimpleNamespace(
            critic=torch.nn.Linear(257, 1),
            value_from_critic_features=lambda features: features[:, 0],
        ),
        critic_quality=PPOLearner.critic_quality,
        qualify_critic=qualify,
    )
    result = bank.qualification(learner, require_support_minimum=True)
    assert result["accuracy_passed"] is True
    assert result["passed"] is False
    assert result["status"] == "INSUFFICIENT_LATE_BODY_SUPPORT"
    assert admitted is False


def test_fixture_rows_are_excluded_from_popart_refit_and_scored_separately() -> None:
    from types import SimpleNamespace

    natural, fixture = CompleteLifeRefitBank(samples_per_life=4), CompleteLifeRefitBank(
        samples_per_life=4
    )
    _add_complete_lives(natural, [1], ["scripted"])
    _add_complete_lives(fixture, [0], ["scripted"], source="fixed_injected_long_body")
    captured: dict[str, torch.Tensor] = {}

    class FitLearner:
        def fit_completed_critic(self, features, returns):
            captured["features"], captured["returns"] = features, returns
            return 0.0

    natural.fit_complete_targets(FitLearner())
    assert captured["features"].shape[0] == 4
    assert torch.equal(captured["features"][:, 0], torch.arange(4, dtype=torch.float32))

    from research.bl_ppo.learner import PPOLearner

    learner = SimpleNamespace(
        policy=SimpleNamespace(
            critic=torch.nn.Linear(257, 1),
            value_from_critic_features=lambda features: torch.zeros(len(features)),
        ),
        critic_quality=PPOLearner.critic_quality,
    )
    score = natural.score_fixture_holdout(learner, fixture)
    assert score["fixture_late_body_rows"] == 4.0
    assert score["metrics_available"] is True
    assert score["diagnostic_only"] is True
    assert score["actor_admission"] is False


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
