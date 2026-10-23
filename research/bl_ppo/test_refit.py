"""Focused E8 complete-life critic-refit evidence tests."""

from __future__ import annotations

import numpy as np
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


def test_amendment_03_solves_natural_fit_rows_in_float64_and_resets_only_head_state() -> None:
    """The prospective solve is bounded, raw-return, and CPU-installable."""
    from types import SimpleNamespace

    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PopArtValueHead

    class Policy:
        def __init__(self) -> None:
            self.critic = PopArtValueHead(257)
            self.actor_parameter = torch.nn.Parameter(torch.tensor(2.0))

        def value_from_critic_features(self, features: torch.Tensor) -> torch.Tensor:
            return self.critic(features)

    policy = Policy()
    learner = SimpleNamespace(
        policy=policy,
        optimizer=torch.optim.Adam([*policy.critic.parameters(), policy.actor_parameter], lr=1e-3),
    )
    # Make Adam hold state that the analytic rewrite must discard.
    (policy.critic.linear.weight.sum() + policy.actor_parameter).backward()
    learner.optimizer.step()
    assert learner.optimizer.state
    actor_state = {
        key: value.clone() for key, value in learner.optimizer.state[policy.actor_parameter].items()
    }
    generator = torch.Generator().manual_seed(41)
    features = torch.randn((280, 257), generator=generator, dtype=torch.float64)
    coefficients = torch.linspace(-0.5, 0.5, 258, dtype=torch.float64)
    returns = torch.cat((torch.ones((280, 1), dtype=torch.float64), features), 1) @ coefficients
    report = PPOLearner.install_exact_return_critic_solution(learner, features, returns)
    predictions = policy.value_from_critic_features(features.float()).double()
    design = torch.cat((torch.ones((280, 1), dtype=torch.float64), features), 1).numpy()
    coefficients, *_ = np.linalg.lstsq(design, returns.numpy(), rcond=None)
    stored_mean, stored_scale = float(policy.critic.mean), float(policy.critic.scale)
    assert report["fit_count"] == 280
    assert report["rank"] == 258
    assert report["optimizer_head_state_reset"] is True
    assert report["installed_fp32_fit_mse"] < 1e-10
    assert stored_mean != float(returns.mean())
    assert stored_scale != float(torch.sqrt(((returns - returns.mean()) ** 2).mean()))
    assert torch.equal(
        policy.critic.linear.weight,
        torch.as_tensor(coefficients[1:] / stored_scale, dtype=torch.float32).reshape(1, 257),
    )
    assert torch.equal(
        policy.critic.linear.bias,
        torch.as_tensor([(coefficients[0] - stored_mean) / stored_scale], dtype=torch.float32),
    )
    assert torch.allclose(predictions, returns, atol=2e-5, rtol=2e-5)
    assert set(learner.optimizer.state) == {policy.actor_parameter}
    assert learner.optimizer.state[policy.actor_parameter].keys() == actor_state.keys()
    for key, value in actor_state.items():
        assert torch.equal(learner.optimizer.state[policy.actor_parameter][key], value)


def test_amendment_03_rejects_finite_float64_values_that_overflow_popart_fp32() -> None:
    from types import SimpleNamespace

    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PopArtValueHead

    critic = PopArtValueHead(257)
    learner = SimpleNamespace(
        policy=SimpleNamespace(
            critic=critic,
            value_from_critic_features=lambda features: critic(features),
        ),
        optimizer=torch.optim.Adam(critic.parameters(), lr=1e-3),
    )
    with pytest.raises(FloatingPointError, match="fp32 PopArt installation"):
        PPOLearner.install_exact_return_critic_solution(
            learner, torch.zeros((2, 257), dtype=torch.float64), torch.full((2,), 1e30)
        )


def test_amendment_04_zero_sum_solve_is_stable_and_repeatable() -> None:
    """C1 uses the fixed Helmert basis and retains fp32 PopArt mapping."""
    from types import SimpleNamespace

    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PopArtValueHead

    class Policy:
        def __init__(self) -> None:
            self.critic = PopArtValueHead(257)
            self.core = SimpleNamespace(norm=torch.nn.LayerNorm(256))

        def value_from_critic_features(self, features):
            return self.critic(features)

    generator = torch.Generator().manual_seed(20261022)
    core = torch.randn((300, 256), generator=generator, dtype=torch.float64)
    core.sub_(core.mean(dim=1, keepdim=True))
    features = torch.cat((core, torch.linspace(0.0, 1.0, 300).double().unsqueeze(1)), dim=1)
    returns = core[:, 0] * 2.0 + features[:, -1] * 3.0

    def solve():
        policy = Policy()
        learner = SimpleNamespace(
            policy=policy, optimizer=torch.optim.Adam(policy.critic.parameters())
        )
        report = PPOLearner.install_exact_return_critic_solution(
            learner, features, returns, zero_sum_basis=True
        )
        return policy, report

    policy_a, report_a = solve()
    policy_b, report_b = solve()
    assert report_a["rank"] == 257
    assert report_a["basis"] == "helmert-zero-sum-256-v1"
    assert abs(report_a["common_mode_sensitivity_per_1e-4_bias"]) <= 1.0
    assert report_a["installed_fp32_fit_mse"] < 1e-8
    assert torch.equal(policy_a.critic.linear.weight, policy_b.critic.linear.weight)
    assert torch.equal(policy_a.critic.linear.bias, policy_b.critic.linear.bias)
    assert report_a == report_b


def test_amendment_04_rank_deficient_report_is_json_safe_and_refuses_nonzero_mean() -> None:
    """C1's rank-deficient path stays finite and the zero-sum guard is strict."""
    import json
    from types import SimpleNamespace

    from research.bl_ppo.learner import PPOLearner
    from research.bl_ppo.model import PopArtValueHead

    critic = PopArtValueHead(257)
    policy = SimpleNamespace(
        critic=critic,
        core=SimpleNamespace(norm=torch.nn.LayerNorm(256)),
        value_from_critic_features=lambda features: critic(features),
    )
    learner = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(critic.parameters()))
    features = torch.zeros((8, 257), dtype=torch.float64)
    report = PPOLearner.install_exact_return_critic_solution(
        learner, features, torch.arange(8, dtype=torch.float64), zero_sum_basis=True
    )
    assert report["rank"] == 1
    assert report["condition_number"] is None
    json.dumps(report, allow_nan=False)
    features[0, 0] = 1e-3
    with pytest.raises(ValueError, match="not zero-sum"):
        PPOLearner.install_exact_return_critic_solution(
            learner, features, torch.arange(8, dtype=torch.float64), zero_sum_basis=True
        )


def test_amendment_03_excludes_supplement_and_preserves_raw_return_dtype() -> None:
    bank = CompleteLifeRefitBank(samples_per_life=1)
    bank.add_batch(
        torch.tensor([1]),
        torch.zeros((1, 257)),
        torch.tensor([900.0]),
        torch.tensor([0.5]),
        torch.tensor([3.25]),
        torch.tensor([True]),
        source="natural",
    )
    bank.add_batch(
        torch.tensor([2]),
        torch.zeros((1, 257)),
        torch.tensor([900.0]),
        torch.tensor([0.5]),
        torch.tensor([99.0]),
        torch.tensor([True]),
        source="refit_long_body_support_supplement",
    )
    captured: dict[str, torch.Tensor] = {}

    class Learner:
        def install_exact_return_critic_solution(self, features, returns):
            captured["features"] = features
            captured["returns"] = returns
            return {"fit_count": int(returns.numel())}

    assert bank.solve_e8_critic_convergence_amendment_03(Learner()) == {"fit_count": 1}
    assert captured["features"].dtype == torch.float64
    assert captured["returns"].dtype == torch.float64
    assert captured["returns"].tolist() == [3.25]


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


def test_large_unscaled_fit_baseline_cannot_false_pass_biased_critic() -> None:
    """Default float32 rank truncation previously admitted this worse critic."""
    from research.bl_ppo.learner import PPOLearner, fit_time_length_baseline

    count = 10_000
    h = torch.linspace(0, 1, count)
    length = 800 + 400 * ((torch.arange(count) * 29) % 997).float() / 997
    returns = 10_000 + 3000 * h + 50 * length
    # Fit and holdout are disjoint rows, each spanning all three columns.
    fit, held = torch.arange(count) % 5 != 0, torch.arange(count) % 5 == 0
    coefficients = fit_time_length_baseline(length[fit], h[fit], returns[fit])
    assert coefficients.dtype == torch.float64
    assert coefficients.tolist() == pytest.approx([10_000, 3000, 50], rel=1e-6)
    quality = PPOLearner.critic_quality(
        returns[held] + 1000,
        returns[held],
        length[held],
        h[held],
        baseline_coefficients=coefficients,
    )
    assert quality["explained_variance"] > 0.99
    assert quality["baseline_mse"] < 0.001
    assert quality["mse"] > 999_000
    assert not quality["passed"]
