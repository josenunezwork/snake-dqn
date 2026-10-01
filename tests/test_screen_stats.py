"""Analytical-oracle tests for src/evaluation/screen_stats.py."""

import math
from statistics import NormalDist

import numpy as np
import pytest

from src.evaluation.screen_stats import (
    FutilityPlan,
    chi2_sf,
    crossed_seed_world_mean,
    curve_auc,
    evaluate_futility_look,
    futility_plan,
    futility_stop_probability,
    gate_pass_probabilities,
    gate_pass_probabilities_for_effect,
    last_k_mean,
    learning_curve_contrast,
    obf_spending,
    per_seed_effects,
    random_effects_pool,
    run_futility_sequence,
    seeds_for_pooled_power,
    stratified_mean_of_means,
)
from src.scripts.eval_stats import student_t_isf, student_t_sf

N = NormalDist()
Z975 = N.inv_cdf(0.975)


# --------------------------------------------------------------------------- distributions


@pytest.mark.parametrize("x", [0.1, 1.0, 3.0, 14.0, 60.0])
def test_chi2_sf_closed_forms(x):
    assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2.0), rel=1e-10)
    assert chi2_sf(x, 1) == pytest.approx(2.0 * (1.0 - N.cdf(math.sqrt(x))), rel=1e-8, abs=1e-15)
    df3 = 2.0 * (1.0 - N.cdf(math.sqrt(x))) + math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0)
    assert chi2_sf(x, 3) == pytest.approx(df3, rel=1e-8, abs=1e-15)


def test_chi2_sf_edges():
    assert chi2_sf(0.0, 4) == 1.0
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0)


# --------------------------------------------------------------------------- pooling


def test_per_seed_effects_mean_and_squared_se():
    out = per_seed_effects([[1.0, 3.0], [0.0, 0.0, 6.0]])
    assert out["effects"] == pytest.approx([2.0, 2.0])
    assert out["variances"] == pytest.approx([2.0 / 2.0, 12.0 / 3.0])
    assert out["counts"] == [2, 3]
    with pytest.raises(ValueError):
        per_seed_effects([[1.0]])


def test_dersimonian_laird_hand_computed():
    # Equal unit variances: mu_FE = 3, Q = 14, C = 3, tau^2 = 11/3, I^2 = 11/14.
    y, v = [1.0, 2.0, 3.0, 6.0], [1.0, 1.0, 1.0, 1.0]
    out = random_effects_pool(y, v, ci_method="z")
    se = math.sqrt(7.0 / 6.0)
    assert out["mu_fixed"] == pytest.approx(3.0)
    assert out["q"] == pytest.approx(14.0)
    assert out["tau2"] == pytest.approx(11.0 / 3.0)
    assert out["i2"] == pytest.approx(11.0 / 14.0)
    assert out["mu"] == pytest.approx(3.0)
    assert out["se"] == pytest.approx(se)
    assert out["ci"] == pytest.approx([3.0 - Z975 * se, 3.0 + Z975 * se])
    df3 = 2.0 * (1.0 - N.cdf(math.sqrt(14.0))) + math.sqrt(28.0 / math.pi) * math.exp(-7.0)
    assert out["q_p_value"] == pytest.approx(df3, rel=1e-8)
    assert out["weights"] == pytest.approx([0.25] * 4)
    # HKSJ: q = sum(w*(y-mu)^2)/(k-1) = 1 here, so SE is unchanged but the critical value is t_3.
    hk = random_effects_pool(y, v, ci_method="hksj")
    t3 = 3.182446305284263
    assert hk["se"] == pytest.approx(se)
    assert hk["critical"] == pytest.approx(t3, rel=1e-9)
    assert hk["ci"] == pytest.approx([3.0 - t3 * se, 3.0 + t3 * se])
    assert hk["p_value"] == pytest.approx(2.0 * student_t_sf(3.0 / se, 3))
    # Prediction interval: mu +/- t_2 sqrt(tau^2 + SE^2).
    half = student_t_isf(0.025, 2) * math.sqrt(11.0 / 3.0 + 7.0 / 6.0)
    assert hk["prediction_interval"] == pytest.approx([3.0 - half, 3.0 + half])


def test_homogeneous_effects_reduce_to_fixed_effect():
    y, v = np.array([1.0, 1.1, 0.9]), np.array([1.0, 2.0, 4.0])
    w = 1.0 / v
    out = random_effects_pool(y, v, ci_method="z")
    assert out["tau2"] == 0.0 and out["i2"] == 0.0
    assert out["mu"] == pytest.approx(float((w * y).sum() / w.sum()))
    assert out["se"] == pytest.approx(math.sqrt(1.0 / w.sum()))
    # q_HK < 1 here; the floored variant must not be narrower than DL.
    raw = random_effects_pool(y, v, ci_method="hksj")
    floor = random_effects_pool(y, v, ci_method="hksj_floor")
    assert raw["se"] < out["se"] == pytest.approx(floor["se"])


def test_pool_rejects_bad_input():
    with pytest.raises(ValueError):
        random_effects_pool([1.0], [1.0])
    with pytest.raises(ValueError):
        random_effects_pool([1.0, 2.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        random_effects_pool([1.0, 2.0], [1.0, 1.0], ci_method="bogus")


def test_hksj_floor_coverage_with_few_heterogeneous_seeds():
    rng = np.random.default_rng(7)
    k, tau2, mu = 5, 1.0, 0.3
    covered_hk = covered_z = 0
    sims = 800
    for _ in range(sims):
        v = rng.uniform(0.2, 1.0, k)
        y = mu + rng.normal(0.0, math.sqrt(tau2), k) + rng.normal(0.0, np.sqrt(v))
        lo, hi = random_effects_pool(y, v)["ci"]
        covered_hk += lo <= mu <= hi
        lo, hi = random_effects_pool(y, v, ci_method="z")["ci"]
        covered_z += lo <= mu <= hi
    assert covered_hk / sims >= 0.92
    assert covered_z < covered_hk  # the textbook z interval under-covers at k = 5


def _two_way_mean_squares(m):
    s, w = m.shape
    g = m.mean()
    ms_s = w * sum((m[i].mean() - g) ** 2 for i in range(s)) / (s - 1)
    ms_w = s * sum((m[:, j].mean() - g) ** 2 for j in range(w)) / (w - 1)
    ss_e = sum(
        (m[i, j] - m[i].mean() - m[:, j].mean() + g) ** 2 for i in range(s) for j in range(w)
    )
    return ms_s, ms_w, ss_e / ((s - 1) * (w - 1))


def test_crossed_mean_variance_formula():
    rng = np.random.default_rng(3)
    m = 2.0 + rng.normal(0, 3, (4, 1)) + rng.normal(0, 5, (1, 12)) + rng.normal(0, 1, (4, 12))
    out = crossed_seed_world_mean(m)
    ms_s, ms_w, ms_e = _two_way_mean_squares(m)
    assert ms_s > ms_e and ms_w > ms_e
    var = (ms_s + ms_w - ms_e) / m.size
    df = (ms_s + ms_w - ms_e) ** 2 / (ms_s**2 / 3 + ms_w**2 / 11 + ms_e**2 / 33)
    assert out["mean"] == pytest.approx(m.mean())
    assert out["se"] == pytest.approx(math.sqrt(var))
    assert out["df"] == pytest.approx(df)
    half = student_t_isf(0.025, df) * math.sqrt(var)
    assert out["ci"] == pytest.approx([m.mean() - half, m.mean() + half])


def test_crossed_mean_truncates_negative_seed_component():
    # Seed rows identical in mean -> MS_seed = 0 < MS_residual; variance falls back to MS_w/(SW).
    m = np.array([[0.0, 4.0, 1.0, 7.0], [1.0, 3.0, 2.0, 6.0]])
    ms_s, ms_w, ms_e = _two_way_mean_squares(m)
    assert ms_s < ms_e < ms_w
    out = crossed_seed_world_mean(m)
    assert out["sigma2_seed"] == 0.0
    assert out["se"] == pytest.approx(math.sqrt(ms_w / m.size))
    assert out["df"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- curves


def test_curve_auc_and_last_k():
    positions = [0.0, 1.0, 3.0]
    assert curve_auc([0.0, 1.0, 3.0], positions) == pytest.approx([1.5])
    assert curve_auc([0.0, 1.0, 3.0], positions, normalize=False) == pytest.approx([4.5])
    assert curve_auc([[2.0, 2.0, 2.0]]) == pytest.approx([2.0])
    assert last_k_mean([[1.0, 5.0, 3.0, 7.0]], 2) == pytest.approx([5.0])
    with pytest.raises(ValueError):
        curve_auc([1.0, 2.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        last_k_mean([[1.0, 2.0]], 3)


def test_learning_curve_contrast_paired_oracle():
    base = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], [5.0, 4.0, 3.0, 2.0]])
    cand = base + np.array([[1.0], [2.0], [3.0]])
    out = learning_curve_contrast(cand, base, last_k=2)
    t2 = student_t_isf(0.025, 2)
    for key in ("auc", "last_k", "final"):
        summary = out[key]
        assert summary["per_seed_delta"] == pytest.approx([1.0, 2.0, 3.0])
        assert summary["mean_delta"] == pytest.approx(2.0)
        assert summary["sd"] == pytest.approx(1.0)
        assert summary["se"] == pytest.approx(1.0 / math.sqrt(3.0))
        assert summary["df"] == 2.0
        assert summary["ci"] == pytest.approx([2.0 - t2 / math.sqrt(3), 2.0 + t2 / math.sqrt(3)])
    assert out["auc"]["baseline_scores"] == pytest.approx([1.5, 1.0, 3.5])
    assert out["last_k"]["baseline_scores"] == pytest.approx([2.5, 1.0, 2.5])


def test_learning_curve_contrast_welch_and_one_sample():
    cand = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    base = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    out = learning_curve_contrast(cand, base, paired=False, last_k=1)["final"]
    var_c, var_b = 1.0 / 3.0, (1.0 / 3.0) / 4.0
    df = (var_c + var_b) ** 2 / (var_c**2 / 2 + var_b**2 / 3)
    assert out["mean_delta"] == pytest.approx(1.5)
    assert out["se"] == pytest.approx(math.sqrt(var_c + var_b))
    assert out["df"] == pytest.approx(df)
    one = learning_curve_contrast([[1.0, 3.0], [3.0, 5.0]], last_k=1)["auc"]
    assert one["mean_delta"] == pytest.approx(3.0)
    assert one["se"] == pytest.approx(1.0)
    assert one["p_value"] == pytest.approx(2.0 * student_t_sf(3.0, 1))
    with pytest.raises(ValueError):
        learning_curve_contrast(cand, base)  # paired needs equal seed counts


# --------------------------------------------------------------------------- planning


def test_conjunction_vs_pooled_gate_oracle():
    out = gate_pass_probabilities(0.8, 3)
    z_a = N.inv_cdf(0.95)
    d = z_a + N.inv_cdf(0.8)
    assert out["conjunction_pass"] == pytest.approx(0.512)
    assert out["pooled_pass"] == pytest.approx(N.cdf(math.sqrt(3.0) * d - z_a))
    assert out["pooled_pass"] > 0.99
    assert out["conjunction_false_pass"] == pytest.approx(0.05**3)
    single = gate_pass_probabilities([0.37])
    assert single["pooled_pass"] == pytest.approx(0.37) == single["conjunction_pass"]
    mixed = gate_pass_probabilities([0.9, 0.6])
    dd = [z_a + N.inv_cdf(p) for p in (0.9, 0.6)]
    assert mixed["pooled_pass"] == pytest.approx(N.cdf(math.hypot(*dd) - z_a))
    with pytest.raises(ValueError):
        gate_pass_probabilities([0.9, 0.01])  # opposite-sign effects
    with pytest.raises(ValueError):
        gate_pass_probabilities(0.8)


def test_gate_pass_for_effect_closed_form_and_monte_carlo():
    z_a = N.inv_cdf(0.95)
    out = gate_pass_probabilities_for_effect(2.0, 1.0, 3)
    assert out["per_seed_pass"] == pytest.approx(N.cdf(2.0 - z_a))
    assert out["pooled_pass"] == pytest.approx(
        gate_pass_probabilities(out["per_seed_pass"], 3)["pooled_pass"]
    )
    het = gate_pass_probabilities_for_effect(2.0, 1.0, 5, tau2=3.0, n_sims=20_000, seed=11)
    assert het["per_seed_pass"] == pytest.approx(N.cdf((2.0 - z_a) / 2.0))
    assert het["pooled_pass"] == pytest.approx(N.cdf(2.0 / (2.0 / math.sqrt(5)) - z_a))
    mc = het["monte_carlo"]
    mc_se = math.sqrt(het["conjunction_pass"] * (1 - het["conjunction_pass"]) / 20_000)
    assert abs(mc["conjunction_pass"] - het["conjunction_pass"]) < 4 * mc_se
    assert mc["pooled_hksj_floor_pass"] < mc["pooled_dl_z_pass"]
    null = gate_pass_probabilities_for_effect(0.0, 1.0, 5, tau2=1.0, n_sims=20_000, seed=5)
    assert null["monte_carlo"]["pooled_hksj_floor_pass"] < 0.05 + 0.006


def test_seeds_for_pooled_power():
    out = seeds_for_pooled_power(1.0, 1.0, alpha=0.025, power=0.8)
    assert out["z_seeds"] == math.ceil((Z975 + N.inv_cdf(0.8)) ** 2)  # 7.85 -> 8
    assert out["t_seeds"] >= out["z_seeds"]
    more = seeds_for_pooled_power(1.0, 1.0, tau2=1.0, alpha=0.025, power=0.8)
    assert more["z_seeds"] == math.ceil(2.0 * (Z975 + N.inv_cdf(0.8)) ** 2)
    with pytest.raises(ValueError):
        seeds_for_pooled_power(0.0, 1.0)


# --------------------------------------------------------------------------- futility


def test_obf_spending_function():
    assert obf_spending(1.0, 0.025) == pytest.approx(0.025)
    assert obf_spending(0.0, 0.025) == 0.0
    expected = 2.0 * (1.0 - N.cdf(N.inv_cdf(1.0 - 0.0125) / math.sqrt(0.5)))
    assert obf_spending(0.5, 0.025) == pytest.approx(expected)


def test_single_look_boundary_is_fixed_sample_quantile():
    plan = futility_plan([20], alpha=0.05)
    assert plan.boundaries[0] == pytest.approx(N.inv_cdf(0.95), abs=1e-9)
    assert plan.alpha_spent[-1] == pytest.approx(0.05)


@pytest.mark.parametrize(
    "looks, expected",
    [
        # Lan-DeMets O'Brien-Fleming spending, one-sided alpha 0.025, equal spacing
        # (published gsDesign sfLDOF values).
        ([50, 100], [2.9626, 1.9686]),
        ([20, 40, 60, 80, 100], [4.8769, 3.3569, 2.6803, 2.2898, 2.0310]),
    ],
)
def test_obf_boundaries_match_published_values(looks, expected):
    plan = futility_plan(looks, alpha=0.025)
    assert plan.boundaries == pytest.approx(expected, abs=2e-4)
    assert plan.alpha_spent[-1] == pytest.approx(0.025, abs=1e-6)
    assert isinstance(plan, FutilityPlan)


def test_boundaries_hold_alpha_by_brownian_simulation():
    plan = futility_plan([8, 16, 24, 32], alpha=0.05)
    rng = np.random.default_rng(1)
    sims = 200_000
    increments = rng.normal(0.0, 1.0, (sims, 4)) * math.sqrt(0.25)
    brownian = np.cumsum(increments, axis=1)
    z = brownian / np.sqrt(np.array(plan.fractions))[None, :]
    crossed = np.any(z <= -np.array(plan.boundaries)[None, :], axis=1)
    assert crossed.mean() == pytest.approx(0.05, abs=3 * math.sqrt(0.05 * 0.95 / sims))


def test_stop_probability_oracles():
    plan = futility_plan([8, 16, 24, 32], alpha=0.05)
    null = futility_stop_probability(plan, 0.0, 10.0)
    assert null["stop_probability"] == pytest.approx(0.05, abs=1e-6)
    assert null["cumulative_stop"] == pytest.approx(list(plan.alpha_spent), abs=1e-6)
    one = futility_plan([25], alpha=0.05)
    harm = futility_stop_probability(one, -6.0, 10.0)  # drift = 6 / (10 / 5) = 3
    assert harm["stop_probability"] == pytest.approx(N.cdf(3.0 - N.inv_cdf(0.95)), abs=1e-9)
    worse = futility_stop_probability(plan, -8.0, 10.0)
    assert (
        worse["stop_probability"] > futility_stop_probability(plan, -4.0, 10.0)["stop_probability"]
    )
    assert worse["expected_worlds"] < 32


def test_evaluate_look_t_to_z_and_decision():
    plan = futility_plan([4, 8], alpha=0.05)
    deltas = [-3.0, -1.0, -2.0, -2.0]
    out = evaluate_futility_look(plan, 0, deltas)
    sd = math.sqrt(2.0 / 3.0)
    t_stat = -2.0 / (sd / 2.0)
    assert out["t_statistic"] == pytest.approx(t_stat)
    assert out["z_statistic"] == pytest.approx(N.inv_cdf(student_t_sf(-t_stat, 3)))
    assert out["stop_clear_loser"] is (out["z_statistic"] <= -plan.boundaries[0])
    with pytest.raises(ValueError):
        evaluate_futility_look(plan, 0, deltas[:3])
    flat = evaluate_futility_look(plan, 0, [-1.0] * 4)
    assert flat["z_statistic"] == -math.inf and flat["stop_clear_loser"]


def test_run_sequence_stops_at_first_crossing_and_respects_available_data():
    plan = futility_plan([4, 8, 12], alpha=0.05, null_delta=0.0)
    rng = np.random.default_rng(2)
    loser = list(-10.0 + rng.normal(0.0, 1.0, 12))
    out = run_futility_sequence(plan, loser)
    assert out["stopped"] and out["stop_look"] == 0 and len(out["looks"]) == 1
    tie = list(rng.normal(0.0, 1.0, 9))
    partial = run_futility_sequence(plan, tie)
    assert len(partial["looks"]) == 2 and not partial["completed_all_looks"]
    null_shift = futility_plan([4, 8], alpha=0.05, null_delta=-50.0)
    assert not run_futility_sequence(null_shift, loser[:8])["stopped"]


def test_plan_validation():
    with pytest.raises(ValueError):
        futility_plan([8, 8])
    with pytest.raises(ValueError):
        futility_plan([100, 100 + 1, 200])  # fraction step < 0.01
    with pytest.raises(ValueError):
        futility_plan([4, 8], alpha=0.6)
    with pytest.raises(ValueError):
        futility_plan([4, 8], max_size=6)


def test_stratified_mean_of_means_welch_oracle():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [10.0, 14.0, 12.0]
    result = stratified_mean_of_means({"a": a, "b": b}, confidence=0.90)
    va, vb = np.var(a, ddof=1) / 4, np.var(b, ddof=1) / 3
    se = math.sqrt(va + vb) / 2
    df = (va + vb) ** 2 / (va**2 / 3 + vb**2 / 2)
    half = student_t_isf(0.05, df) * se
    assert result["mean_of_means"] == pytest.approx((2.5 + 12.0) / 2)
    assert result["se"] == pytest.approx(se) and result["df"] == pytest.approx(df)
    assert result["ci"] == pytest.approx([7.25 - half, 7.25 + half])
    assert result["upper_below_zero"] is False and result["strata_upper_below_zero"] == []
    loser = stratified_mean_of_means({"a": [-5.0, -6.0, -7.0], "b": [1.0, 2.0, 3.0]})
    assert loser["strata_upper_below_zero"] == ["a"]
    with pytest.raises(ValueError):
        stratified_mean_of_means({"a": [1.0]})
    with pytest.raises(ValueError):
        stratified_mean_of_means({})
