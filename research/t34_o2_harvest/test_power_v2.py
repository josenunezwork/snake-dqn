"""Focused, synthetic checks for the O2 V2 pre-data sizing calculation."""

from research.t34_o2_harvest import power_v2


def test_required_allocation_is_balanced_minimal_and_over_1000() -> None:
    n = power_v2.required_n()
    assert n == 1699
    assert 3 * n > 1000
    assert power_v2._joint_lower_bound(n) >= 0.80
    assert power_v2._joint_lower_bound(n - 1) < 0.80


def test_run_records_dependence_robust_power_and_full_go_boundary_disclosure() -> None:
    result = power_v2.run(reps=4_000, seed=7, cap_worlds=1260)
    allocation = result["required_allocation"]
    assert allocation["worlds_total"] == 5097
    assert allocation["dependence_robust_joint_lower_bound"] >= 0.80
    assert result["familywise_null"]["claim"].startswith("Holm at alpha .05")
    assert "0.5 pass ceiling" in result["point_bar_disclosure"]["at_true40"]
    assert result["budget_capped_allocation"]["worlds_per_mix"] == 420
    assert result["budget_capped_allocation"]["status"] == "UNDERPOWERED_CAP"


def test_holm_proxy_is_bounded_at_composite_nulls() -> None:
    result = power_v2.monte_carlo(1699, reps=20_000, seed=8, sds=power_v2.measured_sds())
    scenarios = result["scenarios"]
    assert scenarios["global_null"]["family_any_false_rejection"] <= 0.055
    assert scenarios["b8_null_increment10"]["false_rejection_of_b8_null"] <= 0.055
    assert scenarios["true40_increment_null"]["false_rejection_of_increment_null"] <= 0.055


def test_welch_proxy_df_uses_squared_stratum_variance_estimates() -> None:
    """The finite-n MC denominator is sum(vhat_mix**2 / (n-1)) / 81."""
    sds = power_v2.measured_sds()
    n = 420
    variance = [sds[mix] ** 2 / n for mix in power_v2.MIXES]
    expected = (sum(variance) / 9) ** 2 / (sum(v * v / (n - 1) for v in variance) / 81)
    _, actual = power_v2._statistics(n, sds)
    assert actual == expected
