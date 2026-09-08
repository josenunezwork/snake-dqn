"""Unit tests for the shared paired-evaluation statistics (src/scripts/eval_stats.py).

The paired-CI values are checked against hand-computed numbers so a silent
change to the t-tables or the CI formula fails loudly.
"""

import json
import math

import pytest

from src.scripts.eval_stats import (
    ci95_halfwidth,
    holm,
    holm_three_mix_superiority,
    mass_integral,
    mean,
    paired_delta_pilot_size,
    paired_delta_test,
    paired_stats,
    recommended_seed_count,
    sample_std,
    scripted_noninferiority,
    strict_promotion_decision,
    student_t_isf,
    student_t_sf,
    t_critical_80,
    t_critical_975,
)


class TestTCritical:
    def test_exact_tabulated_values(self):
        assert t_critical_975(4) == pytest.approx(2.7764, abs=1e-4)
        assert t_critical_975(9) == pytest.approx(2.2622, abs=1e-4)
        assert t_critical_80(7) == pytest.approx(0.8960, abs=1e-4)

    def test_between_table_points_is_conservative(self):
        # df=35 falls between 30 and 40: use df=30's LARGER value.
        assert t_critical_975(35) == pytest.approx(2.0423, abs=1e-4)

    def test_normal_limit_beyond_table(self):
        assert t_critical_975(1000) == pytest.approx(1.96, abs=1e-4)
        assert t_critical_80(1000) == pytest.approx(0.8416, abs=1e-4)

    def test_df_below_one_clamped(self):
        assert t_critical_975(0) == t_critical_975(1)


class TestBasicStats:
    def test_mean_and_std(self):
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)
        assert mean([]) == 0.0
        assert sample_std([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(math.sqrt(2.5))
        assert sample_std([7.0]) == 0.0

    def test_ci95_hand_computed(self):
        # sd = sqrt(2.5), se = sd/sqrt(5) = 0.7071068, t(4) = 2.7764
        # -> half-width = 1.96321
        assert ci95_halfwidth([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.96321, abs=1e-4)

    def test_ci95_degenerate(self):
        assert ci95_halfwidth([]) == 0.0
        assert ci95_halfwidth([3.0]) == 0.0


class TestMassIntegral:
    def test_dead_frames_count_in_denominator(self):
        # Alive 3 frames at mass 2, 3, 4, then dead for 7 frames of a
        # 10-frame horizon: integral = 9/10, NOT the alive-mean 3.0.
        alive_mass_sum = 2 + 3 + 4
        assert mass_integral(alive_mass_sum, 10) == pytest.approx(0.9)
        alive_mean = alive_mass_sum / 3
        assert mass_integral(alive_mass_sum, 10) < alive_mean

    def test_full_survival_equals_alive_mean(self):
        assert mass_integral(2 + 3 + 4, 3) == pytest.approx(3.0)

    def test_dying_rich_loses_to_surviving(self):
        # Old bug: mean over alive frames ranked "die at mass 50 on frame 10"
        # above "hold mass 20 for all 100 frames". The integral must not.
        die_rich = mass_integral(50 * 10, 100)
        survive = mass_integral(20 * 100, 100)
        assert survive > die_rich

    def test_zero_frames_rejected(self):
        with pytest.raises(ValueError):
            mass_integral(10.0, 0)


class TestPairedStats:
    def test_hand_computed_delta_ci(self):
        cand = [10.0, 12.0, 14.0, 16.0, 18.0]
        base = [9.0, 10.0, 11.0, 12.0, 13.0]
        stats = paired_stats(cand, base)
        assert stats["deltas"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert stats["mean_delta"] == pytest.approx(3.0)
        assert stats["ci95"] == pytest.approx(1.96321, abs=1e-4)
        assert stats["ci_low"] == pytest.approx(3.0 - 1.96321, abs=1e-4)
        assert stats["wins"] == 5
        assert stats["losses"] == 0
        assert stats["n"] == 5
        assert stats["significant"] is True
        assert stats["regression"] is False

    def test_regression_detected(self):
        cand = [9.0, 10.0, 11.0, 12.0, 13.0]
        base = [10.0, 12.0, 14.0, 16.0, 18.0]
        stats = paired_stats(cand, base)
        assert stats["mean_delta"] == pytest.approx(-3.0)
        assert stats["significant"] is False
        assert stats["regression"] is True

    def test_pairing_beats_unpaired_on_common_noise(self):
        # Common per-seed offsets cancel in the deltas: the paired CI is tight
        # even though both raw series are noisy.
        seed_noise = [0.0, 40.0, -30.0, 10.0, 25.0]
        base = [100.0 + z for z in seed_noise]
        cand = [101.0 + z for z in seed_noise]
        stats = paired_stats(cand, base)
        assert stats["mean_delta"] == pytest.approx(1.0)
        assert stats["ci95"] == pytest.approx(0.0, abs=1e-9)
        assert stats["significant"] is True

    def test_single_seed_never_significant(self):
        stats = paired_stats([5.0], [1.0])
        assert stats["ci95"] == 0.0
        assert stats["significant"] is False
        assert stats["regression"] is False

    def test_mismatched_or_empty_rejected(self):
        with pytest.raises(ValueError):
            paired_stats([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            paired_stats([], [])


class TestRecommendedSeedCount:
    def test_hand_computed_iteration(self):
        # ratio sd/mde = 1: normal start ceil(2.8016^2) = 8, then
        # df=7 -> ceil(3.2606^2) = 11, df=10 -> 10, df=9 -> 10 (converged).
        assert recommended_seed_count(1.0, 1.0) == 10

    def test_larger_variance_needs_more_seeds(self):
        # ratio 2: normal start 32, df>=30 -> ceil(2.8961^2 * 4) = 34.
        assert recommended_seed_count(2.0, 1.0) == 34
        assert recommended_seed_count(2.0, 1.0) > recommended_seed_count(1.0, 1.0)

    def test_zero_variance_returns_minimum(self):
        assert recommended_seed_count(0.0, 1.0) == 2
        assert recommended_seed_count(0.0, 1.0, minimum=5) == 5

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            recommended_seed_count(1.0, 0.0)
        with pytest.raises(ValueError):
            recommended_seed_count(1.0, 1.0, alpha=0.01)
        with pytest.raises(ValueError):
            recommended_seed_count(1.0, 1.0, power=0.9)


class TestStrictNumerics:
    def test_student_t_df_one_matches_independent_cauchy_formula(self):
        # The Cauchy survival function is analytic, independent of the
        # incomplete-beta implementation used by the production function.
        for value in (-10.0, -2.5, -1.0, 0.0, 1.0, 2.5, 10.0):
            expected = 0.5 - math.atan(value) / math.pi
            assert student_t_sf(value, 1) == pytest.approx(expected, rel=5e-12)

    def test_student_t_inverse_round_trips_at_one_sided_gate_probability(self):
        critical = student_t_isf(0.05 / 3.0, 39)
        assert student_t_sf(critical, 39) <= 0.05 / 3.0
        assert student_t_sf(math.nextafter(critical, -math.inf), 39) > 0.05 / 3.0

    def test_holm_matches_exact_boundary_and_step_down_behavior(self):
        rejected, adjusted = holm([0.05 / 3.0, 0.025, 0.05])
        assert rejected == [True, True, True]
        assert adjusted == pytest.approx([0.05, 0.05, 0.05])
        # A second-rank failure stops the third rejection even though 0.03
        # would meet that third-rank threshold of 0.05 on its own.
        rejected, _ = holm([0.001, 0.03, 0.03])
        assert rejected == [True, False, False]


class TestStrictPairedDeltas:
    def test_zero_standard_error_is_json_safe_and_positive_limit_succeeds(self):
        result = paired_delta_test([0.25, 0.25, 0.25])
        assert result["superior"] is True
        assert result["t_statistic"] is None
        assert result["t_statistic_limit"] == "positive_infinity"
        assert result["p_value"] == 0.0
        json.dumps(result, allow_nan=False)

    def test_zero_standard_error_zero_and_negative_limits_are_explicit(self):
        zero = paired_delta_test([0.0, 0.0])
        negative = paired_delta_test([-0.25, -0.25])
        assert (zero["t_statistic"], zero["t_statistic_limit"], zero["p_value"]) == (
            None,
            "zero",
            0.5,
        )
        assert (
            negative["t_statistic"],
            negative["t_statistic_limit"],
            negative["p_value"],
        ) == (None, "negative_infinity", 1.0)

    def test_n_less_than_two_is_explicit_invalid_not_a_significant_result(self):
        result = paired_delta_test([1.0])
        assert result["valid"] is False
        assert result["superior"] is False
        assert result["p_value"] is None
        assert result["descriptive_ci95"] == {
            "half_width": None,
            "low": None,
            "high": None,
        }

    def test_nonfinite_and_boolean_deltas_are_rejected(self):
        with pytest.raises(ValueError):
            paired_delta_test([1.0, float("nan")])
        with pytest.raises(TypeError):
            paired_delta_test([1.0, True])

    def test_holm_requires_three_raw_mix_tests_and_two_rejections(self):
        result = holm_three_mix_superiority(
            {
                "frozen": [0.1, 0.1, 0.1],
                "scripted": [0.2, 0.2, 0.2],
                "mixed": [-0.1, -0.1, -0.1],
            }
        )
        assert result["valid"] is True
        assert result["successful_mixes"] == ["frozen", "scripted"]
        assert result["passes"] is True

        invalid = holm_three_mix_superiority(
            {"frozen": [1.0], "scripted": [1.0, 1.0], "mixed": [1.0, 1.0]}
        )
        assert invalid["valid"] is False
        assert invalid["passes"] is False
        assert invalid["invalid_mixes"] == ["frozen"]

    def test_scripted_noninferiority_boundary_is_strict(self):
        equality = scripted_noninferiority([-1.0, -1.0, -1.0], absolute_delta_ni=1.0)
        assert equality["lower_bound"] == -1.0
        assert equality["passes"] is False
        above = scripted_noninferiority([-0.999, -0.999, -0.999], absolute_delta_ni=1.0)
        assert above["passes"] is True

    def test_strict_decision_joins_holm_and_separate_scripted_ni(self):
        result = strict_promotion_decision(
            {
                "frozen": [0.2, 0.2, 0.2],
                "scripted": [-0.01, -0.01, -0.01],
                "mixed": [0.2, 0.2, 0.2],
            },
            scripted_mix="scripted",
            absolute_delta_ni=0.02,
        )
        assert result["superiority"]["passes"] is True
        assert result["scripted_noninferiority"]["passes"] is True
        assert result["passes"] is True
        assert result["descriptive_combined_ci95"]["n"] == 3

    def test_strict_decision_fails_closed_for_unequal_lengths_and_alt_alpha(self):
        unequal = strict_promotion_decision(
            {
                "frozen": [0.2, 0.2],
                "scripted": [0.2, 0.2, 0.2],
                "mixed": [0.2, 0.2, 0.2],
            },
            scripted_mix="scripted",
            absolute_delta_ni=0.02,
        )
        assert unequal["valid"] is False
        assert unequal["passes"] is False
        with pytest.raises(ValueError, match="fixes family alpha"):
            strict_promotion_decision(
                {
                    "frozen": [0.2, 0.2],
                    "scripted": [0.2, 0.2],
                    "mixed": [0.2, 0.2],
                },
                scripted_mix="scripted",
                absolute_delta_ni=0.02,
                alpha=0.049,
            )


class TestStrictPilotSizing:
    def test_pilot_uses_paired_delta_variance_and_hard_floor(self):
        plan = paired_delta_pilot_size(
            {
                "frozen": [1.0, 1.0, 1.0],
                "scripted": [0.0, 4.0, -4.0],
                "mixed": [1.0, 1.0, 1.0],
            },
            {"frozen": 1.0, "scripted": 1.0, "mixed": 1.0},
        )
        assert plan["minimum_final_worlds"] == 40
        assert plan["required_final_worlds"] >= 40
        assert plan["required_final_worlds"] == plan["per_mix"]["scripted"]["recommended_n"]
        assert plan["joint_power_claim"] is None
        assert plan["planning_alpha_per_mix"] == pytest.approx(0.05 / 3.0)

    def test_pilot_rejects_baseline_style_missing_or_undersized_raw_pairs(self):
        with pytest.raises(ValueError):
            paired_delta_pilot_size(
                {"frozen": [1.0], "scripted": [1.0, 1.0], "mixed": [1.0, 1.0]},
                {"frozen": 1.0, "scripted": 1.0, "mixed": 1.0},
            )
        with pytest.raises(ValueError):
            paired_delta_pilot_size(
                {"frozen": [1.0, 1.0], "scripted": [1.0, 1.0], "mixed": [1.0, 1.0]},
                {"frozen": 1.0, "scripted": 1.0},
            )

    def test_pilot_terminates_at_an_adjacent_integer_fixed_point_cycle(self):
        # Select a paired-delta SD that maps 40 -> 41 and 41 -> 40 under the
        # raw fixed-point formula.  The monotone search must choose 41 rather
        # than oscillating until an arbitrary iteration cap.
        alpha = 0.05 / 3.0
        critical_40 = student_t_isf(alpha, 39) + student_t_isf(0.2, 39)
        critical_41 = student_t_isf(alpha, 40) + student_t_isf(0.2, 40)
        lower = math.sqrt(40.0) / critical_40
        upper = min(math.sqrt(41.0) / critical_40, math.sqrt(40.0) / critical_41)
        ratio = (lower + upper) / 2.0
        two_point_deltas = [0.0, ratio * math.sqrt(2.0)]
        plan = paired_delta_pilot_size(
            {
                "frozen": two_point_deltas,
                "scripted": [0.0, 0.0],
                "mixed": [0.0, 0.0],
            },
            {"frozen": 1.0, "scripted": 1.0, "mixed": 1.0},
        )
        assert plan["per_mix"]["frozen"]["recommended_n"] >= 40
        assert math.isfinite(plan["required_final_worlds"])
