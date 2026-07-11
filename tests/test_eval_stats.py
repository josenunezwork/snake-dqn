"""Unit tests for the shared paired-evaluation statistics (src/scripts/eval_stats.py).

The paired-CI values are checked against hand-computed numbers so a silent
change to the t-tables or the CI formula fails loudly.
"""

import math

import pytest

from src.scripts.eval_stats import (
    ci95_halfwidth,
    mass_integral,
    mean,
    paired_stats,
    recommended_seed_count,
    sample_std,
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
