"""Synthetic accounting checks for paired pooled T13 analysis."""

import numpy as np
import pytest

from research.t13_product_roster_20261017.analyze import metric_samples, validate_episode


def _row(decisions, deaths, histogram):
    return dict(
        decisions=decisions,
        deaths=deaths,
        length_histogram=histogram,
        mass_sum=100,
        boost_decisions=1,
        food_eaten=2,
        kills_made=0,
        kills_taken=0,
        death_causes={"wall": deaths},
    )


def test_pooled_resampling_recomputes_rates_and_quantiles():
    rows = [_row(2, 1, {"2": 2}), _row(8, 0, {"10": 8})]
    weights = np.array([[1, 1], [2, 0], [0, 2]], dtype=float)
    result = metric_samples(rows, weights, horizon=100, snakes=1)
    assert result["deaths_per_1000"].tolist() == [100, 500, 0]
    assert result["length_median"].tolist() == [10, 2, 10]
    assert result["length_p95"].tolist() == [10, 2, 10]
    assert result["length_max"].tolist() == [10, 2, 10]
    assert result["mass_h5000"].tolist() == [1, 1, 1]
    assert result["death_cause_wall"].tolist() == [1, 2, 0]


def test_identical_paired_draws_have_zero_delta():
    rows = [_row(2, 1, {"2": 2}), _row(8, 0, {"10": 8})]
    weights = np.array([[1, 1], [2, 0], [0, 2]], dtype=float)
    left = metric_samples(rows, weights, 5000, 12)
    right = metric_samples(rows, weights, 5000, 12)
    for key in left:
        assert np.array_equal(left[key] - right[key], np.zeros(3))


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError, match="incomplete horizon"):
        validate_episode({"roster": "mixed", "seed": 1, "horizon": 100}, "mixed", 1)
