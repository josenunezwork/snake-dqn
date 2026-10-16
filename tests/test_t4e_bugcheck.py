"""Denominator checks for the saved-bank diagnostic, independent of real artifacts."""

import numpy as np
import pytest

from research.t4e_return_20261016.bugcheck import distribution, summarize


def test_target_gap_distinguishes_branch_population_from_equal_state_pairs() -> None:
    """Different legal-action counts must not masquerade as a paired advantage."""
    rows = {
        "family": np.zeros(5, np.uint8),
        "state_id": np.array([0, 0, 1, 1, 1]),
        "action": np.array([0, 3, 0, 1, 3]),
        "died": np.array([True, False, False, False, False]),
        "y": np.array([0.0, 10.0, 100.0, 100.0, 90.0]),
        "G": np.array([0.0, 10.0, 100.0, 100.0, 90.0]),
        "bootstrap": np.zeros(5),
        "weight": np.ones(5),
        "b8_qa": np.zeros(5),
        "ctrl_2m_qa": np.zeros(5),
        "ctrl_final_qa": np.zeros(5),
    }
    got = summarize(rows)["decisive"]
    assert got["mean_y_gap_boost_minus_nonboost"] == pytest.approx(50 - 200 / 3)
    assert got["paired_state_gap"]["n"] == 2
    assert got["paired_state_gap"]["mean"] == pytest.approx(0)
    assert got["paired_state_positive_fraction"] == pytest.approx(0.5)
    assert got["nonboost"]["survival_rate"] == pytest.approx(2 / 3)
    assert got["boost"]["survival_rate"] == 1
    assert got["boost"]["weighted_huber_derivative_mean"] == pytest.approx(-0.16)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_distribution_rejects_missing_or_nonfinite_evidence(values: list) -> None:
    with pytest.raises(ValueError, match="nonempty finite"):
        distribution(np.asarray(values))
