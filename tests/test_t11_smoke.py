"""Acceptance checks for bounded T11 smoke evidence."""

import pytest

from research.t11_local30_20261017.smoke import validate_summary


def summary():
    return {
        "transitions": 200064,
        "halt": None,
        "actor_exitcodes": [0, 0],
        "args": {"device": "mps"},
        "rate": 1000,
        "wall_seconds": 200,
    }


def test_accepts_packet_overshoot():
    validate_summary(summary(), 0)


@pytest.mark.parametrize(
    "change",
    [
        {"transitions": 199999},
        {"transitions": 8000000},
        {"halt": "boost"},
        {"actor_exitcodes": [0, 1]},
        {"args": {"device": "cpu"}},
        {"rate": 0},
    ],
)
def test_rejects_invalid_evidence(change):
    with pytest.raises(ValueError):
        validate_summary({**summary(), **change}, 0)


def test_rejects_nonzero_exit():
    with pytest.raises(ValueError):
        validate_summary(summary(), 1)
