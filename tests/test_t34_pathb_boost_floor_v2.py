"""Owner-approved relative boost floor (pre-data, 2026-10-20).

0.5 x paired bare delivered boost share <= candidate share <= 0.10, pooled over
the same 144 terminal worlds; a zero bare share gives a zero floor.
Spec: docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md item 2.
"""

from __future__ import annotations

import pytest

from research.t34_pathb import analyze_v2 as analysis
from tests.test_t34_retention_fixes_analyzer import _records, _timing


def _set_boosts(records: list[dict], arm: str, boosts: int, decisions: int = 10000) -> None:
    for row in records:
        if row["kind"] == "terminal":
            row["arms"][arm]["decisions"] = decisions
            row["arms"][arm]["boost_decisions"] = boosts


def _skill(bare: int, paced: int) -> dict:
    records = _records()
    _set_boosts(records, "bare", bare)
    _set_boosts(records, "paced", paced)
    return analysis.skill_summary(records, "paced")


def test_equal_to_bare_share_passes() -> None:
    report = _skill(bare=12, paced=12)  # 0.12% vs 0.12%
    assert report["bare_boost_share"] == pytest.approx(0.0012)
    assert report["boost_floor"] == pytest.approx(0.0006)
    assert report["clauses"]["boost_share"] is True


def test_below_half_of_bare_share_fails() -> None:
    report = _skill(bare=12, paced=5)  # 0.05% < 0.5 x 0.12%
    assert report["boost_share"] == pytest.approx(0.0005)
    assert report["clauses"]["boost_share"] is False


def test_exactly_half_of_bare_share_passes() -> None:
    assert _skill(bare=12, paced=6)["clauses"]["boost_share"] is True


def test_ceiling_is_unchanged() -> None:
    report = _skill(bare=12, paced=1100)  # 11%
    assert report["boost_ceiling"] == 0.10
    assert report["clauses"]["boost_share"] is False
    assert _skill(bare=12, paced=1000)["clauses"]["boost_share"] is True  # 10% inclusive


def test_zero_bare_share_gives_zero_floor() -> None:
    report = _skill(bare=0, paced=0)
    assert report["boost_floor"] == 0.0
    assert report["clauses"]["boost_share"] is True


def test_old_absolute_floor_no_longer_applies() -> None:
    # Realistic bare B8 s3 share (0.12%) would have failed the old 0.5% floor.
    assert _skill(bare=12, paced=12)["boost_share"] < 0.005


def test_decide_uses_relative_floor_end_to_end() -> None:
    records = _records()
    counts = [1665, 1665, 1664, 3, 2, 1]
    for row in records:
        for arm in row["arms"].values():
            arm.update(
                boost_decisions=6,
                boost_share=6 / 5000,
                action_counts=list(counts),
                one_action_share=1665 / 5000,
            )
    report = analysis.decide(records, _timing())
    assert report["paced"]["clauses"]["boost_share"] is True
    assert report["paced"]["bare_boost_share"] == pytest.approx(0.0012)
