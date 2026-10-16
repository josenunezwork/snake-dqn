"""Check that common support alone is not mistaken for matched weights."""

import pytest

from research.t4a_posthoc_20261016.report import SETS, standardize


def test_standardization_matches_population_weights_and_cross_share() -> None:
    support = ["lt400:D1", "ge1000:D1"]
    matched = {}
    for name, counts in zip(SETS, ((9, 1), (1, 9), (99, 1))):
        table = {}
        for bucket, count, rate in zip(("lt400", "ge1000"), counts, (0.2, 0.8)):
            table[bucket] = {
                "1": {
                    "decisions": count,
                    "argmax_change_rate": rate,
                    "cross_boost_rate_all_decisions": rate / 2,
                }
            }
        matched[name] = {"support": support, "rates": {"by_length_and_decile": table}}
    result = standardize({"matched_common_support": matched})
    for rates in result["overall"]["sets"].values():
        assert rates["argmax_change_rate"] == pytest.approx(0.5)
        assert rates["cross_boost_share_of_argmax_changes"] == pytest.approx(0.5)
    assert result["by_decile"]["2"]["sets"] is None
    assert result["by_decile"]["1"] == result["overall"]


def test_standardization_rejects_different_support() -> None:
    matched = {name: {"support": ["lt400:D1"]} for name in SETS}
    matched[SETS[-1]]["support"] = ["ge1000:D1"]
    with pytest.raises(ValueError, match="identical common support"):
        standardize({"matched_common_support": matched})
