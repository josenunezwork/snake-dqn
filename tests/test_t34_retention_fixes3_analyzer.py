"""FIX-FIRST #3 analyzer rules for cap-normalized contention accounting."""

from research.t34_pathb import analyze_v2 as analysis
from research.t34_pathb import retention_v2
from tests.test_t34_pathb_analysis_v2 import _host_summary, _records, _timing


def test_contention_allowance_uses_registered_cap_not_short_elapsed_runtime() -> None:
    timing = _timing()
    # Warmup/between-probe contention stays below 72s despite a short run.
    timing["host_summary"] = _host_summary(contended_seconds=10.0, elapsed_seconds=20.0)
    assert analysis.decide(_records(), timing)["verdict"] == "GO"


def test_contention_accounting_requires_all_cap_normalized_fields() -> None:
    timing = _timing()
    summary = _host_summary(contended_seconds=1.0)
    del summary["contention_budget_seconds"]
    timing["host_summary"] = summary

    report = analysis.decide(_records(), timing)

    assert report["verdict"] == "INCONCLUSIVE"
    assert "contention" in report["reason"]


def test_cap_normalized_contention_exceedance_is_inconclusive() -> None:
    timing = _timing()
    timing["host_summary"] = _host_summary(
        contended_seconds=analysis.PHASE_CAP_SECONDS["timing"] * analysis.MAX_CONTENDED_FRACTION
        + 5.0,
        elapsed_seconds=100,
    )

    assert analysis.decide(_records(), timing)["verdict"] == "INCONCLUSIVE"


def test_supplied_timing_ledger_is_checked_even_when_probe_flags_are_quiet() -> None:
    timing = _timing()
    timing["host_summary"] = _host_summary(contended_seconds=2_885)
    assert analysis.decide(_records(), timing)["verdict"] == "INCONCLUSIVE"
    passed, reason = retention_v2._timing_passes(timing)
    assert not passed
    assert reason.startswith("INCONCLUSIVE")
