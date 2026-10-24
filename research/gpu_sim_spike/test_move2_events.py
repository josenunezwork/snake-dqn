"""The native event inventory must execute real fixtures before it can claim PASS."""

from research.gpu_sim_spike import move2_events


def test_cpu_event_portability_never_claims_native_pass() -> None:
    report = move2_events.run("cpu")
    assert report.get("portability_identity") == "PASS", report
    assert report["status"] == "NOT_RUN"
    assert {row["case"] for row in report["cases"]} == set(move2_events.CASES)
    assert all(row["identity"]["match"] for row in report["cases"])
