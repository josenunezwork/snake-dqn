"""Registered placement boundaries and refusal of incomplete/foreign timing banks."""

from __future__ import annotations

import copy

import pytest

from research.bl_labels.f1_analyze import decide, transport_faults

ROOT_IDS = [f"root-{i:03d}" for i in range(64)]


def attempt(platform: str, elapsed: float = 100.0) -> dict:
    nproc = 2 if platform == "cpu" else 12
    return {
        "platform": platform,
        "source_commit": "d" * 40,
        "complete": True,
        "transport_faults": [],
        "configs": [
            {
                "source_commit": "d" * 40,
                "schedule_label": label,
                "workers": workers,
                "cpu": {"nproc": nproc},
                "root_count": 64,
                "elapsed_seconds": elapsed,
                "roots_per_hour": 64 * 3600 / elapsed,
                "records": [{"root_id": key} for key in ROOT_IDS],
                "integrity": {"digest_checks": 64},
            }
            for label, workers in (("one", 1), ("half", max(1, nproc // 2)), ("all", nproc))
        ],
    }


@pytest.mark.parametrize(
    ("cpu_seconds", "verdict"),
    [(400.0, "CPU_ALL_THREE_SLOTS"), (400.0001, "L4_LABELS_CPU_OVERFLOW")],
)
def test_quarter_rate_boundary(cpu_seconds: float, verdict: str) -> None:
    result = decide([attempt("cpu", cpu_seconds), attempt("l4")], ROOT_IDS)
    assert result["verdict"] == verdict


def test_recovered_transport_fault_still_excludes_platform() -> None:
    cpu = attempt("cpu")
    cpu["transport_faults"] = [{"command": "exec", "status": "failed"}]
    result = decide([cpu, attempt("l4")], ROOT_IDS)
    assert result["verdict"] == "L4_ONLY_CPU_EXCLUDED"


def test_missing_schedule_cannot_decide_ratio() -> None:
    cpu = attempt("cpu")
    cpu["configs"].pop()
    assert decide([cpu, attempt("l4")], ROOT_IDS)["verdict"] == "INCOMPLETE"


@pytest.mark.parametrize("defect", ["duplicate", "foreign", "capacity", "rate", "digest"])
def test_refuse_unregistered_or_malformed_schedule(defect: str) -> None:
    cpu = attempt("cpu")
    if defect == "duplicate":
        cpu["configs"].append(copy.deepcopy(cpu["configs"][0]))
    elif defect == "foreign":
        cpu["configs"][0]["records"][0]["root_id"] = "foreign"
    elif defect == "capacity":
        cpu["configs"][0]["workers"] = 12
    elif defect == "rate":
        cpu["configs"][0]["roots_per_hour"] = 100000
    else:
        cpu["configs"][0]["integrity"]["digest_checks"] = 63
    with pytest.raises(ValueError):
        decide([cpu, attempt("l4")], ROOT_IDS)


def test_l4_not_run_requires_explicit_declaration() -> None:
    assert decide([attempt("cpu")], ROOT_IDS)["verdict"] == "INCOMPLETE"
    assert decide([attempt("cpu")], ROOT_IDS, l4_not_run=True)["verdict"] == "CPU_ONLY_L4_NOT_RUN"


def test_transport_census_requires_evidence_and_counts_budget_exhaustion() -> None:
    with pytest.raises(ValueError):
        transport_faults({"cli_calls": []})
    call = {"status": "completed", "elapsed_seconds": 120.0, "timeout_seconds": 120.0}
    assert transport_faults({"cli_calls": [call]}) == [call]
    call["elapsed_seconds"] = 1.0
    assert transport_faults({"cli_calls": [call]}) == []
