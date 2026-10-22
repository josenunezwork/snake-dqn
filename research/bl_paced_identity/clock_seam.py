"""Fail-closed four-case proof required before L3 identity admission."""

from __future__ import annotations

from typing import Any

SCHEMA = "bl-paced-identity-clock-seam/v1"
EXPECTED = {
    "on-time": (True, "delivered"),
    "publication-late": (False, "cutoff"),
    "consumption-late": (False, "cutoff"),
    "wrong-frame": (False, "stale"),
}


def synthetic_clock_seam_passed(report: Any, source_binding_sha256: str) -> bool:
    """Check only the synthetic fixture summary used by ``audit_preflight``.

    The fixtures exercise journal/replay behavior independently in ``audit.py``.
    They intentionally have no production evidence root, execution provenance, or
    candidate binding, so this helper is never an admission predicate.
    """
    if not isinstance(report, dict) or (
        report.get("schema") != SCHEMA
        or report.get("verdict") != "PASS"
        or report.get("source_binding_sha256") != source_binding_sha256
    ):
        return False
    cases = report.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(EXPECTED):
        return False
    for name, (accepted, reason) in EXPECTED.items():
        case = cases[name]
        if not isinstance(case, dict) or (
            case.get("accepted") is not accepted
            or case.get("reason") != reason
            or case.get("scheduler_verdict") != "PASS"
        ):
            return False
    return True


def clock_seam_passed(
    report: Any, source_binding_sha256: str, *, candidate_sha256: str | None = None
) -> bool:
    """Admit only typed, opened production clock evidence for the expected C.

    A field-only report cannot establish either the source closure or candidate
    identity.  ``None`` is deliberately not a wildcard: callers must calculate
    C from the candidate identity and pass it explicitly.
    """
    from research.bl_evidence.loader import clock_problems

    return not clock_problems(report, source_binding_sha256, candidate_sha256)
