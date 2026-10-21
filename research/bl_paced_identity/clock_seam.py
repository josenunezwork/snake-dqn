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


def clock_seam_passed(report: Any, source_binding_sha256: str) -> bool:
    """Validate every disposition and independent replay, bound to source bytes."""
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
