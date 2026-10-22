"""Verify the two existing release anchors without inheriting them into Claim B.

These immutable releases already underwent their expensive scientific audits. Runtime
validation opens their exact anchored artifacts and checks cross-links, rather than
re-running thousands of historical worlds at every session construction. New identities
need their own packet validator; adding a name or copying these pins grants nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .loader import EvidenceError, FileRef, require

# Authority is the existing owner-approved releases, not hashes supplied by a caller.
RELEASES = {
    "frp3-s12": {
        "checkpoint": "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
        "veto": "v8",
        "method": "free-space-veto/v8-space-and-head(lambda=8.0)",
        "files": {
            "receipt": "5107f3fc540af41f2d81f907d2c9e40302ea4140141833144460f562e5acc5b3",
            "intent": "5deef5a9d415ec135b4c16d59757b486e97b9b5af2068564cfe003a616520485",
            "audit_report": "fa015c2f4b8c38c61d23d387c129fc732c857e6156e638fa1431b412e38a2911",
        },
    },
    "b8-s3": {
        "checkpoint": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
        "veto": "none",
        "method": "none",
        "files": {
            "decision": "8fd97ecc564fdc31a1d77ed50b8112dcffac42d736db20b2b410b67c6ef139fb",
            "audit_report": "c631930c0af441e72e304a08cea401ed1a494c02c3e73071a8a459fb33dde07e",
            "independent_review": (
                "135b1b8b5736619dbf1d57460d606c3a60a4ff4f59019c4ee51392dc5e2aecbe"
            ),
        },
    },
}


def strict_evidence_problems(
    name: str, pin: Mapping[str, Any], *, candidate_sha256: str | None = None
) -> list[str]:
    """Return defects in opened legacy release files; they never qualify assisted C."""
    try:
        require(candidate_sha256 is None, "legacy strict pins do not qualify assisted Claim B C")
        require(name in RELEASES, "no immutable release evidence anchor for this name")
        anchor = RELEASES[name]
        require(isinstance(pin, Mapping), "strict pin must be an object")
        require(pin.get("checkpoint_sha256") == anchor["checkpoint"], "release checkpoint mismatch")
        require(
            pin.get("veto_variant") == anchor["veto"]
            and pin.get("veto_method") == anchor["method"],
            "release veto mismatch",
        )
        receipt = pin.get("strict_receipt")
        require(isinstance(receipt, Mapping), "strict receipt missing")
        require(receipt.get("verdict") == "STRICT_PASS", "strict receipt verdict mismatch")
        require(isinstance(receipt.get("root"), str), "strict evidence root missing")
        root = Path(receipt["root"])
        documents = {}
        refs = {}
        for key, expected in anchor["files"].items():
            ref = FileRef.parse(receipt.get(key))
            require(ref.sha256 == expected, f"strict_receipt.{key} is not the released evidence")
            documents[key] = ref.read(root, completed=False)
            refs[key] = ref
        if name == "frp3-s12":
            body, intent, audit = (documents[key] for key in ("receipt", "intent", "audit_report"))
            candidate = intent["arms"]["candidate"]
            require(
                candidate.get("checkpoint_sha256") == anchor["checkpoint"]
                and candidate.get("method") == anchor["method"],
                "strict intent candidate/veto mismatch",
            )
            require(
                body.get("intent_sha256") == refs["intent"].sha256
                and body.get("audit_report_sha256") == refs["audit_report"].sha256,
                "strict receipt cross-link mismatch",
            )
            require(
                audit.get("status") == "PASS"
                and audit.get("recomputed", {}).get("expected_outcome") == "STRICT_PASS",
                "strict audit clauses failed",
            )
        else:
            decision, audit, review = (
                documents[key] for key in ("decision", "audit_report", "independent_review")
            )
            body = decision.get("decision", {})
            require(
                decision.get("schema") == "t7-phase-r-decision/v1"
                and body.get("status") == "STRICT_PASS"
                and body.get("valid") is True
                and body.get("candidate_selection", {}).get("seed") == 3,
                "T7 decision clauses failed",
            )
            require(
                audit.get("schema") == "t7-phase-r-audit/v1"
                and audit.get("coverage") == {"expected": 5805, "found": 5805, "exact": True}
                and all(
                    audit.get(key) is True
                    for key in ("plan_valid", "records_valid", "decision_valid")
                ),
                "T7 audit clauses failed",
            )
            require(
                review.get("schema") == "t7-independent-result-review/v1"
                and review.get("verdict") == "PASS"
                and review.get("decision") == "STRICT_PASS"
                and review.get("candidate_seed") == 3
                and review.get("errors") == [],
                "T7 independent review clauses failed",
            )
            require(
                review.get("evidence", {}).get("decision.json") == refs["decision"].sha256
                and review.get("evidence", {}).get("audit.json") == refs["audit_report"].sha256,
                "T7 independent review cross-link mismatch",
            )
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return [str(exc)]
    return []
