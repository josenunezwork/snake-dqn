"""Negative evidence tests for the B8 serving-pin owner tool."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from research.t12_b8_serving_20261017 import fill_strict_pin as pin

RESULTS = Path(__file__).resolve().parents[1] / "research/redesign_t7_phase_r_20261016/results"
FILES = ("decision.json", "audit.json", "independent_result_review.json", "plan.json", "start.json")


@pytest.fixture
def evidence_root(tmp_path):
    for name in FILES:
        shutil.copy2(RESULTS / name, tmp_path / name)
    return tmp_path


def problems(root: Path):
    return pin.evidence_problems(
        root,
        {
            "decision": "decision.json",
            "audit_report": "audit.json",
            "independent_review": "independent_result_review.json",
        },
    )


def mutate(path: Path, change) -> None:
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_actual_t7_evidence_recomputes_strict_pass(evidence_root):
    assert problems(evidence_root) == []


def test_review_must_pass_and_have_no_errors(evidence_root):
    mutate(
        evidence_root / "independent_result_review.json",
        lambda doc: doc.update(verdict="FAIL", errors=["independent failure"]),
    )
    failures = problems(evidence_root)
    assert "independent review verdict is not PASS" in failures
    assert "independent review errors are not empty" in failures


def test_decision_selection_cannot_be_relabelled(evidence_root):
    mutate(
        evidence_root / "decision.json",
        lambda doc: doc["decision"]["candidate_selection"].update(seed=2),
    )
    failures = problems(evidence_root)
    assert "T7 rule recomputation differs from decision.json" in failures


def test_audit_must_keep_every_validity_bit(evidence_root):
    mutate(evidence_root / "audit.json", lambda doc: doc.update(decision_valid=False))
    assert "isolated audit is not PASS" in problems(evidence_root)


def test_supplied_forged_audit_path_cannot_bypass_the_immutable_binding(evidence_root):
    forged = evidence_root / "forged-audit.json"
    forged.write_text(
        json.dumps(
            {
                "schema": "wrong",
                "coverage": {"exact": False},
                "plan_valid": True,
                "records_valid": True,
                "decision_valid": True,
                "errors": ["fatal corruption"],
            }
        )
    )
    failures = pin.evidence_problems(
        evidence_root,
        {
            "decision": "decision.json",
            "audit_report": "forged-audit.json",
            "independent_review": "independent_result_review.json",
        },
    )
    assert "audit_report sha256 is not the immutable T7 evidence hash" in failures


def test_filled_pin_refuses_overwrite_without_changing_bytes(evidence_root, tmp_path):
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "schema": "served-checkpoint-strict-pins/v1",
                "checkpoints": {
                    "b8-s3": {
                        "checkpoint_sha256": pin.CHECKPOINT_SHA256,
                        "veto_variant": "none",
                        "veto_method": "none",
                        "strict_receipt": {"already": "filled"},
                    }
                },
            }
        )
    )
    before = pins.read_bytes()
    with pytest.raises(SystemExit, match="already filled"):
        pin.fill(pins, {"verdict": pin.PASS_WORD})
    assert pins.read_bytes() == before
