"""Negative evidence tests for the B8 serving-pin owner tool."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from research.t12_b8_serving_20261017 import fill_strict_pin as pin

RESULTS = Path(__file__).resolve().parents[1] / "research/redesign_t7_phase_r_20261016/results"


def test_committed_t7_envelopes_retain_the_frozen_rule_and_immutable_pins():
    """The synthetic tree supplements, rather than replaces, the governed pins."""
    from research.redesign_t7_phase_r_20261016 import phase_r_t7, rule_spec_t7

    plan = json.loads((RESULTS / "plan.json").read_text())
    rule = rule_spec_t7.rule()
    assert plan["rule"] == rule
    assert plan["rule_sha256"] == phase_r_t7.canonical_sha(rule)
    names = {
        "decision": "decision.json",
        "audit_report": "audit.json",
        "independent_review": "independent_result_review.json",
        "plan": "plan.json",
        "start": "start.json",
    }
    assert {
        key: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        for key, name in names.items()
    } == pin.IMMUTABLE_SHA256


@pytest.fixture
def evidence_root(tmp_path, monkeypatch):
    """Build a complete, small-payload T7 tree for the real frozen recomputation."""
    from research.redesign_t7_phase_r_20261016 import phase_r_t7, rule_spec_t7
    from tests.test_t7_rule_spec import _banks, _entries

    banks, rows = _banks(), _entries()
    for row in rows:
        if row["hero"] == rule_spec_t7.HERO and row["seed"] == 3:
            row["prefix_h5000"]["mass_integral"] += 20.0
            row["record"]["mass_integral"] += 20.0
    records = tmp_path / "records"
    records.mkdir()
    inventory = []

    def write_record(index, row):
        path = records / f"{index:04d}.json"
        path.write_text(json.dumps(row, sort_keys=True))
        inventory.append(
            {
                "path": str(path.relative_to(tmp_path)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    for index, row in enumerate(rows):
        write_record(index, row)
    by_identity = {(r["hero"], r["seed"], r["mix"], r["world_seed"]): r for r in rows}
    index = len(rows)
    for hero in (rule_spec_t7.HERO, rule_spec_t7.INCUMBENT, rule_spec_t7.CHAMPION):
        for seed, worlds in banks.items():
            for mix in rule_spec_t7.MIXES:
                source = by_identity[(hero, seed, mix, worlds[0])]
                control = deepcopy(source)
                control["control"] = True
                control["record"] = deepcopy(source["prefix_h5000"])
                write_record(index, control)
                index += 1
    assert len(inventory) == 5805
    rule = rule_spec_t7.rule()
    plan = {
        "schema": "t7-phase-r-plan/v1",
        "rule": rule,
        "rule_sha256": phase_r_t7.canonical_sha(rule),
        "banks": {str(seed): worlds for seed, worlds in banks.items()},
        "checkpoints": {"B8-noveto|s3": {"sha256": pin.CHECKPOINT_SHA256}},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan, sort_keys=True))
    plan_sha = hashlib.sha256((tmp_path / "plan.json").read_bytes()).hexdigest()
    (tmp_path / "start.json").write_text(json.dumps({"plan_sha256": plan_sha}, sort_keys=True))
    start_sha = hashlib.sha256((tmp_path / "start.json").read_bytes()).hexdigest()
    decision = {
        "plan_sha256": plan_sha,
        "start_sha256": start_sha,
        "records": inventory,
        "decision": rule_spec_t7.decide(rows, banks, True),
    }
    (tmp_path / "decision.json").write_text(json.dumps(decision, sort_keys=True))
    audit = {
        "schema": "t7-phase-r-audit/v1",
        "coverage": {"expected": 5805, "found": 5805, "exact": True},
        "errors": [],
        "plan_valid": True,
        "records_valid": True,
        "decision_valid": True,
    }
    (tmp_path / "audit.json").write_text(json.dumps(audit, sort_keys=True))
    review = {
        "verdict": "PASS",
        "errors": [],
        "candidate_seed": 3,
        "decision": pin.PASS_WORD,
        "evidence": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ("decision.json", "audit.json", "plan.json", "start.json")
        },
    }
    (tmp_path / "independent_result_review.json").write_text(json.dumps(review, sort_keys=True))
    artifact_root = tmp_path / "immutable-artifact"
    shutil.copytree(records, artifact_root / "records")
    for name in (
        "decision.json",
        "audit.json",
        "independent_result_review.json",
        "plan.json",
        "start.json",
    ):
        shutil.copy2(tmp_path / name, artifact_root / name)
    immutable = {
        "decision": "decision.json",
        "audit_report": "audit.json",
        "independent_review": "independent_result_review.json",
        "plan": "plan.json",
        "start": "start.json",
    }
    monkeypatch.setattr(pin, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(
        pin,
        "IMMUTABLE_SHA256",
        {
            key: hashlib.sha256((artifact_root / name).read_bytes()).hexdigest()
            for key, name in immutable.items()
        },
    )
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
