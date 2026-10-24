"""Exact operator and independent-review bindings precede offline admission."""

import json
import subprocess

import pytest

from research.bl_ppo import p1_attempt2_admission as admission
from research.bl_ppo.exports import digest, export_json
from research.bl_ppo.p1_attempt2_pack import ATTEMPT2_RULING, FINALISATION, PIN, TREE


@pytest.fixture
def prerequisites(tmp_path, monkeypatch):
    paths = {
        n: tmp_path / f"{n}.json"
        for n in ("manifest", "codex_v", "claude", "cumulative", "preflight", "output")
    }
    value = {
        "commit": PIN,
        "job_id": "fixture-stage-canary",
        "p1_finalisation_ruling_sha256": FINALISATION,
        "p1_attempt2": {"ruling_sha256": ATTEMPT2_RULING},
        "budget": {
            "unit_cap": 2.0,
            "reserve_units": 0.4,
            "rate_units_per_hour": 1.54,
            "rate_tolerance_units_per_hour": 0,
            "other_assignments": {},
        },
        "timeout_minutes": 30,
        "scientific_admission": False,
        "transport": {"dispatch_ready": False},
        "inputs": [
            {"url": f"https://fixture.invalid/{i}", "sha256": "a" * 64, "size_bytes": 10}
            for i in range(9)
        ],
    }
    export_json(paths["manifest"], value)
    export_json(
        paths["codex_v"],
        {
            "status": "READY_FOR_CLAUDE_CHECK",
            "unresolved_findings": [],
            "ruling_sha256": FINALISATION,
            "attempt2_ruling_sha256": ATTEMPT2_RULING,
            "executable_commit": PIN,
            "manifest_sha256": {"canary-manifest.json": digest(paths["manifest"])},
        },
    )
    export_json(paths["cumulative"], {"fixture": "accounting is independently tested"})
    export_json(
        paths["preflight"],
        {
            "status": "PASS",
            "elapsed_seconds": 25,
            "manifest": {"sha256": digest(paths["manifest"])},
            "inputs": value["inputs"],
        },
    )
    check = {n: True for n in admission.CHECK_FIELDS}
    check.update(
        {
            "status": "PASS",
            "stage": "canary",
            "ruling_sha256": FINALISATION,
            "attempt2_ruling_sha256": ATTEMPT2_RULING,
            "manifest_sha256": digest(paths["manifest"]),
            "codex_v_sha256": digest(paths["codex_v"]),
            "campaign_log_recorded": True,
            "cumulative_accounting_sha256": digest(paths["cumulative"]),
            "public_fetch_preflight_sha256": digest(paths["preflight"]),
            "qualified_transport_tree": TREE,
            "transport_ruling_sha256": admission.TRANSPORT_RULING,
            "a3_3_exception_ruling_sha256": admission.MIRROR_RULING,
            "control_window": {
                "arm_path": str(tmp_path / "arm"),
                "owner_token": "fixture-owner",
                "root_record": str(tmp_path / "root.json"),
                "registration_evidence": str(tmp_path / "registered.json"),
            },
        }
    )
    monkeypatch.setattr(admission, "verify_cumulative_admission", lambda *a, **k: {})
    monkeypatch.setattr(admission, "TIMING_ARMED", tmp_path / "arm")
    (tmp_path / "arm").write_text(
        json.dumps(
            {
                "schema": admission.WINDOW_SCHEMA,
                "kind": "arm",
                "owner": value["job_id"],
                "owner_token": "fixture-owner",
            }
        )
    )
    monkeypatch.setattr(
        admission.subprocess,
        "check_output",
        lambda args, **k: TREE + "\n" if args[1] == "rev-parse" else b"",
    )
    monkeypatch.setattr(
        admission.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=k["input"]),
    )
    return paths, check


def test_canary_admission_binds_checks_into_fresh_sibling(prerequisites, tmp_path):
    paths, check = prerequisites
    export_json(paths["claude"], check)
    admission.admit(stage="canary", broker=tmp_path, **paths)
    result = json.loads(paths["output"].read_text())
    assert result["scientific_admission"] is True
    assert result["p1_attempt2_admission"]["claude_line_check_sha256"] == digest(paths["claude"])
    assert json.loads(paths["manifest"].read_text())["scientific_admission"] is False


@pytest.mark.parametrize("missing", admission.CHECK_FIELDS)
def test_every_operational_condition_is_mandatory(prerequisites, tmp_path, missing):
    paths, check = prerequisites
    check[missing] = False
    export_json(paths["claude"], check)
    with pytest.raises(ValueError, match="operator conditions"):
        admission.admit(stage="canary", broker=tmp_path, **paths)
    assert not paths["output"].exists()


def test_no_historical_canary_can_admit_canary_stage(prerequisites, tmp_path):
    paths, check = prerequisites
    export_json(paths["claude"], check)
    with pytest.raises(ValueError, match="prior canary"):
        admission.admit(stage="canary", broker=tmp_path, canary=tmp_path / "old", **paths)
