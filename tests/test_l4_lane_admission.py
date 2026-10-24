"""Attempt-2 admission wiring for l4-label-faults/v2 (judgment §4 steps 1c, 3-4, 6)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from research.bl_labels import f1_analyze
from research.colab_cli import injection_manifest
from research.rpc_qualification import admission

REPO = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call(**extra: Any) -> dict[str, Any]:
    return {
        "command": "version",
        "status": "completed",
        "elapsed_seconds": 0.1,
        "timeout_seconds": 30.0,
        **extra,
    }


def _acceptance(tmp_path: Path, l4_calls: list[dict[str, Any]]) -> Path:
    """A fix-pin final-only result with both arms and their planned manifests."""
    jobs, execution = [], []
    for name, gpu in (("cpu", "cpu"), ("l4", "L4")):
        job_id = f"acceptance-1-{name}"
        entries = injection_manifest.acceptance_entries(gpu)
        planned = tmp_path / f"{name}.injection.json"
        digest = injection_manifest.write(planned, job_id, entries, purpose="test")
        jobs.append(
            {"name": name, "injection_manifest": str(planned), "injection_manifest_sha256": digest}
        )
        calls = [_call()]
        if name == "cpu":
            silence = entries[0]
            calls.append(
                _call(
                    command="exec",
                    status="failed",
                    elapsed_seconds=30.0,
                    request_id=silence["request_id"],
                    request_sha256=silence["request_sha256"],
                )
            )
            terminal = {"status": "PASS", "failure_class": None, "error": None}
        else:
            calls += l4_calls
            terminal = {
                "status": "INCOMPLETE",
                "failure_class": "ACCOUNTING",
                "error": "AccountingError('supervised assignment disappeared')",
            }
        receipt = tmp_path / f"{name}-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "stop_verified": True,
                    "cleanup_error": None,
                    "cli_calls": calls,
                    **terminal,
                }
            )
        )
        execution.append(
            {
                "job": name,
                "status": "PASS",
                "receipt": str(receipt),
                "receipt_sha256": _sha(receipt),
            }
        )
    result = tmp_path / "acceptance-result.json"
    result.write_text(
        json.dumps(
            {
                "final_plan": {
                    "commit": admission.TRANSPORT_PIN,
                    "jobs": jobs,
                    "execution": execution,
                }
            }
        )
    )
    return result


@pytest.fixture(autouse=True)
def _harness_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """require_final_only (harness gates/source binding) has its own tests."""
    from research.colab_cli import admission as transport_admission

    monkeypatch.setattr(transport_admission, "require_final_only", lambda result, repo: result)


def _exec(status: str) -> dict[str, Any]:
    return _call(
        command="exec",
        status=status,
        elapsed_seconds=30.0 if status == "failed" else 2.0,
        request_id="r" * 32,
        request_sha256="h" * 64,
    )


def test_pin_is_rebound_to_the_fix_commit() -> None:
    assert admission.TRANSPORT_PIN == "4216927615fe80581daf4fb98d608a807c370ef3"
    assert admission.FAULT_DEFINITION == "l4-label-faults/v2"
    assert admission.SCHEMA == "rpc-qualification-profile/v2"


def test_prerequisite_arms_pass_with_recovered_l4_timeout(tmp_path: Path) -> None:
    path = _acceptance(tmp_path, [_exec("failed"), _exec("completed")])
    arms = admission.prerequisite_arms(path, REPO)
    assert arms["l4"]["verdict"] == "PASS" and arms["l4"]["recovered_events"] == 1
    assert arms["cpu"]["verdict"] == "PASS"  # injected silence excluded by identity
    assert admission.require_prerequisite(path, REPO, "l4")["verdict"] == "PASS"


def test_unresolved_l4_timeout_refuses_l4_but_not_cpu(tmp_path: Path) -> None:
    path = _acceptance(tmp_path, [_exec("failed")])
    with pytest.raises(ValueError, match="l4 arm has counted v2 faults"):
        admission.require_prerequisite(path, REPO, "l4")
    assert admission.require_prerequisite(path, REPO, "cpu")["verdict"] == "PASS"


def test_prerequisite_refuses_old_pin_and_tampered_manifest(tmp_path: Path) -> None:
    path = _acceptance(tmp_path, [])
    value = json.loads(path.read_text())
    value["final_plan"]["commit"] = "a18bf6b4f606f023bfd7c035e68a48e64d7f7457"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="v2 fix pin"):
        admission.prerequisite_arms(path, REPO)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    path = _acceptance(fresh, [])
    planned = Path(json.loads(path.read_text())["final_plan"]["jobs"][0]["injection_manifest"])
    planned.chmod(0o644)
    planned.write_text(planned.read_text().replace("absence-probe", "absence-probe-2"))
    with pytest.raises(ValueError, match="changed after planning"):
        admission.prerequisite_arms(path, REPO)


def test_dispatch_needs_reviewer_accept_at_this_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "CLAUDE_V_L4_REQUAL_02_20261024.json"
    monkeypatch.setattr(admission, "REVIEW_PATH", record)
    with pytest.raises(ValueError, match="lacks its independent reviewer ACCEPT"):
        admission.require_review()
    good = {
        "schema": "l4-requal-02-review/v1",
        "verdict": "ACCEPT",
        "transport_pin": admission.TRANSPORT_PIN,
        "source_sha256": {"research/colab_cli/supervisor.py": "0" * 64},
    }
    for bad in ({**good, "verdict": "REVISE"}, {**good, "transport_pin": "a18bf6b4"}):
        record.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="ACCEPT at this pin"):
            admission.require_review()
    record.write_text(json.dumps(good))
    assert admission.require_review()["verdict"] == "ACCEPT"
    requalify = (REPO / "tools/ops/requalify.py").read_text()
    assert 'registry.name == "l4-label-lane-qualified.json"' in requalify
    assert 'require_review(review["commit"])' in requalify


def _dispatch_module():
    spec = importlib.util.spec_from_file_location("ops_dispatch", REPO / "tools/ops/dispatch.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "tools/ops"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(REPO / "tools/ops"))
    return module


def test_lane_record_override_is_narrow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dispatch = _dispatch_module()
    profile = {
        "script": "research/rpc_qualification/colab.sh",
        "rq": {"schema": "rpc-qualification-profile/v2"},
    }
    monkeypatch.delenv(dispatch.LANE_RECORD_ENV, raising=False)
    assert dispatch.qualification_record(tmp_path, profile) == tmp_path / "colab-cli-qualified.json"
    monkeypatch.setenv(dispatch.LANE_RECORD_ENV, str(tmp_path / dispatch.LANE_RECORD_NAME))
    assert (
        dispatch.qualification_record(tmp_path, profile)
        == (tmp_path / dispatch.LANE_RECORD_NAME).resolve()
    )
    with pytest.raises(dispatch.Refused, match="only for the attempt-2"):
        dispatch.qualification_record(tmp_path, {"script": "research/colab_cli/launch.sh"})
    monkeypatch.setenv(dispatch.LANE_RECORD_ENV, str(tmp_path / "elsewhere.json"))
    with pytest.raises(dispatch.Refused, match="must be"):
        dispatch.qualification_record(tmp_path, profile)


def _f1_attempt(tmp_path: Path, calls: list[dict[str, Any]]) -> Path:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    manifest = attempt / "manifest.json"
    manifest.write_text(json.dumps({"env": {"F1_PLATFORM": "l4"}}))
    (attempt / "receipt.json").write_text(
        json.dumps(
            {
                "job_id": "f1",
                "status": "INCOMPLETE",
                "failure_class": "SUPERVISOR",
                "error": "x",
                "stop_verified": True,
                "cleanup_error": None,
                "cli_calls": calls,
                "manifest_sha256": _sha(manifest),
                "exports": [],
            }
        )
    )
    return attempt


def test_f1_bis_reads_the_stall_rule_as_v2(tmp_path: Path) -> None:
    attempt = _f1_attempt(tmp_path, [_exec("failed"), _exec("completed")])
    frozen = f1_analyze.load_attempt(attempt, "b" * 64)
    assert len(frozen["transport_faults"]) == 1  # historical census: recovered still counts
    v2 = f1_analyze.load_attempt(attempt, "b" * 64, fault_definition=f1_analyze.V2_FAULTS)
    assert [row["class"] for row in v2["transport_faults"]] == ["C3"]  # only the terminal
    assert v2["transport_v2"]["recovered_events"] == 1
    with pytest.raises(ValueError, match="not part of"):
        f1_analyze.load_attempt(
            attempt,
            "b" * 64,
            allow_refund_upload_carveout=True,
            fault_definition=f1_analyze.V2_FAULTS,
        )
