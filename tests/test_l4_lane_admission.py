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
    l4_observer_gate: dict[str, Any] | None = None
    for name, gpu in (("cpu", "cpu"), ("l4", "L4")):
        job_id = f"acceptance-1-{name}"
        entries = injection_manifest.acceptance_entries(gpu)
        for entry in entries:
            if entry["command"] == "exec":
                entry["failure_type"] = "timeout"
            elif entry["command"] == "terminal":
                entry.update(
                    {
                        "watchdog_event_id": "mac-loss-watchdog-unassign",
                        "watchdog_plan": {
                            "lease_seconds": 180,
                            "release_bound_seconds": 300,
                            "renewal_suppressed": True,
                            "checkpoint_exported_during_grace": True,
                        },
                    }
                )
            else:
                entry.update(
                    {
                        "remote": "/content/acceptance/" + entry["injection_id"],
                        "operation_id": entry["injection_id"] + "-operation",
                        "attempt_id": entry["injection_id"] + "-attempt",
                        "failure_type": "absence",
                    }
                )
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
                    "endpoint": "https://example.test/accounting",
                    "cli_calls": calls,
                    **terminal,
                }
            )
        )
        if name == "l4":
            l4_observer_gate = {
                "job_id": job_id,
                "planned_event_id": "mac-loss-watchdog-unassign",
                "endpoint": "https://example.test/accounting",
                "lease_seconds": 180,
                "release_bound_seconds": 300,
                "observed_after_lease_seconds": 1,
                "renewal_suppressed": True,
                "checkpoint_exported_during_grace": True,
                "assignment_disappeared_within_bound": True,
            }
            (tmp_path / "acceptance-observer.json").write_text(
                json.dumps(
                    {
                        "profile": "final-only",
                        "gates": {"mac-loss-watchdog-unassign": l4_observer_gate},
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
                    "commit": admission.git_commit(REPO),
                    "jobs": jobs,
                    "execution": execution,
                    "gates": [
                        {
                            "id": "mac-loss-watchdog-unassign",
                            "evidence_record": l4_observer_gate,
                        }
                    ],
                }
            }
        )
    )
    return result


@pytest.fixture(autouse=True)
def _harness_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the admission decision tests independent of classifier fixture shape."""
    from research.bl_labels import l4_faults_v2
    from research.colab_cli import admission as transport_admission

    monkeypatch.setattr(transport_admission, "require_final_only", lambda result, repo: result)

    original = l4_faults_v2.classify_session

    def classified(receipt: dict, _manifest: dict | None = None, **_evidence: Any) -> dict:
        if _manifest is None:
            return original(receipt)
        calls = receipt["cli_calls"]
        c1 = [call for call in calls if call["elapsed_seconds"] > call["timeout_seconds"] + 2]
        l4_unrecovered = (
            receipt["job_id"].endswith("-l4")
            and any(call["status"] == "failed" for call in calls)
            and not any(
                call["status"] == "completed" and call.get("request_id") == "r" * 32
                for call in calls
            )
        )
        counted = ([{"class": "C1"}] if c1 else []) + ([{"class": "C2"}] if l4_unrecovered else [])
        return {
            "verdict": "FAIL" if counted else "PASS",
            "counted": counted,
            "recovered_events": int(not counted and receipt["job_id"].endswith("-l4")),
        }

    monkeypatch.setattr(l4_faults_v2, "classify_session", classified)


def _exec(status: str) -> dict[str, Any]:
    return _call(
        command="exec",
        status=status,
        elapsed_seconds=30.0 if status == "failed" else 2.0,
        request_id="r" * 32,
        request_sha256="h" * 64,
    )


def test_pin_is_rebound_to_the_fix_commit() -> None:
    assert admission.TRANSPORT_PIN == "d68ba07c350d3f642879ee6df2cf9af9d34f0601"
    assert admission.FAULT_DEFINITION == "l4-label-faults/v2"
    assert admission.SCHEMA == "rpc-qualification-profile/v2"


def test_prerequisite_arms_pass_with_recovered_l4_timeout(tmp_path: Path) -> None:
    path = _acceptance(tmp_path, [_exec("failed"), _exec("completed")])
    arms = admission.prerequisite_arms(path, REPO)
    assert arms["l4"]["verdict"] == "PASS" and arms["l4"]["recovered_events"] == 1
    assert arms["cpu"]["verdict"] == "PASS"  # injected silence excluded by identity
    assert admission.require_prerequisite(path, REPO, "l4")["verdict"] == "PASS"


def test_counted_cpu_prerequisite_fault_refuses_l4_admission(tmp_path: Path) -> None:
    """A required CPU-arm fault ends the shared attempt before an L4 wave."""
    path = _acceptance(tmp_path, [_exec("failed"), _exec("completed")])
    result = json.loads(path.read_text())
    cpu = next(row for row in result["final_plan"]["execution"] if row["job"] == "cpu")
    receipt = Path(cpu["receipt"])
    receipt_value = json.loads(receipt.read_text())
    receipt_value["cli_calls"][0]["elapsed_seconds"] = 32.1  # C1: clean L4, faulted CPU arm.
    receipt.write_text(json.dumps(receipt_value))
    cpu["receipt_sha256"] = _sha(receipt)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="required .* arm has counted v2 faults"):
        admission.require_prerequisite(path, REPO, "l4")
    with pytest.raises(ValueError, match="required .* arm has counted v2 faults"):
        admission.require_prerequisite(path, REPO, "cpu")


def test_prerequisite_refuses_old_pin_and_tampered_manifest(tmp_path: Path) -> None:
    path = _acceptance(tmp_path, [])
    value = json.loads(path.read_text())
    value["final_plan"]["commit"] = "a18bf6b4f606f023bfd7c035e68a48e64d7f7457"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reviewed package commit"):
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
    monkeypatch.setattr(admission, "verify_review_sources", lambda *_args: None)
    monkeypatch.setattr(admission, "verify_review_artifacts", lambda *_args: None)
    with pytest.raises(ValueError, match="lacks its independent reviewer ACCEPT"):
        admission.require_review()
    good = {
        "schema": "l4-requal-02-review/v1",
        "verdict": "ACCEPT",
        "transport_pin": admission.TRANSPORT_PIN,
        "package_commit": "0" * 40,
        "source_sha256": {"research/colab_cli/supervisor.py": "0" * 64},
    }
    for bad in ({**good, "verdict": "REVISE"}, {**good, "transport_pin": "a18bf6b4"}):
        record.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="ACCEPT at this pin"):
            admission.require_review()
    record.write_text(json.dumps(good))
    assert admission.require_review()["verdict"] == "ACCEPT"
    requalify = (REPO / "tools/ops/requalify.py").read_text()
    assert 'is_lane_package(repo, review["commit"])' in requalify
    assert 'require_review(repo=repo, package_commit=review["commit"])' in requalify


def test_reviewer_accept_requires_exact_package_commit_and_complete_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial ACCEPT map cannot authorize either prerequisite or later waves."""
    record = tmp_path / "CLAUDE_V_L4_REQUAL_02_20261024.json"
    monkeypatch.setattr(admission, "REVIEW_PATH", record)
    monkeypatch.setattr(admission, "verify_review_artifacts", lambda *_args: None)
    original_verify = admission.verify_review_sources
    inventory = admission.review_source_inventory(REPO)
    good = {
        "schema": "l4-requal-02-review/v1",
        "verdict": "ACCEPT",
        "transport_pin": admission.TRANSPORT_PIN,
        "package_commit": admission.git_commit(REPO),
        "source_sha256": inventory,
    }
    incomplete = {
        **good,
        "source_sha256": {
            "research/colab_cli/supervisor.py": inventory["research/colab_cli/supervisor.py"]
        },
    }
    record.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="complete source inventory"):
        admission.require_review(repo=REPO)
    wrong_commit = {**good, "package_commit": "0" * 40}
    record.write_text(json.dumps(wrong_commit))
    with pytest.raises(ValueError, match="reviewed package commit"):
        admission.require_review(repo=REPO)
    record.write_text(json.dumps(good))
    monkeypatch.setattr(admission, "verify_review_sources", lambda *_args: None)
    assert admission.require_review()["package_commit"] == admission.git_commit(REPO)
    monkeypatch.setattr(admission, "verify_review_sources", original_verify)


def _review_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from research.bl_labels.exports import publish_json

    repo = tmp_path / "repo"
    repo.mkdir()
    pack = tmp_path / "pack"
    pack.mkdir()
    monkeypatch.setattr(admission, "PACK_ROOT", pack)
    monkeypatch.setattr(admission, "PROOF_RELATIVE", "proof.json")
    trace = {"scenarios": {"synthetic": {"baseline": {}, "current": {}}}}
    publish_json(pack / "DIFFERENTIAL_TRACE.json", trace)
    publish_json(
        repo / "proof.json",
        {
            "differential_trace": trace,
            "differential_trace_sha256": _sha(pack / "DIFFERENTIAL_TRACE.json"),
        },
    )
    publish_json(
        pack / "plan.json",
        {
            "source_commit": "0" * 40,
            "transport_pin": admission.TRANSPORT_PIN,
            "transport_tree": admission.TRANSPORT_TREE,
            "fake_proof": {"path": str(repo / "proof.json"), "sha256": _sha(repo / "proof.json")},
        },
    )
    (pack / "GUARDED_OPERATOR_COMMANDS_V6.sh").write_text("# exact helper\n")
    authority = tmp_path / "authority.md"
    authority.write_text("ratified\n")
    monkeypatch.setattr(admission, "AUTHORITY_FILES", {"R": (authority, _sha(authority))})
    return {"repo": repo, "pack": pack, "authority": authority}


@pytest.mark.parametrize("field", ["artifact_sha256", "authority_sha256", "transport_tree"])
def test_accept_checks_artifact_digests_and_authorities(tmp_path, monkeypatch, field):
    paths = _review_artifacts(tmp_path, monkeypatch)
    repo = paths["repo"]
    review = admission.review_artifact_bindings(repo, "0" * 40)
    admission.verify_review_artifacts(review, repo, "0" * 40)
    bad = {**review, field: {} if field.endswith("sha256") else "1" * 40}
    with pytest.raises(ValueError, match="artifact/authority digests"):
        admission.verify_review_artifacts(bad, repo, "0" * 40)


@pytest.mark.parametrize("changed", ["helper", "authority"])
def test_accept_refuses_changed_helper_or_authority(tmp_path, monkeypatch, changed):
    paths = _review_artifacts(tmp_path, monkeypatch)
    repo = paths["repo"]
    review = admission.review_artifact_bindings(repo, "0" * 40)
    path = (
        paths["authority"]
        if changed == "authority"
        else paths["pack"] / "GUARDED_OPERATOR_COMMANDS_V6.sh"
    )
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="authority bytes|artifact/authority digests"):
        admission.verify_review_artifacts(review, repo, "0" * 40)


def test_accept_inventory_includes_executed_ops_wrappers():
    inventory = admission.review_source_inventory(REPO)
    assert {
        "tools/ops/dispatch.py",
        "tools/ops/qualification.py",
        "tools/ops/dispatch.sh",
        "tools/ops/watchd.sh",
    } <= inventory.keys()


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
    # The historical fixture does not carry prospective durable-reply or marker
    # evidence, so it may not be projected as a new-session PASS.
    assert [row["class"] for row in v2["transport_faults"]] == ["C2", "C3"]
    assert v2["transport_v2"]["evidence_missing"]
    assert v2["transport_v2"]["recovered_events"] == 0
    with pytest.raises(ValueError, match="not part of"):
        f1_analyze.load_attempt(
            attempt,
            "b" * 64,
            allow_refund_upload_carveout=True,
            fault_definition=f1_analyze.V2_FAULTS,
        )
