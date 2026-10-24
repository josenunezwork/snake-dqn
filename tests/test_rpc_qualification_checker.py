"""Focused outcome-blind checks for the C1 qualification one-look checker."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from research.bl_labels.exports import publish_json
from research.rpc_qualification import admission, checker


@pytest.fixture(autouse=True)
def _remote_publication(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """Keep the unit test offline; production fetches and reads origin itself."""
    monkeypatch.setattr(
        checker,
        "_remote_json",
        lambda plan, _relative: json.loads((plan.parent / "remote.json").read_text()),
    )
    monkeypatch.setattr(
        checker,
        "require_review",
        lambda *, repo, package_commit: {"artifact_sha256": {"plan": _sha(repo / "plan.json")}},
    )
    monkeypatch.setattr(checker, "ARTIFACT_ROOT", tmp_path / "artifacts")
    # Source-map validation has its own synthetic proof cases.  These fixtures
    # intentionally use an isolated temporary plan repository.
    if request.node.name != "test_fake_proof_refuses_a_stale_or_partial_current_source_map":
        monkeypatch.setattr(checker, "validate_fake_proof", lambda *_args: {})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(root: Path, name: str, platform: str, mode: str = "loop") -> dict:
    output = root / name
    output.mkdir()
    namespace = "rq/" + name
    # Mirror the supervisor's canonical collection layout.
    manifest = output / "manifest.json"
    envelope = {
        "effective": {
            "unit_cap": 0.12 if mode == "injection" else (5.0 if platform == "l4" else 0.30),
            "timeout_minutes": 30 if mode == "injection" else 160,
        }
    }
    job = {
        "job_id": name,
        "mode": mode,
        "platform": platform,
        "namespace": namespace,
        "allocation": "f1",
        "manifest": str(manifest),
        "output_dir": str(output),
        "runtime": str(output / "exports" / "runtime-result.json"),
    }
    from research.colab_cli import injection_manifest

    entries = (
        [
            {
                "injection_id": f"response-loss-{number}",
                "type": "post_send_response_loss",
                "command": "exec",
                "request_id": f"request-{number}",
                "request_sha256": "c" * 64,
                "failure_type": "post_send_response_loss",
                "max_calls": 1,
            }
            for number in range(3)
        ]
        if mode == "injection"
        else []
    )
    planned = root / f"{name}.injection-manifest.json"
    planned_sha256 = injection_manifest.write(planned, name, entries, purpose="test")
    publish_json(
        manifest,
        {
            "job_id": name,
            "commit": "a18bf6b4" + "0" * 32,
            "script": "research/rpc_qualification/colab.sh",
            "output_dir": str(output),
            "source_paths": ["research/rpc_qualification"],
            "timeout_minutes": 30 if mode == "injection" else 160,
            "budget": {
                "unit_cap": envelope["effective"]["unit_cap"],
                "rate_units_per_hour": 0.08 if platform == "cpu" else 1.54,
                "other_assignments": {},
            },
            "rq": {
                "schema": checker.SCHEMA,
                **{key: job[key] for key in ("mode", "platform", "namespace")},
                "bank_sha256": checker.FROZEN_BANK,
                "worker_pin": checker.F1_PIN,
                "frozen_source_sha256": "a" * 64,
                "envelope": envelope,
                "injection_manifest": {
                    "path": str(planned),
                    "sha256": planned_sha256,
                    "request_ids": [entry["request_id"] for entry in entries],
                },
            },
        },
    )
    receipt = {
        "job_id": name,
        "commit": "a18bf6b4" + "0" * 32,
        "generation": "generation-" + name,
        "elapsed_seconds": 60.0,
        "cli_calls": [
            {
                "command": "exec",
                "attempt": 1,
                "timeout_seconds": 10,
                "elapsed_seconds": 1,
                "status": "completed",
                "request_id": "request-" + name,
                "request_sha256": "d" * 64,
            }
        ],
        "marker_census": {"request-" + name: {"started": 1, "done": 1, "code_sha256": "d" * 64}},
        "status": "PASS",
        "manifest_sha256": _sha(manifest),
        "terminal": {
            "returncode": 0,
            "generation": "generation-" + name,
            "job_id": name,
        },
        "stop_verified": True,
        "backend_disappeared": True,
        "contents_transfers": [],
        "transfers": [],
    }
    if mode == "loop":
        unit = {
            "schema": "rpc-qualification-loop/v1",
            "status": "COMPLETE",
            "namespace": namespace,
            "allocation": "f1",
            "index": 0,
            "unit_id": f"{namespace}/f1/00000000",
            "root_id": "root-0",
            "integrity": {
                "digest": "d0",
                "label_row_sha256": "l0",
                "digest_checks": 1,
                "bare_audit_mismatches": 0,
            },
        }
        publish_json(output / "units" / "unit-00000000.json", unit)
        item = output / "units" / "unit-00000000.json"
        export = output / "exports"
        export.mkdir()
        unit_target = export / "units" / "unit-00000000.json"
        unit_target.parent.mkdir()
        for suffix in ("", ".sha256", ".complete"):
            unit_target.with_name(unit_target.name + suffix).write_bytes(
                item.with_name(item.name + suffix).read_bytes()
            )
        rows = []
        for index in range(128):
            source = output / "units" / f"unit-{index:08d}.json"
            value = {
                **unit,
                "index": index,
                "unit_id": f"{namespace}/f1/{index:08d}",
                "root_id": f"root-{index % 64}",
                "integrity": {
                    "digest": f"d{index % 64}",
                    "label_row_sha256": f"l{index % 64}",
                    "digest_checks": 1,
                    "bare_audit_mismatches": 0,
                },
            }
            if index:
                publish_json(source, value)
            target = export / "units" / source.name
            if index:
                for suffix in ("", ".sha256", ".complete"):
                    target.with_name(target.name + suffix).write_bytes(
                        source.with_name(source.name + suffix).read_bytes()
                    )
            rows.append(
                {
                    "index": index,
                    "path": f"units/{source.name}",
                    "sha256": _sha(target),
                    "size": target.stat().st_size,
                }
            )
        publish_json(
            export / "verified-ledger.json",
            {
                "schema": "rpc-qualification-verified-ledger/v1",
                "namespace": namespace,
                "allocation": "f1",
                "bank_sha256": checker.FROZEN_BANK,
                "source_commit": checker.F1_PIN,
                "units": rows,
            },
        )
        publish_json(
            output / "exports" / "runtime-result.json",
            {
                "schema": "rpc-qualification-loop/v1",
                "status": "COMPLETE",
                "namespace": namespace,
                "platform": platform,
                "bank_sha256": checker.FROZEN_BANK,
                "source_commit": checker.F1_PIN,
                "source_closure": {
                    "archive_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "members": 1,
                },
                "workers": 2 if platform == "cpu" else 12,
                "allocation": "f1",
                "root_order": [f"root-{index}" for index in range(64)],
                "loop_seconds": 3600,
                "measured_loop_seconds": 3600,
                "units_completed": 128,
                "root_repeat_counts": {str(index): 2 for index in range(64)},
                "integrity": {
                    "all_attempted_roots_clean": True,
                    "repeat_digest_consistent": True,
                    "bare_audit_mismatches": 0,
                    "digest_checks": 128,
                },
            },
        )
        receipt["exports"] = []
        for target in sorted(export.rglob("*.json")):
            receipt["exports"].append(
                {
                    "path": target.relative_to(export).as_posix(),
                    "sha256": _sha(target),
                    "size": target.stat().st_size,
                }
            )
    publish_json(output / "receipt.json", receipt)
    return job


def _injection(root: Path) -> dict:
    job = _job(root, "injection", "cpu", "injection")
    output = Path(job["output_dir"])
    export = output / "exports"
    export.mkdir()
    units = []
    for number in range(3):
        request = f"request-{number}"
        unit_id = f"{job['namespace']}/f1/{number:08d}"
        code_hash = "c" * 64
        row = {
            "request_id": request,
            "unit_id": unit_id,
            "execution_count": 1,
            "drop_call_index": 0,
        }
        for marker in ("started", "done", "execution_started", "execution_done", "result"):
            path = output / f"{request}.{marker}.json"
            value = {"request_id": request}
            if marker in {"started", "done"}:
                value["sha256"] = code_hash
            if marker == "done":
                value["ok"] = True
            if marker in {"execution_started", "execution_done", "result"}:
                value.update({"unit_id": unit_id, "execution_count": 1})
            publish_json(path, value)
            row[marker] = {
                "path": path.name,
                "local_path": str(path),
                "sha256": _sha(path),
                "size": path.stat().st_size,
            }
        units.append(row)
    ledger_rows = []
    for number in range(3):
        injected = export / "units" / f"unit-{number:08d}.json"
        publish_json(
            injected,
            {
                "schema": "rpc-qualification-loop/v1",
                "status": "COMPLETE",
                "namespace": job["namespace"],
                "allocation": "f1",
                "index": number,
                "unit_id": f"{job['namespace']}/f1/{number:08d}",
                "root_id": f"root-{number}",
                "request_id": f"request-{number}",
                "execution_count": 1,
                "integrity": {
                    "digest": f"d{number}",
                    "label_row_sha256": f"l{number}",
                    "digest_checks": 1,
                    "bare_audit_mismatches": 0,
                },
            },
        )
        ledger_rows.append(
            {
                "index": number,
                "unit_id": f"{job['namespace']}/f1/{number:08d}",
                "path": f"units/unit-{number:08d}.json",
                "sha256": _sha(injected),
                "size": injected.stat().st_size,
            }
        )
        units[number]["result"] = {
            "path": f"units/unit-{number:08d}.json",
            "local_path": str(injected),
            "sha256": _sha(injected),
            "size": injected.stat().st_size,
        }
    publish_json(
        export / "verified-ledger.json",
        {
            "schema": "rpc-qualification-verified-ledger/v1",
            "namespace": job["namespace"],
            "allocation": "f1",
            "source_commit": checker.F1_PIN,
            "bank_sha256": checker.FROZEN_BANK,
            "units": ledger_rows,
        },
    )
    calls = []
    for index, row in enumerate(units):
        row["drop_call_index"] = len(calls)
        calls.append(
            {
                "command": "exec",
                "attempt": 1,
                "timeout_seconds": 10,
                "elapsed_seconds": 1,
                "status": "failed",
                "request_sha256": "c" * 64,
                "failure_type": "post_send_response_loss",
                "injected": True,
                "injection_kind": "post_send_response_loss",
                "unit_id": row["unit_id"],
                "request_id": row["request_id"],
            }
        )
        calls.extend(
            {
                "command": "exec",
                "attempt": attempt,
                "timeout_seconds": 10,
                "elapsed_seconds": 1,
                "status": "completed",
                "request_sha256": "c" * 64,
                "unit_id": row["unit_id"],
                "request_id": row["request_id"],
            }
            for attempt in (2, 3)
        )
    receipt = output / "receipt.json"
    current = json.loads(receipt.read_text())
    current["cli_calls"] = calls
    current["marker_census"] = {
        row["request_id"]: {"started": 1, "done": 1, "code_sha256": "c" * 64} for row in units
    }
    receipt.unlink()
    receipt.with_name(receipt.name + ".sha256").unlink()
    receipt.with_name(receipt.name + ".complete").unlink()
    publish_json(receipt, current)
    publish_json(
        output / "injection-proof.json",
        {
            "schema": "rpc-qualification-injection-proof/v1",
            "status": "COMPLETE",
            "injected_units": units,
            "completed_reissue": {"request_id": "request-0", "same_result_sha256": True},
        },
    )
    runtime = export / "runtime-result.json"
    publish_json(
        runtime,
        {
            "schema": "rpc-qualification-loop/v1",
            "status": "COMPLETE",
            "mode": "injection",
            "namespace": job["namespace"],
            "allocation": "injection",
            "bank_sha256": checker.FROZEN_BANK,
            "source_commit": checker.F1_PIN,
            "source_closure": {
                "archive_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "members": 1,
            },
            "injected_units": 3,
        },
    )
    current["exports"] = []
    for target in sorted(export.rglob("*.json")):
        current["exports"].append(
            {
                "path": target.relative_to(export).as_posix(),
                "sha256": _sha(target),
                "size": target.stat().st_size,
            }
        )
    receipt.unlink()
    receipt.with_name(receipt.name + ".sha256").unlink()
    receipt.with_name(receipt.name + ".complete").unlink()
    publish_json(receipt, current)
    return job


def _plan(root: Path) -> tuple[Path, list[dict]]:
    jobs = [_job(root, f"cpu-{index}", "cpu") for index in range(3)]
    jobs += [_job(root, f"l4-{index}", "l4") for index in range(3)]
    jobs.append(_injection(root))
    fake = root / "fake.json"
    publish_json(fake, {"schema": "rpc-qualification-fake-cli-proof/v1", "status": "PASS"})
    plan = root / "plan.json"
    publish_json(
        plan,
        {
            "schema": checker.SCHEMA,
            "source_commit": "a18bf6b4" + "0" * 32,
            "bank_sha256": checker.FROZEN_BANK,
            "worker_pin": checker.F1_PIN,
            "jobs": jobs,
            "repo": str(root),
            "publication_ref": checker.BRANCH_REF,
            "publication_ledger": "docs/research/results/test-ledger.json",
            "fake_proof": {"path": str(fake), "sha256": _sha(fake)},
        },
    )
    return plan, jobs


def _published(plan: Path, root: Path) -> Path:
    publication = root / "publication.json"
    checker.publish(plan, publication)
    value = json.loads(publication.read_text())
    publish_json(
        root / "remote.json",
        value,
    )
    return publication


def test_publish_then_one_look_verdict_passes_without_outcomes(tmp_path: Path) -> None:
    plan, _jobs = _plan(tmp_path)
    publication = _published(plan, tmp_path)
    verdict = tmp_path / "verdict.json"
    result = checker.verdict(plan, publication, verdict)
    assert result["injection"]["status"] == "PASS", result["injection"]
    assert result["platforms"]["cpu"]["status"] == "PASS"
    assert result["platforms"]["l4"]["status"] == "PASS"
    assert json.loads(verdict.read_text())["one_look"] is True


def test_refuses_short_loop_missing_call_reused_namespace_and_fault(tmp_path: Path) -> None:
    plan, jobs = _plan(tmp_path)
    runtime = Path(jobs[0]["runtime"])
    value = json.loads(runtime.read_text())
    value["loop_seconds"] = 3599
    # The immutable result must be regenerated in a fresh fixture; direct overwrite is prohibited.
    short = tmp_path / "short"
    short.mkdir()
    short_plan, short_jobs = _plan(short)
    short_runtime = Path(short_jobs[0]["runtime"])
    short_value = json.loads(short_runtime.read_text())
    short_value["loop_seconds"] = 3599
    short_runtime.unlink()
    short_runtime.with_name(short_runtime.name + ".sha256").unlink()
    short_runtime.with_name(short_runtime.name + ".complete").unlink()
    publish_json(short_runtime, short_value)
    publication = _published(short_plan, short)
    short_result = checker.verdict(short_plan, publication, short / "verdict.json")
    # The replacement no longer matches the receipt-bound export digest.
    assert short_result["platforms"]["cpu"]["status"] == "FAIL"

    missing = tmp_path / "missing"
    missing.mkdir()
    missing_plan, missing_jobs = _plan(missing)
    publication = _published(missing_plan, missing)
    receipt = Path(missing_jobs[0]["output_dir"]) / "receipt.json"
    receipt.unlink()
    receipt.with_name(receipt.name + ".sha256").unlink()
    receipt.with_name(receipt.name + ".complete").unlink()
    publish_json(
        receipt,
        {
            "status": "PASS",
            "stop_verified": True,
            "manifest_sha256": _sha(Path(missing_jobs[0]["manifest"])),
            "terminal": {"returncode": 0},
        },
    )
    missing_result = checker.verdict(missing_plan, publication, missing / "verdict.json")
    assert missing_result["platforms"]["cpu"]["status"] == "EVIDENCE_MISSING"

    fault = tmp_path / "fault"
    fault.mkdir()
    fault_plan, fault_jobs = _plan(fault)
    publication = _published(fault_plan, fault)
    receipt = Path(fault_jobs[0]["output_dir"]) / "receipt.json"
    row = json.loads(receipt.read_text())
    row["cli_calls"][0]["status"] = "failed"
    receipt.unlink()
    receipt.with_name(receipt.name + ".sha256").unlink()
    receipt.with_name(receipt.name + ".complete").unlink()
    publish_json(receipt, row)
    result = checker.verdict(fault_plan, publication, fault / "verdict.json")
    assert result["platforms"]["cpu"]["status"] == "FAIL"

    reused = tmp_path / "reused"
    reused.mkdir()
    reused_plan, reused_jobs = _plan(reused)
    plan_value = json.loads(reused_plan.read_text())
    plan_value["jobs"][1]["namespace"] = plan_value["jobs"][0]["namespace"]
    reused_plan.unlink()
    reused_plan.with_name(reused_plan.name + ".sha256").unlink()
    reused_plan.with_name(reused_plan.name + ".complete").unlink()
    publish_json(reused_plan, plan_value)
    with pytest.raises(ValueError, match="reuses a namespace"):
        checker.publish(reused_plan, reused / "publication.json")


def test_injection_refuses_duplicate_execution(tmp_path: Path) -> None:
    plan, jobs = _plan(tmp_path)
    injection = next(job for job in jobs if job["mode"] == "injection")
    evidence = Path(injection["output_dir"]) / "injection-proof.json"
    value = json.loads(evidence.read_text())
    value["injected_units"][0]["execution_count"] = 2
    evidence.unlink()
    evidence.with_name(evidence.name + ".sha256").unlink()
    evidence.with_name(evidence.name + ".complete").unlink()
    publish_json(evidence, value)
    publication = _published(plan, tmp_path)
    result = checker.verdict(plan, publication, tmp_path / "verdict.json")
    assert result["injection"]["status"] == "FAIL"
    assert any(fault["source"] == "v2:C5" for fault in result["injection"]["faults"])


def _rewrite_receipt(job: dict, update) -> None:
    receipt = Path(job["output_dir"]) / "receipt.json"
    value = json.loads(receipt.read_text())
    update(value)
    for suffix in ("", ".sha256", ".complete"):
        receipt.with_name(receipt.name + suffix).unlink()
    publish_json(receipt, value)


def _recovered_exec(value: dict) -> None:
    value["cli_calls"] = [
        {
            "command": "exec",
            "attempt": 1,
            "timeout_seconds": 30,
            "elapsed_seconds": 30.01,
            "status": "failed",
            "failure_class": "CONTROL_RPC_TIMEOUT",
            "request_id": "r" * 32,
            "request_sha256": "h" * 64,
        },
        {
            "command": "exec",
            "attempt": 1,
            "timeout_seconds": 30,
            "elapsed_seconds": 2.3,
            "status": "completed",
            "request_id": "r" * 32,
            "request_sha256": "h" * 64,
        },
    ]


def test_v2_recovered_timeout_without_durable_summary_is_not_accepted(tmp_path: Path) -> None:
    plan, jobs = _plan(tmp_path)
    _rewrite_receipt(next(j for j in jobs if j["job_id"] == "l4-0"), _recovered_exec)
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["platforms"]["l4"]["status"] == "FAIL"
    session = next(r for r in result["platforms"]["l4"]["sessions"] if r["session"] == "l4-0")
    assert session["v2"]["counted_by_class"]["C2"] >= 1


def test_v2_unresolved_timeout_fails_the_platform(tmp_path: Path) -> None:
    plan, jobs = _plan(tmp_path)

    def unresolved(value: dict) -> None:
        _recovered_exec(value)
        value["cli_calls"].pop()

    _rewrite_receipt(next(j for j in jobs if j["job_id"] == "l4-1"), unresolved)
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["platforms"]["l4"]["status"] == "FAIL"
    assert any(fault["source"] == "v2:C2" for fault in result["platforms"]["l4"]["faults"])


def test_mandatory_injection_fault_fails_l4_while_optional_cpu_loops_stay_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real fault in the required injection proof ends the whole attempt."""
    plan, jobs = _plan(tmp_path)
    for job in jobs:
        if job["mode"] == "loop" and job["platform"] == "cpu":
            shutil.rmtree(job["output_dir"])
    original = checker._check_injection

    def counted_injection(plan_path: Path, job: dict) -> dict:
        result = original(plan_path, job)
        result["status"] = "FAIL"
        result["counted_faults"] = 1
        result["faults"] = [{"source": "v2:C6", "record": {"identity": "unplanned"}}]
        result["no_duplicate"] = True
        return result

    monkeypatch.setattr(checker, "_check_injection", counted_injection)
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["platforms"]["cpu"]["status"] == "NOT_RUN"
    assert result["platforms"]["l4"]["status"] == "FAIL"
    assert result["platforms"]["l4"]["counted_faults"] >= 1


def test_injection_drop_outside_the_manifest_identity_fails(tmp_path: Path) -> None:
    from research.colab_cli import injection_manifest

    plan, jobs = _plan(tmp_path)
    injection = next(job for job in jobs if job["mode"] == "injection")
    planned = json.loads(Path(injection["manifest"]).read_text())["rq"]["injection_manifest"]
    manifest_path = Path(planned["path"])
    value = json.loads(manifest_path.read_text())
    value["entries"][0]["request_id"] = "not-the-dropped-request"
    for suffix in ("", ".sha256", ".complete"):
        manifest_path.with_name(manifest_path.name + suffix).unlink()
    injection_manifest.write(manifest_path, value["session_job_id"], value["entries"], purpose="t")
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["injection"]["status"] == "EVIDENCE_MISSING"


def test_fake_proof_refuses_a_stale_or_partial_current_source_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pack cannot attest an older supervisor while executing the v2 one."""
    proof = tmp_path / "fake.json"
    expected = {
        "research/colab_cli/supervisor.py": "a" * 64,
        "research/colab_cli/policy.py": "b" * 64,
        "tests/test_colab_cli_rpc_recovery.py": "c" * 64,
    }
    publish_json(
        proof,
        {
            "schema": "rpc-qualification-fake-cli-proof/v1",
            "status": "PASS",
            "live": False,
            "tests": list(checker.FAKE_PROOF_TESTS),
            "source_commit": "0" * 40,
            "source_sha256": {"research/colab_cli/supervisor.py": "old"},
            "transport_tree": "tree",
        },
    )
    monkeypatch.setattr(checker, "fake_proof_source_inventory", lambda _repo: expected)
    with pytest.raises(ValueError, match="source map is incomplete or stale"):
        checker.validate_fake_proof(proof, tmp_path, "0" * 40)


def _replace_json(path: Path, value: dict) -> None:
    """Replace only owned synthetic immutable triplets to model changed evidence."""
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    publish_json(path, value)


@pytest.mark.parametrize("faulty", [False, True])
def test_optional_cpu_manifest_loss_preserves_retained_receipt(tmp_path: Path, faulty: bool):
    plan, jobs = _plan(tmp_path)
    cpu = [j for j in jobs if j["mode"] == "loop" and j["platform"] == "cpu"]
    if faulty:
        _rewrite_receipt(cpu[0], lambda r: r["cli_calls"][0].update(elapsed_seconds=13))
    for job in cpu:
        for suffix in ("", ".sha256", ".complete"):
            Path(job["manifest"] + suffix).unlink()
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["platforms"]["cpu"]["status"] == ("FAIL" if faulty else "EVIDENCE_MISSING")
    assert result["attempt_status"] == ("FAIL" if faulty else "EVIDENCE_MISSING")
    assert result["platforms"]["l4"]["status"] == result["attempt_status"]
    assert result["attempt_counted_faults"] == int(faulty)
    assert len(result["platforms"]["cpu"]["sessions"]) == 3
    if faulty:
        assert result["attempt_faults"][0]["source"] == "v2:C1"


@pytest.mark.parametrize("signal", ["state_dir", "dispatch", "receipt", "runtime", "publication"])
def test_optional_cpu_durable_evidence_blocks_not_run(tmp_path: Path, monkeypatch, signal):
    plan, jobs = _plan(tmp_path)
    cpu = [j for j in jobs if j["mode"] == "loop" and j["platform"] == "cpu"]
    publication = _published(plan, tmp_path) if signal == "publication" else None
    for job in cpu:
        shutil.rmtree(job["output_dir"])
    job = cpu[0]
    value = json.loads(plan.read_text())
    if signal == "state_dir":
        path = tmp_path / "retained-state"
        path.mkdir()
        value["jobs"][0]["state_dir"] = str(path)
    elif signal == "dispatch":
        (checker.ARTIFACT_ROOT / "ops/dispatch" / job["job_id"]).mkdir(parents=True)
    elif signal in {"receipt", "runtime"}:
        path = tmp_path / ("retained-" + signal + ".json")
        path.write_text("{}")
        value["jobs"][0][signal] = str(path)
    if signal != "publication":
        _replace_json(plan, value)
        publication = _published(plan, tmp_path)
    result = checker.verdict(plan, publication, tmp_path / "verdict.json")
    assert result["platforms"]["cpu"]["status"] == "EVIDENCE_MISSING"
    assert result["attempt_status"] == "EVIDENCE_MISSING"
    assert result["attempt_counted_faults"] == 0


def test_injection_child_only_exit_is_nonqualifying_without_counted_fault(tmp_path: Path):
    plan, jobs = _plan(tmp_path)
    job = next(j for j in jobs if j["mode"] == "injection")

    def child_exit(receipt):
        receipt.update(status="INCOMPLETE", failure_class="CHILD_NONZERO_EXIT")
        receipt["terminal"]["returncode"] = 1

    _rewrite_receipt(job, child_exit)
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    injection = result["injection"]
    assert injection["v2"]["session_failures"][0]["kind"] == "NON_TRANSPORT"
    assert set(injection["v2"]["counted_by_class"].values()) == {0}
    assert injection["status"] == "FAIL"
    assert injection["faults"] == []
    assert result["attempt_counted_faults"] == 0
    assert result["attempt_status"] == result["platforms"]["l4"]["status"] == "EVIDENCE_MISSING"
    assert result["attempt_nonqualifying"] == [injection]
    assert result["platforms"]["l4"]["nonqualifying"] == [injection]


@pytest.mark.parametrize("failure", ["wall", "units", "release", "cleanup"])
def test_injection_real_envelope_and_release_failures_still_count(tmp_path: Path, failure):
    plan, jobs = _plan(tmp_path)
    job = next(j for j in jobs if j["mode"] == "injection")
    if failure == "units":
        # Isolate debit enforcement while remaining within the wall envelope.
        manifest = Path(job["manifest"])
        value = json.loads(manifest.read_text())
        value["budget"]["rate_units_per_hour"] = 0.8
        _replace_json(manifest, value)

    def fail(receipt):
        if failure in {"wall", "units"}:
            receipt["elapsed_seconds"] = 1801 if failure == "wall" else 1000
            receipt["manifest_sha256"] = _sha(Path(job["manifest"]))
        elif failure == "release":
            receipt["stop_verified"] = False
        else:
            receipt["cleanup_error"] = "verified cleanup failed"

    _rewrite_receipt(job, fail)
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["attempt_status"] == "FAIL"
    assert result["attempt_counted_faults"] > 0
    assert all(f["source"] == "v2:C3" for f in result["attempt_faults"])


def _final_review(plan: Path, monkeypatch) -> dict:
    """Use real ACCEPT artifact verification with isolated package-source validation."""
    pack = plan.parent
    trace = {"test": "binding"}
    publish_json(pack / "DIFFERENTIAL_TRACE.json", trace)
    fake = pack / "fake.json"
    _replace_json(
        fake,
        {
            "differential_trace": trace,
            "differential_trace_sha256": _sha(pack / "DIFFERENTIAL_TRACE.json"),
        },
    )
    value = json.loads(plan.read_text())
    value.update(transport_pin=admission.TRANSPORT_PIN, transport_tree=admission.TRANSPORT_TREE)
    value["fake_proof"]["sha256"] = _sha(fake)
    _replace_json(plan, value)
    (pack / "GUARDED_OPERATOR_COMMANDS_V7.sh").write_text("# reviewed helper\n")
    authority = pack / "ruling.md"
    authority.write_text("ratified review04\n")
    monkeypatch.setattr(admission, "PACK_ROOT", pack)
    monkeypatch.setattr(admission, "PROOF_RELATIVE", "fake.json")
    monkeypatch.setattr(admission, "AUTHORITY_FILES", {"review04": (authority, _sha(authority))})
    review_path = pack / "accept.json"
    monkeypatch.setattr(admission, "REVIEW_PATH", review_path)
    package = value["source_commit"]
    checked = []

    def sources(review, repo, package_commit):
        assert package_commit == package  # never HEAD after ledger publication
        checked.append(package_commit)

    monkeypatch.setattr(admission, "verify_review_sources", sources)
    monkeypatch.setattr(
        admission, "git_commit", lambda _repo: pytest.fail("must pass frozen commit")
    )
    review = {
        "schema": "l4-requal-02-review/v1",
        "verdict": "ACCEPT",
        "transport_pin": admission.TRANSPORT_PIN,
        "package_commit": package,
        "source_sha256": {},
        **admission.review_artifact_bindings(pack, package),
    }
    review_path.write_text(json.dumps(review))
    monkeypatch.setattr(checker, "require_review", admission.require_review)
    return {"review_path": review_path, "checked": checked, "package": package}


@pytest.mark.parametrize("changed", ["absent", "helper", "authority", "redirected_plan"])
def test_final_verdict_refuses_binding_changes_before_one_look(
    tmp_path: Path, monkeypatch, changed
):
    plan, _jobs = _plan(tmp_path)
    context = _final_review(plan, monkeypatch)
    if changed == "absent":
        context["review_path"].unlink()
    elif changed == "helper":
        (tmp_path / "GUARDED_OPERATOR_COMMANDS_V7.sh").write_text("changed\n")
    elif changed == "authority":
        (tmp_path / "ruling.md").write_text("changed\n")
    else:
        value = json.loads(plan.read_text())
        for job in value["jobs"]:
            if job["mode"] == "loop" and job["platform"] == "cpu":
                for key in ("manifest", "output_dir", "runtime"):
                    job[key] = str(tmp_path / "absent" / job["job_id"] / key)
        replacement = tmp_path / "replacement.json"
        publish_json(replacement, value)
        plan = replacement
    publication = _published(plan, tmp_path)
    with pytest.raises(ValueError, match="ACCEPT|authority bytes"):
        checker.verdict(plan, publication, tmp_path / "verdict.json")
    assert not (tmp_path / "verdict-look-started.json").exists()
    assert not (tmp_path / "verdict.json").exists()


def test_final_verdict_uses_frozen_commit_after_ledger_publication(tmp_path: Path, monkeypatch):
    plan, _jobs = _plan(tmp_path)
    context = _final_review(plan, monkeypatch)
    result = checker.verdict(plan, _published(plan, tmp_path), tmp_path / "verdict.json")
    assert result["attempt_status"] == "PASS"
    assert context["checked"] == [context["package"]]
