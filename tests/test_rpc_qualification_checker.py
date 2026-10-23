"""Focused outcome-blind checks for the C1 qualification one-look checker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.bl_labels.exports import publish_json
from research.rpc_qualification import checker


@pytest.fixture(autouse=True)
def _remote_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unit test offline; production fetches and reads origin itself."""
    monkeypatch.setattr(
        checker,
        "_remote_json",
        lambda plan, _relative: json.loads((plan.parent / "remote.json").read_text()),
    )


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
            }
        ],
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
    assert missing_result["platforms"]["cpu"]["status"] == "FAIL"

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
    assert "duplicated execution" in result["injection"]["reason"]
