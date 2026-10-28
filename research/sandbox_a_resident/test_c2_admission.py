"""Positive admission paths and their independently injected negative twins."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.colab_cli.transfers import DOWNLOAD_CHUNK_BYTES
from research.sandbox_a_resident.test_training_resume import manifest, receipt
from research.sandbox_a_resident.training_resume import (
    ResumeRefused,
    _check_output_root_fault_evidence,
    _scan_retained_text,
    verify_mechanical_stop,
)
from research.sandbox_a_resident.units import digest, publish


@pytest.mark.parametrize("code", [1, 137, 124])
@pytest.mark.parametrize("alias", ["exit_code", "returncode", "exitCode", "returnCode"])
def test_mechanical_receipt_exit_is_explained_by_its_terminal(
    tmp_path: Path, code: int, alias: str
) -> None:
    rec = receipt()
    rec.update(
        failure_class="CHILD_NONZERO_EXIT",
        terminal={
            "job_id": "train-01",
            "generation": "g-1",
            "status": "failed",
            "reason": "timeout",
            "returncode": -15,
        },
        **{alias: code},
    )
    verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)
    rec["terminal"]["error"] = "FloatingPointError injected twin"
    with pytest.raises(ResumeRefused, match="scientific/non-mechanical fault"):
        verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)


@pytest.mark.parametrize(
    "prefix", ["[W1028 07:45:17.1 ProcessGroupNCCL.cpp:1]", "[2026-10-28] INFO", '{"progress":']
)
def test_incomplete_or_bracketed_log_line_uses_raw_fault_scan(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "job.log"
    path.write_text(prefix + " healthy\n")
    _scan_retained_text(path, output_root=tmp_path)
    path.write_text(prefix + " FloatingPointError injected\n")
    with pytest.raises(ResumeRefused, match="scientific/non-mechanical fault"):
        _scan_retained_text(path, output_root=tmp_path)


def test_supervisor_echo_must_equal_validated_receipt(tmp_path: Path) -> None:
    rec = receipt()
    path = tmp_path / "supervisor.log"
    path.write_text("progress\n" + json.dumps(rec) + "\n")
    verify_mechanical_stop(
        rec, manifest(), "b" * 64, output_root=tmp_path, supervisor_log_path=path
    )
    path.write_text(json.dumps({**rec, "elapsed_seconds": 99}) + "\n")
    with pytest.raises(ResumeRefused, match="unknown execution status|non-receipt failure_class"):
        verify_mechanical_stop(
            rec, manifest(), "b" * 64, output_root=tmp_path, supervisor_log_path=path
        )


@pytest.mark.parametrize(
    "name", ["poll-001.json", "state.json.partial", "terminal.json.tmp-a", "receipt.json"]
)
def test_execution_json_is_always_single_document(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    record = {"job_id": "train-01", "generation": "g-1", "status": "running"}
    path.write_text(json.dumps(record))
    _scan_retained_text(
        path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    path.write_text(json.dumps(record) + "\n" + json.dumps(record))
    with pytest.raises(ResumeRefused, match="corrupt poll|partial retained JSON"):
        _scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


@pytest.mark.parametrize("name", ["poll-1.json", "proxy.jsonl", "job.log", "supervisor.log"])
@pytest.mark.parametrize(
    "keys",
    [("job_id", "jobId"), ("job_id", "Job ID"), ("job_id", "job_id"), ("returnCode", "exit_code")],
)
def test_duplicate_normalized_json_keys_refuse_even_when_equal(
    tmp_path: Path, name: str, keys: tuple[str, str]
) -> None:
    path = tmp_path / name
    path.write_text('{"%s":"train-01"}' % keys[0])
    if keys[0] == "returnCode":
        path.write_text('{"%s":0}' % keys[0])
    _scan_retained_text(path, output_root=tmp_path, expected_job_id="train-01")
    value = "0" if keys[0] == "returnCode" else '"train-01"'
    path.write_text('{"%s":%s,"%s":%s}' % (keys[0], value, keys[1], value))
    with pytest.raises(ResumeRefused, match="duplicate normalized JSON key"):
        _scan_retained_text(path, output_root=tmp_path, expected_job_id="train-01")


@pytest.mark.parametrize("name", [".transfer-abcd.1", ".transfer-stage/00000001.part"])
def test_download_chunks_are_slices_at_their_real_offset(tmp_path: Path, name: str) -> None:
    root = tmp_path / "exports/training"
    root.mkdir(parents=True)
    canonical = root / "iteration-0128.pt.xz"
    content = b"a" * DOWNLOAD_CHUNK_BYTES + b"different second chunk"
    publish(canonical, lambda p: p.write_bytes(content))
    chunk = root / name
    chunk.parent.mkdir(exist_ok=True)
    chunk.write_bytes(content[DOWNLOAD_CHUNK_BYTES:])
    _check_output_root_fault_evidence(tmp_path)
    chunk.write_bytes(b"wrong second chunk")
    with pytest.raises(ResumeRefused, match="differs from verified payload"):
        _check_output_root_fault_evidence(tmp_path)


def test_empty_staging_and_verified_json_prefix_are_scanned_safely(tmp_path: Path) -> None:
    root = tmp_path / "exports/training"
    root.mkdir(parents=True)
    (root / "iteration-0128.pt.xz.partial").touch()
    canonical = root / "proxy-0128.json"
    publish(canonical, lambda p: p.write_text('{"status":"PASS","rows":12}\n'))
    part = root / ".transfer-json"
    part.write_bytes(canonical.read_bytes()[:7])
    _check_output_root_fault_evidence(tmp_path)
    # Independent canonical scan is mandatory even if staged bytes match it.
    publish(root / "proxy-0129.json", lambda p: p.write_text('{"error":"FloatingPointError"}'))
    with pytest.raises(ResumeRefused, match="scientific/non-mechanical fault"):
        _check_output_root_fault_evidence(tmp_path)


def test_orphan_full_digest_requires_matching_staged_payload(tmp_path: Path) -> None:
    payload = tmp_path / "poll-001.json.partial"
    payload.write_text('{"status":"running"}')
    sha = tmp_path / "poll-001.json.sha256"
    sha.write_text(digest(payload) + "\n")
    _check_output_root_fault_evidence(tmp_path)
    payload.unlink()
    with pytest.raises(ResumeRefused, match="orphan.*digest|digest.*payload"):
        _check_output_root_fault_evidence(tmp_path)


def test_deep_json_log_refuses_cleanly(tmp_path: Path) -> None:
    import sys

    path = tmp_path / "job.log"
    depth = max(sys.getrecursionlimit() * 10 + 100, 10000)
    path.write_text("progress\n" + "[" * depth + "0" + "]" * depth)
    with pytest.raises(ResumeRefused, match="deep|nested|recursion"):
        _scan_retained_text(path, output_root=tmp_path)


def test_mutable_progress_descriptors_do_not_conflict_with_model_proof(tmp_path: Path) -> None:
    import hashlib

    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    payload = b"opaque checkpoint"
    (training / ".transfer-state").write_bytes(payload)
    model = {
        "path": "training/iteration-0128.pt.xz",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    for i, progress in enumerate((b'{"rows":1}', b'{"rows":2}')):
        (tmp_path / f"poll-{i}.json").write_text(
            json.dumps(
                {
                    "state": {"job_id": "train-01", "generation": "g-1", "status": "running"},
                    "exports": [
                        model,
                        {
                            "path": "training/progress.json",
                            "size": len(progress),
                            "sha256": hashlib.sha256(progress).hexdigest(),
                        },
                    ],
                }
            )
        )
    _check_output_root_fault_evidence(
        tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    record = json.loads((tmp_path / "poll-1.json").read_text())
    record["exports"][0]["sha256"] = "0" * 64
    (tmp_path / "poll-1.json").write_text(json.dumps(record))
    with pytest.raises(ResumeRefused, match="conflicting retained model export bindings"):
        _check_output_root_fault_evidence(
            tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )


def test_unbound_truncated_download_refuses_with_recovery_policy(tmp_path: Path) -> None:
    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    path = training / "iteration-0128.pt.xz.partial"
    path.write_bytes(b"unprovable fragment")
    with pytest.raises(ResumeRefused, match="unverifiable retained model transport partial"):
        _check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("defect", ["path", "error", "middle-exit", "fault"])
def test_native_incomplete_wrapper_footer_needs_exact_receipt_binding(
    tmp_path: Path, defect: str
) -> None:
    rec = {**receipt(), "error": "Refused('deadline reached during telemetry recovery')"}
    path = tmp_path / "broker.log"
    footer = f"INCOMPLETE: {rec['error']}; receipt: /tmp/train-01/receipt.json\nEXIT=1\n"
    path.write_text("healthy progress\n" + footer)
    verify_mechanical_stop(rec, manifest(), "b" * 64, supervisor_log_path=path)
    changed = footer
    if defect == "path":
        changed = footer.replace("/tmp/train-01", "/foreign")
    elif defect == "error":
        changed = footer.replace("telemetry recovery", "unexplained failure")
    elif defect == "middle-exit":
        changed = "EXIT=7\n" + footer
    else:
        changed = "FloatingPointError injected twin\n" + footer
    path.write_text(changed)
    with pytest.raises(ResumeRefused, match="scientific/non-mechanical fault"):
        verify_mechanical_stop(rec, manifest(), "b" * 64, supervisor_log_path=path)


def test_scientific_failed_status_cannot_borrow_mechanical_reason(tmp_path: Path) -> None:
    path = tmp_path / "proxy.json"
    path.write_text('{"metrics":{"status":"measured","reason":"timeout"}}')
    _scan_retained_text(path, output_root=tmp_path)
    path.write_text('{"metrics":{"status":"failed","reason":"timeout"}}')
    with pytest.raises(ResumeRefused, match="failed status"):
        _scan_retained_text(path, output_root=tmp_path)


@pytest.mark.parametrize(
    "field", ["transfers", "contents_transfers", "cli_calls", "control_operations"]
)
def test_only_exact_receipt_scopes_explain_failed_transport_records(
    tmp_path: Path, field: str
) -> None:
    rec = {
        **receipt(),
        field: [
            {
                "status": "failed",
                "operation": "renewal-heartbeat",
                "direction": "upload",
                "remote": "/content/lease-heartbeat.json",
            }
        ],
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(rec))
    verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)
    record = json.loads(path.read_text())
    record[field][0]["finite"] = False
    path.write_text(json.dumps(record))
    # Match against the newly bound receipt too: transport scope never excuses
    # independently explicit scientific failures in that very same record.
    with pytest.raises(ResumeRefused, match="scientific/non-mechanical fault"):
        verify_mechanical_stop(record, manifest(), "b" * 64, output_root=tmp_path)
