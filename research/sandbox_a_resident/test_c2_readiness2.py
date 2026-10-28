"""Exact refusal predicates and twins for the c2 review corrections."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_supervisor_goldens import ELIGIBLE, _evidence
from research.sandbox_a_resident.test_training_resume import manifest, receipt
from research.sandbox_a_resident.units import publish


@pytest.fixture(scope="module", autouse=True)
def private_original_capture_objects(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    repo = Path(__file__).resolve().parents[2]
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=repo, text=True
    ).strip()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path_factory.mktemp("original-capture-objects")))
        mp.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(Path(common) / "objects"))
        for case in ELIGIBLE:
            output, _, man, _ = _evidence(case)
            raw = (output.parent / "pinned-commit.raw").read_bytes()
            restored = (
                subprocess.check_output(
                    ["git", "hash-object", "-w", "-t", "commit", "--stdin"],
                    input=raw,
                    cwd=repo,
                )
                .decode()
                .strip()
            )
            assert restored == man["commit"]
            subprocess.run(["git", "cat-file", "-e", restored + "^{tree}"], cwd=repo, check=True)
        yield


@pytest.mark.parametrize(
    "name,good,bad,message",
    [
        (
            "distribution-screen.json",
            {"material_gaps": []},
            {"material_gaps": [{"feature": "synthetic", "gap": 0.41}]},
            "distribution-screen.json records material distribution gaps",
        ),
        (
            "progress.json",
            {"stage": "update"},
            {"stage": "proxy-failed"},
            "progress.json records failed stage",
        ),
        (
            "stall-watchdog/stall-watchdog.json",
            {"stage": "watching"},
            {"stage": "proxy-failed"},
            "stall-watchdog.json records failed stage",
        ),
        (
            "progress.json",
            {"error": None},
            {"error": "NovelDiagnosticException: synthetic"},
            "progress.json records explicit scientific/non-mechanical fault",
        ),
    ],
)
def test_publication_gate_twins(
    tmp_path: Path, name: str, good: dict, bad: dict, message: str
) -> None:
    path = tmp_path / "exports/training" / name
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(good))
    r._check_output_root_fault_evidence(tmp_path)
    path.write_text(json.dumps(bad))
    with pytest.raises(r.ResumeRefused, match=message):
        r._check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "distribution-screen.json",
        "progress.json",
        "stall-watchdog/stall-watchdog.json",
        "iteration-0016.json",
        "calibrated-boundary.json",
    ],
)
def test_scientific_publications_must_be_mappings(tmp_path: Path, name: str) -> None:
    path = tmp_path / "exports/training" / name
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    with pytest.raises(r.ResumeRefused, match="malformed.*publication|malformed snapshot header"):
        r._check_output_root_fault_evidence(tmp_path)


@pytest.mark.parametrize("case", ELIGIBLE)
@pytest.mark.parametrize(
    "surface",
    [
        "last_traceback_line",
        "last_error_line",
        "job_log_tail",
        "transport_summary",
        "poll",
        "cli_calls",
        "control_operations",
        "transfers",
        "contents_transfers",
        "calibration",
    ],
)
def test_every_exempt_surface_rejects_non_denylisted_exception(case: str, surface: str) -> None:
    _, original, _, _ = _evidence(case)
    rec = copy.deepcopy(original)
    r._scan_json_faults(
        rec,
        label="receipt",
        expected_job_id=rec["job_id"],
        expected_generation=rec["generation"],
        supervisor_transport=True,
        validated_receipt_root=True,
    )
    fault = "NovelDiagnosticException: synthetic sentinel"
    assert not any(marker in fault.lower() for marker in r.EXPLICIT_FAULT_EVIDENCE_MARKERS)
    if surface in {"last_traceback_line", "last_error_line", "job_log_tail"}:
        # Exercise native terminal fields and terminal echoes, including cases
        # whose original receipt had no runtime terminal.
        if rec.get("terminal") is None:
            rec["transport_summary"] = {
                "job_id": rec["job_id"],
                "generation": rec["generation"],
                surface: fault,
            }
        else:
            rec["terminal"][surface] = fault
        # preserve original ACCOUNTING/None terminal envelope
    elif surface in {"cli_calls", "control_operations", "transfers", "contents_transfers"}:
        rec.setdefault(surface, []).append(
            {
                "status": "failed",
                "operation": "renewal-heartbeat",
                "direction": "upload",
                "remote": "/content/lease-heartbeat.json",
                "diagnostic": fault,
            }
        )
    else:
        rec[surface] = {"diagnostic": fault}
    with pytest.raises(r.ResumeRefused, match="explicit scientific/non-mechanical fault"):
        r._scan_json_faults(
            rec,
            label="receipt",
            expected_job_id=rec["job_id"],
            expected_generation=rec["generation"],
            supervisor_transport=True,
            validated_receipt_root=True,
        )


@pytest.mark.parametrize("field", ["last_traceback_line", "last_error_line"])
def test_progress_last_line_is_parsed_and_checked(field: str) -> None:
    rec = {field: 'TRAIN01_PROGRESS {"stage":"update"}'}
    r._scan_json_faults(rec, label="terminal")
    rec[field] = 'TRAIN01_PROGRESS {"error":"synthetic scientific failure"}'
    with pytest.raises(r.ResumeRefused, match="explicit scientific/non-mechanical fault"):
        r._scan_json_faults(rec, label="terminal")


@pytest.mark.parametrize("scope", sorted(r.SCIENTIFIC_DATA_SCOPES))
@pytest.mark.parametrize(
    "status",
    ["error", "fail", "nan", "non_finite", "unstable", "rejected", "skipped", "unknown-new-status"],
)
def test_data_status_allowlist_twins(scope: str, status: str) -> None:
    r._scan_json_faults({scope: {"status": "pass"}}, label="header")
    with pytest.raises(r.ResumeRefused, match="unknown execution status or failed status"):
        r._scan_json_faults({scope: {"status": status}}, label="header")


@pytest.mark.parametrize("scope", sorted(r.SCIENTIFIC_DATA_SCOPES))
def test_data_reason_and_generation_twins(scope: str) -> None:
    rec = {scope: {"status": "pass", "generation": "a" * 32}}
    r._scan_json_faults(rec, label="header", expected_generation="a" * 32)
    rec[scope]["generation"] = "b" * 32
    with pytest.raises(r.ResumeRefused, match="foreign generation"):
        r._scan_json_faults(rec, label="header", expected_generation="a" * 32)
    rec[scope] = {"status": "pass", "reason": "nan_reward"}
    with pytest.raises(r.ResumeRefused, match="non-allowlisted scientific data reason"):
        r._scan_json_faults(rec, label="header")


def test_receipt_reason_never_masked_by_terminal() -> None:
    rec = receipt()
    rec["terminal"] = {
        "job_id": "train-01",
        "generation": "g-1",
        "status": "failed",
        "reason": "timeout",
        "returncode": -15,
    }
    r.verify_mechanical_stop(rec, manifest(), "b" * 64)
    rec["reason"] = "unexplained-scientific-stop"
    with pytest.raises(r.ResumeRefused, match="non-mechanical reason"):
        r.verify_mechanical_stop(rec, manifest(), "b" * 64)


@pytest.mark.parametrize("code", [137, -9, -15])
def test_child_kill_requires_mechanical_reason(code: int) -> None:
    rec = receipt()
    rec["terminal"] = {
        "job_id": "train-01",
        "generation": "g-1",
        "status": "failed",
        "reason": "timeout",
        "returncode": code,
    }
    r.verify_mechanical_stop(rec, manifest(), "b" * 64)
    rec["terminal"]["reason"] = None
    with pytest.raises(r.ResumeRefused, match="non-mechanical child failure"):
        r.verify_mechanical_stop(rec, manifest(), "b" * 64)


@pytest.mark.parametrize(
    "field",
    [
        "job_id",
        "jobId",
        "generation",
        "failureClass",
        "status",
        "reason",
        "exitCode",
        "return_code",
    ],
)
@pytest.mark.parametrize("name", ["job.log", "supervisor.log"])
def test_truncated_execution_json_cannot_bypass_structure(
    tmp_path: Path, field: str, name: str
) -> None:
    path = tmp_path / name
    path.write_text("[2026-10-28] healthy warning\n")
    r._scan_retained_text(path, output_root=tmp_path)
    path.write_text('{"%s": "foreign", "unfinished' % field)
    with pytest.raises(r.ResumeRefused, match="truncated execution JSON"):
        r._scan_retained_text(path, output_root=tmp_path)


@pytest.mark.parametrize("key", ["exitCode", "returnCode", "return_code", "exit_code"])
def test_quoted_text_exit_aliases(tmp_path: Path, key: str) -> None:
    path = tmp_path / "job.log"
    path.write_text('prefix "%s": 3 trailing' % key)
    with pytest.raises(r.ResumeRefused, match="nonzero exit signal"):
        r._scan_retained_text(path, output_root=tmp_path)


def test_metric_keys_and_terminal_counts_are_data() -> None:
    obj = r._json_loads(
        '{"metrics":{"A":1,"a":2,"valueLoss":3,"value_loss":4,"terminal":7,"state":2}}'
    )
    r._scan_json_faults(obj, label="metrics")
    with pytest.raises(r.ResumeRefused, match="duplicate normalized JSON key"):
        r._json_loads('{"metrics":{"A":1,"A":2}}')


def test_complete_mutable_rewrite_and_empty_model_are_not_units(tmp_path: Path) -> None:
    root = tmp_path / "exports/training"
    root.mkdir(parents=True)
    publish(
        root / "progress.json", lambda path: path.write_text('{"stage":"update","iteration":1}')
    )
    path = root / "progress.json.partial"
    path.write_text('{"stage":"update","iteration":2}')
    r._check_output_root_fault_evidence(tmp_path)
    path.write_text('{"stage":"proxy-failed"}')
    with pytest.raises(r.ResumeRefused, match="records failed stage"):
        r._check_output_root_fault_evidence(tmp_path)
    empty = root / "iteration-0000.pt.partial"
    empty.touch()
    assert r._RetainedTransport(tmp_path, None, None).admit_payload(empty, None) is None


def test_missing_scan_root_refuses(tmp_path: Path) -> None:
    with pytest.raises(r.ResumeRefused, match="missing retained scan root"):
        r._check_output_root_fault_evidence(tmp_path / "missing")


@pytest.mark.parametrize("case", ELIGIBLE)
def test_missing_stop_verified_never_admits(case: str) -> None:
    _, rec, man, sha = _evidence(case)
    r.verify_mechanical_stop(rec, man, sha)
    del rec["stop_verified"]
    with pytest.raises(r.ResumeRefused, match="verified released incomplete run"):
        r.verify_mechanical_stop(rec, man, sha)


def test_unreadable_directory_is_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args, **kwargs) -> None:
        args_error = PermissionError(13, "synthetic denied scan", str(tmp_path / "nested"))
        kwargs["onerror"](args_error)
        return iter(())

    monkeypatch.setattr(r.os, "walk", denied)
    with pytest.raises(r.ResumeRefused, match="unreadable retained scan directory"):
        r._check_output_root_fault_evidence(tmp_path)


def test_native_compile_reason_is_exact() -> None:
    r._scan_json_faults(
        {"compile": {"status": "disabled_eager", "reason": "requested mode=none"}}, label="header"
    )
    with pytest.raises(r.ResumeRefused, match="non-allowlisted scientific data reason"):
        r._scan_json_faults(
            {
                "compile": {
                    "status": "disabled_eager",
                    "reason": "NovelDiagnosticException: injected",
                }
            },
            label="header",
        )


@pytest.mark.parametrize("case", ELIGIBLE)
@pytest.mark.parametrize(
    "surface", ["job_log_tail", "last_error_line", "last_traceback_line", "transport_summary"]
)
def test_real_poll_file_fault_twins(tmp_path: Path, case: str, surface: str) -> None:
    import shutil

    original, rec, man, sha = _evidence(case)
    output = tmp_path / "output"
    shutil.copytree(original, output)
    # Prove the original complete producer tree admits before adding this poll.
    r.verify_mechanical_stop(rec, man, sha, output_root=output)
    poll = output / "poll-999999.json"
    record = {"job_id": rec["job_id"], "generation": rec["generation"], "status": "running"}
    healthy = 'TRAIN01_PROGRESS {"stage":"update"}'
    record[surface] = (
        {"diagnostic": "healthy progress"} if surface == "transport_summary" else healthy
    )
    poll.write_text(json.dumps(record))
    r.verify_mechanical_stop(rec, man, sha, output_root=output)
    fault = "NovelDiagnosticException: injected synthetic poll fault"
    record[surface] = {"diagnostic": fault} if surface == "transport_summary" else fault
    poll.write_text(json.dumps(record))
    with pytest.raises(r.ResumeRefused, match="explicit scientific/non-mechanical fault"):
        r.verify_mechanical_stop(rec, man, sha, output_root=output)


@pytest.mark.parametrize("name", ["job.log", "supervisor.log"])
@pytest.mark.parametrize(
    "key",
    ["error", "exception", "last_error_line", "material_gaps", "stage", "Job ID", "Exit Code"],
)
def test_truncated_scientific_fields_never_lose_structural_checks(
    tmp_path: Path, name: str, key: str
) -> None:
    path = tmp_path / name
    path.write_text('{"%s":"synthetic", "unfinished' % key)
    with pytest.raises(r.ResumeRefused, match="truncated execution JSON"):
        r._scan_retained_text(path, output_root=tmp_path)


@pytest.mark.parametrize("name", ["job.log", "supervisor.log"])
def test_complete_json_mechanical_terminal_uses_structured_exit_facts(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / name
    record = {
        "job_id": "train-01",
        "generation": "g-1",
        "status": "failed",
        "reason": "timeout",
        "returncode": -15,
    }
    path.write_text(json.dumps(record))
    r._scan_retained_text(
        path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
    )
    record["reason"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(r.ResumeRefused, match="failed status|non-mechanical child"):
        r._scan_retained_text(
            path, output_root=tmp_path, expected_job_id="train-01", expected_generation="g-1"
        )
