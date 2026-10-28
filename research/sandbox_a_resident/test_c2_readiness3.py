"""Exact-message healthy/fault twins for the readiness-3 review."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from research.sandbox_a_resident import capture_supervisor_goldens as capture
from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_supervisor_goldens import ELIGIBLE, GOLDENS, _evidence
from research.sandbox_a_resident.test_training_resume import manifest, receipt


def refuses(action: Callable[[], object], message: str) -> None:
    with pytest.raises(r.ResumeRefused) as caught:
        action()
    assert str(caught.value) == message


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
@pytest.mark.parametrize("surface", ["last", "tail", "poll", "header", "log", "supervisor"])
def test_nonfinite_json_twins(tmp_path: Path, token: str, surface: str) -> None:
    label = "terminal"
    path = tmp_path / {
        "poll": "poll-001.json",
        "header": "iteration-0016.json",
        "log": "job.log",
        "supervisor": "supervisor.log",
    }.get(surface, "unused")

    def check(value: str) -> None:
        if surface in {"last", "tail"}:
            key = "last_error_line" if surface == "last" else "job_log_tail"
            r._scan_json_faults({key: "TRAIN01_PROGRESS " + value}, label=label)
        else:
            path.write_text(value + "\n")
            r._scan_retained_text(path, output_root=tmp_path)

    check('{"metrics":{"value_loss":1.0}}')
    if token.endswith("999"):
        suffix = ".last_error_line" if surface == "last" else ".job_log_tail"
        base = (
            label + suffix
            if surface in {"last", "tail"}
            else ("supervisor_log:1" if surface == "supervisor" else str(path))
        )
        message = f"{base}.metrics.value_loss records non-finite float"
    else:
        message = "non-finite JSON constant"
    refuses(lambda: check('{"metrics":{"value_loss":' + token + "}}"), message)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_in_memory_deeply_nested_twins(value: float) -> None:
    r._scan_json_faults({"metadata": {"metrics": [1.0]}}, label="header")
    refuses(
        lambda: r._scan_json_faults({"metadata": {"metrics": [value]}}, label="header"),
        "header.metadata.metrics[0] records non-finite float",
    )


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.parametrize("name", ["job.log", "supervisor.log"])
def test_scalar_json_log_line_nonfinite_twins(tmp_path: Path, token: str, name: str) -> None:
    path = tmp_path / name
    path.write_text("1.0\n")
    r._scan_retained_text(path, output_root=tmp_path)
    path.write_text(token + "\n")
    label = "supervisor_log:1" if name == "supervisor.log" else str(path)
    refuses(
        lambda: r._scan_retained_text(path, output_root=tmp_path),
        f"{label} records non-finite float" if token == "1e999" else "non-finite JSON constant",
    )


@pytest.mark.parametrize("surface", ["json", "log", "prefixed-log", "supervisor", "tail"])
def test_fault_in_json_key_twins(tmp_path: Path, surface: str) -> None:
    path = tmp_path / ("supervisor.log" if surface == "supervisor" else "job.log")

    def check(key: str) -> None:
        value = {key: 1}
        if surface == "json":
            r._scan_json_faults(value, label="header")
        elif surface == "tail":
            r._scan_json_faults({"job_log_tail": json.dumps(value)}, label="header")
        else:
            path.write_text(
                ("2026-10-28 INFO " if surface == "prefixed-log" else "") + json.dumps(value) + "\n"
            )
            r._scan_retained_text(path, output_root=tmp_path)

    check("healthy metric")
    label = (
        "header.job_log_tail"
        if surface == "tail"
        else (
            "header"
            if surface == "json"
            else "supervisor_log:1" if surface == "supervisor" else str(path)
        )
    )
    refuses(
        lambda: check("FloatingPointError: nan"),
        f"prior evidence records explicit scientific/non-mechanical fault "
        f"in {label} JSON key: 'floatingpointerror'",
    )


@pytest.mark.parametrize("alias", ["returncode", "exit_code", "exitCode", "rc", "exit_status"])
@pytest.mark.parametrize("code", [1, 137, 124, "-15", True, None])
def test_receipt_exit_healthy_fault_twins(tmp_path: Path, alias: str, code: object) -> None:
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
    )

    def action() -> None:
        r.verify_mechanical_stop(rec, manifest(), "b" * 64, output_root=tmp_path)

    action()  # Native receipts have no root exit fact.
    rec[alias] = -15
    action()  # A same-type mirror carries no conflicting fact.
    rec[alias] = code
    message = (
        "receipt records malformed returncode"
        if code is not None and type(code) is not int
        else (
            "receipt records watchdog timeout exit code 124"
            if code == 124
            else "receipt returncode differs from retained terminal returncode"
        )
    )
    refuses(action, message)


@pytest.mark.parametrize("prefix", ["2026-10-28 INFO ", "[2026-10-28] INFO ", "TRAIN01_PROGRESS "])
def test_prefixed_json_facts_and_raw_prefix_twins(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "job.log"
    path.write_text(prefix + '{"status":"running"}\n')

    def action() -> None:
        r._scan_retained_text(path, output_root=tmp_path)

    action()
    path.write_text(prefix + '{"status":"failed","failure_class":"CHILD_NONZERO_EXIT"}\n')
    refuses(action, f"{path} records unknown execution status or failed status: 'failed'")
    path.write_text("exited with code 3 " + prefix + '{"status":"running"}\n')
    refuses(
        action,
        f"prior evidence records explicit scientific/non-mechanical fault "
        f"in {path}: nonzero exit signal",
    )


@pytest.mark.parametrize(
    "name",
    [
        "canary_observations",
        "research/colab_cli/canary.py",
        "research/colab_cli/canary_observer.py",
    ],
)
def test_exact_producer_key_names_never_exempt_faults(name: str) -> None:
    r._scan_json_faults({name: {}}, label="header")
    refuses(
        lambda: r._scan_json_faults({name + " FloatingPointError: nan": {}}, label="header"),
        "prior evidence records explicit scientific/non-mechanical fault "
        "in header JSON key: 'canary'",
    )
    refuses(
        lambda: r._scan_json_faults({name: {"finite": False}}, label="header"),
        f"header.{name}.finite records explicit scientific/non-mechanical fault",
    )


@pytest.mark.parametrize("surface", ["text", "tail", "supervisor", "json-key"])
@pytest.mark.parametrize(
    "fault,marker",
    [
        ("NovelDiagnosticException('x')", None),
        ("KeyboardInterrupt:", "keyboardinterrupt"),
        ("SystemExit: 3", "systemexit:"),
        ("GeneratorExit", None),
        ("RuntimeWarning: invalid value encountered in divide", None),
        ("DivergenceDetected: synthetic", None),
        ("Traceback (most recent call last):", "traceback (most recent call last)"),
        ("{'returncode': 1}", "nonzero exit signal"),
        ('{"exit_code": "137"}', "malformed returncode"),
        ("exited with code 3", "nonzero exit signal"),
        ("returned non-zero exit status 1", "nonzero exit signal"),
        ('2026-10-28 INFO {"status":"failed","failure_class":"CHILD_NONZERO_EXIT"}', "status"),
    ],
)
def test_novel_diagnostic_forms_twins(
    tmp_path: Path, surface: str, fault: str, marker: str | None
) -> None:
    path = tmp_path / ("supervisor.log" if surface == "supervisor" else "job.log")
    label = (
        "header.job_log_tail"
        if surface == "tail"
        else (
            "header JSON key"
            if surface == "json-key"
            else "supervisor_log:1" if surface == "supervisor" else str(path)
        )
    )

    def check(text: str) -> None:
        if surface == "tail":
            r._scan_json_faults({"job_log_tail": text}, label="header")
        elif surface == "json-key":
            r._scan_json_faults({text: 1}, label="header")
        else:
            path.write_text(text + "\n")
            r._scan_retained_text(path, output_root=tmp_path)

    check("[W1028] healthy diagnostic")
    message = f"prior evidence records explicit scientific/non-mechanical fault in {label}"
    if marker == "status" and surface != "json-key":
        message = f"{label} records unknown execution status or failed status: 'failed'"
    elif marker == "malformed returncode" and surface != "json-key":
        message = f"{label} records malformed returncode"
    elif marker == "nonzero exit signal" or marker == "malformed returncode":
        message += ": nonzero exit signal"
    elif marker and marker != "status":
        message += f": {marker!r}"
    # A prefixed JSON document used as a key is raw text, so its failure class
    # alone is not an exception. Exercise structured forms on diagnostic lines.
    if marker == "status" and surface == "json-key":
        return
    refuses(lambda: check(fault), message)


@pytest.mark.parametrize("suffix", ["", ".partial", ".part", ".tmp-a", ".tmp123"])
@pytest.mark.parametrize(
    "name,healthy,fault,message",
    [
        (
            "distribution-screen.json",
            {"material_gaps": []},
            {"material_gaps": ["synthetic"]},
            "distribution-screen.json records material distribution gaps",
        ),
        (
            "progress.json",
            {"stage": "update"},
            {"stage": "proxy-failed"},
            "progress.json records failed stage",
        ),
        (
            "progress.json",
            {"error": None},
            {"error": "synthetic"},
            "progress.json records explicit scientific/non-mechanical fault",
        ),
        (
            "stall-watchdog/stall-watchdog.json",
            {"stage": "watching"},
            {"stage": "proxy-failed"},
            "stall-watchdog.json records failed stage",
        ),
    ],
)
def test_staged_publication_exact_gate_twins(
    tmp_path: Path, suffix: str, name: str, healthy: dict, fault: dict, message: str
) -> None:
    path = tmp_path / "exports/training" / (name + suffix)
    path.parent.mkdir(parents=True)

    def action() -> None:
        r._check_output_root_fault_evidence(tmp_path)

    path.write_text(json.dumps(healthy))
    action()
    r._scan_retained_text(path, output_root=tmp_path)
    path.write_text(json.dumps(fault))
    refuses(action, message)
    refuses(lambda: r._scan_retained_text(path, output_root=tmp_path), message)
    path.write_text("[]")
    refuses(action, f"malformed scientific publication: {path}")
    refuses(
        lambda: r._scan_retained_text(path, output_root=tmp_path),
        f"malformed scientific publication: {path}",
    )


@pytest.mark.parametrize("suffix", [".part", ".tmp-a", ".partial", ".sha256.tmp-a"])
def test_summary_staged_presence_twins(tmp_path: Path, suffix: str) -> None:
    r._check_output_root_fault_evidence(tmp_path)
    path = tmp_path / "exports/training" / ("summary.json" + suffix)
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    refuses(
        lambda: r._check_output_root_fault_evidence(tmp_path),
        "exported summary.json already exists; prior run completed",
    )


@pytest.mark.parametrize("scenario", capture.SCENARIOS)
def test_existing_capture_never_spawns_or_changes_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    destination = tmp_path / "custom-golden-name"
    destination.mkdir()
    marker = destination / "existing.bin"
    marker.write_bytes(b"immutable golden\x00")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("must not spawn"))
    with pytest.raises(FileExistsError) as caught:
        capture.capture_one(destination, scenario)
    assert str(caught.value) == str(destination)
    assert marker.read_bytes() == b"immutable golden\x00"
    assert sorted(tmp_path.iterdir()) == [destination]


def test_capture_race_never_renames_other_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "custom-golden-name"

    def fake(private: Path, scenario: str) -> dict:
        private.mkdir()
        (private / "new").write_text("ours")
        destination.mkdir()
        (destination / "old").write_text("theirs")
        return {}

    monkeypatch.setattr(capture, "_capture_one", fake)
    with pytest.raises(FileExistsError) as caught:
        capture.capture_one(destination, "wall-cap")
    assert caught.value.filename == str(destination)
    assert (destination / "old").read_text() == "theirs"
    assert not (destination / "new").exists()


@pytest.mark.parametrize("case", ELIGIBLE)
def test_real_producer_liveness_and_exact_fault_twin(tmp_path: Path, case: str) -> None:
    output, rec, man, sha = _evidence(case)
    assert not any(r._record_key(k) == "returncode" for k in rec)
    # No source/archive reads are needed for this execution envelope proof;
    # the full retained corpus is exercised by test_supervisor_goldens.
    r.verify_mechanical_stop(rec, man, sha)
    rec["returncode"] = 124
    refuses(
        lambda: r.verify_mechanical_stop(rec, man, sha),
        "receipt records watchdog timeout exit code 124",
    )


@pytest.mark.parametrize("case", ELIGIBLE)
@pytest.mark.parametrize(
    "fault", ["NovelDiagnosticException('x')", "NovelDiagnosticException: x", "{'returncode': 137}"]
)
def test_producer_receipt_error_fault_precedence_twins(case: str, fault: str) -> None:
    _, rec, man, sha = _evidence(case)
    r.verify_mechanical_stop(rec, man, sha)
    wrapper, payload = rec["error"].split("(", 1)
    body = ast.literal_eval(payload[:-1])
    rec["error"] = f"{wrapper}({body + '; ' + fault!r})"
    message = "prior evidence records explicit scientific/non-mechanical fault in receipt.error"
    if fault.startswith("{"):
        message += ": nonzero exit signal"
    refuses(lambda: r.verify_mechanical_stop(rec, man, sha), message)


@pytest.mark.parametrize("case", ["runtime-timeout", "runtime-lease-expiry"])
def test_faithful_self_release_is_explicitly_inadmissible(case: str) -> None:
    root = GOLDENS.parent / "supervisor_goldens_v2" / case / "output"
    man = json.loads((root / "manifest.json").read_text())
    rec = json.loads((root / "receipt.json").read_text())
    assert rec["terminal"] is None
    assert rec["error"] == "AccountingError('supervised assignment disappeared from sessions')"
    from research.sandbox_a_resident.units import digest

    refuses(
        lambda: r.verify_mechanical_stop(rec, man, digest(root / "manifest.json")),
        "ACCOUNTING receipt records assignment disappearance without a retained terminal; "
        "not admissible",
    )


def test_custom_leaf_host_loss_success_and_fault_twin(tmp_path: Path) -> None:
    destination = tmp_path / "custom-golden-name"
    result = capture.capture_one(destination, "provider-loss-no-receipt")
    assert result["receipt_sha256"] is None
    assert (destination / "output/manifest.json").is_file()
    assert not (tmp_path / "provider-loss-no-receipt").exists()
    before = {
        str(p.relative_to(destination)): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError) as caught:
        capture.capture_one(destination, "provider-loss-no-receipt")
    assert str(caught.value) == str(destination)
    assert before == {
        str(p.relative_to(destination)): p.read_bytes()
        for p in destination.rglob("*")
        if p.is_file()
    }
    shutil.rmtree(destination)


def test_native_validator_remains_active_except_for_root(tmp_path: Path) -> None:
    from research.colab_cli import runtime

    validate = capture._offline_spec_validator(runtime, tmp_path)
    spec = {
        "job_id": "offline",
        "generation": "a" * 32,
        "root": str(tmp_path),
        "command": ["python"],
        "env": {},
        "timeout_seconds": 3,
        "lease_seconds": 180,
        "checkpoint_grace_seconds": 1,
        "max_export_bytes": 1024,
        "release_endpoint": "offline-fake-endpoint",
    }
    assert validate(spec) == spec
    with pytest.raises(runtime.SpecError) as caught:
        validate({**spec, "env": {"BAD": 1}})
    assert str(caught.value) == "env must map strings to strings"


@pytest.mark.parametrize("name", ["job.log", "other.json", "exports/training/progress.json.tmp-a"])
def test_diagnostic_size_bound_twins(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    r._scan_retained_text(path, output_root=tmp_path)
    with path.open("wb") as stream:
        stream.truncate(r.MAX_RETAINED_JSON_BYTES + 1)
    refuses(
        lambda: r._scan_retained_text(path, output_root=tmp_path),
        f"oversized retained diagnostic: {path}",
    )


@pytest.mark.parametrize("generation", ["B" * 32, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"])
def test_foreign_data_generation_formats_twins(generation: str) -> None:
    r._scan_json_faults(
        {"metadata": {"generation": 3}}, label="header", expected_generation="a" * 32
    )
    refuses(
        lambda: r._scan_json_faults(
            {"metadata": {"generation": generation}}, label="header", expected_generation="a" * 32
        ),
        "header.metadata belongs to foreign generation or malformed identity",
    )


def test_nested_supervisor_name_grants_no_transport_exemption(tmp_path: Path) -> None:
    path = tmp_path / "exports/training/supervisor.log"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"running"}\n')
    r._scan_retained_text(path, output_root=tmp_path)
    path.write_text(json.dumps(receipt()) + "\n")
    refuses(
        lambda: r._scan_retained_text(path, output_root=tmp_path, validated_receipt=receipt()),
        f"{path} records unknown execution status or failed status: 'incomplete'",
    )


def test_truncated_list_never_masquerades_as_bracketed_warning(tmp_path: Path) -> None:
    path = tmp_path / "job.log"
    path.write_text("[W1028] healthy diagnostic\n")

    def action() -> None:
        r._scan_retained_text(path, output_root=tmp_path)

    action()
    path.write_text('[{"status":"failed"}] x\n')
    refuses(action, f"{path} records truncated execution JSON")


def test_diagnostic_marker_never_exempts_exception_in_path() -> None:
    r._scan_json_faults(
        {"job_log_tail": "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/tmp/sealed"}, label="header"
    )

    refuses(
        lambda: r._scan_json_faults(
            {
                "job_log_tail": (
                    "STALL_WATCHDOG_DIAGNOSTICS sealed_dir=/tmp/" "NovelDiagnosticException('x')"
                )
            },
            label="header",
        ),
        "prior evidence records explicit scientific/non-mechanical fault in header.job_log_tail",
    )


@pytest.mark.parametrize("twin", ["append", "replace", "prefix"])
def test_exact_historical_pytest_banner_twins(tmp_path: Path, twin: str) -> None:
    path = tmp_path / "supervisor.log"
    banner = "!!!!!!!!! _pytest.outcomes.Exit: macOS OpenMP-isolated pytest finished !!!!!!!!!"
    path.write_text(banner + "\n")
    r._scan_retained_text(path, output_root=tmp_path)
    fault = {
        "append": banner + " NovelDiagnosticException('x')",
        "replace": banner.replace("finished", "NovelDiagnosticException('x')"),
        "prefix": "NovelDiagnosticException('x') " + banner,
    }[twin]
    path.write_text(fault + "\n")
    refuses(
        lambda: r._scan_retained_text(path, output_root=tmp_path),
        "prior evidence records explicit scientific/non-mechanical fault in supervisor_log:1",
    )
