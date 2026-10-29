"""Exact-message regression twins plus retained real TRAIN-01 liveness."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from research.sandbox_a_resident import training_resume as r
from research.sandbox_a_resident.test_c2_readiness3 import refuses
from research.sandbox_a_resident.units import publish

FIXTURE = Path(__file__).parent / "fixtures/attempt15_liveness"
WARNING = (
    "/content/snake-cli/59e68cce673b46b1960ab9eb9287337b/.venv/lib/python3.12/"
    "site-packages/torch/_inductor/compile_fx.py:312: UserWarning: "
    "TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. "
    "Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
)


@pytest.fixture(scope="module")
def attempt15(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("real-attempt15")
    with tarfile.open(FIXTURE / "evidence.tar.xz") as archive:
        archive.extractall(root, filter="data")
    inventory = json.loads((FIXTURE / "inventory.json").read_text())
    assert set(inventory) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    for name, row in inventory.items():
        path = root / name
        assert path.stat().st_size == row["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    return root


def test_real_attempt15_log_and_every_poll_admit_end_to_end(attempt15: Path) -> None:
    output = attempt15 / "output"
    manifest = json.loads((output / "manifest.json").read_text())
    gen = json.loads((output / "poll-000000.json").read_text())["state"]["generation"]
    audit = r._check_supervisor_log_fault_evidence(
        attempt15 / "supervisor.log",
        expected_job_id=manifest["job_id"],
        expected_generation=gen,
        allow_terminal_receipt_enospc=True,
        expected_output_root=Path(manifest["output_dir"]),
    )
    assert audit["line_count"] == 1428
    assert audit["matched_job_records"] == 1355
    assert r._check_output_root_fault_evidence(
        output, expected_job_id=manifest["job_id"], expected_generation=gen
    ) == (1279, 1279)


@pytest.mark.parametrize("surface", ["text", "tail", "poll", "log"])
@pytest.mark.parametrize("fault", ["append", "replace", "prefix"])
def test_exact_torch_warning_fault_twins(tmp_path: Path, surface: str, fault: str) -> None:
    def action(text: str) -> None:
        if surface == "text":
            r._refuse_if_fault_text(text, field_name="warning")
        elif surface == "tail":
            r._scan_json_faults({"job_log_tail": text + "\n  warnings.warn(\n"}, label="warning")
        elif surface == "poll":
            path = tmp_path / "poll-000000.json"
            path.write_text(json.dumps({"job_log_tail": text + "\n  warnings.warn(\n"}))
            r._scan_retained_text(path, output_root=tmp_path)
        else:
            path = tmp_path / "supervisor.log"
            path.write_text(json.dumps({"job_log_tail": text + "\n  warnings.warn(\n"}) + "\n")
            r._check_supervisor_log_fault_evidence(path)

    action(WARNING)
    value = {
        "append": WARNING + " NovelDiagnosticException('x')",
        "replace": WARNING.replace("TensorFloat32", "NovelDiagnosticException('x')"),
        "prefix": "NovelDiagnosticException('x') " + WARNING,
    }[fault]
    label = {
        "text": "warning",
        "tail": "warning.job_log_tail",
        "poll": f"{tmp_path / 'poll-000000.json'}.job_log_tail",
        "log": "supervisor_log:1.job_log_tail",
    }[surface]
    refuses(
        lambda: action(value),
        f"prior evidence records explicit scientific/non-mechanical fault in {label}",
    )


def test_real_truncated_warning_first_tail_line() -> None:
    r._scan_json_faults({"job_log_tail": WARNING[1:] + "\n  warnings.warn(\n"}, label="tail")
    refuses(
        lambda: r._scan_json_faults(
            {"job_log_tail": WARNING[1:] + " NovelDiagnosticException('x')"}, label="tail"
        ),
        "prior evidence records explicit scientific/non-mechanical fault in tail.job_log_tail",
    )


@pytest.mark.parametrize(
    "fault", ["NovelDiagnosticException('x')", '"finite": false', "value_loss=nan"]
)
def test_first_tail_fragment_still_scans_faults(fault: str) -> None:
    fragment = ' "first_nonfinite_term": null, "stage": "learner-finite-check", "update": 14}'
    r._scan_json_faults({"job_log_tail": fragment}, label="poll")
    suffix = {
        "NovelDiagnosticException('x')": "",
        '"finite": false': ": '\"finite\": false'",
        "value_loss=nan": ": non-finite numeric text",
    }[fault]
    refuses(
        lambda: r._scan_json_faults({"job_log_tail": fragment + " " + fault}, label="poll"),
        "prior evidence records explicit scientific/non-mechanical fault in poll.job_log_tail"
        + suffix,
    )
    refuses(
        lambda: r._scan_json_faults({"job_log_tail": "healthy\n" + fragment}, label="poll"),
        "poll.job_log_tail records truncated execution JSON",
    )


@pytest.mark.parametrize("key", ["path", "dest", "sha256"])
@pytest.mark.parametrize("surface", ["structured", "json-log"])
def test_inventory_values_are_scanned_before_raw_mask(key: str, surface: str) -> None:
    def action(value: str) -> None:
        record = {key: value}
        if surface == "structured":
            r._scan_json_faults(record, label="record")
        else:
            r._scan_diagnostic_record_line(
                json.dumps(record), label="record", expected_job_id=None, expected_generation=None
            )

    action("healthy/data")
    refuses(
        lambda: action("ValueError: x"),
        f"prior evidence records explicit scientific/non-mechanical fault "
        f"in record.{key}: 'valueerror:'",
    )


@pytest.mark.parametrize("value", ["value_loss=nan", "loss: inf", "loss=-Infinity", "loss=+NaN"])
@pytest.mark.parametrize("surface", ["text", "tail", "json-log"])
def test_plain_nonfinite_numeric_twins(value: str, surface: str) -> None:
    def action(text: str) -> None:
        if surface == "text":
            r._refuse_if_fault_text(text, field_name="numeric")
        elif surface == "tail":
            r._scan_json_faults({"job_log_tail": text}, label="numeric")
        else:
            r._scan_diagnostic_record_line(
                json.dumps({"diagnostic": text}),
                label="numeric",
                expected_job_id=None,
                expected_generation=None,
            )

    action("value_loss=0.01 loss: 0.1")
    label = {"text": "numeric", "tail": "numeric.job_log_tail", "json-log": "numeric.diagnostic"}[
        surface
    ]
    refuses(
        lambda: action(value),
        f"prior evidence records explicit scientific/non-mechanical fault "
        f"in {label}: non-finite numeric text",
    )


def test_truncated_scientific_staging_prefix_twins(tmp_path: Path) -> None:
    training = tmp_path / "exports/training"
    training.mkdir(parents=True)
    content = '{"stage": "collect", "ticks": 32}'
    publish(training / "progress.json", lambda dst: dst.write_text(content))
    staged = training / "progress.json.partial"
    staged.write_text(content[:18])
    r._check_output_root_fault_evidence(tmp_path)
    staged.write_text('{"error": "NovelDiagnosticException(')
    refuses(
        lambda: r._check_output_root_fault_evidence(tmp_path),
        f"prior evidence records explicit scientific/non-mechanical fault in {staged}",
    )


def test_capture_failed_directory_never_masks_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.sandbox_a_resident import capture_supervisor_goldens as capture

    destination = tmp_path / "golden"
    failed = tmp_path / "golden.failed"
    failed.mkdir()
    (failed / "old").write_text("old")

    def explode(path: Path, scenario: str) -> None:
        path.mkdir()
        (path / "new").write_text("new")
        raise RuntimeError("original producer failure")

    monkeypatch.setattr(capture, "_capture_one", explode)
    with pytest.raises(RuntimeError, match="^original producer failure$"):
        capture.capture_one(destination, "wall-cap")
    assert (failed / "old").read_text() == "old"
    assert not (failed / "new").exists()


@pytest.mark.parametrize("surface", ["log", "poll"])
def test_real_attempt15_retained_fault_twin(tmp_path: Path, attempt15: Path, surface: str) -> None:
    import shutil

    if surface == "log":
        path = tmp_path / "supervisor.log"
        original = (attempt15 / "supervisor.log").read_text()
        path.write_text("NovelDiagnosticException('before retained evidence')\n" + original)
        manifest = json.loads((attempt15 / "output/manifest.json").read_text())
        refuses(
            lambda: r._check_supervisor_log_fault_evidence(
                path,
                expected_job_id=manifest["job_id"],
                allow_terminal_receipt_enospc=True,
                expected_output_root=Path(manifest["output_dir"]),
            ),
            "prior evidence records explicit scientific/non-mechanical fault in supervisor_log:1",
        )
    else:
        output = tmp_path / "output"
        shutil.copytree(attempt15 / "output", output)
        path = output / "poll-000000.json"
        record = json.loads(path.read_text())
        record["job_log_tail"] = "value_loss=nan"
        path.write_text(json.dumps(record))
        refuses(
            lambda: r._check_output_root_fault_evidence(output),
            f"prior evidence records explicit scientific/non-mechanical fault "
            f"in {path}.job_log_tail: non-finite numeric text",
        )


def test_warning_source_line_fault_twin() -> None:
    def action(source: str) -> None:
        r._scan_json_faults({"job_log_tail": WARNING + "\n" + source}, label="warning")

    action("  warnings.warn(")
    refuses(
        lambda: action("  warnings.warn( NovelDiagnosticException('x')"),
        "prior evidence records explicit scientific/non-mechanical fault in warning.job_log_tail",
    )


def test_exact_empty_watchdog_inventory_path_twins() -> None:
    path = "training/stall-watchdog/faulthandler-stacks.log"
    r._scan_json_faults({"exports": [{"path": path, "size": 0}]}, label="poll")
    refuses(
        lambda: r._scan_json_faults({"exports": [{"path": path, "size": 1}]}, label="poll"),
        "poll exported non-empty faulthandler-stacks.log (stall watchdog fired)",
    )
    refuses(
        lambda: r._scan_json_faults(
            {"exports": [{"path": path + " NovelDiagnosticException('x')", "size": 0}]},
            label="poll",
        ),
        "prior evidence records explicit scientific/non-mechanical fault "
        "in poll.exports[0].path: 'faulthandler'",
    )
