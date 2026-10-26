"""Outcome-blind mismatch exports keep the frozen admission decision intact."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from research.bl_ppo.exports import verify_export
from research.bl_residual import f5_environment as environment
from research.bl_residual import f5_job, f5_qualification
from research.colab_cli.runtime import _exports
from tests.test_bl_residual_f5_environment import _receipt


def _mismatch() -> tuple[dict, dict]:
    expected = _receipt()
    actual = _receipt(uuid="B", node="host-b", root="/venv-b")
    actual["cpu_isa"]["model name"] = "different host CPU"
    actual["gpu_driver_query"]["driver_version"] = "999"
    actual["gpu_driver_query_raw"] = (
        ", ".join(
            actual["gpu_driver_query"][key]
            for key in ("name", "uuid", "driver_version", "compute_cap", "memory_total")
        )
        + "\n"
    )
    return expected, actual


def test_mismatch_lists_all_canonical_keys_and_preserves_raw_observation() -> None:
    expected, actual = _mismatch()
    before = copy.deepcopy(actual)
    with pytest.raises(environment.F5CrossAssignmentRefusal) as raised:
        environment.compare_environment(expected, actual)
    diagnostic = raised.value.diagnostic
    assert diagnostic["differing_canonical_keys"] == [
        "/cpu_isa/model name",
        "/gpu_driver_query/driver_version",
        "/gpu_driver_query_raw",
    ]
    assert diagnostic["actual_raw_sha256"] == environment.receipt_sha256(actual)
    assert diagnostic["expected_raw_sha256"] == environment.receipt_sha256(expected)
    assert diagnostic["actual_canonical_sha256"] == environment.receipt_sha256(
        environment.canonical_environment(actual)
    )
    assert actual == before
    assert diagnostic["admissible"] is False
    assert diagnostic["scientific_use_authorized"] is False
    assert diagnostic["fit_usable"] is False
    # Only refusal status/metadata changes; no observed field is repaired.
    raw = diagnostic["raw_receipt"]
    assert {k: v for k, v in raw.items() if k != "_admission_failures"} == {
        **actual,
        "capture_status": "REFUSED",
    }
    for candidate in (diagnostic, raw):
        with pytest.raises(environment.F5EnvironmentRefusal):
            environment.validate_environment(candidate)
        with pytest.raises(environment.F5EnvironmentRefusal):
            environment.compare_environment(expected, candidate)


def test_differing_keys_escape_paths_and_report_missing_packages_and_exact_lists() -> None:
    assert environment._differing_canonical_keys(
        {"a/b~c": {"leaf": "old"}, "packages": {"extra": {}}, "list": [1, 2]},
        {"a/b~c": {"leaf": "new"}, "packages": {}, "list": [1, 3]},
    ) == ["/a~1b~0c/leaf", "/list", "/packages/extra"]


def test_equal_assignment_does_not_export_diagnostics(tmp_path: Path) -> None:
    expected = _receipt()
    actual = _receipt(uuid="B", node="host-b", root="/venv-b")
    output = tmp_path / "refusal.json"
    report = f5_job._environment_evidence(
        {"environment_policy": environment.policy_binding(), "environment_capture": expected},
        actual,
        refusal_output=output,
    )
    assert (
        report["comparison"]["expected_canonical_sha256"]
        == report["comparison"]["actual_canonical_sha256"]
    )
    assert not output.exists()


def test_job_admission_exports_before_wrapping_refusal(monkeypatch, tmp_path: Path) -> None:
    expected, actual = _mismatch()
    monkeypatch.setattr(f5_job, "_environment_configuration_failures", lambda: [])
    monkeypatch.setattr(f5_job, "capture_environment", lambda: copy.deepcopy(actual))
    output = tmp_path / "refusal.json"
    with pytest.raises(f5_job.F5FitRefusal, match="cross-assignment"):
        f5_job._environment_admission(
            {"environment_policy": environment.policy_binding(), "environment_capture": expected},
            refusal_output=output,
        )
    verify_export(output)
    assert json.loads(output.read_text())["differing_canonical_keys"]


def test_real_canary_entrypoint_exports_before_workload_and_remains_failed(
    monkeypatch, tmp_path: Path
) -> None:
    expected, actual = _mismatch()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "b8.pth").write_bytes(b"opaque placeholder, never loaded")
    fixtures = inputs / "fixtures.json"
    # Isolate hardware/source/input admission; exercise the real CLI and mismatch path.
    monkeypatch.setattr(f5_qualification, "_require_cuda", lambda device: None)
    monkeypatch.setattr(f5_qualification, "load_fixtures", lambda path: ([], "f" * 64))
    monkeypatch.setattr(f5_qualification, "digest", lambda path: f5_job.B8_SHA256)
    monkeypatch.setattr(
        f5_job,
        "_control",
        lambda path: {
            "environment_policy": environment.policy_binding(),
            "environment_capture": expected,
            "qualification_fixture_sha256": "f" * 64,
            "qualification_b8_sha256": f5_job.B8_SHA256,
        },
    )
    monkeypatch.setattr(f5_job, "_source_admission", lambda control: None)
    monkeypatch.setattr(f5_job, "_environment_admission", lambda control: actual)

    def no_workload(*args, **kwargs):
        pytest.fail("mismatch must refuse before crossing a workload boundary")

    monkeypatch.setattr(f5_job, "_assignment_boundary", no_workload)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "f5_qualification",
            "--inputs",
            str(inputs),
            "--fixtures",
            str(fixtures),
            "--output",
            str(output),
            "--transport-only",
            "--canary-tiny-workload",
        ],
    )
    with pytest.raises(environment.F5CrossAssignmentRefusal, match="cross-assignment"):
        f5_qualification.main()
    path = output / "environment/cross-assignment-refusal.json"
    checksum = verify_export(path)
    inventory = _exports(tmp_path, 2 * 1024**3)
    assert inventory == [
        {
            "path": "environment/cross-assignment-refusal.json",
            "sha256": checksum,
            "size": path.stat().st_size,
        }
    ]
    assert not (output / "qualification").exists()
    assert not (output / "timing-units").exists()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        f5_qualification.main()
    assert path.read_bytes() == original


def test_malformed_and_same_assignment_receipts_keep_existing_refusals(tmp_path: Path) -> None:
    expected = _receipt()
    actual = _receipt(uuid="B")
    with pytest.raises(environment.F5EnvironmentRefusal, match="same-assignment") as raised:
        environment.compare_environment(expected, actual, mode="same-assignment")
    assert not isinstance(raised.value, environment.F5CrossAssignmentRefusal)
    actual["unknown"] = "refuse"
    with pytest.raises(environment.F5EnvironmentRefusal) as raised:
        f5_job._environment_evidence(
            {"environment_policy": environment.policy_binding(), "environment_capture": expected},
            actual,
            refusal_output=tmp_path / "refusal.json",
        )
    assert not isinstance(raised.value, environment.F5CrossAssignmentRefusal)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("operation", ["prepare", "fit"])
def test_existing_output_never_gains_refusal_exports(monkeypatch, tmp_path, operation) -> None:
    expected, actual = _mismatch()
    output = tmp_path / "completed-output"
    output.mkdir()
    prior = output / "existing-receipt.json"
    prior.write_bytes(b"immutable prior evidence\n")
    control = {
        "environment_policy": environment.policy_binding(),
        "environment_capture": expected,
    }
    monkeypatch.setattr(f5_job, "_control", lambda path: control)
    monkeypatch.setattr(f5_job, "_source_admission", lambda control: None)
    monkeypatch.setattr(f5_job, "_environment_configuration_failures", lambda: [])
    monkeypatch.setattr(f5_job, "capture_environment", lambda: actual)
    if operation == "prepare":
        with pytest.raises(FileExistsError, match="fresh"):
            f5_job.prepare(tmp_path, output)
    else:
        with pytest.raises(f5_job.F5FitRefusal, match="cross-assignment"):
            f5_job.fit_pair(tmp_path, tmp_path, tmp_path, tmp_path, output)
    assert list(output.iterdir()) == [prior]
    assert prior.read_bytes() == b"immutable prior evidence\n"


def test_existing_verified_fit_can_still_skip(monkeypatch, tmp_path) -> None:
    output = tmp_path / "completed-output"
    output.mkdir()
    expected = _receipt()
    control = {
        "environment_policy": environment.policy_binding(),
        "environment_capture": expected,
    }
    monkeypatch.setattr(f5_job, "_control", lambda path: control)
    monkeypatch.setattr(f5_job, "_source_admission", lambda control: None)
    monkeypatch.setattr(f5_job, "_environment_configuration_failures", lambda: [])
    monkeypatch.setattr(f5_job, "capture_environment", lambda: expected)
    monkeypatch.setattr(f5_job, "_bound_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(f5_job, "digest", lambda path: f5_job.B8_SHA256)
    monkeypatch.setattr(f5_job, "_verified_pair", lambda *args: {"verified": "unit"})
    result = f5_job.fit_pair(tmp_path, tmp_path, tmp_path, tmp_path, output)
    assert result["kind"] == "SKIPPED_VERIFIED_PAIRED_FIT_UNIT"
    assert not list(output.iterdir())
