"""Refused captures export diagnostic evidence without admitting science."""

from __future__ import annotations

import copy
import json
import subprocess
from types import SimpleNamespace

import pytest

from research.bl_ppo.exports import verify_export
from research.bl_residual import f5_environment, f5_job
from research.colab_cli.runtime import _exports


def test_capture_cli_exports_all_diagnostics_before_refusal(monkeypatch, tmp_path):
    raw = {"gpu_uuid": "01234567-89ab-cdef-0123-456789abcdef", "cpu_count": 12}
    evidence = {
        "raw_receipt": raw,
        "acquisition_failures": ["package read failed"],
        "validation_failures": ["UUID mismatch", "capability mismatch"],
        "admission_failures": ["package read failed", "UUID mismatch", "capability mismatch"],
        "admissible": False,
        "evidence_only": True,
        "scientific_use_authorized": False,
    }
    output = tmp_path / "output" / "environment.json"
    monkeypatch.setattr(f5_job, "_environment_admission", lambda *a, **k: copy.deepcopy(evidence))
    monkeypatch.setattr(
        f5_job.sys, "argv", ["f5_job", "capture-environment", "--output", str(output)]
    )
    with pytest.raises(f5_job.F5FitRefusal, match="package read failed; UUID mismatch; capability"):
        f5_job.main()
    assert not output.exists()
    evidence["raw_receipt"] = f5_environment.refuse_environment_receipt(
        raw, evidence["admission_failures"]
    )
    assert (
        json.loads(output.with_name("environment-raw.json").read_text()) == evidence["raw_receipt"]
    )
    diagnostic = output.with_name("environment-diagnostic.json")
    assert json.loads(diagnostic.read_text()) == evidence
    verify_export(diagnostic)
    # The actual runtime scanner discovers completed diagnostic payloads on a
    # nonzero child exit; supervisor.execute collects them before terminal_failure.
    inventory = _exports(tmp_path, 2 * 1024**3)
    assert {entry["path"] for entry in inventory} == {
        "environment-raw.json",
        "environment-diagnostic.json",
    }
    assert all(
        entry["sha256"] == verify_export(tmp_path / "output" / entry["path"]) for entry in inventory
    )
    before = diagnostic.read_bytes()
    with pytest.raises(FileExistsError):
        f5_job.export_environment_capture(output)
    assert diagnostic.read_bytes() == before


def test_admissible_capture_also_publishes_diagnostics(monkeypatch, tmp_path):
    evidence = {
        "raw_receipt": {"fixture": "valid"},
        "admission_failures": [],
        "admissible": True,
        "evidence_only": True,
    }
    monkeypatch.setattr(f5_job, "_environment_admission", lambda *a, **k: evidence)
    output = tmp_path / "environment.json"
    exported = f5_job.export_environment_capture(output)
    assert exported["sha256"] == verify_export(output)
    assert json.loads(output.read_text()) == evidence["raw_receipt"]
    assert output.with_name("environment-diagnostic.json").exists()


def test_admission_aggregates_config_and_raw_failures(monkeypatch):
    monkeypatch.setattr(
        f5_job,
        "_environment_configuration_failures",
        lambda: ["runtime version", "allocated affinity"],
    )
    monkeypatch.setattr(
        f5_environment,
        "capture_environment_diagnostic",
        lambda: {
            "raw_receipt": {"gpu": "NVIDIA L4"},
            "acquisition_failures": ["wheel receipt"],
            "validation_failures": ["memory", "UUID"],
            "admissible": False,
        },
    )
    result = f5_job._environment_admission(None, diagnostic=True)
    assert result["admission_failures"] == [
        "runtime version",
        "allocated affinity",
        "wheel receipt",
        "memory",
        "UUID",
    ]
    assert not result["admissible"]
    assert result["raw_receipt"] == f5_environment.refuse_environment_receipt(
        {"gpu": "NVIDIA L4"}, result["admission_failures"]
    )
    assert not result["scientific_use_authorized"]


def test_runtime_configuration_checks_independent_domains(monkeypatch):
    monkeypatch.setattr(f5_job.sys, "platform", "linux")
    monkeypatch.setattr(f5_job.sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(f5_job.torch, "__version__", f5_job.L4_TORCH)
    monkeypatch.setattr(f5_job.torch.version, "cuda", f5_job.L4_CUDA)
    monkeypatch.setattr(f5_job.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(f5_job.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        f5_job.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="NVIDIA L4")
    )
    monkeypatch.setattr(f5_job.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(
        f5_job.os, "sched_getaffinity", lambda _: {2, 4, 6, 8, 10, 12}, raising=False
    )
    monkeypatch.setattr(f5_job.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(f5_job.torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(f5_job.torch, "use_deterministic_algorithms", lambda _: None)
    monkeypatch.setattr(f5_job.torch, "get_default_dtype", lambda: f5_job.torch.float32)
    monkeypatch.setattr(f5_job.torch, "is_autocast_enabled", lambda _: False)
    for key in f5_environment.THREAD_ENV_KEYS:
        monkeypatch.setenv(key, ":4096:8" if key == "CUBLAS_WORKSPACE_CONFIG" else "1")
    assert f5_job._environment_configuration_failures() == []
    monkeypatch.setattr(f5_job.os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(f5_job.torch.version, "cuda", "12.8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    failures = f5_job._environment_configuration_failures()
    assert len(failures) == 3
    assert any("CUDA build" in value for value in failures)
    assert any("allocated vCPUs" in value for value in failures)
    assert any("MKL_NUM_THREADS" in value for value in failures)


def test_l40_name_cannot_pass_l4_configuration(monkeypatch):
    # Reuse the domain fixture setup and then reject an ambiguous family substring.
    test_runtime_configuration_checks_independent_domains(monkeypatch)
    monkeypatch.setattr(
        f5_job.torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="NVIDIA L40S")
    )
    assert any(
        "exact NVIDIA L4 name" in value for value in f5_job._environment_configuration_failures()
    )


@pytest.fixture(params=["failed-parseable-smi", "configuration-only"])
def refused_capture(request, monkeypatch, tmp_path):
    """Real capture/export refusal with otherwise admissible isolated-L4 observations."""
    from tests.test_bl_residual_f5_capture import RAW_SMI, _install_isolated_l4

    _install_isolated_l4(monkeypatch, tmp_path)
    successful = f5_environment.capture_environment()
    f5_environment.validate_environment(successful)
    monkeypatch.setattr(f5_job, "_environment_configuration_failures", lambda: [])
    if request.param == "failed-parseable-smi":

        def failed(*args, **kwargs):
            raise subprocess.CalledProcessError(
                3, args[0], output=RAW_SMI, stderr="driver warning\n"
            )

        monkeypatch.setattr(f5_environment.subprocess, "check_output", failed)
        # The collector's raw content must refuse even without job-layer marking.
        collected = f5_environment.capture_environment_diagnostic()
        assert not collected["admissible"]
        assert collected["raw_receipt"]["capture_status"] == "REFUSED"
        assert collected["raw_receipt"]["_acquisition_failures"]
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_environment.validate_environment(collected["raw_receipt"])
    else:
        monkeypatch.setattr(
            f5_job, "_environment_configuration_failures", lambda: ["configuration-only refusal"]
        )
    output = tmp_path / "output" / "environment.json"
    with pytest.raises(f5_job.F5FitRefusal):
        f5_job.export_environment_capture(output)
    assert not output.exists()
    raw_path = output.with_name("environment-raw.json")
    diagnostic_path = output.with_name("environment-diagnostic.json")
    verify_export(raw_path)
    verify_export(diagnostic_path)
    raw = json.loads(raw_path.read_text())
    wrapper = json.loads(diagnostic_path.read_text())
    assert wrapper["admissible"] is False
    assert raw == wrapper["raw_receipt"]
    assert raw["capture_status"] == "REFUSED"
    assert raw["_admission_failures"]
    assert raw["gpu_driver_query_raw"] == RAW_SMI
    if request.param == "failed-parseable-smi":
        assert raw["_acquisition_failures"] == wrapper["acquisition_failures"]
        assert raw["_acquisition_evidence"]["gpu_driver_query_raw"]["stderr"] == "driver warning\n"
    else:
        assert wrapper["acquisition_failures"] == wrapper["validation_failures"] == []
        assert raw["_admission_failures"] == ["configuration-only refusal"]
    # A fresh, successful current capture cannot legitimize the refused reference.
    monkeypatch.setattr(f5_job, "capture_environment", lambda: copy.deepcopy(successful))
    monkeypatch.setattr(
        f5_environment,
        "capture_environment_diagnostic",
        lambda: {
            "raw_receipt": copy.deepcopy(successful),
            "acquisition_failures": [],
            "validation_failures": [],
            "admissible": True,
        },
    )
    monkeypatch.setattr(f5_job, "_environment_configuration_failures", lambda: [])
    return raw, wrapper, successful


def test_refused_raw_cannot_escape_receipt_validation(refused_capture):
    raw, wrapper, _ = refused_capture
    for candidate in (raw, wrapper):
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_environment.validate_environment(candidate)


@pytest.mark.parametrize("mode", ["same-assignment", "cross-assignment"])
def test_refused_raw_cannot_escape_environment_comparison(refused_capture, mode):
    raw, wrapper, successful = refused_capture
    for candidate in (raw, wrapper):
        for expected, actual in (
            (candidate, candidate),
            (candidate, successful),
            (successful, candidate),
        ):
            with pytest.raises(f5_environment.F5EnvironmentRefusal):
                f5_environment.compare_environment(expected, actual, mode=mode)


def test_refused_raw_cannot_escape_control_admission(refused_capture):
    raw, wrapper, _ = refused_capture
    for candidate in (raw, wrapper):
        control = {
            "environment_policy": f5_environment.policy_binding(),
            "environment_capture": candidate,
        }
        with pytest.raises(f5_job.F5FitRefusal):
            f5_job._environment_admission(control)
        diagnostic = f5_job._environment_admission(control, diagnostic=True)
        assert not diagnostic["admissible"]
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_environment.validate_environment(diagnostic["raw_receipt"])


def test_refused_reference_sweep(refused_capture, monkeypatch, tmp_path):
    from research.bl_ppo.exports import export_json
    from research.bl_residual import f5_package, f5_qualification

    raw, wrapper, successful = refused_capture
    inputs = tmp_path / "bootstrap"
    inputs.mkdir()
    checkpoint = inputs / "b8.pth"
    checkpoint.write_bytes(b"synthetic checkpoint")
    monkeypatch.setattr(f5_qualification, "digest", lambda _: f5_job.B8_SHA256)
    monkeypatch.setattr(f5_package, "source_inventory", lambda _: {"fixture": "source"})
    (inputs / "bootstrap-control.json").write_text(
        json.dumps(
            {
                "schema": "bl-f5-bootstrap-qualification/v1",
                "source_hashes": {"fixture": "source"},
                "environment_policy": f5_environment.policy_binding(),
                "qualification_fixture_sha256": "fixture-sha",
                "b8_sha256": f5_job.B8_SHA256,
            }
        )
    )
    snapshot = tmp_path / "snapshot.pt"
    required = [
        "model_residual",
        "optimizer_state",
        "updates_completed",
        "losses",
        "identity",
        "recipe",
        "plan_sha256",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "plan_generator_state",
    ]
    export_json(snapshot, {key: None for key in required})
    monkeypatch.setattr(f5_job.torch, "load", lambda *a, **k: {key: None for key in required})
    for candidate in (raw, wrapper):
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_environment.canonical_environment(candidate)
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_job._assignment_boundary(candidate)
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_job._environment_evidence(
                {
                    "environment_policy": f5_environment.policy_binding(),
                    "environment_capture": candidate,
                },
                successful,
            )
        (inputs / "environment.json").write_text(json.dumps(candidate))
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_qualification._bootstrap_admission(inputs, inputs, "fixture-sha", checkpoint)
        with pytest.raises(f5_environment.F5EnvironmentRefusal):
            f5_job._resume_state(snapshot, None, candidate)


@pytest.mark.parametrize("status", [None, "REFUSED", "success", True, {}, []])
def test_successful_provenance_is_mandatory(status):
    from tests.test_bl_residual_f5_environment import _receipt

    receipt = _receipt()
    receipt["capture_status"] = status
    with pytest.raises(f5_environment.F5EnvironmentRefusal):
        f5_environment.validate_environment(receipt)
    del receipt["capture_status"]
    with pytest.raises(f5_environment.F5EnvironmentRefusal):
        f5_environment.validate_environment(receipt)
