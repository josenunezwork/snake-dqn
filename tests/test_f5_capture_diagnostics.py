"""Refused captures export diagnostic evidence without admitting science."""

from __future__ import annotations

import copy
import json
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
    assert json.loads(output.with_name("environment-raw.json").read_text()) == raw
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
    assert result["raw_receipt"] == {"gpu": "NVIDIA L4"}
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
