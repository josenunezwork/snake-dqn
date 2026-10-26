"""Exhaustive refusal diagnostics for F5 environment receipts."""

from __future__ import annotations

from research.bl_residual import f5_environment as environment
from tests.test_bl_residual_f5_environment import _receipt


def test_collect_environment_failures_keeps_independent_schema_thread_package_and_gpu_errors() -> (
    None
):
    receipt = _receipt()
    receipt["schema"] = "bad-schema"
    receipt["schema_sha256"] = "0" * 64
    receipt["cpu_count"] = 0
    receipt["allocated_cpu_affinity"] = [0]
    receipt["torch"] = ""
    receipt["python_build"] = []
    receipt["torch_threads"] = {"intra": 0, "inter": True}
    receipt["installed_packages"]["torch"]["installed_contents_sha256"] = "0" * 64
    receipt["gpu_uuid"] = "different"
    receipt["gpu"] = "Tesla L4"
    receipt["gpu_capability"] = [8, 6]
    receipt["gpu_memory"] = 23036 * environment.MIB_BYTES

    failures = environment.collect_environment_failures(receipt)

    assert set(failures) >= {
        "F5 environment receipt schema differs",
        "F5 environment schema source pin differs",
        "F5 environment cpu_count is malformed",
        "F5 environment allocated_cpu_affinity is malformed",
        "F5 environment torch must be a nonempty string",
        "F5 environment python_build is malformed",
        "F5 environment torch thread value is malformed",
        "F5 environment installed contents digest differs: torch",
        "F5 environment GPU UUID provenance fields disagree",
        "F5 environment GPU name provenance fields disagree",
        "F5 environment parsed GPU capability differs",
        "F5 environment parsed GPU memory differs",
    }


def test_diagnostic_preserves_partial_cpu_gpu_evidence_after_smi_and_package_failures(
    monkeypatch, tmp_path
) -> None:
    """One source failure does not erase or re-read independent observations."""
    from tests.test_bl_residual_f5_capture import _install_isolated_l4

    _install_isolated_l4(monkeypatch, tmp_path)
    calls = []

    def bad_smi():
        calls.append("smi")
        return "unparseable raw L4 driver output\n"

    original = environment._package_receipt

    def package(name, wheel):
        calls.append(name)
        if name in {"torch", "numpy"}:
            raise OSError(f"package {name} failed")
        return original(name, wheel)

    monkeypatch.setattr(environment, "_driver_query_raw", bad_smi)
    monkeypatch.setattr(environment, "_package_receipt", package)
    diagnostic = environment.capture_environment_diagnostic()
    raw = diagnostic["raw_receipt"]
    assert not diagnostic["admissible"]
    assert raw["gpu"] == "NVIDIA L4"
    assert raw["cpu_count"] == 6
    assert raw["gpu_driver_query_raw"] == "unparseable raw L4 driver output\n"
    assert raw["gpu_driver_query"] is None
    assert raw["installed_packages"]["numpy"] is None
    assert raw["installed_packages"]["torch"] is None
    assert raw["installed_packages"]["numba"]
    assert calls.count("smi") == 1
    assert calls.count("torch") == 1
    assert calls.count("numpy") == 1
    failures = diagnostic["acquisition_failures"]
    assert len(failures) == 3
    assert any("gpu_driver_query" in value for value in failures)
    assert any("package numpy failed" in value for value in failures)
    assert any("package torch failed" in value for value in failures)
    assert diagnostic["validation_failures"]


def test_wheel_refusal_does_not_mask_package_metadata_failures() -> None:
    receipt = _receipt()
    receipt["locked_wheels"] = None
    receipt["installed_packages"]["torch"]["installed_metadata"]["METADATA"]["raw_base64"] = "!!!"
    failures = environment.collect_environment_failures(receipt)
    assert any("raw metadata is malformed: torch: METADATA" in value for value in failures)


def test_failed_smi_command_preserves_stdout_stderr_and_still_collects(monkeypatch, tmp_path):
    import subprocess

    from tests.test_bl_residual_f5_capture import RAW_SMI, _install_isolated_l4

    _install_isolated_l4(monkeypatch, tmp_path)

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(3, args[0], output=RAW_SMI, stderr="driver warning\n")

    monkeypatch.setattr(environment.subprocess, "check_output", failed)
    diagnostic = environment.capture_environment_diagnostic()
    assert not diagnostic["admissible"]
    assert diagnostic["raw_receipt"]["gpu_driver_query_raw"] == RAW_SMI
    assert diagnostic["raw_receipt"]["gpu_driver_query"]["name"] == "NVIDIA L4"
    evidence = diagnostic["acquisition_evidence"]["gpu_driver_query_raw"]
    assert evidence["stdout"] == RAW_SMI
    assert evidence["stderr"] == "driver warning\n"
    assert diagnostic["raw_receipt"]["installed_packages"]["torch"]


def test_real_colab_domains_and_sparse_affinity_are_kept_separate():
    receipt = _receipt(uuid="01234567-89ab-cdef-0123-456789abcdef")
    receipt["cpu_count"] = 12
    receipt["allocated_cpu_affinity"] = [8, 10, 12, 14, 16, 18]
    receipt["cpu"] = "x86_64"
    receipt["cpu_isa"] = {"model name": "Intel(R) Xeon(R) CPU @ 2.20GHz", "flags": "avx2 fma"}
    receipt["platform"] = "Linux-6.6.105+-x86_64-with-glibc2.35"
    receipt["os_kernel"].update(
        release="6.6.105+", version="#1 SMP PREEMPT_DYNAMIC", processor="x86_64"
    )
    receipt["cudnn"] = "91002"
    receipt["gpu_driver_query"]["driver_version"] = "550.54.15"
    receipt["gpu_driver_query_raw"] = receipt["gpu_driver_query_raw"].replace(
        ", 550,", ", 550.54.15,"
    )
    assert receipt["cuda"] == "12.6"
    assert receipt["gpu_driver_query"]["driver_version"] == "550.54.15"
    assert receipt["torch"] == "2.9.1+cu126"
    environment.validate_environment(receipt)
    assert environment.collect_environment_failures(receipt) == []
    # Equality remains exact for each invariant's own domain.
    import copy

    changed = copy.deepcopy(receipt)
    changed["os_kernel"]["release"] = "6.6.106+"
    import pytest

    with pytest.raises(environment.F5EnvironmentRefusal, match="invariant"):
        environment.compare_environment(receipt, changed, mode="cross-assignment")


def test_uuid_tail_is_case_exact_without_normalizing_torch():
    receipt = _receipt(uuid="abcdef67-89ab-cdef-0123-456789abcdef")
    receipt["gpu_uuid"] = receipt["gpu_uuid"].upper()
    failures = environment.collect_environment_failures(receipt)
    assert "F5 environment GPU UUID provenance fields disagree" in failures
