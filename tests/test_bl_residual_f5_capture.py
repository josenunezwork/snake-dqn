"""Isolated-L4 regressions for the F5 environment capture receipt.

These tests deliberately construct an installed-distribution inventory from the
frozen lock, rather than manufacturing a completed receipt.  They exercise the
real package-receipt and receipt-validation paths with only host, CUDA, and
wheelhouse I/O replaced.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_residual import f5_environment as environment

L4_MIB = 23034
L4_BYTES = L4_MIB * 1024 * 1024
L4_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
RAW_SMI = f"NVIDIA L4, {L4_UUID}, 550.54.15, 8.9, {L4_MIB}\n"


def _locked_names() -> list[str]:
    """Return the normalized package names from the adopted L4 lock."""
    lock = (
        Path(environment.__file__).with_name("f5_adoption_environment_20261026_v6")
        / "requirements-f5-l4-cu126.lock"
    )
    names: list[str] = []
    for line in lock.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==[^\\\s]+", line)
        if match:
            names.append(match.group(1).lower().replace("_", "-"))
    assert names and len(names) == len(set(names))
    assert "scipy" not in names
    return names


def _locked_versions() -> dict[str, str]:
    """Return the exact normalized names and versions in the frozen L4 lock."""
    lock = (
        Path(environment.__file__).with_name("f5_adoption_environment_20261026_v6")
        / "requirements-f5-l4-cu126.lock"
    )
    versions: dict[str, str] = {}
    for line in lock.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\\\s]+)", line)
        if match:
            versions[match.group(1).lower().replace("_", "-")] = match.group(2)
    assert set(versions) == set(_locked_names())
    return versions


def _path_distribution(root: Path, name: str, version: str) -> importlib.metadata.PathDistribution:
    """Build a minimal installed package that ``_package_receipt`` can inspect."""
    site_packages = root / "lib" / "python3.12" / "site-packages"
    package_name = name.replace("-", "_")
    package = site_packages / package_name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(f"PACKAGE = {name!r}\n")
    dist_info = site_packages / f"{package_name}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
    (dist_info / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: fixture\n")
    (dist_info / "RECORD").write_text(
        "\n".join(
            (
                f"{package_name}/__init__.py,,",
                f"{dist_info.name}/METADATA,,",
                f"{dist_info.name}/WHEEL,,",
                f"{dist_info.name}/RECORD,,",
            )
        )
        + "\n"
    )
    return importlib.metadata.PathDistribution(dist_info)


def _inventory(root: Path, names: list[str]) -> dict[str, importlib.metadata.PathDistribution]:
    versions = _locked_versions()
    return {name: _path_distribution(root, name, versions.get(name, "1.0")) for name in names}


def _wheel_manifest(names: list[str]) -> dict[str, dict[str, str]]:
    """Provide the lock-selected wheel closure at the permitted I/O seam."""
    return {
        name: {
            "filename": f"{name}-1.0-py3-none-any.whl",
            "raw_url": f"file:///fixture-wheelhouse/{name}-1.0-py3-none-any.whl",
            "canonical_url": f"file://<WHEELHOUSE>/{name}-1.0-py3-none-any.whl",
            "sha256": hashlib.sha256(f"locked-wheel:{name}".encode()).hexdigest(),
        }
        for name in names
    }


def _install_isolated_l4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    names: list[str] | None = None,
    wheel_names: list[str] | None = None,
    torch_bytes: int = L4_BYTES,
) -> list[str]:
    """Install only host/hardware and wheelhouse seams around real capture code."""
    installed_names = _locked_names() if names is None else names
    selected_wheels = installed_names if wheel_names is None else wheel_names
    inventory = _inventory(tmp_path / "isolated-l4", installed_names)

    def distributions() -> list[importlib.metadata.PathDistribution]:
        return list(inventory.values())

    def distribution(name: str) -> importlib.metadata.PathDistribution:
        normalized = name.lower().replace("_", "-")
        try:
            return inventory[normalized]
        except KeyError as exc:
            raise importlib.metadata.PackageNotFoundError(name) from exc

    original_is_file = Path.is_file
    original_read_text = Path.read_text

    def is_file(path: Path) -> bool:
        return str(path) == "/proc/cpuinfo" or original_is_file(path)

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/cpuinfo":
            return "model name\t: isolated L4 host\nflags\t: avx2 fma\n"
        return original_read_text(path, *args, **kwargs)

    props = SimpleNamespace(name="NVIDIA L4", uuid=L4_UUID, total_memory=torch_bytes)
    uname = SimpleNamespace(
        _asdict=lambda: {
            "system": "Linux",
            "node": "isolated-l4",
            "release": "6.1",
            "version": "#1",
            "machine": "x86_64",
            "processor": "isolated CPU",
        }
    )
    seen_commands: list[list[str]] = []

    def check_output(command: list[str], **kwargs: object) -> str:
        seen_commands.append(command)
        assert kwargs == {"text": True, "timeout": 15}
        return RAW_SMI

    monkeypatch.setattr(environment.importlib.metadata, "distributions", distributions)
    monkeypatch.setattr(environment.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(environment, "_wheel_manifest", lambda: _wheel_manifest(selected_wheels))
    monkeypatch.setattr(environment.Path, "is_file", is_file)
    monkeypatch.setattr(environment.Path, "read_text", read_text)
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(environment.os, "sched_getaffinity", lambda _: set(range(6)), raising=False)
    for name in environment.THREAD_ENV_KEYS:
        monkeypatch.setenv(name, ":4096:8" if name == "CUBLAS_WORKSPACE_CONFIG" else "1")
    monkeypatch.setattr(environment.platform, "uname", lambda: uname)
    monkeypatch.setattr(environment.platform, "platform", lambda: "Linux-6.1-x86_64")
    monkeypatch.setattr(environment.platform, "processor", lambda: "isolated CPU")
    monkeypatch.setattr(environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(environment.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(environment.torch.cuda, "get_device_properties", lambda _: props)
    monkeypatch.setattr(environment.torch.cuda, "get_device_capability", lambda _: (8, 9))
    versions = _locked_versions()
    monkeypatch.setattr(environment.torch, "__version__", versions["torch"])
    monkeypatch.setattr(environment.np, "__version__", versions["numpy"])
    monkeypatch.setattr(environment.subprocess, "check_output", check_output)
    monkeypatch.setattr(environment, "_capture_test_smi_commands", seen_commands, raising=False)
    return installed_names


def _capture_isolated_l4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: object
) -> dict[str, object]:
    _install_isolated_l4(monkeypatch, tmp_path, **kwargs)
    receipt = environment.capture_environment()
    assert environment._capture_test_smi_commands == [
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ]
    return receipt


def test_capture_uses_exact_isolated_lock_closure_without_optional_scipy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The actual capture succeeds with the 69-package frozen L4 inventory."""
    names = _install_isolated_l4(monkeypatch, tmp_path)
    assert len(names) == 69
    parsed, raw = environment._driver_query()
    assert raw == RAW_SMI
    assert parsed["memory_total"] == str(L4_MIB)

    receipt = environment.capture_environment()

    assert receipt["installed_numeric_packages"]["scipy"] is None
    assert "scipy" not in receipt["installed_packages"]
    assert "scipy" not in receipt["locked_wheels"]
    assert receipt["gpu_memory"] == L4_BYTES
    assert receipt["gpu_driver_query"]["memory_total"] == str(L4_MIB)
    assert receipt["memory_units"] == {
        "gpu_memory": "bytes",
        "gpu_driver_query.memory_total": "MiB",
    }
    environment.validate_environment(receipt)


@pytest.mark.parametrize("missing", ["numba", "torch"])
def test_capture_refuses_missing_required_installed_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    names = [name for name in _locked_names() if name != missing]
    _install_isolated_l4(monkeypatch, tmp_path, names=names, wheel_names=names)
    with pytest.raises(environment.F5EnvironmentRefusal):
        environment.capture_environment()


def test_capture_refuses_unlocked_ambient_scipy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = [*_locked_names(), "scipy"]
    _install_isolated_l4(monkeypatch, tmp_path, names=names, wheel_names=_locked_names())
    with pytest.raises(environment.F5EnvironmentRefusal):
        environment.capture_environment()


def test_capture_refuses_missing_required_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = _locked_names()
    _install_isolated_l4(
        monkeypatch,
        tmp_path,
        names=names,
        wheel_names=[name for name in names if name != "llvmlite"],
    )
    with pytest.raises(environment.F5EnvironmentRefusal):
        environment.capture_environment()


def test_capture_refuses_locked_nonnumeric_wheel_missing_from_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    names = _locked_names()
    omitted = "absl-py"
    assert omitted in names
    _install_isolated_l4(
        monkeypatch,
        tmp_path,
        names=[name for name in names if name != omitted],
        wheel_names=names,
    )
    with pytest.raises(environment.F5EnvironmentRefusal, match="installed closure"):
        environment.capture_environment()


@pytest.mark.parametrize(("module_name", "version"), [("torch", "0.0"), ("np", "0.0")])
def test_capture_refuses_torch_or_numpy_version_alias_contradiction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module_name: str, version: str
) -> None:
    _install_isolated_l4(monkeypatch, tmp_path)
    monkeypatch.setattr(getattr(environment, module_name), "__version__", version)
    with pytest.raises(environment.F5EnvironmentRefusal, match="version disagrees"):
        environment.capture_environment()


def test_capture_in_fresh_isolated_python_blocks_scipy_import(tmp_path: Path) -> None:
    """Run the actual capture helper in ``-I`` with the synthetic L4 inventory."""
    script = textwrap.dedent(f"""
        import builtins
        import runpy
        import sys
        from pathlib import Path

        sys.path.insert(0, {str(Path.cwd())!r})
        helpers = runpy.run_path({str(Path(__file__))!r})
        from _pytest.monkeypatch import MonkeyPatch

        monkeypatch = MonkeyPatch()
        try:
            helpers["_install_isolated_l4"](monkeypatch, Path({str(tmp_path)!r}))
            original_import = builtins.__import__
            def block_scipy(name, *args, **kwargs):
                if name.split(".", 1)[0] == "scipy":
                    raise ImportError("scipy intentionally absent from isolated F5 L4 runtime")
                return original_import(name, *args, **kwargs)
            builtins.__import__ = block_scipy
            try:
                try:
                    __import__("scipy")
                except ImportError:
                    pass
                else:
                    raise AssertionError("scipy import was not blocked")
                receipt = helpers["environment"].capture_environment()
                assert receipt["installed_numeric_packages"]["scipy"] is None
                assert receipt["gpu_memory"] == helpers["L4_BYTES"]
            finally:
                builtins.__import__ = original_import
        finally:
            monkeypatch.undo()
        """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **dict(os.environ),
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("torch_bytes", "accepted"),
    [
        (L4_BYTES, True),
        (L4_BYTES + 1024 * 1024 - 1, True),
        (L4_BYTES - 1024 * 1024 + 1, True),
        (L4_BYTES + 1024 * 1024, False),
        (L4_BYTES - 1024 * 1024, False),
        (L4_MIB, False),
    ],
)
def test_capture_normalizes_torch_bytes_against_smi_mib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, torch_bytes: int, accepted: bool
) -> None:
    _install_isolated_l4(monkeypatch, tmp_path, torch_bytes=torch_bytes)
    if accepted:
        receipt = environment.capture_environment()
        assert receipt["gpu_memory"] == torch_bytes
        environment.validate_environment(receipt)
    else:
        with pytest.raises(environment.F5EnvironmentRefusal, match="memory"):
            environment.capture_environment()


def _assigned_copy(receipt: dict[str, object], *, uuid: str) -> dict[str, object]:
    """Make the documented assignment-local GPU identity change consistently."""
    changed = copy.deepcopy(receipt)
    changed["gpu_uuid"] = uuid
    driver = changed["gpu_driver_query"]
    assert isinstance(driver, dict)
    driver["uuid"] = uuid
    changed["gpu_driver_query_raw"] = str(changed["gpu_driver_query_raw"]).replace(L4_UUID, uuid)
    return changed


@pytest.mark.parametrize("memory_field", ["gpu_memory", "gpu_driver_query_memory"])
def test_cross_assignment_keeps_each_raw_memory_measurement_invariant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, memory_field: str
) -> None:
    baseline = _capture_isolated_l4(monkeypatch, tmp_path, torch_bytes=L4_BYTES + 512 * 1024)
    assert isinstance(baseline["gpu_memory"], int)
    fractional = _assigned_copy(baseline, uuid="GPU-fedcba98-7654-3210-fedc-ba9876543210")
    accepted = environment.compare_environment(baseline, fractional, mode="cross-assignment")
    assert accepted["actual_raw_sha256"] != accepted["expected_raw_sha256"]
    if memory_field == "gpu_memory":
        fractional["gpu_memory"] = int(baseline["gpu_memory"]) - 1
    else:
        driver = fractional["gpu_driver_query"]
        assert isinstance(driver, dict)
        driver["memory_total"] = str(L4_MIB + 1)
        fractional["gpu_driver_query_raw"] = (
            f"NVIDIA L4, {driver['uuid']}, 550.54.15, 8.9, {L4_MIB + 1}\n"
        )
    environment.validate_environment(fractional)
    with pytest.raises(environment.F5EnvironmentRefusal, match="invariant"):
        environment.compare_environment(baseline, fractional, mode="cross-assignment")
