"""T4A receipt-binary runtime binding checks; no compiler or CUDA required."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.gpu_sim_spike import native_compile, t4a_native


def _report(commit: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    import torch

    monkeypatch.setattr(torch.version, "cuda", "12.6")
    hashes, specs = native_compile.inventory(native_compile.ROOT)
    monkeypatch.setattr(native_compile, "inventory", lambda *args: (hashes, specs))
    rows = []
    for spec in specs:
        name = spec["name"]
        payload = tmp_path / "inputs/native/native-binaries" / f"{name}.so"
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes((name + " binary").encode())
        rows.append(
            {
                "name": name,
                "path": f"native-binaries/{name}.so",
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "size": payload.stat().st_size,
            }
        )
    report = {
        "schema": native_compile.SCHEMA,
        "status": "PASS",
        "commit": commit,
        "source_sha256": hashes,
        "extensions": specs,
        "architectures": native_compile.ARCHES,
        "kernels_launched": False,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvcc": "nvcc test",
        "binaries": rows,
    }
    path = tmp_path / "inputs/native/native-compile.json"
    path.write_text(json.dumps(report))
    return path, report


def test_runtime_environment_is_explicit_and_hash_bound() -> None:
    digest = "a" * 64
    assert t4a_native.runtime_environment(digest)[t4a_native.REPORT_ENV] == t4a_native.REPORT_DEST
    assert (
        t4a_native.runtime_environment(digest)[native_compile.T4A_ARCH_MODE_ENV]
        == native_compile.T4A_A100_ARCH_MODE
    )
    with pytest.raises(RuntimeError):
        t4a_native.runtime_environment("bad")


def test_report_refuses_none_or_missing_binary_descriptors(tmp_path, monkeypatch) -> None:
    path, report = _report("a" * 40, tmp_path, monkeypatch)
    report["binaries"][0]["sha256"] = None
    path.write_text(json.dumps(report))
    monkeypatch.setenv("COLAB_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv(native_compile.T4A_BINARY_ENV, "1")
    monkeypatch.setenv(t4a_native.REPORT_ENV, t4a_native.REPORT_DEST)
    monkeypatch.setenv(t4a_native.REPORT_SHA_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="SHA-256"):
        t4a_native.preload_from_environment("a" * 40)


def test_preload_installs_exact_verified_modules_without_loader_rebuild(
    tmp_path, monkeypatch
) -> None:
    import importlib

    commit = "a" * 40
    path, report = _report(commit, tmp_path, monkeypatch)
    monkeypatch.setenv("COLAB_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv(native_compile.T4A_BINARY_ENV, "1")
    monkeypatch.setenv(t4a_native.REPORT_ENV, t4a_native.REPORT_DEST)
    monkeypatch.setenv(t4a_native.REPORT_SHA_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    modules = {}

    def fake_load(name: str, binary: Path):
        value = SimpleNamespace(__file__=str(binary), marker=name)
        modules[name] = value
        return value

    monkeypatch.setattr(t4a_native, "_load_extension", fake_load)
    for _name, (module_name, attribute) in t4a_native._LOADERS.items():
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, attribute, None)
    loaded = t4a_native.preload_from_environment(commit)
    assert loaded == {row["name"]: row["sha256"] for row in report["binaries"]}
    for name, (module_name, attribute) in t4a_native._LOADERS.items():
        assert getattr(importlib.import_module(module_name), attribute) is modules[name]


def test_preload_refuses_tampered_runtime_binary_before_import(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    path, report = _report(commit, tmp_path, monkeypatch)
    binary = tmp_path / "inputs/native" / report["binaries"][0]["path"]
    binary.write_bytes(b"changed")
    monkeypatch.setenv("COLAB_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv(native_compile.T4A_BINARY_ENV, "1")
    monkeypatch.setenv(t4a_native.REPORT_ENV, t4a_native.REPORT_DEST)
    monkeypatch.setenv(t4a_native.REPORT_SHA_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="binary differs"):
        t4a_native.preload_from_environment(commit)


def test_runtime_refuses_architecture_mode_different_from_report(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    path, _report_body = _report(commit, tmp_path, monkeypatch)
    monkeypatch.setenv("COLAB_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv(native_compile.T4A_BINARY_ENV, "1")
    monkeypatch.setenv(native_compile.T4A_ARCH_MODE_ENV, native_compile.T4A_A100_ARCH_MODE)
    monkeypatch.setenv(t4a_native.REPORT_ENV, t4a_native.REPORT_DEST)
    monkeypatch.setenv(t4a_native.REPORT_SHA_ENV, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="architecture mode differs"):
        t4a_native.preload_from_environment(commit)
