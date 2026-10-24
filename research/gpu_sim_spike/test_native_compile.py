"""Offline compiler orchestration and fail-closed A100 admission tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.gpu_sim_spike import native_compile as native
from research.gpu_sim_spike import native_compile_pack, sparse_units

PIN = "a" * 40


def test_inventory_covers_every_cuda_source_and_production_flags():
    hashes, specs = native.inventory(native.ROOT)
    sources = {p.name for p in (native.ROOT / native.PACKAGE).glob("*.cu")}
    assert len(specs) == len(sources) == 5
    assert {Path(s["sources"][1]).name for s in specs} == sources
    exact = next(s for s in specs if "vector_geometry" in s["name"])
    assert exact["extra_cuda_cflags"] == [
        "-O3",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    ]
    assert all(str(native.PACKAGE + "/" + name) in hashes for name in sources)


def test_compile_builds_all_extensions_without_kernel_calls(tmp_path, monkeypatch):
    import torch
    from torch.utils import cpp_extension

    calls = []
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(native.subprocess, "check_output", lambda *a, **kw: "nvcc test")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "bogus")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "old-cache"))
    monkeypatch.setenv("MAX_JOBS", "99")
    report = native.run(tmp_path / "output/native-compile.json", PIN)
    assert report["status"] == "PASS" and report["kernels_launched"] is False
    assert len(calls) == 5
    assert report["architectures"] == "8.0;8.9"
    assert all(c["verbose"] for c in calls)
    assert all(Path(c["sources"][0]).suffix == ".cpp" for c in calls)
    sparse_units._verify(tmp_path / "output/native-compile.json")
    with pytest.raises(ValueError, match="fresh"):
        native.run(tmp_path / "output/native-compile.json", PIN)


def test_failed_compile_never_publishes_pass(tmp_path, monkeypatch):
    import torch
    from torch.utils import cpp_extension

    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "old-cache"))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setenv("MAX_JOBS", "2")

    def fail(**kwargs):
        raise RuntimeError("nvcc ambiguous overload")

    monkeypatch.setattr(cpp_extension, "load", fail)
    output = tmp_path / "output/native-compile.json"
    with pytest.raises(RuntimeError, match="ambiguous"):
        native.run(output, PIN)
    assert not output.exists() and not Path(str(output) + ".complete").exists()


def _proof(tmp_path, monkeypatch):
    hashes, specs = native.inventory(native.ROOT)
    monkeypatch.setattr(native, "inventory", lambda *a: (hashes, specs))
    output = tmp_path / "compile-output"
    exports = output / "exports"
    report = exports / "native-compile.json"
    sparse_units.publish_json(
        report,
        {
            "schema": native.SCHEMA,
            "status": "PASS",
            "commit": PIN,
            "architectures": native.ARCHES,
            "source_sha256": hashes,
            "extensions": specs,
            "kernels_launched": False,
        },
    )
    saved = output / "manifest.json"
    saved.write_text(
        json.dumps(
            {"commit": PIN, "script": f"{native.PACKAGE}/native_compile.sh", "job_id": "compile"}
        )
    )
    receipt = {
        "commit": PIN,
        "job_id": "compile",
        "manifest_sha256": native.digest(saved),
        "status": "PASS",
        "stop_verified": True,
        "error": None,
        "cleanup_error": None,
        "failure_class": None,
        "terminal": {"returncode": 0, "job_id": "compile"},
        "exports": [
            {"path": p.name, "size": p.stat().st_size, "sha256": native.digest(p)}
            for p in exports.iterdir()
        ],
    }
    path = output / "receipt.json"
    path.write_text(json.dumps(receipt))
    manifest = {
        "commit": PIN,
        "requirements": {"gpu": "A100"},
        "script": f"{native.PACKAGE}/colab.sh",
        "canary": {"mode": "probe"},
        "native_cuda_compile": {"receipt": {"path": str(path), "sha256": native.digest(path)}},
    }
    return path, receipt, manifest


def test_a100_compile_gate_accepts_verified_same_pin_proof(tmp_path, monkeypatch):
    _, _, manifest = _proof(tmp_path, monkeypatch)
    native.require_for_a100_canary(manifest)


@pytest.mark.parametrize("failure", ["missing", "pin", "stop", "terminal", "hash", "payload"])
def test_a100_compile_gate_refuses_missing_stale_or_tampered_proof(tmp_path, monkeypatch, failure):
    path, receipt, manifest = _proof(tmp_path, monkeypatch)
    if failure == "missing":
        manifest.pop("native_cuda_compile")
    elif failure == "pin":
        manifest["commit"] = "b" * 40
    elif failure == "stop":
        receipt["stop_verified"] = False
        path.write_text(json.dumps(receipt))
    elif failure == "terminal":
        receipt["terminal"]["returncode"] = 1
        path.write_text(json.dumps(receipt))
    elif failure == "hash":
        manifest["native_cuda_compile"]["receipt"]["sha256"] = "0" * 64
    elif failure == "payload":
        (path.parent / "exports/native-compile.json").write_text("{}")
    with pytest.raises(ValueError):
        native.require_for_a100_canary(manifest)


def test_compile_job_is_bounded_host_only_on_l4(tmp_path):
    value = native_compile_pack.manifest(PIN, tmp_path)
    assert value["requirements"]["gpu"] == "L4"
    assert value["device"] == "cpu"
    assert value["timeout_minutes"] == 20
    assert value["budget"]["unit_cap"] == 1.2
    assert value["resume_exemption"]["kind"] == "short-canary"
    native.require_for_a100_canary(value)  # The compiler job must not require itself.


def test_allowed_a100_substitution_also_requires_compile_proof():
    value = {
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4", "A100"]},
        "script": f"{native.PACKAGE}/colab.sh",
        "canary": {"mode": "probe"},
    }
    with pytest.raises(ValueError, match="requires native CUDA compile PASS"):
        native.require_for_a100_canary(value)
