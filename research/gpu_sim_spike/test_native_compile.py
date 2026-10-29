"""Offline compiler orchestration and fail-closed A100 admission tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    assert report["architectures"] == "8.0;8.9;9.0"
    assert all(c["verbose"] for c in calls)
    assert all(Path(c["sources"][0]).suffix == ".cpp" for c in calls)
    sparse_units._verify(tmp_path / "output/native-compile.json")
    with pytest.raises(ValueError, match="fresh"):
        native.run(tmp_path / "output/native-compile.json", PIN)


def test_a100_repair_mode_compiles_only_architecture_80(tmp_path, monkeypatch):
    import torch
    from torch.utils import cpp_extension

    compiled = tmp_path / "compiler-cache"
    compiled.mkdir()
    calls = []
    monkeypatch.setattr(torch.version, "cuda", "12.6")

    def fake_load(**kwargs):
        path = compiled / f"{kwargs['name']}.so"
        path.write_bytes(kwargs["name"].encode())
        calls.append(kwargs)
        return SimpleNamespace(__file__=str(path))

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setattr(native.subprocess, "check_output", lambda *a, **kw: "nvcc test")
    monkeypatch.setenv(native.T4A_ARCH_MODE_ENV, native.T4A_A100_ARCH_MODE)
    monkeypatch.setenv(native.T4A_BINARY_ENV, "1")
    report = native.run(tmp_path / "output/native-compile.json", PIN)
    assert report["architecture_mode"] == native.T4A_A100_ARCH_MODE
    assert report["architectures"] == native.T4A_A100_ARCHES
    assert len(calls) == 5


def test_a100_repair_mode_refuses_nonexporting_compile(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setenv(native.T4A_ARCH_MODE_ENV, native.T4A_A100_ARCH_MODE)
    monkeypatch.delenv(native.T4A_BINARY_ENV, raising=False)
    with pytest.raises(ValueError, match="immutable binary exports"):
        native.run(tmp_path / "output/native-compile.json", PIN)


def test_t4a_opt_in_exports_every_loaded_binary_with_immutable_sidecars(tmp_path, monkeypatch):
    import torch
    from torch.utils import cpp_extension

    compiled = tmp_path / "compiler-cache"
    compiled.mkdir()
    calls = []
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(native.subprocess, "check_output", lambda *a, **kw: "nvcc test")

    def fake_load(**kwargs):
        path = compiled / f"{kwargs['name']}.so"
        path.write_bytes(kwargs["name"].encode())
        calls.append(kwargs["name"])
        return SimpleNamespace(__file__=str(path))

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    monkeypatch.setenv(native.T4A_BINARY_ENV, "1")
    output = tmp_path / "output/native-compile.json"
    report = native.run(output, PIN)
    assert {row["name"] for row in report["binaries"]} == set(calls)
    for row in report["binaries"]:
        payload = output.parent / row["path"]
        assert payload.is_file()
        assert payload.with_name(payload.name + ".sha256").is_file()
        assert payload.with_name(payload.name + ".complete").is_file()


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
            {"path": p.name, "size": p.stat().st_size, "sha256": native.digest(p)} for p in [report]
        ],
    }
    receipt["terminal"]["exports"] = receipt["exports"]
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


def _binary_proof(tmp_path, monkeypatch):
    path, receipt, manifest = _proof(tmp_path, monkeypatch)
    report_path = path.parent / "exports/native-compile.json"
    report = json.loads(report_path.read_text())
    rows = []
    for spec in report["extensions"]:
        name = spec["name"]
        payload = path.parent / "exports/native-binaries" / f"{name}.so"
        payload.parent.mkdir(exist_ok=True)
        payload.write_bytes(name.encode())
        payload.with_name(payload.name + ".sha256").write_text(
            f"{native.digest(payload)}  {payload.name}\n"
        )
        payload.with_name(payload.name + ".complete").write_text("complete\n")
        row = {
            "name": name,
            "path": f"native-binaries/{name}.so",
            "sha256": native.digest(payload),
            "size": payload.stat().st_size,
        }
        rows.append(row)
        receipt["exports"].append(
            {"path": row["path"], "sha256": row["sha256"], "size": row["size"]}
        )
    report["binaries"] = rows
    report_path.write_text(json.dumps(report))
    report_path.with_name(report_path.name + ".sha256").write_text(
        f"{native.digest(report_path)}  {report_path.name}\n"
    )
    receipt["exports"][0] = {
        "path": "native-compile.json",
        "sha256": native.digest(report_path),
        "size": report_path.stat().st_size,
    }
    receipt["terminal"]["exports"] = receipt["exports"]
    path.write_text(json.dumps(receipt))
    manifest["native_cuda_compile"] = {
        "receipt": {"path": str(path), "sha256": native.digest(path)}
    }
    return path, receipt, manifest


def test_a100_compile_gate_accepts_verified_same_pin_proof(tmp_path, monkeypatch):
    _, _, manifest = _proof(tmp_path, monkeypatch)
    native.require_for_a100_canary(manifest)


def test_t4a_compiler_repair_exemption_requires_exact_authorities_and_architecture(
    tmp_path, monkeypatch
):
    _path, _receipt, manifest = _proof(tmp_path, monkeypatch)
    names = [f"native-binaries/{spec['name']}.so" for spec in native.inventory(native.ROOT)[1]]
    manifest.update(
        {
            "script": f"{native.PACKAGE}/native_compile.sh",
            "device": "cpu",
            "env": {
                native.T4A_BINARY_ENV: "1",
                native.T4A_ARCH_MODE_ENV: native.T4A_A100_ARCH_MODE,
                "TORCH_CUDA_ARCH_LIST": native.T4A_A100_ARCHES,
            },
            "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "est_minutes": 15},
            "timeout_minutes": 30,
            "job_id": f"t4a-native-compile-a100-{PIN[:12]}-repair01",
            "resume_exemption": {"kind": "short-canary", "timeout_minutes": 30},
            "scientific_admission": {
                "amendment_sha256": native.T4A_COMPILE_AMENDMENT_SHA256,
                "compile_repair_sha256": native.T4A_COMPILE_REPAIR_SHA256,
            },
            "inputs": [
                {
                    "sha256": native.T4A_COMPILE_AMENDMENT_SHA256,
                    "dest": "inputs/authority/JUDGMENT_T4A_COMPILE_AMENDMENT.md",
                },
                {
                    "sha256": native.T4A_COMPILE_REPAIR_SHA256,
                    "dest": "inputs/authority/t4a-compile-repair.md",
                },
            ],
            "required_exports": ["native-compile.json", *names],
        }
    )
    manifest.pop("native_cuda_compile")
    native.require_for_a100_canary(manifest)
    manifest["env"][native.T4A_ARCH_MODE_ENV] = "wrong-mode"
    with pytest.raises(ValueError, match="requires native CUDA compile PASS"):
        native.require_for_a100_canary(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest["inputs"].pop(),
        lambda manifest: manifest["inputs"].pop(0),
        lambda manifest: manifest["scientific_admission"].__setitem__(
            "compile_repair_sha256", "0" * 64
        ),
        lambda manifest: manifest["scientific_admission"].__setitem__("amendment_sha256", "0" * 64),
        lambda manifest: manifest["env"].__setitem__("TORCH_CUDA_ARCH_LIST", native.ARCHES),
        lambda manifest: manifest.__setitem__("job_id", "t4a-native-compile-a100-wrong-repair01"),
        lambda manifest: manifest["resume_exemption"].__setitem__("timeout_minutes", 10),
        lambda manifest: manifest["required_exports"].pop(),
    ],
)
def test_t4a_compiler_repair_exemption_refuses_missing_or_mismatched_binding(
    tmp_path, monkeypatch, mutation
):
    _path, _receipt, manifest = _proof(tmp_path, monkeypatch)
    names = [f"native-binaries/{spec['name']}.so" for spec in native.inventory(native.ROOT)[1]]
    manifest.update(
        {
            "script": f"{native.PACKAGE}/native_compile.sh",
            "device": "cpu",
            "env": {
                native.T4A_BINARY_ENV: "1",
                native.T4A_ARCH_MODE_ENV: native.T4A_A100_ARCH_MODE,
                "TORCH_CUDA_ARCH_LIST": native.T4A_A100_ARCHES,
            },
            "requirements": {"gpu": "A100", "allowed_gpus": ["A100"], "est_minutes": 15},
            "timeout_minutes": 30,
            "job_id": f"t4a-native-compile-a100-{PIN[:12]}-repair01",
            "resume_exemption": {"kind": "short-canary", "timeout_minutes": 30},
            "scientific_admission": {
                "amendment_sha256": native.T4A_COMPILE_AMENDMENT_SHA256,
                "compile_repair_sha256": native.T4A_COMPILE_REPAIR_SHA256,
            },
            "inputs": [
                {
                    "sha256": native.T4A_COMPILE_AMENDMENT_SHA256,
                    "dest": "inputs/authority/JUDGMENT_T4A_COMPILE_AMENDMENT.md",
                },
                {
                    "sha256": native.T4A_COMPILE_REPAIR_SHA256,
                    "dest": "inputs/authority/t4a-compile-repair.md",
                },
            ],
            "required_exports": ["native-compile.json", *names],
        }
    )
    manifest.pop("native_cuda_compile")
    mutation(manifest)
    with pytest.raises(ValueError, match="requires native CUDA compile PASS"):
        native.require_for_a100_canary(manifest)


def test_binding_accepts_complete_opt_in_binary_receipt_and_refuses_tamper(tmp_path, monkeypatch):
    path, receipt, _manifest = _binary_proof(tmp_path, monkeypatch)
    proof = {"path": str(path), "sha256": native.digest(path)}
    assert native.binding(path, PIN) == proof
    first = receipt["exports"][1]["path"]
    (path.parent / "exports" / first).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="export inventory"):
        native.binding(path, PIN)


def test_binding_refuses_tampered_binary_sidecar(tmp_path, monkeypatch):
    path, receipt, _manifest = _binary_proof(tmp_path, monkeypatch)
    first = receipt["exports"][1]["path"]
    payload = path.parent / "exports" / first
    payload.with_name(payload.name + ".sha256").write_text("0" * 64 + "  wrong.so\n")
    with pytest.raises(ValueError, match="binary sidecars"):
        native.binding(path, PIN)


def test_binding_refuses_architecture_mode_and_architecture_mismatch(tmp_path, monkeypatch):
    path, receipt, _manifest = _binary_proof(tmp_path, monkeypatch)
    report = path.parent / "exports/native-compile.json"
    body = json.loads(report.read_text())
    body["architecture_mode"] = native.T4A_A100_ARCH_MODE
    report.write_text(json.dumps(body))
    _reseal_report(path, receipt, report)
    with pytest.raises(ValueError, match="architectures"):
        native.binding(path, PIN)


def test_binding_refuses_a100_repair_report_without_all_binaries(tmp_path, monkeypatch):
    path, receipt, _manifest = _proof(tmp_path, monkeypatch)
    report = path.parent / "exports/native-compile.json"
    body = json.loads(report.read_text())
    body["architecture_mode"] = native.T4A_A100_ARCH_MODE
    body["architectures"] = native.T4A_A100_ARCHES
    report.write_text(json.dumps(body))
    report.with_name(report.name + ".sha256").write_text(
        f"{native.digest(report)}  {report.name}\n"
    )
    receipt["exports"] = [
        {"path": report.name, "size": report.stat().st_size, "sha256": native.digest(report)}
    ]
    receipt["terminal"]["exports"] = receipt["exports"]
    saved = path.parent / "manifest.json"
    manifest = json.loads(saved.read_text())
    manifest["env"] = {
        native.T4A_ARCH_MODE_ENV: native.T4A_A100_ARCH_MODE,
        "TORCH_CUDA_ARCH_LIST": native.T4A_A100_ARCHES,
    }
    manifest["requirements"] = {
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 15,
    }
    saved.write_text(json.dumps(manifest))
    receipt["manifest_sha256"] = native.digest(saved)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="repair binary inventory"):
        native.binding(path, PIN)


def _reseal_report(path: Path, receipt: dict, report: Path) -> None:
    report.with_name(report.name + ".sha256").write_text(
        f"{native.digest(report)}  {report.name}\n"
    )
    receipt["exports"][0] = {
        "path": report.name,
        "size": report.stat().st_size,
        "sha256": native.digest(report),
    }
    receipt["terminal"]["exports"] = receipt["exports"]
    path.write_text(json.dumps(receipt))


def test_binding_refuses_repair_manifest_architecture_mode_mismatch(tmp_path, monkeypatch):
    path, receipt, _manifest = _binary_proof(tmp_path, monkeypatch)
    report = path.parent / "exports/native-compile.json"
    body = json.loads(report.read_text())
    body["architecture_mode"] = native.T4A_A100_ARCH_MODE
    body["architectures"] = native.T4A_A100_ARCHES
    report.write_text(json.dumps(body))
    _reseal_report(path, receipt, report)
    saved = path.parent / "manifest.json"
    manifest = json.loads(saved.read_text())
    manifest["env"] = {native.T4A_ARCH_MODE_ENV: "wrong-mode"}
    saved.write_text(json.dumps(manifest))
    receipt["manifest_sha256"] = native.digest(saved)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="architecture mode"):
        native.binding(path, PIN)


def test_binding_refuses_repair_report_with_non_a100_manifest(tmp_path, monkeypatch):
    path, receipt, _manifest = _binary_proof(tmp_path, monkeypatch)
    report = path.parent / "exports/native-compile.json"
    body = json.loads(report.read_text())
    body["architecture_mode"] = native.T4A_A100_ARCH_MODE
    body["architectures"] = native.T4A_A100_ARCHES
    report.write_text(json.dumps(body))
    _reseal_report(path, receipt, report)
    saved = path.parent / "manifest.json"
    manifest = json.loads(saved.read_text())
    manifest["env"] = {
        native.T4A_ARCH_MODE_ENV: native.T4A_A100_ARCH_MODE,
        "TORCH_CUDA_ARCH_LIST": native.T4A_A100_ARCHES,
    }
    manifest["requirements"] = {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 12,
        "est_minutes": 15,
    }
    saved.write_text(json.dumps(manifest))
    receipt["manifest_sha256"] = native.digest(saved)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="A100 scope"):
        native.binding(path, PIN)


def test_binding_refuses_duplicate_binary_export_rows(tmp_path, monkeypatch):
    path, receipt, _manifest = _binary_proof(tmp_path, monkeypatch)
    receipt["exports"].append(dict(receipt["exports"][1]))
    receipt["terminal"]["exports"] = receipt["exports"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="malformed"):
        native.binding(path, PIN)


@pytest.mark.parametrize(
    "failure",
    ["missing", "pin", "stop", "terminal", "hash", "payload", "sidecar", "marker", "inventory"],
)
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
    elif failure in {"sidecar", "marker"}:
        suffix = ".sha256" if failure == "sidecar" else ".complete"
        (path.parent / ("exports/native-compile.json" + suffix)).write_text("tampered\n")
    elif failure == "inventory":
        receipt["exports"].append(dict(receipt["exports"][0]))
        path.write_text(json.dumps(receipt))
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


@pytest.mark.parametrize("architectures", [native.ARCHES, native.LEGACY_ARCHES])
@pytest.mark.parametrize("gpu", ["H100", "A100"])
def test_compile_binding_checks_requested_hardware_architecture(
    tmp_path, monkeypatch, architectures, gpu
):
    path, receipt, manifest = _proof(tmp_path, monkeypatch)
    report = path.parent / "exports/native-compile.json"
    body = json.loads(report.read_text())
    body["architectures"] = architectures
    report.write_text(json.dumps(body))
    _reseal_report(path, receipt, report)
    proof = {"path": str(path), "sha256": native.digest(path)}
    manifest["requirements"] = {"gpu": gpu, "allowed_gpus": [gpu]}
    manifest["native_cuda_compile"]["receipt"] = proof
    if gpu == "H100" and architectures == native.LEGACY_ARCHES:
        with pytest.raises(ValueError, match="lacks H100 architecture 9.0"):
            native.binding(path, PIN, requested_gpu=gpu)
        with pytest.raises(ValueError, match="lacks H100 architecture 9.0"):
            native.require_for_a100_canary(manifest)
    else:
        assert native.binding(path, PIN, requested_gpu=gpu) == proof
        native.require_for_a100_canary(manifest)


def test_allowed_h100_substitution_requires_hopper_compile_architecture(tmp_path, monkeypatch):
    path, receipt, manifest = _proof(tmp_path, monkeypatch)
    report = path.parent / "exports/native-compile.json"
    body = json.loads(report.read_text())
    body["architectures"] = native.LEGACY_ARCHES
    report.write_text(json.dumps(body))
    _reseal_report(path, receipt, report)
    manifest["native_cuda_compile"]["receipt"]["sha256"] = native.digest(path)
    manifest["requirements"] = {"gpu": "L4", "allowed_gpus": ["L4", "H100"]}
    with pytest.raises(ValueError, match="lacks H100 architecture 9.0"):
        native.require_for_a100_canary(manifest)
