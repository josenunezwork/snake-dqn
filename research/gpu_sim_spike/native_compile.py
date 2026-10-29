"""Compile every native extension without launching kernels or scoring outcomes.

The host compiler is a CPU workload. An L4 allocation supplies the CUDA toolkit;
explicit fatbin targets also compile A100 code. This is never gameplay proof.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

PACKAGE = "research/gpu_sim_spike"
ROOT = Path(__file__).resolve().parents[2]
ARCHES = "8.0;8.9"
T4A_A100_ARCH_MODE = "a100-8.0"
T4A_A100_ARCHES = "8.0"
SCHEMA = "gpu-native-compile/v1"
T4A_BINARY_ENV = "T4A_NATIVE_BINARIES"
T4A_ARCH_MODE_ENV = "T4A_NATIVE_ARCH_MODE"
T4A_COMPILE_AMENDMENT_SHA256 = "1eac049440eaad214400493037c88b7098abbde530be0125ea15b30363c1a57f"
T4A_COMPILE_REPAIR_SHA256 = "6900219f9313b1c37ca8c0e98c4bf5e786805fa2614b35308e229d5b4ae9b5a4"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architectures_for_mode(mode: object) -> str:
    """Return the frozen CUDA architecture list for one explicit T4A mode."""
    if mode in (None, ""):
        return ARCHES
    if mode == T4A_A100_ARCH_MODE:
        return T4A_A100_ARCHES
    raise ValueError("unknown T4A native architecture mode")


def report_architectures(report: dict[str, Any]) -> tuple[str | None, str]:
    """Validate a report's architecture declaration without broadening old reports."""
    mode = report.get("architecture_mode")
    if mode is not None and not isinstance(mode, str):
        raise ValueError("native compile architecture mode is malformed")
    architectures = architectures_for_mode(mode)
    if report.get("architectures") != architectures:
        raise ValueError("native compile architectures differ")
    return mode, architectures


def _publish_binary(path: Path, source: Path) -> None:
    """Publish one compiled shared object with immutable runtime sidecars."""
    if (
        path.exists()
        or Path(str(path) + ".sha256").exists()
        or Path(str(path) + ".complete").exists()
    ):
        raise ValueError(f"native binary export already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_copy(source, path)
    _atomic_create(path.with_name(path.name + ".sha256"), f"{digest(path)}  {path.name}\n".encode())
    _atomic_create(path.with_name(path.name + ".complete"), b"complete\n")


def _atomic_copy(source: Path, destination: Path) -> None:
    """Hard-link a fully fsynced copy into an absent immutable destination."""
    with (
        source.open("rb") as input_stream,
        tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".native-", delete=False
        ) as output_stream,
    ):
        temporary = Path(output_stream.name)
        while block := input_stream.read(1024 * 1024):
            output_stream.write(block)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"native binary export already exists: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".native-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"native binary export already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def inventory(repo: Path, commit: str | None = None) -> tuple[dict, list[dict]]:
    """Read native sources and production load() arguments, optionally from Git."""
    if commit:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, PACKAGE], cwd=repo, text=True
        ).splitlines()

        def read(name: str) -> bytes:
            return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)

    else:
        names = [str(p.relative_to(repo)) for p in (repo / PACKAGE).glob("*") if p.is_file()]

        def read(name: str) -> bytes:
            return (repo / name).read_bytes()

    cuda = sorted(name for name in names if name.endswith(".cu"))
    if not cuda:
        raise ValueError("native compile inventory is empty")
    hashes, specs = {}, []
    for source in cuda:
        stem = source[:-3]
        paths = [source, stem + ".cpp", stem + ".py"]
        for name in paths:
            hashes[name] = hashlib.sha256(read(name)).hexdigest()
        tree = ast.parse(read(stem + ".py"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "load"
        ]
        if len(calls) != 1:
            raise ValueError(f"expected exactly one production load() in {stem}.py")
        args = {kw.arg: kw.value for kw in calls[0].keywords}
        if calls[0].args or set(args) != {
            "name",
            "sources",
            "extra_cflags",
            "extra_cuda_cflags",
            "verbose",
        }:
            raise ValueError(f"unhandled production load() arguments: {stem}.py")
        # Fail closed if the production loader's source list stops using these twins.
        expected = ast.parse(
            f'[str(root / "{Path(stem).name}.cpp"), str(root / "{Path(stem).name}.cu")]',
            mode="eval",
        ).body
        if ast.dump(args["sources"]) != ast.dump(expected):
            raise ValueError(f"unhandled production sources: {stem}.py")
        spec = {
            key: ast.literal_eval(args[key])
            for key in ("name", "extra_cflags", "extra_cuda_cflags")
        }
        specs.append({**spec, "sources": [stem + ".cpp", source]})
    for name in ("native_compile.py", "native_compile.sh", "native_compile_requirements.txt"):
        path = f"{PACKAGE}/{name}"
        hashes[path] = hashlib.sha256(read(path)).hexdigest()
    return hashes, specs


def run(output: Path, commit: str) -> dict[str, Any]:
    """Compile/link/load production extensions in a fresh cache, with no kernel calls."""
    import torch
    from torch.utils.cpp_extension import load

    if output.exists() or (output.parent.parent / "native-compile.build").exists():
        raise ValueError("native compile requires fresh output and build directories")
    if torch.version.cuda is None:
        raise ValueError("native compile requires CUDA PyTorch and nvcc")
    mode = os.environ.get(T4A_ARCH_MODE_ENV)
    architectures = architectures_for_mode(mode)
    export_binaries = os.environ.get(T4A_BINARY_ENV) == "1"
    if mode == T4A_A100_ARCH_MODE and not export_binaries:
        raise ValueError("T4A A100 architecture mode requires immutable binary exports")
    os.environ["TORCH_CUDA_ARCH_LIST"] = architectures
    os.environ["MAX_JOBS"] = "2"
    hashes, specs = inventory(ROOT)
    build = output.parent.parent / "native-compile.build"
    build.mkdir(parents=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build)
    began = time.monotonic()
    binary_rows: list[dict[str, str | int]] = []
    for spec in specs:
        print(f"Compiling {spec['name']} targets={architectures}", flush=True)
        extension = load(
            **{key: value for key, value in spec.items() if key != "sources"},
            sources=[str(ROOT / name) for name in spec["sources"]],
            verbose=True,
        )
        if export_binaries:
            source = Path(str(extension.__file__)).resolve(strict=True)
            relative = f"native-binaries/{spec['name']}.so"
            target = output.parent / relative
            _publish_binary(target, source)
            binary_rows.append(
                {
                    "name": str(spec["name"]),
                    "path": relative,
                    "sha256": digest(target),
                    "size": target.stat().st_size,
                }
            )
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "commit": commit,
        "architectures": architectures,
        "source_sha256": hashes,
        "extensions": specs,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "elapsed_seconds": time.monotonic() - began,
        "kernels_launched": False,
    }
    if mode:
        result["architecture_mode"] = mode
    if export_binaries:
        result["binaries"] = binary_rows
    from research.gpu_sim_spike.sparse_units import publish_json

    publish_json(output, result)
    return result


def binding(path: Path, commit: str, repo: Path = ROOT) -> dict[str, str]:
    """Verify supervisor success/release and the hash-bound complete native export."""
    path = path.resolve(strict=True)
    receipt = json.loads(path.read_text())
    saved = path.parent / "manifest.json"
    if (
        receipt.get("status") != "PASS"
        or receipt.get("stop_verified") is not True
        or receipt.get("error")
        or receipt.get("cleanup_error")
        or receipt.get("failure_class")
        or receipt.get("commit") != commit
        or receipt.get("manifest_sha256") != digest(saved)
    ):
        raise ValueError("native compile requires same-pin PASS with verified stop")
    manifest = json.loads(saved.read_text())
    if (
        manifest.get("commit") != commit
        or manifest.get("script") != f"{PACKAGE}/native_compile.sh"
        or manifest.get("job_id") != receipt.get("job_id")
    ):
        raise ValueError("native compile manifest identity differs")
    terminal = receipt.get("terminal") or {}
    if terminal.get("returncode") != 0 or terminal.get("job_id") != receipt.get("job_id"):
        raise ValueError("native compile terminal is absent or failed")
    rows = receipt.get("exports")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("native compile export inventory is malformed")
    # runtime._exports inventories payloads only. Sidecars are transferred and
    # checked separately; they must never be invented as receipt inventory rows.
    exports = {row.get("path"): row for row in rows}
    if len(exports) != len(rows) or any(not isinstance(name, str) or not name for name in exports):
        raise ValueError("native compile export inventory is malformed")
    if "native-compile.json" not in exports:
        raise ValueError("native compile export inventory differs")
    if terminal.get("exports") != rows:
        raise ValueError("native compile terminal export inventory differs")
    for name in exports:
        payload = path.parent / "exports" / name
        row = exports.get(name, {})
        if (
            payload.is_symlink()
            or row.get("sha256") != digest(payload)
            or row.get("size") != payload.stat().st_size
        ):
            raise ValueError("native compile export inventory differs")
    report_path = path.parent / "exports/native-compile.json"
    from research.gpu_sim_spike.sparse_units import _verify

    _verify(report_path)
    report = json.loads(report_path.read_text())
    hashes, specs = inventory(repo, commit)
    try:
        report_mode, report_arches = report_architectures(report)
    except ValueError as exc:
        raise ValueError("native compile source/flags/architectures/PASS differs") from exc
    if manifest.get("env", {}).get(T4A_ARCH_MODE_ENV, "") != (report_mode or ""):
        raise ValueError("native compile manifest architecture mode differs")
    if (
        report_mode == T4A_A100_ARCH_MODE
        and manifest.get("env", {}).get("TORCH_CUDA_ARCH_LIST") != report_arches
    ):
        raise ValueError("native compile manifest architectures differ")
    if report_mode == T4A_A100_ARCH_MODE and manifest.get("requirements", {}) != {
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 15,
    }:
        raise ValueError("native compile repair A100 scope differs")
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "PASS"
        or report.get("commit") != commit
        or report.get("source_sha256") != hashes
        or report.get("extensions") != specs
        or report.get("kernels_launched") is not False
    ):
        raise ValueError("native compile source/flags/architectures/PASS differs")
    binaries = report.get("binaries")
    if report_mode == T4A_A100_ARCH_MODE and binaries is None:
        raise ValueError("native compile repair binary inventory is required")
    if binaries is not None:
        if not isinstance(binaries, list) or len(binaries) != len(specs):
            raise ValueError("native compile binary inventory differs")
        expected_names = {spec["name"] for spec in specs}
        reported_names = {row.get("name") for row in binaries if isinstance(row, dict)}
        if reported_names != expected_names:
            raise ValueError("native compile binary inventory differs")
        expected_paths = set()
        for row in binaries:
            if not isinstance(row, dict):
                raise ValueError("native compile binary inventory differs")
            name, binary_path = row.get("name"), row.get("path")
            expected = f"native-binaries/{name}.so"
            if (
                name not in expected_names
                or binary_path != expected
                or not isinstance(row.get("size"), int)
                or row["size"] <= 0
                or not isinstance(row.get("sha256"), str)
                or len(row["sha256"]) != 64
            ):
                raise ValueError("native compile binary inventory differs")
            expected_paths.add(expected)
            payload = path.parent / "exports" / expected
            export = exports.get(expected)
            if (
                export is None
                or payload.is_symlink()
                or export.get("sha256") != row["sha256"]
                or export.get("size") != row["size"]
                or digest(payload) != row["sha256"]
                or payload.stat().st_size != row["size"]
            ):
                raise ValueError("native compile binary payload differs")
            try:
                sidecar_digest = _verify(payload)
            except (OSError, ValueError) as exc:
                raise ValueError("native compile binary sidecars differ") from exc
            if sidecar_digest != row["sha256"]:
                raise ValueError("native compile binary sidecars differ")
        if set(exports) != {"native-compile.json", *expected_paths}:
            raise ValueError("native compile export inventory differs")
    elif set(exports) != {"native-compile.json"}:
        raise ValueError("native compile export inventory differs")
    return {"path": str(path), "sha256": digest(path)}


def require_for_a100_canary(manifest: dict[str, Any], repo: Path = ROOT) -> None:
    """Refuse H100/A100 GPU-sprint probes before any allocation side effects."""
    if not (
        (
            manifest.get("requirements", {}).get("gpu") in {"A100", "H100"}
            or any(
                gpu in manifest.get("requirements", {}).get("allowed_gpus", [])
                for gpu in ("A100", "H100")
            )
        )
        and manifest.get("canary", {}).get("mode") == "probe"
        and str(manifest.get("script", "")).startswith(PACKAGE + "/")
    ):
        return
    # The repair ruling permits exactly its A100 compile-only prerequisite to
    # remain a probe. It cannot be generalized into a canary bypass.
    admission = manifest.get("scientific_admission")
    commit = manifest.get("commit")
    repair_input = {
        "sha256": T4A_COMPILE_REPAIR_SHA256,
        "dest": "inputs/authority/t4a-compile-repair.md",
    }
    amendment_input = {
        "sha256": T4A_COMPILE_AMENDMENT_SHA256,
        "dest": "inputs/authority/JUDGMENT_T4A_COMPILE_AMENDMENT.md",
    }
    requirements = manifest.get("requirements", {})
    environment = manifest.get("env", {})
    if (
        manifest.get("script") == f"{PACKAGE}/native_compile.sh"
        and manifest.get("device") == "cpu"
        and environment.get(T4A_BINARY_ENV) == "1"
        and environment.get(T4A_ARCH_MODE_ENV) == T4A_A100_ARCH_MODE
        and environment.get("TORCH_CUDA_ARCH_LIST") == T4A_A100_ARCHES
        and requirements.get("gpu") == "A100"
        and requirements.get("allowed_gpus") == ["A100"]
        and requirements.get("est_minutes") == 15
        and manifest.get("timeout_minutes") == 30
        and isinstance(commit, str)
        and len(commit) == 40
        and manifest.get("job_id") == f"t4a-native-compile-a100-{commit[:12]}-repair01"
        and isinstance(manifest.get("resume_exemption"), dict)
        and manifest["resume_exemption"].get("kind") == "short-canary"
        and manifest["resume_exemption"].get("timeout_minutes") == 30
        and isinstance(admission, dict)
        and admission.get("amendment_sha256") == T4A_COMPILE_AMENDMENT_SHA256
        and admission.get("compile_repair_sha256") == T4A_COMPILE_REPAIR_SHA256
        and any(
            isinstance(row, dict)
            and row.get("sha256") == repair_input["sha256"]
            and row.get("dest") == repair_input["dest"]
            for row in manifest.get("inputs", [])
        )
        and any(
            isinstance(row, dict)
            and row.get("sha256") == amendment_input["sha256"]
            and row.get("dest") == amendment_input["dest"]
            for row in manifest.get("inputs", [])
        )
        and set(manifest.get("required_exports", []))
        == {
            "native-compile.json",
            *[
                f"native-binaries/{spec['name']}.so"
                for spec in inventory(repo, manifest["commit"])[1]
            ],
        }
    ):
        return
    proof = manifest.get("native_cuda_compile", {}).get("receipt", {})
    if not isinstance(proof, dict) or not proof.get("path") or not proof.get("sha256"):
        raise ValueError("H100/A100 canary requires native CUDA compile PASS receipt")
    if binding(Path(proof["path"]), manifest["commit"], repo) != proof:
        raise ValueError("native compile receipt SHA256 differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    run(args.output, args.commit)


if __name__ == "__main__":
    main()
