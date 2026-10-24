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
import time
from pathlib import Path
from typing import Any

PACKAGE = "research/gpu_sim_spike"
ROOT = Path(__file__).resolve().parents[2]
ARCHES = "8.0;8.9"
SCHEMA = "gpu-native-compile/v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    os.environ["TORCH_CUDA_ARCH_LIST"] = ARCHES
    os.environ["MAX_JOBS"] = "2"
    hashes, specs = inventory(ROOT)
    build = output.parent.parent / "native-compile.build"
    build.mkdir(parents=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(build)
    began = time.monotonic()
    for spec in specs:
        print(f"Compiling {spec['name']} targets={ARCHES}", flush=True)
        load(
            **{key: value for key, value in spec.items() if key != "sources"},
            sources=[str(ROOT / name) for name in spec["sources"]],
            verbose=True,
        )
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "commit": commit,
        "architectures": ARCHES,
        "source_sha256": hashes,
        "extensions": specs,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "elapsed_seconds": time.monotonic() - began,
        "kernels_launched": False,
    }
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
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("native compile export inventory is malformed")
    # runtime._exports inventories payloads only. Sidecars are transferred and
    # checked separately; they must never be invented as receipt inventory rows.
    exports = {row.get("path"): row for row in rows}
    if set(exports) != {"native-compile.json"}:
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
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "PASS"
        or report.get("commit") != commit
        or report.get("source_sha256") != hashes
        or report.get("extensions") != specs
        or report.get("architectures") != ARCHES
        or report.get("kernels_launched") is not False
    ):
        raise ValueError("native compile source/flags/architectures/PASS differs")
    return {"path": str(path), "sha256": digest(path)}


def require_for_a100_canary(manifest: dict[str, Any], repo: Path = ROOT) -> None:
    """Refuse A100 GPU-sprint probes before any allocation side effects."""
    if not (
        (
            manifest.get("requirements", {}).get("gpu") == "A100"
            or "A100" in manifest.get("requirements", {}).get("allowed_gpus", [])
        )
        and manifest.get("canary", {}).get("mode") == "probe"
        and str(manifest.get("script", "")).startswith(PACKAGE + "/")
    ):
        return
    proof = manifest.get("native_cuda_compile", {}).get("receipt", {})
    if not isinstance(proof, dict) or not proof.get("path") or not proof.get("sha256"):
        raise ValueError("A100 canary requires native CUDA compile PASS receipt")
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
