"""Bind T4A CUDA execution to the exact shared objects from its compile receipt.

This is intentionally a T4A runtime adapter.  Production extension loaders keep
their ordinary lazy-build behavior for every historical packet; a T4A packet
opts in through ``T4A_NATIVE_BINARIES=1`` and preloads these verified modules in
the same Python process before the benchmark imports its production paths.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from research.gpu_sim_spike import native_compile

REPORT_DEST = "inputs/native/native-compile.json"
REPORT_ENV = "T4A_NATIVE_REPORT"
REPORT_SHA_ENV = "T4A_NATIVE_REPORT_SHA256"

_LOADERS = {
    "snake_dqn_a0_event_v1": ("research.gpu_sim_spike.event_cuda", "_EXTENSION"),
    "snake_dqn_e1_flood_cuda_v2": ("research.gpu_sim_spike.flood_cuda", "_EXTENSION"),
    "snake_dqn_move_v1": ("research.gpu_sim_spike.move_cuda", "_EXTENSION"),
    "snake_dqn_v8_indexed_v1": ("research.gpu_sim_spike.v8_indexed", "_EXTENSION"),
    "snake_dqn_vector_geometry_exact_v2": (
        "research.gpu_sim_spike.vector_geometry",
        "_EXTENSION",
    ),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hex(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise RuntimeError(f"T4A native {label} must be a lowercase SHA-256")
    return value


def _report_rows(
    report: dict[str, Any], commit: str, *, pinned_sources: bool
) -> list[dict[str, Any]]:
    # The compile receipt is checked against Git on the Mac.  A runtime packet
    # is a sparse uploaded tree and need not contain its Git object database;
    # there we bind the report to its checked-out source bytes instead.
    hashes, specs = native_compile.inventory(
        native_compile.ROOT, commit if pinned_sources else None
    )
    if (
        report.get("schema") != native_compile.SCHEMA
        or report.get("status") != "PASS"
        or report.get("commit") != commit
        or report.get("source_sha256") != hashes
        or report.get("extensions") != specs
        or report.get("architectures") != native_compile.ARCHES
        or report.get("kernels_launched") is not False
        or not isinstance(report.get("torch"), str)
        or not isinstance(report.get("cuda"), str)
        or not isinstance(report.get("nvcc"), str)
        or not report["nvcc"].strip()
    ):
        raise RuntimeError("T4A native report source/toolchain/commit identity differs")
    rows = report.get("binaries")
    if not isinstance(rows, list) or len(rows) != len(specs):
        raise RuntimeError("T4A native report binary inventory is malformed")
    expected = {str(spec["name"]) for spec in specs}
    names = {row.get("name") for row in rows if isinstance(row, dict)}
    if names != expected or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("T4A native report binary names differ")
    for row in rows:
        name = row["name"]
        if (
            name not in _LOADERS
            or row.get("path") != f"native-binaries/{name}.so"
            or not isinstance(row.get("size"), int)
            or row["size"] <= 0
        ):
            raise RuntimeError("T4A native report binary descriptor differs")
        _require_hex(row.get("sha256"), "binary hash")
    return sorted(rows, key=lambda row: str(row["name"]))


def private_upload_inputs(proof: dict[str, Any], commit: str) -> list[dict[str, str]]:
    """Return immutable private upload rows for one verified compile receipt.

    The receipt itself remains Mac-side provenance.  Canary/main upload the
    report and shared objects, whose hashes are already sealed by that receipt.
    """
    if native_compile.binding(Path(str(proof.get("path", ""))), commit) != proof:
        raise RuntimeError("T4A native compile receipt binding differs")
    receipt = Path(proof["path"])
    report = receipt.parent / "exports/native-compile.json"
    rows = _report_rows(json.loads(report.read_text()), commit, pinned_sources=True)
    output = [
        {
            "path": str(report),
            "sha256": _digest(report),
            "dest": REPORT_DEST,
        }
    ]
    for row in rows:
        binary = receipt.parent / "exports" / str(row["path"])
        if binary.is_symlink() or not binary.is_file() or _digest(binary) != row["sha256"]:
            raise RuntimeError("T4A native compile binary export differs")
        output.append(
            {
                "path": str(binary),
                "sha256": str(row["sha256"]),
                "dest": "inputs/native/" + str(row["path"]),
            }
        )
    return output


def runtime_environment(report_sha256: str) -> dict[str, str]:
    """Return the fixed environment pointers consumed by the T4A shell entrypoint."""
    return {
        native_compile.T4A_BINARY_ENV: "1",
        REPORT_ENV: REPORT_DEST,
        REPORT_SHA_ENV: _require_hex(report_sha256, "report hash"),
    }


def _load_extension(name: str, path: Path) -> Any:
    if name in sys.modules:
        existing = sys.modules[name]
        if Path(str(getattr(existing, "__file__", ""))).resolve() != path.resolve():
            raise RuntimeError(f"T4A native extension name collision: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"T4A native binary is not importable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def preload_from_environment(commit: str) -> dict[str, str]:
    """Verify and install receipt binaries into production loader globals in-process."""
    if os.environ.get(native_compile.T4A_BINARY_ENV) != "1":
        raise RuntimeError("T4A native binary mode is required for CUDA execution")
    root = Path(os.environ.get("COLAB_INPUT_DIR", "")).resolve()
    relative = os.environ.get(REPORT_ENV)
    if not root.is_dir() or relative != REPORT_DEST:
        raise RuntimeError("T4A native report input location differs")
    report_path = root / relative
    if report_path.is_symlink() or not report_path.is_file():
        raise RuntimeError("T4A native report is absent")
    if _digest(report_path) != _require_hex(os.environ.get(REPORT_SHA_ENV), "report hash"):
        raise RuntimeError("T4A native report hash differs")
    report = json.loads(report_path.read_text())
    rows = _report_rows(report, commit, pinned_sources=False)
    import torch

    if report.get("torch") != torch.__version__ or report.get("cuda") != torch.version.cuda:
        raise RuntimeError("T4A native runtime toolchain differs from compile receipt")
    loaded: dict[str, str] = {}
    for row in rows:
        name, relative_binary = str(row["name"]), str(row["path"])
        binary = root / "inputs/native" / relative_binary
        if binary.is_symlink() or not binary.is_file() or binary.stat().st_size != row["size"]:
            raise RuntimeError(f"T4A native binary differs: {name}")
        if _digest(binary) != row["sha256"]:
            raise RuntimeError(f"T4A native binary hash differs: {name}")
        extension = _load_extension(name, binary)
        module_name, attribute = _LOADERS[name]
        production = __import__(module_name, fromlist=[attribute])
        setattr(production, attribute, extension)
        if hasattr(production, "_ERROR"):
            setattr(production, "_ERROR", None)
        loaded[name] = str(row["sha256"])
    return loaded
