"""Source/toolchain-bound native cache transport; never allocates a CUDA runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike.native_compile import ROOT, inventory
from research.gpu_sim_spike.t4a_native import _LOADERS, _load_extension
from research.spatial_student_pilot.resume import _publish_bytes

NAME = "native-cache.json"
SCHEMA = "spatial-native-cache/v1"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate(report: dict[str, Any], *, commit: str | None = None) -> None:
    hashes, specs = inventory(ROOT, commit)
    expected = {s["name"] for s in specs if s["name"] != "snake_dqn_v8_indexed_v1"}
    if (
        report.get("schema") != SCHEMA
        or report.get("source_sha256") != hashes
        or report.get("extensions") != specs
        or {r["name"] for r in report.get("binaries", [])} != expected
        or len(report["binaries"]) != len(expected)
        or report.get("platform") != "linux-x86_64"
        or not str(report.get("torch", "")).startswith("2.9.1+")
        or report.get("cuda") != "12.6"
        or report.get("architecture") not in {"8.0", "9.0"}
    ):
        raise ValueError("native cache source/architecture/inventory differs")
    for row in report["binaries"]:
        if (
            row["path"] != "native/" + row["name"] + ".so"
            or type(row["size"]) is not int
            or not 0 < row["size"] <= 32 * 1024**2
        ):
            raise ValueError("invalid native cache binary")


def upload_inputs(receipt_path: Path, commit: str) -> list[dict[str, str]]:
    """Admit source-identical binaries from stopped, hash-verified prior exports.

    This deliberately does not weaken native_compile.binding's same-pin contract.
    Cache reuse binds native source/flags equality rather than gameplay identity;
    every source commit, original receipt and binary digest remains in provenance.
    """
    receipt = json.loads(receipt_path.read_text())
    manifest = receipt_path.parent / "manifest.json"
    if (
        receipt.get("stop_verified") is not True
        or receipt.get("cleanup_error")
        or not manifest.is_file()
        or digest(manifest) != receipt.get("manifest_sha256")
        or json.loads(manifest.read_text())["commit"] != receipt.get("commit")
    ):
        raise ValueError("native cache requires a verified stopped supervisor receipt")
    exports = {r["path"]: r for r in receipt["exports"]}
    report_path = receipt_path.parent / "exports/spatial" / NAME
    report = json.loads(report_path.read_text())
    _validate(report, commit=commit)
    if report.get("source_commit") != receipt.get("commit"):
        raise ValueError("native cache original source commit differs from receipt")
    rows = [(NAME, digest(report_path)), *[(r["path"], r["sha256"]) for r in report["binaries"]]]
    inputs = []
    for relative, expected in rows:
        path = report_path.parent / relative
        entry = exports.get("spatial/" + relative, {})
        if (
            path.is_symlink()
            or not path.is_file()
            or digest(path) != expected
            or entry.get("sha256") != expected
            or entry.get("size") != path.stat().st_size
        ):
            raise ValueError("native cache lacks verified binary export binding")
        inputs.append(
            dict(path=str(path.resolve()), dest="native-cache/" + relative, sha256=expected)
        )
    return inputs


def preload(input_dir: Path) -> bool:
    """Install checked shared objects before any lazy production loader runs."""
    relative = os.environ.get("SPATIAL_NATIVE_CACHE_SHA256")
    if not relative:
        return False
    architecture = "%d.%d" % torch.cuda.get_device_capability()
    cache_hashes = json.loads(relative)
    if architecture not in cache_hashes:
        # A packet can bind the available A100 cache while H100 stays cold.
        # Never load a different architecture's binary as a fallback.
        return False
    relative = cache_hashes[architecture]
    root = input_dir / "native-cache" / architecture
    report_path = root / NAME
    if digest(report_path) != relative:
        raise ValueError("native cache report digest differs")
    report = json.loads(report_path.read_text())
    _validate(report)
    architecture = "%d.%d" % torch.cuda.get_device_capability()
    if (
        report["architecture"] != architecture
        or report["torch"] != torch.__version__
        or report["cuda"] != torch.version.cuda
    ):
        raise ValueError("native cache toolchain or GPU differs")
    for row in report["binaries"]:
        path = root / row["path"]
        if path.is_symlink() or path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError("native cache binary digest differs")
        module_name, attr = _LOADERS[row["name"]]
        setattr(importlib.import_module(module_name), attr, _load_extension(row["name"], path))
    return True


def export_cache(output: Path) -> dict[str, Any]:
    """Retain successfully loaded binaries for a later identically sourced packet."""
    hashes, specs = inventory(ROOT)
    binaries = []
    for spec in specs:
        name = spec["name"]
        if name == "snake_dqn_v8_indexed_v1":
            continue
        module_name, attr = _LOADERS[name]
        extension = getattr(sys.modules[module_name], attr)
        source = Path(extension.__file__)
        relative = "native/" + name + ".so"
        payload = source.read_bytes()
        _publish_bytes(output / relative, payload)
        sha = hashlib.sha256(payload).hexdigest()
        _publish_bytes(output / (relative + ".sha256"), (sha + "\n").encode())
        _publish_bytes(output / (relative + ".complete"), b"complete\n")
        binaries.append(dict(name=name, path=relative, size=len(payload), sha256=sha))
    report = dict(
        schema=SCHEMA,
        source_sha256=hashes,
        extensions=specs,
        binaries=binaries,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        platform="linux-x86_64",
        architecture="%d.%d" % torch.cuda.get_device_capability(),
        source_commit=os.environ.get("SPATIAL_SOURCE_COMMIT"),
    )
    raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    _publish_bytes(output / NAME, raw)
    _publish_bytes(output / (NAME + ".sha256"), (hashlib.sha256(raw).hexdigest() + "\n").encode())
    _publish_bytes(output / (NAME + ".complete"), b"complete\n")
    return dict(bytes=sum(r["size"] for r in binaries), binaries=len(binaries))
