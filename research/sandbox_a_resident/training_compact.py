"""Lossless compact transport for one verified TRAIN-01 full-state snapshot.

The train01-11 post-preparation snapshot ``iteration-0000.pt`` is 1.33 GB,
almost all of it padded and mostly-zero simulator/opponent ring buffers, and
cannot be staged within the qualified transport's fixed whole-staging upload
cap.  The derivative is one XZ stream (LZMA2 preset 6, SHA-256 integrity check)
of the *exact original file bytes*.  At runtime it is decoded into a
byte-identical ``iteration-0000.pt`` (refusing on any size or SHA-256 mismatch),
the ordinary ledger verification and ``torch.load`` resume path consume that
file, and every decoded tensor plus the canonical non-tensor structure are then
checked against a Mac-built manifest of the original decoded state before any
state is restored.  Any mismatch refuses; nothing is retried or repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from research.sandbox_a_resident.training_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_SCHEMA_V2,
    unit_id,
    validate_compact_resume,
    validate_compact_source_packet,
)
from research.sandbox_a_resident.units import digest, publish

MANIFEST_SCHEMA = "sandbox-a-train-compact-snapshot/v1"
COMPACT_SNAPSHOT_EXPORT_SCHEMA = "sandbox-a-train-compact-export/v1"
XZ_PRESET = 6
CODEC = {
    "container": "xz",
    "filter": "lzma2",
    "preset": XZ_PRESET,
    "check": "sha256",
    "streams": 1,
    "scope": "exact original file bytes",
}
# Decoder memory guard: preset 6 needs about 9 MiB.
DECODER_MEMLIMIT = 64 * 1024 * 1024
READ_BYTES = 1024 * 1024
DECODE_BLOCK_BYTES = 8 * 1024 * 1024
Progress = Callable[[str, Mapping[str, Any]], None]


class _PublicationProgress:
    """Report completed byte work, never periodic claims from a timer thread."""

    def __init__(self, callback: Progress | None) -> None:
        self.callback = callback
        self.last = time.monotonic()
        self.stage: str | None = None

    def __call__(self, stage: str, values: Mapping[str, Any]) -> None:
        now = time.monotonic()
        if self.callback is not None and (stage != self.stage or now - self.last >= 30):
            self.callback(stage, values)
            self.last = now
            self.stage = stage


class CompactResumeRefused(ValueError):
    """A compact derivative, its manifest or the decoded state fails verification."""


def _tensor_sha256(tensor: torch.Tensor, on_progress: Progress | None = None) -> str:
    flat = tensor.detach().contiguous().reshape(-1)
    raw = memoryview(flat.view(torch.uint8).numpy())
    hasher = hashlib.sha256()
    for offset in range(0, len(raw), READ_BYTES):
        block = raw[offset : offset + READ_BYTES]
        hasher.update(block)
        if on_progress is not None:
            on_progress("snapshot-tensor-hash", {"tensor_bytes_done": offset + len(block)})
    return hasher.hexdigest()


def describe_state(state: Any, on_progress: Progress | None = None) -> dict[str, Any]:
    """Return per-tensor SHA-256 records and a canonical structure digest.

    The walk is deterministic (insertion order) and refuses unsupported types,
    so two equal descriptions imply equal tensors (dtype, shape, stride,
    storage offset, autograd flag, bytes) and equal non-tensor structure.
    """
    tensors: list[dict[str, Any]] = []

    def walk(value: Any, path: tuple[Any, ...]) -> Any:
        kind = type(value)
        if kind is torch.Tensor:
            if value.device.type != "cpu":
                raise CompactResumeRefused("decoded tensors must be CPU-mapped")
            tensors.append(
                {
                    "path": list(path),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "storage_offset": int(value.storage_offset()),
                    "requires_grad": bool(value.requires_grad),
                    "nbytes": int(value.numel() * value.element_size()),
                    "sha256": _tensor_sha256(value, on_progress),
                }
            )
            return ["tensor", len(tensors) - 1]
        if kind is dict or kind is OrderedDict:
            items = []
            for key, item in value.items():
                if type(key) not in (str, int):
                    raise CompactResumeRefused(f"unsupported state key type at {list(path)}")
                items.append(
                    [["str" if type(key) is str else "int", key], walk(item, (*path, key))]
                )
            return ["dict" if kind is dict else "odict", items]
        if kind is list or kind is tuple:
            return [
                "list" if kind is list else "tuple",
                [walk(item, (*path, index)) for index, item in enumerate(value)],
            ]
        if kind is bool:
            return ["bool", value]
        if kind is int:
            return ["int", str(value)]
        if kind is float:
            return ["float", value.hex()]
        if kind is str:
            return ["str", value]
        if value is None:
            return ["none"]
        if kind is np.ndarray:
            array = np.ascontiguousarray(value)
            return [
                "ndarray",
                array.dtype.str,
                list(array.shape),
                hashlib.sha256(array.tobytes()).hexdigest(),
            ]
        raise CompactResumeRefused(f"unsupported state type {kind.__qualname__} at {list(path)}")

    skeleton = walk(state, ())
    encoded = json.dumps(skeleton, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {
        "structure_sha256": hashlib.sha256(encoded).hexdigest(),
        "tensor_count": len(tensors),
        "tensor_bytes": sum(row["nbytes"] for row in tensors),
        "tensors": tensors,
    }


def decode_stream(
    derivative: Path,
    sink: Callable[[bytes], Any],
    *,
    expected_size: int,
    expected_sha256: str,
    on_progress: Progress | None = None,
) -> None:
    """Stream-decode one XZ stream, bounding output and verifying size and SHA-256."""
    if type(expected_size) is not int or expected_size < 1:
        raise CompactResumeRefused("expected decoded size must be a positive integer")
    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=DECODER_MEMLIMIT)
    hasher = hashlib.sha256()
    total = 0
    try:
        with derivative.open("rb") as source:
            while not decompressor.eof:
                if decompressor.needs_input:
                    chunk = source.read(READ_BYTES)
                    if not chunk:
                        raise CompactResumeRefused("truncated compact snapshot stream")
                else:
                    chunk = b""
                block = decompressor.decompress(chunk, max_length=DECODE_BLOCK_BYTES)
                total += len(block)
                if total > expected_size:
                    raise CompactResumeRefused("decoded snapshot exceeds its original size")
                hasher.update(block)
                sink(block)
                if on_progress is not None and block:
                    on_progress("snapshot-roundtrip", {"decoded_bytes": total})
            if decompressor.unused_data or source.read(1):
                raise CompactResumeRefused("trailing bytes after the compact snapshot stream")
    except lzma.LZMAError as error:
        raise CompactResumeRefused(f"corrupt compact snapshot stream: {error}") from error
    if total != expected_size or hasher.hexdigest() != expected_sha256:
        raise CompactResumeRefused("decoded snapshot differs from the original size/SHA-256")


def _compress(original: Path, destination: Path, on_progress: Progress | None = None) -> None:
    compressor = lzma.LZMACompressor(
        format=lzma.FORMAT_XZ, check=lzma.CHECK_SHA256, preset=XZ_PRESET
    )
    with original.open("rb") as source, destination.open("xb") as target:
        total = 0
        for block in iter(lambda: source.read(16 * READ_BYTES), b""):
            target.write(compressor.compress(block))
            total += len(block)
            if on_progress is not None:
                on_progress("snapshot-compress", {"original_bytes_done": total})
        target.write(compressor.flush())
        target.flush()
        os.fsync(target.fileno())


def _verified_original(original: Path) -> tuple[int, str]:
    sidecar = Path(str(original) + ".sha256")
    complete = Path(str(original) + ".complete")
    if not (original.is_file() and sidecar.is_file() and complete.is_file()):
        raise CompactResumeRefused(f"incomplete original snapshot: {original.name}")
    if any(path.is_symlink() for path in (original, sidecar, complete)):
        raise CompactResumeRefused(f"symlinked original snapshot: {original.name}")
    expected = sidecar.read_text().strip()
    if digest(original) != expected:
        raise CompactResumeRefused(f"corrupt original snapshot: {original.name}")
    return original.stat().st_size, expected


def _infer_iteration(original: Path, iteration: int | None) -> int | None:
    match = re.fullmatch(r"iteration-([0-9]{4})\.pt", original.name)
    inferred = int(match.group(1)) if match is not None else None
    if iteration is None:
        return inferred
    if type(iteration) is not int or iteration < 0:
        raise CompactResumeRefused("iteration must be a nonnegative integer")
    if inferred is not None and inferred != iteration:
        raise CompactResumeRefused("snapshot filename iteration differs from requested iteration")
    return iteration


def encode_snapshot(
    original: Path,
    *,
    unit: str | None = None,
    source_packet_id: str | None = None,
    iteration: int | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Create-only Mac encoding: derivative and manifest, each with completion sidecars.

    The derivative is fully decoded and verified against the original bytes
    before the manifest (of the original file's decoded tensors) is published.
    """
    resolved_iteration = _infer_iteration(original, iteration)
    if unit is None:
        if resolved_iteration is None:
            raise CompactResumeRefused("unit or canonical iteration-NNNN.pt filename required")
        resolved_unit = unit_id(resolved_iteration)
    else:
        if resolved_iteration is not None and unit != unit_id(resolved_iteration):
            raise CompactResumeRefused("unit differs from canonical snapshot iteration")
        resolved_unit = unit
    if source_packet_id is not None:
        if resolved_iteration is None:
            raise CompactResumeRefused("source_packet_id requires a canonical snapshot iteration")
        try:
            validate_compact_source_packet(source_packet_id, resolved_iteration)
        except ValueError as error:
            raise CompactResumeRefused(str(error)) from error
    size, sha256 = _verified_original(original)
    derivative = Path(str(original) + COMPACT_DERIVATIVE_SUFFIX)
    manifest_path = Path(str(original) + COMPACT_MANIFEST_SUFFIX)
    if manifest_path.exists() or Path(str(manifest_path) + ".partial").exists():
        raise FileExistsError(manifest_path)
    encoded = publish(derivative, lambda path: _compress(original, path))
    decode_stream(derivative, lambda block: None, expected_size=size, expected_sha256=sha256)
    state = torch.load(original, map_location="cpu", weights_only=False)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "unit": resolved_unit,
        "codec": CODEC,
        "original": {"name": original.name, "size": size, "sha256": sha256},
        "derivative": {
            "name": derivative.name,
            "size": encoded["size"],
            "sha256": encoded["sha256"],
        },
        **describe_state(state),
    }
    if source_packet_id is not None:
        assert resolved_iteration is not None
        manifest["source_packet_id"] = source_packet_id
        manifest["iteration"] = resolved_iteration
    described = publish(
        manifest_path,
        lambda path: path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n"),
    )
    result: dict[str, Any] = {"derivative": encoded, "manifest": described}
    if source_packet_id is not None:
        assert resolved_iteration is not None
        runtime_stem = f"resume/iteration-{resolved_iteration:04d}.pt"
        allowlist_entry: dict[str, Any] = {
            "source_packet_id": source_packet_id,
            "iteration": resolved_iteration,
            "unit": resolved_unit,
            "original": {"path": runtime_stem, "size": size, "sha256": sha256},
            "derivative": {
                "path": runtime_stem + COMPACT_DERIVATIVE_SUFFIX,
                "size": encoded["size"],
                "sha256": encoded["sha256"],
            },
            "manifest": {
                "path": runtime_stem + COMPACT_MANIFEST_SUFFIX,
                "size": described["size"],
                "sha256": described["sha256"],
            },
        }
        if source_commit is not None:
            allowlist_entry["source_commit"] = source_commit
        result["allowlist_entry"] = allowlist_entry
    return result


def _safe_input(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts or raw.parts[0] != "resume":
        raise CompactResumeRefused("unsafe compact resume runtime path")
    return root / raw


def _bound_file(path: Path, descriptor: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise CompactResumeRefused(f"compact resume input missing: {path.name}")
    if path.stat().st_size != descriptor["size"] or digest(path) != descriptor["sha256"]:
        raise CompactResumeRefused(f"compact resume input differs from its binding: {path.name}")


def load_manifest(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a manifest only after its bytes and declarations match the binding."""
    _bound_file(path, entry["manifest"])
    manifest = json.loads(path.read_text())
    original, derivative = entry["original"], entry["derivative"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("unit") != entry["unit"]
        or manifest.get("codec") != CODEC
        or manifest.get("original")
        != {
            "name": Path(original["path"]).name,
            "size": original["size"],
            "sha256": original["sha256"],
        }
        or manifest.get("derivative")
        != {
            "name": Path(derivative["path"]).name,
            "size": derivative["size"],
            "sha256": derivative["sha256"],
        }
        or not isinstance(manifest.get("tensors"), list)
        or manifest.get("tensor_count") != len(manifest["tensors"])
    ):
        raise CompactResumeRefused("compact snapshot manifest does not match its binding")
    if entry.get("schema") == COMPACT_RESUME_SCHEMA_V2:
        legacy_train01_11 = (
            entry.get("source_packet_id") == "packet-train01-11"
            and entry.get("iteration") == 0
            and "source_packet_id" not in manifest
            and "iteration" not in manifest
        )
        source_packet_mismatch = "source_packet_id" in manifest and manifest.get(
            "source_packet_id"
        ) != entry.get("source_packet_id")
        if not legacy_train01_11 and (
            source_packet_mismatch or manifest.get("iteration") != entry.get("iteration")
        ):
            raise CompactResumeRefused(
                "compact snapshot manifest source_packet_id/iteration differs from its binding"
            )
    return manifest


def materialize_compact_resume(
    ledger_payload: Mapping[str, Any],
    input_root: Path,
    *,
    allowlist_path: Path | None = None,
) -> dict[str, Any]:
    """Decode the allowlisted derivative into its byte-identical original, create-only.

    Returns the verified manifest; the caller must pass the loaded state to
    :func:`verify_decoded_state` before restoring anything.
    """
    entry = validate_compact_resume(dict(ledger_payload), allowlist_path=allowlist_path)
    if entry is None:
        raise CompactResumeRefused("ledger has no compact resume binding")
    derivative = _safe_input(input_root, entry["derivative"]["path"])
    manifest_path = _safe_input(input_root, entry["manifest"]["path"])
    target = _safe_input(input_root, entry["original"]["path"])
    partial = target.with_name(target.name + ".partial")
    sidecar = Path(str(target) + ".sha256")
    complete = Path(str(target) + ".complete")
    if target.exists() or target.is_symlink() or partial.exists() or partial.is_symlink():
        raise CompactResumeRefused("decoded original already present; refusing to overwrite")
    if sidecar.is_symlink() or complete.is_symlink():
        raise CompactResumeRefused("symlinked compact resume sidecar")
    if sidecar.exists() != complete.exists():
        raise CompactResumeRefused("incomplete pre-staged compact resume sidecars")
    if sidecar.exists() and sidecar.read_text().strip() != entry["original"]["sha256"]:
        raise CompactResumeRefused("pre-staged sidecar differs from compact resume binding")
    _bound_file(derivative, entry["derivative"])
    manifest = load_manifest(manifest_path, entry)
    with partial.open("xb") as stream:
        try:
            decode_stream(
                derivative,
                stream.write,
                expected_size=entry["original"]["size"],
                expected_sha256=entry["original"]["sha256"],
            )
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            partial.unlink()
            raise
    # Hard-link publication refuses a racing writer, unlike overwrite-capable rename.
    os.link(partial, target)
    partial.unlink()
    if not sidecar.exists():
        sidecar.write_text(entry["original"]["sha256"] + "\n")
        complete.write_text("complete\n")
    return manifest


def publish_compact_snapshot(
    output: Path,
    iteration: int,
    writer: Callable[[Path], None],
    *,
    metadata: Mapping[str, Any],
    source_packet_id: str | None = None,
    on_progress: Progress | None = None,
) -> dict[str, Any]:
    """Publish a lossless compact periodic snapshot at the source without writing raw ``.pt``.

    The temporary raw ``.pt`` is serialized outside ``output`` (in system ``/tmp``)
    so runtime ``output.rglob('*.complete')`` export scanners never observe scratch
    files or transient subdirectories.  The published ``.pt.xz`` stream is fully
    round-trip decoded and checked against the raw size and SHA-256 before the
    per-tensor ``.pt.xz.manifest`` and completion-last ``.json`` envelope are
    published.  CPU/CUDA RNG state is preserved verbatim.
    """
    if type(iteration) is not int or iteration < 0:
        raise CompactResumeRefused("iteration must be a nonnegative integer")
    progress = _PublicationProgress(on_progress)
    identity = unit_id(iteration)
    if source_packet_id is not None:
        try:
            validate_compact_source_packet(source_packet_id, iteration)
        except ValueError as error:
            raise CompactResumeRefused(str(error)) from error
    output.mkdir(parents=True, exist_ok=True)
    stem = f"iteration-{iteration:04d}"
    raw_target = output / f"{stem}.pt"
    derivative_path = output / f"{stem}.pt{COMPACT_DERIVATIVE_SUFFIX}"
    manifest_path = output / f"{stem}.pt{COMPACT_MANIFEST_SUFFIX}"
    info_path = output / f"{stem}.json"
    for candidate in (raw_target, derivative_path, manifest_path, info_path):
        for suffix in ("", ".partial", ".sha256", ".complete"):
            probe = Path(f"{candidate}{suffix}")
            if probe.exists() or probe.is_symlink():
                raise FileExistsError(probe)

    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else None
    )
    try:
        with tempfile.TemporaryDirectory(prefix=f"train01-compact-{stem}-") as scratch_dir:
            scratch_pt = Path(scratch_dir) / f"{stem}.pt"
            writer(scratch_pt)
            orig_size = scratch_pt.stat().st_size
            progress("snapshot-serialized", {"original_bytes": orig_size})
            hasher = hashlib.sha256()
            with scratch_pt.open("rb") as stream:
                total = 0
                for block in iter(lambda: stream.read(READ_BYTES), b""):
                    hasher.update(block)
                    total += len(block)
                    progress("snapshot-file-hash", {"original_bytes_done": total})
            orig_sha256 = hasher.hexdigest()
            loaded_state = torch.load(scratch_pt, map_location="cpu", weights_only=False)
            progress("snapshot-loaded", {"original_bytes": orig_size})
            state_description = describe_state(loaded_state, progress)
            del loaded_state
            encoded = publish(derivative_path, lambda dst: _compress(scratch_pt, dst, progress))
            progress("snapshot-compressed", {"derivative_bytes": encoded["size"]})
            decode_stream(
                derivative_path,
                lambda _block: None,
                expected_size=orig_size,
                expected_sha256=orig_sha256,
                on_progress=progress,
            )
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    manifest_payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "unit": identity,
        "iteration": iteration,
        "codec": CODEC,
        "original": {"name": f"{stem}.pt", "size": orig_size, "sha256": orig_sha256},
        "derivative": {
            "name": derivative_path.name,
            "size": encoded["size"],
            "sha256": encoded["sha256"],
        },
        **state_description,
    }
    if source_packet_id is not None:
        manifest_payload["source_packet_id"] = source_packet_id
    described = publish(
        manifest_path,
        lambda dst: dst.write_text(json.dumps(manifest_payload, indent=1, sort_keys=True) + "\n"),
    )
    original_descriptor = {
        "id": identity,
        "path": str(raw_target.resolve()),
        "size": orig_size,
        "sha256": orig_sha256,
    }
    compact_descriptor: dict[str, Any] = {
        "schema": COMPACT_SNAPSHOT_EXPORT_SCHEMA,
        "codec": CODEC,
        "original": {"name": f"{stem}.pt", "size": orig_size, "sha256": orig_sha256},
        "derivative": {
            "name": derivative_path.name,
            "size": encoded["size"],
            "sha256": encoded["sha256"],
        },
        "manifest": {
            "name": manifest_path.name,
            "size": described["size"],
            "sha256": described["sha256"],
        },
    }
    if source_packet_id is not None:
        compact_descriptor["source_packet_id"] = source_packet_id
    envelope = {
        "schema": "sandbox-a-train-snapshot/v1",
        "id": identity,
        "iteration": iteration,
        "snapshot": original_descriptor,
        "compact_snapshot": compact_descriptor,
        "metadata": dict(metadata),
    }
    info = publish(
        info_path,
        lambda dst: dst.write_text(json.dumps(envelope, sort_keys=True) + "\n"),
    )
    return {
        "id": identity,
        "iteration": iteration,
        "exports": [encoded, described, info],
        "original": original_descriptor,
        "compact_snapshot": compact_descriptor,
    }


def _verified_export_file(path: Path) -> tuple[int, str]:
    sidecar = Path(f"{path}.sha256")
    complete = Path(f"{path}.complete")
    if not (path.is_file() and sidecar.is_file() and complete.is_file()):
        raise CompactResumeRefused(f"incomplete compact snapshot export: {path.name}")
    if any(item.is_symlink() for item in (path, sidecar, complete)):
        raise CompactResumeRefused(f"symlinked compact snapshot export: {path.name}")
    expected = sidecar.read_text().strip().split()[0]
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise CompactResumeRefused(f"malformed SHA-256 sidecar: {path.name}")
    if digest(path) != expected:
        raise CompactResumeRefused(f"corrupt compact snapshot export: {path.name}")
    return path.stat().st_size, expected


def verify_compact_snapshot_export(
    export_root: Path,
    iteration: int,
    *,
    source_packet_id: str | None = None,
    source_commit: str | None = None,
    require_actor: bool = True,
) -> dict[str, Any]:
    """Verify a compact-at-source (or Mac-encoded) snapshot export end-to-end on Mac.

    Checks completion-last sidecars, envelope bindings, and manifest declarations,
    then stream-decodes ``iteration-NNNN.pt.xz`` to a temporary ``/tmp`` file
    verifying exact decoded size and SHA-256, loads CPU-mapped tensors, and runs
    :func:`verify_decoded_state` across every tensor and non-tensor structure.
    """
    if type(iteration) is not int or iteration < 0:
        raise CompactResumeRefused("iteration must be a nonnegative integer")
    if export_root.is_symlink() or not export_root.is_dir():
        raise CompactResumeRefused(f"export root is missing or symlinked: {export_root}")
    if source_packet_id is not None:
        try:
            validate_compact_source_packet(source_packet_id, iteration)
        except ValueError as error:
            raise CompactResumeRefused(str(error)) from error
    identity = unit_id(iteration)
    stem = f"iteration-{iteration:04d}"
    derivative_path = export_root / f"{stem}.pt{COMPACT_DERIVATIVE_SUFFIX}"
    manifest_path = export_root / f"{stem}.pt{COMPACT_MANIFEST_SUFFIX}"
    info_path = export_root / f"{stem}.json"
    actor_path = export_root / f"{stem}-actor.pth"

    deriv_size, deriv_sha = _verified_export_file(derivative_path)
    man_size, man_sha = _verified_export_file(manifest_path)
    info_size, info_sha = _verified_export_file(info_path)
    actor_desc: dict[str, Any] | None = None
    if require_actor:
        actor_size, actor_sha = _verified_export_file(actor_path)
        actor_desc = {"name": actor_path.name, "size": actor_size, "sha256": actor_sha}

    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CompactResumeRefused(f"unreadable snapshot envelope: {info_path.name}") from error
    if (
        not isinstance(info, dict)
        or info.get("schema") != "sandbox-a-train-snapshot/v1"
        or info.get("id") != identity
        or info.get("iteration") != iteration
    ):
        raise CompactResumeRefused(f"malformed snapshot envelope: {info_path.name}")
    snapshot = info.get("snapshot")
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("id") != identity
        or type(snapshot.get("size")) is not int
        or snapshot["size"] < 1
        or not isinstance(snapshot.get("sha256"), str)
        or len(snapshot["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in snapshot["sha256"])
    ):
        raise CompactResumeRefused(f"malformed snapshot original descriptor: {info_path.name}")
    orig_size = int(snapshot["size"])
    orig_sha = str(snapshot["sha256"])

    compact_header = info.get("compact_snapshot")
    if compact_header is not None:
        if (
            not isinstance(compact_header, dict)
            or compact_header.get("schema") != COMPACT_SNAPSHOT_EXPORT_SCHEMA
            or compact_header.get("codec") != CODEC
            or compact_header.get("original")
            != {"name": f"{stem}.pt", "size": orig_size, "sha256": orig_sha}
            or compact_header.get("derivative")
            != {"name": derivative_path.name, "size": deriv_size, "sha256": deriv_sha}
            or compact_header.get("manifest")
            != {"name": manifest_path.name, "size": man_size, "sha256": man_sha}
        ):
            raise CompactResumeRefused(
                f"compact_snapshot envelope disagrees with exported files: {info_path.name}"
            )
        if (
            source_packet_id is not None
            and "source_packet_id" in compact_header
            and compact_header["source_packet_id"] != source_packet_id
        ):
            raise CompactResumeRefused(
                f"compact_snapshot source_packet_id differs from {source_packet_id!r}"
            )

    runtime_stem = f"resume/{stem}.pt"
    binding: dict[str, Any] = {
        "unit": identity,
        "iteration": iteration,
        "original": {"path": runtime_stem, "size": orig_size, "sha256": orig_sha},
        "derivative": {
            "path": runtime_stem + COMPACT_DERIVATIVE_SUFFIX,
            "size": deriv_size,
            "sha256": deriv_sha,
        },
        "manifest": {
            "path": runtime_stem + COMPACT_MANIFEST_SUFFIX,
            "size": man_size,
            "sha256": man_sha,
        },
    }
    if source_packet_id is not None:
        binding["schema"] = COMPACT_RESUME_SCHEMA_V2
        binding["source_packet_id"] = source_packet_id
    loaded_manifest = load_manifest(manifest_path, binding)
    if "iteration" in loaded_manifest and loaded_manifest["iteration"] != iteration:
        raise CompactResumeRefused("compact snapshot manifest iteration differs from envelope")

    with tempfile.TemporaryDirectory(prefix=f"train01-verify-{stem}-") as scratch_dir:
        decoded_pt = Path(scratch_dir) / f"{stem}.pt"
        with decoded_pt.open("xb") as stream:
            decode_stream(
                derivative_path,
                stream.write,
                expected_size=orig_size,
                expected_sha256=orig_sha,
            )
        decoded_state = torch.load(decoded_pt, map_location="cpu", weights_only=False)
        verify_decoded_state(decoded_state, loaded_manifest)
        del decoded_state

    result: dict[str, Any] = {
        "status": "PASS",
        "unit": identity,
        "iteration": iteration,
        "original": binding["original"],
        "derivative": binding["derivative"],
        "manifest": binding["manifest"],
        "info": {"name": info_path.name, "size": info_size, "sha256": info_sha},
        "actor": actor_desc,
        "tensor_count": loaded_manifest["tensor_count"],
        "tensor_bytes": loaded_manifest["tensor_bytes"],
        "structure_sha256": loaded_manifest["structure_sha256"],
    }
    if source_packet_id is not None:
        allowlist_entry: dict[str, Any] = {
            "source_packet_id": source_packet_id,
            "iteration": iteration,
            "unit": identity,
            "original": binding["original"],
            "derivative": binding["derivative"],
            "manifest": binding["manifest"],
        }
        if source_commit is not None:
            allowlist_entry["source_commit"] = source_commit
        result["allowlist_entry"] = allowlist_entry
    return result


def verify_decoded_state(state: Any, manifest: Mapping[str, Any]) -> None:
    """Refuse unless the decoded state matches every original tensor and its structure."""
    description = describe_state(state)
    expected = manifest.get("tensors")
    if not isinstance(expected, list) or len(description["tensors"]) != len(expected):
        raise CompactResumeRefused("decoded tensor count differs from the original manifest")
    for actual, original in zip(description["tensors"], expected):
        if actual != original:
            raise CompactResumeRefused(f"decoded tensor differs from original: {actual['path']}")
    if (
        description["structure_sha256"] != manifest.get("structure_sha256")
        or description["tensor_bytes"] != manifest.get("tensor_bytes")
        or description["tensor_count"] != manifest.get("tensor_count")
    ):
        raise CompactResumeRefused("decoded state structure differs from the original manifest")


def decode_to_bytes(derivative: Path, *, expected_size: int, expected_sha256: str) -> bytes:
    """Decode fully into memory (verification tooling; the runtime streams to disk)."""
    blocks: list[bytes] = []
    decode_stream(
        derivative, blocks.append, expected_size=expected_size, expected_sha256=expected_sha256
    )
    return b"".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode", help="create-only Mac encoding of one verified snapshot")
    encode.add_argument("original", type=Path)
    encode.add_argument("--unit")
    encode.add_argument("--source-packet-id")
    encode.add_argument("--iteration", type=int)
    encode.add_argument("--source-commit")
    verify = sub.add_parser(
        "verify", help="verify one compact-at-source or Mac-encoded snapshot export"
    )
    verify.add_argument("export_root", type=Path)
    verify.add_argument("--iteration", required=True, type=int)
    verify.add_argument("--source-packet-id")
    verify.add_argument("--source-commit")
    verify.add_argument("--no-require-actor", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "verify":
        result = verify_compact_snapshot_export(
            args.export_root.resolve(),
            args.iteration,
            source_packet_id=args.source_packet_id,
            source_commit=args.source_commit,
            require_actor=not args.no_require_actor,
        )
    else:
        result = encode_snapshot(
            args.original.resolve(),
            unit=args.unit,
            source_packet_id=args.source_packet_id,
            iteration=args.iteration,
            source_commit=args.source_commit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
