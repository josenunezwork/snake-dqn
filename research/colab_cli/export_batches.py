"""Bounded batched recovery of small immutable Colab exports.

The supervisor normally performs three Contents GETs for every immutable
export.  This module reduces that overhead for small exports by asking the
idle remote kernel to create one, deliberately uncompressed ZIP archive.  The
archive is only a transport container: local publication retains the original
payload, SHA sidecar, and completion-marker bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol, Sequence

from research.colab_cli.transfers import TransferError, download_file

MAX_BATCH_ENTRIES = 64
MAX_BATCH_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_SIDECAR_BYTES = 1024
MAX_MARKER_BYTES = 4096
_DIGEST = re.compile(r"[0-9a-f]{64}")


class ExportBatchError(RuntimeError):
    """A batched immutable-export recovery could not be proven safe."""


class _CLI(Protocol):
    renew: Callable[[], None]

    def recoverable_rpc(self, code: str, timeout: float = ...) -> Any: ...


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ExportBatchError("export path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(piece in {"", ".", ".."} for piece in value.split("/")):
        raise ExportBatchError(f"unsafe export path: {value!r}")
    return str(path)


def _validated_entry(entry: dict[str, Any], limit: int) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ExportBatchError("export entry must be an object")
    path = _safe_relative(entry.get("path"))
    size = entry.get("size")
    digest = entry.get("sha256")
    if type(size) is not int or size < 0 or size > limit:
        raise ExportBatchError("export exceeds bounded whole-file transfer limit")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        raise ExportBatchError("invalid export digest")
    # Keep any inventory fields byte-for-byte equivalent to the supervisor's
    # source entry.  Its caller compares repeated poll inventories for
    # immutability, so a transport helper must not silently discard metadata.
    result = dict(entry)
    result.update(path=path, size=size, sha256=digest)
    return result


def _validate_entries(entries: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if type(limit) is not int or limit <= 0:
        raise ExportBatchError("batch limit must be a positive integer")
    rows = [_validated_entry(entry, limit) for entry in entries]
    names: set[str] = set()
    for entry in rows:
        for name in (entry["path"], entry["path"] + ".sha256", entry["path"] + ".complete"):
            if name in names:
                raise ExportBatchError("export paths collide with immutable sidecars")
            names.add(name)
    return rows


def small_batches(
    entries: Sequence[dict[str, Any]], limit: int = MAX_BATCH_ENTRIES
) -> list[list[dict[str, Any]]]:
    """Return ordered batches, retaining oversize exports as singleton fallbacks.

    Every supplied entry appears once in the returned order.  A singleton whose
    payload exceeds :data:`MAX_BATCH_PAYLOAD_BYTES` is deliberately left for
    the caller's ordinary ``export_file`` path.  Other singleton batches are
    also valid fallback units, which keeps supervisor control flow simple.
    """
    observed_maximum = max(
        (entry.get("size", 0) if isinstance(entry, dict) else 0 for entry in entries), default=0
    )
    rows = _validate_entries(entries, max(MAX_BATCH_PAYLOAD_BYTES, observed_maximum))
    if type(limit) is not int or not 1 <= limit <= MAX_BATCH_ENTRIES:
        raise ExportBatchError(f"batch limit must be 1..{MAX_BATCH_ENTRIES}")
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0
    for entry in rows:
        if entry["size"] > MAX_BATCH_PAYLOAD_BYTES:
            if current:
                result.append(current)
                current, current_size = [], 0
            result.append([entry])
            continue
        if current and (
            len(current) >= limit or current_size + entry["size"] > MAX_BATCH_PAYLOAD_BYTES
        ):
            result.append(current)
            current, current_size = [], 0
        current.append(entry)
        current_size += entry["size"]
    if current:
        result.append(current)
    return result


def _batch_code(root: str, entries: list[dict[str, Any]], archive: str) -> str:
    """Produce the self-contained, single-use remote archive operation."""
    spec = {
        "root": root,
        "entries": entries,
        "archive": archive,
        "archive_limit": MAX_ARCHIVE_BYTES,
    }
    return """# SNAKE_EXPORT_BATCH
import hashlib, os, stat, zipfile
from pathlib import Path, PurePosixPath
spec = PAYLOAD
root = Path(spec['root'])
archive = Path(spec['archive'])
if root.is_symlink() or archive.exists() or archive.parent != root:
    raise RuntimeError('refusing existing or misplaced batch archive')
root_real = root.resolve(strict=True)
output = root / 'output'
if output.is_symlink() or not output.is_dir() or not output.resolve().is_relative_to(root_real):
    raise RuntimeError('unsafe export output root')
written = set()
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=False) as out:
    for entry in spec['entries']:
        rel = PurePosixPath(entry['path'])
        if (len(entry['path']) > 4096 or rel.is_absolute()
                or any(part in ('', '.', '..') for part in entry['path'].split('/'))):
            raise RuntimeError('unsafe export path')
        for suffix, bound in (('', entry['size']), ('.sha256', 1024), ('.complete', 4096)):
            name = entry['path'] + suffix
            source = output
            for part in PurePosixPath(name).parts:
                source = source / part
                if source.is_symlink():
                    raise RuntimeError('export member has symlink ancestor')
            before = source.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_size > bound:
                raise RuntimeError('export member is not a bounded regular file')
            if name in written:
                raise RuntimeError('duplicate archive member')
            with source.open('rb') as handle:
                data = handle.read(bound + 1)
            after = source.lstat()
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                    before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
                raise RuntimeError('export member changed while archiving')
            if len(data) != before.st_size or len(data) > bound:
                raise RuntimeError('export member read exceeded its bound')
            if suffix == '' and hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise RuntimeError('payload digest mismatch while archiving')
            if suffix == '.sha256':
                try:
                    sidecar = data.decode('utf-8').split()
                except UnicodeDecodeError as exc:
                    raise RuntimeError('digest sidecar is not text') from exc
                if not sidecar or sidecar[0] != entry['sha256']:
                    raise RuntimeError('digest sidecar mismatch while archiving')
            written.add(name)
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            out.writestr(info, data)
if archive.stat().st_size > spec['archive_limit']:
    raise RuntimeError('batch archive exceeds bounded download limit')
with archive.open('rb') as _snake_archive_handle:
    digest = hashlib.file_digest(_snake_archive_handle, 'sha256').hexdigest()
result = {'path': str(archive), 'size': archive.stat().st_size, 'sha256': digest,
          'members': sorted(written)}
""".replace("PAYLOAD", repr(spec))


def _read_archive(
    archive: Path, entries: list[dict[str, Any]]
) -> dict[str, tuple[bytes, bytes, bytes]]:
    expected = {
        name
        for entry in entries
        for name in (entry["path"], entry["path"] + ".sha256", entry["path"] + ".complete")
    }
    try:
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            if len(members) != len(expected) or {member.filename for member in members} != expected:
                raise ExportBatchError("batch archive member set differs from inventory")
            if any(
                member.is_dir()
                or member.flag_bits & 0x1
                or member.compress_type != zipfile.ZIP_STORED
                or member.file_size < 0
                or member.compress_size != member.file_size
                or (stat.S_IFMT(member.external_attr >> 16) not in {0, stat.S_IFREG})
                for member in members
            ):
                raise ExportBatchError("batch archive has unsupported member type")
            total = sum(member.file_size for member in members)
            if total > MAX_BATCH_PAYLOAD_BYTES + len(entries) * (
                MAX_SIDECAR_BYTES + MAX_MARKER_BYTES
            ):
                raise ExportBatchError("batch archive expands beyond bounded member total")
            info = {member.filename: member for member in members}
            result: dict[str, tuple[bytes, bytes, bytes]] = {}
            for entry in entries:
                payload_info = info[entry["path"]]
                sidecar_info = info[entry["path"] + ".sha256"]
                marker_info = info[entry["path"] + ".complete"]
                if (
                    payload_info.file_size != entry["size"]
                    or sidecar_info.file_size > MAX_SIDECAR_BYTES
                    or marker_info.file_size > MAX_MARKER_BYTES
                ):
                    raise ExportBatchError("batch archive member violates export bounds")
                payload = source.read(payload_info)
                sidecar = source.read(sidecar_info)
                marker = source.read(marker_info)
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ExportBatchError("batch payload digest mismatch")
                try:
                    sidecar_text = sidecar.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ExportBatchError("batch digest sidecar is not text") from exc
                fields = sidecar_text.split()
                if not fields or fields[0] != entry["sha256"]:
                    raise ExportBatchError("batch digest sidecar mismatch")
                result[entry["path"]] = (payload, sidecar, marker)
            return result
    except zipfile.BadZipFile as exc:
        raise ExportBatchError("batch archive is invalid") from exc


def _target(output: Path, relative: str) -> Path:
    """Make a local target without following a directory or leaf symlink."""
    if output.is_symlink():
        raise ExportBatchError("local export root is a symlink")
    current = output
    for part in ("exports", *PurePosixPath(relative).parts[:-1]):
        current = current / part
        if current.is_symlink():
            raise ExportBatchError("local export destination has a symlink ancestor")
        if current.exists() and not current.is_dir():
            raise ExportBatchError("local export destination parent is not a directory")
        current.mkdir(exist_ok=True)
        if current.is_symlink():
            raise ExportBatchError("local export destination has a symlink ancestor")
    return current / PurePosixPath(relative).name


def _publish(target: Path, payload: bytes, sidecar: bytes, marker: bytes) -> None:
    sidecar_target, marker_target = Path(str(target) + ".sha256"), Path(str(target) + ".complete")
    if (
        target.exists()
        or target.is_symlink()
        or sidecar_target.exists()
        or sidecar_target.is_symlink()
        or marker_target.exists()
        or marker_target.is_symlink()
    ):
        raise ExportBatchError("local immutable export destination already exists")
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".batch-export-") as temporary:
        temp = Path(temporary)
        data, digest, complete = temp / "data", temp / "sha256", temp / "complete"
        data.write_bytes(payload)
        digest.write_bytes(sidecar)
        complete.write_bytes(marker)
        os.replace(data, target)
        os.replace(digest, sidecar_target)
        os.replace(complete, marker_target)


def export_batch(
    cli: _CLI,
    root: str,
    entries: Sequence[dict[str, Any]],
    output: Path,
    limit: int,
    on_verified: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Recover one small batch and publish each verified immutable triple.

    Archives are intentionally retained remotely under the job root.  Their
    random names make a recovery operation non-destructive; retaining at most
    eight MiB per batch trades bounded scratch disk for no unsafe cleanup RPC.
    """
    rows = _validate_entries(entries, limit)
    if (
        not rows
        or len(rows) > MAX_BATCH_ENTRIES
        or sum(row["size"] for row in rows) > MAX_BATCH_PAYLOAD_BYTES
    ):
        raise ExportBatchError("invalid small export batch")
    root_path = PurePosixPath(root)
    if not isinstance(root, str) or not root_path.is_absolute() or "/../" in root:
        raise ExportBatchError("unsafe remote batch root")
    archive = root.rstrip("/") + "/.snake-export-batch-" + uuid.uuid4().hex + ".zip"
    reply = cli.recoverable_rpc(_batch_code(root.rstrip("/"), rows, archive))
    if not isinstance(reply, dict) or reply.get("path") != archive:
        raise ExportBatchError("remote batch archive acknowledgement is invalid")
    archive_size, archive_digest = reply.get("size"), reply.get("sha256")
    if (
        type(archive_size) is not int
        or not 0 <= archive_size <= MAX_ARCHIVE_BYTES
        or not isinstance(archive_digest, str)
        or not _DIGEST.fullmatch(archive_digest)
    ):
        raise ExportBatchError("remote batch archive metadata is invalid")
    expected_members = sorted(
        name
        for row in rows
        for name in (row["path"], row["path"] + ".sha256", row["path"] + ".complete")
    )
    if reply.get("members") != expected_members:
        raise ExportBatchError("remote batch archive inventory is invalid")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output, prefix=".batch-download-") as temporary:
        local_archive = Path(temporary) / "archive.zip"
        try:
            download_file(
                cli,
                archive,
                local_archive,
                archive_size,
                renew=cli.renew,
                expected_sha256=archive_digest,
            )
        except TransferError as exc:
            raise ExportBatchError(f"batch archive download failed: {exc}") from exc
        data = _read_archive(local_archive, rows)
    for row in rows:
        _publish(_target(output, row["path"]), *data[row["path"]])
        if on_verified is not None:
            on_verified(row)
