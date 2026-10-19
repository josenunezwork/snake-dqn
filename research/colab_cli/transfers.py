"""Bounded-memory transport over the official Colab CLI ContentsClient.

The installed CLI base64-encodes a complete file. This adapter constrains each
ContentsClient request to ``CHUNK_BYTES`` and uses the selected kernel only to
assemble or split already-uploaded local files. It never sends OAuth material
to the runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

CHUNK_BYTES = 8 * 1024 * 1024
CONTENTS_ATTEMPTS = 3


class TransferError(RuntimeError):
    """A transfer did not produce a verified immutable file."""


class PreparedDownload:
    """Remote immutable split created while the kernel is known to be idle."""

    def __init__(self, parts: str, chunks: int, size: int, sha256: str) -> None:
        self.parts, self.chunks, self.size, self.sha256 = parts, chunks, size, sha256


class _CLI(Protocol):
    session: str

    def call(self, *args: str, timeout: float = ...) -> str: ...

    def rpc(self, code: str, timeout: float = ...) -> Any: ...


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _timeout(size: int) -> float:
    """Per chunk timeout: 60 seconds minimum, 120 seconds maximum."""
    return float(min(120, max(60, 15 + math.ceil(size / (512 * 1024)))))


def _contents_call(
    cli: _CLI, args: tuple[str, ...], timeout: float, renew: Callable[[], None] | None
) -> None:
    """Retry only an idempotent ContentsClient transfer, at most three times.

    A retry repeats immutable bytes to the same temporary remote chunk or
    overwrites the same temporary local chunk.  ``CLI.call`` limits every
    attempt by its current broker deadline.  RPCs, assignments, and jobs never
    come through this helper.
    """
    failure: BaseException | None = None
    for _ in range(CONTENTS_ATTEMPTS):
        if renew:
            renew()
        try:
            cli.call(*args, timeout=timeout)
            if renew:
                renew()
            return
        except BaseException as exc:
            failure = exc
    assert failure is not None
    raise failure


def _rpc(cli: _CLI, operation: str, spec: dict[str, Any], timeout: float) -> Any:
    """Use one small, auditable RPC. Mutating RPCs are never retried here."""
    payload = json.dumps(spec, sort_keys=True)
    if operation == "UPLOAD_ASSEMBLE":
        code = """# SNAKE_TRANSFER_UPLOAD_ASSEMBLE
import hashlib, os
from pathlib import Path
spec = PAYLOAD
parts = Path(spec['parts'])
temporary = Path(spec['temporary'])
target = Path(spec['target'])
temporary.parent.mkdir(parents=True, exist_ok=True)
hash_ = hashlib.sha256()
with temporary.open('xb') as out:
    for index in range(spec['chunks']):
        with (parts / f'{index:08d}.part').open('rb') as source:
            while block := source.read(1024 * 1024):
                out.write(block)
                hash_.update(block)
    out.flush()
    os.fsync(out.fileno())
if hash_.hexdigest() != spec['sha256'] or temporary.stat().st_size != spec['size']:
    raise RuntimeError('assembled upload digest or size mismatch')
os.replace(temporary, target)
result = {'sha256': hash_.hexdigest(), 'size': spec['size']}
""".replace("PAYLOAD", payload)
    elif operation == "DOWNLOAD_SPLIT":
        code = """# SNAKE_TRANSFER_DOWNLOAD_SPLIT
import hashlib, os
from pathlib import Path
spec = PAYLOAD
source = Path(spec['source'])
parts = Path(spec['parts'])
parts.mkdir(parents=True, exist_ok=False)
hash_ = hashlib.sha256()
count = 0
with source.open('rb') as handle:
    while block := handle.read(spec['chunk_bytes']):
        with (parts / f'{count:08d}.part').open('xb') as out:
            out.write(block)
            out.flush()
            os.fsync(out.fileno())
        hash_.update(block)
        count += 1
if hash_.hexdigest() != spec['sha256'] or source.stat().st_size != spec['size']:
    raise RuntimeError('export changed while splitting')
result = {'sha256': hash_.hexdigest(), 'size': spec['size'], 'chunks': count}
""".replace("PAYLOAD", payload)
    elif operation == "REMOVE_TREE":
        code = """# SNAKE_TRANSFER_REMOVE_TREE
import shutil
from pathlib import Path
spec = PAYLOAD
path = Path(spec['path'])
if not path.name.startswith('.snake-transfer-'):
    raise RuntimeError('refusing non-transfer cleanup')
if path.is_dir():
    shutil.rmtree(path, ignore_errors=True)
else:
    path.unlink(missing_ok=True)
result = True
""".replace("PAYLOAD", payload)
    elif operation == "CREATE_DIR":
        code = """# SNAKE_TRANSFER_CREATE_DIR
from pathlib import Path
spec = PAYLOAD
path = Path(spec['path'])
if not path.name.startswith('.snake-transfer-'):
    raise RuntimeError('refusing non-transfer directory')
path.mkdir(parents=True, exist_ok=False)
result = True
""".replace("PAYLOAD", payload)
    else:  # pragma: no cover
        raise AssertionError(operation)
    return cli.rpc(code, timeout=timeout)


def _cleanup(cli: _CLI, path: str, renew: Callable[[], None] | None) -> None:
    try:
        if renew:
            renew()
        _rpc(cli, "REMOVE_TREE", {"path": path}, 60)
        if renew:
            renew()
    except Exception:
        pass


def _staging(remote: str, ident: str) -> tuple[str, str]:
    """Keep temporary directories beside, never below, a possibly-file target."""
    path = PurePosixPath(remote)
    if not path.is_absolute() or not path.name:
        raise TransferError(f"remote path must be absolute file path: {remote!r}")
    parent = str(path.parent)
    base = f".snake-transfer-{ident}"
    return f"{parent}/{base}", f"{parent}/{base}.assembling"


def upload_file(cli: _CLI, local: Path, remote: str, renew: Callable[[], None]) -> str:
    """Chunk-upload an immutable local input then atomically publish it remotely."""
    local = Path(local)
    if not local.is_file():
        raise TransferError(f"upload source is not a file: {local}")
    size, digest = local.stat().st_size, _digest(local)
    if size <= CHUNK_BYTES:
        _contents_call(
            cli, ("upload", "-s", cli.session, str(local), remote), _timeout(size), renew
        )
        if _digest(local) != digest or local.stat().st_size != size:
            raise TransferError("upload source changed during transfer")
        return digest
    ident = uuid.uuid4().hex
    parts, temporary = _staging(remote, ident)
    chunks = math.ceil(size / CHUNK_BYTES)
    try:
        # ContentsClient PUT does not create missing parents.  This is a
        # single non-retried mutation; an uncertain RPC fences later assembly.
        renew()
        _rpc(cli, "CREATE_DIR", {"path": parts}, _timeout(CHUNK_BYTES))
        renew()
        with tempfile.TemporaryDirectory(prefix="snake-colab-upload-") as directory:
            chunk = Path(directory) / "chunk"
            with local.open("rb") as source:
                for index in range(chunks):
                    data = source.read(CHUNK_BYTES)
                    if not data:
                        raise TransferError("source changed while upload was read")
                    chunk.write_bytes(data)
                    _contents_call(
                        cli,
                        ("upload", "-s", cli.session, str(chunk), f"{parts}/{index:08d}.part"),
                        _timeout(len(data)),
                        renew,
                    )
        if _digest(local) != digest or local.stat().st_size != size:
            raise TransferError("upload source changed during transfer")
        renew()
        reply = _rpc(
            cli,
            "UPLOAD_ASSEMBLE",
            {
                "parts": parts,
                "temporary": temporary,
                "target": remote,
                "chunks": chunks,
                "sha256": digest,
                "size": size,
            },
            _timeout(CHUNK_BYTES),
        )
        renew()
        if (
            not isinstance(reply, dict)
            or reply.get("sha256") != digest
            or reply.get("size") != size
        ):
            raise TransferError("kernel did not acknowledge immutable upload assembly")
        return digest
    finally:
        _cleanup(cli, parts, renew)
        _cleanup(cli, temporary, renew)


def prepare_download(
    cli: _CLI,
    remote: str,
    size: int,
    renew: Callable[[], None] | None = None,
    *,
    expected_sha256: str | None = None,
) -> PreparedDownload:
    """Split an export while the kernel is idle, before long-running work begins."""
    if type(size) is not int or size <= CHUNK_BYTES:
        raise TransferError("prepare_download is only for files larger than a transfer chunk")
    if expected_sha256 is None or len(expected_sha256) != 64:
        raise TransferError("prepare_download requires immutable expected_sha256")
    parts, _ = _staging(remote, uuid.uuid4().hex)
    if renew:
        renew()
    reply = _rpc(
        cli,
        "DOWNLOAD_SPLIT",
        {
            "source": remote,
            "parts": parts,
            "chunk_bytes": CHUNK_BYTES,
            "size": size,
            "sha256": expected_sha256,
        },
        _timeout(CHUNK_BYTES),
    )
    if renew:
        renew()
    count = math.ceil(size / CHUNK_BYTES)
    if (
        not isinstance(reply, dict)
        or reply.get("sha256") != expected_sha256
        or reply.get("size") != size
    ):
        _cleanup(cli, parts, renew)
        raise TransferError("kernel did not acknowledge immutable export split")
    if reply.get("chunks") != count:
        _cleanup(cli, parts, renew)
        raise TransferError("kernel returned invalid export chunk count")
    return PreparedDownload(parts, count, size, expected_sha256)


def download_file(
    cli: _CLI,
    remote: str,
    local: Path,
    size: int,
    renew: Callable[[], None] | None = None,
    *,
    expected_sha256: str | None = None,
    prepared: PreparedDownload | None = None,
) -> str:
    """Chunk-download an immutable export, digest check it, and atomically publish."""
    if type(size) is not int or size < 0:
        raise TransferError("download size must be a nonnegative integer")
    if expected_sha256 is None:
        raise TransferError("download requires immutable expected_sha256")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise TransferError("download expected_sha256 must be lowercase SHA-256")
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if size <= CHUNK_BYTES:
        _contents_call(
            cli, ("download", "-s", cli.session, remote, str(local)), _timeout(size), renew
        )
        if local.stat().st_size != size or _digest(local) != expected_sha256:
            local.unlink(missing_ok=True)
            raise TransferError("download digest or size mismatch")
        return expected_sha256
    plan = prepared or prepare_download(cli, remote, size, renew, expected_sha256=expected_sha256)
    if plan.size != size or plan.sha256 != expected_sha256:
        raise TransferError("prepared download does not match immutable export inventory")
    try:
        with tempfile.NamedTemporaryFile(
            dir=local.parent, prefix=".transfer-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                for index in range(plan.chunks):
                    part = Path(str(temporary) + f".{index}")
                    expected_size = min(CHUNK_BYTES, size - index * CHUNK_BYTES)
                    _contents_call(
                        cli,
                        (
                            "download",
                            "-s",
                            cli.session,
                            f"{plan.parts}/{index:08d}.part",
                            str(part),
                        ),
                        _timeout(expected_size),
                        renew,
                    )
                    try:
                        if part.stat().st_size != expected_size:
                            raise TransferError("downloaded chunk has unexpected size")
                        with part.open("rb") as source:
                            while block := source.read(1024 * 1024):
                                handle.write(block)
                    finally:
                        part.unlink(missing_ok=True)
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        if temporary.stat().st_size != size or _digest(temporary) != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise TransferError("download digest or size mismatch")
        os.replace(temporary, local)
        return expected_sha256
    finally:
        _cleanup(cli, plan.parts, renew)
