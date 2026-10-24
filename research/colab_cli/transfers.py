"""Bounded-memory transport over the official Colab CLI ContentsClient.

The installed CLI base64-encodes a complete file. This adapter constrains each
ContentsClient request to a bounded upload/download chunk and uses the selected kernel only to
assemble or split already-uploaded local files. It never sends OAuth material
to the runtime.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import inspect
import itertools
import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Protocol

from research.colab_cli import policy

# Uploads travel through ContentsClient JSON/base64 PUTs, where smaller writes
# are materially more reliable on a fresh VM.  Downloads do not have that
# failure mode, so retain the established 8 MiB bound for export recovery.
UPLOAD_CHUNK_BYTES = 2 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
CONTENTS_ATTEMPTS = 3
# Only initial input staging shares this cap. Later transfers start fresh.
UPLOAD_TOTAL_TIMEOUT_SECONDS = 5 * 60
TRANSFER_MIN_TIMEOUT_SECONDS = 300
TRANSFER_BYTES_PER_SECOND = 512 * 1024


# Evidence identities use independent counters, never the jitter/request RNG.
_TRANSFER_IDS = itertools.count(1)
_PARENT_TRANSFER: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "parent_transfer", default=None
)
_CHILD_TRANSFER: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "child_transfer", default=None
)


def transfer_evidence(command: str) -> dict[str, Any]:
    """Metadata for this physical transfer; control PUTs carry no C6 claim."""
    parent, child = _PARENT_TRANSFER.get(), _CHILD_TRANSFER.get()
    if command not in {"upload", "download"}:
        return {}
    if parent is None or child is None:
        if command == "upload":
            return {"transfer_role": "control", "transfer_plane": "control"}
        return {"transfer_role": "consumed"}
    return {
        "transfer_role": "consumed",
        "parent_transfer_id": parent["transfer_id"],
        "transfer_id": child["transfer_id"],
        "transfer_attempt": child["attempts"],
    }


def record_verified_downloads(
    cli: _CLI, *, first_call: int, remote: str, digest: str, remotes: set[str]
) -> None:
    """Bind an already hash-checked data/sidecar pair to its observed calls.

    Used only after the acceptance observer's existing remote-sidecar versus
    Mac-payload check. This performs no transfer, hashing or new decision.
    """
    rows = [
        row
        for row in cli.call_records[first_call:]
        if row.get("command") == "download"
        and row.get("remote") in remotes
        and row.get("status") == "completed"
    ]
    if not rows:
        return
    parent_id = f"transfer-{next(_TRANSFER_IDS)}"
    children = []
    for row in rows:
        child_id = f"contents-{next(_TRANSFER_IDS)}"
        row.update(
            parent_transfer_id=parent_id,
            transfer_id=child_id,
            transfer_attempt=1,
            verified_sha256=digest,
        )
        children.append(
            {
                "transfer_id": child_id,
                "parent_transfer_id": parent_id,
                "remote": row["remote"],
                "attempts": 1,
                "status": "completed",
                "verified_sha256": digest,
                "started_monotonic": row["started_monotonic"],
                "finished_monotonic": row["finished_monotonic"],
            }
        )
    cli.transfer_records.append(
        {
            "transfer_id": parent_id,
            "direction": "download",
            "remote": remote,
            "status": "verified",
            "sha256": digest,
            "verified_sha256": digest,
            "verification_kind": "observed-checkpoint-sidecar",
            "children": children,
            "started_monotonic": rows[0]["started_monotonic"],
            "finished_monotonic": rows[-1]["finished_monotonic"],
            "elapsed_seconds": rows[-1]["finished_monotonic"] - rows[0]["started_monotonic"],
        }
    )


@contextlib.contextmanager
def consumed_metadata(cli: _CLI, remote: str, digest: str) -> Iterator[None]:
    """Bind metadata children to an export whose existing checks all passed.

    The caller marks verified only after its ordinary digest/sidecar/marker
    checks and publication. This adds no readback or verification decision.
    """
    started = time.monotonic()
    record = {
        "transfer_id": f"transfer-{next(_TRANSFER_IDS)}",
        "direction": "download",
        "remote": remote,
        "started_monotonic": started,
        "status": "failed",
        "children": [],
        "verification_kind": "export-triplet",
    }
    token = _PARENT_TRANSFER.set(record)
    try:
        yield
        record.update(status="verified", sha256=digest, verified_sha256=digest)
    finally:
        _PARENT_TRANSFER.reset(token)
        finished = time.monotonic()
        record.update(finished_monotonic=finished, elapsed_seconds=finished - started)
        for child in record["children"]:
            if record["status"] == "verified":
                child["verified_sha256"] = digest
        for call in getattr(cli, "call_records", []):
            if (
                call.get("parent_transfer_id") == record["transfer_id"]
                and record["status"] == "verified"
            ):
                call["verified_sha256"] = digest
        if hasattr(cli, "transfer_records"):
            cli.transfer_records.append(record)


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
    """Per chunk timeout: 120 seconds minimum, 180 seconds maximum."""
    return float(min(180, max(120, 30 + math.ceil(size / (512 * 1024)))))


def transfer_timeout(size: int) -> float:
    """Allow setup plus size-scaled transport, still bounded by the session cap."""
    return max(TRANSFER_MIN_TIMEOUT_SECONDS, 120 + size / TRANSFER_BYTES_PER_SECOND)


def _observed_transfer(direction: str) -> Callable:
    """Measure whole-file goodput including hashing, retries and lease renewal."""

    def decorate(function: Callable) -> Callable:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def observed(cli: _CLI, *args: Any, **kwargs: Any) -> str:
            caller = inspect.currentframe().f_back
            call_site = f"{caller.f_code.co_filename}:{caller.f_lineno}"
            del caller
            arguments = signature.bind(cli, *args, **kwargs).arguments
            remote = str(arguments["remote"])
            local = Path(arguments["local"])
            size = (
                (local.stat().st_size if local.is_file() else 0)
                if direction == "upload"
                else arguments["size"]
            )
            started = time.monotonic()
            record = {
                "transfer_id": f"transfer-{next(_TRANSFER_IDS)}",
                "started_monotonic": started,
                "children": [],
                "direction": direction,
                "remote": remote,
                "size_bytes": size,
                "call_site": call_site,
                "status": "failed",
            }
            token = _PARENT_TRANSFER.set(record)
            try:
                result = function(cli, *args, **kwargs)
                record.update(status="verified", sha256=result, verified_sha256=result)
                return result
            except Exception as exc:
                raise TransferError(f"{direction} {remote} called at {call_site}: {exc}") from exc
            finally:
                _PARENT_TRANSFER.reset(token)
                finished = time.monotonic()
                elapsed = max(finished - started, 1e-9)
                record.update(
                    finished_monotonic=finished,
                    elapsed_seconds=elapsed,
                    bytes_per_second=size / elapsed if record["status"] == "verified" else None,
                )
                for child in record["children"]:
                    if record["status"] == "verified":
                        child["verified_sha256"] = record["sha256"]
                for call in getattr(cli, "call_records", []):
                    if (
                        call.get("parent_transfer_id") == record["transfer_id"]
                        and record["status"] == "verified"
                    ):
                        call["verified_sha256"] = record["sha256"]
                if hasattr(cli, "transfer_records"):
                    cli.transfer_records.append(record)

        return observed

    return decorate


def _is_retryable_contents_error(exc: BaseException) -> bool:
    """Return whether a failed ContentsClient request may safely be repeated.

    ``CLI.call`` turns command failures into ``Refused`` strings, so importing
    its exception type here would create a dependency cycle.  Keep this narrow:
    only connection and timeout language emitted by requests/the CLI qualifies.
    Authentication, path, quota, and kernel errors must fail on their first
    observation.
    """
    return policy.transient_error(exc)


def _bounded_timeout(timeout: float, deadline: float | None, call_site: str = "transfer") -> float:
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TransferError(f"total upload time cap reached at {call_site}")
    return min(timeout, remaining)


def _contents_call(
    cli: _CLI,
    args: tuple[str, ...],
    timeout: float,
    renew: Callable[[], None] | None,
    *,
    deadline: float | None = None,
) -> None:
    """Retry only an idempotent ContentsClient transfer, at most three times.

    A retry repeats immutable bytes to the same temporary remote chunk or
    overwrites the same temporary local chunk.  ``CLI.call`` limits every
    attempt by its current broker deadline.  RPCs, assignments, and jobs never
    come through this helper.
    """
    failure: BaseException | None = None
    parent = _PARENT_TRANSFER.get()
    child = {
        "transfer_id": f"contents-{next(_TRANSFER_IDS)}",
        "parent_transfer_id": parent["transfer_id"] if parent else None,
        "remote": args[-1] if args[0] == "upload" else args[-2],
        "direction": args[0],
        "attempts": 0,
        "status": "failed",
        "calls": [],
    }
    if parent is not None:
        parent["children"].append(child)
    for attempt in range(CONTENTS_ATTEMPTS):
        if renew:
            renew()
        started = time.monotonic()
        child["attempts"] = attempt + 1
        child.setdefault("started_monotonic", started)
        token = _CHILD_TRANSFER.set(child)
        outcome = "failed"
        try:
            cli.call(*args, timeout=_bounded_timeout(timeout, deadline, f"{args[0]} {args[-1]}"))
            outcome = "completed"
        except Exception as exc:
            failure = exc
            if not _is_retryable_contents_error(exc):
                raise
            if attempt + 1 < CONTENTS_ATTEMPTS:
                delay = policy.BACKOFF_SECONDS[attempt]
                if deadline is not None:
                    delay = min(delay, max(0, deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
        else:
            _CHILD_TRANSFER.set(None)
            # Renewal is a separate mutating RPC.  Never mistake its failure
            # for a failed idempotent ContentsClient write and retry the write.
            if renew:
                renew()
            return
        finally:
            _CHILD_TRANSFER.reset(token)
            finished = time.monotonic()
            child.update(finished_monotonic=finished, status=outcome)
            child["elapsed_seconds"] = finished - child["started_monotonic"]
            child["calls"].append(
                {
                    "attempt": attempt + 1,
                    "started_monotonic": started,
                    "finished_monotonic": finished,
                    "elapsed_seconds": finished - started,
                    "timeout_seconds": timeout,
                    "status": outcome,
                }
            )
    assert failure is not None
    raise TransferError("ContentsClient transient retry budget exhausted") from failure


def _rpc(cli: _CLI, operation: str, spec: dict[str, Any], timeout: float) -> Any:
    """Recover reads or durable acknowledgements, never blindly replay mutations."""
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
path.mkdir(parents=True, exist_ok=True)
result = True
""".replace("PAYLOAD", payload)
    elif operation == "VERIFY_FILE":
        code = """# SNAKE_TRANSFER_VERIFY_FILE
import hashlib
from pathlib import Path
spec = PAYLOAD
path = Path(spec['path'])
hash_ = hashlib.sha256()
with path.open('rb') as source:
    while block := source.read(1024 * 1024):
        hash_.update(block)
result = {'sha256': hash_.hexdigest(), 'size': path.stat().st_size}
""".replace("PAYLOAD", payload)
    else:  # pragma: no cover
        raise AssertionError(operation)
    # These operations repeat the same fact/control state at a UUID path.
    # File assembly/splitting remains single-dispatch and fenced on ambiguity.
    if operation in {"VERIFY_FILE", "CREATE_DIR", "REMOVE_TREE"} and hasattr(cli, "poll_rpc"):
        return cli.poll_rpc(code, timeout=timeout)
    if operation in {"UPLOAD_ASSEMBLE", "DOWNLOAD_SPLIT"} and hasattr(cli, "recoverable_rpc"):
        return cli.recoverable_rpc(code, timeout=timeout)
    return cli.rpc(code, timeout=timeout)


def _verify_remote_file(
    cli: _CLI, remote: str, size: int, digest: str, *, deadline: float | None = None
) -> None:
    """Require the runtime to hash the published file before accepting upload."""
    reply = _rpc(
        cli,
        "VERIFY_FILE",
        {"path": remote},
        _bounded_timeout(policy.TELEMETRY_SECONDS, deadline, f"VERIFY_FILE {remote}"),
    )
    if not isinstance(reply, dict) or reply.get("sha256") != digest or reply.get("size") != size:
        raise TransferError("remote upload digest or size mismatch")


def _cleanup(
    cli: _CLI,
    path: str,
    renew: Callable[[], None] | None,
    *,
    deadline: float | None = None,
) -> None:
    try:
        if deadline is not None and time.monotonic() >= deadline:
            return
        if renew:
            renew()
        _rpc(
            cli, "REMOVE_TREE", {"path": path}, _bounded_timeout(policy.TELEMETRY_SECONDS, deadline)
        )
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


@_observed_transfer("upload")
def upload_file(
    cli: _CLI,
    local: Path,
    remote: str,
    renew: Callable[[], None],
    *,
    deadline: float | None = None,
) -> str:
    """Chunk-upload an immutable local input then atomically publish it remotely."""
    local = Path(local)
    if not local.is_file():
        raise TransferError(f"upload source is not a file: {local}")
    size, digest = local.stat().st_size, _digest(local)
    transfer_deadline = time.monotonic() + transfer_timeout(size)
    # The staging ceiling applies only to input PUTs. Renewals and kernel RPCs
    # must retain their own/session bounds even when staging has expired.
    deadline = min(deadline, transfer_deadline) if deadline is not None else transfer_deadline
    if size <= UPLOAD_CHUNK_BYTES:
        _contents_call(
            cli,
            ("upload", "-s", cli.session, str(local), remote),
            _timeout(size),
            renew,
            deadline=deadline,
        )
        if _digest(local) != digest or local.stat().st_size != size:
            raise TransferError("upload source changed during transfer")
        _verify_remote_file(cli, remote, size, digest, deadline=transfer_deadline)
        return digest
    ident = uuid.uuid4().hex
    parts, temporary = _staging(remote, ident)
    chunks = math.ceil(size / UPLOAD_CHUNK_BYTES)
    try:
        # ContentsClient PUT does not create missing parents. Directory creation
        # is idempotent; recovery still requires independent kernel-idle proof.
        renew()
        _rpc(
            cli,
            "CREATE_DIR",
            {"path": parts},
            _bounded_timeout(policy.TELEMETRY_SECONDS, transfer_deadline, "CREATE_DIR"),
        )
        renew()
        with tempfile.TemporaryDirectory(prefix="snake-colab-upload-") as directory:
            chunk = Path(directory) / "chunk"
            with local.open("rb") as source:
                for index in range(chunks):
                    data = source.read(UPLOAD_CHUNK_BYTES)
                    if not data:
                        raise TransferError("source changed while upload was read")
                    chunk.write_bytes(data)
                    _contents_call(
                        cli,
                        ("upload", "-s", cli.session, str(chunk), f"{parts}/{index:08d}.part"),
                        _timeout(len(data)),
                        renew,
                        deadline=deadline,
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
            _bounded_timeout(policy.TELEMETRY_SECONDS, transfer_deadline, "UPLOAD_ASSEMBLE"),
        )
        renew()
        if (
            not isinstance(reply, dict)
            or reply.get("sha256") != digest
            or reply.get("size") != size
        ):
            raise TransferError("kernel did not acknowledge immutable upload assembly")
        _verify_remote_file(cli, remote, size, digest, deadline=transfer_deadline)
        return digest
    finally:
        _cleanup(cli, parts, renew, deadline=transfer_deadline)
        _cleanup(cli, temporary, renew, deadline=transfer_deadline)


def upload_contents(cli: _CLI, local: Path, remote: str, *, deadline: float | None = None) -> None:
    """Bound a small idempotent PUT without a kernel RPC (credentials/controls)."""
    caller = inspect.currentframe().f_back
    call_site = f"{caller.f_code.co_filename}:{caller.f_lineno}"
    del caller
    size = local.stat().st_size
    if size > UPLOAD_CHUNK_BYTES:
        raise TransferError("control upload exceeds ContentsClient chunk size")
    transfer_deadline = time.monotonic() + transfer_timeout(size)
    if deadline is not None:
        transfer_deadline = min(deadline, transfer_deadline)
    try:
        _contents_call(
            cli,
            ("upload", "-s", cli.session, str(local), remote),
            _timeout(size),
            None,
            deadline=transfer_deadline,
        )
    except Exception as exc:
        raise TransferError(f"control upload {remote} called at {call_site}: {exc}") from exc


def prepare_download(
    cli: _CLI,
    remote: str,
    size: int,
    renew: Callable[[], None] | None = None,
    *,
    expected_sha256: str | None = None,
    deadline: float | None = None,
) -> PreparedDownload:
    """Split an export while the kernel is idle, before long-running work begins."""
    if type(size) is not int or size <= DOWNLOAD_CHUNK_BYTES:
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
            "chunk_bytes": DOWNLOAD_CHUNK_BYTES,
            "size": size,
            "sha256": expected_sha256,
        },
        _bounded_timeout(policy.TELEMETRY_SECONDS, deadline, f"DOWNLOAD_SPLIT {remote}"),
    )
    if renew:
        renew()
    count = math.ceil(size / DOWNLOAD_CHUNK_BYTES)
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


@_observed_transfer("download")
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
    deadline = time.monotonic() + transfer_timeout(size)
    if size <= DOWNLOAD_CHUNK_BYTES:
        _contents_call(
            cli,
            ("download", "-s", cli.session, remote, str(local)),
            _timeout(size),
            renew,
            deadline=deadline,
        )
        if local.stat().st_size != size or _digest(local) != expected_sha256:
            local.unlink(missing_ok=True)
            raise TransferError("download digest or size mismatch")
        return expected_sha256
    plan = prepared or prepare_download(
        cli, remote, size, renew, expected_sha256=expected_sha256, deadline=deadline
    )
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
                    expected_size = min(DOWNLOAD_CHUNK_BYTES, size - index * DOWNLOAD_CHUNK_BYTES)
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
                        deadline=deadline,
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
        _cleanup(cli, plan.parts, renew, deadline=deadline)
