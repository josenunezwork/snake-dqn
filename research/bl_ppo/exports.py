"""Create-only exports consumed by the unchanged Colab supervisor."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

_BUDGET: ContextVar[tuple[Path, int, int] | None] = ContextVar("bl_ppo_export_budget", default=None)


@contextmanager
def limit_exports(root: Path, *, total_bytes: int, file_bytes: int) -> Iterator[None]:
    """Apply a frozen job-wide on-disk allowance, including export sidecars."""
    if any(type(value) is not int or value <= 0 for value in (total_bytes, file_bytes)):
        raise ValueError("export byte allowances must be positive integers")
    token = _BUDGET.set((Path(root).resolve(), total_bytes, file_bytes))
    try:
        yield
    finally:
        _BUDGET.reset(token)


def digest(path: Path) -> str:
    """Hash a file without loading a checkpoint into memory."""
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bl-export-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # atomic create-only, including competing writers
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    """Publish immutable payload, digest, then the completion marker last."""
    path = Path(path)
    if any(Path(str(path) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError(f"export already exists: {path}")
    budget = _BUDGET.get()
    if budget is not None:
        root, total_limit, file_limit = budget
        if not path.resolve().is_relative_to(root):
            raise ValueError("export escaped its frozen job output root")
        if len(payload) > file_limit:
            raise ValueError("export exceeds manifest max_export_bytes per-file allowance")
        existing = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        added = len(payload) + 64 + 2 + len(path.name.encode()) + 1 + len(b"complete\n")
        if existing + added > total_limit:
            raise ValueError("exports exceed manifest max_total_export_bytes allowance")
    _publish(path, payload)
    checksum = hashlib.sha256(payload).hexdigest()
    _publish(Path(str(path) + ".sha256"), f"{checksum}  {path.name}\n".encode())
    _publish(Path(str(path) + ".complete"), b"complete\n")
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": str(path), "sha256": checksum, "size": len(payload)}


def export_json(path: Path, value: Any) -> dict[str, Any]:
    """Export canonical finite JSON."""
    return export_bytes(path, (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())


def export_torch(path: Path, value: Any, *, max_payload_bytes: int | None = None) -> dict[str, Any]:
    """Export a complete torch snapshot with supervisor sidecars."""
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    if max_payload_bytes is not None and buffer.tell() > max_payload_bytes:
        raise ValueError("snapshot exceeds its predeclared export byte bound")
    return export_bytes(path, buffer.getvalue())


def verify_export(path: Path) -> str:
    """Refuse partial or altered exports before using them as inputs."""
    path = Path(path)
    for suffix in ("", ".sha256", ".complete"):
        part = Path(str(path) + suffix)
        if part.is_symlink() or not part.is_file():
            raise ValueError(f"missing or unsafe export component: {part}")
    if Path(str(path) + ".complete").read_text() != "complete\n":
        raise ValueError("invalid completion marker")
    words = Path(str(path) + ".sha256").read_text().split()
    observed = digest(path)
    if words != [observed, path.name]:
        raise ValueError("export digest/name mismatch")
    return observed
