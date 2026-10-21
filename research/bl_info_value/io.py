"""Create-only supervisor-compatible immutable exports."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    """Hash a regular file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def export_bytes(path: Path, payload: bytes) -> None:
    """Publish payload, checksum, then completion marker; never overwrite."""
    for candidate in (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete")):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(candidate)
    _publish(path, payload)
    _publish(Path(str(path) + ".sha256"), (hashlib.sha256(payload).hexdigest() + "\n").encode())
    _publish(Path(str(path) + ".complete"), b"complete\n")


def write_json(path: Path, value: Any) -> None:
    """Publish a finite canonical JSON export with supervisor sidecars."""
    export_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Publish compressed non-object arrays with completion fencing."""
    stream = io.BytesIO()
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays prohibited")
    np.savez_compressed(stream, **arrays)
    export_bytes(path, stream.getvalue())


def verify_export(path: Path) -> None:
    """Require immutable sidecars before consuming an exported payload."""
    if not Path(str(path) + ".complete").is_file():
        raise ValueError(f"incomplete export: {path}")
    if Path(str(path) + ".sha256").read_text().split()[0] != sha256(path):
        raise ValueError(f"export digest mismatch: {path}")
