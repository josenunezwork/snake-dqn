"""Validate immutable capture inputs, with a source-code-only snapshot fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_capture_inputs(
    inputs: Mapping[str, str], *, snapshot_root: Path, repo: Path
) -> dict[str, dict[str, str]]:
    """Validate current data or the exact archived bytes of changed repository Python.

    Checkpoints, arrays, manifests and every non-Python input always require
    exact current bytes. Snapshot recovery applies only to Python source under
    the worktree, and records the original path and digest without rewriting
    the immutable capture receipt.
    """
    snapshots = {}
    for source, expected in inputs.items():
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"invalid capture input SHA-256: {source}")
        original = Path(source)
        actual = _digest(original) if original.is_file() else None
        if actual == expected:
            continue
        if original.suffix != ".py" or not original.resolve().is_relative_to(repo.resolve()):
            raise ValueError(f"outside candidate input changed: {source}")
        snapshot = snapshot_root / f"{expected}.py"
        if not snapshot.is_file() or _digest(snapshot) != expected:
            raise ValueError(f"outside candidate source snapshot unavailable or changed: {source}")
        snapshots[f"{source}@{expected}"] = {
            "original_path": str(original),
            "original_sha256": expected,
            "snapshot_path": str(snapshot),
            "snapshot_sha256": expected,
            "live_source_sha256": actual or "missing",
        }
    return snapshots
