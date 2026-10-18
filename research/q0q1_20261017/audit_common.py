"""Shared fail-closed primitives for the Q0 CPU audit jobs.

These helpers intentionally do no training.  Q0 may read immutable receipts and
may run the registered bounded development observation packets only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

T23_SEEDS = (7101, 7102, 7103, 7104, 7105)
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
HORIZON = 5000
MIX_COUNTS = {"scripted": 16, "frozen": 8, "mixed": 8}


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic receipt after making its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(payload) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def iter_json(root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield readable JSON receipts, never treating malformed data as evidence."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        value = load_json(path)
        if value is not None:
            yield path, value


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return None if not rows else sum(rows) / len(rows)


def stratum(length: int) -> str:
    """The preregistered coarse length cuts, plus an explicit middle band."""
    if length < 700:
        return "short_lt700"
    if length >= 800:
        return "long_ge800"
    return "middle_700_799"
