"""Bounded lossless tensor input I/O; no simulator imports on the GPU."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any

import torch

from research.doom_auc_smoke.evidence import atomic


def save_gzip(path: Path, payload: dict[str, Any]) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic(path, gzip.compress(buffer.getvalue(), mtime=0))


def load_gzip(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("compressed input exceeds 64 MiB")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(512 * 1024**2 + 1)
    if len(raw) > 512 * 1024**2:
        raise ValueError("decoded input exceeds 512 MiB")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
