"""Durable checkpoint serialization helpers."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any, Union

import torch


def atomic_torch_save(payload: Any, destination: Union[str, Path]) -> None:
    """Durably save ``payload`` without risking an existing checkpoint.

    The temporary file lives beside the destination, which makes ``os.replace``
    an atomic same-filesystem swap.  Cleanup deliberately catches
    :class:`BaseException`, so an interrupted save does not leave a temporary
    checkpoint behind.

    Args:
        payload: Value accepted by :func:`torch.save`.
        destination: File to create or replace.
    """
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as checkpoint_file:
            torch.save(payload, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(tmp_path, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
