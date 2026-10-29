"""Stage exact checkpoint bytes; stable content-addressed paths across allocations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from research.bl_served_search.exports import digest
from research.watch_lead_p_successor.units import CHECKPOINTS

B8_URL = (
    "https://github.com/josenunezwork/snake-"
    "dqn/releases/download/s1-inputs-20261019/B8_s3_final.pth"
)
B8_BYTES = 6_562_005
STABLE_ROOT = Path("/tmp/p-watch-lead-successor-checkpoints-v1")
TRANSFER_CAP_SECONDS = 150
REQUIRED_MARGIN_SECONDS = 45


def stage(inputs: Path, root: Path = STABLE_ROOT) -> Path:
    """Hash source and final bytes, atomic create-only publication; no reserialization."""
    if root.is_symlink():
        raise ValueError("unsafe checkpoint stage")
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKPOINTS.items():
        source = inputs / name
        if source.is_symlink() or not source.is_file() or digest(source) != expected:
            raise ValueError("checkpoint input identity mismatch")
        target = root / name
        if not target.exists():
            fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=root)
            try:
                with os.fdopen(fd, "wb") as handle, source.open("rb") as src:
                    while chunk := src.read(1024 * 1024):
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if digest(Path(temporary)) != expected:
                    raise ValueError("checkpoint copy mismatch")
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    pass  # concurrent immutable staging is allowed only for identical bytes
            finally:
                Path(temporary).unlink(missing_ok=True)
        if target.is_symlink() or digest(target) != expected:
            raise ValueError("checkpoint stage collision")
    return root


def require_transfer_margin(measured: dict) -> dict:
    """The future target receipt must demonstrate headroom, not merely cap survival."""
    seconds = measured.get("transfer_seconds")
    staging = measured.get("elapsed_seconds")
    if (
        type(seconds) not in (int, float)
        or not 0 < seconds <= TRANSFER_CAP_SECONDS - REQUIRED_MARGIN_SECONDS
        or type(staging) not in (int, float)
        or not 0 < staging <= 300
    ):
        raise ValueError("target transfer must measure <=105 s and total staging <=300 s")
    return {
        "transfer_seconds": seconds,
        "margin_seconds": TRANSFER_CAP_SECONDS - seconds,
        "staging_seconds": staging,
    }
