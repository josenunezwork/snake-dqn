#!/usr/bin/env python3
"""Publish hardware evidence alongside, without editing, a supervisor receipt."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def bound(path: Path) -> dict:
    """Bind an immutable local evidence file."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def publish(manifest: Path, destination: Path) -> None:
    """Keep requested and observed hardware distinct, including early failure."""
    packet = json.loads(manifest.read_text())
    output = Path(packet["output_dir"])
    receipt_path, hardware_path = output / "receipt.json", output / "hardware.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    hardware = json.loads(hardware_path.read_text()) if hardware_path.is_file() else None
    value = {
        "schema": "colab-gpu-dispatch-receipt/v1",
        "job_id": packet["job_id"],
        "manifest": bound(manifest),
        "requested_gpu": packet["requirements"]["gpu"],
        "observed_hardware": hardware,
        "hardware_evidence": bound(hardware_path) if hardware is not None else None,
        "supervisor_receipt": bound(receipt_path) if receipt else None,
        "endpoint": receipt.get("endpoint"),
        "status": receipt.get("status", "NO_SUPERVISOR_RECEIPT"),
        "stop_verified": receipt.get("stop_verified", False),
        "budget": packet["budget"],
        "usage_before": receipt.get("usage_before"),
        "usage_after": receipt.get("usage_after"),
        "accounting": receipt.get("accounting"),
    }
    fd, temporary = tempfile.mkstemp(prefix=".gpu-receipt-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Path(sys.argv[2]))
