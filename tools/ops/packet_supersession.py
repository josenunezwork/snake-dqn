"""Read-only packet retirement check shared by preflight and dispatch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def require_current_packet(manifest: dict[str, Any], path: Path, *, root: Path) -> None:
    """Refuse retired identities before operational side effects or test execution."""
    registry = root / "tools/ops/packet-supersessions.json"
    value = json.loads(registry.read_text())
    if value.get("schema") != "snake-packet-supersessions/v1":
        raise ValueError("malformed packet supersession registry")
    for row in value["superseded"]:
        if "manifest_sha256" in row:
            manifest_path = path / "dispatch-manifest.json" if path.is_dir() else path
            matches = (
                hashlib.sha256(manifest_path.read_bytes()).hexdigest() == row["manifest_sha256"]
            )
        else:
            matches = manifest.get("job_id") in row["job_ids"] or row["packet_dir"] in path.parts
        if matches:
            raise ValueError(
                f"packet superseded: {row['packet_dir']}; use {row['successor']} "
                "after independent review"
            )
