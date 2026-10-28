"""Read-only packet retirement check shared by preflight and dispatch."""

from __future__ import annotations

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
        if manifest.get("job_id") in row["job_ids"] or row["packet_dir"] in path.parts:
            raise ValueError(
                f"packet superseded: {row['packet_dir']}; use {row['successor']} "
                "after independent review"
            )
