"""Read-only packet retirement check shared by preflight and dispatch."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def require_current_packet(
    manifest: dict[str, Any], path: Path, *, root: Path, original_bytes: bytes | None = None
) -> None:
    """Refuse retired identities using the bytes already loaded by the caller."""
    try:
        value = json.loads((root / "tools/ops/packet-supersessions.json").read_text())
        if (
            not isinstance(value, dict)
            or value.get("schema") != "snake-packet-supersessions/v1"
            or not isinstance(value.get("superseded"), list)
        ):
            raise ValueError("malformed packet supersession registry")
        rows = value["superseded"]
        for row in rows:
            if (
                not isinstance(row, dict)
                or any(
                    not isinstance(row.get(k), str) or not row[k].strip()
                    for k in ("packet_dir", "successor", "reason")
                )
                or not isinstance(row.get("job_ids"), list)
                or any(not isinstance(item, str) or not item for item in row["job_ids"])
                or (
                    "manifest_sha256" in row
                    and (
                        not isinstance(row["manifest_sha256"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", row["manifest_sha256"]) is None
                    )
                )
            ):
                raise ValueError("malformed packet supersession registry row")
        for row in rows:
            if "manifest_sha256" in row:
                if original_bytes is None:
                    raise ValueError("packet retirement check requires loaded manifest bytes")
                matches = hashlib.sha256(original_bytes).hexdigest() == row["manifest_sha256"]
            else:
                matches = (
                    manifest.get("job_id") in row["job_ids"] or row["packet_dir"] in path.parts
                )
            if matches:
                raise ValueError(
                    f"packet superseded: {row['packet_dir']}; use {row['successor']} "
                    "after independent review"
                )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed or unreadable packet supersession registry") from exc
