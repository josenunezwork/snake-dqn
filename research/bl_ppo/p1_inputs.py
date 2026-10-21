"""Bounded runtime snapshot downloads; only the tiny catalog crosses Contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from research.bl_ppo.exports import export_bytes, verify_export

MAX_BYTES = 2 * 1024**3
SCHEMA = "bl-p1-runtime-downloads/v1"


def validate_catalog(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Refuse unbounded, unpinned or path-escaping runtime inputs."""
    if value.get("schema") != SCHEMA or not isinstance(value.get("files"), list):
        raise ValueError("invalid P1 runtime download catalog")
    names: set[str] = set()
    total = 0
    for row in value["files"]:
        name = row.get("name")
        size = row.get("size_bytes")
        checksum = row.get("sha256", "")
        url = urlsplit(row.get("url", ""))
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or name in names
            or name.endswith((".sha256", ".complete"))
        ):
            raise ValueError("unsafe or duplicate P1 download name")
        if type(size) is not int or not 0 < size <= MAX_BYTES:
            raise ValueError("P1 download must have a bounded positive byte size")
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("P1 download requires lowercase SHA256")
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("P1 runtime download requires HTTPS without embedded credentials")
        names.add(name)
        total += size
    if not names or total > MAX_BYTES:
        raise ValueError("P1 transitive download set is empty or exceeds 2 GiB")
    return value["files"]


def materialize(catalog: Path, output: Path) -> dict[str, Any]:
    """Download once, verify before publication, and create supervisor sidecars."""
    rows = validate_catalog(json.loads(catalog.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    verified = []
    for row in rows:
        request = urllib.request.Request(row["url"], headers={"User-Agent": "snake-p1/1"})
        chunks = []
        total = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("P1 download redirected away from HTTPS")
            while block := response.read(min(1024**2, row["size_bytes"] - total + 1)):
                total += len(block)
                if total > row["size_bytes"]:
                    raise ValueError("P1 runtime input exceeds registered size")
                chunks.append(block)
        payload = b"".join(chunks)
        if total != row["size_bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("P1 runtime input size/hash mismatch")
        path = output / row["name"]
        verified.append(export_bytes(path, payload))
        verify_export(path)
    return {"schema": SCHEMA, "status": "VERIFIED", "files": verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.catalog, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
