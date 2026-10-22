"""Bounded runtime snapshot downloads; only the tiny catalog crosses Contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from research.bl_ppo.exports import digest, export_bytes, verify_export

MAX_BYTES = 2 * 1024**3
MAX_CATALOG_BYTES = 8 * 1024**3
SCHEMA = "bl-p1-runtime-downloads/v1"


def unpack_bundle(bundle: Path, output: Path) -> dict[str, Any]:
    """Extract the small pinned flat-file input bundle without path/link escapes."""
    if bundle.stat().st_size > 6 * 1024**2:
        raise ValueError("P1 compressed input bundle exceeds 6 MiB")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names: set[str] = set()
        total = 0
        for member in members:
            if (
                not member.isfile()
                or Path(member.name).name != member.name
                or member.name in {"", ".", ".."}
                or member.name in names
                or (output / member.name).exists()
                or (output / member.name).is_symlink()
            ):
                raise ValueError("unsafe or conflicting P1 bundle member")
            total += member.size
            names.add(member.name)
        if total > 64 * 1024**2 or not members:
            raise ValueError("P1 bundle is empty or exceeds 64 MiB unpacked")
        for member in members:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("missing P1 bundle payload")
            with (output / member.name).open("xb") as target:
                target.write(stream.read())
    for name in names:
        if name.endswith(".complete"):
            verify_export(output / name.removesuffix(".complete"))
    return {"status": "VERIFIED", "files": len(members), "bytes": total}


def validate_catalog(
    value: dict[str, Any], *, maximum_total_bytes: int = MAX_CATALOG_BYTES
) -> list[dict[str, Any]]:
    """Refuse unbounded, unpinned or path-escaping runtime inputs."""
    if value.get("schema") != SCHEMA or not isinstance(value.get("files"), list):
        raise ValueError("invalid P1 runtime download catalog")
    # Preserve legacy catalogues' aggregate 2 GiB contract. New catalogues may
    # explicitly freeze a larger total; individual files retain the qualified
    # supervisor's 2 GiB cap. No Contents upload carries these payloads.
    total_cap = value.get("max_total_bytes", MAX_BYTES)
    if type(total_cap) is not int or not 0 < total_cap <= min(
        MAX_CATALOG_BYTES, maximum_total_bytes
    ):
        raise ValueError("P1 download aggregate cap must be declared within 8 GiB")
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
    if not names or total > total_cap:
        raise ValueError("P1 transitive download set is empty or exceeds declared aggregate cap")
    return value["files"]


def materialize(catalog: Path, output: Path, *, control: Path | None = None) -> dict[str, Any]:
    """Download once, verify before publication, and create supervisor sidecars."""
    cap = MAX_BYTES
    if control is not None:
        from research.bl_ppo.export_budget import validate_control_budget

        verify_export(control)
        frozen = json.loads(control.read_text())
        budget = validate_control_budget(frozen)
        if digest(catalog) != frozen.get("runtime_downloads_sha256"):
            raise ValueError("runtime catalog differs from frozen control digest")
        cap = budget["download_closure_cap_bytes"]
    rows = validate_catalog(json.loads(catalog.read_text()), maximum_total_bytes=cap)
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--catalog", type=Path)
    mode.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    result = (
        unpack_bundle(args.bundle, args.output)
        if args.bundle
        else materialize(args.catalog, args.output, control=args.control)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
