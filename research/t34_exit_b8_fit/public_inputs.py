"""Materialize portable, hash-pinned inputs for the ExIt-B8 fit entrypoint.

Publication is deliberately fail-closed: catalog validation completes before a
network connection is opened, and unresolved identities are not downloadable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.request import urlopen

CHUNK_BYTES = 1024 * 1024
CATALOG_PATH = Path(__file__).with_name("PUBLIC_INPUTS.json")


class PublicInputError(ValueError):
    """A public input catalog is incomplete, unsafe, or has drifted."""


def _safe_dest(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PublicInputError("each public input requires a destination")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise PublicInputError(f"unsafe public destination: {value!r}")
    return str(path)


def load_catalog(path: Path) -> list[dict[str, Any]]:
    """Read a complete catalog without contacting any URL."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicInputError(f"invalid public input catalog: {path}") from exc
    rows = value.get("inputs") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise PublicInputError("catalog must contain public inputs")
    names: set[str] = set()
    destinations: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PublicInputError("catalog rows must be objects")
        name, dest = row.get("name"), _safe_dest(row.get("dest"))
        if not isinstance(name, str) or not name or name in names or dest in destinations:
            raise PublicInputError("catalog names and destinations must be unique")
        names.add(name)
        destinations.add(dest)
        size, digest, url = row.get("size_bytes"), row.get("sha256"), row.get("url")
        if not isinstance(size, int) or size <= 0:
            raise PublicInputError(f"publication identity is pending for {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise PublicInputError(f"publication identity is pending for {name}")
        if (
            not isinstance(url, str)
            or not url.startswith("https://")
            or len(url) == len("https://")
        ):
            raise PublicInputError(f"publication is pending for {name}; no download was attempted")
        members = row.get("members", [])
        if not isinstance(members, list):
            raise PublicInputError(f"members must be a list for {name}")
        for member in members:
            if not isinstance(member, dict):
                raise PublicInputError(f"invalid archive member for {name}")
            _safe_dest(member.get("dest"))
            member_hash = member.get("sha256")
            if (
                not isinstance(member.get("name"), str)
                or not isinstance(member_hash, str)
                or len(member_hash) != 64
            ):
                raise PublicInputError(f"archive member identity is incomplete for {name}")
    return rows


def _download(
    row: dict[str, Any], out_dir: Path, opener: Callable[..., Any], timeout: float
) -> Path:
    target = out_dir / str(row["dest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size, expected_hash = int(row["size_bytes"]), str(row["sha256"])
    written = 0
    digest = hashlib.sha256()
    created = False
    try:
        with target.open("xb") as stream:
            created = True
            with opener(str(row["url"]), timeout=timeout) as response:
                while True:
                    block = response.read(min(CHUNK_BYTES, expected_size - written + 1))
                    if not block:
                        break
                    written += len(block)
                    if written > expected_size:
                        raise PublicInputError(f"public input exceeds exact size: {row['name']}")
                    digest.update(block)
                    stream.write(block)
        if written != expected_size or digest.hexdigest() != expected_hash:
            raise PublicInputError(f"public input size or hash mismatch: {row['name']}")
    except Exception:
        if created:
            target.unlink(missing_ok=True)
        raise
    return target


def _extract_members(row: dict[str, Any], archive: Path, out_dir: Path) -> list[Path]:
    """Extract only named league members and verify each member's digest."""
    members = row.get("members", [])
    if not members:
        return []
    outputs: list[Path] = []
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            available = {member.name: member for member in bundle.getmembers() if member.isfile()}
            for spec in members:
                name, dest, digest = (
                    spec.get("name"),
                    _safe_dest(spec.get("dest")),
                    spec.get("sha256"),
                )
                member = available.get(name)
                if not isinstance(name, str) or member is None or not isinstance(digest, str):
                    raise PublicInputError(f"league member is absent or unpinned: {name}")
                target = out_dir / dest
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as stream:
                    if source is None:
                        raise PublicInputError(f"unreadable league member: {name}")
                    hasher = hashlib.sha256()
                    while block := source.read(CHUNK_BYTES):
                        hasher.update(block)
                        stream.write(block)
                if hasher.hexdigest() != digest:
                    target.unlink(missing_ok=True)
                    raise PublicInputError(f"league member hash mismatch: {name}")
                outputs.append(target)
    except tarfile.TarError as exc:
        raise PublicInputError("invalid league archive") from exc
    return outputs


def materialize(
    catalog_path: Path,
    out_dir: Path,
    *,
    opener: Callable[..., Any] = urlopen,
    timeout: float = 30.0,
    allow_existing: bool = False,
) -> list[Path]:
    """Download all validated inputs to one fresh portable runtime directory."""
    rows = load_catalog(catalog_path)
    if out_dir.exists() and not allow_existing:
        raise PublicInputError(f"output directory must be fresh: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=allow_existing)
    downloaded = [_download(row, out_dir, opener, timeout) for row in rows]
    for row, archive in zip(rows, downloaded):
        downloaded.extend(_extract_members(row, archive, out_dir))
    for row, path in zip(rows, downloaded):
        if row.get("complete_marker"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_name(path.name + ".sha256").write_text(digest + f"  {path.name}\n")
            path.with_name(path.name + ".complete").write_text("complete\n")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "inputs": [
                    str(path)
                    for path in materialize(
                        args.catalog, args.out_dir, allow_existing=args.allow_existing
                    )
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
