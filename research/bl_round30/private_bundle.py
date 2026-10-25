"""Deterministic, private R30 input bundles with fail-closed extraction.

The normal metadata archive intentionally rejects scientific-sized payloads.
R30 transports the admitted *training-only* input directory as one private
bundle instead.  The bundle is lossless: files are copied byte-for-byte and an
internal allowlist binds every destination, size, and SHA-256 before anything
is published in the runtime input directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "bl-round30-private-input-bundle/v1"
MANIFEST = ".r30-private-bundle-manifest.json"
MAX_FILES = 64
# This is only a hard archive safety ceiling.  The pack additionally rejects
# each actual bundle using the recent verified L4 upload measurement and the
# ordinary 150-second bar.  Only the exact-input native L4 canary can prove
# the bar; whole staging remains independently capped at 300 seconds.
MAX_BUNDLE_BYTES = 48 * 1024**2
MAX_UNPACKED_BYTES = 128 * 1024**2
_BASE_NAMES = {
    "control.json",
    "inventory.json",
    "approval.json",
    "verified-ledger.json",
    "series.json",
    "train.npz",
    "successor-gate.json",
    "diagnosis.json",
    "unit-0000.pth",
    "allocation.json",
    "partition-manifest.json",
    "coverage-approval.json",
    "life-id-map.json",
    "sampler-plan.npz",
    "sampler-plan.json",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or PurePosixPath(name).is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or name == MANIFEST
    ):
        raise ValueError("unsafe private-bundle destination")
    return name


def _allowlisted_name(name: str) -> str:
    """Restrict private transport to the worker's declared input vocabulary."""
    name = _name(name)
    base = name.removesuffix(".complete").removesuffix(".sha256")
    if base not in _BASE_NAMES:
        raise ValueError("foreign private-bundle destination")
    return name


def _target(root: Path, name: str) -> Path:
    target = root / name
    for candidate in (target, *target.parents):
        if candidate.is_symlink():
            raise ValueError("symlink in private-bundle destination")
        if candidate != target and candidate.exists() and not candidate.is_dir():
            raise ValueError("non-directory private-bundle parent")
        if candidate == root:
            break
    return target


def _entries(source: Path, *, exclude: set[str]) -> list[tuple[str, Path, int, str]]:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("private-bundle source must be a real directory")
    rows = []
    for path in sorted(source.iterdir()):
        if path.name in exclude:
            continue
        name = _allowlisted_name(path.name)
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe private-bundle source type")
        rows.append((name, path, path.stat().st_size, sha256(path)))
    if not rows or len(rows) > MAX_FILES:
        raise ValueError("empty or excessive private-bundle inventory")
    if sum(row[2] for row in rows) > MAX_UNPACKED_BYTES:
        raise ValueError("private-bundle unpacked inputs exceed R30 transport bound")
    return rows


def bundle(source: Path, archive: Path, *, exclude: set[str] | None = None) -> dict[str, Any]:
    """Create one deterministic private gzip/tar bundle without overwriting evidence."""
    excluded = set() if exclude is None else set(exclude)
    entries = _entries(source, exclude=excluded)
    archive = Path(archive)
    if archive.exists() or archive.is_symlink():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": SCHEMA,
        "files": [
            {"path": name, "size": size, "sha256": checksum} for name, _, size, checksum in entries
        ],
    }
    with tempfile.NamedTemporaryFile(dir=archive.parent) as temporary:
        with gzip.GzipFile(fileobj=temporary, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                payload = (json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n").encode()
                record = tarfile.TarInfo(MANIFEST)
                record.size, record.mode, record.mtime = len(payload), 0o444, 0
                record.uid = record.gid = 0
                record.uname = record.gname = ""
                tar.addfile(record, io.BytesIO(payload))
                for name, path, size, checksum in entries:
                    record = tarfile.TarInfo(name)
                    record.size, record.mode, record.mtime = size, 0o444, 0
                    record.uid = record.gid = 0
                    record.uname = record.gname = ""
                    with path.open("rb") as stream:
                        tar.addfile(record, stream)
                    if path.stat().st_size != size or sha256(path) != checksum:
                        raise ValueError("private-bundle source changed while bundling")
        temporary.flush()
        os.fsync(temporary.fileno())
        if temporary.tell() > MAX_BUNDLE_BYTES:
            raise ValueError("private-bundle exceeds conservative standard-bar size cap")
        os.link(temporary.name, archive)
    return {
        "schema": SCHEMA,
        "path": str(archive.resolve()),
        "sha256": sha256(archive),
        "size_bytes": archive.stat().st_size,
        "unpacked_bytes": sum(row[2] for row in entries),
        "files": len(entries),
    }


def unpack(archive: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify an allowlisted bundle completely, then publish members atomically."""
    archive, root = Path(archive), Path(root)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size > MAX_BUNDLE_BYTES
        or sha256(archive) != expected_sha256
        or root.is_symlink()
        or not root.is_dir()
    ):
        raise ValueError("private-bundle hash/type/size differs")
    with tempfile.TemporaryDirectory(prefix=".r30-private-verify-", dir=root) as temporary:
        staged = Path(temporary)
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if not members or len(members) > MAX_FILES + 1 or any(not m.isfile() for m in members):
                raise ValueError("private-bundle links/special/excessive members")
            by_name = {member.name: member for member in members}
            if len(by_name) != len(members) or MANIFEST not in by_name:
                raise ValueError("private-bundle missing manifest or duplicate member")
            stream = tar.extractfile(by_name.pop(MANIFEST))
            assert stream is not None
            manifest = json.load(stream)
            if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
                raise ValueError("foreign private-bundle manifest")
            rows = manifest.get("files")
            if not isinstance(rows, list) or not rows or len(rows) > MAX_FILES:
                raise ValueError("invalid private-bundle allowlist")
            names, total = [], 0
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
                    raise ValueError("invalid private-bundle row")
                name = _allowlisted_name(row["path"])
                size, checksum = row["size"], row["sha256"]
                if (
                    type(size) is not int
                    or not 0 <= size <= MAX_UNPACKED_BYTES
                    or not re.fullmatch(r"[0-9a-f]{64}", str(checksum))
                ):
                    raise ValueError("invalid private-bundle size/digest")
                names.append(name)
                total += size
            if (
                len(set(names)) != len(names)
                or total > MAX_UNPACKED_BYTES
                or set(names) != set(by_name)
            ):
                raise ValueError("private-bundle allowlist/member set differs")
            for row in rows:
                name = row["path"]
                target = _target(root, name)
                if target.exists():
                    raise FileExistsError(target)
                member = by_name[name]
                if member.size != row["size"]:
                    raise ValueError("private-bundle member size differs")
                staged_path = staged / name
                stream = tar.extractfile(member)
                assert stream is not None
                with staged_path.open("xb") as output:
                    while chunk := stream.read(1024**2):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256(staged_path) != row["sha256"]:
                    raise ValueError("private-bundle member digest differs")
            # Preserve payload -> checksum -> completion publication.  A
            # runtime failure can therefore never expose a false completion
            # marker for an incompletely published export triple.
            for name in sorted(
                names,
                key=lambda value: (value.endswith(".complete"), value.endswith(".sha256"), value),
            ):
                os.link(staged / name, _target(root, name))
    return {"status": "VERIFIED", "files": len(names), "bytes": total, "sha256": expected_sha256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("unpack",))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.root, args.sha256), sort_keys=True))


if __name__ == "__main__":
    main()
