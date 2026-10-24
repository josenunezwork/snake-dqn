"""Package-side, create-only metadata bundles; independent of Colab transport.

The supervisor binds the tar digest. The package verifies that digest again,
validates every member before publication, and leaves original ledger readers
and sidecar checks in place. Large payloads and public-fetch catalogues stay
as direct supervisor inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "snake-dqn-metadata-archive/v1"
INVENTORY = ".metadata-manifest.json"
MAX_FILE_BYTES = 2 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
MAX_FILES = 4096


def sha256(path: Path) -> str:
    """Hash without interpreting payloads or loading model objects."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or value in {INVENTORY, "metadata.tar"}
    ):
        raise ValueError("unsafe or reserved metadata destination")
    return value


def _unique(names: list[str]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("duplicate metadata destination")
    values = set(names)
    for name in names:
        if any(str(parent) in values for parent in PurePosixPath(name).parents):
            raise ValueError("overlapping metadata destinations")


def _safe_target(root: Path, name: str) -> Path:
    target = root / name
    for path in (target, *target.parents):
        if path.is_symlink():
            raise ValueError("symlink in metadata destination")
        if path != target and path.exists() and not path.is_dir():
            raise ValueError("non-directory metadata parent")
        if path == root:
            break
    return target


def _sidecars(staged: Path, root: Path, names: set[str]) -> None:
    """Recheck intact export triples, including directly staged large payloads."""
    bases = {
        name.removesuffix(suffix)
        for name in names
        for suffix in (".sha256", ".complete")
        if name.endswith(suffix)
    }
    for base in sorted(bases):
        paths = []
        for suffix in ("", ".sha256", ".complete"):
            name = base + suffix
            path = staged / name if name in names else _safe_target(root, name)
            if path.is_symlink() or not path.is_file():
                raise ValueError("missing or unsafe metadata export component")
            paths.append(path)
        payload, sidecar, marker = paths
        if marker.read_bytes() != b"complete\n":
            raise ValueError("invalid metadata completion marker")
        if sidecar.read_text().split() != [sha256(payload), payload.name]:
            raise ValueError("metadata export digest/name mismatch")


def bundle_inputs(
    inputs: list[dict[str, Any]], archive: Path, *, reuse_verified: bool = False
) -> list[dict[str, Any]]:
    """Replace small local rows with one deterministic hash-bound metadata tar.

    Byte-identical sidecars and markers are members, not regenerated. Direct
    URL inputs, public-inputs.json and payloads above 2 MiB remain direct rows.
    The tar is atomically create-only; its local sidecars are not upload rows.
    """
    names = [_name(row["dest"]) for row in inputs]
    _unique(names)
    selected = []
    retained = []
    for row in inputs:
        if "url" in row:
            retained.append(dict(row))
            continue
        path = Path(row["path"])
        if path.is_symlink() or not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("metadata source hash/type differs")
        if path.stat().st_size > MAX_FILE_BYTES or row["dest"] == "public-inputs.json":
            retained.append(dict(row))
        else:
            selected.append((row["dest"], path, row["sha256"], path.stat().st_size))
    if not selected or len(selected) > MAX_FILES:
        raise ValueError("empty or excessive metadata inventory")
    selected.sort()
    total = sum(row[3] for row in selected)
    if total > MAX_TOTAL_BYTES:
        raise ValueError("metadata exceeds unpacked byte bound")
    manifest = {
        "schema": SCHEMA,
        "files": [
            {"path": name, "sha256": digest, "size": size} for name, _, digest, size in selected
        ],
    }
    archive = Path(archive)
    existing = any(
        Path(str(archive) + suffix).exists() or Path(str(archive) + suffix).is_symlink()
        for suffix in ("", ".sha256", ".complete")
    )
    if existing and not reuse_verified:
        raise FileExistsError(archive)
    if existing:
        _sidecars(
            archive.parent, archive.parent, {archive.name + ".sha256", archive.name + ".complete"}
        )
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent) as temporary:
        with tarfile.open(fileobj=temporary, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            data = (json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n").encode()
            member = tarfile.TarInfo(INVENTORY)
            member.size, member.mode = len(data), 0o444
            tar.addfile(member, io.BytesIO(data))
            for name, path, digest, size in selected:
                member = tarfile.TarInfo(name)
                member.size, member.mode = size, 0o444
                with path.open("rb") as stream:
                    tar.addfile(member, stream)
                if path.stat().st_size != size or sha256(path) != digest:
                    raise ValueError("metadata source changed while bundling")
        temporary.flush()
        os.fsync(temporary.fileno())
        if existing:
            if sha256(Path(temporary.name)) != sha256(archive):
                raise ValueError("immutable metadata archive differs")
        else:
            os.link(temporary.name, archive)
    if existing:
        return [
            *retained,
            {"path": str(archive.resolve()), "dest": "metadata.tar", "sha256": sha256(archive)},
        ]
    checksum = sha256(archive)
    for suffix, content in (
        (".sha256", f"{checksum}  {archive.name}\n".encode()),
        (".complete", b"complete\n"),
    ):
        with tempfile.NamedTemporaryFile(dir=archive.parent) as sidecar:
            sidecar.write(content)
            sidecar.flush()
            os.fsync(sidecar.fileno())
            os.link(sidecar.name, Path(str(archive) + suffix))
    directory = os.open(archive.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return [
        *retained,
        {"path": str(archive.resolve()), "dest": "metadata.tar", "sha256": sha256(archive)},
    ]


def unpack(archive: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    """Fully verify a bound tar, then publish original files, markers last.

    Refuses duplicate/foreign/missing members, links, special files, path escapes,
    digest/size mismatches and existing destinations. No tar extraction API is
    used. Failure during validation publishes no members; late conflicts cannot
    overwrite files and completion markers are always published last.
    """
    archive, root = Path(archive), Path(root)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or archive.is_symlink()
        or not archive.is_file()
        or archive.stat().st_size > MAX_TOTAL_BYTES + 8 * 1024**2
        or sha256(archive) != expected_sha256
    ):
        raise ValueError("metadata archive hash/type/size differs")
    if root.is_symlink() or not root.is_dir() or any(p.is_symlink() for p in root.parents):
        raise ValueError("metadata root must be an existing real directory")
    with tempfile.TemporaryDirectory(prefix=".metadata-verify-", dir=root) as temporary:
        staged = Path(temporary)
        with tarfile.open(archive, "r:") as tar:
            members = tar.getmembers()
            if len(members) > MAX_FILES + 1 or not members:
                raise ValueError("excessive or empty metadata archive")
            if any(not member.isfile() for member in members):
                raise ValueError("metadata archive links/special files are forbidden")
            by_name = {member.name: member for member in members}
            if len(by_name) != len(members) or INVENTORY not in by_name:
                raise ValueError("duplicate members or missing internal manifest")
            inventory_member = by_name.pop(INVENTORY)
            if not 0 < inventory_member.size <= MAX_FILE_BYTES:
                raise ValueError("metadata manifest size differs")
            inventory_stream = tar.extractfile(inventory_member)
            assert inventory_stream is not None
            manifest = json.load(inventory_stream)
            if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
                raise ValueError("foreign metadata manifest")
            rows = manifest.get("files")
            if not isinstance(rows, list) or not 0 < len(rows) <= MAX_FILES:
                raise ValueError("invalid metadata file inventory")
            names = []
            total = 0
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
                    raise ValueError("invalid metadata inventory row")
                name = _name(row["path"])
                size, digest = row["size"], row["sha256"]
                if (
                    type(size) is not int
                    or not 0 <= size <= MAX_FILE_BYTES
                    or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                ):
                    raise ValueError("invalid metadata size/digest")
                names.append(name)
                total += size
            _unique(names)
            if total > MAX_TOTAL_BYTES or set(names) != set(by_name):
                raise ValueError("metadata manifest/member set or total differs")
            for row in rows:
                name = row["path"]
                target = _safe_target(root, name)
                if target.exists():
                    raise FileExistsError(target)
                member = by_name[name]
                if member.size != row["size"]:
                    raise ValueError("metadata member size differs")
                path = staged / name
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(member)
                assert stream is not None
                with path.open("xb") as output:
                    while chunk := stream.read(1024**2):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if sha256(path) != row["sha256"]:
                    raise ValueError("metadata member digest differs")
        _sidecars(staged, root, set(names))
        # Stable ordering preserves payload -> sidecar -> completion semantics.
        for name in sorted(
            names, key=lambda n: (n.endswith(".complete"), n.endswith(".sha256"), n)
        ):
            target = _safe_target(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged / name, target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    return {"status": "VERIFIED", "files": len(names), "bytes": total, "sha256": expected_sha256}


def main() -> None:
    """Run the exact package-side VM admission path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["unpack"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.root, args.sha256), sort_keys=True))


if __name__ == "__main__":
    main()
