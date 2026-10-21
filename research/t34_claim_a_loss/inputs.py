"""Pack and unpack the excluded A5 v2 exports as one hash-bound D0 input.

Mac side (:func:`pack`): every complete world export under the given roots (world
JSON + ``.sha256`` sidecar + ``.complete`` marker) is checked against its sidecar,
must belong to the excluded A5 v2 namespace (``t34-exit-a5/v2/``) and carry B8 s3
as carrier and rollout policy.  The files go into a deterministic ``tar.gz``
(sorted members, mtime 0, uid/gid 0, mode 0644) with an ``index.json`` listing each
world's id, relative path and SHA-256.  The tarball is a supervisor-uploaded input.

Runtime side (:func:`unpack`): the tarball's SHA-256 must equal the manifest's,
members must be plain files under safe relative paths, and every world must match
its index digest.  Nothing else is read.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "t34-claim-a-d0-inputs/v1"
A5_PREFIX = "t34-exit-a5/v2/"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
EXPECTED_A5_WORLDS = 39


class InputsRefused(ValueError):
    """The exports cannot be packed or unpacked as the registered D0 input."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _world_files(roots: list[Path]) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for number, root in enumerate(roots):
        for path in sorted(Path(root).resolve(strict=True).rglob("world-*.json")):
            if path.with_name(path.name + ".complete").is_file():
                found.append((number, path))
    return found


def pack(
    roots: list[Path],
    out: Path,
    *,
    expect_worlds: int = EXPECTED_A5_WORLDS,
    b8_sha256: str = B8_SHA256,
    prefix: str = A5_PREFIX,
) -> dict[str, Any]:
    files = _world_files(roots)
    if len(files) != expect_worlds:
        raise InputsRefused(f"expected {expect_worlds} complete A5 worlds, found {len(files)}")
    members: dict[str, bytes] = {}
    index = []
    for number, (root_number, path) in enumerate(files):
        data = path.read_bytes()
        digest = sha256_bytes(data)
        sidecar = path.with_name(path.name + ".sha256").read_text().split()[0]
        if sidecar != digest:
            raise InputsRefused(f"{path}: bytes differ from the .sha256 sidecar")
        raw = json.loads(data)
        world_id = str(raw["world_id"])
        if not world_id.startswith(prefix):
            raise InputsRefused(f"{world_id} is not an excluded A5 v2 world")
        if str(raw["checkpoint_sha256"]) != b8_sha256:
            raise InputsRefused(f"{world_id}: carrier is not B8 s3")
        if str(raw.get("rollout_policy_sha256", raw["checkpoint_sha256"])) != b8_sha256:
            raise InputsRefused(f"{world_id}: rollout policy is not B8 s3")
        # One flat directory per source root keeps member names short and unique.
        rel = f"worlds/r{root_number}-{number:03d}/{path.name}"
        members[rel] = data
        members[rel + ".sha256"] = f"{digest}  {path.name}\n".encode()
        members[rel + ".complete"] = path.with_name(path.name + ".complete").read_bytes()
        index.append({"world_id": world_id, "path": rel, "sha256": digest, "source": str(path)})
    ids = [item["world_id"] for item in index]
    if len(set(ids)) != len(ids):
        raise InputsRefused("duplicate world ids in the A5 exports")
    index_doc = {
        "schema": SCHEMA,
        "worlds": sorted(index, key=lambda item: item["world_id"]),
        "b8_sha256": b8_sha256,
        "prefix": prefix,
    }
    members["index.json"] = (json.dumps(index_doc, indent=2, sort_keys=True) + "\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(members):
                info = tarfile.TarInfo(name)
                info.size = len(members[name])
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(members[name]))
    payload = buffer.getvalue()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(payload)
    return {
        "path": str(out.resolve()),
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
        "worlds": len(index),
        "index_sha256": sha256_bytes(members["index.json"]),
    }


def unpack(tarball: Path, sha256: str, dest: Path) -> list[Path]:
    """Verify and extract; return the world JSON paths in index order."""
    if sha256_file(tarball) != sha256:
        raise InputsRefused("D0 input tarball does not match its manifest SHA-256")
    dest.mkdir(parents=True, exist_ok=False)
    with tarfile.open(tarball, mode="r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise InputsRefused(f"unsafe tar member {member.name!r}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            assert source is not None
            with target.open("xb") as stream:
                stream.write(source.read())
    index = json.loads((dest / "index.json").read_text())
    if index.get("schema") != SCHEMA:
        raise InputsRefused("D0 input index has the wrong schema")
    paths = []
    for item in index["worlds"]:
        path = dest / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise InputsRefused(f"{item['world_id']}: extracted bytes differ from the index")
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-worlds", type=int, default=EXPECTED_A5_WORLDS)
    args = parser.parse_args()
    print(json.dumps(pack(args.root, args.out, expect_worlds=args.expect_worlds), sort_keys=True))


if __name__ == "__main__":
    main()
