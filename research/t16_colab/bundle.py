#!/usr/bin/env python3
"""Create and safely unpack the non-git inputs for the T16 Colab smoke.

The archive intentionally contains only the checkpoint and demo bytes that the
T16 runner reads.  The manifest is verified in full before any payload is
written to the extraction directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Sequence

SCHEMA = "t16-colab-inputs/v1"
ROOT = "t16-colab-inputs"
MANIFEST_NAME = f"{ROOT}/manifest.json"
ARTIFACT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
CHECKPOINT_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")


class BundleError(RuntimeError):
    """Raised when a T16 input bundle is malformed or cannot be verified."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path`` without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise BundleError(f"unsafe archive path: {value!r}")
    return path


def _archive_name(relative: str) -> str:
    return str(PurePosixPath(ROOT) / _safe_relative(relative))


def default_inputs(
    artifacts: Path = ARTIFACT_ROOT, repo: Path = CHECKPOINT_ROOT.parent
) -> Dict[str, Path]:
    """Return the minimal external file set read by the T16 notebook.

    ``repo`` supplies the source ``saved_snakes`` directory used by the Grid
    identity fixture. It defaults to the established source checkout so a T16
    worktree can export the bundle without copying those immutable inputs.
    """
    demo = artifacts / "redesign-m2b-20261008" / "data"
    checkpoints = repo / "saved_snakes"
    return {
        "init/B8_s3_final.pth": artifacts / "redesign-m4b-mac-20261013" / "B8_s3" / "final.pth",
        "demo/margin.json": demo / "margin.json",
        "demo/round-12/chunk-0000-part-00.npz": demo / "round-12" / "chunk-0000-part-00.npz",
        "checkpoints/frp3_m3_s12_u60000_20261005.pth": checkpoints
        / "frp3_m3_s12_u60000_20261005.pth",
        "checkpoints/champion_a5_freespace_20260621.pth": checkpoints
        / "champion_a5_freespace_20260621.pth",
        "checkpoints/best_apex_fs.pth": checkpoints / "best_apex_fs.pth",
        "checkpoints/best_apex_stage1_fs.pth": checkpoints / "best_apex_stage1_fs.pth",
        "checkpoints/best_apex_pre_fs.pth": checkpoints / "best_apex_pre_fs.pth",
    }


def export_bundle(output: Path, inputs: Mapping[str, Path]) -> dict:
    """Write a compressed bundle and return its manifest.

    Args:
        output: Destination ``.tar.gz`` path.
        inputs: Mapping from relative archive path to an existing regular file.
    """
    if not inputs:
        raise BundleError("refusing to write an empty input bundle")
    entries = []
    normalized: list[tuple[str, Path]] = []
    for relative, source in sorted(inputs.items()):
        _safe_relative(relative)
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise BundleError(f"required input is not a regular file: {source}")
        normalized.append((relative, source))
        entries.append(
            {
                "path": relative,
                "sha256": sha256_file(source),
                "size_bytes": source.stat().st_size,
            }
        )
    manifest = {"schema": SCHEMA, "files": entries}
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stable gzip header and tar metadata let the owner re-export the exact same
    # bytes and use the notebook's outer archive pin, independent of local mtimes.
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
        for relative, source in normalized:
            info = archive.gettarinfo(str(source), arcname=_archive_name(relative))
            if not info.isreg():
                raise BundleError(f"required input is not a regular file: {source}")
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            info.pax_headers = {}
            with source.open("rb") as handle:
                archive.addfile(info, handle)
    return manifest


def _load_and_verify(archive: tarfile.TarFile) -> tuple[dict, dict[str, tarfile.TarInfo]]:
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise BundleError("archive has duplicate member names")
    by_name = {member.name: member for member in members}
    manifest_member = by_name.get(MANIFEST_NAME)
    if manifest_member is None or not manifest_member.isreg():
        raise BundleError("archive has no regular manifest")
    stream = archive.extractfile(manifest_member)
    if stream is None:
        raise BundleError("cannot read archive manifest")
    try:
        manifest = json.loads(stream.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError("manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise BundleError("unexpected manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise BundleError("manifest must declare at least one file")
    expected = {MANIFEST_NAME}
    verified: dict[str, tarfile.TarInfo] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise BundleError("manifest file entry is not an object")
        relative = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        fields_valid = (
            isinstance(relative, str) and isinstance(digest, str) and isinstance(size, int)
        )
        if not fields_valid:
            raise BundleError("manifest file entry has invalid fields")
        member_name = _archive_name(relative)
        if member_name in expected:
            raise BundleError(f"duplicate manifest path: {relative}")
        expected.add(member_name)
        member = by_name.get(member_name)
        if member is None or not member.isreg() or member.size != size:
            raise BundleError(f"missing or malformed payload: {relative}")
        payload = archive.extractfile(member)
        if payload is None:
            raise BundleError(f"cannot read payload: {relative}")
        hasher = hashlib.sha256()
        for chunk in iter(lambda: payload.read(1024 * 1024), b""):
            hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise BundleError(f"SHA-256 mismatch: {relative}")
        verified[relative] = member
    if set(names) != expected:
        raise BundleError("archive contains members not declared by its manifest")
    return manifest, verified


def verify_and_extract(archive_path: Path, destination: Path) -> Path:
    """Verify every declared payload before writing the bundle below ``destination``.

    The returned directory is ``destination / t16-colab-inputs``. Existing
    payloads are replaced only after the archive has completely verified.
    """
    archive_path, destination = Path(archive_path), Path(destination)
    with tarfile.open(archive_path, "r:gz") as archive:
        _, verified = _load_and_verify(archive)
        target = destination / ROOT
        staging = destination / f".{ROOT}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            for relative, member in verified.items():
                output = staging / Path(relative)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise BundleError(f"cannot read verified payload: {relative}")
                with output.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
            manifest_member = archive.getmember(MANIFEST_NAME)
            manifest_source = archive.extractfile(manifest_member)
            if manifest_source is None:
                raise BundleError("cannot read verified manifest")
            (staging / "manifest.json").write_bytes(manifest_source.read())
            if target.exists():
                shutil.rmtree(target)
            staging.replace(target)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    return target


def _parse_input(value: str) -> tuple[str, Path]:
    try:
        relative, source = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--input must be ARCHIVE_PATH=SOURCE_PATH") from error
    try:
        _safe_relative(relative)
    except BundleError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return relative, Path(source)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export or verified-extract command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="write the minimal T16 input tarball")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--artifacts", type=Path, default=ARTIFACT_ROOT)
    export.add_argument("--repo", type=Path, default=CHECKPOINT_ROOT.parent)
    export.add_argument("--input", type=_parse_input, action="append", default=[])
    unpack = subparsers.add_parser("extract", help="verify every payload before extraction")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        inputs = default_inputs(args.artifacts, args.repo)
        inputs.update(dict(args.input))
        manifest = export_bundle(args.output, inputs)
        print(
            json.dumps(
                {
                    "archive": str(args.output),
                    "files": len(manifest["files"]),
                    "sha256": sha256_file(args.output),
                    "size_bytes": args.output.stat().st_size,
                }
            )
        )
        return 0
    target = verify_and_extract(args.archive, args.destination)
    print(json.dumps({"input_root": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
