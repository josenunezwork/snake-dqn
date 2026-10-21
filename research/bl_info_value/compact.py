"""Compact, hash-bound transport for the E3 selected T3 rows.

The v1 bundle is deliberately consumed as an immutable source: its selected
observation rows become byte bindings while the checkpoint payload is moved to
a separately published, immutable archive.  This keeps the runtime's
reconstruction inputs small without changing the selected rows or roster.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_info_value.catalog import check_path, unpack
from research.bl_info_value.io import export_bytes, sha256, verify_export, write_json

_OBS_FIELDS = ("local", "global", "scalars", "mask")
_MAX_UNPACKED_BYTES = 512 * 1024**2


def array_binding(value: np.ndarray) -> dict[str, Any]:
    """Return a portable byte-identity binding for one non-object array."""
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("object arrays cannot be hash-bound")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _tar_member(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _tar_gzip(payloads: dict[str, bytes]) -> bytes:
    """Build a deterministic gzip tar from exact member bytes."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for name in sorted(payloads):
                payload = payloads[name]
                archive.addfile(_tar_member(name, payload), io.BytesIO(payload))
    return buffer.getvalue()


def _rows(task: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    """Copy original selected rows in the frozen task-index order."""
    indices = [int(index) for index in task["indices"]]
    if len(indices) != len(set(indices)):
        raise ValueError("selected source indices are not unique")
    if sha256(source) != task["source_sha256"]:
        raise ValueError("source shard digest differs from task binding")
    with np.load(source, allow_pickle=False) as shard:
        missing = set(_OBS_FIELDS).difference(shard.files)
        if missing:
            raise ValueError(f"source shard lacks observations: {sorted(missing)}")
        if any(index < 0 or index >= len(shard["decision"]) for index in indices):
            raise ValueError("selected source index is outside shard")
        result = []
        for index in indices:
            row = {
                name: np.asarray(shard[name][index]).tolist()
                for name in shard.files
                if name not in _OBS_FIELDS
            }
            row["obs_hashes"] = {
                name: array_binding(np.asarray(shard[name][index])) for name in _OBS_FIELDS
            }
            result.append(row)
    return result


def compact(archive: Path, out: Path, expected: str) -> dict[str, Any]:
    """Convert one verified v1 E3 archive into compact inputs and checkpoints.

    ``out`` is create-only.  The source archive must itself have immutable
    supervisor sidecars, then is unpacked through the v1 verifier before any
    selected source row or checkpoint is read.
    """
    archive, out = check_path(Path(archive)), check_path(Path(out))
    verify_export(archive)
    if out.exists():
        raise FileExistsError(out)
    with tempfile.TemporaryDirectory(prefix="bl-e3-compact-") as temporary:
        source_root = Path(temporary) / "source"
        source_catalog = unpack(archive, source_root, expected)
        if source_catalog.get("schema") != "bl-e3-input/v1":
            raise ValueError("compact conversion requires a v1 input archive")
        checkpoint_paths = source_catalog.get("checkpoints")
        if not isinstance(checkpoint_paths, dict) or not checkpoint_paths:
            raise ValueError("v1 catalog lacks checkpoints")
        checkpoint_payloads: dict[str, bytes] = {}
        for name, digest in sorted(checkpoint_paths.items()):
            path = source_root / name
            if not _checkpoint_name(name) or not path.is_file() or sha256(path) != digest:
                raise ValueError(f"checkpoint mismatch: {name}")
            checkpoint_payloads[name] = path.read_bytes()

        tasks = []
        for original in source_catalog["tasks"]:
            task = json.loads(json.dumps(original))
            source = source_root / str(task["source_npz"])
            task["row_bindings"] = _rows(task, source)
            tasks.append(task)
        catalog = dict(source_catalog)
        catalog.update(
            schema="bl-e3-input/v2",
            source_archive_sha256=expected,
            tasks=tasks,
            files={},
            checkpoint_archive={},
        )
        checkpoint_bytes = _tar_gzip(checkpoint_payloads)
        catalog["checkpoint_archive"] = {
            "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "size_bytes": len(checkpoint_bytes),
        }
        catalog_bytes = (json.dumps(catalog, sort_keys=True) + "\n").encode()
        input_bytes = _tar_gzip({"catalog.json": catalog_bytes})

        out.mkdir(parents=True)
        write_json(out / "catalog.json", catalog)
        export_bytes(out / "inputs.tar.gz", input_bytes)
        export_bytes(out / "checkpoints.tar.gz", checkpoint_bytes)
    return catalog


def _safe_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _checkpoint_name(name: str) -> bool:
    """Accept only direct checkpoint files under the fixed archive directory."""
    path = Path(name)
    return _safe_name(name) and path.parts[:1] == ("checkpoints",) and len(path.parts) == 2


def hydrate(catalog: dict[str, Any], inputs: Path, checkpoint_archive: Path) -> None:
    """Verify and extract exactly the separately bound checkpoint archive."""
    expected = catalog.get("checkpoints")
    archive_binding = catalog.get("checkpoint_archive")
    if not isinstance(expected, dict) or not isinstance(archive_binding, dict):
        raise ValueError("compact catalog lacks checkpoint bindings")
    size = archive_binding.get("size_bytes")
    digest = archive_binding.get("sha256")
    if type(size) is not int or size < 1 or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("invalid checkpoint archive binding")
    checkpoint_archive, inputs = Path(checkpoint_archive), Path(inputs)
    if not checkpoint_archive.is_file() or checkpoint_archive.stat().st_size != size:
        raise ValueError("checkpoint archive size mismatch")
    if sha256(checkpoint_archive) != digest:
        raise ValueError("checkpoint archive digest mismatch")
    target = inputs / "checkpoints"
    if target.exists():
        raise ValueError("checkpoint target already exists")
    with tarfile.open(checkpoint_archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("checkpoint archive members differ from catalog")
        if sum(member.size for member in members) > _MAX_UNPACKED_BYTES:
            raise ValueError("checkpoint archive exceeds bound")
        for member in members:
            if not member.isfile() or not _checkpoint_name(member.name):
                raise ValueError("unsafe checkpoint archive member")
            if member.size < 1:
                raise ValueError("empty checkpoint archive member")
        target.mkdir()
        try:
            for member in members:
                destination = inputs / member.name
                if not destination.resolve().is_relative_to(target.resolve()):
                    raise ValueError("checkpoint destination escapes target")
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("checkpoint member is unreadable")
                with destination.open("xb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                if sha256(destination) != expected[member.name]:
                    raise ValueError(f"checkpoint digest mismatch: {member.name}")
        except Exception:
            for path in target.glob("*"):
                path.unlink()
            target.rmdir()
            raise


def main() -> None:
    """Run the create-only compact conversion from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    catalog = compact(args.archive, args.out, args.archive_sha256)
    print(json.dumps({"schema": catalog["schema"], "tasks": len(catalog["tasks"])}))


if __name__ == "__main__":
    main()
