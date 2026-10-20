"""Lossless, hash-validated local replay cache for the T34 fit loop.

The cache deliberately stores a world at a time.  It is therefore useful for a
large replay bank without changing the loader's bounded-memory contract: rows
are reconstructed from memory-mapped arrays only while that world is yielded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np

from .interface import ReplaySource, ReplayWorld, normalize_row, validate_row

_VERSION = 1
_MANIFEST = "manifest.json"
_MANIFEST_SHA = "manifest.sha256"
_ROW_ARRAY_FIELDS = ("local", "global", "scalars", "q_prev", "forbidden_mask", "full_sight")
_OPTIONAL_NUMERIC_COLUMNS = ("hindsight_log_free_region", "hindsight_log_free_region_mask")


class CacheValidationError(RuntimeError):
    """A cache is incomplete, corrupt, or does not belong to these inputs."""


def _canonical(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("cache identity must be JSON-serializable finite data") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_metadata(value: Any) -> Any:
    """Convert nested export metadata without changing its JSON value/types."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        # json.dumps below rejects NaN/inf, which must never silently enter an
        # export cache as metadata.
        json.dumps(value, allow_nan=False)
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("cache metadata mappings require string keys")
        return {key: _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    raise ValueError(f"cache metadata value has unsupported type {type(value).__name__}")


def _world_paths(split: str, index: int) -> tuple[str, str]:
    base = f"{split}/world-{index:07d}"
    return base, f"{base}/metadata.jsonl"


def _write_world(
    root: Path, split: str, index: int, world: ReplayWorld, *, include_tower: bool
) -> dict[str, Any]:
    """Write one retained world and return only small manifest metadata."""
    base, metadata_rel = _world_paths(split, index)
    directory = root / base
    directory.mkdir(parents=True)
    rows = list(world.rows)
    if not rows:
        raise CacheValidationError(f"verified world {world.world_id} has no rows")
    normalized = [dict(normalize_row(row)) for row in rows]
    for row in normalized:
        # Match the loader's np.asarray treatment for required observations and
        # action labels.  This also permits a v2 producer to provide an
        # explicit JSON-list forbidden mask alongside a shadow row whose mask
        # was derived as an ndarray.
        for key in _ROW_ARRAY_FIELDS:
            if key in row:
                row[key] = np.asarray(row[key])
        # NaN is a valid masked auxiliary target.  Keeping these in their own
        # numeric mmap columns preserves its exact bytes rather than trying to
        # encode it in JSON metadata (where NaN is deliberately forbidden).
        for key in _OPTIONAL_NUMERIC_COLUMNS:
            if key in row:
                row[key] = np.asarray(row[key])
    if not include_tower:
        for row in normalized:
            # These fields are only consumed by the optional S1a tower.  Their
            # absence makes an S1b cache materially smaller by design.
            row.pop("full_sight", None)
            row.pop("hindsight_log_free_region", None)
            row.pop("hindsight_log_free_region_mask", None)
    for row in normalized:
        validate_row(row)
    keys = set().union(*(row.keys() for row in normalized))
    array_keys = sorted(
        key for key in keys if any(isinstance(row.get(key), np.ndarray) for row in normalized)
    )
    # A tensor column must exist for every row so one mmap remains a true dense
    # array.  Scalar metadata may naturally differ between LAD and shadow rows.
    if any(
        key not in row or not isinstance(row[key], np.ndarray)
        for row in normalized
        for key in array_keys
    ):
        raise CacheValidationError(f"world {world.world_id} has sparse or mixed array fields")
    artifacts: dict[str, dict[str, Any]] = {}
    for key in array_keys:
        values = [row[key] for row in normalized]
        first = values[0]
        if any(value.dtype != first.dtype or value.shape != first.shape for value in values[1:]):
            raise CacheValidationError(
                f"world {world.world_id} has inconsistent {key} array shape/dtype"
            )
        relative = f"{base}/{key}.npy"
        target = root / relative
        mapped = np.lib.format.open_memmap(
            target, mode="w+", dtype=first.dtype, shape=(len(values), *first.shape)
        )
        for row_index, value in enumerate(values):
            mapped[row_index] = value
        mapped.flush()
        del mapped
        artifacts[relative] = {"sha256": _sha256(target), "bytes": target.stat().st_size}

    offsets: list[int] = []
    metadata_path = root / metadata_rel
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in normalized:
            offsets.append(handle.tell())
            metadata = {
                key: _json_metadata(value) for key, value in row.items() if key not in array_keys
            }
            handle.write(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            handle.write("\n")
    artifacts[metadata_rel] = {
        "sha256": _sha256(metadata_path),
        "bytes": metadata_path.stat().st_size,
    }
    return {
        "world_id": world.world_id,
        "verified": True,
        "hash_expected": world.hash_expected,
        "hash_actual": world.hash_actual,
        "reason": world.reason,
        "chunk_id": world.chunk_id,
        "base": base,
        "metadata": metadata_rel,
        "metadata_offsets": offsets,
        "array_keys": array_keys,
        "rows": len(normalized),
        "artifacts": artifacts,
    }


def _drop_check(chunks: Mapping[str, list[int]]) -> None:
    bad = {name: counts for name, counts in chunks.items() if counts[1] / counts[0] > 0.02}
    if bad:
        raise CacheValidationError(
            f"replay hash mismatch rate exceeds 2% in complete chunk(s): {bad}"
        )


def _cache_split(
    root: Path,
    source: ReplaySource,
    split: str,
    worlds: Iterable[ReplayWorld],
    *,
    include_tower: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[int]]]:
    retained: list[dict[str, Any]] = []
    counts = {"searched": 0, "unsearched": 0, "rows": 0, "worlds_seen": 0, "worlds_dropped": 0}
    chunks: dict[str, list[int]] = {}
    for world in worlds:
        counts["worlds_seen"] += 1
        chunk = world.chunk_id or "unpartitioned"
        chunks.setdefault(chunk, [0, 0])[0] += 1
        if not world.verified:
            counts["worlds_dropped"] += 1
            chunks[chunk][1] += 1
            continue
        if not bool(getattr(source, "is_fake", False)) and (
            len(world.hash_expected) != 64
            or len(world.hash_actual) != 64
            or world.hash_expected != world.hash_actual
        ):
            raise CacheValidationError(
                f"verified production world {world.world_id} lacks matching sha256 proof"
            )
        entry = _write_world(root, split, len(retained), world, include_tower=include_tower)
        retained.append(entry)
        counts["rows"] += entry["rows"]
        # Every row was normalized and validated in _write_world.
        for row in world.rows:
            normalized = normalize_row(row)
            counts["searched" if bool(normalized["searched"]) else "unsearched"] += 1
    _drop_check(chunks)
    return retained, counts, chunks


def build_cache(
    source: ReplaySource,
    path: str | Path,
    *,
    identity: dict[str, Any],
    include_tower: bool = False,
) -> "CachedReplaySource":
    """Regenerate a source exactly once into a create-only local cache.

    ``identity`` is caller-owned input/export provenance.  It is compared byte
    for byte (canonical JSON) when reopening, so a cache can never be reused
    with a different export or frozen model input.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"cache target already exists (create-only): {destination}")
    if not isinstance(identity, dict):
        raise TypeError("cache identity must be a dict")
    identity_json = _canonical(identity)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    started = perf_counter()
    try:
        training, train_counts, chunks = _cache_split(
            temporary, source, "training", source.replay_epoch(1), include_tower=include_tower
        )
        inventory = getattr(source, "training_row_counts", None)
        has_inventory = callable(inventory)
        if has_inventory:
            expected = inventory(1)
            actual = (train_counts["searched"], train_counts["unsearched"])
            if tuple(expected) != actual:
                raise CacheValidationError(
                    "training_row_counts drift: counts must describe verified retained stream; "
                    f"expected={expected}, actual={actual}"
                )
        heldout_method = getattr(source, "replay_heldout", None)
        heldout: list[dict[str, Any]] = []
        heldout_counts: dict[str, int] | None = None
        heldout_chunks: dict[str, list[int]] | None = None
        if callable(heldout_method):
            heldout, heldout_counts, heldout_chunks = _cache_split(
                temporary, source, "heldout", heldout_method(1), include_tower=include_tower
            )
        artifacts = {
            artifact: proof
            for entry in (*training, *heldout)
            for artifact, proof in entry["artifacts"].items()
        }
        byte_count = sum(int(proof["bytes"]) for proof in artifacts.values())
        array_bytes = sum(
            int(proof["bytes"])
            for artifact, proof in artifacts.items()
            if artifact.endswith(".npy")
        )
        metadata_bytes = byte_count - array_bytes
        heldout_rows = 0 if heldout_counts is None else heldout_counts["rows"]
        total_rows = train_counts["rows"] + heldout_rows
        manifest = {
            "version": _VERSION,
            "complete": True,
            "identity": json.loads(identity_json),
            "identity_canonical": identity_json,
            "include_tower": bool(include_tower),
            "is_fake": bool(getattr(source, "is_fake", False)),
            "has_training_row_counts": has_inventory,
            "training": training,
            "heldout": heldout,
            "artifacts": artifacts,
            "stats": {
                "regen_seconds": perf_counter() - started,
                "rows": total_rows,
                "training_rows": train_counts["rows"],
                "heldout_rows": heldout_rows,
                "bytes": byte_count,
                "array_bytes": array_bytes,
                "metadata_bytes": metadata_bytes,
                "bytes_per_row": byte_count / max(1, total_rows),
                "regen_seconds_per_row": (perf_counter() - started) / max(1, total_rows),
                "training": train_counts,
                "training_chunks": chunks,
                "heldout": heldout_counts,
                "heldout_chunks": heldout_chunks,
            },
        }
        with (temporary / _MANIFEST).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        manifest_sha = _sha256(temporary / _MANIFEST)
        (temporary / _MANIFEST_SHA).write_text(f"{manifest_sha}  {_MANIFEST}\n", encoding="ascii")
        # Rename makes a complete manifest appear only after every data file is closed.
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return CachedReplaySource(destination, identity=identity)


class CachedReplaySource:
    """Read-only replay source backed by a verified, lossless mmap cache."""

    def __init__(self, path: str | Path, *, identity: dict[str, Any]) -> None:
        self.path = Path(path)
        manifest_path = self.path / _MANIFEST
        if not manifest_path.is_file():
            raise CacheValidationError(f"incomplete cache: missing {_MANIFEST}")
        manifest_sidecar = self.path / _MANIFEST_SHA
        if not manifest_sidecar.is_file():
            raise CacheValidationError(f"incomplete cache: missing {_MANIFEST_SHA}")
        parts = manifest_sidecar.read_text(encoding="ascii").split()
        if len(parts) != 2 or parts[1] != _MANIFEST:
            raise CacheValidationError("invalid cache manifest sha sidecar")
        expected_manifest_sha = parts[0]
        if len(expected_manifest_sha) != 64 or _sha256(manifest_path) != expected_manifest_sha:
            raise CacheValidationError("cache manifest hash mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheValidationError("invalid cache manifest") from exc
        if manifest.get("version") != _VERSION or manifest.get("complete") is not True:
            raise CacheValidationError("incomplete or unsupported cache manifest")
        if manifest.get("identity_canonical") != _canonical(identity):
            raise CacheValidationError("cache identity mismatch; refusing stale cache")
        if manifest.get("identity") != identity:
            raise CacheValidationError("cache identity serialization mismatch")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise CacheValidationError("cache manifest has no artifacts")
        for relative, proof in artifacts.items():
            candidate = self.path / relative
            if not candidate.is_file() or _sha256(candidate) != proof.get("sha256"):
                raise CacheValidationError(f"cache artifact hash mismatch: {relative}")
        self.manifest: dict[str, Any] = manifest
        self.is_fake = bool(manifest.get("is_fake", False))
        self.cache_is_fake = self.is_fake
        if bool(manifest.get("has_training_row_counts")):
            self.training_row_counts = self._training_row_counts

    def __getattribute__(self, name: str) -> Any:
        # Do not let a cache invent a held-out interface its original source
        # did not have.  This matters for fake preflights, whose normal fit
        # fallback is the training stream.
        if name == "replay_heldout":
            manifest = object.__getattribute__(self, "manifest")
            if not manifest.get("heldout"):
                raise AttributeError("source did not cache a held-out replay stream")
        return object.__getattribute__(self, name)

    def _training_row_counts(self, epoch: int) -> tuple[int, int]:
        del epoch
        counts = self.manifest["stats"]["training"]
        return int(counts["searched"]), int(counts["unsearched"])

    def _replay(self, split: str) -> Iterable[ReplayWorld]:
        for entry in self.manifest.get(split, []):
            arrays = {
                key: np.load(
                    self.path / entry["base"] / f"{key}.npy", mmap_mode="r", allow_pickle=False
                )
                for key in entry["array_keys"]
            }
            metadata_path = self.path / entry["metadata"]
            rows = []
            with metadata_path.open("r", encoding="utf-8") as handle:
                for row_index, line in enumerate(handle):
                    metadata = json.loads(line)
                    rows.append(
                        {**metadata, **{key: values[row_index] for key, values in arrays.items()}}
                    )
            if len(rows) != int(entry["rows"]):
                raise CacheValidationError(
                    f"cache metadata row count mismatch: {entry['world_id']}"
                )
            yield ReplayWorld(
                world_id=entry["world_id"],
                rows=rows,
                verified=True,
                hash_expected=entry.get("hash_expected", ""),
                hash_actual=entry.get("hash_actual", ""),
                reason=entry.get("reason", ""),
                chunk_id=entry.get("chunk_id", ""),
            )

    def replay_epoch(self, epoch: int) -> Iterable[ReplayWorld]:
        """Epoch has no effect: cached rows are the stable source sequence."""
        del epoch
        yield from self._replay("training")

    def replay_heldout(self, epoch: int) -> Iterable[ReplayWorld]:
        del epoch
        yield from self._replay("heldout")
