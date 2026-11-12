"""Game-independent create-only artifact, hard-link, and rotating JSONL export publishers."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import research.arena_ppo.runtime as _arena_rt
from research.arena_ppo.runtime import sha256_file


def artifact_or_sidecar_exists(path: Path | str) -> bool:
    """Return ``True`` if ``path`` or either of its ``.sha256`` / ``.complete`` sidecars exists."""
    target = Path(path)
    return (
        target.exists() or Path(f"{target}.sha256").exists() or Path(f"{target}.complete").exists()
    )


_artifact_or_sidecar_exists = artifact_or_sidecar_exists


def _publish_sidecars_for_existing_payload(target: Path, digest: str) -> None:
    """Publish create-only ``.sha256`` and ``.complete`` sidecars with single-inode stability.

    Uses ``os.replace`` from a single-inode temp file after verifying non-existence so
    ``complete_path`` never transiently has ``st_nlink == 2`` or mutates ``st_ctime_ns`` after
    becoming visible to ``research.colab_cli.runtime._exports``.
    """
    sha_path = Path(f"{target}.sha256")
    complete_path = Path(f"{target}.complete")
    for existing in (sha_path, complete_path):
        if existing.exists():
            raise FileExistsError(f"Refusing to overwrite immutable sidecar path: {existing}")

    sha_tmp = Path(f"{sha_path}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with sha_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}  {target.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if sha_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable sidecar path: {sha_path}")
        os.replace(sha_tmp, sha_path)
    finally:
        if sha_tmp.exists():
            sha_tmp.unlink()

    comp_tmp = Path(f"{complete_path}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with comp_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if complete_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable sidecar path: {complete_path}")
        os.replace(comp_tmp, complete_path)
    finally:
        if comp_tmp.exists():
            comp_tmp.unlink()


def write_create_only_artifact(
    path: Path | str,
    writer: Callable[[Path], None] | bytes | bytearray | memoryview,
) -> str:
    """Write ``path`` once as a fresh single-inode (`st_nlink == 1`) artifact with sidecars."""
    target = Path(path)
    if artifact_or_sidecar_exists(target):
        raise FileExistsError(f"Refusing to overwrite immutable artifact path: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if isinstance(writer, (bytes, bytearray, memoryview)):
            with tmp_path.open("wb") as handle:
                handle.write(bytes(writer))
                handle.flush()
                os.fsync(handle.fileno())
        else:
            writer(tmp_path)
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())

        digest = sha256_file(tmp_path)
        if artifact_or_sidecar_exists(target):
            raise FileExistsError(f"Refusing to overwrite immutable artifact path: {target}")
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    _publish_sidecars_for_existing_payload(target, digest)
    _arena_rt._fsync_directory(target.parent)
    return digest


def write_create_only_json(path: Path | str, payload: dict[str, Any]) -> str:
    """Serialize a JSON mapping with create-only ``.sha256`` and ``.complete`` markers."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return write_create_only_artifact(path, encoded)


def publish_hardlinked_artifacts(
    source_path: Path | str,
    targets: Sequence[Path | str],
    *,
    staging_dir: Path | str | None = None,
) -> str:
    """Publish a persistent work-dir file into immutable create-only paths via hard links.

    Must ONLY be used when ``source_path`` lives in a persistent work directory that outlives
    export scanning (never a temporary scratch directory that is deleted before exit).
    """
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Missing source payload for hard-link publication: {src}")
    dst_list = [Path(t) for t in targets]
    if not dst_list:
        raise ValueError("targets must contain at least one destination path")
    if len({str(p.resolve()) if p.parent.exists() else str(p) for p in dst_list}) != len(dst_list):
        raise ValueError(f"Duplicate destination paths in targets: {dst_list}")

    for dst in dst_list:
        if artifact_or_sidecar_exists(dst):
            raise FileExistsError(f"Refusing to overwrite immutable artifact path: {dst}")
    for dst in dst_list:
        dst.parent.mkdir(parents=True, exist_ok=True)

    digest = sha256_file(src)
    src_stat = src.stat()
    linked_directly = False
    if src_stat.st_nlink == 1:
        created_direct: list[Path] = []
        try:
            for dst in dst_list:
                os.link(src, dst)
                created_direct.append(dst)
            linked_directly = True
        except OSError:
            for partial in created_direct:
                partial.unlink(missing_ok=True)
            linked_directly = False

    if not linked_directly:
        stage_parent = Path(staging_dir) if staging_dir is not None else dst_list[0].parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        fd, stage_name = tempfile.mkstemp(
            prefix=f".{dst_list[0].name}.stage.",
            dir=str(stage_parent),
        )
        os.close(fd)
        stage_path = Path(stage_name)
        created_staged: list[Path] = []
        try:
            shutil.copyfile(src, stage_path)
            with stage_path.open("rb") as handle:
                os.fsync(handle.fileno())
            for dst in dst_list:
                try:
                    os.link(stage_path, dst)
                except OSError:
                    fd_d, dst_tmp_name = tempfile.mkstemp(
                        prefix=f".{dst.name}.tmp.",
                        dir=str(dst.parent),
                    )
                    os.close(fd_d)
                    dst_tmp = Path(dst_tmp_name)
                    try:
                        shutil.copyfile(stage_path, dst_tmp)
                        with dst_tmp.open("rb") as h_d:
                            os.fsync(h_d.fileno())
                        os.replace(dst_tmp, dst)
                    finally:
                        dst_tmp.unlink(missing_ok=True)
                created_staged.append(dst)
        except BaseException:
            for partial in created_staged:
                partial.unlink(missing_ok=True)
            raise
        finally:
            stage_path.unlink(missing_ok=True)

    for dst in dst_list:
        _publish_sidecars_for_existing_payload(dst, digest)

    seen_parents: set[Path] = set()
    for dst in dst_list:
        parent = dst.parent.resolve()
        if parent not in seen_parents:
            seen_parents.add(parent)
            _arena_rt._fsync_directory(dst.parent)
    return digest


def exported_payload_bytes(out_dir: Path) -> int:
    """Sum byte sizes of all non-sidecar payload files recursively under ``out_dir``."""
    if not out_dir.is_dir():
        return 0
    total = 0
    for p in out_dir.rglob("*"):
        if p.is_file() and not p.name.endswith((".sha256", ".complete")):
            try:
                total += int(p.stat().st_size)
            except OSError:
                pass
    return total


_exported_payload_bytes = exported_payload_bytes


class RotatingCreateOnlyJSONLWriter:
    """Append-only rotating JSONL writer publishing single-inode ``.complete`` segments."""

    def __init__(
        self,
        log_dir: Path | str,
        *,
        prefix: str = "train_log",
        rotate_every_records: int = 64,
        rotate_every_seconds: float = 900.0,
    ) -> None:
        if int(rotate_every_records) < 1:
            raise ValueError(f"rotate_every_records must be >= 1, got {rotate_every_records}")
        if float(rotate_every_seconds) <= 0.0:
            raise ValueError(f"rotate_every_seconds must be > 0, got {rotate_every_seconds}")
        self.log_dir = Path(log_dir)
        self.prefix = str(prefix)
        self.rotate_every_records = int(rotate_every_records)
        self.rotate_every_seconds = float(rotate_every_seconds)

        self._next_segment_idx = 1
        self._active_path: Path | None = None
        self._active_handle: Any = None
        self._active_records = 0
        self._active_opened_mono = 0.0
        self.total_records = 0
        self.completed_segments: list[Path] = []

    def _open_next_segment(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        seg_path = self.log_dir / f"{self.prefix}_{self._next_segment_idx:05d}.jsonl"
        if artifact_or_sidecar_exists(seg_path):
            raise FileExistsError(f"Refusing to overwrite immutable log segment: {seg_path}")
        self._active_handle = seg_path.open("xb")
        self._active_path = seg_path
        self._active_records = 0
        self._active_opened_mono = time.monotonic()
        self._next_segment_idx += 1

    def append(self, record: dict[str, Any]) -> Path:
        """Write a single JSON line to the active segment, rotating when thresholds hit."""
        if self._active_handle is None:
            self._open_next_segment()
        assert self._active_path is not None
        current_path = self._active_path
        encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self._active_handle.write(encoded)
        self._active_handle.flush()
        self._active_records += 1
        self.total_records += 1

        if (
            self._active_records >= self.rotate_every_records
            or (time.monotonic() - self._active_opened_mono) >= self.rotate_every_seconds
        ):
            self.seal_current_segment()
        return current_path

    def seal_current_segment(self) -> Path | None:
        """Flush, fsync, close, and publish ``.sha256`` + ``.complete`` for the active segment."""
        if self._active_handle is None or self._active_path is None:
            return None
        seg_path = self._active_path
        handle = self._active_handle
        self._active_handle = None
        self._active_path = None
        self._active_records = 0
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

        digest = sha256_file(seg_path)
        _publish_sidecars_for_existing_payload(seg_path, digest)
        _arena_rt._fsync_directory(self.log_dir)
        self.completed_segments.append(seg_path)
        return seg_path

    def close(self) -> list[Path]:
        """Seal any open segment and return all completed segment paths."""
        self.seal_current_segment()
        return list(self.completed_segments)
