"""Bounded, two-stage full-state checkpoint persistence (opt-in T21).

The caller owns the update/actor barrier and passes an immutable CPU snapshot. No
filesystem or Drive operation runs on the learner thread. A manifest is committed
last; incomplete directories are never resume candidates. Only load trusted run
snapshots: torch's full-state pickle can contain Python objects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1


def capture_state(value: Any) -> Any:
    """Detach state at a caller-owned mutation boundary.

    CUDA tensors use pinned host storage and asynchronous D2H copies; a single
    event is awaited before returning so the caller can safely mutate its model.
    CPU replay arrays must be copied too: their cost scales with replay size and
    is not a universal sub-50-ms guarantee. Serialization is never done here.
    """
    import torch

    events = {}

    def clone(item):
        if isinstance(item, torch.Tensor):
            item = item.detach()
            if item.device.type == "cuda":
                result = torch.empty_like(item, device="cpu", pin_memory=True)
                with torch.cuda.device(item.device):
                    result.copy_(item, non_blocking=True)
                    events[item.device] = torch.cuda.current_stream(item.device)
                return result
            return item.cpu().clone()
        if isinstance(item, np.ndarray):
            return item.copy()
        if isinstance(item, dict):
            return type(item)((key, clone(val)) for key, val in item.items())
        if isinstance(item, list):
            return [clone(val) for val in item]
        if isinstance(item, tuple):
            return tuple(clone(val) for val in item)
        return copy.deepcopy(item)

    result = clone(value)
    for device, stream in events.items():
        with torch.cuda.device(device):
            event = torch.cuda.Event()
            event.record(stream)
            event.synchronize()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _read_manifest(path: Path, verify: bool = True) -> dict:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("invalid manifest object")
    if manifest.get("schema") != SCHEMA or manifest.get("snapshot") != path.parent.name:
        raise ValueError("invalid full-state manifest identity/schema")
    if not isinstance(manifest.get("transitions"), int) or manifest["transitions"] < 0:
        raise ValueError("invalid transition counter")
    if not isinstance(manifest.get("created_ns"), int):
        raise ValueError("invalid creation timestamp")
    files = manifest.get("files")
    if (
        not isinstance(files, list)
        or len(files) != 1
        or not isinstance(files[0], dict)
        or files[0].get("name") != "state.pt"
    ):
        raise ValueError("invalid checkpoint file inventory")
    for entry in files:
        target = path.parent / entry["name"]
        if target.is_symlink() or target.stat().st_size != entry["size_bytes"]:
            raise ValueError("checkpoint file size mismatch")
        if verify and _sha256(target) != entry["sha256"]:
            raise ValueError("checkpoint sha256 mismatch")
    return manifest


def discover_snapshots(directory: str | Path, verify: bool = True) -> list[dict]:
    """Return complete, valid manifests newest first; ignore interrupted/corrupt saves."""
    root = Path(directory)
    result = []
    paths = (
        [root / "manifest.json"]
        if (root / "manifest.json").is_file()
        else root.glob("snapshot-*/manifest.json")
    )
    for path in paths:
        try:
            if not path.parent.is_symlink():
                result.append(_read_manifest(path, verify=verify))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(result, key=lambda item: (item["transitions"], item["created_ns"]), reverse=True)


def load_latest_snapshot(directory: str | Path) -> tuple[dict, dict]:
    """Verify checksums before unpickling the newest usable trusted snapshot."""
    import torch

    root = Path(directory)
    for manifest in discover_snapshots(root):
        folder = root if root.name == manifest["snapshot"] else root / manifest["snapshot"]
        try:
            state = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
            if not isinstance(state, dict):
                continue
            return state, manifest
        except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
            continue
    raise ValueError(f"no complete verified full-state snapshot in {root}")


class AsyncCheckpointer:
    """One bounded local writer plus one independent durable-copy worker.

    At most one CPU snapshot is pending serialization; callers skip busy intervals
    rather than accumulating RAM. Durable copies are queued by local path. Local
    retention protects the one active copy; queued copies coalesce to the newest
    snapshot. A slow Drive can temporarily retain one extra file beyond ``keep``.
    Failed copies remain subject to normal retention, bounding disk use.
    """

    def __init__(
        self,
        local_dir: str | Path,
        durable_dir: str | Path | None = None,
        keep: int = 2,
        retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.durable_dir = Path(durable_dir) if durable_dir is not None else None
        if keep < 1 or retries < 1 or retry_delay < 0:
            raise ValueError("keep/retries must be positive and retry_delay nonnegative")
        if self.durable_dir is not None and self.local_dir.resolve() == self.durable_dir.resolve():
            raise ValueError("local and durable checkpoint directories must differ")
        self.keep, self.retries, self.retry_delay = keep, retries, retry_delay
        self._local = queue.Queue(maxsize=1)
        self._durable = queue.Queue(maxsize=1)
        self._events = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._prune_lock = threading.Lock()
        self._pending: set[str] = set()
        self._latest_name = None
        self._local_complete_name = None
        self._durable_complete_name = None
        self._last_local_failed_transition = -1
        self._last_local_transition = -1
        self._last_failed_transition = -1
        self._last_durable_transition = -1
        self._active = False
        self._closed = False
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True, name="ckpt-local")
        self._copier = threading.Thread(target=self._copy_loop, daemon=True, name="ckpt-durable")
        self._writer.start()
        self._copier.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active

    def submit(self, state: dict, transitions: int, metadata: dict | None = None) -> bool:
        """Transfer ownership of already-detached state; return false instead of waiting."""
        with self._lock:
            if self._closed or self._active:
                return False
            self._active = True
        self._local.put_nowait((state, int(transitions), dict(metadata or {})))
        return True

    def poll_events(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def _prune(self, directory: Path, protect: set[str] | None = None) -> None:
        # Hashes were checked on commit; retention only needs valid manifest inventory.
        for manifest in discover_snapshots(directory, verify=False)[self.keep :]:
            if manifest["snapshot"] not in (protect or set()):
                shutil.rmtree(directory / manifest["snapshot"])

    def _prune_local(self, protected: set[str]) -> None:
        with self._prune_lock:
            with self._lock:
                protected = set(self._pending)
            self._prune(self.local_dir, protected)

    def _write_loop(self) -> None:
        import torch

        while not self._stop.is_set() or not self._local.empty():
            try:
                state, transitions, metadata = self._local.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.perf_counter()
            name = f"snapshot-{transitions:020d}-{uuid.uuid4().hex[:8]}"
            with self._lock:
                self._latest_name = name
            temporary = self.local_dir / f".tmp-{name}"
            try:
                temporary.mkdir(parents=True)
                data = temporary / "state.pt"
                with data.open("wb") as stream:
                    torch.save(state, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                manifest = {
                    "schema": SCHEMA,
                    "snapshot": name,
                    "transitions": transitions,
                    "created_ns": time.time_ns(),
                    "files": [
                        {
                            "name": "state.pt",
                            "sha256": _sha256(data),
                            "size_bytes": data.stat().st_size,
                        }
                    ],
                    "metadata": metadata,
                }
                _atomic_json(temporary / "manifest.json", manifest)
                os.replace(temporary, self.local_dir / name)
                _fsync_dir(self.local_dir)
                with self._lock:
                    self._local_complete_name = name
                    self._last_local_transition = max(self._last_local_transition, transitions)
                    if self.durable_dir is not None:
                        self._pending.add(name)
                        # Coalesce an unstarted upload to the newest snapshot. Only
                        # one active and one waiting copy can protect local files.
                        try:
                            older = self._durable.get_nowait()
                        except queue.Empty:
                            pass
                        else:
                            self._pending.discard(older)
                            self._durable.task_done()
                            self._events.put(
                                {
                                    "event": "checkpoint_copy_superseded",
                                    "snapshot": older,
                                    "replacement": name,
                                }
                            )
                        self._durable.put_nowait(name)
                    protected = set(self._pending)
                self._prune_local(protected)
                self._events.put(
                    {
                        "event": "checkpoint_local_complete",
                        "snapshot": name,
                        "transitions": transitions,
                        "local_seconds": time.perf_counter() - started,
                        "size_bytes": (
                            data.stat().st_size
                            if data.exists()
                            else manifest["files"][0]["size_bytes"]
                        ),
                    }
                )
            except Exception as exc:
                with self._lock:
                    self._last_local_failed_transition = max(
                        self._last_local_failed_transition, transitions
                    )
                self._events.put(
                    {
                        "event": "checkpoint_local_failed",
                        "snapshot": name,
                        "error": repr(exc),
                        "local_seconds": time.perf_counter() - started,
                    }
                )
            finally:
                # A failed partial is not resumable, and must not grow disk forever.
                shutil.rmtree(temporary, ignore_errors=True)
                del state
                with self._lock:
                    self._active = False
                self._local.task_done()

    def _copy_snapshot(self, name: str) -> None:
        assert self.durable_dir is not None
        source = self.local_dir / name
        temporary = self.durable_dir / f".tmp-{name}-{uuid.uuid4().hex[:8]}"
        try:
            temporary.mkdir(parents=True)
            manifest = _read_manifest(source / "manifest.json")
            for entry in manifest["files"]:
                dest = temporary / entry["name"]
                with (source / entry["name"]).open("rb") as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
                if _sha256(dest) != entry["sha256"]:
                    raise ValueError("durable copy sha256 mismatch")
            # Publish files first and the manifest LAST, even on Drive FUSE where a
            # directory rename may not have local-filesystem atomic semantics.
            final = self.durable_dir / name
            final.mkdir(exist_ok=True)
            for entry in manifest["files"]:
                os.replace(temporary / entry["name"], final / entry["name"])
            _atomic_json(final / "manifest.json", manifest)
            _fsync_dir(self.durable_dir)
            self._prune(self.durable_dir)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _copy_loop(self) -> None:
        while not self._stop.is_set() or self._writer.is_alive() or not self._durable.empty():
            try:
                name = self._durable.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.perf_counter()
            for attempt in range(1, self.retries + 1):
                try:
                    self._copy_snapshot(name)
                    self._events.put(
                        {
                            "event": "checkpoint_durable_complete",
                            "snapshot": name,
                            "attempts": attempt,
                            "durable_seconds": time.perf_counter() - started,
                        }
                    )
                    with self._lock:
                        self._durable_complete_name = name
                        self._last_durable_transition = max(
                            self._last_durable_transition, int(name.split("-")[1])
                        )
                        self._pending.discard(name)
                        protected = set(self._pending)
                    self._prune_local(protected)
                    break
                except Exception as exc:
                    self._events.put(
                        {
                            "event": "checkpoint_durable_failed",
                            "snapshot": name,
                            "attempt": attempt,
                            "error": repr(exc),
                            "durable_seconds": time.perf_counter() - started,
                        }
                    )
                    if attempt < self.retries:
                        time.sleep(self.retry_delay * 2 ** (attempt - 1))
            else:
                with self._lock:
                    self._last_failed_transition = max(
                        self._last_failed_transition, int(name.split("-")[1])
                    )
                    self._pending.discard(name)
                    protected = set(self._pending)
                try:
                    self._prune_local(protected)
                except OSError as exc:
                    self._events.put({"event": "checkpoint_retention_failed", "error": repr(exc)})
            # Failed snapshots remain among the newest K local recovery points;
            # later successful copies supersede earlier durability failures.
            self._durable.task_done()

    def wait_local(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.005)
        return not self.busy

    def close(self, timeout: float = 30.0) -> bool:
        """Bounded shutdown; false means pending work remains, never release the runtime."""
        with self._lock:
            self._closed = True
        self._stop.set()
        deadline = time.monotonic() + timeout
        for worker in (self._writer, self._copier):
            worker.join(max(0.0, deadline - time.monotonic()))
        with self._lock:
            pending = (
                bool(self._pending)
                or self._latest_name != self._local_complete_name
                or (
                    self.durable_dir is not None
                    and self._latest_name != self._durable_complete_name
                )
            )
        return not (self._writer.is_alive() or self._copier.is_alive() or pending)
