"""Versioned, verified Watch frame recordings; no model execution at playback."""

from __future__ import annotations

import gc
import gzip
import hashlib
import io
import json
import logging
import math
import mmap
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "snake-watch-replay/v1"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_SHA = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
ASSISTED_CONTROLLERS = frozenset({"exact-stop-l2-prime", "exact-stop-l1", "exact-stop-l1-os"})
CONTROLLERS = ASSISTED_CONTROLLERS | {"bare-b8"}
ENV_REPLAY_CONTROLLER = "SNAKE_WATCH_REPLAY_CONTROLLER"
REPLAY_CONTROLLER_ALL = "all"
VALID_REPLAY_CONTROLLERS = frozenset({REPLAY_CONTROLLER_ALL, *ASSISTED_CONTROLLERS})
MAX_COMPRESSED = 128 * 1024**2
MAX_DECODED = 512 * 1024**2
MAX_LINE = 2 * 1024**2
COOPERATIVE_YIELD_FRAMES = 64
COOPERATIVE_YIELD_SLICE_SECONDS = 0.00025
COOPERATIVE_YIELD_MIN_WAIT_SECONDS = 0.001
COOPERATIVE_YIELD_DUTY_FACTOR = 4.0
GC_GEN0_THRESHOLD = 50_000
GC_GEN1_THRESHOLD = 20
GC_GEN2_THRESHOLD = 20

_GC_SUPPRESS_LOCK = threading.Lock()
_gc_suppress_depth = 0
_gc_restore_enabled = False


def _ensure_frame_gc_thresholds() -> None:
    """Keep Gen-0 GC threshold above a single Watch frame's container working set."""
    t0, t1, t2 = gc.get_threshold()
    if t0 == 0:
        return
    if t0 < GC_GEN0_THRESHOLD or t1 < GC_GEN1_THRESHOLD or t2 < GC_GEN2_THRESHOLD:
        gc.set_threshold(
            max(t0, GC_GEN0_THRESHOLD),
            max(t1, GC_GEN1_THRESHOLD),
            max(t2, GC_GEN2_THRESHOLD),
        )


def _enter_gc_suppress() -> None:
    global _gc_suppress_depth, _gc_restore_enabled
    with _GC_SUPPRESS_LOCK:
        if _gc_suppress_depth == 0:
            _gc_restore_enabled = gc.isenabled()
            if _gc_restore_enabled:
                gc.disable()
        _gc_suppress_depth += 1


def _exit_gc_suppress() -> None:
    global _gc_suppress_depth
    with _GC_SUPPRESS_LOCK:
        _gc_suppress_depth -= 1
        if _gc_suppress_depth == 0 and _gc_restore_enabled:
            gc.enable()


_REQUIRED_FRAME_KEYS = frozenset(
    {
        "arena",
        "food",
        "stats",
        "session",
        "inspector",
        "netviz",
        "obs_spec",
        "architecture",
        "checkpoint_name",
        "serving_contract",
        "play",
        "hero_raster",
    }
)
_SNAKE_IDS = frozenset(range(6))
_SNAKE_KEYS = ("segments", "head", "length", "alive", "boosting", "color", "is_hero", "name")
_OPTIONAL_DICT_KEYS = ("inspector", "netviz", "hero_raster", "serving_contract")
_ARENA_NUMERIC_KEYS = ("width", "height", "segment", "wall")
_INSPECTOR_NUMERIC_LIST_KEYS = ("q_values", "state", "free_space")
_INSPECTOR_ACTION_KEYS = ("chosen", "executed_action", "replay_bare_action")
_NUM_TYPES = (int, float)
_ISFINITE = math.isfinite


def _reject_nonfinite(_token: str) -> Any:
    raise ValueError("non-finite JSON constant")


_FINITE_JSON_DECODER = json.JSONDecoder(parse_constant=_reject_nonfinite)


def canonical(value: Any) -> bytes:
    """Encode portable finite JSON, including a record delimiter."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _decode_json_object(line: bytes) -> dict:
    value = _FINITE_JSON_DECODER.decode(line.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("replay record must be an object")
    return value


class _BufferReader(io.RawIOBase):
    """Zero-copy readable stream over bytes or an anonymous mmap buffer."""

    def __init__(self, buf: bytes | bytearray | memoryview | mmap.mmap) -> None:
        self._buf = buf
        self._size = len(buf)
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = pos
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + pos
        elif whence == io.SEEK_END:
            new_pos = self._size + pos
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self._pos = min(new_pos, self._size)
        return self._pos

    def readinto(self, b: Any) -> int:
        if getattr(self._buf, "closed", False):
            return 0
        avail = self._size - self._pos
        if avail <= 0:
            return 0
        with memoryview(b) as dest, memoryview(self._buf) as src:
            dest_b = dest.cast("B")
            count = min(len(dest_b), avail)
            dest_b[:count] = src[self._pos : self._pos + count]
            self._pos += count
            return count


def _records(payload: bytes | bytearray | memoryview | mmap.mmap) -> Iterator[dict]:
    total = 0
    with (
        _BufferReader(payload) as raw,
        io.BufferedReader(raw) as buf,
        gzip.GzipFile(fileobj=buf, mode="rb") as stream,
    ):
        while line := stream.readline(MAX_LINE + 1):
            total += len(line)
            if len(line) > MAX_LINE or total > MAX_DECODED:
                raise ValueError("replay decompression limit")
            yield _decode_json_object(line)


def _read_pinned_payload(path: Path) -> tuple[bytes | mmap.mmap, str]:
    """Read bounded compressed replay bytes into an anonymous mmap so close() unmaps pages."""
    size = path.stat().st_size
    if size > MAX_COMPRESSED:
        raise ValueError("oversized replay")
    if size == 0:
        with path.open("rb") as stream:
            extra = stream.read(1)
        if extra:
            raise ValueError("incomplete replay")
        return b"", hashlib.sha256(b"").hexdigest()
    mm = mmap.mmap(-1, size)
    try:
        pos = 0
        with path.open("rb") as stream, memoryview(mm) as mv:
            while pos < size:
                n = stream.readinto(mv[pos:])
                if not n:
                    break
                pos += n
            extra = stream.read(1)
            if pos != size or extra:
                raise ValueError("oversized replay" if extra else "incomplete replay")
            digest = hashlib.sha256(mv).hexdigest()
        return mm, digest
    except BaseException:
        mm.close()
        raise


def validate_header(header: dict, *, product: bool = True) -> None:
    """Reject wrong controllers, checkpoints, horizon and world contracts."""
    if (
        header.get("schema") != SCHEMA
        or header.get("controller") not in CONTROLLERS
        or header.get("checkpoint_shas") != {"hero": B8_SHA, "opponents": FRP3_SHA}
        or header.get("world") != "watch6-mechanics-v2-continuous-respawn"
        or header.get("protocol_version") != 2
        or type(header.get("horizon")) is not int
        or not 1 <= header["horizon"] <= 6000
        or (product and (header["horizon"] != 6000 or header.get("fixture") is not False))
        or type(header.get("seed")) is not int
        or not isinstance(header.get("unit_id"), str)
        or not header["unit_id"]
        or not isinstance(header.get("stats"), dict)
    ):
        raise ValueError("foreign replay metadata")


def resolve_replay_controller(controller: str | None = None) -> tuple[str | None, str | None]:
    """Resolve SNAKE_WATCH_REPLAY_CONTROLLER, failing closed on unknown values."""
    raw = os.environ.get(ENV_REPLAY_CONTROLLER) if controller is None else controller
    if raw is None or not str(raw).strip():
        return REPLAY_CONTROLLER_ALL, None
    normalized = str(raw).strip().lower()
    if normalized not in VALID_REPLAY_CONTROLLERS:
        return None, (
            f"Invalid {ENV_REPLAY_CONTROLLER}={raw!r}; expected all, "
            "exact-stop-l2-prime, exact-stop-l1, or exact-stop-l1-os. Using live B8."
        )
    return normalized, None


def _check_replay_files(path: Path) -> tuple[Path, Path]:
    sha_path = Path(str(path) + ".sha256")
    complete_path = Path(str(path) + ".complete")
    paths = (path, sha_path, complete_path)
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("incomplete replay")
    if path.stat().st_size > MAX_COMPRESSED or sha_path.stat().st_size > 1024:
        raise ValueError("oversized replay")
    if complete_path.stat().st_size > 4096 or complete_path.read_bytes() != b"complete\n":
        raise ValueError("replay completion marker")
    return sha_path, complete_path


def peek_header(path: Path, *, product: bool = True) -> dict:
    """Read and validate only line 1 of a completed .watch.gz recording."""
    _check_replay_files(path)
    with path.open("rb") as raw_stream, gzip.GzipFile(fileobj=raw_stream) as stream:
        line = stream.readline(MAX_LINE + 1)
    if not line:
        raise ValueError("truncated replay")
    if len(line) > MAX_LINE:
        raise ValueError("replay decompression limit")
    header = _decode_json_object(line)
    validate_header(header, product=product)
    return header


def _number(value: Any) -> bool:
    return type(value) in _NUM_TYPES and _ISFINITE(value)


def _point(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    x, y = value[0], value[1]
    return type(x) in _NUM_TYPES and _ISFINITE(x) and type(y) in _NUM_TYPES and _ISFINITE(y)


def validate_frame(frame: dict, index: int) -> None:
    """Require the existing protocol plus the six recorded roster members."""
    snakes = frame.get("snakes")
    if (
        frame.get("type") != "frame"
        or frame.get("protocol_version") != 2
        or type(frame.get("frame")) is not int
        or frame["frame"] != index
        or frame.get("mechanics_version") != 2
        or not _REQUIRED_FRAME_KEYS <= frame.keys()
        or not isinstance(snakes, list)
        or len(snakes) != 6
        or any(not isinstance(s, dict) or type(s.get("id")) is not int for s in snakes)
        or {s["id"] for s in snakes} != _SNAKE_IDS
        or not isinstance(frame["session"], dict)
        or not isinstance(frame["arena"], dict)
        or not isinstance(frame["stats"], dict)
        or not isinstance(frame["food"], list)
        or not all(_point(p) for p in frame["food"])
        or any(frame[k] is not None and not isinstance(frame[k], dict) for k in _OPTIONAL_DICT_KEYS)
        or not isinstance(frame["obs_spec"], str)
        or not isinstance(frame["architecture"], str)
        or not isinstance(frame["checkpoint_name"], (str, type(None)))
        or frame["session"].get("mode") != "watch"
        or frame["session"].get("watch_roster") != "mixed"
        or frame.get("play") is not None
    ):
        raise ValueError(f"invalid Watch replay frame {index}")
    for snake in snakes:
        for key in _SNAKE_KEYS:
            if key not in snake:
                raise ValueError("incomplete recorded snake")
        segments = snake["segments"]
        head = snake["head"]
        length = snake["length"]
        color = snake["color"]
        if (
            not isinstance(segments, list)
            or not all(_point(p) for p in segments)
            or (head is not None and not _point(head))
            or type(length) is not int
            or length < 0
            or type(snake["alive"]) is not bool
            or type(snake["boosting"]) is not bool
            or type(snake["is_hero"]) is not bool
            or not isinstance(snake["name"], str)
            or not isinstance(color, list)
            or len(color) != 3
            or any(type(c) is not int or not 0 <= c <= 255 for c in color)
        ):
            raise ValueError("invalid recorded snake field types")
    arena = frame["arena"]
    if any(not _number(arena.get(k)) for k in _ARENA_NUMERIC_KEYS):
        raise ValueError("invalid recorded arena")

    if any(v is not None and not _number(v) for v in frame["stats"].values()):
        raise ValueError("invalid recorded stats")
    inspector = frame["inspector"]
    if inspector is not None:
        for key in _INSPECTOR_NUMERIC_LIST_KEYS:
            value = inspector.get(key)
            if value is not None and (
                not isinstance(value, list) or not all(_number(v) for v in value)
            ):
                raise ValueError("invalid inspector numeric fields")
        for key in _INSPECTOR_ACTION_KEYS:
            value = inspector.get(key)
            if value is not None and (type(value) is not int or not -1 <= value < 6):
                raise ValueError("invalid inspector action")
        if (
            "replay_search_override" in inspector
            and type(inspector["replay_search_override"]) is not bool
        ):
            raise ValueError("invalid inspector search flag")
        labels = inspector.get("action_labels")
        if labels is not None and (
            not isinstance(labels, list) or not all(isinstance(v, str) for v in labels)
        ):
            raise ValueError("invalid inspector labels")


def verify_payload(
    payload: bytes | bytearray | memoryview | mmap.mmap,
    *,
    product: bool = True,
    stop: threading.Event | None = None,
) -> dict:
    """Validate bounded compressed bytes before a producer marks them complete."""
    if stop is not None:
        _ensure_frame_gc_thresholds()
    if len(payload) > MAX_COMPRESSED:
        raise ValueError("oversized replay")
    records = _records(payload)
    try:
        _enter_gc_suppress()
        try:
            header = next(records)
            validate_header(header, product=product)
        finally:
            _exit_gc_suppress()
        count = 0
        slice_start = time.perf_counter() if stop is not None else 0.0
        while True:
            frame = None
            _enter_gc_suppress()
            try:
                frame = next(records, None)
                if frame is None:
                    break
                count += 1
                validate_frame(frame, count)
            finally:
                frame = None
                _exit_gc_suppress()
            # Serving verification cooperatively releases the GIL at least every
            # 64 records and whenever a single verification slice reaches 0.25 ms
            # so heavy late-game frames cannot starve live Watch fallback ticks.
            # Producer verification keeps its original synchronous path.
            if stop is not None:
                if stop.is_set():
                    raise InterruptedError("replay bank closed")
                elapsed = time.perf_counter() - slice_start
                if (
                    count % COOPERATIVE_YIELD_FRAMES == 0
                    or elapsed >= COOPERATIVE_YIELD_SLICE_SECONDS
                ):
                    wait_s = max(
                        COOPERATIVE_YIELD_MIN_WAIT_SECONDS,
                        elapsed * COOPERATIVE_YIELD_DUTY_FACTOR,
                    )
                    if stop.wait(wait_s):
                        raise InterruptedError("replay bank closed")
                    slice_start = time.perf_counter()
        if count != header["horizon"]:
            raise ValueError("truncated replay")
        return header
    finally:
        records.close()


class Recording:
    """Pin compressed bytes once; verify every record before exposing frame one."""

    def __init__(
        self,
        path: Path,
        *,
        product: bool = True,
        cache: dict | None = None,
        stop: threading.Event | None = None,
    ) -> None:
        self._records: Iterator[dict] | None = None
        self._mmap: mmap.mmap | None = None
        if stop is not None:
            _ensure_frame_gc_thresholds()
        sha_path, _ = _check_replay_files(path)
        payload, self.sha256 = _read_pinned_payload(path)
        if isinstance(payload, mmap.mmap):
            self._mmap = payload
        try:
            if sha_path.read_text().split() != [self.sha256, path.name]:
                raise ValueError("replay SHA mismatch")
            key = (self.sha256, product)
            if cache is not None and key in cache:
                self.header = cache[key]
            else:
                self.header = verify_payload(payload, product=product, stop=stop)
                if cache is not None:
                    cache[key] = self.header
            self._records = _records(payload)
            _enter_gc_suppress()
            try:
                next(self._records)
                self.frame = next(self._records)
            finally:
                _exit_gc_suppress()
            self.path = path
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the lazy decompression cursor and unmap the pinned compressed payload."""
        records, self._records = getattr(self, "_records", None), None
        if records is not None:
            close_records = getattr(records, "close", None)
            if callable(close_records):
                close_records()
        mm, self._mmap = getattr(self, "_mmap", None), None
        if mm is not None and not mm.closed:
            mm.close()

    def __del__(self) -> None:
        self.close()

    def step(self) -> bool:
        """Advance one frame; false asks the caller to rotate or use live Watch."""
        if self._records is None:
            return False
        if self.frame["frame"] >= self.header["horizon"]:
            has_more = next(self._records, None) is not None
            if not has_more:
                self.close()
            return has_more
        _enter_gc_suppress()
        prev_frame = self.frame
        try:
            self.frame = None  # type: ignore[assignment]
            frame = next(self._records, None)
            if frame is None:
                self.frame = prev_frame
                self.close()
                return False
            self.frame = frame
            return True
        except BaseException:
            self.frame = prev_frame
            raise
        finally:
            prev_frame = None
            _exit_gc_suppress()

    def snapshot(self, session: Any) -> dict:
        """Keep recorded inspector/roster, overlay only present playback controls."""
        return {
            **self.frame,
            "paused": not session.playing,
            "viewer_count": session.viewer_count,
            "session": {
                **self.frame["session"],
                "playing": session.playing,
                "speed": session.speed,
                "speed_ceiling": session.speed_ceiling,
                "error": session.last_error,
                "watch_source": "replay",
                "replay": self.info(),
            },
        }

    def info(self) -> dict:
        return {
            "label": (
                "Recorded assisted hero"
                if self.header["controller"] in ASSISTED_CONTROLLERS
                else "Recorded bare B8"
            ),
            "controller": self.header["controller"],
            "unit_id": self.header["unit_id"],
            "seed": self.header["seed"],
            "sha256": self.sha256,
            "frame": self.frame["frame"],
            "horizon": self.header["horizon"],
        }


class ReplayBank:
    """Verify once per SHA and prefetch rotations off the session/tick thread.

    The bounded queue holds compressed recordings, never decoded whole sessions.
    Every rotation rereads and hashes the file in the worker; the cursor then
    lazily decodes frames from its pinned in-memory bytes without disk I/O.
    Verified recordings become available during discovery. A bank containing
    only one distinct unit plays once, then returns to live B8 rather than
    immediately repeating that session. Two distinct units permit rotation.
    """

    def __init__(
        self,
        root: Path,
        *,
        controller: str | None = None,
        shuffle_discovery: bool = True,
    ) -> None:
        _ensure_frame_gc_thresholds()
        self.root = root
        self.controller, self.error = resolve_replay_controller(controller)
        self._shuffle_discovery = bool(shuffle_discovery)
        self._ready: queue.Queue = queue.Queue(maxsize=2)
        self._closed = threading.Event()
        self._done = threading.Event()
        self._worker = threading.Thread(
            target=self._prefetch, name="watch-replay-prefetch", daemon=True
        )
        self._worker.start()

    @property
    def pending(self) -> bool:
        return not self._done.is_set() or not self._ready.empty()

    def close(self) -> None:
        """Cancel verification/queue waits and reap the worker within two seconds."""
        self._closed.set()
        self._worker.join(timeout=2)
        if self._worker.is_alive():
            logging.getLogger(__name__).warning("Replay worker close timed out")

    def pick(self) -> Recording | None:
        """Nonblocking rotation: no glob, stat, read, hash, or full validation."""
        try:
            record = self._ready.get_nowait()
        except queue.Empty:
            if self._done.is_set() and self.error is None:
                self.error = "No verified replay without an immediate repeat; using live B8."
            return None
        self.error = None
        return record

    def _enqueue(self, record: Recording) -> bool:
        """Use the queue's condition wakeup; timed waits also observe close."""
        while not self._closed.is_set():
            try:
                self._ready.put(record, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _prefetch(self) -> None:
        cache: dict = {}
        logger = logging.getLogger(__name__)
        try:
            if self.controller is None:
                return
            paths = sorted(self.root.glob("*.watch.gz"))[:1024]
            if self._shuffle_discovery:
                random.SystemRandom().shuffle(paths)
            verified = []
            last = None
            seen: set[str] = set()
            for path in paths:
                if self._closed.is_set():
                    return
                try:
                    if self.controller != REPLAY_CONTROLLER_ALL:
                        header = peek_header(path)
                        if header["controller"] != self.controller:
                            continue
                    record = Recording(path, cache=cache, stop=self._closed)
                    if (
                        self.controller != REPLAY_CONTROLLER_ALL
                        and record.header["controller"] != self.controller
                    ):
                        close_rec = getattr(record, "close", None)
                        if callable(close_rec):
                            close_rec()
                        raise ValueError("replaced replay controller")
                    uid = record.header["unit_id"]
                    verified.append((path, uid))
                    if uid != last and uid not in seen:
                        if not self._enqueue(record):
                            close_rec = getattr(record, "close", None)
                            if callable(close_rec):
                                close_rec()
                            record = None
                            return
                        last = uid
                        seen.add(uid)
                    else:
                        close_rec = getattr(record, "close", None)
                        if callable(close_rec):
                            close_rec()
                    record = None
                except InterruptedError:
                    record = None
                    return
                except Exception as exc:
                    record = None
                    logger.warning("Skipping replay %s: %s", path, exc)
            while not self._closed.is_set():
                candidates = [(p, uid) for p, uid in verified if uid != last and uid not in seen]
                if not candidates:
                    seen.clear()
                    candidates = [(p, uid) for p, uid in verified if uid != last]
                if not candidates:
                    self.error = "No verified replay without an immediate repeat; using live B8."
                    return
                random.SystemRandom().shuffle(candidates)
                chosen = None
                for path, uid in candidates:
                    try:
                        chosen = Recording(path, cache=cache, stop=self._closed)
                        # A replaced file must retain its bank identity and controller filter.
                        if chosen.header["unit_id"] != uid:
                            close_rec = getattr(chosen, "close", None)
                            if callable(close_rec):
                                close_rec()
                            raise ValueError("replaced replay identity")
                        if (
                            self.controller != REPLAY_CONTROLLER_ALL
                            and chosen.header["controller"] != self.controller
                        ):
                            close_rec = getattr(chosen, "close", None)
                            if callable(close_rec):
                                close_rec()
                            raise ValueError("replaced replay controller")
                        break
                    except InterruptedError:
                        chosen = None
                        return
                    except Exception as exc:
                        logger.warning("Skipping replay %s: %s", path, exc)
                        verified.remove((path, uid))
                        chosen = None
                if chosen is None:
                    continue
                chosen_uid = chosen.header["unit_id"]
                if not self._enqueue(chosen):
                    close_rec = getattr(chosen, "close", None)
                    if callable(close_rec):
                        close_rec()
                    chosen = None
                    return
                chosen = None
                last = chosen_uid
                seen.add(last)
        except Exception as exc:
            self.error = f"Replay bank unavailable; using live B8: {exc}"
            logger.exception("Replay prefetch failed")
        finally:
            self._done.set()
