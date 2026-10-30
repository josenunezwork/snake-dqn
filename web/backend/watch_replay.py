"""Versioned, verified Watch frame recordings; no model execution at playback."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import logging
import math
import queue
import random
import threading
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "snake-watch-replay/v1"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_SHA = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
MAX_COMPRESSED = 128 * 1024**2
MAX_DECODED = 512 * 1024**2
MAX_LINE = 2 * 1024**2


def canonical(value: Any) -> bytes:
    """Encode portable finite JSON, including a record delimiter."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _records(payload: bytes) -> Iterator[dict]:
    total = 0
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        while line := stream.readline(MAX_LINE + 1):
            total += len(line)
            if len(line) > MAX_LINE or total > MAX_DECODED:
                raise ValueError("replay decompression limit")
            value = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(value, dict):
                raise ValueError("replay record must be an object")
            yield value


def validate_header(header: dict, *, product: bool = True) -> None:
    """Reject wrong controllers, checkpoints, horizon and world contracts."""
    if (
        header.get("schema") != SCHEMA
        or header.get("controller") not in {"exact-stop-l2-prime", "bare-b8"}
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


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _point(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_number(v) for v in value)


def validate_frame(frame: dict, index: int) -> None:
    """Require the existing protocol plus the six recorded roster members."""
    required = {
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
    if (
        frame.get("type") != "frame"
        or frame.get("protocol_version") != 2
        or type(frame.get("frame")) is not int
        or frame.get("frame") != index
        or frame.get("mechanics_version") != 2
        or not required <= frame.keys()
        or not isinstance(frame.get("snakes"), list)
        or len(frame["snakes"]) != 6
        or any(not isinstance(s, dict) or type(s.get("id")) is not int for s in frame["snakes"])
        or {s.get("id") for s in frame["snakes"]} != set(range(6))
        or any(not isinstance(frame[k], dict) for k in ("session", "arena", "stats"))
        or not isinstance(frame["food"], list)
        or not all(_point(p) for p in frame["food"])
        or any(
            frame[k] is not None and not isinstance(frame[k], dict)
            for k in ("inspector", "netviz", "hero_raster", "serving_contract")
        )
        or any(not isinstance(frame[k], str) for k in ("obs_spec", "architecture"))
        or not isinstance(frame["checkpoint_name"], (str, type(None)))
        or frame["session"].get("mode") != "watch"
        or frame["session"].get("watch_roster") != "mixed"
        or frame.get("play") is not None
    ):
        raise ValueError(f"invalid Watch replay frame {index}")
    for snake in frame["snakes"]:
        for key in ("segments", "head", "length", "alive", "boosting", "color", "is_hero", "name"):
            if key not in snake:
                raise ValueError("incomplete recorded snake")
        if (
            not isinstance(snake["segments"], list)
            or not all(_point(p) for p in snake["segments"])
            or (snake["head"] is not None and not _point(snake["head"]))
            or type(snake["length"]) is not int
            or snake["length"] < 0
            or any(type(snake[k]) is not bool for k in ("alive", "boosting", "is_hero"))
            or not isinstance(snake["name"], str)
            or not isinstance(snake["color"], list)
            or len(snake["color"]) != 3
            or any(type(c) is not int or not 0 <= c <= 255 for c in snake["color"])
        ):
            raise ValueError("invalid recorded snake field types")
    if any(not _number(frame["arena"].get(k)) for k in ("width", "height", "segment", "wall")):
        raise ValueError("invalid recorded arena")

    if any(v is not None and not _number(v) for v in frame["stats"].values()):
        raise ValueError("invalid recorded stats")
    inspector = frame["inspector"]
    if inspector is not None:
        for key in ("q_values", "state", "free_space"):
            value = inspector.get(key)
            if value is not None and (
                not isinstance(value, list) or not all(_number(v) for v in value)
            ):
                raise ValueError("invalid inspector numeric fields")
        for key in ("chosen", "executed_action", "replay_bare_action"):
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
    payload: bytes, *, product: bool = True, stop: threading.Event | None = None
) -> dict:
    """Validate bounded compressed bytes before a producer marks them complete."""
    if len(payload) > MAX_COMPRESSED:
        raise ValueError("oversized replay")
    records = _records(payload)
    header = next(records)
    validate_header(header, product=product)
    count = 0
    for count, frame in enumerate(records, 1):
        validate_frame(frame, count)
        # Serving verification cooperatively releases the GIL between bounded
        # records. Producer verification keeps its original synchronous path.
        if stop is not None and stop.wait(0.001):
            raise InterruptedError("replay bank closed")
    if count != header["horizon"]:
        raise ValueError("truncated replay")
    return header


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
        paths = [path, Path(str(path) + ".sha256"), Path(str(path) + ".complete")]
        if any(p.is_symlink() or not p.is_file() for p in paths):
            raise ValueError("incomplete replay")
        if path.stat().st_size > MAX_COMPRESSED or paths[1].stat().st_size > 1024:
            raise ValueError("oversized replay")
        if paths[2].stat().st_size > 4096 or paths[2].read_bytes() != b"complete\n":
            raise ValueError("replay completion marker")
        with path.open("rb") as stream:
            payload = stream.read(MAX_COMPRESSED + 1)
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if paths[1].read_text().split() != [self.sha256, path.name]:
            raise ValueError("replay SHA mismatch")
        key = (self.sha256, product)
        if cache is not None and key in cache:
            self.header = cache[key]
        else:
            self.header = verify_payload(payload, product=product, stop=stop)
            if cache is not None:
                cache[key] = self.header
        self._records = _records(payload)
        next(self._records)
        self.frame = next(self._records)
        self.path = path

    def step(self) -> bool:
        """Advance one frame; false asks the caller to rotate or use live Watch."""
        frame = next(self._records, None)
        if frame is None:
            return False
        self.frame = frame
        return True

    def snapshot(self, session: Any) -> dict:
        """Keep recorded inspector/roster, overlay only present playback controls."""
        frame = copy.deepcopy(self.frame)
        frame.update(paused=not session.playing, viewer_count=session.viewer_count)
        frame["session"].update(
            playing=session.playing,
            speed=session.speed,
            speed_ceiling=session.speed_ceiling,
            error=session.last_error,
            watch_source="replay",
            replay=self.info(),
        )
        return frame

    def info(self) -> dict:
        return {
            "label": (
                "Recorded assisted hero"
                if self.header["controller"] == "exact-stop-l2-prime"
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

    def __init__(self, root: Path) -> None:
        self.root = root
        self.error: str | None = None
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
            paths = sorted(self.root.glob("*.watch.gz"))[:1024]
            verified = []
            last = None
            seen: set[str] = set()
            for path in paths:
                if self._closed.is_set():
                    return
                try:
                    record = Recording(path, cache=cache, stop=self._closed)
                    uid = record.header["unit_id"]
                    verified.append((path, uid))
                    if uid != last and uid not in seen:
                        if not self._enqueue(record):
                            return
                        last = uid
                        seen.add(uid)
                except InterruptedError:
                    return
                except Exception as exc:
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
                        # A replaced file must retain its bank identity.
                        if chosen.header["unit_id"] != uid:
                            raise ValueError("replaced replay identity")
                        break
                    except InterruptedError:
                        return
                    except Exception as exc:
                        logger.warning("Skipping replay %s: %s", path, exc)
                        verified.remove((path, uid))
                        chosen = None
                if chosen is None:
                    continue
                if not self._enqueue(chosen):
                    return
                last = chosen.header["unit_id"]
                seen.add(last)
        except Exception as exc:
            self.error = f"Replay bank unavailable; using live B8: {exc}"
            logger.exception("Replay prefetch failed")
        finally:
            self._done.set()
