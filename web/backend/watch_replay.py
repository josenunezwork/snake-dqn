"""Versioned, verified Watch frame recordings; no model execution at playback."""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import random
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
        or frame.get("frame") != index
        or frame.get("mechanics_version") != 2
        or not required <= frame.keys()
        or not isinstance(frame.get("snakes"), list)
        or len(frame["snakes"]) != 6
        or {s.get("id") for s in frame["snakes"]} != set(range(6))
        or frame["session"].get("mode") != "watch"
        or frame["session"].get("watch_roster") != "mixed"
        or frame.get("play") is not None
    ):
        raise ValueError(f"invalid Watch replay frame {index}")
    for snake in frame["snakes"]:
        for key in ("segments", "head", "length", "alive", "boosting", "color", "is_hero", "name"):
            if key not in snake:
                raise ValueError("incomplete recorded snake")


def verify_payload(payload: bytes, *, product: bool = True) -> dict:
    """Validate bounded compressed bytes before a producer marks them complete."""
    if len(payload) > MAX_COMPRESSED:
        raise ValueError("oversized replay")
    records = _records(payload)
    header = next(records)
    validate_header(header, product=product)
    count = 0
    for count, frame in enumerate(records, 1):
        validate_frame(frame, count)
    if count != header["horizon"]:
        raise ValueError("truncated replay")
    return header


class Recording:
    """Pin compressed bytes once; verify every record before exposing frame one."""

    def __init__(self, path: Path, *, product: bool = True) -> None:
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
        self.header = verify_payload(payload, product=product)
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
    """Shuffle each local bank cycle, excluding the last session on rotation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.last: str | None = None
        self.seen: set[str] = set()
        self.error: str | None = None

    def pick(self) -> Recording | None:
        paths = sorted(self.root.glob("*.watch.gz"))[:1024]
        random.SystemRandom().shuffle(paths)
        repeat_cycle: list[Path] = []
        for path in paths:
            try:
                record = Recording(path)
                if record.header["unit_id"] != self.last:
                    if record.header["unit_id"] in self.seen:
                        repeat_cycle.append(path)
                    else:
                        self.last = record.header["unit_id"]
                        self.seen.add(self.last)
                        self.error = None
                        return record
            except Exception as exc:
                self.error = f"Replay verification failed: {path.name}: {exc}"
                # Verification failure immediately returns control to live B8.
                return None
        if not repeat_cycle:
            self.error = "No verified replay without an immediate repeat; using live B8."
            return None
        self.seen.clear()
        chosen = Recording(repeat_cycle[0])
        self.last = chosen.header["unit_id"]
        self.seen.add(self.last)
        self.error = None
        return chosen
