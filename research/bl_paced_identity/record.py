"""Create-only, hash-chained event journals for paced identity evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from research.bl_paced_identity.events import (
    Event,
    EventValidationError,
    canonical_bytes,
    canonical_clone,
    sha256,
)

SCHEMA = "bl-paced-identity-journal/v1"
_HEX = set("0123456789abcdef")


class JournalError(RuntimeError):
    """Raised when a paced journal is incomplete, unsafe, or internally inconsistent."""


class JournalOverflow(JournalError):
    """Raised when the on-tick ring could have dropped identity evidence."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".paced-journal-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"create-only export already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def export_json(path: Path, value: Any) -> str:
    """Publish finite canonical JSON plus immutable digest and completion sidecars."""
    path = Path(path)
    sidecars = (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    if any(item.exists() for item in sidecars):
        raise FileExistsError(f"create-only export already exists: {path}")
    payload = canonical_bytes(value)
    _publish(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        _publish(Path(str(path) + ".sha256"), f"{digest}  {path.name}\n".encode("utf-8"))
        _publish(Path(str(path) + ".complete"), b"complete\n")
    except Exception as exc:
        raise JournalError(f"incomplete export sidecars: {path}") from exc
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest


def read_export(path: Path) -> dict[str, Any]:
    """Read a completed JSON object only after verifying every sidecar."""
    path = Path(path)
    pieces = (path, Path(str(path) + ".sha256"), Path(str(path) + ".complete"))
    if any(piece.is_symlink() or not piece.is_file() for piece in pieces):
        raise JournalError(f"missing or unsafe export component: {path}")
    if pieces[2].read_bytes() != b"complete\n":
        raise JournalError(f"invalid completion marker: {path}")
    observed = _file_sha256(path)
    if pieces[1].read_text(encoding="utf-8").split() != [observed, path.name]:
        raise JournalError(f"digest/name mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JournalError(f"invalid JSON export: {path}") from exc
    if not isinstance(value, dict):
        raise JournalError(f"expected JSON object: {path}")
    if canonical_bytes(value) != path.read_bytes():
        raise JournalError(f"noncanonical JSON export: {path}")
    return value


class Journal:
    """A bounded coordinator ring that drains into immutable hash-chained chunks."""

    def __init__(
        self,
        directory: Path,
        identity: Mapping[str, Any],
        initial_state: Mapping[str, Any],
        *,
        capacity: int = 256,
        chunk_size: int = 64,
        binding_sha256: str | None = None,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise JournalError("capacity must be a positive integer")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or not 1 <= chunk_size <= capacity
        ):
            raise JournalError("chunk_size must be in 1..capacity")
        if not isinstance(identity, Mapping) or not isinstance(initial_state, Mapping):
            raise JournalError("identity and initial_state must be objects")
        if binding_sha256 is not None and not _is_sha256(binding_sha256):
            raise JournalError("binding_sha256 must be a lowercase sha256 or None")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._events: list[Event] = []
        self._capacity = capacity
        self._chunk_size = chunk_size
        self._next_seq = 0
        self._last_time_ns = -1
        self._next_chunk = 0
        identity_snapshot = canonical_clone(dict(identity))
        initial_state_snapshot = canonical_clone(dict(initial_state))
        self._previous_chunk_sha = sha256(
            {"schema": SCHEMA, "identity": identity_snapshot, "binding_sha256": binding_sha256}
        )
        self._chunk_shas: list[str] = []
        self._invalid_reason: str | None = None
        self._sealed = False
        self._header = {
            "schema": SCHEMA,
            "identity": identity_snapshot,
            "initial_state": initial_state_snapshot,
            "capacity": capacity,
            "chunk_size": chunk_size,
            "binding_sha256": binding_sha256,
            "chain_initial_sha256": self._previous_chunk_sha,
        }
        self._header_sha = export_json(self.directory / "header.json", self._header)

    def emit(self, time_ns: int, kind: str, **data: Any) -> Event:
        """Append one coordinator event without doing filesystem I/O."""
        if self._sealed:
            raise JournalError("cannot emit after terminal seal")
        if self._invalid_reason is not None:
            raise JournalError(f"journal is invalid: {self._invalid_reason}")
        if len(self._events) >= self._capacity:
            self._invalid_reason = "ring_overflow"
            raise JournalOverflow("journal ring overflow; evidence is invalid")
        try:
            event = Event(seq=self._next_seq, time_ns=time_ns, kind=kind, data=dict(data))
        except EventValidationError as exc:
            self._invalid_reason = "invalid_event"
            raise JournalError(str(exc)) from exc
        if event.time_ns < self._last_time_ns:
            self._invalid_reason = "backwards_time"
            raise JournalError("event time_ns moved backwards")
        self._events.append(Event.from_dict(event.as_dict()))
        self._next_seq += 1
        self._last_time_ns = event.time_ns
        return event

    def drain(self) -> list[str]:
        """Flush full chunks off tick; callers choose when this I/O is safe."""
        if self._sealed:
            raise JournalError("cannot drain after terminal seal")
        if self._invalid_reason is not None:
            raise JournalError(f"journal is invalid: {self._invalid_reason}")
        written: list[str] = []
        while self._events:
            selected = self._events[: self._chunk_size]
            chunk = {
                "schema": SCHEMA,
                "index": self._next_chunk,
                "header_sha256": self._header_sha,
                "previous_chunk_sha256": self._previous_chunk_sha,
                "events": [event.as_dict() for event in selected],
            }
            path = self.directory / f"chunk-{self._next_chunk:08d}.json"
            digest = export_json(path, chunk)
            del self._events[: len(selected)]
            self._previous_chunk_sha = digest
            self._chunk_shas.append(digest)
            self._next_chunk += 1
            written.append(digest)
        return written

    def seal(self) -> dict[str, Any]:
        """Flush remaining evidence and publish exactly one terminal seal."""
        if self._sealed:
            raise JournalError("journal already sealed")
        if self._invalid_reason is not None:
            raise JournalError(f"cannot seal invalid journal: {self._invalid_reason}")
        self.drain()
        seal = {
            "schema": SCHEMA,
            "status": "SEALED",
            "header_sha256": self._header_sha,
            "event_count": self._next_seq,
            "chunk_count": self._next_chunk,
            "chunk_sha256": list(self._chunk_shas),
            "terminal_chunk_sha256": self._previous_chunk_sha,
        }
        export_json(self.directory / "seal.json", seal)
        self._sealed = True
        return seal


def read_journal(directory: Path) -> tuple[dict[str, Any], list[Event], dict[str, Any]]:
    """Read a sealed journal and reject any missing, reordered, or altered evidence."""
    directory = Path(directory)
    header = read_export(directory / "header.json")
    if header.get("schema") != SCHEMA:
        raise JournalError("unsupported journal schema")
    required_header = {
        "schema",
        "identity",
        "initial_state",
        "capacity",
        "chunk_size",
        "binding_sha256",
        "chain_initial_sha256",
    }
    if set(header) != required_header or not isinstance(header["identity"], dict):
        raise JournalError("invalid journal header")
    if (
        isinstance(header["capacity"], bool)
        or not isinstance(header["capacity"], int)
        or header["capacity"] < 1
        or isinstance(header["chunk_size"], bool)
        or not isinstance(header["chunk_size"], int)
        or not 1 <= header["chunk_size"] <= header["capacity"]
    ):
        raise JournalError("invalid journal limits")
    if header["binding_sha256"] is not None and not _is_sha256(header["binding_sha256"]):
        raise JournalError("invalid header binding")
    if header["chain_initial_sha256"] != sha256(
        {
            "schema": SCHEMA,
            "identity": header["identity"],
            "binding_sha256": header["binding_sha256"],
        }
    ):
        raise JournalError("invalid chain initial digest")
    seal = read_export(directory / "seal.json")
    required_seal = {
        "schema",
        "status",
        "header_sha256",
        "event_count",
        "chunk_count",
        "chunk_sha256",
        "terminal_chunk_sha256",
    }
    if set(seal) != required_seal or seal["schema"] != SCHEMA or seal["status"] != "SEALED":
        raise JournalError("missing or invalid terminal seal")
    header_sha = _file_sha256(directory / "header.json")
    if seal["header_sha256"] != header_sha:
        raise JournalError("seal header mismatch")
    if (
        not isinstance(seal["chunk_count"], int)
        or isinstance(seal["chunk_count"], bool)
        or seal["chunk_count"] < 0
    ):
        raise JournalError("invalid seal chunk count")
    if (
        not isinstance(seal["event_count"], int)
        or isinstance(seal["event_count"], bool)
        or seal["event_count"] < 0
    ):
        raise JournalError("invalid seal event count")
    if (
        not isinstance(seal["chunk_sha256"], list)
        or len(seal["chunk_sha256"]) != seal["chunk_count"]
    ):
        raise JournalError("seal chunk count mismatch")
    expected_names = {f"chunk-{index:08d}.json" for index in range(seal["chunk_count"])}
    expected_parts = {
        f"{name}{suffix}" for name in expected_names for suffix in ("", ".sha256", ".complete")
    }
    observed_parts = {path.name for path in directory.glob("chunk-*")}
    if observed_parts != expected_parts:
        raise JournalError("journal chunks are missing or unexpected")
    previous = header.get("chain_initial_sha256")
    if not isinstance(previous, str):
        raise JournalError("invalid chain initial digest")
    events: list[Event] = []
    last_time = -1
    for index in range(seal["chunk_count"]):
        path = directory / f"chunk-{index:08d}.json"
        chunk = read_export(path)
        required_chunk = {"schema", "index", "header_sha256", "previous_chunk_sha256", "events"}
        if set(chunk) != required_chunk or chunk["schema"] != SCHEMA or chunk["index"] != index:
            raise JournalError(f"invalid chunk structure: {path}")
        if chunk["header_sha256"] != header_sha or chunk["previous_chunk_sha256"] != previous:
            raise JournalError(f"broken chunk chain: {path}")
        digest = _file_sha256(path)
        if seal["chunk_sha256"][index] != digest:
            raise JournalError(f"sealed chunk digest mismatch: {path}")
        if not isinstance(chunk["events"], list) or not chunk["events"]:
            raise JournalError(f"invalid event chunk: {path}")
        if len(chunk["events"]) > header["chunk_size"]:
            raise JournalError(f"chunk exceeds frozen chunk_size: {path}")
        for raw in chunk["events"]:
            try:
                event = Event.from_dict(raw)
            except EventValidationError as exc:
                raise JournalError(f"invalid event in {path}") from exc
            if event.seq != len(events) or event.time_ns < last_time:
                raise JournalError("out-of-order event sequence or timestamp")
            events.append(event)
            last_time = event.time_ns
        previous = digest
    if seal["event_count"] != len(events) or seal["terminal_chunk_sha256"] != previous:
        raise JournalError("terminal seal count or chain mismatch")
    return header, events, seal
