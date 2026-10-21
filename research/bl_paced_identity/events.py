"""Immutable, canonical event types for the paced-identity transport boundary."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EventValidationError(ValueError):
    """Raised when evidence cannot safely enter the paced journal."""


def canonical_bytes(value: Any) -> bytes:
    """Return the one finite, order-independent JSON representation for evidence."""
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EventValidationError("value is not finite canonical JSON") from exc


def sha256(value: Any) -> str:
    """Hash canonical evidence bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_clone(value: Any) -> Any:
    """Deep-copy one JSON value while proving it is canonical and finite."""
    return json.loads(canonical_bytes(value))


def _required_string(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise EventValidationError(f"{field} must be a non-empty string")


def _sha(value: str, field: str) -> None:
    _required_string(value, field)
    if _SHA256.fullmatch(value) is None:
        raise EventValidationError(f"{field} must be a lowercase sha256")


def _nonnegative(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventValidationError(f"{field} must be a non-negative integer")


def _nonempty_metadata(value: Any, field: str) -> Any:
    """Validate and snapshot identity metadata without imposing its schema."""
    if value is None or (isinstance(value, (str, list, dict)) and not value):
        raise EventValidationError(f"{field} must be non-empty")
    return canonical_clone(value)


@dataclass(frozen=True)
class Request:
    """The complete binding for a result that may be consumed by a live tick."""

    session: str
    life: int
    source_frame: int
    target_frame: int
    checkpoint_sha: str
    shadow_epoch: int
    root_digest: str
    request_id: str
    payload_sha: str
    cutoff_ns: int

    def __post_init__(self) -> None:
        _required_string(self.session, "session")
        _nonnegative(self.life, "life")
        _nonnegative(self.source_frame, "source_frame")
        _nonnegative(self.target_frame, "target_frame")
        if self.target_frame < self.source_frame:
            raise EventValidationError("target_frame precedes source_frame")
        _sha(self.checkpoint_sha, "checkpoint_sha")
        _nonnegative(self.shadow_epoch, "shadow_epoch")
        _sha(self.root_digest, "root_digest")
        _required_string(self.request_id, "request_id")
        _sha(self.payload_sha, "payload_sha")
        _nonnegative(self.cutoff_ns, "cutoff_ns")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CandidateIdentity:
    """Frozen candidate configuration, including the declared worker envelope."""

    planner_sha: str
    scheduler: str
    horizon: Any
    tie_rule: str
    workers: int
    max_pending: int
    qos: Any
    warmup: Any
    fallback: Any
    recorder: Any
    load_envelope: Any
    checkpoint_sha: str

    def __post_init__(self) -> None:
        _sha(self.planner_sha, "planner_sha")
        _required_string(self.scheduler, "scheduler")
        _required_string(self.tie_rule, "tie_rule")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or not 1 <= self.workers <= 4
        ):
            raise EventValidationError("workers must be in 1..4")
        if (
            isinstance(self.max_pending, bool)
            or not isinstance(self.max_pending, int)
            or not 1 <= self.max_pending <= 8
        ):
            raise EventValidationError("max_pending must be in 1..8")
        if self.workers > self.max_pending:
            raise EventValidationError("workers cannot exceed max_pending")
        _sha(self.checkpoint_sha, "checkpoint_sha")
        object.__setattr__(self, "horizon", _nonempty_metadata(self.horizon, "horizon"))
        object.__setattr__(self, "qos", _nonempty_metadata(self.qos, "qos"))
        object.__setattr__(self, "warmup", _nonempty_metadata(self.warmup, "warmup"))
        object.__setattr__(self, "fallback", _nonempty_metadata(self.fallback, "fallback"))
        object.__setattr__(self, "recorder", _nonempty_metadata(self.recorder, "recorder"))
        object.__setattr__(
            self, "load_envelope", _nonempty_metadata(self.load_envelope, "load_envelope")
        )
        canonical_bytes(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def authorized(self, *, owner_multi_worker: bool = False) -> bool:
        """Return whether the frozen envelope is currently permitted to run.

        Build and fixture configurations may describe the full W<=4/K<=8 envelope.
        Runtime admission remains fail-closed at W=1/K=1 until the owner explicitly
        authorizes the multi-worker envelope.
        """
        if type(owner_multi_worker) is not bool:
            raise EventValidationError("owner_multi_worker must be an explicit boolean")
        return owner_multi_worker is True or (self.workers == 1 and self.max_pending == 1)

    def require_authorized(self, *, owner_multi_worker: bool = False) -> None:
        if not self.authorized(owner_multi_worker=owner_multi_worker):
            raise EventValidationError("multi-worker envelope requires owner authorization")


@dataclass(frozen=True)
class Event:
    """A sequenced coordinator observation; payloads are always JSON objects."""

    seq: int
    time_ns: int
    kind: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        _nonnegative(self.seq, "seq")
        _nonnegative(self.time_ns, "time_ns")
        _required_string(self.kind, "kind")
        if not isinstance(self.data, dict):
            raise EventValidationError("data must be an object")
        object.__setattr__(self, "data", canonical_clone(self.data))

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Event":
        if not isinstance(value, Mapping) or set(value) != {"seq", "time_ns", "kind", "data"}:
            raise EventValidationError("event must contain exactly seq, time_ns, kind, data")
        data = value["data"]
        if not isinstance(data, dict):
            raise EventValidationError("event data must be an object")
        return cls(seq=value["seq"], time_ns=value["time_ns"], kind=value["kind"], data=data)
