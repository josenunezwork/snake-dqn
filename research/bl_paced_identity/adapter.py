"""Frozen offered-root adapter for the paced bare-shadow boundary.

The adapter only turns caller-supplied, immutable bare-shadow roots into
requests.  It does not create roots, call a planner, or qualify a candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .events import EventValidationError, Request, canonical_bytes, canonical_clone, sha256

HORIZON_CHECKPOINTS = (32, 96, 192, 300)
DECISION_SPACING = 5
MAX_LEAD_FRAMES = 432
_NANOSECONDS = 1_000_000_000


class RootSelectionController(Protocol):
    """The small journal boundary the adapter needs before selecting a root."""

    def progress(self, value: dict[str, Any]) -> None: ...

    def read_clock(self, label: str) -> int: ...


@dataclass(frozen=True)
class TickSchedule:
    """Absolute integer tick schedule with no accumulated floating-point drift."""

    epoch_ns: int
    frames_per_second: int

    def __post_init__(self) -> None:
        if type(self.epoch_ns) is not int or self.epoch_ns < 0:
            raise EventValidationError("epoch_ns must be a non-negative integer")
        if type(self.frames_per_second) is not int or self.frames_per_second < 1:
            raise EventValidationError("frames_per_second must be a positive integer")

    def at_frame(self, frame: int) -> int:
        """Return the absolute tick time using one exact integer division."""
        if type(frame) is not int or frame < 0:
            raise EventValidationError("frame must be a non-negative integer")
        return self.epoch_ns + frame * _NANOSECONDS // self.frames_per_second


class FrozenTargetRoot:
    """A deep-frozen reachable target root supplied by the game-specific caller.

    ``root`` and ``planner_input`` are stored as canonical bytes.  The public
    accessors always return fresh JSON values, so later caller mutation cannot
    change the digest or a request built from this root.
    """

    __slots__ = (
        "session",
        "life",
        "source_frame",
        "target_frame",
        "checkpoint_sha",
        "shadow_epoch",
        "horizon_checkpoint",
        "_root_bytes",
        "_planner_input_bytes",
        "root_digest",
        "_sealed",
    )

    def __init__(
        self,
        *,
        session: str,
        life: int,
        source_frame: int,
        target_frame: int,
        checkpoint_sha: str,
        shadow_epoch: int,
        horizon_checkpoint: int,
        root: Mapping[str, Any],
        planner_input: Mapping[str, Any],
    ) -> None:
        if not isinstance(session, str) or not session:
            raise EventValidationError("session must be a non-empty string")
        for value, name in (
            (life, "life"),
            (source_frame, "source_frame"),
            (target_frame, "target_frame"),
            (shadow_epoch, "shadow_epoch"),
        ):
            if type(value) is not int or value < 0:
                raise EventValidationError(f"{name} must be a non-negative integer")
        if target_frame <= source_frame:
            raise EventValidationError("target_frame must be after source_frame")
        if not isinstance(root, Mapping) or not root:
            raise EventValidationError("root must be a non-empty object")
        if not isinstance(planner_input, Mapping):
            raise EventValidationError("planner_input must be an object")
        if type(horizon_checkpoint) is not int or horizon_checkpoint not in HORIZON_CHECKPOINTS:
            raise EventValidationError("horizon_checkpoint must be one of 32, 96, 192, 300")
        # Request validates the SHA when this root becomes a request; doing it
        # here prevents roots that can never bind a valid request.
        if (
            not isinstance(checkpoint_sha, str)
            or len(checkpoint_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in checkpoint_sha)
        ):
            raise EventValidationError("checkpoint_sha must be a lowercase sha256")
        self.session = session
        self.life = life
        self.source_frame = source_frame
        self.target_frame = target_frame
        self.checkpoint_sha = checkpoint_sha
        self.shadow_epoch = shadow_epoch
        self.horizon_checkpoint = horizon_checkpoint
        self._root_bytes = canonical_bytes(dict(root))
        self._planner_input_bytes = canonical_bytes(dict(planner_input))
        self.root_digest = hashlib.sha256(self._root_bytes).hexdigest()
        self._sealed = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("FrozenTargetRoot is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("FrozenTargetRoot is immutable")

    @property
    def root(self) -> dict[str, Any]:
        """Return a fresh canonical snapshot of the supplied target root."""
        return json.loads(self._root_bytes)

    @property
    def planner_input(self) -> dict[str, Any]:
        """Return a fresh canonical snapshot of the planner input."""
        return json.loads(self._planner_input_bytes)


@dataclass(frozen=True, init=False)
class PreparedRequest:
    """One immutable request/input pair ready for normal controller admission."""

    request: Request
    observed_ns: int
    _payload_bytes: bytes

    def __init__(
        self, request: Request, input_payload: Mapping[str, Any], observed_ns: int
    ) -> None:
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "observed_ns", observed_ns)
        object.__setattr__(self, "_payload_bytes", canonical_bytes(dict(input_payload)))

    @property
    def input_payload(self) -> dict[str, Any]:
        """Return a fresh input snapshot bound by ``request.payload_sha``."""
        return json.loads(self._payload_bytes)


class CandidateRequestAdapter:
    """Select the earliest future offered root and bind it to an exact request."""

    def __init__(self, schedule: TickSchedule) -> None:
        self.schedule = schedule

    @staticmethod
    def _context(context: Mapping[str, Any]) -> dict[str, Any]:
        required = ("session", "life", "frame", "checkpoint_sha", "shadow_epoch")
        if not isinstance(context, Mapping) or any(key not in context for key in required):
            raise EventValidationError("root selection context is incomplete")
        copied = canonical_clone(dict(context))
        for key in ("life", "frame", "shadow_epoch"):
            if type(copied[key]) is not int or copied[key] < 0:
                raise EventValidationError(f"context {key} must be a non-negative integer")
        if not isinstance(copied["session"], str) or not copied["session"]:
            raise EventValidationError("context session must be a non-empty string")
        return copied

    @staticmethod
    def _eligible(
        root: FrozenTargetRoot, context: Mapping[str, Any], schedule: TickSchedule, observed_ns: int
    ) -> bool:
        if (
            root.session != context["session"]
            or root.life != context["life"]
            or root.source_frame != context["frame"]
            or root.checkpoint_sha != context["checkpoint_sha"]
            or root.shadow_epoch != context["shadow_epoch"]
        ):
            return False
        lead = root.target_frame - root.source_frame
        return (
            root.target_frame > context["frame"]
            and lead <= MAX_LEAD_FRAMES
            and lead % DECISION_SPACING == 0
            and root.horizon_checkpoint in HORIZON_CHECKPOINTS
            and schedule.at_frame(root.target_frame - 2) > observed_ns
        )

    def prepare(
        self,
        controller: RootSelectionController,
        *,
        context: Mapping[str, Any],
        roots: Iterable[FrozenTargetRoot],
        progress: Mapping[str, Any],
    ) -> PreparedRequest | None:
        """Journal observations, then select and freeze the earliest eligible root.

        ``roots`` may originate from a game-specific provider.  The adapter
        records the provider's progress and an absolute clock observation before
        iterating it; it never infers reachability or calls a planner itself.
        """
        frozen_context = self._context(context)
        controller.progress(canonical_clone(dict(progress)))
        observed_ns = controller.read_clock("candidate_root_selection")
        eligible = [
            root
            for root in roots
            if self._eligible(root, frozen_context, self.schedule, observed_ns)
        ]
        if not eligible:
            return None
        # This total order is independent of the caller's provider iteration.
        root = min(
            eligible,
            key=lambda item: (
                item.target_frame,
                item.root_digest,
                item.horizon_checkpoint,
                hashlib.sha256(item._planner_input_bytes).hexdigest(),
            ),
        )
        payload = canonical_clone(
            {
                "horizon_checkpoint": root.horizon_checkpoint,
                "planner_input": root.planner_input,
                "root": root.root,
                "root_digest": root.root_digest,
            }
        )
        request_id = sha256(
            {
                "checkpoint_sha": root.checkpoint_sha,
                "life": root.life,
                "payload_sha": sha256(payload),
                "root_digest": root.root_digest,
                "session": root.session,
                "shadow_epoch": root.shadow_epoch,
                "source_frame": root.source_frame,
                "target_frame": root.target_frame,
            }
        )
        request = Request(
            session=root.session,
            life=root.life,
            source_frame=root.source_frame,
            target_frame=root.target_frame,
            checkpoint_sha=root.checkpoint_sha,
            shadow_epoch=root.shadow_epoch,
            root_digest=root.root_digest,
            request_id=request_id,
            payload_sha=sha256(payload),
            publish_cutoff_ns=self.schedule.at_frame(root.target_frame - 2),
            consume_cutoff_ns=self.schedule.at_frame(root.target_frame + 1),
        )
        return PreparedRequest(request=request, input_payload=payload, observed_ns=observed_ns)
