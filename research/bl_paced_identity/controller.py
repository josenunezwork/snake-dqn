"""Instrumented tick-consumption boundary; no simulator or planner work on tick.

The caller supplies already-captured immutable roots and a target-root adapter.
This module does not replace a research planner or alter the served application.
All nondeterministic observations enter through explicit methods and are journaled.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict
from typing import Any, Callable

from .events import CandidateIdentity, Request, canonical_bytes, sha256


class PacedController:
    """Bounded, nonwaiting controller with exact target and post-validation fences."""

    def __init__(
        self,
        identity: CandidateIdentity,
        journal: Any,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        synthetic: bool = False,
    ) -> None:
        self.identity = identity
        self.journal = journal
        self.clock = clock
        self.pending: dict[str, dict[str, Any]] = {}
        self.seen: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.active: dict[str, Any] | None = None
        self.epoch = 0
        self.faulted = False
        self.synthetic = synthetic

    def _emit(self, kind: str, **data: Any) -> Any:
        return self.journal.emit(self.clock(), kind, **copy.deepcopy(data))

    def _fault(self, request_id: str, reason: str) -> None:
        self.faulted = True
        self._emit("fault", request_id=request_id, reason=reason)

    def submit(self, request: Request, payload: dict[str, Any]) -> bool:
        """Admit a complete root; queued, running and unconsumed all occupy K."""
        raw = asdict(request)
        event = self._emit("submit_attempt", request=raw, payload=payload)
        rid = request.request_id
        if sha256(payload) != request.payload_sha:
            self._fault(rid, "request_payload_sha")
            return False
        if rid in self.seen:
            if self.seen[rid] != {"request": raw, "payload": payload}:
                self._fault(rid, "conflicting_duplicate")
                return False
            reason = "duplicate"
        elif event.time_ns >= request.publish_cutoff_ns:
            reason = "expired"
        elif request.target_frame - request.source_frame > 432:
            reason = "lead_exceeded"
        elif len(self.pending) >= self.identity.max_pending:
            reason = "saturated"
        else:
            self.seen[rid] = copy.deepcopy({"request": raw, "payload": payload})
            self.pending[rid] = {
                "request": raw,
                "visible": None,
                "fenced": False,
                "stage": "queued",
            }
            self._emit("submit", request_id=rid)
            return True
        self._emit("refusal", request_id=rid, reason=reason)
        return False

    def worker_complete(self, request_id: str, completed_ns: int) -> None:
        """Completion is telemetry, never coordinator availability or delivery."""
        self._emit("worker_complete", request_id=request_id, completed_ns=completed_ns)
        pending = self.pending.get(request_id)
        if pending is not None:
            if pending["stage"] != "running" and not self.synthetic:
                self._fault(request_id, "untracked_completion")
                return
            pending["stage"] = "completed"

    def dispatch(self, request_id: str) -> bool:
        """Journal and validate one EDF queued-to-running process transition."""
        self._emit("dispatch_attempt", request_id=request_id)
        queued = [
            item
            for item in self.pending.values()
            if item["stage"] == "queued" and not item["fenced"]
        ]
        next_item = min(
            queued,
            key=lambda item: (
                item["request"]["publish_cutoff_ns"],
                item["request"]["target_frame"],
                item["request"]["request_id"],
            ),
            default=None,
        )
        running = sum(item["stage"] == "running" for item in self.pending.values())
        if (
            next_item is None
            or next_item["request"]["request_id"] != request_id
            or running >= self.identity.workers
        ):
            self._fault(request_id, "dispatch_identity")
            return False
        next_item["stage"] = "running"
        self._emit("dispatch", request_id=request_id)
        return True

    def publish(self, request_id: str, payload: dict[str, Any], completed_ns: int) -> None:
        """Observe a response at the coordinator; store the entire result."""
        event = self._emit(
            "result_visible", request_id=request_id, payload=payload, completed_ns=completed_ns
        )
        value = {"payload": payload, "completed_ns": completed_ns}
        if request_id in self.results:
            if self.results[request_id] != value:
                self._fault(request_id, "conflicting_result")
            return
        self.results[request_id] = copy.deepcopy(value)
        pending = self.pending.get(request_id)
        if pending is None:
            return
        if pending["stage"] != "completed" and not self.synthetic:
            self._fault(request_id, "untracked_completion")
            return
        if pending["fenced"]:
            del self.pending[request_id]
            return
        pending["visible"] = copy.deepcopy({**value, "visible_ns": event.time_ns})

    def worker_error(self, request_id: str, error: str) -> None:
        """Retain all worker failures; a failed root always falls back."""
        self._emit("worker_error", request_id=request_id, error=error)
        self.pending.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        """Fence publication without pretending an unfinished worker disappeared."""
        self._emit("cancel", request_id=request_id)
        pending = self.pending.get(request_id)
        if pending is not None:
            if pending["visible"] is not None:
                del self.pending[request_id]
            else:
                pending["fenced"] = True
        if self.active and self.active["request_id"] == request_id:
            self.active["cancelled"] = True
        self._emit("cancellation", request_id=request_id)

    def invalidate(self, reason: str, shadow_epoch: int) -> None:
        """Fence after respawn, host pause/resume, checkpoint or world change."""
        if shadow_epoch <= self.epoch:
            raise ValueError("invalidation epoch must advance")
        self._emit("invalidate", reason=reason, shadow_epoch=shadow_epoch)
        self._fence(reason, shadow_epoch)

    def _fence(self, reason: str, shadow_epoch: int) -> None:
        self.epoch = shadow_epoch
        for rid, item in list(self.pending.items()):
            if item["visible"] is not None:
                del self.pending[rid]
            else:
                item["fenced"] = True
        self._emit("invalidation", reason=reason, shadow_epoch=shadow_epoch)

    def progress(self, value: dict[str, Any]) -> None:
        """Record worker reads that influence root selection, before using them."""
        self._emit("progress", value=value)

    def read_clock(self, label: str) -> int:
        value = self.clock()
        self.journal.emit(value, "clock_read", label=label, value_ns=value)
        return value

    def begin_poll(self, context: dict[str, Any], bare: int, mask: list[bool]) -> int | None:
        """Start one tick; None means validation is required before consumption."""
        if self.active is not None:
            raise RuntimeError("unfinished tick validation")
        if type(bare) is not int or not 0 <= bare < 6 or len(mask) != 6:
            raise ValueError("six-action bare context required")
        event = self._emit("poll", context=context, bare=bare, mask=mask)
        available = sorted(
            (
                item
                for item in self.pending.values()
                if item["visible"] is not None
                and not item["fenced"]
                and item["request"]["target_frame"] <= context["frame"]
            ),
            key=lambda item: (
                item["request"]["publish_cutoff_ns"],
                item["request"]["target_frame"],
                item["request"]["request_id"],
            ),
        )
        if not available:
            return self._decision(None, bare, "unavailable", False, bare)
        item = available[0]
        raw = item["request"]
        rid = raw["request_id"]
        self.active = {
            "request_id": rid,
            "context": copy.deepcopy(context),
            "bare": bare,
            "mask": list(mask),
            "valid": False,
        }
        if (
            event.time_ns >= raw["consume_cutoff_ns"]
            or item["visible"]["visible_ns"] >= raw["publish_cutoff_ns"]
        ):
            return self._reject("cutoff")
        if not self._matches(raw, context) or context["shadow_epoch"] != self.epoch:
            return self._reject("stale")
        return None

    @staticmethod
    def _matches(request: dict[str, Any], context: dict[str, Any]) -> bool:
        return all(
            request[key] == context["frame" if key == "target_frame" else key]
            for key in (
                "session",
                "life",
                "target_frame",
                "checkpoint_sha",
                "shadow_epoch",
                "root_digest",
            )
        )

    def validation_complete(self, request_id: str, valid: bool) -> None:
        event = self._emit("validation_complete", request_id=request_id, valid=valid)
        if self.active is None or self.active["request_id"] != request_id:
            raise RuntimeError("validation without matching poll")
        self.active.update(valid=bool(valid), validated_ns=event.time_ns)

    def consume(self, request_id: str) -> int:
        """Re-read the clock after validation; equality at cutoff is a miss."""
        self._emit("consume", request_id=request_id)
        active = self.active
        if active is None or active["request_id"] != request_id:
            raise RuntimeError("consumption without matching poll")
        item = self.pending.get(request_id)
        if item is None or item["fenced"] or active.get("cancelled"):
            return self._commit(active, active["bare"], "stale", False, self.clock())
        raw = item["request"]
        visible = item["visible"]
        payload = visible["payload"]
        action = payload.get("action")
        rejection = None
        if not active["valid"] or "validated_ns" not in active:
            rejection = "validation_failed"
        elif (
            payload.get("request") != raw
            or payload.get("bare") != active["bare"]
            or payload.get("mask") != active["mask"]
        ):
            rejection = "invalid_payload"
        elif type(action) is not int or not 0 <= action < 6 or not active["mask"][action]:
            rejection = "invalid_action"
        # Sample after recorder/validation work. Commit controller state before
        # any further recorder call; the journal observes that committed action.
        commit_ns = self.clock()
        if (
            commit_ns >= raw["consume_cutoff_ns"]
            or active.get("validated_ns", commit_ns) >= raw["consume_cutoff_ns"]
            or visible["visible_ns"] >= raw["publish_cutoff_ns"]
            or visible["completed_ns"] >= raw["publish_cutoff_ns"]
        ):
            rejection = "cutoff"
        if rejection is not None:
            return self._commit(active, active["bare"], rejection, False, commit_ns)
        return self._commit(active, action, "delivered", True, commit_ns)

    def _commit(
        self, active: dict[str, Any], action: int, reason: str, accepted: bool, commit_ns: int
    ) -> int:
        """Commit in-memory action/state before recording the commit observation."""
        request_id = active["request_id"]
        self.pending.pop(request_id, None)
        self.active = None
        self.journal.emit(commit_ns, "commit", request_id=request_id)
        result = self._decision(
            request_id, action, reason, accepted, active["bare"], time_ns=commit_ns
        )
        if accepted and result != active["bare"]:
            self._fence("intervention", self.epoch + 1)
        return result

    def _reject(self, reason: str) -> int:
        active = self.active
        if active is None:
            raise RuntimeError("rejection without active poll")
        self.pending.pop(active["request_id"], None)
        self.active = None
        return self._decision(active["request_id"], active["bare"], reason, False, active["bare"])

    def _decision(
        self,
        rid: str | None,
        action: int,
        reason: str,
        accepted: bool,
        bare: int,
        *,
        time_ns: int | None = None,
    ) -> int:
        self.journal.emit(
            self.clock() if time_ns is None else time_ns,
            "decision",
            request_id=rid,
            action=action,
            reason=reason,
            accepted=accepted,
            intervened=accepted and action != bare,
        )
        return action

    def poll(
        self,
        context: dict[str, Any],
        bare: int,
        mask: list[bool],
        validate: Callable[[dict[str, Any]], bool],
    ) -> int:
        """Validate visible results immediately, then consume or return bare."""
        action = self.begin_poll(context, bare, mask)
        if action is not None:
            return action
        rid = self.active["request_id"]  # type: ignore[index]
        try:
            valid = bool(validate(copy.deepcopy(self.pending[rid]["visible"]["payload"])))
        except Exception:
            valid = False
        self.validation_complete(rid, valid)
        self.read_clock("post_validation")
        return self.consume(rid)

    def game_state(self, snapshot: dict[str, Any]) -> None:
        """Persist post-action state/counters from the real game adapter."""
        canonical_bytes(snapshot)
        self._emit("game_state", **snapshot)
