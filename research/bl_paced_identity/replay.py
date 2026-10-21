"""Independent deterministic availability-trace reducer.

This module deliberately does not import the served controller.  It consumes
only the journal's external observations and recomputes the scheduler-side
outcomes that a paced controller is allowed to produce.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
)

from research.bl_paced_identity.events import Event, EventValidationError, Request
from research.bl_paced_identity.events import sha256 as evidence_sha256
from research.bl_paced_identity.record import JournalError, read_journal

MAX_LEAD_FRAMES = 432
REPLAY_SCHEMA = "bl-paced-identity-replay/v1"
_OUTPUT_KINDS = {
    "submit",
    "refusal",
    "decision",
    "invalidation",
    "cancellation",
    "fault",
    "dispatch",
}


class ReplayGame(Protocol):
    """Small game boundary used only for the independent full replay."""

    def reset(self, initial_state: Mapping[str, Any]) -> None: ...

    def context(self) -> Mapping[str, Any]: ...

    def bare(self) -> int: ...

    def mask(self) -> List[bool]: ...

    def submission(self, now_ns: int) -> Mapping[str, Any]: ...

    def validate(self, payload: Mapping[str, Any]) -> bool: ...

    def observe(self, kind: str, time_ns: int, data: Mapping[str, Any]) -> None: ...

    def finish_expectations(self) -> Mapping[str, Any]: ...

    def step(self, action: int) -> Mapping[str, Any]: ...


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _request_data(value: Any) -> Dict[str, Any]:
    data = _plain(value)
    if not isinstance(data, dict):
        raise ValueError("request must serialize to an object")
    return data


def _context_matches(request: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    return (
        request.get("session") == context.get("session")
        and request.get("life") == context.get("life")
        and request.get("target_frame") == context.get("frame")
        and request.get("checkpoint_sha") == context.get("checkpoint_sha")
        and request.get("shadow_epoch") == context.get("shadow_epoch")
        and request.get("root_digest") == context.get("root_digest")
    )


def _legal(action: Any, mask: Sequence[Any]) -> bool:
    return (
        isinstance(action, int)
        and not isinstance(action, bool)
        and 0 <= action < len(mask)
        and bool(mask[action])
    )


class _Reducer:
    """A standalone model of availability, fencing, and tick consumption."""

    def __init__(self, header: Mapping[str, Any]) -> None:
        identity = header.get("identity", {})
        if not isinstance(identity, Mapping):
            identity = {}
        self.capacity = int(
            header.get("K", header.get("max_outstanding", identity.get("max_pending", 1)))
        )
        if not 1 <= self.capacity <= 8:
            raise ValueError("K must be in [1, 8]")
        self.workers = int(identity.get("workers", header.get("W", 1)))
        if not 1 <= self.workers <= 4:
            raise ValueError("W must be in [1, 4]")
        self.requests: Dict[str, MutableMapping[str, Any]] = {}
        self.seen_submits: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.stale_consumes: Dict[str, int] = {}
        self.consume_intents: set[str] = set()
        initial_state = header.get("initial_state", {})
        if not isinstance(initial_state, Mapping):
            initial_state = {}
        self.epoch = int(header.get("initial_shadow_epoch", initial_state.get("shadow_epoch", 0)))
        self.active_request: Optional[str] = None
        self.worker_completions: Dict[str, int] = {}
        self.structural_error: Optional[str] = None
        self.synthetic = bool(initial_state.get("synthetic", False))
        self.outputs: List[Dict[str, Any]] = []
        self.invalid = False
        self.actions: List[int] = []
        self.state_snapshots: List[Any] = []
        self.score_snapshots: List[Any] = []

    def emit(self, kind: str, **data: Any) -> None:
        self.outputs.append({"kind": kind, "data": _plain(data)})

    def _decision(
        self,
        request_id: Optional[str],
        bare: int,
        reason: str,
        *,
        accepted: bool = False,
        action: Any = None,
    ) -> None:
        chosen = action if accepted else bare
        intervened = bool(accepted and chosen != bare)
        self.emit(
            "decision",
            request_id=request_id,
            action=chosen,
            reason=reason,
            accepted=accepted,
            intervened=intervened,
        )
        self.actions.append(chosen)

    def _release(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def _invalidate(self, reason: str, epoch: int) -> None:
        # Completed visible answers have no future value after a fence.  Work
        # that is still queued/running remains charged to K until it reports.
        if epoch <= self.epoch:
            self.invalid = True
            self.structural_error = "nonadvancing_epoch"
            return
        self.epoch = epoch
        for request_id, item in list(self.requests.items()):
            if item.get("visible"):
                validation = item.get("validation")
                if validation is not None:
                    self.stale_consumes[request_id] = validation["bare"]
                self._release(request_id)
            else:
                item["fenced"] = True
                validation = item.pop("validation", None)
                if validation is not None:
                    self.stale_consumes[request_id] = validation["bare"]
        self.emit("invalidation", reason=reason, shadow_epoch=epoch)
        self.active_request = None

    def submit(self, data: Mapping[str, Any], now: int) -> None:
        request = _request_data(data.get("request"))
        try:
            Request(**request)
        except (EventValidationError, TypeError):
            self.invalid = True
            self.structural_error = "invalid_request"
            self.emit("fault", request_id=request.get("request_id"), reason="invalid_request")
            return
        payload = _plain(data.get("payload", {}))
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            self.invalid = True
            self.emit("fault", request_id=request_id, reason="invalid_request")
            return
        canonical = {"request": request, "payload": payload}
        if evidence_sha256(payload) != request.get("payload_sha"):
            self.invalid = True
            self.emit("fault", request_id=request_id, reason="request_payload_sha")
            return
        if request_id in self.seen_submits:
            if self.seen_submits[request_id] == canonical:
                self.emit("refusal", request_id=request_id, reason="duplicate")
            else:
                self.invalid = True
                self.emit("fault", request_id=request_id, reason="conflicting_duplicate")
            return
        if now >= int(request.get("cutoff_ns", -1)):
            self.emit("refusal", request_id=request_id, reason="expired")
            return
        if (
            int(request.get("target_frame", -1)) - int(request.get("source_frame", 0))
            > MAX_LEAD_FRAMES
        ):
            self.emit("refusal", request_id=request_id, reason="lead_exceeded")
            return
        if len(self.requests) >= self.capacity:
            self.emit("refusal", request_id=request_id, reason="saturated")
            return
        self.seen_submits[request_id] = canonical
        self.requests[request_id] = {
            "request": request,
            "submit_payload": payload,
            "visible": False,
            "fenced": False,
            "cancelled": False,
            "stage": "queued",
        }
        self.emit("submit", request_id=request_id)

    def visible(self, data: Mapping[str, Any]) -> None:
        request_id = data.get("request_id")
        result = {
            "payload": _plain(data.get("payload", {})),
            "completed_ns": data.get("completed_ns"),
        }
        if (
            isinstance(result["completed_ns"], bool)
            or not isinstance(result["completed_ns"], int)
            or result["completed_ns"] < 0
            or result["completed_ns"] > data.get("_event_time_ns", -1)
            or (
                request_id in self.worker_completions
                and self.worker_completions[request_id] != result["completed_ns"]
            )
        ):
            self.invalid = True
            self.structural_error = "invalid_completion"
            return
        if request_id in self.results:
            if self.results[request_id] != result:
                self.invalid = True
                self.emit("fault", request_id=request_id, reason="conflicting_result")
            return
        self.results[request_id] = result
        item = self.requests.get(request_id)
        if item is None:
            # A cancelled/invalidation-fenced result is permitted late evidence;
            # it cannot re-open a slot or become available.
            return
        if not self.synthetic and item.get("stage") != "completed":
            self.invalid = True
            self.structural_error = "untracked_completion"
            self.emit("fault", request_id=request_id, reason="untracked_completion")
            return
        if item.get("cancelled") or item.get("fenced"):
            self._release(request_id)
            return
        item["visible"] = True
        item["result_payload"] = result["payload"]
        item["completed_ns"] = data.get("completed_ns")

    def dispatch(self, data: Mapping[str, Any]) -> None:
        request_id = data.get("request_id")
        queued = [
            (rid, item)
            for rid, item in self.requests.items()
            if item.get("stage") == "queued"
            and not item.get("fenced")
            and not item.get("cancelled")
        ]
        running = sum(item.get("stage") == "running" for item in self.requests.values())
        if not queued or running >= self.workers:
            self.invalid = True
            self.structural_error = "dispatch_identity"
            self.emit("fault", request_id=request_id, reason="dispatch_identity")
            return
        expected_id, item = min(
            queued,
            key=lambda row: (
                int(row[1]["request"].get("cutoff_ns", -1)),
                int(row[1]["request"].get("target_frame", -1)),
                row[0],
            ),
        )
        if request_id != expected_id:
            self.invalid = True
            self.structural_error = "dispatch_identity"
            self.emit("fault", request_id=request_id, reason="dispatch_identity")
            return
        item["stage"] = "running"
        self.emit("dispatch", request_id=request_id)

    def poll(self, data: Mapping[str, Any], now: int) -> None:
        context = _plain(data.get("context", {}))
        bare = data.get("bare")
        mask = list(data.get("mask", []))
        if self.active_request is not None:
            self.invalid = True
            self.structural_error = "overlapping_validation"
            return
        candidates = [
            (request_id, item)
            for request_id, item in self.requests.items()
            if item.get("visible")
            and not item.get("cancelled")
            and "validation" not in item
            and item["request"].get("target_frame", -1) <= context.get("frame", -1)
        ]
        if not candidates:
            self._decision(None, bare, "unavailable")
            return
        request_id, item = min(
            candidates,
            key=lambda row: (
                int(row[1]["request"].get("cutoff_ns", -1)),
                int(row[1]["request"].get("target_frame", -1)),
                row[0],
            ),
        )
        if now >= int(item["request"].get("cutoff_ns", -1)):
            self._release(request_id)
            self._decision(request_id, bare, "cutoff")
            return
        if (
            item.get("fenced")
            or not _context_matches(item["request"], context)
            or context.get("shadow_epoch") != self.epoch
        ):
            self._release(request_id)
            self._decision(request_id, bare, "stale")
            return
        item["validation"] = {"context": context, "bare": bare, "mask": mask}
        self.active_request = request_id

    def validation_complete(self, data: Mapping[str, Any], now: int) -> None:
        item = self.requests.get(data.get("request_id"))
        if (
            item is None
            or "validation" not in item
            or self.active_request != data.get("request_id")
        ):
            if data.get("request_id") in self.stale_consumes:
                return
            self.invalid = True
            self.structural_error = "unknown_validation"
            return
        item["validation"].update({"valid": bool(data.get("valid")), "completed_ns": now})

    def consume(self, data: Mapping[str, Any], now: int) -> None:
        """Record a consume intent; the post-serialization commit decides it."""
        del now
        request_id = data.get("request_id")
        item = self.requests.get(request_id)
        if item is None:
            if request_id in self.stale_consumes:
                self.consume_intents.add(request_id)
                return
            self.invalid = True
            self.structural_error = "unknown_consume"
            return
        if "validation" not in item or self.active_request != request_id:
            self.invalid = True
            self.structural_error = "unknown_consume"
            return
        item["intent"] = True
        self.consume_intents.add(request_id)

    def commit(self, data: Mapping[str, Any], now: int) -> None:
        request_id = data.get("request_id")
        if request_id not in self.consume_intents:
            self.invalid = True
            self.structural_error = "commit_without_consume"
            return
        self.consume_intents.remove(request_id)
        item = self.requests.get(request_id)
        if item is None or "validation" not in item:
            if request_id in self.stale_consumes:
                self._decision(request_id, self.stale_consumes.pop(request_id), "stale")
            else:
                self.invalid = True
                self.structural_error = "unknown_consume"
            return
        if not item.pop("intent", False):
            self.invalid = True
            self.structural_error = "commit_without_consume"
            return
        self.active_request = None
        validation = item["validation"]
        bare = validation["bare"]
        request = item["request"]
        if item.get("fenced") or item.get("cancelled"):
            self._release(request_id)
            self._decision(request_id, bare, "stale")
            return
        if (
            now >= int(request.get("cutoff_ns", -1))
            or int(validation.get("completed_ns", now)) >= int(request.get("cutoff_ns", -1))
            or int(item.get("completed_ns", now)) >= int(request.get("cutoff_ns", -1))
        ):
            self._release(request_id)
            self._decision(request_id, bare, "cutoff")
            return
        if not validation.get("valid", False):
            self._release(request_id)
            self._decision(request_id, bare, "validation_failed")
            return
        payload = item.get("result_payload")
        valid_payload = isinstance(payload, dict) and payload.get("request") == request
        valid_payload = (
            valid_payload
            and payload.get("bare") == bare
            and payload.get("mask") == validation["mask"]
        )
        if not valid_payload:
            self._release(request_id)
            self._decision(request_id, bare, "invalid_payload")
            return
        action = payload.get("action")
        if not _legal(action, validation["mask"]):
            self._release(request_id)
            self._decision(request_id, bare, "invalid_action")
            return
        self._release(request_id)
        self._decision(request_id, bare, "delivered", accepted=True, action=action)
        if action != bare:
            self._invalidate("intervention", int(request.get("shadow_epoch", 0)) + 1)

    def cancel(self, data: Mapping[str, Any]) -> None:
        request_id = data.get("request_id")
        item = self.requests.get(request_id)
        if item is not None:
            if item.get("visible"):
                validation = item.get("validation")
                if validation is not None:
                    self.stale_consumes[request_id] = validation["bare"]
                self._release(request_id)
            else:
                item["cancelled"] = True
                validation = item.pop("validation", None)
                if validation is not None:
                    self.stale_consumes[request_id] = validation["bare"]
        self.emit("cancellation", request_id=request_id)

    def worker_error(self, data: Mapping[str, Any]) -> None:
        request_id = data.get("request_id")
        item = self.requests.get(request_id)
        queued_cleanup = (
            item is not None
            and item.get("stage") == "queued"
            and (item.get("fenced") or item.get("cancelled"))
            and data.get("error")
            in {"cancelled_before_dispatch", "transport_closed_before_dispatch"}
        )
        if (
            item is not None
            and not self.synthetic
            and item.get("stage") != "running"
            and not queued_cleanup
        ):
            self.invalid = True
            self.structural_error = "worker_error_untracked"
        if item is not None and item.get("validation") is not None:
            self.stale_consumes[request_id] = item["validation"]["bare"]
        self._release(request_id)

    def handle(self, event: Event) -> None:
        data = _plain(event.data)
        data["_event_time_ns"] = event.time_ns
        if event.kind == "submit_attempt":
            self.submit(data, event.time_ns)
        elif event.kind == "result_visible":
            self.visible(data)
        elif event.kind == "poll":
            self.poll(data, event.time_ns)
        elif event.kind == "validation_complete":
            self.validation_complete(data, event.time_ns)
        elif event.kind == "consume":
            self.consume(data, event.time_ns)
        elif event.kind == "commit":
            self.commit(data, event.time_ns)
        elif event.kind == "invalidate":
            self._invalidate(str(data.get("reason")), int(data.get("shadow_epoch", 0)))
        elif event.kind == "cancel":
            self.cancel(data)
        elif event.kind == "worker_error":
            self.worker_error(data)
        elif event.kind in {"pause", "resume"}:
            self._invalidate(event.kind, int(data.get("shadow_epoch", 0)))
        elif event.kind == "worker_complete":
            item = self.requests.get(data.get("request_id"))
            completed = data.get("completed_ns")
            request_id = data.get("request_id")
            if (
                isinstance(completed, bool)
                or not isinstance(completed, int)
                or completed < 0
                or completed > event.time_ns
            ):
                self.invalid = True
                self.structural_error = "invalid_completion"
            else:
                if request_id in self.worker_completions:
                    if self.worker_completions[request_id] != completed:
                        self.invalid = True
                        self.structural_error = "conflicting_completion"
                    return
                self.worker_completions[request_id] = completed
                if item is not None:
                    if not self.synthetic and item.get("stage") != "running":
                        self.invalid = True
                        self.structural_error = "worker_complete_untracked"
                        return
                    item["completed_ns"] = completed
                    item["stage"] = "completed"
        elif event.kind == "dispatch_attempt":
            self.dispatch(data)
        elif event.kind == "clock_read":
            if data.get("value_ns") != event.time_ns:
                self.invalid = True
                self.structural_error = "clock_mismatch"
        elif event.kind in {"progress", "game_state"}:
            return
        else:
            self.invalid = True
            self.emit("fault", request_id=None, reason="unknown_event")


def _divergence(
    event: Event | None,
    *,
    expected: Any,
    observed: Any,
    actions: Sequence[int],
    states: Sequence[Any],
    scores: Sequence[Any],
) -> Dict[str, Any]:
    return {
        "request_id": None if event is None else _plain(event.data).get("request_id"),
        "frame": None if event is None else _plain(event.data).get("context", {}).get("frame"),
        "event": None if event is None else event.kind,
        "expected": _plain(expected),
        "observed": _plain(observed),
        "action_prefix": list(actions),
        "state_prefix": list(states),
        "score_prefix": list(scores),
    }


def replay_events(
    header: Mapping[str, Any],
    events: Iterable[Event],
    game_factory: Optional[Callable[[Mapping[str, Any]], ReplayGame]] = None,
) -> Dict[str, Any]:
    """Replay a sealed availability trace without importing controller code.

    Scheduler replay can pass without a game hook.  The overall verdict remains
    ``INCOMPLETE`` in that mode because scores and state digests were not
    independently regenerated.
    """
    stream = list(events)
    reducer = _Reducer(header)
    identity = header.get("identity", {})
    if not isinstance(identity, Mapping):
        identity = {}
    candidate_sha256 = evidence_sha256(dict(identity))
    binding_sha256 = header.get("binding_sha256")
    expected_by_trigger: Dict[int, List[Dict[str, Any]]] = {}
    trigger_seq: Optional[int] = None
    for recorded in stream:
        if recorded.kind in _OUTPUT_KINDS:
            expected_by_trigger.setdefault(-1 if trigger_seq is None else trigger_seq, []).append(
                {"kind": recorded.kind, "data": _plain(recorded.data)}
            )
        else:
            trigger_seq = recorded.seq
    game: Optional[ReplayGame] = None
    hooks_ready = False
    if game_factory is not None:
        game = game_factory(header)
        hooks_ready = all(
            callable(getattr(game, name, None))
            for name in (
                "reset",
                "context",
                "bare",
                "mask",
                "step",
                "submission",
                "validate",
                "observe",
                "finish_expectations",
            )
        )
        if not hooks_ready:
            return {
                "schema": REPLAY_SCHEMA,
                "verdict": "INCOMPLETE",
                "scheduler_verdict": "INCOMPLETE",
                "action_hash": _digest([]),
                "state_hash": _digest([]),
                "score_hash": _digest([]),
                "first_divergence": None,
                "evidence_error": "missing_independent_game_hooks",
                "candidate_sha256": candidate_sha256,
                "binding_sha256": binding_sha256,
                "qualification": "PRECHECK_ONLY",
                "counts": {
                    "events": len(stream),
                    "expected_outputs": 0,
                    "outputs": 0,
                    "decisions": 0,
                },
            }
        game.reset(_plain(header.get("initial_state", {})))
    divergence: Optional[Dict[str, Any]] = None
    last_seq: Optional[int] = None
    pending_actions: List[int] = []
    for event in stream:
        try:
            Event.from_dict(event.as_dict())
        except (AttributeError, EventValidationError):
            divergence = _divergence(
                None,
                expected="valid Event",
                observed=event,
                actions=reducer.actions,
                states=reducer.state_snapshots,
                scores=reducer.score_snapshots,
            )
            break
        if (last_seq is None and event.seq != 0) or (
            last_seq is not None and event.seq != last_seq + 1
        ):
            divergence = _divergence(
                event,
                expected="contiguous sequence",
                observed=event.seq,
                actions=reducer.actions,
                states=reducer.state_snapshots,
                scores=reducer.score_snapshots,
            )
            break
        if last_seq is not None and event.time_ns < stream[event.seq - 1].time_ns:
            divergence = _divergence(
                event,
                expected="monotonic time_ns",
                observed=event.time_ns,
                actions=reducer.actions,
                states=reducer.state_snapshots,
                scores=reducer.score_snapshots,
            )
            break
        last_seq = event.seq
        if event.kind in _OUTPUT_KINDS:
            continue
        if (
            game is not None
            and hooks_ready
            and event.kind
            in {
                "clock_read",
                "progress",
                "worker_complete",
                "result_visible",
                "worker_error",
                "invalidate",
                "pause",
                "resume",
                "cancel",
            }
        ):
            try:
                game.observe(event.kind, event.time_ns, _plain(event.data))
            except Exception as exc:
                divergence = _divergence(
                    event,
                    expected="accepted exogenous observation",
                    observed=f"adapter error: {type(exc).__name__}",
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
        if event.kind == "submit_attempt" and game is not None and hooks_ready:
            try:
                expected_submission = _plain(game.submission(event.time_ns))
            except Exception as exc:
                divergence = _divergence(
                    event,
                    expected="independent submission",
                    observed=f"adapter error: {type(exc).__name__}",
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
            if expected_submission != _plain(event.data):
                divergence = _divergence(
                    event,
                    expected=expected_submission,
                    observed=event.data,
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
        if event.kind == "validation_complete" and game is not None and hooks_ready:
            request_id = event.data.get("request_id")
            pending = reducer.requests.get(request_id)
            payload = (
                pending.get("result_payload")
                if pending is not None
                else reducer.results.get(request_id, {}).get("payload")
            )
            if not isinstance(payload, Mapping):
                divergence = _divergence(
                    event,
                    expected="visible payload before validation",
                    observed=payload,
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
            try:
                expected_valid = bool(game.validate(payload))
            except Exception as exc:
                divergence = _divergence(
                    event,
                    expected="independent validation",
                    observed=f"adapter error: {type(exc).__name__}",
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
            if expected_valid != bool(event.data.get("valid")):
                divergence = _divergence(
                    event,
                    expected=expected_valid,
                    observed=event.data.get("valid"),
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
        before = len(reducer.outputs)
        reducer.handle(event)
        produced = reducer.outputs[before:]
        expected_outputs = expected_by_trigger.pop(event.seq, [])
        if produced != expected_outputs:
            divergence = _divergence(
                event,
                expected=expected_outputs,
                observed=produced,
                actions=reducer.actions,
                states=reducer.state_snapshots,
                scores=reducer.score_snapshots,
            )
            break
        for output in produced:
            if output["kind"] == "decision":
                pending_actions.append(output["data"]["action"])
        if event.kind == "poll" and game is not None:
            observed_context = _plain(game.context())
            expected_context = _plain(event.data.get("context", {}))
            if (
                observed_context != expected_context
                or game.bare() != event.data.get("bare")
                or game.mask() != event.data.get("mask")
            ):
                divergence = _divergence(
                    event,
                    expected={
                        "context": expected_context,
                        "bare": event.data.get("bare"),
                        "mask": event.data.get("mask"),
                    },
                    observed={
                        "context": observed_context,
                        "bare": game.bare(),
                        "mask": game.mask(),
                    },
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
        if event.kind == "game_state" and game is not None:
            if not pending_actions:
                divergence = _divergence(
                    event,
                    expected="decision before game_state",
                    observed="none",
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
            action = pending_actions.pop(0)
            observed = game.step(action)
            expected = _plain(event.data.get("snapshot", event.data))
            fields_are_typed = (
                isinstance(observed.get("state_digest"), str)
                and bool(observed.get("state_digest"))
                and all(
                    isinstance(observed.get(key), int) and not isinstance(observed.get(key), bool)
                    for key in ("mass_sum", "alive_frames", "deaths")
                )
            )
            if not fields_are_typed or _plain(observed) != expected:
                divergence = _divergence(
                    event,
                    expected=expected,
                    observed=observed,
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
                break
            reducer.state_snapshots.append(observed.get("state_digest", observed))
            reducer.score_snapshots.append(
                {key: observed.get(key) for key in ("mass_sum", "alive_frames", "deaths")}
            )
    if divergence is None and expected_by_trigger:
        divergence = _divergence(
            None,
            expected="no untriggered internal output",
            observed=expected_by_trigger,
            actions=reducer.actions,
            states=reducer.state_snapshots,
            scores=reducer.score_snapshots,
        )
    if divergence is None and game is not None and hooks_ready:
        try:
            expectations = _plain(game.finish_expectations())
        except Exception as exc:
            divergence = _divergence(
                None,
                expected="independent finish expectations",
                observed=f"adapter error: {type(exc).__name__}",
                actions=reducer.actions,
                states=reducer.state_snapshots,
                scores=reducer.score_snapshots,
            )
        else:
            submissions = [_plain(event.data) for event in stream if event.kind == "submit_attempt"]
            dispatches = [
                output["data"]["request_id"]
                for output in reducer.outputs
                if output["kind"] == "dispatch"
            ]
            if (
                not isinstance(expectations, Mapping)
                or expectations.get("submissions") != submissions
                or expectations.get("dispatches") != dispatches
            ):
                divergence = _divergence(
                    None,
                    expected=expectations,
                    observed={"submissions": submissions, "dispatches": dispatches},
                    actions=reducer.actions,
                    states=reducer.state_snapshots,
                    scores=reducer.score_snapshots,
                )
    if divergence is None and reducer.invalid:
        divergence = _divergence(
            None,
            expected="valid availability trace",
            observed=reducer.structural_error or "recorded fault",
            actions=reducer.actions,
            states=reducer.state_snapshots,
            scores=reducer.score_snapshots,
        )
    if (
        divergence is None
        and game is not None
        and hooks_ready
        and (not reducer.actions or pending_actions)
    ):
        divergence = _divergence(
            None,
            expected="complete independent game hooks and one game_state per decision",
            observed={
                "hooks_ready": hooks_ready,
                "decisions": len(reducer.actions),
                "unstepped": len(pending_actions),
            },
            actions=reducer.actions,
            states=reducer.state_snapshots,
            scores=reducer.score_snapshots,
        )
    scheduler_verdict = "PASS" if divergence is None and not reducer.invalid else "INVALID_IDENTITY"
    overall = (
        "INVALID_IDENTITY"
        if scheduler_verdict == "INVALID_IDENTITY"
        else ("PASS" if game is not None and hooks_ready else "INCOMPLETE")
    )
    initial_state = header.get("initial_state", {})
    if not isinstance(initial_state, Mapping):
        initial_state = {}
    state_events = [event.data for event in stream if event.kind == "game_state"]
    digest_schema = "t34-shadow-full-digest/v3"
    full_v3 = (
        initial_state.get("state_digest_schema") == digest_schema
        and bool(state_events)
        and all(event.get("state_digest_schema") == digest_schema for event in state_events)
    )
    contract = initial_state.get("replay_contract", {})
    if not isinstance(contract, Mapping):
        contract = {}

    def is_sha(value: Any) -> bool:
        return (
            isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")
        )

    qualified = (
        overall == "PASS"
        and initial_state.get("synthetic") is False
        and {"game_state", "opponent_state", "rng_state", "source_snapshot"}.issubset(initial_state)
        and all(
            initial_state[key]
            for key in ("game_state", "opponent_state", "rng_state", "source_snapshot")
        )
        and full_v3
        and contract.get("independent_scheduler") is True
        and is_sha(contract.get("replay_reference_sha256"))
        and is_sha(binding_sha256)
    )
    return {
        "schema": REPLAY_SCHEMA,
        "verdict": overall,
        "scheduler_verdict": scheduler_verdict,
        "action_hash": _digest(reducer.actions),
        "state_hash": _digest(reducer.state_snapshots),
        "score_hash": _digest(reducer.score_snapshots),
        "first_divergence": divergence,
        "candidate_sha256": candidate_sha256,
        "binding_sha256": binding_sha256,
        "qualification": "REPLAY_QUALIFIED" if qualified else "PRECHECK_ONLY",
        "counts": {
            "events": len(stream),
            "expected_outputs": len(reducer.outputs)
            + sum(len(outputs) for outputs in expected_by_trigger.values()),
            "outputs": len(reducer.outputs),
            "decisions": len(reducer.actions),
        },
    }


def replay_journal(
    directory: Any, game_factory: Optional[Callable[[Mapping[str, Any]], ReplayGame]] = None
) -> Dict[str, Any]:
    """Read a sealed journal before replaying it; missing evidence is incomplete."""
    try:
        header, events, seal = read_journal(directory)
    except JournalError as exc:
        message = str(exc)
        corrupt = any(word in message for word in ("mismatch", "broken", "out-of-order", "altered"))
        return {
            "schema": REPLAY_SCHEMA,
            "verdict": "INVALID_IDENTITY" if corrupt else "INCOMPLETE",
            "scheduler_verdict": "INVALID_IDENTITY" if corrupt else "INCOMPLETE",
            "action_hash": _digest([]),
            "state_hash": _digest([]),
            "score_hash": _digest([]),
            "first_divergence": {"event": "journal", "observed": message} if corrupt else None,
            "evidence_error": message,
            "candidate_sha256": None,
            "binding_sha256": None,
            "qualification": "PRECHECK_ONLY",
            "counts": {"events": 0, "expected_outputs": 0, "outputs": 0, "decisions": 0},
        }
    try:
        report = replay_events(header, events, game_factory)
    except Exception as exc:
        return {
            "schema": REPLAY_SCHEMA,
            "verdict": "INVALID_IDENTITY",
            "scheduler_verdict": "INVALID_IDENTITY",
            "candidate_sha256": evidence_sha256(header["identity"]),
            "binding_sha256": header.get("binding_sha256"),
            "qualification": "PRECHECK_ONLY",
            "first_divergence": {
                "event": "replay_adapter",
                "error": type(exc).__name__,
                "observed": str(exc),
            },
            "counts": {"events": len(events), "expected_outputs": 0, "outputs": 0, "decisions": 0},
            "seal": seal,
        }
    report["seal"] = seal
    return report
