"""Nonblocking D500 controller for Amendment 2 Path B.

The adapter supplies immutable captured snapshots after a real step. The caller
only records compact identities and submits opaque objects: every engine
mutation, clone, digest, confirmation, advancement, and search is worker-only.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections import deque
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple

MAX_RECENT_DECISIONS = 256
MAX_PENDING_IDENTITIES = 501


@dataclass(frozen=True)
class SessionIdentity:
    """Full fence for one session, hero life, checkpoint, and frame."""

    session_id: str
    frame: int
    life: int
    checkpoint: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.checkpoint:
            raise ValueError("session_id and checkpoint must be non-empty")
        if self.frame < 0 or self.life < 0:
            raise ValueError("frame and life must be non-negative")

    def continuity_key(self) -> Tuple[str, int, str]:
        return self.session_id, self.life, self.checkpoint


@dataclass(frozen=True)
class _Binding:
    identity: SessionIdentity
    digest: Optional[str]
    shadow_id: Optional[int] = None
    epoch: int = 0


@dataclass(frozen=True)
class _ScheduledAction:
    binding: _Binding
    target_frame: int
    action: int
    intervention_kind: str


@dataclass(frozen=True)
class _WorkerResult:
    value: Any
    cpu_seconds: float


class PathBControllerV2:
    """One worker, one job, and one future action with mandatory bare fallback."""

    def __init__(
        self,
        engine: Any,
        *,
        executor: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
        max_recent_decisions: int = MAX_RECENT_DECISIONS,
    ) -> None:
        if max_recent_decisions < 1:
            raise ValueError("max_recent_decisions must be positive")
        if int(getattr(engine, "depth", 500)) != 500:
            raise ValueError("Path B v2 requires a D500 shadow engine")
        self._engine = engine
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="t34-pathb-v2"
        )
        self._clock = clock
        self._future: Optional[Any] = None
        self._job_kind: Optional[str] = None
        self._pending: Optional[_Binding] = None
        self._last_confirmed: Optional[_Binding] = None
        self._scheduled: Optional[_ScheduledAction] = None
        self._confirmations: Deque[Tuple[int, int, str]] = deque(maxlen=MAX_PENDING_IDENTITIES)
        self._latest_snapshot: Any = None
        self._latest_binding: Optional[_Binding] = None
        self._confirmation_overflow = False
        self._expected_digests: Dict[int, str] = {}
        self._epoch = 0
        self._progress_lock = Lock()
        self._latest_post_frame = -1
        self._force_refork = False
        self._last_identity: Optional[SessionIdentity] = None
        self._submitted_wall: Optional[float] = None
        self._job_sequence = 0
        self._pending_sequence: Optional[int] = None
        self._worker_cpu_seconds = 0.0
        self._shadow_cpu_seconds = 0.0
        self._search_cpu_seconds = 0.0
        self._first_result_latency_records: Deque[Dict[str, float | int]] = deque(
            maxlen=MAX_RECENT_DECISIONS
        )
        self._unconditional_faults: Deque[Dict[str, Any]] = deque(maxlen=32)
        self._closed = False
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=max_recent_decisions)
        self._last: Optional[Dict[str, Any]] = None
        self._counts: Dict[str, int] = {
            "decisions": 0,
            "requested": 0,
            "maintenance_submitted": 0,
            "maintenance_dropped_busy": 0,
            "applied": 0,
            "accepted": 0,
            "stale_discarded": 0,
            "late_discarded": 0,
            "deadline_miss": 0,
            "fallback": 0,
            "worker_errors": 0,
            "invalid_results": 0,
            "unconditional_faults": 0,
        }
        self._fallback_reasons: Dict[str, int] = {}

    @property
    def busy(self) -> bool:
        return self._future is not None

    @property
    def scheduled_frame(self) -> Optional[int]:
        return None if self._scheduled is None else self._scheduled.target_frame

    def decide(
        self, identity: SessionIdentity, bare_action: int, action_mask: Sequence[bool]
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose immediately; this action-boundary method never waits."""
        report = self._new_report(identity, bare_action)
        self._collect_finished(identity, report)
        scheduled = self._scheduled
        if scheduled is None:
            return self._finish(bare_action, report, "bare_no_scheduled_action")
        if identity.continuity_key() != scheduled.binding.identity.continuity_key():
            self._scheduled = None
            report["proposed_action"] = scheduled.action
            report["stale_discarded"] = True
            return self._finish(bare_action, report, "stale_scheduled_action")
        if scheduled.target_frame < identity.frame:
            self._scheduled = None
            report["proposed_action"] = scheduled.action
            report["late_discarded"] = True
            report["deadline_miss"] = True
            return self._finish(bare_action, report, "late_scheduled_action")
        if scheduled.target_frame > identity.frame:
            report["proposed_action"] = scheduled.action
            return self._finish(bare_action, report, "scheduled_future_action")
        self._scheduled = None
        report["proposed_action"] = scheduled.action
        if not self._schedule_confirmed_for_target(identity, scheduled):
            report["stale_discarded"] = True
            return self._finish(bare_action, report, "unconfirmed_scheduled_action")
        if not self._legal_action(scheduled.action, action_mask):
            return self._finish(bare_action, report, "invalid_scheduled_action")
        report["accepted"] = True
        report["applied"] = scheduled.action != bare_action
        report["intervention_kind"] = scheduled.intervention_kind
        if report["applied"]:
            self._fence("intervention_executed")
        return self._finish(scheduled.action, report, None)

    def after_step(
        self, live_snapshot: Any, identity: SessionIdentity, *, digest: Optional[str] = None
    ) -> None:
        """Submit opaque maintenance or log a bounded dropped-maintenance event.

        The snapshot must already be adapter-created and immutable. This method
        does not call the engine, clone, hash, or wait, so catch-up can never
        delay the next game tick.
        """
        with self._progress_lock:
            self._latest_post_frame = max(self._latest_post_frame, identity.frame)
        if self._closed:
            return
        if not isinstance(digest, str) or not digest:
            self._record_event(identity, "missing_confirmation_digest")
            return
        if (
            self._last_identity is not None
            and identity.continuity_key() != self._last_identity.continuity_key()
        ):
            self._fence("continuity_changed")
        self._last_identity = identity
        expected = self._expected_digests.get(identity.frame)
        if expected is not None and expected != digest:
            self._fence("predicted_digest_mismatch")
        elif expected is not None and self._last_confirmed is not None:
            self._last_confirmed = _Binding(
                identity, digest, self._last_confirmed.shadow_id, self._last_confirmed.epoch
            )
            self._expected_digests.pop(identity.frame, None)
        binding = self._binding(identity, digest)
        if live_snapshot is not None:
            self._latest_snapshot, self._latest_binding = live_snapshot, binding
        if len(self._confirmations) == self._confirmations.maxlen:
            self._confirmation_overflow = True
            self._fence("confirmation_overflow")
            binding = self._binding(identity, digest)
        self._confirmations.append((identity.frame, identity.life, digest))
        self._collect_finished(identity, None)
        if self._future is not None:
            self._counts["maintenance_dropped_busy"] += 1
            self._record_event(identity, "maintenance_dropped_busy")
            return
        if live_snapshot is None:
            # A prior captured source is not a safe substitute after the
            # worker became free: let the adapter capture the next real state.
            self._record_event(identity, "maintenance_deferred_no_snapshot")
            return
        if self._latest_snapshot is None or self._latest_binding is None:
            self._record_event(identity, "maintenance_deferred_no_snapshot")
            return
        self._submit_maintenance(self._latest_snapshot, self._latest_binding)

    def state(self) -> Dict[str, Any]:
        return {
            "active": not self._closed,
            "label": "assisted hero",
            "busy": self.busy,
            "scheduled_frame": self.scheduled_frame,
            "counts": dict(self._counts),
            "fallback_reasons": dict(self._fallback_reasons),
            "last_confirmed": self._identity_dict(self._last_confirmed),
            "pending_confirmations": len(self._confirmations),
            "confirmation_overflow": self._confirmation_overflow,
            "expected_digest_count": len(self._expected_digests),
            "worker_cpu_seconds": self._worker_cpu_seconds,
            "shadow_cpu_seconds": self._shadow_cpu_seconds,
            "search_cpu_seconds": self._search_cpu_seconds,
            "first_result_latency_records": list(self._first_result_latency_records),
            "unconditional_faults": list(self._unconditional_faults),
            "last": self._last,
            "recent": list(self._recent),
        }

    def settle_between_ticks(
        self, identity: SessionIdentity, check_cap: Optional[Callable[[], None]] = None
    ) -> bool:
        """Drain one worker only between unpaced descriptive-harness ticks.

        This method is deliberately public for the unpaced descriptive arm.
        Paced code must use ``decide`` and ``after_step`` only.  It advances no
        world state and never rewinds: it only waits for an already submitted
        maintenance/search job, allowing a completed maintenance job to submit
        and drain its one chained search request.
        """
        if self._closed:
            return True
        while self._future is not None:
            self._run_cap_check(check_cap)
            fault_count = self._counts["unconditional_faults"]
            try:
                self._future.result(timeout=0.25)
            except concurrent.futures.TimeoutError:
                self._run_cap_check(check_cap)
                continue
            except Exception:
                # The future is done even when it raised.  Reuse the normal
                # collector so typed P7 faults reach state()/the harness.
                self._collect_finished(identity, None)
                self._run_cap_check(check_cap)
                if self._closed or self._counts["unconditional_faults"] > fault_count:
                    return False
                continue
            self._collect_finished(identity, None)
            self._run_cap_check(check_cap)
        return True

    def close(self, wait: bool = False) -> None:
        if self._closed:
            if wait:
                self._executor.shutdown(wait=True, cancel_futures=False)
            return
        self._closed = True
        self._scheduled = None
        cancel = getattr(self._engine, "cancel", None)
        if callable(cancel):
            cancel()
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _run_cap_check(self, check_cap: Optional[Callable[[], None]]) -> None:
        if check_cap is None:
            return
        try:
            check_cap()
        except Exception:
            self.close(wait=False)
            raise

    def _submit_maintenance(self, live_snapshot: Any, binding: _Binding) -> None:
        try:
            confirmations = tuple(self._confirmations)
            request = self._engine.make_maintenance_request(
                live_snapshot,
                frame=binding.identity.frame,
                life=binding.identity.life,
                checkpoint=binding.identity.checkpoint,
                refork=self._last_confirmed is None
                or self._confirmation_overflow
                or self._force_refork,
                confirmations=confirmations,
            )
            self._future = self._executor.submit(self._run_stage, self._engine.maintain, request)
            self._job_kind, self._pending = "maintenance", binding
            self._submitted_wall = self._clock()
            self._job_sequence += 1
            self._pending_sequence = self._job_sequence
            self._counts["maintenance_submitted"] += 1
            self._confirmations.clear()
            self._confirmation_overflow = False
            self._force_refork = False
        except Exception as exc:
            self._future = self._job_kind = self._pending = None
            self._record_event(binding.identity, "maintenance_submit_%s" % type(exc).__name__)

    def _submit_search(self, request: Any, binding: _Binding) -> None:
        try:
            try:
                request = replace(
                    request, latest_progress=self._live_progress, source_epoch=binding.epoch
                )
            except TypeError:
                # Tiny fake requests in focused controller tests are mutable;
                # the real frozen engine request takes the ``replace`` path.
                request.latest_progress = self._live_progress
                request.source_epoch = binding.epoch
            self._future = self._executor.submit(self._run_stage, self._engine.run_search, request)
            self._job_kind, self._pending = "search", binding
            self._submitted_wall = self._clock()
            self._job_sequence += 1
            self._pending_sequence = self._job_sequence
            self._counts["requested"] += 1
        except Exception as exc:
            self._future = self._job_kind = self._pending = None
            self._record_event(binding.identity, "search_submit_%s" % type(exc).__name__)

    def _collect_finished(self, current: SessionIdentity, report: Optional[Dict[str, Any]]) -> None:
        if self._future is None or self._pending is None or not self._future.done():
            return
        future, kind, binding, sequence = (
            self._future,
            self._job_kind,
            self._pending,
            self._pending_sequence,
        )
        self._future = self._job_kind = self._pending = self._pending_sequence = None
        try:
            raw_result = future.result()
        except Exception as exc:
            if self._is_nonfinite_q(exc):
                self._record_unconditional_fault("nonfinite_q", binding.identity, report)
                return
            self._counts["worker_errors"] += 1
            self._mark(report, "worker_error", "worker_%s" % type(exc).__name__, binding.identity)
            return
        if isinstance(raw_result, _WorkerResult):
            result = raw_result.value
            self._worker_cpu_seconds += raw_result.cpu_seconds
            if kind == "maintenance":
                self._shadow_cpu_seconds += raw_result.cpu_seconds
            elif kind == "search":
                self._search_cpu_seconds += raw_result.cpu_seconds
            if report is not None:
                report["worker_cpu_seconds"] = raw_result.cpu_seconds
            if kind == "search" and self._submitted_wall is not None:
                latency_seconds = self._clock() - self._submitted_wall
                self._first_result_latency_records.append(
                    {
                        "sequence": -1 if sequence is None else sequence,
                        "milliseconds": latency_seconds * 1000.0,
                    }
                )
                if report is not None:
                    report["first_result_wall_seconds"] = latency_seconds
        else:
            result = raw_result
        if kind == "maintenance":
            self._consume_maintenance(result, binding, current, report)
        elif kind == "search":
            self._consume_search(result, binding, current, report)
        else:
            self._mark(report, "worker_error", "unknown_worker_job", binding.identity)

    def _consume_maintenance(
        self,
        result: Any,
        binding: _Binding,
        current: SessionIdentity,
        report: Optional[Dict[str, Any]],
    ) -> None:
        matches = (
            binding.epoch == self._epoch
            and getattr(result, "source_frame", None) == binding.identity.frame
            and getattr(result, "source_life", None) == binding.identity.life
            and getattr(result, "source_checkpoint", None) == binding.identity.checkpoint
            and (binding.digest is None or getattr(result, "source_digest", None) == binding.digest)
            and current.continuity_key() == binding.identity.continuity_key()
            and bool(getattr(result, "confirmed", False))
            and isinstance(getattr(result, "shadow_id", None), int)
        )
        if not matches:
            self._mark(report, "stale_discarded", "stale_maintenance_result", binding.identity)
            return
        source_binding = _Binding(
            binding.identity,
            getattr(result, "source_digest", None),
            getattr(result, "shadow_id"),
            binding.epoch,
        )
        self._last_confirmed = source_binding
        expected = getattr(result, "expected_digests", ())
        if isinstance(expected, tuple):
            self._expected_digests = {int(frame): str(digest) for frame, digest in expected}
            if not self._reconcile_confirmations(current):
                return
        request = getattr(result, "search_request", None)
        if request is not None and self._scheduled is None:
            if (
                getattr(request, "shadow_id", None) == source_binding.shadow_id
                and getattr(request, "source_digest", None) == source_binding.digest
            ):
                self._submit_search(request, source_binding)

    def _consume_search(
        self,
        result: Any,
        binding: _Binding,
        current: SessionIdentity,
        report: Optional[Dict[str, Any]],
    ) -> None:
        status = getattr(result, "status", None)
        target, action = getattr(result, "target_frame", None), getattr(result, "action", None)
        if status == "LATE" or (isinstance(target, int) and target <= current.frame):
            if report is not None:
                report["deadline_miss"] = True
            self._mark(report, "late_discarded", "late_search_result", binding.identity)
            return
        if status == "INVALID":
            self._counts["invalid_results"] += 1
            self._mark(report, "worker_error", "invalid_search_result", binding.identity)
            return
        if status == "FAULT":
            marker = getattr(result, "unconditional_fault", None)
            if marker == "nonfinite_q":
                self._record_unconditional_fault(marker, binding.identity, report)
                return
            self._counts["invalid_results"] += 1
            self._mark(report, "worker_error", "unknown_search_fault", binding.identity)
            return
        if status == "NO_IMPROVEMENT":
            valid_no_improvement = (
                binding.epoch == self._epoch
                and getattr(result, "shadow_id", None) == binding.shadow_id
                and getattr(result, "source_digest", None) == binding.digest
                and current.continuity_key() == binding.identity.continuity_key()
                and self._last_confirmed is not None
                and self._last_confirmed.shadow_id == binding.shadow_id
                and self._last_confirmed.epoch == self._epoch
            )
            if valid_no_improvement:
                if report is not None:
                    report["fallback_reason"] = "no_improving_action"
                return
        fresh = (
            binding.epoch == self._epoch
            and status == "IMPROVING"
            and bool(getattr(result, "completed", False))
            and isinstance(target, int)
            and isinstance(action, int)
            and getattr(result, "shadow_id", None) == binding.shadow_id
            and getattr(result, "source_digest", None) == binding.digest
            and current.continuity_key() == binding.identity.continuity_key()
            and self._last_confirmed is not None
            and self._last_confirmed.shadow_id == binding.shadow_id
            and self._last_confirmed.epoch == self._epoch
        )
        if not fresh:
            self._mark(
                report, "stale_discarded", "stale_or_nonimproving_search_result", binding.identity
            )
            return
        kind = getattr(result, "intervention_kind", "none")
        if kind not in {"rescue", "delay", "none"}:
            self._counts["invalid_results"] += 1
            self._mark(report, "worker_error", "invalid_intervention_kind", binding.identity)
            return
        self._scheduled = _ScheduledAction(binding, target, action, kind)
        if report is not None:
            report.update(
                proposed_action=action,
                search_published=True,
                scheduled_frame=target,
                intervention_kind=kind,
            )

    def _schedule_confirmed_for_target(
        self, identity: SessionIdentity, scheduled: _ScheduledAction
    ) -> bool:
        confirmed = self._last_confirmed
        return bool(
            confirmed is not None
            and confirmed.shadow_id == scheduled.binding.shadow_id
            and confirmed.epoch == self._epoch
            and confirmed.identity.frame >= identity.frame
        )

    def _reconcile_confirmations(self, current: SessionIdentity) -> bool:
        """Apply all buffered post-step digests covered by a new worker map."""
        if self._last_confirmed is None:
            return False
        for frame, life, digest in self._confirmations:
            expected = self._expected_digests.get(frame)
            if expected is None:
                continue
            if expected != digest:
                self._fence("predicted_digest_mismatch")
                return False
            self._last_confirmed = _Binding(
                SessionIdentity(current.session_id, frame, life, current.checkpoint),
                digest,
                self._last_confirmed.shadow_id,
                self._last_confirmed.epoch,
            )
            self._expected_digests.pop(frame, None)
        return True

    @staticmethod
    def _legal_action(action: Any, mask: Sequence[bool]) -> bool:
        return (
            isinstance(action, int)
            and not isinstance(action, bool)
            and 0 <= action < len(mask)
            and bool(mask[action])
        )

    @staticmethod
    def _new_report(identity: SessionIdentity, bare_action: int) -> Dict[str, Any]:
        return {
            "session_id": identity.session_id,
            "frame": identity.frame,
            "life": identity.life,
            "checkpoint": identity.checkpoint,
            "proposed_action": None,
            "executed_action": bare_action,
            "accepted": False,
            "applied": False,
            "stale_discarded": False,
            "late_discarded": False,
            "deadline_miss": False,
            "fallback_reason": None,
        }

    @staticmethod
    def _run_stage(function: Callable[[Any], Any], request: Any) -> _WorkerResult:
        started = time.thread_time()
        return _WorkerResult(function(request), time.thread_time() - started)

    def _fence(self, reason: str) -> None:
        """Fence outputs immediately; a running worker is never cancelled unsafely."""
        self._scheduled = None
        with self._progress_lock:
            self._epoch += 1
        self._last_confirmed = None
        self._expected_digests.clear()
        self._force_refork = True
        self._record_event(self._last_identity or SessionIdentity("fenced", 0, 0, "fenced"), reason)

    @staticmethod
    def _is_nonfinite_q(exc: Exception) -> bool:
        """Recognize the engine's typed P7 unconditional-fault boundary."""
        try:
            from research.t34_pathb.engine_v2 import NonFiniteQ

            return isinstance(exc, NonFiniteQ)
        except ImportError:
            return type(exc).__name__ == "NonFiniteQ"

    def _record_unconditional_fault(
        self, marker: str, identity: SessionIdentity, report: Optional[Dict[str, Any]]
    ) -> None:
        record = {
            "fault": marker,
            "marker": marker,
            "session_id": identity.session_id,
            "frame": identity.frame,
            "life": identity.life,
            "checkpoint": identity.checkpoint,
        }
        self._unconditional_faults.append(record)
        self._counts["unconditional_faults"] += 1
        if report is not None:
            report["unconditional_fault"] = marker
            report["fallback_reason"] = "unconditional_fault_%s" % marker
        else:
            self._last = {**self._new_report(identity, -1), **record}

    def _binding(self, identity: SessionIdentity, digest: Optional[str]) -> _Binding:
        with self._progress_lock:
            return _Binding(identity, digest, None, self._epoch)

    def _live_progress(self) -> Tuple[int, int]:
        """Return scalar progress only; safe for the worker to read repeatedly."""
        with self._progress_lock:
            return self._latest_post_frame, self._epoch

    def _mark(
        self, report: Optional[Dict[str, Any]], flag: str, reason: str, identity: SessionIdentity
    ) -> None:
        if report is not None:
            report[flag] = True
            report["fallback_reason"] = reason
        else:
            self._record_event(identity, reason)

    def _finish(
        self, action: int, report: Dict[str, Any], fallback: Optional[str]
    ) -> Tuple[int, Dict[str, Any]]:
        report["executed_action"] = action
        if report["fallback_reason"] is None:
            report["fallback_reason"] = fallback
        self._counts["decisions"] += 1
        for flag in ("accepted", "applied", "stale_discarded", "late_discarded", "deadline_miss"):
            if report[flag]:
                self._counts[flag] += 1
        if report["fallback_reason"] is not None:
            self._counts["fallback"] += 1
            reason = report["fallback_reason"]
            self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
        self._last = report
        self._recent.append(report)
        return action, report

    def _record_event(self, identity: SessionIdentity, reason: str) -> None:
        event = self._new_report(identity, -1)
        event.update(event=reason, fallback_reason=reason)
        self._last = event
        self._recent.append(event)
        self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1

    @staticmethod
    def _identity_dict(binding: Optional[_Binding]) -> Optional[Dict[str, Any]]:
        if binding is None:
            return None
        return {
            "session_id": binding.identity.session_id,
            "frame": binding.identity.frame,
            "life": binding.identity.life,
            "checkpoint": binding.identity.checkpoint,
            "shadow_id": binding.shadow_id,
            "digest": binding.digest,
        }
