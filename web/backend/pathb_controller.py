"""Nonblocking, identity-bound controller for T34 Path B assistance.

The controller deliberately knows nothing about game state.  The serving hook owns
snapshot construction and action legality; this small boundary owns the one-request
deadline contract so it can be tested without a running web service.
"""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import os
import pickle
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple


MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_RECENT_DECISIONS = 256

Identity = Tuple[Any, ...]
Worker = Callable[[bytes], Dict[str, Any]]


def _set_math_thread_env() -> None:
    """Set inherited numerical-library caps before a spawned child imports them."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _pin_math_threads() -> None:
    """Keep the spawned worker within the registered one-math-thread budget."""
    _set_math_thread_env()
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        # Torch is optional for the controller unit tests and may already have
        # configured its inter-op pool in an embedding process.
        pass


def _run_worker(worker: Worker, payload: bytes) -> Dict[str, Any]:
    """Run in the child and stamp completion at the source of the result."""
    result = worker(payload)
    if not isinstance(result, dict):
        raise TypeError("Path B worker must return a dict")
    stamped = dict(result)
    stamped["_completion_monotonic"] = time.monotonic()
    if _encoded_size(stamped) > MAX_RESPONSE_BYTES:
        raise ValueError("Path B worker response exceeds 64 KiB")
    return stamped


def _encoded_size(value: Dict[str, Any]) -> int:
    """Bound the object that crosses the process IPC boundary."""
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except (pickle.PickleError, TypeError, ValueError):
        return MAX_RESPONSE_BYTES + 1


class DeadlineController:
    """A single spawned lookahead request that can never delay an action tick.

    ``submit`` accepts only one snapshot at a time.  ``poll`` never waits: it
    consumes a completed reply only for the exact caller identity and only before
    the snapshot's absolute deadline.  An overdue running future deliberately
    remains outstanding until it eventually completes and is collected.
    """

    def __init__(
        self,
        worker: Worker,
        clock: Callable[[], float] = time.monotonic,
        executor: Optional[Any] = None,
    ) -> None:
        # This runs before ``ProcessPoolExecutor`` creates the spawned child, so
        # BLAS/Numba see the limits during import as well as in its initializer.
        _set_math_thread_env()
        self._worker = worker
        self._clock = clock
        self._executor = executor or concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_pin_math_threads,
        )
        self._future: Optional[Any] = None
        self._identity: Optional[Identity] = None
        self._deadline: Optional[float] = None
        self._closed = False
        self._nonblocking_shutdown = False
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=MAX_RECENT_DECISIONS)
        self._counts: Dict[str, int] = {
            "requested": 0,
            "applied": 0,
            "stale_discarded": 0,
            "deadline_miss": 0,
            "fallback": 0,
        }
        self._fallback_reasons: Dict[str, int] = {}
        self._last: Optional[Dict[str, Any]] = None

    def submit(self, payload: bytes, identity: Identity, deadline: float) -> bool:
        """Submit an immutable snapshot, returning false for rejected requests."""
        if self._closed or self._future is not None or not isinstance(payload, bytes):
            return False
        if len(payload) > MAX_REQUEST_BYTES or not math.isfinite(deadline):
            return False
        self._future = self._executor.submit(_run_worker, self._worker, payload)
        self._identity = tuple(identity)
        self._deadline = deadline
        self._counts["requested"] += 1
        return True

    @property
    def busy(self) -> bool:
        """Whether the one permitted request is still outstanding."""
        return self._future is not None

    def poll(
        self,
        identity: Identity,
        bare: int,
        mask: Sequence[bool],
        validate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Return the bare or advisory action immediately, with an audit record."""
        requested = self._future is not None
        report = self._new_report(requested, bare)
        if self._future is None:
            return self._finish(bare, report, "no_request")
        if not self._future.done():
            if self._clock() >= self._deadline:  # type: ignore[operator]
                report["deadline_miss"] = True
                return self._finish(bare, report, "deadline_computing")
            return self._finish(bare, report, "pending")

        future, expected, deadline = self._future, self._identity, self._deadline
        self._clear_pending()
        if tuple(identity) != expected:
            report["stale_discarded"] = True
            return self._finish(bare, report, "identity_mismatch")
        try:
            result = future.result()
        except Exception:
            return self._finish(bare, report, "worker_error")
        if not isinstance(result, dict) or _encoded_size(result) > MAX_RESPONSE_BYTES:
            return self._finish(bare, report, "malformed_response")
        completed = result.get("_completion_monotonic")
        if not isinstance(completed, (int, float)) or not math.isfinite(completed):
            return self._finish(bare, report, "malformed_response")
        now = self._clock()
        if completed >= deadline or now >= deadline:  # type: ignore[operator]
            report["deadline_miss"] = True
            return self._finish(bare, report, "deadline_miss")

        proposed = result.get("proposed_action")
        result_bare = result.get("bare_action")
        result_mask = result.get("mask")
        report["proposed_action"] = proposed
        try:
            action_context_matches = result_bare == bare and list(result_mask) == list(mask)
        except TypeError:
            action_context_matches = False
        if not action_context_matches:
            report["stale_discarded"] = True
            return self._finish(bare, report, "action_context_mismatch")
        if validate is not None:
            try:
                valid_world = bool(validate(result))
            except Exception:
                valid_world = False
            if not valid_world:
                report["stale_discarded"] = True
                return self._finish(bare, report, "world_mismatch")
        if not self._legal_action(proposed, mask):
            return self._finish(bare, report, "invalid_action")
        report["accepted"] = True
        # ``applied`` measures an actual intervention.  A verified proposal that
        # keeps the bare action is useful evidence, but is not an intervention.
        report["applied"] = int(proposed) != bare
        return self._finish(int(proposed), report, None)

    def state(self) -> Dict[str, Any]:
        """Return JSON-safe state for the assisted Watch UI and inspector."""
        return {
            "active": self._future is not None,
            "label": "assisted hero",
            "counts": dict(self._counts),
            "fallback_reasons": dict(self._fallback_reasons),
            "last": self._last,
            "recent": list(self._recent),
        }

    def close(self, wait: bool = False) -> None:
        """Make subsequent submissions fail without blocking a game tick."""
        if self._closed and (not wait or not self._nonblocking_shutdown):
            return
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._nonblocking_shutdown = not wait
        elif wait:
            # The tick path returns promptly; an explicit harness close can
            # subsequently join that same already-shutting-down worker.
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._nonblocking_shutdown = False

    def record(self, log: Dict[str, Any]) -> None:
        """Add a synchronous diagnostic or snapshot failure to the same audit ring."""
        self._record(log)

    def _clear_pending(self) -> None:
        self._future = None
        self._identity = None
        self._deadline = None

    @staticmethod
    def _legal_action(action: Any, mask: Sequence[bool]) -> bool:
        return (
            isinstance(action, int)
            and not isinstance(action, bool)
            and 0 <= action < len(mask)
            and bool(mask[action])
        )

    @staticmethod
    def _new_report(requested: bool, bare: int) -> Dict[str, Any]:
        return {
            "requested": requested,
            "applied": False,
            "accepted": False,
            "stale_discarded": False,
            "deadline_miss": False,
            "fallback_reason": None,
            "proposed_action": None,
            "executed_action": bare,
        }

    def _finish(self, action: int, report: Dict[str, Any], fallback: Optional[str]) -> Tuple[int, Dict[str, Any]]:
        report["executed_action"] = action
        report["fallback_reason"] = fallback
        self._record(report)
        return action, report

    def _record(self, report: Dict[str, Any]) -> None:
        """Record by reference so serving may enrich the same log after ``poll``."""
        self._counts.setdefault("decisions", 0)
        self._counts["decisions"] += 1
        if report.get("applied"):
            self._counts["applied"] += 1
        if report.get("accepted"):
            self._counts.setdefault("accepted", 0)
            self._counts["accepted"] += 1
        if report.get("stale_discarded"):
            self._counts["stale_discarded"] += 1
        if report.get("deadline_miss"):
            self._counts["deadline_miss"] += 1
        fallback = report.get("fallback_reason")
        if fallback is not None:
            self._counts["fallback"] += 1
            self._fallback_reasons[fallback] = self._fallback_reasons.get(fallback, 0) + 1
        self._last = report
        self._recent.append(report)
