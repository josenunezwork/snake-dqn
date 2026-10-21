"""Spawn-only, one-math-thread process transport with bounded EDF admission.

The transport has no Colab dependency and never drives a game clock. Collection
is nonblocking. Calls that start processes or drain journals belong off tick.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import time
from dataclasses import asdict
from typing import Any, Callable

from .controller import PacedController
from .events import Request, canonical_bytes

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _pin_worker() -> None:
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    # The parent is nice'd as well. Never increase a worker's priority.
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except ImportError:
        pass


def _compute(
    worker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    request: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    result = worker(request, payload)
    canonical_bytes(result)
    if not isinstance(result, dict):
        raise TypeError("worker result must be a JSON object")
    return result, time.monotonic_ns()


class ProcessTransport:
    """EDF queue with W spawned processes and K coordinator-owned identities."""

    def __init__(
        self,
        controller: PacedController,
        worker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        *,
        owner_multi_worker: bool = False,
    ) -> None:
        if os.uname().sysname != "Darwin":
            raise RuntimeError("served-search transport is Mac-only")
        controller.identity.require_authorized(owner_multi_worker=owner_multi_worker)
        for name in THREAD_ENV:
            os.environ[name] = "1"
        os.environ["SNAKE_DQN_DEVICE"] = "cpu"
        self.controller = controller
        self.worker = worker
        self.pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=controller.identity.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_pin_worker,
        )
        self.queued: dict[str, tuple[Request, dict[str, Any]]] = {}
        self.running: dict[str, concurrent.futures.Future[Any]] = {}
        self.closed = False

    def submit(self, request: Request, payload: dict[str, Any]) -> bool:
        """Journal admission; dispatch is explicitly off tick."""
        if self.closed:
            raise RuntimeError("transport closed")
        if not self.controller.submit(request, payload):
            return False
        import json

        self.queued[request.request_id] = (request, json.loads(canonical_bytes(payload)))
        return True

    def dispatch(self) -> None:
        """Start at most W unfinished jobs in absolute-cutoff order, off tick."""
        if self.closed:
            raise RuntimeError("transport closed")
        while self.queued and len(self.running) < self.controller.identity.workers:
            rid = min(
                self.queued,
                key=lambda key: (
                    self.queued[key][0].cutoff_ns,
                    self.queued[key][0].target_frame,
                    key,
                ),
            )
            request, payload = self.queued.pop(rid)
            state = self.controller.pending.get(rid)
            if state is None or state["fenced"]:
                self.controller.worker_error(rid, "cancelled_before_dispatch")
                continue
            if not self.controller.dispatch(rid):
                raise RuntimeError("EDF/W dispatch identity fault")
            try:
                self.running[rid] = self.pool.submit(
                    _compute, self.worker, asdict(request), payload
                )
            except Exception as exc:
                self.controller.worker_error(rid, type(exc).__name__)

    def collect(self) -> None:
        """Observe all available results without waiting, retaining every error."""
        for rid, future in list(self.running.items()):
            if not future.done():
                continue
            del self.running[rid]
            try:
                payload, completed_ns = future.result()
            except Exception as exc:
                self.controller.worker_error(rid, type(exc).__name__)
                continue
            self.controller.worker_complete(rid, completed_ns)
            self.controller.publish(rid, payload, completed_ns)

    def close(self) -> None:
        """Explicit off-tick join; never kills any unrelated process or session."""
        if self.closed:
            return
        self.closed = True
        for rid in self.queued:
            self.controller.cancel(rid)
            self.controller.worker_error(rid, "transport_closed_before_dispatch")
        self.queued.clear()
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.collect()
