"""Serving-only startup GC policy and counters-only pause accounting.

Authority: round-25/JUDGMENT.md, sha256
 df3c55403da147195dd0508c0f5185e955f20d939ae4f7853e1d2ba2f14d20e7.
Only the serving lifespan (or an explicit qualification process) applies it;
importing this module never changes GC in training or pool workers.
"""

from __future__ import annotations

import gc
import os
import platform
import sys
import time

from web.backend import release_monitoring

POLICY_ENV = "SNAKE_SERVE_GC_POLICY"
DEFAULT_POLICY = "default"
_policy = DEFAULT_POLICY
_applied = False


class GcTickMeter:
    """Count completed collections without retaining event objects in the callback."""

    __slots__ = (
        "start0",
        "start1",
        "start2",
        "count0",
        "count1",
        "count2",
        "pause0",
        "pause1",
        "pause2",
        "h0_0",
        "h0_1",
        "h0_2",
        "h0_3",
        "h0_4",
        "h1_0",
        "h1_1",
        "h1_2",
        "h1_3",
        "h1_4",
        "h2_0",
        "h2_1",
        "h2_2",
        "h2_3",
        "h2_4",
        "hook",
    )

    def __init__(self) -> None:
        self.start0 = 0
        self.start1 = 0
        self.start2 = 0
        self.count0 = 0
        self.count1 = 0
        self.count2 = 0
        self.pause0 = 0
        self.pause1 = 0
        self.pause2 = 0
        self.h0_0 = 0
        self.h0_1 = 0
        self.h0_2 = 0
        self.h0_3 = 0
        self.h0_4 = 0
        self.h1_0 = 0
        self.h1_1 = 0
        self.h1_2 = 0
        self.h1_3 = 0
        self.h1_4 = 0
        self.h2_0 = 0
        self.h2_1 = 0
        self.h2_2 = 0
        self.h2_3 = 0
        self.h2_4 = 0
        self.hook = self.callback

    def install(self) -> None:
        """Register this meter once (qualification may use a separate meter)."""
        if self.hook not in gc.callbacks:
            gc.callbacks.append(self.hook)

    def close(self) -> None:
        """Remove only this meter's callback."""
        if self.hook in gc.callbacks:
            gc.callbacks.remove(self.hook)

    def callback(self, phase: str, info: dict) -> None:
        """Store only integers on an existing slotted instance; no event containers."""
        generation = info["generation"]
        if phase == "start":
            if generation == 0:
                self.start0 = time.monotonic_ns()
            elif generation == 1:
                self.start1 = time.monotonic_ns()
            elif generation == 2:
                self.start2 = time.monotonic_ns()
        elif phase == "stop":
            if generation == 0:
                elapsed = time.monotonic_ns() - self.start0
                self.pause0 += elapsed
                self.count0 += 1
                if elapsed < 1000000:
                    self.h0_0 += 1
                elif elapsed < 5000000:
                    self.h0_1 += 1
                elif elapsed < 10000000:
                    self.h0_2 += 1
                elif elapsed < 20000000:
                    self.h0_3 += 1
                else:
                    self.h0_4 += 1
            elif generation == 1:
                elapsed = time.monotonic_ns() - self.start1
                self.pause1 += elapsed
                self.count1 += 1
                if elapsed < 1000000:
                    self.h1_0 += 1
                elif elapsed < 5000000:
                    self.h1_1 += 1
                elif elapsed < 10000000:
                    self.h1_2 += 1
                elif elapsed < 20000000:
                    self.h1_3 += 1
                else:
                    self.h1_4 += 1
            elif generation == 2:
                elapsed = time.monotonic_ns() - self.start2
                self.pause2 += elapsed
                self.count2 += 1
                if elapsed < 1000000:
                    self.h2_0 += 1
                elif elapsed < 5000000:
                    self.h2_1 += 1
                elif elapsed < 10000000:
                    self.h2_2 += 1
                elif elapsed < 20000000:
                    self.h2_3 += 1
                else:
                    self.h2_4 += 1

    def drain(self) -> dict:
        """Return completed collections since the previous drain, then reset."""
        c0, c1, c2 = self.count0, self.count1, self.count2
        p0, p1, p2 = self.pause0, self.pause1, self.pause2
        h0_0 = self.h0_0
        h0_1 = self.h0_1
        h0_2 = self.h0_2
        h0_3 = self.h0_3
        h0_4 = self.h0_4
        h1_0 = self.h1_0
        h1_1 = self.h1_1
        h1_2 = self.h1_2
        h1_3 = self.h1_3
        h1_4 = self.h1_4
        h2_0 = self.h2_0
        h2_1 = self.h2_1
        h2_2 = self.h2_2
        h2_3 = self.h2_3
        h2_4 = self.h2_4
        self.h0_0 = 0
        self.h0_1 = 0
        self.h0_2 = 0
        self.h0_3 = 0
        self.h0_4 = 0
        self.h1_0 = 0
        self.h1_1 = 0
        self.h1_2 = 0
        self.h1_3 = 0
        self.h1_4 = 0
        self.h2_0 = 0
        self.h2_1 = 0
        self.h2_2 = 0
        self.h2_3 = 0
        self.h2_4 = 0
        self.count0 = self.count1 = self.count2 = 0
        self.pause0 = self.pause1 = self.pause2 = 0
        histograms = {
            "gen0": {
                "0-1": h0_0,
                "1-5": h0_1,
                "5-10": h0_2,
                "10-20": h0_3,
                "20+": h0_4,
            },
            "gen1": {
                "0-1": h1_0,
                "1-5": h1_1,
                "5-10": h1_2,
                "10-20": h1_3,
                "20+": h1_4,
            },
            "gen2": {
                "0-1": h2_0,
                "1-5": h2_1,
                "5-10": h2_2,
                "10-20": h2_3,
                "20+": h2_4,
            },
        }
        return {
            "gc_pause_ms": (p0 + p1 + p2) / 1e6,
            "gc_histograms": histograms,
            "gc_gen0_count": c0,
            "gc_gen1_count": c1,
            "gc_gen2_count": c2,
            "gc_gen0_pause_ms": p0 / 1e6,
            "gc_gen1_pause_ms": p1 / 1e6,
            "gc_gen2_pause_ms": p2 / 1e6,
            **state(),
        }


meter = GcTickMeter()


def state() -> dict:
    """Expose the applied policy, never an un-applied environment request."""
    return {"gc_policy": _policy, "gc_freeze_count": gc.get_freeze_count()}


def drain_tick() -> dict:
    """Drain the process meter for the five serving telemetry fields."""
    values = meter.drain()
    return {
        key: values[key]
        for key in (
            "gc_pause_ms",
            "gc_gen2_count",
            "gc_gen2_pause_ms",
            "gc_freeze_count",
            "gc_policy",
        )
    }


def apply_startup_policy(*, stage: str) -> dict:
    """Apply exactly once, before constructing any live session; emit its receipt."""
    global _applied, _policy
    if _applied:
        raise RuntimeError("GC startup policy already applied in this process")
    policy = os.environ.get(POLICY_ENV, DEFAULT_POLICY)
    if policy not in ("default", "freeze_startup"):
        raise ValueError(f"invalid {POLICY_ENV}: {policy!r}")
    if not stage:
        raise ValueError("startup stage must be nonempty")
    _applied = True
    _policy = policy
    thresholds = gc.get_threshold()
    tracked_before = len(gc.get_objects())
    collected = None
    if policy == "freeze_startup":
        collected = gc.collect(2)
        gc.freeze()
        assert gc.get_freeze_count() > 0 and gc.isenabled()
    assert gc.get_threshold() == thresholds
    receipt = {
        "policy": policy,
        "interpreter": sys.version,
        "build": platform.python_build(),
        "thresholds": thresholds,
        "enabled": gc.isenabled(),
        "tracked_before": tracked_before,
        "frozen_count": gc.get_freeze_count(),
        "tracked_after": len(gc.get_objects()),
        "collect2_result": collected,
        "stage": stage,
        "monotonic_ns": time.monotonic_ns(),
    }
    meter.install()
    release_monitoring.emit("gc_policy", **receipt)
    return receipt
