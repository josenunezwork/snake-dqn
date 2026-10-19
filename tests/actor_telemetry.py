"""Shared contract for the M3-C actor's Q0b accounting fields.

``actor_main`` messages gained per-message accounting after the frozen git baselines that
the M6/T4 bitwise-parity tests replay (``836646c`` and ``f3375b26``).  These fields are
additive telemetry: they are excluded from legacy payload parity and validated here instead.
``produced_monotonic`` is a ``time.perf_counter()`` stamp, so it never compares across runs.
"""

from __future__ import annotations

import numbers

POST_BASELINE_ACTOR_TELEMETRY = frozenset(
    {
        "truncation_rows_dropped",
        "pending_nstep_rows",
        "world_frames",
        "pool_decisions",
        "long_body_decisions",
        "produced_monotonic",
    }
)

# Fields that depend on wall-clock time, not on the simulated trajectory.
WALL_CLOCK_ACTOR_FIELDS = frozenset({"env_s", "timing", "produced_monotonic"})

_COUNTS = POST_BASELINE_ACTOR_TELEMETRY - {"produced_monotonic"}


def assert_post_baseline_actor_telemetry(msgs, ctx) -> None:
    """Every message carries the accounting fields: non-negative integer counts and a
    non-decreasing monotonic production stamp."""
    last = float("-inf")
    for f, msg in enumerate(msgs):
        assert POST_BASELINE_ACTOR_TELEMETRY <= msg.keys(), (ctx, f)
        for key in _COUNTS:
            value = msg[key]
            assert isinstance(value, numbers.Integral) and value >= 0, (ctx, f, key, value)
        stamp = msg["produced_monotonic"]
        assert isinstance(stamp, float) and stamp >= last, (ctx, f, stamp, last)
        last = stamp
    assert any(msg["world_frames"] > 0 for msg in msgs), ctx


def without_post_baseline_actor_telemetry(msg) -> dict:
    return {k: v for k, v in msg.items() if k not in POST_BASELINE_ACTOR_TELEMETRY}
