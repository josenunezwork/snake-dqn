"""Durable JSONL telemetry for the B8 serving-release observation window.

The records deliberately go to stderr through a dedicated logger, so the normal
server supervisor log is the durable source consumed by ``monitor_b8_release``.
Every registry resolution is recorded. Gameplay samples cover the released B8
default and explicit b8-s3 selection; rollback sessions do not contribute B8 samples.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Mapping

LOG_PREFIX = "b8-release-monitor:"
logger = logging.getLogger(__name__)


def _ensure_visible() -> None:
    """Make release records visible with web/serve.py's warning-level Uvicorn."""
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    # ``hasHandlers`` also sees an ancestor handler.  This logger intentionally
    # does not propagate (to keep JSONL records unmodified), so it needs its own
    # stderr handler even when Uvicorn configured the root logger.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def emit(event: str, **fields: Any) -> None:
    """Write one parseable, timestamped release-monitoring event to stderr."""
    _ensure_visible()
    payload = {"event": event, "timestamp": time.time(), **fields}
    logger.info("%s %s", LOG_PREFIX, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def served_resolution(choice: Mapping[str, object]) -> None:
    """Record a served resolution, including any fail-closed fallback reason."""
    emit(
        "resolution",
        requested=choice.get("requested"),
        name=choice.get("name"),
        sha256=choice.get("sha256"),
        fallback_reason=choice.get("reason"),
    )


def mixed_roster_fallback(*, reason: str) -> None:
    """Record a released-default mixed Watch refusal before all-B8 fallback."""
    emit("mixed_roster_fallback", reason=reason)


def watch_tick(
    *,
    checkpoint: str,
    roster_mode: str,
    roster_size: int,
    speed: float,
    decisions: int,
    deaths: int,
    update_ms: float,
    latency_eligible: bool,
) -> None:
    """Record one B8 Watch hero update at the default server speed.

    ``decisions`` comes from the actual hero ``update`` call after the respawn
    phase. ``deaths`` is one when the hero appears in
    GameState.frame_death_causes after the update.
    The measured interval is exactly GameState.update, matching T12B's update
    distribution rather than websocket or browser timing.
    """
    emit(
        "watch_tick",
        checkpoint=checkpoint,
        roster_mode=str(roster_mode),
        roster_size=int(roster_size),
        speed=float(speed),
        decisions=int(decisions),
        deaths=int(deaths),
        update_ms=float(update_ms),
        latency_eligible=bool(latency_eligible),
    )


def play_tick(
    *, checkpoint: str, driven_ai: int, speed: float, update_ms: float, latency_eligible: bool
) -> None:
    """Record one B8 Play update for the separately qualified Play latency path."""
    emit(
        "play_tick",
        checkpoint=checkpoint,
        driven_ai=int(driven_ai),
        speed=float(speed),
        update_ms=float(update_ms),
        latency_eligible=bool(latency_eligible),
    )


def ego2s_exception(*, checkpoint: str, mode: str, error: BaseException) -> None:
    """Record an exception from the live ego2s update path before re-raising it."""
    emit(
        "ego2s_exception",
        checkpoint=checkpoint,
        mode=mode,
        error_type=type(error).__name__,
        error=str(error),
    )
