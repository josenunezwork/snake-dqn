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
_TELEMETRY_ERROR_INTERVAL_S = 60.0
_last_telemetry_error_ts = 0.0


def _log_telemetry_error(exc: BaseException) -> None:
    """Rate-limit internal telemetry failures so they never flood or raise."""
    global _last_telemetry_error_ts
    try:
        now = time.monotonic()
        if (
            _last_telemetry_error_ts
            and (now - _last_telemetry_error_ts) < _TELEMETRY_ERROR_INTERVAL_S
        ):
            return
        _last_telemetry_error_ts = now
        logging.getLogger("web.backend.served_checkpoint").warning(
            "%s suppressed telemetry error: %s", LOG_PREFIX, exc
        )
    except Exception:
        pass


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
    try:
        _ensure_visible()
        payload = {"event": event, "timestamp": time.time(), **fields}
        logger.info("%s %s", LOG_PREFIX, json.dumps(payload, sort_keys=True, separators=(",", ":")))
    except Exception as exc:
        _log_telemetry_error(exc)


def served_resolution(choice: Mapping[str, object]) -> None:
    """Record a served resolution, including any fail-closed fallback reason."""
    try:
        emit(
            "resolution",
            requested=choice.get("requested"),
            name=choice.get("name"),
            sha256=choice.get("sha256"),
            fallback_reason=choice.get("reason"),
        )
    except Exception as exc:
        _log_telemetry_error(exc)


def mixed_roster_fallback(*, reason: str) -> None:
    """Record a released-default mixed Watch refusal before all-B8 fallback."""
    try:
        emit("mixed_roster_fallback", reason=reason)
    except Exception as exc:
        _log_telemetry_error(exc)


def play_mixed_roster_fallback(*, reason: str) -> None:
    """Record an opt-in mixed Play refusal before all-B8 fallback."""
    try:
        emit("play_mixed_roster_fallback", reason=reason)
    except Exception as exc:
        _log_telemetry_error(exc)


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
    try:
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
    except Exception as exc:
        _log_telemetry_error(exc)


def play_tick(
    *, checkpoint: str, driven_ai: int, speed: float, update_ms: float, latency_eligible: bool
) -> None:
    """Record one B8 Play update for the separately qualified Play latency path."""
    try:
        emit(
            "play_tick",
            checkpoint=checkpoint,
            driven_ai=int(driven_ai),
            speed=float(speed),
            update_ms=float(update_ms),
            latency_eligible=bool(latency_eligible),
        )
    except Exception as exc:
        _log_telemetry_error(exc)


def ego2s_exception(*, checkpoint: str, mode: str, error: BaseException) -> None:
    """Record an exception from the live ego2s update path before re-raising it."""
    try:
        emit(
            "ego2s_exception",
            checkpoint=checkpoint,
            mode=mode,
            error_type=type(error).__name__,
            error=str(error),
        )
    except Exception as exc:
        _log_telemetry_error(exc)


def watch_replay_active(
    *,
    checkpoint: str,
    roster_mode: str,
    roster_size: int,
    controller: str | None,
    speed: float,
    status: str = "active",
) -> None:
    """Record Watch replay selection or pending state before the first summary flush."""
    try:
        emit(
            "watch_replay_active",
            checkpoint=str(checkpoint),
            roster_mode=str(roster_mode),
            roster_size=int(roster_size),
            controller=None if controller is None else str(controller),
            speed=float(speed),
            status=str(status),
        )
    except Exception as exc:
        _log_telemetry_error(exc)


def watch_replay_summary(
    *,
    checkpoint: str,
    roster_mode: str,
    roster_size: int,
    controller: str,
    speed: float,
    frames: int,
    update_p95_ms: float,
    max_update_ms: float,
    mean_update_ms: float = 0.0,
    first_timestamp: float | None = None,
) -> None:
    """Record aggregated Watch replay playback telemetry without per-frame logging."""
    try:
        fields: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "roster_mode": str(roster_mode),
            "roster_size": int(roster_size),
            "controller": str(controller),
            "speed": float(speed),
            "frames": int(frames),
            "update_p95_ms": round(float(update_p95_ms), 4),
            "max_update_ms": round(float(max_update_ms), 4),
            "mean_update_ms": round(float(mean_update_ms), 4),
        }
        if first_timestamp is not None:
            fields["first_timestamp"] = float(first_timestamp)
        emit("watch_replay_summary", **fields)
    except Exception as exc:
        _log_telemetry_error(exc)


def watch_replay_failure(
    *,
    checkpoint: str,
    roster_mode: str,
    roster_size: int,
    controller: str | None,
    stage: str,
    error: BaseException | str,
) -> None:
    """Record a Watch replay step/snapshot failure that fell back to live B8."""
    try:
        error_type = type(error).__name__ if isinstance(error, BaseException) else "ReplayError"
        emit(
            "replay_step_failure",
            checkpoint=str(checkpoint),
            roster_mode=str(roster_mode),
            roster_size=int(roster_size),
            controller=None if controller is None else str(controller),
            stage=str(stage),
            error_type=error_type,
            reason=str(error),
        )
    except Exception as exc:
        _log_telemetry_error(exc)


def watch_replay_bank_exhausted(
    *,
    checkpoint: str,
    roster_mode: str,
    roster_size: int,
    reason: str,
) -> None:
    """Record Watch replay bank exhaustion or unavailability falling back to live B8."""
    try:
        emit(
            "replay_bank_exhausted",
            checkpoint=str(checkpoint),
            roster_mode=str(roster_mode),
            roster_size=int(roster_size),
            reason=str(reason),
        )
    except Exception as exc:
        _log_telemetry_error(exc)
