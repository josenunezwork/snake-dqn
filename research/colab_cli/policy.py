"""Transport budgets derived from acceptance runs 3--8, not job/spend caps.

Every caller must narrow these allowances to its existing operation/session
budget. A latency allowance never authorizes a replacement assignment or job.
"""

from __future__ import annotations

import re
import subprocess

READ_SECONDS = 120.0
RPC_SECONDS = 120.0
# Successful colab exec calls complete within <=43s (p99 28s across 13.7k calls);
# cap read-only status/renewal polls at 60s while keeping mutating RPCs at 120s.
POLL_RPC_SECONDS = 60.0
# Foreground staging controls must leave headroom for immutable payload PUTs.
# These are attempt/recovery limits, not replacements for the 150/300s bars.
STAGING_CONTROL_SECONDS = 30.0
# Source fetch has its own 120s subprocess budget plus identity/checkout work.
# init 10 + config 5 + fetch 120 + rev-parse 10 + checkout 10 = 155s.
# Keep ordinary staging controls short; source alone gets 165s, with no grace.
SOURCE_CONTROL_SECONDS = 165.0
# Allow at least two full retries after a hung 165s source-fetch exec
# (3 * 165s = 495s plus independent idle probes and 2s/5s backoff).
SOURCE_RECOVERY_SECONDS = 540.0
# A 30s exec stall followed by a 30s idle-helper stall must leave room for
# fresh idle proof and a retry. All work still shares the 300s staging cap.
STAGING_RECOVERY_SECONDS = 120.0
STAGING_HEARTBEAT_SECONDS = 60.0
# One full exec timeout must leave room for independent idle proof and a read
# retry. Contents heartbeats continue under the unchanged runtime lease.
TELEMETRY_SECONDS = 300.0
# Owner 2026-10-29: base 10 minutes, effective 20 minutes (standing 2x rule).
# Only healthy owned assignments receive this consecutive-failure window.
EXEC_POLL_RECOVERY_SECONDS = 1200.0
EXEC_POLL_BACKOFF_SECONDS = (2.0, 5.0, 10.0, 20.0, 30.0)
ALLOCATION_SECONDS = 180.0
# Runtime allocation is not readiness of its Contents and kernel transports.
# This separate startup allowance still borrows from the whole-session caps.
WARMUP_SECONDS = 300.0
WARMUP_PROBE_SECONDS = 30.0
WARMUP_INTERVAL_SECONDS = 5.0
ATTEMPTS = 3
BACKOFF_SECONDS = (2.0, 5.0)
# ~8.5% of colab exec calls hang; two 120s hangs in one 300s export window occur
# on ~0.7-1% of multi-chunk periodic downloads. Deferring up to 3 consecutive
# failed cycles (refusing on attempt 4, <= 4 * 300s = 1200s == EXEC_POLL_RECOVERY_SECONDS)
# reduces i.i.d. false aborts below 1e-7 while still catching dead links within 20 min.
MAX_PERIODIC_EXPORT_DEFERRALS = 3
MAX_PERIODIC_EXPORT_ATTEMPTS = MAX_PERIODIC_EXPORT_DEFERRALS + 1
READ_COMMANDS = frozenset({"version", "whoami", "sessions", "usage", "url"})


READ_ONLY_POLL_TRANSIENT_MARKERS = (
    "websocketconnectionclosed",
    "connection is already closed",
    "socket is already closed",
    "connectionclosederror",
    "connectionclosedok",
    "connection closed",
)

_HTTP_4XX_RE = re.compile(
    r"(?:"
    r"\bhttp(?:/\d(?:\.\d)?)?\s+4\d\d\b"
    r"|\bhttp(?:status)?\s*error\b[^\n]{0,48}\b4\d\d\b"
    r"|\b4\d\d\s+client\s+error\b"
    r"|\bclient\s+error\b[^\n]{0,24}\b4\d\d\b"
    r"|\b(?:status(?:[_\s]+code)?|response|returned\s+error)\s*[=:]?\s*4\d\d\b"
    r"|(?<![:\w])4\d\d(?:\s*[:-])?\s+(?:"
    r"bad request|unauthorized|payment required|forbidden|not found|method not allowed"
    r"|not acceptable|proxy authentication|request timeout|conflict|gone|length required"
    r"|precondition|payload too large|content too large|uri too long|unsupported media"
    r"|range not satisfiable|expectation failed|misdirected|unprocessable|locked"
    r"|failed dependency|too early|upgrade required|too many requests"
    r"|access denied|permission denied"
    r")\b"
    r")",
    re.IGNORECASE,
)


def has_http_4xx(text: str) -> bool:
    """Return whether diagnostic text reports an HTTP 4xx client error."""
    return bool(_HTTP_4XX_RE.search(text))


def _normalized_error_message(exc: BaseException) -> str:
    message = str(exc).lower()
    # Rich's CLI traceback includes adjacent source lines. A permanent guard
    # refusal can therefore display the reconnect-required raise (or timeout
    # text) without actually raising it. Classify the terminal guard exception,
    # preserving genuine saved-kernel reconnect recovery after idle proof.
    guard_errors = re.findall(r"(?:^|\n)(?:cli_guard: )?guarderror: ([^\n]+)", message)
    if guard_errors:
        return guard_errors[-1]
    return message


def transient_error(exc: BaseException) -> bool:
    """Retry transport outages only; auth, quota, hashes and kernel errors fail."""
    if isinstance(exc, (ConnectionError, TimeoutError, subprocess.TimeoutExpired)):
        return True
    message = _normalized_error_message(exc)
    if has_http_4xx(message):
        return False
    return bool(
        re.search(r"\b(?:500|502|503|504)\b", message)
        or any(
            marker in message
            for marker in (
                "connection reset",
                "connection aborted",
                "connection refused",
                "connection error",
                "connectionerror",
                "nameresolutionerror",
                "failed to resolve",
                "remotedisconnected",
                "network is unreachable",
                "temporarily unavailable",
                # Creation was acknowledged and saved before websocket setup.
                # Only independent idle proof may clear exec_uncertain.
                "tracked kernel reconnect required",
                "timed out",
                "timeout",
            )
        )
    )


def transient_poll_error(exc: BaseException, *, mutating: bool = False) -> bool:
    """Classify transient colab exec failures, adding WebSocket closes on read-only polls."""
    if transient_error(exc):
        return True
    if mutating:
        return False
    message = _normalized_error_message(exc)
    return any(marker in message for marker in READ_ONLY_POLL_TRANSIENT_MARKERS)
