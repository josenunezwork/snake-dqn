"""Transport budgets derived from acceptance runs 3--8, not job/spend caps.

Every caller must narrow these allowances to its existing operation/session
budget. A latency allowance never authorizes a replacement assignment or job.
"""

from __future__ import annotations

import re
import subprocess

READ_SECONDS = 120.0
RPC_SECONDS = 120.0
# One full exec timeout must leave room for independent idle proof and a read
# retry. Contents heartbeats continue under the unchanged runtime lease.
TELEMETRY_SECONDS = 300.0
ALLOCATION_SECONDS = 180.0
# Runtime allocation is not readiness of its Contents and kernel transports.
# This separate startup allowance still borrows from the whole-session caps.
WARMUP_SECONDS = 300.0
WARMUP_PROBE_SECONDS = 30.0
WARMUP_INTERVAL_SECONDS = 5.0
ATTEMPTS = 3
BACKOFF_SECONDS = (2.0, 5.0)
READ_COMMANDS = frozenset({"version", "whoami", "sessions", "usage", "url"})


def transient_error(exc: BaseException) -> bool:
    """Retry transport outages only; auth, quota, hashes and kernel errors fail."""
    if isinstance(exc, (ConnectionError, TimeoutError, subprocess.TimeoutExpired)):
        return True
    message = str(exc).lower()
    # Rich's CLI traceback includes adjacent source lines. A permanent guard
    # refusal can therefore display the reconnect-required raise (or timeout
    # text) without actually raising it. Classify the terminal guard exception,
    # preserving genuine saved-kernel reconnect recovery after idle proof.
    guard_errors = re.findall(r"(?:^|\n)(?:cli_guard: )?guarderror: ([^\n]+)", message)
    if guard_errors:
        message = guard_errors[-1]
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
