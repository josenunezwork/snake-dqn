"""Transport budgets derived from acceptance runs 3--8, not job/spend caps.

Every caller must narrow these allowances to its existing operation/session
budget. A latency allowance never authorizes a replacement assignment or job.
"""

from __future__ import annotations

import random
import re
import subprocess

READ_SECONDS = 120.0
RPC_SECONDS = 30.0
REQUEST_SECONDS = 120.0
# Short exec attempts leave room for independent idle proof and retries.
# Contents heartbeats continue under the unchanged runtime lease.
TELEMETRY_SECONDS = 300.0
ALLOCATION_SECONDS = 180.0
# Runtime allocation is not readiness of its Contents and kernel transports.
# This separate startup allowance still borrows from the whole-session caps.
WARMUP_SECONDS = 300.0
WARMUP_PROBE_SECONDS = 30.0
WARMUP_INTERVAL_SECONDS = 5.0
MAX_REQUEST_POLLS = 300
# l4-label-faults/v2 (JUDGMENT_L4_FAULT_DEFINITION_20261024 §4): after an
# uncertain exec, at most four bounded idle-proof/re-issue attempts under the
# same request ID, all inside a 150-second per-request recovery ceiling that is
# itself narrowed by the operation, session and priced caps (first cap wins).
RECOVERY_ATTEMPTS = 4
RECOVERY_SECONDS = 150.0
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


def recovery_probe_retryable(exc: BaseException) -> bool:
    """An idle-proof GET that failed by transport or by exhausting its own bound.

    Its helper budget exhaustion is a slow read, not proof of state, so another
    bounded probe may follow inside the recovery ceiling. Auth, quota, malformed
    state and kernel errors still fail immediately.
    """
    if transient_error(exc):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("read helper operation budget exhausted", "timed out", "timeoutexpired")
    )


def poll_delay(attempt: int) -> float:
    """Exponential backoff with jitter; capped independently of session budgets."""
    return random.uniform(0.75, 1.25) * min(8.0, 2.0 ** min(attempt, 3))
