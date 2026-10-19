"""Transport budgets derived from acceptance runs 3--5, not job/spend caps.

Every caller must narrow these allowances to its existing operation/session
budget. A latency allowance never authorizes a replacement assignment or job.
"""

from __future__ import annotations

import re
import subprocess

READ_SECONDS = 120.0
RPC_SECONDS = 120.0
SOURCE_RPC_SECONDS = 150.0
ALLOCATION_SECONDS = 180.0
ATTEMPTS = 3
BACKOFF_SECONDS = (2.0, 5.0)
READ_COMMANDS = frozenset({"version", "whoami", "sessions", "usage", "url"})


def transient_error(exc: BaseException) -> bool:
    """Retry transport outages only; auth, quota, hashes and kernel errors fail."""
    if isinstance(exc, (ConnectionError, TimeoutError, subprocess.TimeoutExpired)):
        return True
    message = str(exc).lower()
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
                "timed out",
                "timeout",
            )
        )
    )
