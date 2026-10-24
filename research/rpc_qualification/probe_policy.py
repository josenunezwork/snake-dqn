"""Scoped lane policy hardening; the frozen transport subtree stays untouched."""

from __future__ import annotations

import contextlib
import re
import subprocess
from collections.abc import Iterator

from research.colab_cli import policy


def recovery_probe_retryable(exc: BaseException) -> bool:
    """Retry only a transient terminal error or the exact helper exhaustion."""
    if isinstance(exc, (ConnectionError, TimeoutError, subprocess.TimeoutExpired)):
        return True
    message = str(exc).lower()
    terminal = re.findall(r"(?:^|\n)(?:cli_guard: )?guarderror: ([^\n]+)", message)
    if terminal:
        message = terminal[-1]
    # Pass only the normalized terminal message to the frozen transient parser.
    # Never re-scan the discarded traceback context for timeout strings.
    return policy.transient_error(RuntimeError(message)) or (
        message == "read helper operation budget exhausted"
    )


@contextlib.contextmanager
def hardened_probe_policy() -> Iterator[None]:
    """Install the reviewed policy for one qualification run and always restore."""
    original = policy.recovery_probe_retryable
    policy.recovery_probe_retryable = recovery_probe_retryable
    try:
        yield
    finally:
        policy.recovery_probe_retryable = original
