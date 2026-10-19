"""Pinned public-input validation for the Colab CLI transport.

The broker verifies every public object on the Mac before an allocation.  The
runtime repeats the same size and digest check before it starts the trainer.
"""

from __future__ import annotations

import hashlib
import math
import re
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

PUBLIC_INPUT_FETCH_ATTEMPTS = 3
PUBLIC_INPUT_FETCH_TIMEOUT_SECONDS = 30
# This is a shared cap for fetching all public inputs in a runtime.  It is
# deliberately separate from the ContentsClient upload staging cap.
PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS = 300
READ_BYTES = 1024 * 1024


class InputError(RuntimeError):
    """A public input could not be fetched as its pinned identity."""


@contextmanager
def _hard_timeout(seconds: float):
    """Interrupt a trickling socket read; urllib's timeout is inactivity-only."""
    if (
        not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        raise InputError("public input preflight requires a main-thread interval timer")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise InputError("public input preflight cannot replace an active interval timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum: int, _frame: Any) -> None:
        raise InputError("public input preflight total timeout")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def validate_public_input(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized public-input row or reject its unsafe shape."""
    url = raw.get("url")
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    if not isinstance(url, str) or not re.fullmatch(r"https://[^\s]+", url):
        raise InputError("public input url must be an HTTPS URL")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise InputError("public input sha256 must be lowercase SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise InputError("public input size_bytes must be a nonnegative integer")
    return {"url": url, "sha256": digest, "size_bytes": size}


def preflight(
    rows: Iterable[dict[str, Any]],
    *,
    opener: Callable[..., Any] = urlopen,
    timeout_seconds: float = PUBLIC_INPUT_FETCH_TIMEOUT_SECONDS,
    total_timeout_seconds: float = PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Download and verify each distinct public input before allocation.

    The bytes are streamed only to calculate the pinned identity, so this does
    not leave an unverified copy behind on the Mac.  Results are suitable for
    an admission receipt and include measured transfer timing.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise InputError("public input request timeout must be finite and positive")
    if not math.isfinite(total_timeout_seconds) or total_timeout_seconds <= 0:
        raise InputError("public input total timeout must be finite and positive")
    deadline = time.monotonic() + total_timeout_seconds
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    with _hard_timeout(total_timeout_seconds):
        for raw in rows:
            row = validate_public_input(raw)
            identity = (row["url"], row["sha256"], row["size_bytes"])
            if identity in seen:
                continue
            seen.add(identity)
            last_error: BaseException | None = None
            for attempt in range(1, PUBLIC_INPUT_FETCH_ATTEMPTS + 1):
                if time.monotonic() >= deadline:
                    raise InputError("public input preflight total timeout") from last_error
                started = time.monotonic()
                count = 0
                digest = hashlib.sha256()
                try:
                    request = Request(row["url"], headers={"Accept-Encoding": "identity"})
                    with opener(
                        request, timeout=min(timeout_seconds, max(0.1, deadline - started))
                    ) as reply:
                        while True:
                            if time.monotonic() >= deadline:
                                raise InputError("public input preflight total timeout")
                            chunk = reply.read(READ_BYTES)
                            if not chunk:
                                break
                            count += len(chunk)
                            if count > row["size_bytes"]:
                                raise InputError("public input exceeded pinned size")
                            digest.update(chunk)
                    if time.monotonic() >= deadline:
                        raise InputError("public input preflight total timeout")
                    elapsed = time.monotonic() - started
                    if count != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                        raise InputError("public input size or SHA-256 mismatch")
                    results.append(
                        {
                            **row,
                            "attempt": attempt,
                            "elapsed_seconds": elapsed,
                            "bytes_per_second": count / max(elapsed, 1e-9),
                        }
                    )
                    break
                except (InputError, OSError, URLError) as exc:
                    last_error = exc
                    if isinstance(exc, InputError) and "mismatch" in str(exc):
                        raise InputError(f"public input preflight failed: {exc}") from exc
            else:
                raise InputError(f"public input preflight failed: {last_error}") from last_error
    return results
