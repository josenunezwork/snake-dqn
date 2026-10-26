"""Outcome-free parent supervision of the one F4 worker, including finalization."""

from __future__ import annotations

import math
import os
import select
import signal
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from research.bl_ppo.exports import export_json


def kill_receipt_path(output: Path) -> Path:
    """Keep watchdog evidence outside the immutable aggregate-only result bundle."""
    return output.with_name(output.name + ".watchdog-kill.json")


def supervise(
    worker: Callable[[Callable[[], None]], None],
    *,
    kill_receipt: Path,
    seconds: float = 3600.0,
    on_kill: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Fork before native imports; ACK deadline arming before allowing a claim.

    The parent never runs inference or parses outcomes. Its wait loop and SIGKILL
    do not depend on worker Python callbacks. The unreaped child PID cannot be
    reused, and only that exact owned child is killed. No completion retry exists.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("watchdog ceiling must be finite and positive")
    if any(Path(str(kill_receipt) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError("watchdog kill evidence already exists; no retry authorized")
    parent, child = socket.socketpair()
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        pid = os.fork()
    except BaseException:
        parent.close()
        child.close()
        raise
    if pid == 0:
        parent.close()
        armed = False

        def arm() -> None:
            nonlocal armed
            if armed:
                raise RuntimeError("watchdog may only be armed once")
            child.sendall(b"A")
            if child.recv(1) != b"K":
                raise RuntimeError("independent watchdog did not acknowledge arming")
            armed = True

        code = 1
        try:
            worker(arm)
            if not armed:
                raise RuntimeError("worker exited without arming its watchdog")
            code = 0
        except BaseException:
            # Preserve the worker's sealed failure; never print exception data.
            pass
        finally:
            try:
                child.close()
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(code)

    child.close()
    started: float | None = None
    reaped = False
    old_handlers = {}

    def interrupted(signum: int, _frame: object) -> None:
        raise RuntimeError(f"F4 supervisor interrupted by signal {signum}")

    def kill_and_record(reason: str) -> None:
        nonlocal reaped
        # A repeated terminal signal must not interrupt kill/reap/fsync evidence.
        saved = {
            signum: signal.signal(signum, signal.SIG_IGN)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            # Kill first: durable disk IO must never delay enforcement of the ceiling.
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            reaped = True
            record = {
                "schema": "bl-residual-f4-postmortem-watchdog-kill/v1",
                "purpose": "DEVELOPMENT_ONLY",
                "f4_disposition": "STOP_FUNDING",
                "worker_pid": pid,
                "signal": "SIGKILL",
                "wait_status": status,
                "reason": reason,
                "ceiling_seconds": seconds,
                "elapsed_seconds": None if started is None else time.monotonic() - started,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "retry_authorized": False,
            }
            export_json(kill_receipt, record)
            if on_kill is not None:
                on_kill(record)
        finally:
            for signum, handler in saved.items():
                signal.signal(signum, handler)

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.signal(signum, interrupted)
        while True:
            # Test the ceiling before accepting an exit: finalization completing
            # after the ceiling is never accepted as successful.
            remaining = None if started is None else seconds - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                kill_and_record("hard_ceiling")
                raise RuntimeError("F4 watchdog killed worker at ceiling; no retry authorized")
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                reaped = True
                if status != 0 or started is None:
                    raise RuntimeError("F4 worker incomplete; no retry authorized")
                return
            if started is None:
                readable, _, _ = select.select([parent], [], [], 0.02)
                if readable:
                    if parent.recv(1) != b"A":
                        raise RuntimeError("worker lost watchdog handshake")
                    started = time.monotonic()
                    parent.sendall(b"K")
            else:
                time.sleep(min(0.02, max(0.0, remaining)))
    finally:
        parent.close()
        try:
            if not reaped:
                kill_and_record("supervisor_interrupted")
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
