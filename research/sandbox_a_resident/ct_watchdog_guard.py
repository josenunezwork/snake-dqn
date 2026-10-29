"""Independent hard-stop process; never inspect the worker's Python frames.

Private pipe protocol: newline-terminated absolute monotonic deadlines; EOF
disarms. Only StallWatchdog owns the writer. No filesystem or traceback work is
allowed between deadline expiry and SIGKILL. Run as a standalone stdlib script.
"""

from __future__ import annotations

import math
import os
import select
import signal
import sys
import time


def guard(parent_pid: int) -> None:
    """Kill only our still-living parent when its completed-work lease expires."""
    # Mechanical preemption targets the whole process group. Remain in that
    # group for final SIGTERM/SIGKILL cleanup, but protect progress rearming
    # while the worker handles its graceful SIGUSR1 fence. Publish readiness
    # only after both dispositions are installed. EOF still disarms on death.
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    deadline: float | None = None
    pending = b""
    os.write(1, b"R")
    while True:
        wait = None if deadline is None else max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([0], [], [], wait)
        if readable:
            chunk = os.read(0, 4096)
            if not chunk:
                return
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                value = float(line)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid watchdog deadline")
                deadline = value
            if len(pending) > 64:
                raise ValueError("oversize watchdog command")
        elif deadline is not None and time.monotonic() >= deadline:
            # Parent identity + the private pipe avoid killing a recycled PID.
            # No logging, lock, dump, or Python in the target is required here.
            if os.getppid() == parent_pid:
                os.kill(parent_pid, signal.SIGKILL)
            return


if __name__ == "__main__":
    guard(int(sys.argv[1]))
