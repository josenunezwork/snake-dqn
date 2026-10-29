"""Bound owned subprocess trees, including nested admission launchers."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Any


def run(args: list[str], *, timeout: float, check: bool = False, **kwargs: Any) -> Any:
    """Run in an owned group; terminate descendants and reap on every interruption.

    Nested callers forward TERM while waiting, so each group owner cleans its
    own descendants before the outer owner escalates to KILL. Ordinary children
    inherit their owner's group. No unrelated process is selected by name.
    """
    capture = kwargs.pop("capture_output", False)
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = subprocess.Popen(args, start_new_session=True, **kwargs)
    previous = None
    install_handler = threading.current_thread() is threading.main_thread()

    def interrupted(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    def terminate() -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Kill surviving descendants even if the immediate child already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()

    if install_handler:
        previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        terminate()
        raise
    finally:
        if install_handler:
            signal.signal(signal.SIGTERM, previous)
    result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result
