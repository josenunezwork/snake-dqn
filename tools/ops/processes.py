"""Bound owned subprocess trees, including nested admission launchers."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from typing import Any


def run(
    args: list[str],
    *,
    timeout: float,
    check: bool = False,
    descendant_groups: bool = True,
    **kwargs: Any,
) -> Any:
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

    def owned_groups() -> set[int]:
        """Snapshot descendants before TERM can orphan a nested group leader."""
        inventory = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,pgid="],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        rows = [
            tuple(map(int, row.split())) for row in inventory.stdout.splitlines() if row.strip()
        ]
        owned = {process.pid}
        while True:
            expanded = owned | {pid for pid, parent, _group in rows if parent in owned}
            if expanded == owned:
                break
            owned = expanded
        # Each newly-created session/group must have a leader in this owned
        # tree. Never signal the caller's group or an unrelated existing group.
        return {process.pid} | {
            group for pid, _parent, group in rows if pid in owned and group in owned
        }

    def signal_group(group: int, signum: int) -> None:
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can return EPERM for an already-exited orphan group. Do
            # not suppress an actual cleanup failure: independently refuse if
            # the owned PGID still contains any non-zombie process.
            inventory = subprocess.run(
                ["/bin/ps", "-axo", "pgid=,stat="],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            for row in inventory.stdout.splitlines():
                fields = row.split()
                if fields and int(fields[0]) == group and not fields[1].startswith("Z"):
                    raise

    def terminate() -> None:
        inventory_error = None
        try:
            groups = owned_groups() if descendant_groups else {process.pid}
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            groups = {process.pid}
            inventory_error = exc
        for group in groups:
            signal_group(group, signal.SIGTERM)
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Kill surviving descendants even if the immediate child already exited.
        # Reap every observed group even if an outer launcher was killed before
        # its nested timeout handler could escalate. Group ownership survives
        # reparenting because it was captured before the first signal.
        for group in groups:
            signal_group(group, signal.SIGKILL)
        process.communicate()
        if inventory_error is not None:
            raise subprocess.SubprocessError(
                "owned root group reaped; descendant inventory unavailable, cleanup unverified"
            ) from inventory_error

    if install_handler:
        previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            # A nested validator may have caught its own subprocess timeout
            # and exited nonzero, leaving descendants in this owned group.
            terminate()
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
