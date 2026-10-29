"""Minimal, lease-aware launcher for the factory nice interim.

This is intentionally separate from the unqualified full companion supervisor.
It has two routes only: low-priority light work and one reviewed, serial test
command.  The latter owns both the shared test-slot lock and the Mac lease for
its complete lifetime, so it cannot overlap a factory, timing, or calibration
window.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from research.compute.factory_companion_commands import (
    CommandValidationError,
    _light,
    validate_command,
)
from research.mac_lease import LeaseHeld, held

TEST_SLOT_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/factory-companions/test-slot.lock"
)
TEST_DEADLINE_SECONDS = 120.0
_CLEANUP_RESERVE_SECONDS = 6.0
_TERM_GRACE_SECONDS = 2.0
_KILL_GRACE_SECONDS = 2.0
TIMING_ARMED_PATH = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed")
_THREAD_ENV = {
    "SNAKE_DQN_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "OMP_THREAD_LIMIT": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTEST_ADDOPTS": "",
    "PYTEST_PLUGINS": "",
}
_LIGHT_EXECUTABLES = frozenset({"codex", "claude", "rg", "cat", "sed", "jq", "git", "bash"})


class InterimLaunchError(RuntimeError):
    """The small interim lane cannot safely admit the requested command."""


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


@contextmanager
def serial_test_slot(path: Path | None = None) -> Iterator[None]:
    """Acquire the host-wide test slot without waiting for another operation.

    Recovery host checks use this same context through their complete process
    observation and receipt publication.  A numerical test therefore cannot
    run while recovery is proving that tests have stopped.
    """
    path = TEST_SLOT_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InterimLaunchError("factory serial test slot is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _nice_floor() -> int:
    current = os.getpriority(os.PRIO_PROCESS, 0)
    if current < 15:
        os.setpriority(os.PRIO_PROCESS, 0, 15)
    observed = os.getpriority(os.PRIO_PROCESS, 0)
    if observed < 15:
        raise InterimLaunchError("could not establish nice >= 15")
    return observed


def _env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(_THREAD_ENV)
    return environment


def _reject_test_command(
    command: Sequence[str], *, review_manifest: str, review_sha256: str, worktree: Path
) -> None:
    """Require the existing source-pinned explicit-node review manifest."""
    try:
        validate_command("test", command, str(worktree), review_manifest, review_sha256)
    except CommandValidationError as exc:
        raise InterimLaunchError(f"test route is not a reviewed serial command: {exc}") from exc
    nodes = command[10:]
    if any("train" in node.lower() or "eval" in node.lower() for node in nodes):
        raise InterimLaunchError("test route refuses training or evaluation targets")


@contextmanager
def _elapsed_alarm(deadline: float) -> Iterator[None]:
    """Bound review, lock/lease admission and execution before cleanup reserve."""
    remaining = _remaining(deadline)
    if remaining <= 0:
        raise InterimLaunchError("test review exceeded its 120-second complete deadline")

    def expired(_signal: int, _frame: object) -> None:
        raise InterimLaunchError("test review exceeded its 120-second complete deadline")

    prior = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    entered = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
        if previous_timer[0] > 0:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous_timer[0] - (time.monotonic() - entered)),
                previous_timer[1],
            )


def _reject_light_command(command: Sequence[str]) -> None:
    if not command:
        raise InterimLaunchError("light route requires a command")
    name = Path(command[0]).name
    if name not in _LIGHT_EXECUTABLES:
        raise InterimLaunchError("light route accepts only reviewed advice or read-only tools")
    if name in {"codex", "claude"}:
        if len(command) < 2 or command[1] not in {"exec", "--print"}:
            raise InterimLaunchError(
                "light client route accepts only codex/claude noninteractive work"
            )
        found = shutil.which(name)
        allowed = (
            "/usr/local/bin/",
            "/opt/homebrew/bin/",
            str(Path.home() / ".codex") + "/",
            str(Path.home() / ".local/bin") + "/",
        )
        if (
            command[0] != name
            or not found
            or not any(found.startswith(prefix) for prefix in allowed)
        ):
            raise InterimLaunchError("light client executable is outside the reviewed client path")
        return
    try:
        _light(command, Path.cwd())
    except CommandValidationError as exc:
        raise InterimLaunchError(f"light route is not a reviewed read-only command: {exc}") from exc


def _spawn(command: Sequence[str]) -> subprocess.Popen[bytes]:
    parent_nice = _nice_floor()
    # nice(1) takes a relative increment.  It preserves a less favorable
    # inherited priority (for example nice 19) and all descendants inherit it.
    increment = max(0, 15 - parent_nice)
    child = subprocess.Popen(
        ["/usr/bin/nice", "-n", str(increment), *command],
        env=_env(),
        start_new_session=True,
    )
    actual = os.getpriority(os.PRIO_PROCESS, child.pid)
    if actual < 15:
        _cleanup(child, time.monotonic() + 2)
        raise InterimLaunchError("child did not inherit nice >= 15")
    return child


def _group_members(pgid: int, deadline: float) -> set[int]:
    """Return executable members of our process group from one process-table read."""
    remaining = _remaining(deadline)
    if remaining <= 0:
        raise InterimLaunchError("owned cleanup exhausted its 120-second deadline")
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        timeout=min(1.0, remaining),
    )
    members = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(field.isdigit() for field in fields[:2]):
            if int(fields[1]) == pgid and not fields[2].startswith("Z"):
                members.add(int(fields[0]))
    return members


def _cleanup(child: subprocess.Popen[bytes], deadline: float) -> None:
    """Stop only the owned process group, bounded by the launch deadline."""
    for sig, grace in (
        (signal.SIGTERM, _TERM_GRACE_SECONDS),
        (signal.SIGKILL, _KILL_GRACE_SECONDS),
    ):
        child.poll()
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            child.poll()
            return
        available = max(0, _remaining(deadline))
        if sig == signal.SIGTERM:
            available /= 2
        signal_deadline = min(deadline, time.monotonic() + min(grace, available))
        try:
            while _remaining(signal_deadline) > 0.05:
                child.poll()
                if not _group_members(child.pid, signal_deadline):
                    return
                time.sleep(min(0.02, max(0.0, _remaining(signal_deadline))))
        except (OSError, subprocess.SubprocessError, InterimLaunchError):
            # Missing observation is incomplete, but cannot prevent the owned
            # group receiving KILL before we release either admission fence.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.poll()
            raise
    raise InterimLaunchError("owned test process group did not stop before deadline")


def _wait(child: subprocess.Popen[bytes], deadline: float) -> int:
    while child.poll() is None and _remaining(deadline) > 0:
        if TIMING_ARMED_PATH.exists():
            raise InterimLaunchError("timing/preemption arm is active; serial test stopped")
        time.sleep(min(0.05, _remaining(deadline)))
    if child.poll() is None:
        raise InterimLaunchError("test route exceeded its 120-second complete deadline")
    return int(child.returncode)


def run_light(command: Sequence[str]) -> int:
    """Run a low-priority advice/read-only command without taking the Mac lease."""
    _reject_light_command(command)
    child = _spawn(command)
    return child.wait()


def run_test(
    command: Sequence[str],
    *,
    review_manifest: str,
    review_sha256: str,
    deadline_seconds: float = TEST_DEADLINE_SECONDS,
) -> int:
    """Run one reviewed test under the slot and shared Mac lease, or refuse."""
    if (
        isinstance(deadline_seconds, bool)
        or not isinstance(deadline_seconds, (int, float))
        or not math.isfinite(deadline_seconds)
        or not 0 < deadline_seconds <= TEST_DEADLINE_SECONDS
    ):
        raise InterimLaunchError("test deadline must be positive and no greater than 120 seconds")
    # This outer alarm also bounds Mac-lease release, including its mutation
    # lock, after the inner work alarm is cancelled for owned cleanup.
    with _elapsed_alarm(time.monotonic() + deadline_seconds):
        return _run_test(
            command,
            review_manifest=review_manifest,
            review_sha256=review_sha256,
            deadline_seconds=deadline_seconds,
        )


def _run_test(
    command: Sequence[str], *, review_manifest: str, review_sha256: str, deadline_seconds: float
) -> int:
    """Retain both admission fences through work, cleanup and lease release."""
    started = time.monotonic()
    deadline = started + deadline_seconds
    cleanup_reserve = min(_CLEANUP_RESERVE_SECONDS, deadline_seconds * 0.6)
    work_deadline = deadline - cleanup_reserve

    def interrupted(_signal: int, _frame: object) -> None:
        raise InterimLaunchError("serial test interrupted; owned cleanup required")

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupted)
        with _elapsed_alarm(work_deadline):
            _reject_test_command(
                command=command,
                review_manifest=review_manifest,
                review_sha256=review_sha256,
                worktree=Path.cwd(),
            )
        with serial_test_slot():
            if TIMING_ARMED_PATH.exists():
                raise InterimLaunchError("timing/preemption arm is active; serial test refused")
            if _remaining(deadline) <= cleanup_reserve:
                raise InterimLaunchError("test admission consumed the complete deadline")
            try:
                # Entering the lease includes its mutation lock; bound that
                # blocking admission too. ExitStack keeps the lease through
                # cleanup after the work alarm has been cancelled.
                from contextlib import ExitStack

                with ExitStack() as stack:
                    with _elapsed_alarm(work_deadline):
                        stack.enter_context(
                            held(
                                "factory-interim-serial-test",
                                "F0 interim reviewed serial test",
                                seconds=deadline_seconds,
                            )
                        )
                    if TIMING_ARMED_PATH.exists():
                        raise InterimLaunchError(
                            "timing/preemption arm is active; serial test refused"
                        )
                    child = _spawn(command)
                    try:
                        with _elapsed_alarm(work_deadline):
                            status = _wait(child, work_deadline)
                    finally:
                        for sig in old_handlers:
                            signal.signal(sig, signal.SIG_IGN)
                        _cleanup(child, deadline)
            except LeaseHeld as exc:
                raise InterimLaunchError(
                    "factory/timing lease is active; serial test refused"
                ) from exc
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    if _remaining(deadline) < 0:
        raise InterimLaunchError("test route exceeded its 120-second complete deadline")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="launch F0 nice-interim local work")
    parser.add_argument("--class", dest="klass", required=True, choices=("light", "test"))
    parser.add_argument("--review-manifest")
    parser.add_argument("--review-sha256")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        if args.klass == "light":
            if args.review_manifest is not None or args.review_sha256 is not None:
                raise InterimLaunchError("light route does not accept a test review manifest")
            code = run_light(command)
        else:
            code = run_test(
                command,
                review_manifest=args.review_manifest or "",
                review_sha256=args.review_sha256 or "",
            )
    except InterimLaunchError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
