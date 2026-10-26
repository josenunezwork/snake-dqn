"""Mac-only runtime fence for the once-only F4 harm post-mortem exporter.

This module deliberately owns no payload paths and computes no diagnostic
values.  It establishes the host conditions that must hold before the caller
creates its separate, create-only claim, and supplies a cheap callback that
the rowwise held-out loop can invoke while it is reading allowed inputs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from research import mac_lease
from research.compute.thermal_guard import ThermalGuard

WALL_SECONDS = 60 * 60
FINALIZATION_SECONDS = 30.0
GUARD_INTERVAL_SECONDS = 30.0
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class PostmortemRuntimeError(RuntimeError):
    """The post-mortem runtime is no longer safe to continue."""


class PostmortemRuntimeDeadline(PostmortemRuntimeError):
    """The single claimed run exhausted its fixed wall-clock envelope."""


class PostmortemRuntimeSignal(PostmortemRuntimeError):
    """An operator signal ended the claimed run before terminal export."""


class _LeaseContext(Protocol):
    def __enter__(self) -> mac_lease.MacLease: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool | None: ...


def _on_ac_power(runner: Callable[..., object] = subprocess.run) -> bool:
    """Return true only for an unambiguous macOS AC reading."""
    try:
        completed = runner(
            ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10.0, check=True
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return "AC Power" in str(getattr(completed, "stdout", ""))


def _lid_open(runner: Callable[..., object] = subprocess.run) -> bool:
    """Return true only when macOS explicitly reports an open lid."""
    try:
        completed = runner(
            ["ioreg", "-r", "-k", "AppleClamshellState"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=True,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    return '"AppleClamshellState" = No' in str(getattr(completed, "stdout", ""))


def _require_darwin() -> None:
    if sys.platform != "darwin":
        raise PostmortemRuntimeError("F4 post-mortem exporter is Mac Darwin CPU only")


def _set_exact_nice() -> int:
    """Set nice to 15; a more-nice inherited process cannot be promoted."""
    current = os.nice(0)
    if current > 15:
        raise PostmortemRuntimeError(
            f"inherited nice {current}; cannot lower it to required nice 15"
        )
    if current < 15:
        os.nice(15 - current)
    actual = os.nice(0)
    if actual != 15:
        raise PostmortemRuntimeError(f"required nice 15, found {actual}")
    return actual


def _configure_one_thread_torch() -> None:
    """Validate the shell fence before lazily configuring torch native pools."""
    expected = {"SNAKE_DQN_DEVICE": "cpu", **{name: "1" for name in THREAD_ENVIRONMENT}}
    mismatched = [name for name, value in expected.items() if os.environ.get(name) != value]
    if mismatched:
        raise PostmortemRuntimeError(
            "F4 post-mortem requires inherited fixed environment: " + ", ".join(mismatched)
        )
    # Torch initializes native pools during import.  The exporter must enter
    # this fence before any inference module imports it.
    if "torch" in sys.modules:
        raise PostmortemRuntimeError("torch was imported before F4 runtime thread fence")
    import torch  # pylint: disable=import-outside-toplevel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise PostmortemRuntimeError("torch did not accept required one-thread configuration")


@dataclass(frozen=True)
class RuntimeEvidence:
    """Host facts suitable for a count-only terminal receipt."""

    lease: dict[str, object]
    nice: int
    threads: int
    _guard: ThermalGuard = field(repr=False, compare=False)
    _lid_reader: Callable[[], bool] = field(repr=False, compare=False)
    _wake: subprocess.Popen[object] = field(repr=False, compare=False)

    def checkpoint(self) -> None:
        """Revalidate every process-wide host invariant without reading a payload."""
        if os.nice(0) != 15:
            raise PostmortemRuntimeError("F4 post-mortem runtime nice level changed from 15")
        expected = {"SNAKE_DQN_DEVICE": "cpu", **{name: "1" for name in THREAD_ENVIRONMENT}}
        if any(os.environ.get(name) != value for name, value in expected.items()):
            raise PostmortemRuntimeError("F4 post-mortem runtime thread/device environment changed")
        if self._wake.poll() is not None:
            raise PostmortemRuntimeError("owned caffeinate helper exited during F4 post-mortem")
        torch = sys.modules.get("torch")
        if torch is None or torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise PostmortemRuntimeError("F4 post-mortem torch thread configuration changed")

    def deadline(self) -> "ClaimDeadline":
        """Create the claim-to-terminal-export deadline after host admission.

        The entrypoint must enter this immediately before its create-only claim
        and retain it through its terminal result or failure receipt.
        """
        return ClaimDeadline(
            guard=self._guard, lid_reader=self._lid_reader, runtime_check=self.checkpoint
        )


class ClaimDeadline:
    """Main-process SIGALRM deadline and periodic host guard without a watcher."""

    def __init__(
        self,
        *,
        seconds: float = WALL_SECONDS,
        guard: ThermalGuard,
        lid_reader: Callable[[], bool],
        runtime_check: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        signal_module: object = signal,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if seconds <= 0:
            raise ValueError("claim deadline must be positive")
        self.seconds = float(seconds)
        self.guard = guard
        self.lid_reader = lid_reader
        self.runtime_check = runtime_check
        self.clock = clock
        self.signal = signal_module
        self.started: float | None = None
        self._compute_seconds = max(0.0, self.seconds - FINALIZATION_SECONDS)
        self._hard_deadline_armed = False
        self._checking_host = False
        self._old_handlers: dict[int, object] = {}
        self._old_timer: tuple[float, float] | None = None
        self.progress = progress or self._write_progress

    @staticmethod
    def _write_progress(row: dict[str, Any]) -> None:
        print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)

    def _check_host(self) -> None:
        if self.runtime_check is not None:
            self.runtime_check()
        if not self.lid_reader():
            raise PostmortemRuntimeError("F4 post-mortem host guard: lid closed or unreadable")
        reading = self.guard.check()
        if not reading["ok"]:
            raise PostmortemRuntimeError(
                "F4 post-mortem host guard: " + ", ".join(str(x) for x in reading["reasons"])
            )

    def _elapsed(self) -> float:
        assert self.started is not None, "claim deadline is not active"
        return self.clock() - self.started

    def _arm_hard_deadline(self) -> None:
        """Keep the absolute 60-minute kill switch active through finalization."""
        if self._hard_deadline_armed:
            return
        remaining = self.seconds - self._elapsed()
        if remaining <= 0:
            os._exit(124)
        self._hard_deadline_armed = True
        self.signal.setitimer(self.signal.ITIMER_REAL, max(0.001, remaining))

    def _stop_compute(self, reason: str) -> None:
        self._arm_hard_deadline()
        raise PostmortemRuntimeDeadline(reason)

    def _compute_remaining(self) -> float:
        return self._compute_seconds - self._elapsed()

    def _arm_next_guard(self) -> None:
        """Schedule another host guard before entering a potentially blocking probe."""
        remaining = self._compute_remaining()
        if remaining <= 0:
            self._stop_compute("F4 post-mortem compute window expired; finalization reserve only")
        self.signal.setitimer(
            self.signal.ITIMER_REAL, min(GUARD_INTERVAL_SECONDS, max(0.001, remaining))
        )

    def checkpoint(self) -> None:
        """Check wall and host state; suitable as the held-out observer callback."""
        if self._elapsed() >= self._compute_seconds:
            self._stop_compute("F4 post-mortem compute window expired; finalization reserve only")
        try:
            self._check_host()
        except PostmortemRuntimeError:
            self._arm_hard_deadline()
            raise
        # Host probes may each block for up to ten seconds.  Recheck against
        # the same absolute monotonic deadline after they return.
        if self._elapsed() >= self._compute_seconds:
            self._stop_compute("F4 post-mortem compute window expired; finalization reserve only")

    def guard_callback(self) -> Callable[..., None]:
        """Return a permissive callback for row/mask observers (arguments ignored)."""

        def callback(*_args: object, **_kwargs: object) -> None:
            self.checkpoint()

        return callback

    def _alarm(self, _signum: int, _frame: object) -> None:
        if self._hard_deadline_armed:
            os._exit(124)
        if self._compute_remaining() <= 0:
            self._stop_compute("F4 post-mortem compute window expired; finalization reserve only")
        self.progress(
            {
                "event": "f4_postmortem_runtime_guard",
                "elapsed_seconds": round(self._elapsed(), 3),
                "seconds_left": round(max(0.0, self.seconds - self._elapsed()), 3),
            }
        )
        # Arm first: AC/lid/thermal probes may each block, and a nested alarm
        # must retain the absolute compute/finalization boundary rather than
        # wait for the interrupted probe to return.
        self._arm_next_guard()
        if self._checking_host:
            return
        self._checking_host = True
        try:
            self._check_host()
        except PostmortemRuntimeError:
            self._arm_hard_deadline()
            raise
        finally:
            self._checking_host = False
        if self._compute_remaining() <= 0:
            self._stop_compute("F4 post-mortem compute window expired; finalization reserve only")

    def _terminal_signal(self, signum: int, _frame: object) -> None:
        self._arm_hard_deadline()
        raise PostmortemRuntimeSignal(f"F4 post-mortem interrupted by signal {signum}")

    def __enter__(self) -> "ClaimDeadline":
        if not hasattr(self.signal, "setitimer") or not hasattr(self.signal, "getitimer"):
            raise PostmortemRuntimeError("SIGALRM timer is required for F4 post-mortem deadline")
        if hasattr(self.signal, "pthread_sigmask") and self.signal is signal:
            # signal.signal itself will enforce main-thread use; make the error
            # explicit so an exporter cannot silently lose its run ceiling.
            import threading

            if threading.current_thread() is not threading.main_thread():
                raise PostmortemRuntimeError(
                    "F4 claim deadline must run on the main process thread"
                )
        self.started = self.clock()
        existing_timer = self.signal.getitimer(self.signal.ITIMER_REAL)
        if existing_timer != (0.0, 0.0):
            raise PostmortemRuntimeError("F4 claim deadline refuses a preexisting SIGALRM timer")
        for signum in (self.signal.SIGALRM, self.signal.SIGINT, self.signal.SIGTERM):
            self._old_handlers[signum] = self.signal.getsignal(signum)
        self.signal.signal(self.signal.SIGALRM, self._alarm)
        self.signal.signal(self.signal.SIGINT, self._terminal_signal)
        self.signal.signal(self.signal.SIGTERM, self._terminal_signal)
        # Arm before any host probe: each probe has its own bounded timeout,
        # but their combined time must never push this claimed run past 3600s.
        self._old_timer = self.signal.setitimer(
            self.signal.ITIMER_REAL, min(GUARD_INTERVAL_SECONDS, max(0.001, self._compute_seconds))
        )
        try:
            self.checkpoint()
        except BaseException:
            # No claim has been created if entry admission fails, so restore
            # process signal state rather than leaving a timer behind.
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.signal.setitimer(self.signal.ITIMER_REAL, 0.0)
        for signum, handler in self._old_handlers.items():
            self.signal.signal(signum, handler)
        return False


@contextmanager
def claim_deadline(
    *,
    seconds: float = WALL_SECONDS,
    guard: ThermalGuard,
    lid_reader: Callable[[], bool],
    clock: Callable[[], float] = time.monotonic,
    runtime_check: Callable[[], None] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[ClaimDeadline]:
    """Apply the fixed claim-to-terminal-export ceiling in the main process."""
    deadline = ClaimDeadline(
        seconds=seconds,
        guard=guard,
        lid_reader=lid_reader,
        clock=clock,
        runtime_check=runtime_check,
        progress=progress,
    )
    with deadline:
        yield deadline


def _wait_for_lease(
    factory: Callable[[], _LeaseContext], *, sleep: Callable[[float], None], poll_seconds: float
) -> "_EnteredLease":
    """Wait for a research lease release; never preempt or replace another owner."""
    while True:
        context = factory()
        try:
            lease = context.__enter__()
        except mac_lease.LeaseHeld:
            sleep(poll_seconds)
            continue
        return _EnteredLease(context, lease)


class _EnteredLease:
    """Adapt a lease context already entered by the non-preemptive wait loop."""

    def __init__(self, context: _LeaseContext, lease: mac_lease.MacLease) -> None:
        self.context = context
        self.lease = lease

    def __enter__(self) -> mac_lease.MacLease:
        return self.lease

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool | None:
        return self.context.__exit__(exc_type, exc, traceback)


@contextmanager
def exclusive_runtime(
    *,
    owner: str = "f4-postmortem-harm-20261025-01",
    purpose: str = "F4 once-only harm post-mortem export",
    lease_path: Path | None = None,
    seconds: float = WALL_SECONDS,
    poll_seconds: float = GUARD_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    runner: Callable[..., object] = subprocess.run,
    popen: Callable[..., subprocess.Popen[object]] = subprocess.Popen,
) -> Iterator[RuntimeEvidence]:
    """Acquire the Mac lease and wake assertion before a caller may claim inputs.

    The caller must enter :func:`claim_deadline` immediately before writing its
    create-only claim.  No payload is accepted by this function.
    """
    _require_darwin()
    if seconds != WALL_SECONDS:
        raise ValueError("F4 post-mortem runtime ceiling is fixed at 60 minutes")
    if poll_seconds <= 0 or poll_seconds > GUARD_INTERVAL_SECONDS:
        raise ValueError("lease polling must be positive and at most 30 seconds")
    _set_exact_nice()
    _configure_one_thread_torch()
    if not _on_ac_power(runner) or not _lid_open(runner):
        raise PostmortemRuntimeError("F4 post-mortem requires AC power and an open lid")
    guard = ThermalGuard(ac_reader=lambda: _on_ac_power(runner))
    initial = guard.check()
    if not initial["ok"]:
        raise PostmortemRuntimeError(
            "F4 post-mortem thermal guard: " + ", ".join(initial["reasons"])
        )

    def lease_factory() -> _LeaseContext:
        return mac_lease.held(owner, purpose, seconds=seconds, path=lease_path)

    held_context = _wait_for_lease(lease_factory, sleep=sleep, poll_seconds=poll_seconds)
    wake: subprocess.Popen[object] | None = None
    try:
        # A helper, never an inference worker: the exporter remains one process.
        wake = popen(["/usr/bin/caffeinate", "-is", "-w", str(os.getpid())])
        if wake.poll() is not None:
            raise PostmortemRuntimeError("owned caffeinate helper exited at startup")
        yield RuntimeEvidence(
            lease=held_context.lease.receipt,
            nice=15,
            threads=1,
            _guard=guard,
            _lid_reader=lambda: _lid_open(runner),
            _wake=wake,
        )
    finally:
        if wake is not None and wake.poll() is None:
            wake.terminate()
            try:
                wake.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                wake.kill()
                wake.wait(timeout=10.0)
        held_context.__exit__(None, None, None)
