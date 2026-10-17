"""Opt-in in-process profiler for ``train_m3c`` processes (perf-profile, 2026-10-13).

py-spy needs root on macOS, so the trainer can profile itself instead. Off by default: with
``M3C_PROFILE`` unset (or ``off``) :meth:`ProcessProfile.from_env` returns ``None`` and the
call sites only compare that ``None`` once per frame / loop iteration (no profiler is
imported, nothing is timed, the training state is never read or written).

Environment:

* ``M3C_PROFILE``: ``cprofile`` (deterministic, every call; inflates Python-call-heavy code)
  or ``pyinstrument`` (statistical; ``pip install pyinstrument``; the timing thread is used,
  so the per-call cost is small). Unset / empty / ``off``: disabled.
* ``M3C_PROFILE_DIR``: where each process writes its dump (default: ``<--out>/profile``).
* ``M3C_PROFILE_WARM_S``: seconds after the process created its profiler before sampling
  starts (default 0), to keep numba JIT / segment construction / warm-up out of the window.
* ``M3C_PROFILE_INTERVAL``: pyinstrument sampling interval in seconds (default 0.001).
* ``M3C_PROFILE_WARM_FROM_FIRST_TICK=1``: start the warm-up clock at the first learner
  loop / actor frame, excluding model loading and process setup (default off).
* ``M3C_PROFILE_DURATION_S``: stop at the first tick after this many active seconds
  (default 0 = capture until exit). Actual elapsed time is always recorded in metadata.

Per process (``role`` = ``learner`` / ``actor<j>``) the dump is ``<role>.pstats`` (cProfile)
or ``<role>.pyisession`` (pyinstrument JSON session), a readable ``<role>.txt`` and
``<role>.meta.json`` (pid, mode, profiled seconds, ticks = frames / loop iterations profiled,
plus the caller's counters). ``research/perf_profile_20261013/analyze_profiles.py`` maps the
dumps onto actor / learner components. Dumps are written at process exit (an actor in its
``finally``, the learner after ``summary.json``) by :meth:`ProcessProfile.safe_dump`: a failed
dump is reported on stderr and never changes the run's outcome.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROFILE_ENV = "M3C_PROFILE"
PROFILE_DIR_ENV = "M3C_PROFILE_DIR"
PROFILE_WARM_ENV = "M3C_PROFILE_WARM_S"
PROFILE_INTERVAL_ENV = "M3C_PROFILE_INTERVAL"
PROFILE_DURATION_ENV = "M3C_PROFILE_DURATION_S"
PROFILE_FIRST_TICK_ENV = "M3C_PROFILE_WARM_FROM_FIRST_TICK"
MODES = ("cprofile", "pyinstrument")
OFF = ("", "0", "off", "none")


def profile_mode() -> Optional[str]:
    """The requested mode, ``None`` when profiling is off (raises on an unknown mode)."""
    mode = os.environ.get(PROFILE_ENV, "").strip().lower()
    if mode in OFF:
        return None
    if mode not in MODES:
        raise ValueError(f"{PROFILE_ENV} must be one of {MODES} (or unset), got {mode!r}")
    return mode


def save_session_compact(session: Any, path: Path) -> int:
    """Save a pyinstrument session with identical stacks merged (same format, loadable by
    ``pyinstrument.session.Session.load``; a run's ~10^5 samples collapse to ~10^3 stacks).
    Returns the number of merged records."""
    from pyinstrument.session import Session

    merged: Dict[tuple, float] = {}
    for stack, t in session.frame_records:
        key = tuple(stack)
        merged[key] = merged.get(key, 0.0) + t
    compact = Session(
        frame_records=[(list(k), t) for k, t in merged.items()],
        start_time=session.start_time,
        duration=session.duration,
        min_interval=session.min_interval,
        max_interval=session.max_interval,
        sample_count=session.sample_count,
        start_call_stack=session.start_call_stack,
        target_description=session.target_description,
        cpu_time=session.cpu_time,
        sys_path=session.sys_path,
        sys_prefixes=session.sys_prefixes,
    )
    compact.save(str(path))
    return len(merged)


class ProcessProfile:
    """One process's profiler: created idle, started by :meth:`tick` after the warm-up."""

    def __init__(
        self,
        mode: str,
        role: str,
        out_dir: Path,
        warm_s: float = 0.0,
        interval: float = 0.001,
        duration_s: float = 0.0,
        warm_from_first_tick: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "pyinstrument":  # fail at process start, not after the warm-up
            try:
                import pyinstrument  # noqa: F401
            except ImportError as exc:  # pragma: no cover - depends on the venv
                raise RuntimeError(
                    f"{PROFILE_ENV}=pyinstrument needs `pip install pyinstrument`"
                ) from exc
        self.mode, self.role = mode, str(role)
        self.out_dir = Path(out_dir)
        self.interval = float(interval)
        self.created = time.perf_counter()
        self.start_at = self.created + max(0.0, float(warm_s))
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError(f"{PROFILE_DURATION_ENV} must be finite and >= 0")
        self.duration_s = float(duration_s)
        self.warm_from_first_tick = bool(warm_from_first_tick)
        self._first_tick: Optional[float] = None
        self.t_start: Optional[float] = None
        self.t_stop: Optional[float] = None
        self.ticks = 0  # tick() calls while running (frames / loop iterations profiled)
        self.meta: Dict[str, Any] = {}
        self._prof: Any = None
        self._session: Any = None

    @classmethod
    def from_env(cls, role: str, default_dir: Optional[Path] = None) -> Optional["ProcessProfile"]:
        """The profiler requested by the environment, or ``None`` (the default: off)."""
        mode = profile_mode()
        if mode is None:
            return None
        out = os.environ.get(PROFILE_DIR_ENV) or (
            str(Path(default_dir) / "profile") if default_dir else "m3c_profile"
        )
        warm = float(os.environ.get(PROFILE_WARM_ENV) or 0.0)
        interval = float(os.environ.get(PROFILE_INTERVAL_ENV) or 0.001)
        return cls(
            mode,
            role,
            Path(out),
            warm,
            interval,
            duration_s=float(os.environ.get(PROFILE_DURATION_ENV) or 0.0),
            warm_from_first_tick=os.environ.get(PROFILE_FIRST_TICK_ENV, "0") == "1",
        )

    @property
    def running(self) -> bool:
        return self._prof is not None and self.t_stop is None

    def tick(self) -> bool:
        """Call once per frame / loop iteration; returns True on the call that starts it."""
        if self.t_stop is not None:
            return False
        now = time.perf_counter()
        if self.warm_from_first_tick and self._first_tick is None:
            self.start_at = now + (self.start_at - self.created)
            self._first_tick = now
        if self._prof is not None:
            if self.duration_s and now - self.t_start >= self.duration_s:
                self.stop()
                return False
            self.ticks += 1
            return False
        if now < self.start_at:
            return False
        self._start()
        return True

    def _start(self) -> None:
        if self.mode == "cprofile":
            import cProfile

            self._prof = cProfile.Profile()
            self._prof.enable()
        else:
            os.environ.setdefault("PYINSTRUMENT_IGNORE_OVERHEAD_WARNING", "1")
            from pyinstrument import Profiler

            self._prof = Profiler(
                interval=self.interval, async_mode="disabled", use_timing_thread=True
            )
            self._prof.start()
        self.t_start = time.perf_counter()

    def stop(self) -> None:
        if not self.running:
            return
        if self.mode == "cprofile":
            self._prof.disable()
        else:
            self._session = self._prof.stop()
        self.t_stop = time.perf_counter()

    def dump(self, extra: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Stop (if running) and write the dump; returns the meta path (``None`` if the
        profiler never started, e.g. the process ended inside the warm-up)."""
        self.stop()
        if self._prof is None:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / self.role
        if self.mode == "cprofile":
            import io
            import pstats

            self._prof.dump_stats(str(base) + ".pstats")
            s = io.StringIO()
            st = pstats.Stats(self._prof, stream=s)
            st.sort_stats("tottime").print_stats(60)
            st.sort_stats("cumulative").print_stats(60)
            text = s.getvalue()
        else:
            self.meta["stacks"] = save_session_compact(
                self._session, Path(str(base) + ".pyisession")
            )
            self.meta["samples"] = self._session.sample_count
            from pyinstrument.renderers import ConsoleRenderer

            text = ConsoleRenderer(unicode=False, color=False, show_all=False).render(self._session)
        (Path(str(base) + ".txt")).write_text(text)
        meta = {
            "role": self.role,
            "pid": os.getpid(),
            "mode": self.mode,
            "interval": self.interval if self.mode == "pyinstrument" else None,
            "warm_s": self.start_at - self.created,
            "profiled_s": (self.t_stop or time.perf_counter()) - (self.t_start or 0.0),
            "ticks": self.ticks,
            **self.meta,
            **(extra or {}),
        }
        if self.duration_s or self.warm_from_first_tick:
            meta.update(
                requested_duration_s=self.duration_s,
                warm_from_first_tick=self.warm_from_first_tick,
                warm_s=self.start_at - (self._first_tick or self.created),
            )
        path = Path(str(base) + ".meta.json")
        path.write_text(json.dumps(meta, indent=1, default=str) + "\n")
        return path

    def safe_dump(self, extra: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """:meth:`dump` for a ``finally`` block: a failed dump is reported on stderr and never
        raises (the trainer's own teardown and exit status must not depend on the profiler)."""
        try:
            return self.dump(extra)
        except Exception as exc:  # noqa: BLE001
            print(f"[{PROFILE_ENV}] {self.role}: profile dump failed: {exc!r}", file=sys.stderr)
            return None
