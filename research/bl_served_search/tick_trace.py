"""Bounded, preallocated tracing for the served Watch timing probe.

The recorder intentionally records *inclusive* and *exclusive* timing separately.
Consumers must not sum parents and children: ``game_update`` includes its children,
while ``*_exclusive`` removes direct nested child time.  The module owns no output
files; callers export :meth:`TraceRecorder.rows` after the paced run has stopped.
"""

from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

FPS_DEFAULT = 24
UPDATE_P95_LIMIT_NS = 11_688_000
SPAN_NAMES = (
    "session_step",
    "game_update",
    "snapshot",
    "dto",
    "observation",
    "snake_update",
    "hero_forward",
    "hero_action_selection",
    "opponent_forward",
    "opponent_action_selection",
    "opponent_veto",
    "simulation_collision",
    "food_spawn_respawn",
    "ipc_lock",
)
_SPAN_INDEX = {name: index for index, name in enumerate(SPAN_NAMES)}
# These hooks describe work performed by the serving step itself.  A snapshot can
# legitimately call some of the same policy/snake helpers for inspector output;
# that work belongs to DTO/snapshot timing, not to step inference/update spans.
_STEP_INTERNAL_SPANS = frozenset(
    {
        "observation",
        "snake_update",
        "hero_forward",
        "hero_action_selection",
        "opponent_forward",
        "opponent_action_selection",
        "opponent_veto",
        "simulation_collision",
        "food_spawn_respawn",
    }
)

_TICK_DTYPE = np.dtype(
    [
        ("tick", "<i8"),
        ("target_ns", "<i8"),
        ("wake_ns", "<i8"),
        ("start_ns", "<i8"),
        ("end_ns", "<i8"),
        ("start_cpu_ns", "<i8"),
        ("end_cpu_ns", "<i8"),
        ("game_age", "<i8"),
        ("hero_length", "<i8"),
        ("opponent_lengths", "<i8", (5,)),
        ("opponent_alive", "<i8", (5,)),
        ("food_count", "<i8"),
        ("alive_count", "<i8"),
        ("respawns", "<i8"),
        ("thread_id", "<i8"),
        ("gc_start", "<i8", (3,)),
        ("gc_stop", "<i8", (3,)),
        ("gc_overlap_ns", "<i8", (3,)),
    ]
)


class TraceError(RuntimeError):
    """The trace protocol was used outside a valid tick."""


class TraceOverflowError(TraceError):
    """A bounded trace cannot silently discard a tick."""


class _Span:
    """One stable context-manager object created with its recorder hook."""

    __slots__ = ("recorder", "name")

    def __init__(self, recorder: "TraceRecorder", name: str) -> None:
        self.recorder = recorder
        self.name = name

    def __enter__(self) -> "_Span":
        self.recorder._enter_span(self.name)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.recorder._exit_span(self.name)
        return False


class _TimedLock:
    """RLock-compatible proxy which charges only acquisition wait to a span."""

    __slots__ = ("_lock", "_recorder")

    def __init__(self, lock: Any, recorder: "TraceRecorder") -> None:
        self._lock = lock
        self._recorder = recorder

    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        if self._recorder._is_active_on_current_thread():
            with self._recorder.span("ipc_lock"):
                return self._lock.acquire(*args, **kwargs)
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> Any:
        return self._lock.release()

    def __enter__(self) -> "_TimedLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.release()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lock, name)


class TraceRecorder:
    """Fixed-capacity trace recorder for one tick-owning thread.

    ``begin_tick`` and ``end_tick`` delimit a tick.  All mutable arrays are
    allocated at construction; small Python temporaries remain. The recorder
    does no JSON serialization, disk I/O, or bound-method wrapper construction
    on a tick. The bounded
    wrappers add normal Python call overhead, which the observer-control arm
    measures separately.  Overflow poisons the recorder so a partial timing
    panel is never presented as a pass.
    """

    def __init__(
        self,
        capacity: int,
        *,
        wall_clock_ns: Callable[[], int] = time.perf_counter_ns,
        cpu_clock_ns: Callable[[], int] = time.thread_time_ns,
    ) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._wall_clock_ns = wall_clock_ns
        self._cpu_clock_ns = cpu_clock_ns
        self._ticks = np.zeros(capacity, dtype=_TICK_DTYPE)
        self._wall_inclusive = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._wall_exclusive = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._cpu_inclusive = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._cpu_exclusive = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._span_calls = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._span_wall_start_ns = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._span_wall_end_ns = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._span_cpu_start_ns = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._span_cpu_end_ns = np.zeros((capacity, len(SPAN_NAMES)), dtype=np.int64)
        self._count = 0
        self._active = -1
        self._failed = False
        # A bounded nesting stack prevents per-span list allocation in the tick.
        self._stack = np.zeros((64, 5), dtype=np.int64)
        self._stack_depth = 0
        self._span_objects = {name: _Span(self, name) for name in SPAN_NAMES}
        self._gc_open = np.zeros(3, dtype=np.int64)
        self._gc_open_active = np.zeros(3, dtype=np.bool_)
        self._gc_callback_installed = False

    @property
    def count(self) -> int:
        return self._count

    @property
    def failed(self) -> bool:
        return self._failed

    def begin_tick(self, index: int, target_ns: int, wake_ns: int) -> None:
        """Start one tick, recording the absolute target and actual wake time."""
        if self._failed:
            raise TraceOverflowError("recorder is invalid after an earlier overflow")
        if self._active >= 0:
            raise TraceError("cannot begin a tick before ending the active tick")
        if self._count >= self.capacity:
            self._failed = True
            raise TraceOverflowError("trace capacity exhausted; timing panel is invalid")
        if not all(isinstance(value, int) for value in (index, target_ns, wake_ns)):
            raise TraceError("tick identity and clock values must be integers")
        row = self._count
        now = int(self._wall_clock_ns())
        cpu = int(self._cpu_clock_ns())
        tick = self._ticks[row]
        tick["tick"] = index
        tick["target_ns"] = target_ns
        tick["wake_ns"] = wake_ns
        tick["start_ns"] = now
        tick["end_ns"] = 0
        tick["start_cpu_ns"] = cpu
        tick["end_cpu_ns"] = 0
        tick["game_age"] = -1
        tick["hero_length"] = -1
        tick["opponent_lengths"] = -1
        tick["opponent_alive"] = -1
        tick["food_count"] = -1
        tick["alive_count"] = -1
        tick["respawns"] = -1
        tick["thread_id"] = threading.get_ident()
        self._active = row
        self._stack_depth = 0

    def end_tick(self, end_ns: int | None = None) -> None:
        """Finish the current tick; ``end_ns`` is for deterministic unit tests."""
        row = self._require_active()
        if self._stack_depth:
            raise TraceError("cannot end tick with an open timing span")
        end = int(self._wall_clock_ns()) if end_ns is None else int(end_ns)
        self._ticks[row]["end_ns"] = end
        self._ticks[row]["end_cpu_ns"] = int(self._cpu_clock_ns())
        self._active = -1
        self._count += 1

    def span(self, name: str) -> _Span:
        """Return a reusable nested span context manager for a declared span."""
        try:
            return self._span_objects[name]
        except KeyError as error:
            raise ValueError(f"unknown trace span {name!r}") from error

    @contextmanager
    def attach(self, session: Any) -> Iterator["TraceRecorder"]:
        """Attach stable, per-instance hooks and restore every hook on exit.

        The production classes are never changed.  Optional hooks are installed
        only where an instance supplies a callable attribute, which makes this
        safe for the real Watch6 session and small test doubles alike.
        """
        restores: list[tuple[Any, str, Any, bool]] = []
        hooked: set[tuple[int, str]] = set()
        build_frame_restore: tuple[Any, Any] | None = None

        def wrap(obj: Any, attribute: str, span_name: str) -> None:
            key = (id(obj), attribute)
            if key in hooked:
                return
            original = getattr(obj, attribute, None)
            if not callable(original):
                return
            had_instance_attribute = attribute in vars(obj)

            def traced(*args: Any, **kwargs: Any) -> Any:
                if not self._is_active_on_current_thread() or (
                    span_name in _STEP_INTERNAL_SPANS and not self._in_step_internal_context()
                ):
                    return original(*args, **kwargs)
                with self.span(span_name):
                    result = original(*args, **kwargs)
                if attribute == "respawn":
                    row = self._require_active()
                    if self._ticks[row]["respawns"] < 0:
                        self._ticks[row]["respawns"] = 0
                    self._ticks[row]["respawns"] += 1
                return result

            setattr(obj, attribute, traced)
            restores.append((obj, attribute, original, had_instance_attribute))
            hooked.add(key)

        # The wrappers are constructed once here, never in a tick.  session.step
        # brackets the complete serving endpoint; game.update is the 11.688ms bar.
        original_lock = getattr(session, "_lock", None)
        if original_lock is not None and callable(getattr(original_lock, "acquire", None)):
            had_instance_lock = "_lock" in vars(session)
            session._lock = _TimedLock(original_lock, self)
            restores.append((session, "_lock", original_lock, had_instance_lock))
        wrap(session, "step", "session_step")
        wrap(session, "snapshot", "snapshot")
        try:
            from web.backend import serialize

            original_build_frame = serialize.build_frame

            def traced_build_frame(candidate_session: Any, *args: Any, **kwargs: Any) -> Any:
                # GameSession imports this function lazily inside snapshot().
                # Time only the builder, excluding that import and RLock wait.
                if (
                    candidate_session is session
                    and self._active >= 0
                    and self._ticks[self._active]["thread_id"] == threading.get_ident()
                ):
                    with self.span("dto"):
                        return original_build_frame(candidate_session, *args, **kwargs)
                return original_build_frame(candidate_session, *args, **kwargs)

            serialize.build_frame = traced_build_frame
            build_frame_restore = (serialize, original_build_frame)
        except ImportError:
            pass
        game = getattr(session, "game", None)
        if game is not None:
            wrap(game, "update", "game_update")
            wrap(game, "handle_collisions", "simulation_collision")
            for name in ("manage_food", "spawn_food", "check_food_consumption"):
                wrap(game, name, "food_spawn_respawn")
            food_manager = getattr(game, "food_manager", None)
            for name in ("maintain_count", "spawn", "consume_at"):
                if food_manager is not None:
                    wrap(food_manager, name, "food_spawn_respawn")
            for snake in tuple(getattr(game, "snakes", ())):
                forward_span = (
                    "hero_forward"
                    if getattr(snake, "id", None) == getattr(session, "hero_id", None)
                    else "opponent_forward"
                )
                action_span = (
                    "hero_action_selection"
                    if forward_span == "hero_forward"
                    else "opponent_action_selection"
                )
                wrap(snake, "update", "snake_update")
                wrap(snake, "get_state", "observation")
                wrap(snake, "respawn", "food_spawn_respawn")
                policy = getattr(snake, "policy", None)
                if policy is not None:
                    # B8 does the live raster and batched CNN forward in
                    # prepare_frame -> _ensure_frame, before snake.update.
                    # The net hook gives its forward a distinct nested span.
                    wrap(policy, "_ensure_frame", "observation")
                    wrap(getattr(policy, "net", None), "forward", forward_span)
                    # Legacy Apex policies dispatch through policy.dqn(state),
                    # so their actual module forward is here rather than
                    # select_action/act (which are fallback paths).
                    wrap(getattr(policy, "dqn", None), "forward", forward_span)
                    wrap(policy, "select_action", action_span)
                    wrap(policy, "act", action_span)
                wrap(getattr(snake, "safety_veto", None), "apply", "opponent_veto")
        self._install_gc_callback()
        try:
            yield self
        finally:
            if build_frame_restore is not None:
                module, original_build_frame = build_frame_restore
                module.build_frame = original_build_frame
            self._remove_gc_callback()
            for obj, attribute, original, had_instance_attribute in reversed(restores):
                if had_instance_attribute:
                    setattr(obj, attribute, original)
                else:
                    delattr(obj, attribute)

    @contextmanager
    def attach_telemetry(self, session: Any) -> Iterator["TraceRecorder"]:
        """Route lock wait and GC activity to this recorder without method hooks.

        Production Hub dispatches the step and snapshot through independent worker
        threads.  This context can nest inside a step recorder's full attachment:
        the outer lock proxy records the active snapshot tick while the inner one
        simply delegates because its step tick is inactive on that thread.
        """
        original_lock = getattr(session, "_lock", None)
        restore_lock: tuple[Any, str, Any, bool] | None = None
        if original_lock is not None and callable(getattr(original_lock, "acquire", None)):
            had_instance_lock = "_lock" in vars(session)
            session._lock = _TimedLock(original_lock, self)
            restore_lock = (session, "_lock", original_lock, had_instance_lock)
        self._install_gc_callback()
        try:
            yield self
        finally:
            self._remove_gc_callback()
            if restore_lock is not None:
                obj, attribute, original, had_instance_attribute = restore_lock
                if had_instance_attribute:
                    setattr(obj, attribute, original)
                else:
                    delattr(obj, attribute)

    def capture_session_context(self, session: Any) -> None:
        """Store bounded per-tick workload fields after an actual Watch step."""
        row = self._require_active()
        game = session.game
        snakes = tuple(getattr(game, "snakes", ()))
        hero_id = getattr(session, "hero_id", None)
        hero = next((snake for snake in snakes if getattr(snake, "id", None) == hero_id), None)
        self._ticks[row]["game_age"] = int(getattr(game, "frame", -1))
        self._ticks[row]["hero_length"] = int(getattr(hero, "length", -1)) if hero else -1
        opponent_index = 0
        for snake in snakes:
            if snake is hero or opponent_index >= 5:
                continue
            self._ticks[row]["opponent_lengths"][opponent_index] = int(getattr(snake, "length", -1))
            self._ticks[row]["opponent_alive"][opponent_index] = int(
                bool(getattr(snake, "is_alive", False))
            )
            opponent_index += 1
        self._ticks[row]["food_count"] = len(getattr(game, "food", ()))
        self._ticks[row]["alive_count"] = int(getattr(game, "alive_snakes", -1))
        # ``attach``'s respawn wrapper records actual calls, including a snake
        # which respawns then dies in the same tick. A completed Watch step with
        # no respawns is explicitly zero rather than an unknown partial tick.
        if self._ticks[row]["respawns"] < 0:
            self._ticks[row]["respawns"] = 0

    def arrays(self) -> dict[str, np.ndarray]:
        """Return copied buffers after the tick loop; never call from a live tick."""
        if self._active >= 0:
            raise TraceError("cannot export a live trace")
        stop = self._count
        return {
            "ticks": self._ticks[:stop].copy(),
            "span_wall_ns": self._wall_inclusive[:stop].copy(),
            "span_wall_exclusive_ns": self._wall_exclusive[:stop].copy(),
            "span_cpu_ns": self._cpu_inclusive[:stop].copy(),
            "span_cpu_exclusive_ns": self._cpu_exclusive[:stop].copy(),
            "span_calls": self._span_calls[:stop].copy(),
            "span_wall_start_ns": self._span_wall_start_ns[:stop].copy(),
            "span_wall_end_ns": self._span_wall_end_ns[:stop].copy(),
            "span_cpu_start_ns": self._span_cpu_start_ns[:stop].copy(),
            "span_cpu_end_ns": self._span_cpu_end_ns[:stop].copy(),
            "span_names": np.asarray(SPAN_NAMES),
        }

    def rows(self) -> list[dict[str, Any]]:
        """Build JSON-ready rows outside the tick loop."""
        arrays = self.arrays()
        rows: list[dict[str, Any]] = []
        for row_index, tick in enumerate(arrays["ticks"]):
            row = {
                name: (
                    int(tick[name])
                    if tick.dtype[name].shape == ()
                    else [int(v) for v in tick[name]]
                )
                for name in tick.dtype.names or ()
            }
            row["spans"] = {
                name: {
                    "wall_ns": int(arrays["span_wall_ns"][row_index, span_index]),
                    "wall_exclusive_ns": int(
                        arrays["span_wall_exclusive_ns"][row_index, span_index]
                    ),
                    "cpu_ns": int(arrays["span_cpu_ns"][row_index, span_index]),
                    "cpu_exclusive_ns": int(arrays["span_cpu_exclusive_ns"][row_index, span_index]),
                    "calls": int(arrays["span_calls"][row_index, span_index]),
                    "first_wall_start_ns": int(arrays["span_wall_start_ns"][row_index, span_index]),
                    "last_wall_end_ns": int(arrays["span_wall_end_ns"][row_index, span_index]),
                    "first_cpu_start_ns": int(arrays["span_cpu_start_ns"][row_index, span_index]),
                    "last_cpu_end_ns": int(arrays["span_cpu_end_ns"][row_index, span_index]),
                }
                for span_index, name in enumerate(SPAN_NAMES)
            }
            rows.append(row)
        return rows

    def _require_active(self) -> int:
        if self._failed:
            raise TraceOverflowError("recorder is invalid after an earlier overflow")
        if self._active < 0:
            raise TraceError("no active tick")
        if self._ticks[self._active]["thread_id"] != threading.get_ident():
            raise TraceError(
                "tick trace crossed threads; begin/end and spans need one tick-owning thread"
            )
        return self._active

    def _is_active_on_current_thread(self) -> bool:
        """Return whether this recorder owns a tick on this calling thread."""
        return (
            not self._failed
            and self._active >= 0
            and self._ticks[self._active]["thread_id"] == threading.get_ident()
        )

    def _in_step_internal_context(self) -> bool:
        """Whether a step-internal hook is nested under the serving update path."""
        if not self._is_active_on_current_thread():
            return False
        return any(
            self._stack[depth, 0] in (_SPAN_INDEX["session_step"], _SPAN_INDEX["game_update"])
            for depth in range(self._stack_depth)
        )

    def _install_gc_callback(self) -> None:
        if self._gc_callback_installed:
            raise TraceError("GC callback is already installed")
        gc.callbacks.append(self._gc_callback)
        self._gc_callback_installed = True

    def _remove_gc_callback(self) -> None:
        if not self._gc_callback_installed:
            return
        try:
            gc.callbacks.remove(self._gc_callback)
        except ValueError:
            pass
        self._gc_callback_installed = False

    def _enter_span(self, name: str) -> None:
        row = self._require_active()
        if self._stack_depth >= len(self._stack):
            self._failed = True
            raise TraceOverflowError("trace nesting capacity exhausted; timing panel is invalid")
        wall_now, cpu_now = int(self._wall_clock_ns()), int(self._cpu_clock_ns())
        entry = self._stack[self._stack_depth]
        entry[:] = (_SPAN_INDEX[name], wall_now, cpu_now, 0, 0)
        self._stack_depth += 1
        if self._span_calls[row, _SPAN_INDEX[name]] == 0:
            self._span_wall_start_ns[row, _SPAN_INDEX[name]] = wall_now
            self._span_cpu_start_ns[row, _SPAN_INDEX[name]] = cpu_now
        self._span_calls[row, _SPAN_INDEX[name]] += 1

    def _exit_span(self, name: str) -> None:
        row = self._require_active()
        if not self._stack_depth or self._stack[self._stack_depth - 1, 0] != _SPAN_INDEX[name]:
            raise TraceError(f"trace span nesting mismatch at {name!r}")
        self._stack_depth -= 1
        span, wall_started, cpu_started, child_wall, child_cpu = self._stack[self._stack_depth]
        wall_elapsed = max(0, int(self._wall_clock_ns()) - wall_started)
        cpu_elapsed = max(0, int(self._cpu_clock_ns()) - cpu_started)
        # Multiple hooks intentionally share a coarse group (for example,
        # check_food_consumption -> food_manager.consume_at).  Count all calls,
        # but only the outermost same-group invocation contributes inclusive
        # wall/CPU so grouped timing cannot exceed its containing tick.
        same_group_ancestor = any(
            self._stack[depth, 0] == span for depth in range(self._stack_depth)
        )
        if not same_group_ancestor:
            self._wall_inclusive[row, span] += wall_elapsed
            self._cpu_inclusive[row, span] += cpu_elapsed
        # Summing every node's exclusive residual remains an honest partition
        # of the coarse group's outer interval, including same-group leaves.
        self._wall_exclusive[row, span] += max(0, wall_elapsed - child_wall)
        self._cpu_exclusive[row, span] += max(0, cpu_elapsed - child_cpu)
        self._span_wall_end_ns[row, span] = wall_started + wall_elapsed
        self._span_cpu_end_ns[row, span] = cpu_started + cpu_elapsed
        if self._stack_depth:
            self._stack[self._stack_depth - 1, 3] += wall_elapsed
            self._stack[self._stack_depth - 1, 4] += cpu_elapsed

    def _gc_callback(self, phase: str, info: Mapping[str, Any]) -> None:
        if self._active < 0 or self._failed:
            return
        if self._ticks[self._active]["thread_id"] != threading.get_ident():
            return
        generation = int(info.get("generation", -1))
        if generation not in (0, 1, 2):
            return
        now = int(self._wall_clock_ns())
        row = self._active
        if phase == "start":
            self._ticks[row]["gc_start"][generation] += 1
            self._gc_open[generation] = now
            self._gc_open_active[generation] = True
        elif phase == "stop":
            self._ticks[row]["gc_stop"][generation] += 1
            started = int(self._gc_open[generation]) if self._gc_open_active[generation] else now
            self._gc_open_active[generation] = False
            self._ticks[row]["gc_overlap_ns"][generation] += max(0, now - started)


def _percentiles(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    data = np.asarray(values, dtype=np.int64)
    return {
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": int(data.max()),
    }


def _bands(
    values: Sequence[Mapping[str, Any]], field: str, measure: str
) -> dict[str, dict[str, float | int]]:
    bands: dict[str, list[int]] = {}
    for row in values:
        value = int(row.get(field, -1))
        if value < 0:
            continue
        label = (
            f"{(value // 100) * 100}-{(value // 100) * 100 + 99}"
            if field == "game_age"
            else f"{(value // 50) * 50}-{(value // 50) * 50 + 49}"
        )
        bands.setdefault(label, []).append(int(row[measure]))
    return {label: _percentiles(samples) for label, samples in sorted(bands.items())}


def summarize_breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize span and GC accounting shared by all probe transport arms.

    This deliberately does not inspect tick cadence or clocks, so production can
    use it after combining its independently-clocked step and snapshot rows.
    Callers that accept external JSON should translate ``ValueError`` into their
    own invalid-result protocol.
    """
    span_summary = {
        name: {
            "wall_ns": _percentiles(
                [int(row.get("spans", {}).get(name, {}).get("wall_ns", 0)) for row in rows]
            ),
            "wall_exclusive_ns": _percentiles(
                [
                    int(row.get("spans", {}).get(name, {}).get("wall_exclusive_ns", 0))
                    for row in rows
                ]
            ),
            "thread_cpu_ns": _percentiles(
                [int(row.get("spans", {}).get(name, {}).get("cpu_ns", 0)) for row in rows]
            ),
            "thread_cpu_exclusive_ns": _percentiles(
                [int(row.get("spans", {}).get(name, {}).get("cpu_exclusive_ns", 0)) for row in rows]
            ),
        }
        for name in SPAN_NAMES
    }
    gc_overlap: list[int] = []
    gc_starts = [0, 0, 0]
    gc_stops = [0, 0, 0]
    for source in rows:
        overlap = source.get("gc_overlap_ns", (0, 0, 0))
        starts = source.get("gc_start", (0, 0, 0))
        stops = source.get("gc_stop", (0, 0, 0))
        if any(
            not isinstance(values, (list, tuple))
            or len(values) != 3
            or any(type(value) is not int or value < 0 for value in values)
            for values in (overlap, starts, stops)
        ):
            raise ValueError("invalid gc counters")
        gc_overlap.append(sum(overlap))
        for generation in range(3):
            gc_starts[generation] += starts[generation]
            gc_stops[generation] += stops[generation]
    return {
        "span_summary": span_summary,
        "gc": {
            "overlap_ns": _percentiles(gc_overlap),
            "start_counts": gc_starts,
            "stop_counts": gc_stops,
        },
    }


def reduce_trace(
    rows: Sequence[Mapping[str, Any]],
    expected_ticks: int,
    fps: int = FPS_DEFAULT,
    *,
    require_update: bool = True,
) -> dict[str, Any]:
    """Reduce a complete trace without changing its pass/fail rules.

    A tick ending exactly at the next absolute target is an overrun, by design.
    ``INCOMPLETE`` is reserved for a wrong row count; malformed timestamps and
    identities are ``INVALID``.  Timing PASS requires both zero overruns and an
    update p95 no greater than 11.688 ms.
    """
    if (
        not isinstance(expected_ticks, int)
        or expected_ticks < 0
        or not isinstance(fps, int)
        or fps <= 0
    ):
        raise ValueError("expected_ticks must be non-negative and fps must be positive")
    if len(rows) != expected_ticks:
        return {
            "verdict": "INCOMPLETE",
            "expected_ticks": expected_ticks,
            "observed_ticks": len(rows),
        }
    required = {
        "tick",
        "target_ns",
        "wake_ns",
        "start_ns",
        "end_ns",
        "start_cpu_ns",
        "end_cpu_ns",
    }
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if not required <= set(source):
            return {"verdict": "INVALID", "reason": "missing required tick field", "row": index}
        try:
            row = {name: int(source[name]) for name in required}
        except (TypeError, ValueError):
            return {"verdict": "INVALID", "reason": "non-integer tick field", "row": index}
        if (
            row["tick"] != index
            or row["start_ns"] < row["wake_ns"]
            or row["end_ns"] < row["start_ns"]
            or row["end_cpu_ns"] < row["start_cpu_ns"]
            or (
                index
                and row["target_ns"] - normalized[0]["target_ns"] != index * 1_000_000_000 // fps
            )
        ):
            return {"verdict": "INVALID", "reason": "invalid tick ordering", "row": index}
        spans = source.get("spans")
        if not isinstance(spans, Mapping):
            return {"verdict": "INVALID", "reason": "missing spans", "row": index}
        tick_wall_ns = row["end_ns"] - row["start_ns"]
        tick_cpu_ns = row["end_cpu_ns"] - row["start_cpu_ns"]
        required_span_fields = {
            "wall_ns",
            "wall_exclusive_ns",
            "cpu_ns",
            "cpu_exclusive_ns",
            "calls",
            "first_wall_start_ns",
            "last_wall_end_ns",
            "first_cpu_start_ns",
            "last_cpu_end_ns",
        }
        for span_name, span in spans.items():
            if not isinstance(span, Mapping):
                return {"verdict": "INVALID", "reason": "invalid span", "row": index}
            if not required_span_fields <= set(span):
                return {"verdict": "INVALID", "reason": "incomplete span", "row": index}
            for field in required_span_fields:
                value = span[field]
                if type(value) is not int or value < 0:
                    return {"verdict": "INVALID", "reason": "invalid span duration", "row": index}
            if (
                span["wall_exclusive_ns"] > span["wall_ns"]
                or span["cpu_exclusive_ns"] > span["cpu_ns"]
                or span["wall_ns"] > tick_wall_ns
                or span["cpu_ns"] > tick_cpu_ns
            ):
                return {"verdict": "INVALID", "reason": "span exceeds tick envelope", "row": index}
            if span["calls"] == 0 and any(
                span[field] for field in required_span_fields - {"calls"}
            ):
                return {"verdict": "INVALID", "reason": "empty span has timing", "row": index}
            if span["calls"]:
                if (
                    not row["start_ns"]
                    <= span["first_wall_start_ns"]
                    <= span["last_wall_end_ns"]
                    <= row["end_ns"]
                    or not row["start_cpu_ns"]
                    <= span["first_cpu_start_ns"]
                    <= span["last_cpu_end_ns"]
                    <= row["end_cpu_ns"]
                ):
                    return {
                        "verdict": "INVALID",
                        "reason": "span timestamps outside tick",
                        "row": index,
                    }
        update_span = spans.get("game_update")
        if require_update and (
            not isinstance(update_span, Mapping)
            or update_span.get("calls") != 1
            or type(update_span.get("wall_ns")) is not int
            or int(update_span["wall_ns"]) < 0
        ):
            return {"verdict": "INVALID", "reason": "missing exact game_update span", "row": index}
        normalized.append(row)
    update_ns = [int(row.get("spans", {}).get("game_update", {}).get("wall_ns", 0)) for row in rows]
    tick_ns = [row["end_ns"] - row["start_ns"] for row in normalized]
    thread_cpu_ns = [row["end_cpu_ns"] - row["start_cpu_ns"] for row in normalized]
    wake_lateness = [row["wake_ns"] - row["target_ns"] for row in normalized]
    epoch = normalized[0]["target_ns"] if normalized else 0
    next_targets = [epoch + (row["tick"] + 1) * 1_000_000_000 // fps for row in normalized]
    overrun_severity = [
        max(0, row["end_ns"] - next_target) for row, next_target in zip(normalized, next_targets)
    ]
    overruns = [
        row["tick"]
        for row, next_target in zip(normalized, next_targets)
        if row["end_ns"] >= next_target
    ]
    longest, current = 0, 0
    for row, next_target in zip(normalized, next_targets):
        if row["end_ns"] >= next_target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    update_summary = _percentiles(update_ns)
    try:
        breakdowns = summarize_breakdowns(rows)
    except ValueError:
        return {"verdict": "INVALID", "reason": "invalid gc counters"}
    timing_pass = not overruns and update_summary["p95"] <= UPDATE_P95_LIMIT_NS
    enriched = [dict(row, tick_ns=tick_ns[index]) for index, row in enumerate(rows)]
    return {
        "verdict": "PASS" if timing_pass else "FAIL",
        "expected_ticks": expected_ticks,
        "observed_ticks": len(rows),
        "fps": fps,
        "update_p95_limit_ns": UPDATE_P95_LIMIT_NS,
        "timing_pass": timing_pass,
        "absolute_overruns": len(overruns),
        "overrun_ticks": overruns,
        "overrun_severity_ns": _percentiles(overrun_severity),
        "longest_consecutive_overruns": longest,
        "update_ns": update_summary,
        "tick_ns": _percentiles(tick_ns),
        "tick_thread_cpu_ns": _percentiles(thread_cpu_ns),
        "wall_minus_thread_cpu_ns": _percentiles(
            [wall - cpu for wall, cpu in zip(tick_ns, thread_cpu_ns)]
        ),
        "wake_lateness_ns": _percentiles(wake_lateness),
        "late_wake_count": sum(value > 0 for value in wake_lateness),
        "late_wake_lateness_ns": _percentiles([value for value in wake_lateness if value > 0]),
        **breakdowns,
        "by_game_age": _bands(enriched, "game_age", "tick_ns"),
        "by_hero_length": _bands(enriched, "hero_length", "tick_ns"),
    }
