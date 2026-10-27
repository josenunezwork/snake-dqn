"""Focused unit tests for bounded served Watch timing instrumentation."""

from __future__ import annotations

import gc
import threading

import pytest

from research.bl_served_search.tick_trace import (
    TraceError,
    TraceOverflowError,
    TraceRecorder,
    reduce_trace,
)
from web.backend import serialize


class Clock:
    def __init__(self) -> None:
        self.wall = 0
        self.cpu = 0

    def wall_ns(self) -> int:
        return self.wall

    def cpu_ns(self) -> int:
        return self.cpu

    def advance(self, wall: int, cpu: int | None = None) -> None:
        self.wall += wall
        self.cpu += wall if cpu is None else cpu


def recorder(clock: Clock, capacity: int = 2) -> TraceRecorder:
    return TraceRecorder(capacity, wall_clock_ns=clock.wall_ns, cpu_clock_ns=clock.cpu_ns)


def test_nested_spans_keep_inclusive_and_exclusive_accounting() -> None:
    clock = Clock()
    trace = recorder(clock)
    trace.begin_tick(0, 0, 0)
    with trace.span("game_update"):
        clock.advance(10, 6)
        with trace.span("observation"):
            clock.advance(5, 3)
        clock.advance(7, 4)
    trace.end_tick()
    row = trace.rows()[0]["spans"]
    assert row["game_update"] == {
        "wall_ns": 22,
        "wall_exclusive_ns": 17,
        "cpu_ns": 13,
        "cpu_exclusive_ns": 10,
        "calls": 1,
        "first_wall_start_ns": 0,
        "last_wall_end_ns": 22,
        "first_cpu_start_ns": 0,
        "last_cpu_end_ns": 13,
    }
    assert row["observation"]["wall_ns"] == 5


def test_nested_same_group_counts_calls_without_double_counting_inclusive_time() -> None:
    clock = Clock()
    trace = recorder(clock)
    trace.begin_tick(0, 0, 0)
    with trace.span("game_update"):
        with trace.span("food_spawn_respawn"):
            clock.advance(5)
            with trace.span("food_spawn_respawn"):
                clock.advance(10)
            clock.advance(5)
    trace.end_tick()
    food = trace.rows()[0]["spans"]["food_spawn_respawn"]
    assert food["calls"] == 2
    assert food["wall_ns"] == 20
    assert food["wall_exclusive_ns"] == 20
    assert reduce_trace(trace.rows(), expected_ticks=1)["verdict"] == "PASS"


def test_gc_counts_and_overlap_are_fixed_per_generation() -> None:
    clock = Clock()
    trace = recorder(clock)
    trace.begin_tick(0, 0, 0)
    trace._gc_callback("start", {"generation": 2})
    clock.advance(9)
    trace._gc_callback("stop", {"generation": 2})
    trace.end_tick()
    row = trace.rows()[0]
    assert row["gc_start"] == [0, 0, 1]
    assert row["gc_stop"] == [0, 0, 1]
    assert row["gc_overlap_ns"] == [0, 0, 9]


def test_overflow_fails_closed_and_subsequent_use_refuses() -> None:
    clock = Clock()
    trace = recorder(clock, capacity=1)
    trace.begin_tick(0, 0, 0)
    trace.end_tick()
    with pytest.raises(TraceOverflowError):
        trace.begin_tick(1, 1, 1)
    with pytest.raises(TraceOverflowError):
        trace.begin_tick(1, 1, 1)


def test_mismatched_nested_span_is_rejected() -> None:
    clock = Clock()
    trace = recorder(clock)
    trace.begin_tick(0, 0, 0)
    trace._enter_span("game_update")
    with pytest.raises(TraceError, match="nesting"):
        trace._exit_span("snapshot")


class _Game:
    def __init__(self) -> None:
        self.frame = 8
        self.food = [(1, 2)]
        self.alive_snakes = 1
        self.snakes = []

    def update(self) -> str:
        return "updated"


class _Session:
    def __init__(self) -> None:
        self.game = _Game()
        self.hero_id = 0
        self._lock = threading.RLock()

    def step(self) -> str:
        with self._lock:
            return self.game.update()

    def snapshot(self) -> dict:
        from web.backend.serialize import build_frame

        return build_frame(self)


class _Forward:
    def forward(self, state: object) -> object:
        return state


class _Policy:
    def __init__(self) -> None:
        self.dqn = _Forward()

    def select_action(self, state: object) -> int:
        self.dqn.forward(state)
        return 0


class _Snake:
    def __init__(self, identifier: int, policy: _Policy) -> None:
        self.id = identifier
        self.policy = policy
        self.is_alive = True

    def update(self, *_: object) -> None:
        self.policy.select_action(None)


def test_shared_policy_nested_forward_uses_distinct_stage_names() -> None:
    clock = Clock()
    trace = recorder(clock)
    session = _Session()
    policy = _Policy()
    session.game.snakes = [_Snake(1, policy), _Snake(2, policy)]
    with trace.attach(session):
        trace.begin_tick(0, 0, 0)
        for snake in session.game.snakes:
            snake.update()
        trace.end_tick()
    spans = trace.rows()[0]["spans"]
    assert spans["opponent_action_selection"]["calls"] == 2
    assert spans["opponent_forward"]["calls"] == 2
    assert spans["opponent_forward"]["wall_ns"] == 0
    assert spans["opponent_action_selection"]["wall_ns"] == 0


def test_attach_constructs_hooks_once_and_restores_methods() -> None:
    clock = Clock()
    trace = recorder(clock)
    session = _Session()
    original_step = session.step
    original_update = session.game.update
    original_builder = serialize.build_frame
    serialize.build_frame = lambda value: {"frame": value.game.frame}
    try:
        with trace.attach(session):
            trace.begin_tick(0, 0, 0)
            assert session.step() == "updated"
            assert session.snapshot() == {"frame": 8}
            trace.end_tick()
            spans = trace.rows()[0]["spans"]
            assert spans["session_step"]["calls"] == 1
            assert spans["game_update"]["calls"] == 1
            assert spans["snapshot"]["calls"] == 1
            assert spans["dto"]["calls"] == 1
            assert spans["ipc_lock"]["calls"] == 1
    finally:
        serialize.build_frame = original_builder
    assert session.step.__func__ is original_step.__func__
    assert session.game.update.__func__ is original_update.__func__
    assert isinstance(session._lock, type(threading.RLock()))
    assert trace._gc_callback not in gc.callbacks


def tick(*, end: int, update: int, target: int = 0, wake: int = 0, index: int = 0) -> dict:
    return {
        "tick": index,
        "target_ns": target,
        "wake_ns": wake,
        "start_ns": wake,
        "end_ns": end,
        "start_cpu_ns": 0,
        "end_cpu_ns": end,
        "game_age": index,
        "hero_length": 10,
        "spans": {
            "game_update": {
                "wall_ns": update,
                "wall_exclusive_ns": update,
                "cpu_ns": update,
                "cpu_exclusive_ns": update,
                "calls": 1,
                "first_wall_start_ns": 0,
                "last_wall_end_ns": update,
                "first_cpu_start_ns": 0,
                "last_cpu_end_ns": update,
            }
        },
    }


def test_reducer_enforces_equality_overrun_and_update_bar() -> None:
    interval = 1_000_000_000 // 24
    result = reduce_trace([tick(end=interval, update=11_688_000)], expected_ticks=1)
    assert result["verdict"] == "FAIL"
    assert result["absolute_overruns"] == 1
    passed = reduce_trace([tick(end=interval - 1, update=11_688_000)], expected_ticks=1)
    assert passed["verdict"] == "PASS"
    too_slow = reduce_trace([tick(end=interval - 1, update=11_688_001)], expected_ticks=1)
    assert too_slow["verdict"] == "FAIL"


def test_reducer_uses_rational_absolute_targets_not_floor_frame_intervals() -> None:
    target_one = 1_000_000_000 // 24
    next_target = 2 * 1_000_000_000 // 24
    first = tick(end=target_one - 1, update=1)
    second = tick(end=next_target - 1, update=1, target=target_one, index=1)
    assert reduce_trace([first, second], expected_ticks=2)["verdict"] == "PASS"
    second["end_ns"] = next_target
    assert reduce_trace([first, second], expected_ticks=2)["absolute_overruns"] == 1


def test_reducer_reports_incomplete_and_invalid_rows() -> None:
    assert reduce_trace([], expected_ticks=1)["verdict"] == "INCOMPLETE"
    invalid = reduce_trace([tick(end=1, update=1, index=7)], expected_ticks=1)
    assert invalid["verdict"] == "INVALID"
    missing_update = tick(end=1, update=1)
    missing_update["spans"] = {}
    assert reduce_trace([missing_update], expected_ticks=1)["verdict"] == "INVALID"
    negative = tick(end=1, update=1)
    negative["spans"]["game_update"]["wall_ns"] = -1
    assert reduce_trace([negative], expected_ticks=1)["verdict"] == "INVALID"
