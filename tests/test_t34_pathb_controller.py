"""Unit tests for the serving-independent T34 deadline controller."""

from __future__ import annotations

from typing import Any, Optional

from web.backend.pathb_controller import DeadlineController, MAX_RECENT_DECISIONS


class Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class Future:
    def __init__(self, value: Any = None, done: bool = True, error: Optional[Exception] = None) -> None:
        self.value, self.ready, self.error, self.cancelled = value, done, error, False

    def done(self) -> bool:
        return self.ready

    def result(self) -> Any:
        if self.error:
            raise self.error
        return self.value

    def cancel(self) -> bool:
        self.cancelled = True
        return True


class Executor:
    def __init__(self) -> None:
        self.future = Future()
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, *_args: Any) -> Future:
        return self.future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def controller(clock: Clock, executor: Executor) -> DeadlineController:
    return DeadlineController(lambda _: {}, clock=clock, executor=executor)


def result(action: int = 2, completed: float = 100.1, bare: int = 1, mask: list[bool] | None = None) -> dict:
    return {"proposed_action": action, "bare_action": bare, "mask": mask or [True, True, True], "_completion_monotonic": completed}


def test_applies_only_a_timely_matching_legal_result() -> None:
    clock, executor = Clock(), Executor()
    executor.future.value = result()
    subject = controller(clock, executor)
    assert subject.submit(b"snapshot", ("s", 8, 2, "sha", "fp"), 101.0)
    action, report = subject.poll(("s", 8, 2, "sha", "fp"), 1, [True, True, True])
    assert action == 2
    assert report["applied"] and report["executed_action"] == 2
    assert subject.state()["active"] is False


def test_busy_and_oversize_are_rejected_without_replacing_request() -> None:
    clock, executor = Clock(), Executor()
    executor.future.ready = False
    subject = controller(clock, executor)
    assert subject.submit(b"x", (1,), 101)
    assert not subject.submit(b"y", (2,), 101)
    assert not subject.submit(b"x" * (64 * 1024 * 1024 + 1), (3,), 101)
    assert subject.state()["active"]


def test_overdue_computation_stays_active_until_it_is_collected() -> None:
    clock, executor = Clock(102), Executor()
    executor.future.ready = False
    subject = controller(clock, executor)
    subject.submit(b"x", (1,), 101)
    action, report = subject.poll((1,), 1, [True, True])
    assert action == 1 and report["fallback_reason"] == "deadline_computing"
    assert subject.state()["active"]
    executor.future.ready, executor.future.value = True, result(action=0, completed=100.5, bare=1, mask=[True, True])
    action, report = subject.poll((1,), 1, [True, True])
    assert action == 1 and report["deadline_miss"]
    assert not subject.state()["active"]


def test_identity_or_action_context_mismatch_discards_completed_result() -> None:
    clock, executor = Clock(), Executor()
    executor.future.value = result()
    subject = controller(clock, executor)
    subject.submit(b"x", (1,), 101)
    action, report = subject.poll((2,), 1, [True, True, True])
    assert action == 1 and report["stale_discarded"]

    executor.future = Future(result(bare=0))
    subject.submit(b"x", (1,), 101)
    action, report = subject.poll((1,), 1, [True, True, True])
    assert action == 1 and report["fallback_reason"] == "action_context_mismatch"


def test_validator_guards_the_live_pre_action_world() -> None:
    clock, executor = Clock(), Executor()
    executor.future.value = result()
    subject = controller(clock, executor)
    subject.submit(b"x", (1,), 101)
    action, report = subject.poll((1,), 1, [True, True, True], validate=lambda _: False)
    assert action == 1
    assert report["stale_discarded"]
    assert report["fallback_reason"] == "world_mismatch"


def test_invalid_worker_response_and_close_are_safe_and_nonblocking() -> None:
    clock, executor = Clock(), Executor()
    executor.future.value = result(action=8)
    subject = controller(clock, executor)
    subject.submit(b"x", (1,), 101)
    action, report = subject.poll((1,), 1, [True, True, True])
    assert action == 1 and report["fallback_reason"] == "invalid_action"
    subject.close()
    assert executor.shutdown_calls == [(False, True)]
    assert not subject.submit(b"x", (1,), 101)


def test_state_keeps_a_bounded_audit_ring_and_all_report_fields() -> None:
    clock, executor = Clock(), Executor()
    subject = controller(clock, executor)
    for _ in range(MAX_RECENT_DECISIONS + 10):
        _, report = subject.poll((1,), 1, [True, True])
        assert set(report) == {"requested", "applied", "accepted", "stale_discarded", "deadline_miss", "fallback_reason", "proposed_action", "executed_action"}
    state = subject.state()
    assert state["label"] == "assisted hero"
    assert len(state["recent"]) == MAX_RECENT_DECISIONS
