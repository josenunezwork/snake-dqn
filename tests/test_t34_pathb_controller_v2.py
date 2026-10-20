"""No-blocking and identity tests for the Amendment 2 controller boundary."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from research.t34_pathb.controller_v2 import PathBControllerV2, SessionIdentity, _WorkerResult


class FakeExecutor:
    def __init__(self) -> None:
        self.futures: list[Future[Any]] = []
        self.calls: list[tuple[Any, tuple[Any, ...]]] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, function: Any, *args: Any) -> Future[Any]:
        self.calls.append((function, args))
        future: Future[Any] = Future()
        self.futures.append(future)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class ImmediateTimeoutFuture:
    def done(self) -> bool:
        return False

    def result(self, timeout: float | None = None) -> Any:
        del timeout
        from concurrent.futures import TimeoutError

        raise TimeoutError()


@dataclass(frozen=True)
class Request:
    frame: int
    life: int
    checkpoint: str


class FakeEngine:
    depth = 500

    def make_maintenance_request(
        self,
        _snapshot: Any,
        *,
        frame: int,
        life: int,
        checkpoint: str,
        refork: bool,
        confirmations: tuple[tuple[int, int, str], ...],
    ) -> Request:
        self.last_refork = refork
        self.confirmations = confirmations
        return Request(frame, life, checkpoint)

    def maintain(self, _request: Request) -> Any:
        raise AssertionError("worker completion is controlled by FakeExecutor")

    def run_search(self, _request: Any) -> Any:
        raise AssertionError("worker completion is controlled by FakeExecutor")


def ident(
    frame: int, *, session: str = "s", life: int = 0, checkpoint: str = "b8-s3"
) -> SessionIdentity:
    return SessionIdentity(session, frame, life, checkpoint)


def maintenance(
    identity: SessionIdentity,
    *,
    shadow: int = 7,
    request: Any = None,
    digest: str = "d",
    expected: tuple[tuple[int, str], ...] = (),
) -> Any:
    return SimpleNamespace(
        source_frame=identity.frame,
        source_life=identity.life,
        source_checkpoint=identity.checkpoint,
        source_digest=digest,
        shadow_id=shadow,
        confirmed=True,
        search_request=request,
        expected_digests=expected,
    )


def controller() -> tuple[PathBControllerV2, FakeEngine, FakeExecutor]:
    engine, executor = FakeEngine(), FakeExecutor()
    return PathBControllerV2(engine, executor=executor), engine, executor


def ready_search(
    controller: PathBControllerV2,
    executor: FakeExecutor,
    *,
    expected: tuple[tuple[int, str], ...] = (),
) -> None:
    controller.after_step(object(), ident(1), digest="d")
    executor.futures[0].set_result(
        maintenance(
            ident(1), request=SimpleNamespace(shadow_id=7, source_digest="d"), expected=expected
        )
    )
    controller.after_step(object(), ident(2), digest="later")
    assert len(executor.futures) == 2


def test_worker_owns_all_engine_calls_and_tick_never_waits() -> None:
    controller_, engine, executor = controller()
    controller_.after_step(object(), ident(1), digest="d")
    assert executor.calls[0][0].__name__ == "_run_stage"
    assert executor.calls[0][1][0].__self__ is engine
    assert executor.calls[0][1][0].__name__ == "maintain"
    assert controller_.busy

    action, report = controller_.decide(ident(1), 0, [True] * 6)
    assert action == 0
    assert report["fallback_reason"] == "bare_no_scheduled_action"
    # While catch-up is running, no second snapshot or engine operation queues.
    controller_.after_step(object(), ident(2), digest="later")
    assert len(executor.calls) == 1
    assert controller_.state()["counts"]["maintenance_dropped_busy"] == 1


def test_none_snapshot_never_reuses_an_old_snapshot_after_worker_completion() -> None:
    controller_, _engine, executor = controller()
    controller_.after_step(object(), ident(1), digest="d")
    executor.futures[0].set_result(maintenance(ident(1)))
    controller_.after_step(None, ident(2), digest="d2")
    assert len(executor.calls) == 1
    assert controller_.state()["last"]["event"] == "maintenance_deferred_no_snapshot"


def test_worker_cpu_telemetry_updates_when_after_step_collects_completion() -> None:
    controller_, _engine, executor = controller()
    controller_.after_step(object(), ident(1), digest="d")
    executor.futures[0].set_result(_WorkerResult(maintenance(ident(1)), 0.125))
    controller_.after_step(None, ident(2), digest="d2")
    state = controller_.state()
    assert state["worker_cpu_seconds"] == pytest.approx(0.125)
    assert state["shadow_cpu_seconds"] == pytest.approx(0.125)
    assert state["search_cpu_seconds"] == 0.0


def test_future_result_executes_only_after_fresh_confirmed_identity() -> None:
    controller_, _engine, executor = controller()
    ready_search(controller_, executor)
    executor.futures[1].set_result(
        SimpleNamespace(
            status="IMPROVING",
            completed=True,
            shadow_id=7,
            source_digest="d",
            target_frame=5,
            action=2,
            intervention_kind="rescue",
        )
    )
    action, future = controller_.decide(ident(2), 0, [True] * 6)
    assert action == 0
    assert future["proposed_action"] == 2
    assert future["fallback_reason"] == "scheduled_future_action"
    # The next completed maintenance confirms the state immediately before the
    # target; without it the controller must keep taking bare actions.
    controller_.after_step(object(), ident(5), digest="newer")
    executor.futures[2].set_result(maintenance(ident(5), shadow=7, digest="newer"))
    action, applied = controller_.decide(ident(5), 0, [True] * 6)
    assert action == 2
    assert applied["proposed_action"] == applied["executed_action"] == 2
    assert applied["fallback_reason"] is None
    assert applied["intervention_kind"] == "rescue"
    assert controller_.state()["last_confirmed"] is None


def test_late_stale_or_changed_checkpoint_results_are_bare() -> None:
    controller_, _engine, executor = controller()
    ready_search(controller_, executor)
    executor.futures[1].set_result(
        SimpleNamespace(
            status="IMPROVING",
            completed=True,
            shadow_id=7,
            source_digest="d",
            target_frame=2,
            action=2,
        )
    )
    action, late = controller_.decide(ident(2), 0, [True] * 6)
    assert action == 0 and late["late_discarded"]

    controller_, _engine, executor = controller()
    ready_search(controller_, executor)
    executor.futures[1].set_result(
        SimpleNamespace(
            status="IMPROVING",
            completed=True,
            shadow_id=7,
            source_digest="d",
            target_frame=3,
            action=2,
        )
    )
    controller_.decide(ident(2, checkpoint="other"), 0, [True] * 6)
    assert controller_.scheduled_frame is None


def test_logs_are_bounded_and_identity_is_validated() -> None:
    controller_, _engine, executor = controller()
    for frame in range(505):
        controller_.after_step(object(), ident(frame), digest=str(frame))
    assert controller_.state()["pending_confirmations"] == 501
    controller_.close()
    assert executor.shutdown_calls == [(False, True)]
    with pytest.raises(ValueError):
        SessionIdentity("", 0, 0, "b8")
    with pytest.raises(ValueError):
        SessionIdentity("s", -1, 0, "b8")


def test_rejects_non_d500_engine() -> None:
    engine = FakeEngine()
    engine.depth = 300
    with pytest.raises(ValueError, match="D500"):
        PathBControllerV2(engine, executor=FakeExecutor())


def test_intervening_predicted_digest_mismatch_fences_future_action() -> None:
    controller_, _engine, executor = controller()
    ready_search(controller_, executor, expected=((3, "expected"),))
    executor.futures[1].set_result(
        SimpleNamespace(
            status="IMPROVING",
            completed=True,
            shadow_id=7,
            source_digest="d",
            target_frame=4,
            action=2,
            intervention_kind="rescue",
        )
    )
    controller_.decide(ident(2), 0, [True] * 6)
    controller_.after_step(object(), ident(3), digest="mismatch")
    action, report = controller_.decide(ident(4), 0, [True] * 6)
    assert action == 0
    assert report["fallback_reason"] == "bare_no_scheduled_action"


def test_matching_predicted_digest_advances_confirmation_while_search_is_busy() -> None:
    controller_, _engine, executor = controller()
    ready_search(controller_, executor, expected=((4, "match"),))
    executor.futures[1].set_result(
        SimpleNamespace(
            status="IMPROVING",
            completed=True,
            shadow_id=7,
            source_digest="d",
            target_frame=4,
            action=2,
            intervention_kind="rescue",
        )
    )
    controller_.decide(ident(2), 0, [True] * 6)
    controller_.after_step(object(), ident(4), digest="match")
    action, report = controller_.decide(ident(4), 0, [True] * 6)
    assert action == 2
    assert report["intervention_kind"] == "rescue"


def test_worker_request_keeps_root_binding_after_busy_digest_reconciliation() -> None:
    controller_, engine, executor = controller()
    controller_.after_step(object(), ident(0), digest="d0")
    controller_.after_step(None, ident(1), digest="d1")
    controller_.after_step(None, ident(2), digest="d2")
    executor.futures[0].set_result(
        maintenance(
            ident(0),
            request=SimpleNamespace(shadow_id=7, source_digest="d0"),
            digest="d0",
            expected=((1, "d1"), (2, "d2")),
        )
    )
    controller_.after_step(object(), ident(3), digest="d3")
    assert executor.calls[1][1][0].__self__ is engine
    assert executor.calls[1][1][0].__name__ == "run_search"
    # The queued d1/d2 confirmations advanced the watermark, but the search
    # request stays bound to its f0 source digest.
    assert controller_._pending is not None
    assert controller_._pending.identity.frame == 0
    assert controller_._pending.digest == "d0"


def test_delayed_search_request_reads_live_scalar_progress_without_tick_engine_work() -> None:
    controller_, _engine, executor = controller()
    ready_search(controller_, executor)
    request = executor.calls[1][1][1]
    for frame in range(3, 9):
        controller_.after_step(None, ident(frame), digest="d%s" % frame)
    assert request.latest_progress() == (8, 0)
    assert request.source_epoch == 0


def test_unpaced_settle_drains_completed_maintenance_without_world_work() -> None:
    controller_, _engine, executor = controller()
    controller_.after_step(object(), ident(0), digest="d0")
    executor.futures[0].set_result(maintenance(ident(0), digest="d0"))
    assert controller_.settle_between_ticks(ident(0))
    assert not controller_.busy


def test_unpaced_settle_closes_and_propagates_a_cap_failure() -> None:
    controller_, _engine, executor = controller()
    controller_._future = ImmediateTimeoutFuture()
    controller_._pending = controller_._binding(ident(0), "d0")
    controller_._job_kind = "maintenance"

    def fail_cap() -> None:
        raise RuntimeError("cap")

    with pytest.raises(RuntimeError, match="cap"):
        controller_.settle_between_ticks(ident(0), check_cap=fail_cap)
    assert executor.shutdown_calls == [(False, True)]


def test_p7_nonfinite_q_fault_is_preserved_separately_from_invalid_search() -> None:
    controller_, _engine, executor = controller()
    ready_search(controller_, executor)
    executor.futures[1].set_result(
        SimpleNamespace(
            status="FAULT", unconditional_fault="nonfinite_q", target_frame=None, action=None
        )
    )
    action, fault = controller_.decide(ident(2), 0, [True] * 6)
    assert action == 0
    assert fault["unconditional_fault"] == "nonfinite_q"
    assert fault["fallback_reason"] == "unconditional_fault_nonfinite_q"
    assert controller_.state()["unconditional_faults"][0]["fault"] == "nonfinite_q"

    controller_, _engine, executor = controller()
    ready_search(controller_, executor)
    executor.futures[1].set_result(
        SimpleNamespace(status="INVALID", target_frame=None, action=None)
    )
    action, invalid = controller_.decide(ident(2), 0, [True] * 6)
    assert action == 0
    assert invalid["fallback_reason"] == "invalid_search_result"
    assert controller_.state()["unconditional_faults"] == []


def test_unpaced_settle_routes_typed_nonfinite_worker_failure_to_p7_fault_state() -> None:
    from research.t34_pathb.engine_v2 import NonFiniteQ

    controller_, _engine, executor = controller()
    controller_.after_step(object(), ident(0), digest="d0")
    executor.futures[0].set_exception(NonFiniteQ("nan"))
    assert not controller_.settle_between_ticks(ident(0))
    assert controller_.state()["unconditional_faults"][0]["fault"] == "nonfinite_q"


def test_close_wait_true_joins_an_earlier_nonblocking_close() -> None:
    controller_, _engine, executor = controller()
    controller_.close(wait=False)
    controller_.close(wait=True)
    assert executor.shutdown_calls == [(False, True), (True, False)]


def test_real_engine_constructs_without_running_shadow_work() -> None:
    from research.t34_pathb.engine_v2 import PathBSearchEngine

    executor = FakeExecutor()
    controller_ = PathBControllerV2(PathBSearchEngine(), executor=executor)
    assert not controller_.busy
    controller_.close()
