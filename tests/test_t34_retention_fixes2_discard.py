"""Host-pause discard fences for the T34 Path B assistance seam."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import research.t34_pathb.controller_v2 as controller_module
from research.t34_pathb.controller_v2 import PathBControllerV2, SessionIdentity


class FakeExecutor:
    """Manual futures keep worker completion timing under test control."""

    def __init__(self) -> None:
        self.futures: list[Future[Any]] = []

    def submit(self, _function: Any, *_args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        self.futures.append(future)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait, cancel_futures


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
        del refork, confirmations
        return Request(frame, life, checkpoint)

    def maintain(self, _request: Request) -> Any:
        raise AssertionError("FakeExecutor controls completion")

    def run_search(self, _request: Any) -> Any:
        raise AssertionError("FakeExecutor controls completion")


def identity(frame: int) -> SessionIdentity:
    return SessionIdentity("pause-row", frame, 0, "b8-s3")


def maintenance(source: SessionIdentity, request: Any) -> Any:
    return SimpleNamespace(
        source_frame=source.frame,
        source_life=source.life,
        source_checkpoint=source.checkpoint,
        source_digest="d0",
        shadow_id=9,
        confirmed=True,
        search_request=request,
        expected_digests=(),
    )


def test_host_pause_discards_running_request_and_late_completion_cannot_publish() -> None:
    executor = FakeExecutor()
    controller = PathBControllerV2(FakeEngine(), executor=executor)
    controller.after_step(object(), identity(0), digest="d0")

    assert controller.discard_pending_assistance() is True
    state = controller.state()
    assert state["counts"]["fallback"] == 1
    assert state["fallback_reasons"]["host_pause"] == 1
    # The same still-running worker was already discarded; a second pause may
    # not charge the same request again.
    assert controller.discard_pending_assistance() is False
    assert controller.state()["counts"]["fallback"] == 1

    executor.futures[0].set_result(
        maintenance(identity(0), SimpleNamespace(shadow_id=9, source_digest="d0"))
    )
    action, report = controller.decide(identity(1), 1, [True] * 6)

    assert action == 1
    assert report["fallback_reason"] == "stale_maintenance_result"
    assert controller.scheduled_frame is None
    assert controller.state()["counts"]["stale_discarded"] == 1


def test_host_pause_discards_ready_action_before_next_row_decision() -> None:
    controller = PathBControllerV2(FakeEngine(), executor=FakeExecutor())
    binding = controller._binding(identity(0), "d0")
    controller._last_confirmed = SimpleNamespace(
        shadow_id=9, epoch=binding.epoch, identity=identity(3)
    )
    controller._scheduled = controller_module._ScheduledAction(binding, 3, 2, "rescue")

    assert controller.discard_pending_assistance() is True
    action, report = controller.decide(identity(3), 1, [True] * 6)

    assert action == 1
    assert report["fallback_reason"] == "bare_no_scheduled_action"
    assert controller.scheduled_frame is None
    # One row-level host-pause fallback plus the next decision's bare fallback.
    assert controller.state()["counts"]["fallback"] == 2
    assert controller.state()["fallback_reasons"]["host_pause"] == 1


def test_idle_host_pause_does_not_invent_a_fallback() -> None:
    controller = PathBControllerV2(FakeEngine(), executor=FakeExecutor())

    assert controller.discard_pending_assistance() is False
    assert controller.state()["counts"]["fallback"] == 0
    assert controller.state()["fallback_reasons"] == {}
