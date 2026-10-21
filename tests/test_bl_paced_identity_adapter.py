"""Focused deterministic contracts for the frozen paced-root request adapter."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from research.bl_paced_identity.adapter import (
    CandidateRequestAdapter,
    FrozenTargetRoot,
    TickSchedule,
)
from research.bl_paced_identity.controller import PacedController
from research.bl_paced_identity.events import CandidateIdentity, EventValidationError, sha256
from research.bl_paced_identity.record import Journal, read_journal
from research.bl_paced_identity.replay import replay_events

SHA = "a" * 64


class _Controller:
    def __init__(self, now: int) -> None:
        self.now = now
        self.calls: list[tuple[str, object]] = []

    def progress(self, value: dict[str, object]) -> None:
        self.calls.append(("progress", value))

    def read_clock(self, label: str) -> int:
        self.calls.append(("clock", label))
        return self.now


def _context(frame: int = 10) -> dict[str, object]:
    return {
        "session": "fixture-session",
        "life": 3,
        "frame": frame,
        "checkpoint_sha": SHA,
        "shadow_epoch": 7,
    }


def _root(
    target: int = 15, *, horizon: int = 32, root: dict[str, object] | None = None
) -> FrozenTargetRoot:
    return FrozenTargetRoot(
        session="fixture-session",
        life=3,
        source_frame=10,
        target_frame=target,
        checkpoint_sha=SHA,
        shadow_epoch=7,
        horizon_checkpoint=horizon,
        root={"frame": target, "state": "reachable"} if root is None else root,
        planner_input={"bare_action": 1},
    )


def test_adapter_selects_earliest_future_five_decision_root_and_binds_two_cutoffs() -> None:
    adapter = CandidateRequestAdapter(TickSchedule(epoch_ns=101, frames_per_second=3))
    controller = _Controller(now=777)
    prepared = adapter.prepare(
        controller,
        context=_context(),
        roots=[_root(20, horizon=300), _root(15, horizon=32)],
        progress={"offered": 2},
    )

    assert prepared is not None
    assert prepared.observed_ns == 777
    assert controller.calls == [("progress", {"offered": 2}), ("clock", "candidate_root_selection")]
    assert prepared.request.target_frame == 15
    assert prepared.request.publish_cutoff_ns == adapter.schedule.at_frame(13)
    assert prepared.request.consume_cutoff_ns == adapter.schedule.at_frame(16)
    assert prepared.request.consume_cutoff_ns > prepared.request.publish_cutoff_ns
    assert prepared.request.root_digest == sha256({"frame": 15, "state": "reachable"})
    assert prepared.request.payload_sha == sha256(prepared.input_payload)
    assert asdict(prepared.request)["request_id"] == sha256(
        {
            "checkpoint_sha": SHA,
            "life": 3,
            "payload_sha": prepared.request.payload_sha,
            "root_digest": prepared.request.root_digest,
            "session": "fixture-session",
            "shadow_epoch": 7,
            "source_frame": 10,
            "target_frame": 15,
        }
    )


def test_adapter_freezes_supplied_root_and_filters_ineligible_roots() -> None:
    supplied = {"frame": 15, "state": {"value": 1}}
    frozen = _root(root=supplied)
    supplied["state"]["value"] = 99
    adapter = CandidateRequestAdapter(TickSchedule(epoch_ns=0, frames_per_second=60))
    prepared = adapter.prepare(
        _Controller(now=1),
        context=_context(),
        roots=[
            _root(16),  # Not at the offered five-decision cadence.
            _root(445),  # Exceeds the 432-decision lead cap.
            frozen,
        ],
        progress={"worker": {"ready": True}},
    )

    assert prepared is not None
    assert prepared.request.target_frame == 15
    assert prepared.input_payload["root"] == {"frame": 15, "state": {"value": 1}}
    returned = frozen.root
    returned["state"]["value"] = 88
    assert frozen.root == {"frame": 15, "state": {"value": 1}}
    with pytest.raises(AttributeError):
        frozen.root_digest = "b" * 64
    with pytest.raises(AttributeError):
        frozen._root_bytes = b"{}\n"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del frozen._sealed
    with pytest.raises(AttributeError):
        del frozen.target_frame
    payload = prepared.input_payload
    payload["root"]["state"]["value"] = 77
    assert prepared.input_payload["root"] == {"frame": 15, "state": {"value": 1}}


def test_adapter_declines_roots_with_wrong_context_or_missing_horizon_contract() -> None:
    adapter = CandidateRequestAdapter(TickSchedule(epoch_ns=0, frames_per_second=30))
    wrong_epoch = FrozenTargetRoot(
        session="fixture-session",
        life=3,
        source_frame=10,
        target_frame=15,
        checkpoint_sha=SHA,
        shadow_epoch=8,
        horizon_checkpoint=96,
        root={"state": "wrong-epoch"},
        planner_input={},
    )

    assert (
        adapter.prepare(
            _Controller(now=1), context=_context(), roots=[wrong_epoch], progress={"offered": 1}
        )
        is None
    )
    with pytest.raises(EventValidationError, match="horizon_checkpoint"):
        _root(horizon=True)  # type: ignore[arg-type]


def test_adapter_skips_expired_root_and_uses_later_root_at_publish_boundary() -> None:
    adapter = CandidateRequestAdapter(TickSchedule(epoch_ns=0, frames_per_second=1_000_000_000))
    # target 15 has Dpub=13; equality is expired. target 20 has Dpub=18.
    prepared = adapter.prepare(
        _Controller(now=13),
        context=_context(),
        roots=[_root(15), _root(20)],
        progress={"offered": 2},
    )

    assert prepared is not None
    assert prepared.request.target_frame == 20


def test_adapter_tie_order_is_independent_of_provider_iteration() -> None:
    adapter = CandidateRequestAdapter(TickSchedule(epoch_ns=0, frames_per_second=1))
    first = _root(15, horizon=96)
    second = FrozenTargetRoot(
        session="fixture-session",
        life=3,
        source_frame=10,
        target_frame=15,
        checkpoint_sha=SHA,
        shadow_epoch=7,
        horizon_checkpoint=32,
        root={"frame": 15, "state": "reachable"},
        planner_input={"bare_action": 4},
    )
    left = adapter.prepare(
        _Controller(now=0), context=_context(), roots=[first, second], progress={}
    )
    right = adapter.prepare(
        _Controller(now=0), context=_context(), roots=[second, first], progress={}
    )

    assert left is not None and right is not None
    assert left.request.request_id == right.request.request_id
    assert left.input_payload == right.input_payload


def test_adapter_request_reaches_controller_and_independent_reducer(tmp_path: Path) -> None:
    identity = CandidateIdentity(
        SHA,
        "edf",
        {"checkpoints": [32, 96, 192, 300]},
        "fixture",
        1,
        1,
        {"qos": "fixture"},
        {"warmup": "fixture"},
        {"fallback": "bare"},
        {"recorder": "fixture"},
        {"load": "fixture"},
        SHA,
    )
    controller_clock = _Controller(now=12)
    journal = Journal(tmp_path / "journal", asdict(identity), {"synthetic": True})
    controller = PacedController(
        identity, journal, clock=lambda: controller_clock.now, synthetic=True
    )
    controller.invalidate("fixture", 7)
    prepared = CandidateRequestAdapter(TickSchedule(0, 1_000_000_000)).prepare(
        controller, context=_context(), roots=[_root()], progress={"offered": 1}
    )

    assert prepared is not None and controller.submit(prepared.request, prepared.input_payload)
    controller.worker_complete(prepared.request.request_id, 12)
    controller.publish(
        prepared.request.request_id,
        {"request": asdict(prepared.request), "action": 2, "bare": 0, "mask": [True] * 6},
        12,
    )
    controller_clock.now = 15
    context = {**_context(), "frame": 15, "root_digest": prepared.request.root_digest}
    assert controller.poll(context, 0, [True] * 6, lambda _: True) == 2
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"
