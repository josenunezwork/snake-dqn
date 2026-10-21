"""Focused adversarial tests for the independent paced availability replay."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from research.bl_paced_identity.events import Event, Request, sha256
from research.bl_paced_identity.replay import replay_events


def request(request_id: str = "r1", *, cutoff_ns: int = 100, target_frame: int = 7) -> Request:
    return Request(
        session="session",
        life=2,
        source_frame=3,
        target_frame=target_frame,
        checkpoint_sha="c" * 64,
        shadow_epoch=0,
        root_digest="d" * 64,
        request_id=request_id,
        payload_sha=sha256({}),
        cutoff_ns=cutoff_ns,
    )


def event(seq: int, kind: str, data: dict[str, Any], *, time_ns: int = 1) -> Event:
    return Event(seq=seq, time_ns=time_ns, kind=kind, data=data)


def decision(
    request_id: str | None, action: int, reason: str, accepted: bool = False
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "action": action,
        "reason": reason,
        "accepted": accepted,
        "intervened": accepted and action != 0,
    }


def poll() -> dict[str, Any]:
    return {
        "context": {
            "session": "session",
            "life": 2,
            "frame": 7,
            "checkpoint_sha": "c" * 64,
            "shadow_epoch": 0,
            "root_digest": "d" * 64,
        },
        "bare": 0,
        "mask": [True] * 6,
    }


def test_delivered_action_is_recomputed_from_visible_payload_not_recorded_decision() -> None:
    req = request()
    payload = {"request": asdict(req), "action": 2, "bare": 0, "mask": [True] * 6}
    events = [
        event(0, "submit_attempt", {"request": asdict(req), "payload": {}}),
        event(1, "submit", {"request_id": "r1"}),
        event(
            2,
            "result_visible",
            {"request_id": "r1", "payload": payload, "completed_ns": 10},
            time_ns=10,
        ),
        event(3, "poll", poll(), time_ns=11),
        event(4, "validation_complete", {"request_id": "r1", "valid": True}, time_ns=12),
        event(5, "consume", {"request_id": "r1"}, time_ns=13),
        event(6, "commit", {"request_id": "r1"}, time_ns=13),
        event(7, "decision", decision("r1", 2, "delivered", True), time_ns=13),
        event(8, "invalidation", {"reason": "intervention", "shadow_epoch": 1}, time_ns=13),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["scheduler_verdict"] == "PASS"
    assert report["verdict"] == "INCOMPLETE"
    assert report["counts"]["decisions"] == 1


def test_cutoff_at_exact_nanosecond_falls_back_to_bare() -> None:
    req = request(cutoff_ns=10)
    payload = {"request": asdict(req), "action": 2, "bare": 0, "mask": [True] * 6}
    events = [
        event(0, "submit_attempt", {"request": asdict(req), "payload": {}}),
        event(1, "submit", {"request_id": "r1"}),
        event(
            2,
            "result_visible",
            {"request_id": "r1", "payload": payload, "completed_ns": 9},
            time_ns=9,
        ),
        event(3, "poll", poll(), time_ns=9),
        event(4, "validation_complete", {"request_id": "r1", "valid": True}, time_ns=9),
        event(5, "consume", {"request_id": "r1"}, time_ns=10),
        event(6, "commit", {"request_id": "r1"}, time_ns=10),
        event(7, "decision", decision("r1", 0, "cutoff"), time_ns=10),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["scheduler_verdict"] == "PASS"
    assert report["counts"]["decisions"] == 1


def test_conflicting_duplicate_is_invalid_identity() -> None:
    first = request("same")
    second = request("same", target_frame=8)
    events = [
        event(0, "submit_attempt", {"request": asdict(first), "payload": {}}),
        event(1, "submit", {"request_id": "same"}),
        event(2, "submit_attempt", {"request": asdict(second), "payload": {}}),
        event(3, "fault", {"request_id": "same", "reason": "conflicting_duplicate"}),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["scheduler_verdict"] == "INVALID_IDENTITY"


def test_cancelled_running_request_holds_capacity_until_publication() -> None:
    first, second, third = request("r1"), request("r2"), request("r3")
    events = [
        event(0, "submit_attempt", {"request": asdict(first), "payload": {}}),
        event(1, "submit", {"request_id": "r1"}),
        event(2, "cancel", {"request_id": "r1"}),
        event(3, "cancellation", {"request_id": "r1"}),
        event(4, "submit_attempt", {"request": asdict(second), "payload": {}}),
        event(5, "refusal", {"request_id": "r2", "reason": "saturated"}),
        event(
            6,
            "result_visible",
            {"request_id": "r1", "payload": {}, "completed_ns": 6},
            time_ns=6,
        ),
        event(7, "submit_attempt", {"request": asdict(third), "payload": {}}, time_ns=6),
        event(8, "submit", {"request_id": "r3"}, time_ns=6),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["scheduler_verdict"] == "PASS"


def test_recorded_internal_output_order_is_checked_exactly() -> None:
    req = request()
    events = [
        event(0, "submit_attempt", {"request": asdict(req), "payload": {}}),
        event(1, "refusal", {"request_id": "r1", "reason": "duplicate"}),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["scheduler_verdict"] == "INVALID_IDENTITY"
    assert report["first_divergence"] is not None


def test_stale_epoch_is_rejected_even_when_request_fields_otherwise_match() -> None:
    req = request()
    payload = {"request": asdict(req), "action": 2, "bare": 0, "mask": [True] * 6}
    stale_poll = poll()
    stale_poll["context"]["shadow_epoch"] = 1
    events = [
        event(0, "submit_attempt", {"request": asdict(req), "payload": {}}),
        event(1, "submit", {"request_id": "r1"}),
        event(
            2,
            "result_visible",
            {"request_id": "r1", "payload": payload, "completed_ns": 2},
            time_ns=2,
        ),
        event(3, "poll", stale_poll, time_ns=3),
        event(4, "decision", decision("r1", 0, "stale"), time_ns=3),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["scheduler_verdict"] == "PASS"


def test_backwards_event_time_invalidates_identity() -> None:
    events = [
        event(0, "progress", {"value": {}}, time_ns=2),
        event(1, "progress", {"value": {}}, time_ns=1),
    ]

    report = replay_events({"K": 1, "initial_state": {"synthetic": True}}, events)

    assert report["verdict"] == "INVALID_IDENTITY"
    assert report["first_divergence"]["expected"] == "monotonic time_ns"


class _TamperedSubmissionGame:
    """Independent adapter whose regenerated root differs from the journal."""

    def reset(self, initial_state: dict[str, Any]) -> None:
        del initial_state

    def context(self) -> dict[str, Any]:
        return {}

    def bare(self) -> int:
        return 0

    def mask(self) -> list[bool]:
        return [True] * 6

    def submission(self, now_ns: int) -> dict[str, Any]:
        del now_ns
        changed = asdict(request())
        changed["target_frame"] = 8
        return {"request": changed, "payload": {}}

    def validate(self, payload: dict[str, Any]) -> bool:
        del payload
        return True

    def observe(self, kind: str, time_ns: int, data: dict[str, Any]) -> None:
        del kind, time_ns, data

    def finish_expectations(self) -> dict[str, Any]:
        return {"submissions": [], "dispatches": []}

    def step(self, action: int) -> dict[str, Any]:
        del action
        return {"state_digest": "unused", "mass_sum": 0, "alive_frames": 0, "deaths": 0}


def test_full_replay_rejects_tampered_submission_even_when_scheduler_trace_matches() -> None:
    req = request()
    events = [
        event(0, "submit_attempt", {"request": asdict(req), "payload": {}}),
        event(1, "submit", {"request_id": "r1"}),
    ]

    report = replay_events(
        {"K": 1, "initial_state": {"synthetic": True}},
        events,
        lambda _header: _TamperedSubmissionGame(),
    )

    assert report["verdict"] == "INVALID_IDENTITY"
    assert report["first_divergence"]["event"] == "submit_attempt"


def test_full_replay_detects_omitted_independent_submission_opportunity() -> None:
    class _ExpectedButAbsent(_TamperedSubmissionGame):
        def finish_expectations(self) -> dict[str, Any]:
            return {
                "submissions": [{"request": asdict(request()), "payload": {}}],
                "dispatches": [],
            }

    report = replay_events(
        {"K": 1, "initial_state": {"synthetic": True}},
        [],
        lambda _header: _ExpectedButAbsent(),
    )

    assert report["verdict"] == "INVALID_IDENTITY"


def test_dispatch_recomputes_edf_and_tracks_two_running_workers() -> None:
    later = request("later", cutoff_ns=40)
    earlier = request("earlier", cutoff_ns=30)
    events = [
        event(0, "submit_attempt", {"request": asdict(later), "payload": {}}),
        event(1, "submit", {"request_id": "later"}),
        event(2, "submit_attempt", {"request": asdict(earlier), "payload": {}}),
        event(3, "submit", {"request_id": "earlier"}),
        event(4, "dispatch_attempt", {"request_id": "earlier"}),
        event(5, "dispatch", {"request_id": "earlier"}),
        event(6, "dispatch_attempt", {"request_id": "later"}),
        event(7, "dispatch", {"request_id": "later"}),
        event(8, "worker_complete", {"request_id": "earlier", "completed_ns": 8}, time_ns=8),
        event(
            9,
            "result_visible",
            {"request_id": "earlier", "payload": {}, "completed_ns": 8},
            time_ns=9,
        ),
    ]

    report = replay_events({"identity": {"workers": 2, "max_pending": 3}}, events)

    assert report["scheduler_verdict"] == "PASS"


def test_production_result_visibility_without_dispatch_and_completion_is_invalid() -> None:
    req = request()
    events = [
        event(0, "submit_attempt", {"request": asdict(req), "payload": {}}),
        event(1, "submit", {"request_id": "r1"}),
        event(
            2, "result_visible", {"request_id": "r1", "payload": {}, "completed_ns": 2}, time_ns=2
        ),
        event(3, "fault", {"request_id": "r1", "reason": "untracked_completion"}, time_ns=2),
    ]

    report = replay_events({"identity": {"workers": 1, "max_pending": 1}}, events)

    assert report["verdict"] == "INVALID_IDENTITY"


class _OneTickIndependentGame(_TamperedSubmissionGame):
    """A tiny contract test, never a scored or exported production run."""

    def context(self) -> dict[str, Any]:
        return poll()["context"]

    def finish_expectations(self) -> dict[str, Any]:
        return {"submissions": [], "dispatches": []}

    def step(self, action: int) -> dict[str, Any]:
        assert action == 0
        return {
            "state_digest": "a" * 64,
            "state_digest_schema": "t34-shadow-full-digest/v3",
            "mass_sum": 1,
            "alive_frames": 1,
            "deaths": 0,
        }


def _one_tick_trace() -> list[Event]:
    return [
        event(0, "poll", poll()),
        event(1, "decision", decision(None, 0, "unavailable")),
        event(2, "game_state", _OneTickIndependentGame().step(0)),
    ]


def test_meaningful_omission_rejects_otherwise_complete_bare_game_trace() -> None:
    class ExpectedRequest(_OneTickIndependentGame):
        def finish_expectations(self) -> dict[str, Any]:
            return {
                "submissions": [{"request": asdict(request()), "payload": {}}],
                "dispatches": [],
            }

    report = replay_events(
        {"initial_state": {"synthetic": True}}, _one_tick_trace(), lambda _: ExpectedRequest()
    )
    assert report["verdict"] == "INVALID_IDENTITY"
    assert "submission" in str(report["first_divergence"])


def test_production_contract_can_qualify_but_missing_hooks_and_tiny_state_cannot() -> None:
    header = {
        "identity": {"workers": 1, "max_pending": 1},
        "binding_sha256": "b" * 64,
        "initial_state": {
            "synthetic": False,
            "state_digest_schema": "t34-shadow-full-digest/v3",
            "game_state": {"fixture": 1},
            "opponent_state": {"fixture": 1},
            "rng_state": {"fixture": 1},
            "source_snapshot": {"fixture": 1},
            "replay_contract": {"independent_scheduler": True, "replay_reference_sha256": "a" * 64},
        },
    }
    report = replay_events(header, _one_tick_trace(), lambda _: _OneTickIndependentGame())
    assert report["verdict"] == "PASS"
    assert report["qualification"] == "REPLAY_QUALIFIED"
    header["initial_state"]["synthetic"] = True
    assert (
        replay_events(header, _one_tick_trace(), lambda _: _OneTickIndependentGame())[
            "qualification"
        ]
        == "PRECHECK_ONLY"
    )
    assert replay_events(header, _one_tick_trace(), lambda _: object())["verdict"] == "INCOMPLETE"
