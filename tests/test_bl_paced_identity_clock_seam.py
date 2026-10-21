"""Round-09 Z-2: distinct publication/consumption fences under a fake clock."""

from dataclasses import asdict

import pytest

from research.bl_paced_identity.controller import PacedController
from research.bl_paced_identity.events import EventValidationError
from research.bl_paced_identity.fixtures import (
    FixtureClock,
    fixture_identity,
    fixture_request,
    fixture_result,
    request_context,
)
from research.bl_paced_identity.record import Journal, read_journal
from research.bl_paced_identity.replay import replay_events


@pytest.mark.parametrize(
    "case,visible_ns,consume_ns,frame,accepted,reason",
    [
        ("on-time", 79, 100, 5, True, "delivered"),
        ("publication-late", 80, 100, 5, False, "cutoff"),
        ("consumption-late", 79, 110, 5, False, "cutoff"),
        ("wrong-frame", 79, 100, 6, False, "stale"),
    ],
)
def test_two_cutoff_four_case_dispositions_replay(
    tmp_path, case, visible_ns, consume_ns, frame, accepted, reason
):
    # T_r=100, tick=10: D_pub=80 and D_cons=110. Worker finishes
    # before publication in every case; IPC visibility is what matters.
    clock = FixtureClock(1)
    identity = fixture_identity()
    journal = Journal(tmp_path / case, asdict(identity), {"synthetic": True})
    controller = PacedController(identity, journal, clock=clock, synthetic=True)
    request = fixture_request(publish_cutoff_ns=80, consume_cutoff_ns=110)
    assert controller.submit(request, {"root": "fixture"})
    clock.now = 70
    controller.worker_complete(request.request_id, 70)
    clock.now = visible_ns
    controller.publish(request.request_id, fixture_result(request), 70)
    clock.now = 100
    context = {**request_context(request), "frame": frame}
    action = controller.begin_poll(context, 0, [True] * 6)
    if action is None:
        controller.validation_complete(request.request_id, True)
        clock.now = consume_ns
        action = controller.consume(request.request_id)
    assert action == (2 if accepted else 0)
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    disposition = next(event.data for event in events if event.kind == "decision")
    assert disposition["accepted"] is accepted
    assert disposition["reason"] == reason
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"


@pytest.mark.parametrize("publication,consumption", [(80, 80), (81, 80), (-1, 80)])
def test_invalid_cutoff_order_rejected(publication, consumption):
    with pytest.raises(EventValidationError):
        fixture_request(publish_cutoff_ns=publication, consume_cutoff_ns=consumption)


def test_validation_can_cross_publication_but_cannot_cross_consumption(tmp_path):
    clock = FixtureClock(1)
    identity = fixture_identity()
    journal = Journal(tmp_path / "validation", asdict(identity), {"synthetic": True})
    controller = PacedController(identity, journal, clock=clock, synthetic=True)
    request = fixture_request(publish_cutoff_ns=80, consume_cutoff_ns=110)
    controller.submit(request, {"root": "fixture"})
    clock.now = 79
    controller.publish(request.request_id, fixture_result(request), 70)
    clock.now = 100
    assert controller.begin_poll(request_context(request), 0, [True] * 6) is None
    clock.now = 110
    controller.validation_complete(request.request_id, True)
    assert controller.consume(request.request_id) == 0
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"
