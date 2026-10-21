"""Real coordinator/reducer cross-checks on the complete adversarial fixture set."""

from dataclasses import asdict, replace

import pytest

from research.bl_paced_identity.controller import PacedController
from research.bl_paced_identity.fixtures import (
    FixtureClock,
    fixture_identity,
    fixture_request,
    fixture_result,
    request_context,
    run_adversarial_fixtures,
)
from research.bl_paced_identity.receipt import check_paced, freeze_family, make_receipt
from research.bl_paced_identity.record import Journal, read_journal
from research.bl_paced_identity.replay import replay_events
from research.bl_paced_identity.transport import ProcessTransport


def test_adversarial_interleavings_match_independent_reducer(tmp_path):
    fixtures = run_adversarial_fixtures(tmp_path / "fixtures")
    assert fixtures["fixture_count"] >= 26
    assert all(row["expected"] == row["observed"] for row in fixtures["fixtures"])
    boundary = {row["fixture_id"]: row for row in fixtures["fixtures"]}
    assert boundary["cutoff-99"]["decisions"][0]["accepted"]
    assert not boundary["cutoff-100"]["decisions"][0]["accepted"]
    assert not boundary["validation-crossing-cutoff"]["decisions"][0]["accepted"]
    assert boundary["intervention-root-five-frames-later"]["decisions"][1]["accepted"]


@pytest.mark.parametrize("race", ["cancel", "invalidate"])
def test_validation_raced_by_fence_falls_back_and_replays(tmp_path, race):
    identity = fixture_identity()
    journal = Journal(tmp_path / race, asdict(identity), {"synthetic": True})
    clock = FixtureClock()
    controller = PacedController(identity, journal, clock=clock, synthetic=True)
    request = fixture_request()
    controller.submit(request, {"root": "fixture"})
    controller.publish("r1", fixture_result(request), 1)
    assert controller.begin_poll(request_context(request), 0, [True] * 6) is None
    controller.validation_complete("r1", True)
    if race == "cancel":
        controller.cancel("r1")
    else:
        controller.invalidate("respawn", 1)
    assert controller.consume("r1") == 0
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"


def test_full_worker_envelope_buildable_but_process_launch_needs_owner(tmp_path):
    identity = fixture_identity(4, 8)
    journal = Journal(tmp_path / "multi", asdict(identity), {"synthetic": True})
    controller = PacedController(identity, journal)
    with pytest.raises(ValueError, match="owner"):
        ProcessTransport(controller, lambda *_: {})
    with pytest.raises(ValueError, match="boolean"):
        identity.require_authorized(owner_multi_worker="PENDING")
    for workers in range(1, 5):
        for capacity in range(workers, 9):
            assert fixture_identity(workers, capacity).workers == workers


def test_deadline_and_payload_rejections_and_idempotent_consumption(tmp_path):
    identity = fixture_identity()
    journal = Journal(tmp_path / "payload", asdict(identity), {"synthetic": True})
    controller = PacedController(identity, journal, clock=FixtureClock(), synthetic=True)
    request = fixture_request()
    controller.submit(request, {"root": "fixture"})
    result = fixture_result(request)
    result["request"] = asdict(replace(request, checkpoint_sha="a" * 64))
    controller.publish("r1", result, 1)
    assert controller.poll(request_context(request), 0, [True] * 6, lambda _: True) == 0
    controller.publish("r1", result, 1)
    assert controller.poll(request_context(request), 0, [True] * 6, lambda _: True) == 0
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"
    assert sum(bool(event.data["accepted"]) for event in events if event.kind == "decision") == 0


def test_family_freezes_only_present_nominees_and_reserves_planner_slot():
    family = freeze_family({"bare_R": "a" * 64, "assisted_B8": "b" * 64})
    assert family["holm_thresholds"] == [0.025, 0.05]
    assert family["absent_reserved_slot_excluded"]
    three = freeze_family({"bare_R": "a" * 64, "assisted_B8": "b" * 64, "planner_over_R": "c" * 64})
    assert three["holm_thresholds"] == [0.05 / 3, 0.025, 0.05]
    with pytest.raises(ValueError):
        freeze_family({"bare_R": "a" * 64}, strict_bank_open=True)


def test_synthetic_pass_never_becomes_three_oracle_qualification():
    common = dict(
        identity=asdict(fixture_identity()),
        semantic={"verdict": "PASS", "qualification": "PRECHECK_ONLY"},
        replay={"verdict": "PASS"},
        paced={"verdict": "NOT_RUN"},
        source_hashes={"x": "a" * 64},
        input_hashes={"x": "b" * 64},
        reference_hashes={"x": "c" * 64},
        commands=["fixture"],
        journal_seals=[{"status": "SEALED"}],
        forecast={"units": 0},
    )
    receipt = make_receipt(**common)
    assert receipt["verdict"] == "INCOMPLETE"
    assert receipt["l3_identity_prerequisites"] == "BLOCKED"
    assert receipt["efficacy_unblinding"] == "BLOCKED"
    common["replay"] = {"verdict": "INVALID_IDENTITY", "first_divergence": {"event": 1}}
    assert make_receipt(**common)["verdict"] == "INVALID_IDENTITY"


def test_injected_intervention_cannot_pass_paced_oracle():
    report = check_paced(
        [{"run_id": "injected", "synthetic": True}],
        expected_run_ids=["injected"],
        candidate_sha256="a" * 64,
        frozen_order_sha256="b" * 64,
    )
    assert report["verdict"] == "INCOMPLETE"


def test_recorder_cost_before_commit_cannot_backdate_delivery(tmp_path):
    clock = FixtureClock(1)

    class SlowRecorder(Journal):
        def emit(self, time_ns, kind, **data):
            event = super().emit(time_ns, kind, **data)
            if kind == "consume":
                clock.now = 101
            return event

    identity = fixture_identity()
    journal = SlowRecorder(tmp_path / "slow-recorder", asdict(identity), {"synthetic": True})
    controller = PacedController(identity, journal, clock=clock, synthetic=True)
    request = fixture_request()
    controller.submit(request, {"root": "fixture"})
    controller.publish("r1", fixture_result(request), 1)
    clock.now = 99
    assert controller.poll(request_context(request), 0, [True] * 6, lambda _: True) == 0
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    decision = next(event for event in events if event.kind == "decision")
    assert decision.time_ns >= 100
    assert decision.data["reason"] == "cutoff"
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"


def test_production_queued_cancel_cleanup_releases_k_and_replays(tmp_path):
    identity = fixture_identity()
    journal = Journal(tmp_path / "queue-cleanup", asdict(identity), {"synthetic": False})
    controller = PacedController(identity, journal, clock=FixtureClock())
    request = fixture_request()
    controller.submit(request, {"root": "fixture"})
    controller.cancel("r1")
    controller.worker_error("r1", "transport_closed_before_dispatch")
    assert not controller.pending
    journal.seal()
    header, events, _ = read_journal(journal.directory)
    assert replay_events(header, events)["scheduler_verdict"] == "PASS"
