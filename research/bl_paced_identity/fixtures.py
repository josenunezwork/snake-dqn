"""Synthetic interleavings, kept separate from fresh paced-operation evidence."""

from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from .controller import PacedController
from .events import CandidateIdentity, Request, sha256
from .record import Journal, export_json, read_journal
from .replay import replay_events

FIXTURE_CHECKPOINT = "c" * 64


class FixtureClock:
    """Integer clock controlled by test events; it never sleeps or paces a game."""

    def __init__(self, now: int = 1) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def fixture_identity(workers: int = 1, max_pending: int = 1) -> CandidateIdentity:
    return CandidateIdentity(
        planner_sha=sha256({"fixture_planner": "six-action/v1"}),
        scheduler="edf-cutoff-target-request/v1",
        horizon={"checkpoints": [32, 96, 192, 300], "max_lead_frames": 432},
        tie_rule="survival_then_bare_then_Q_then_index",
        workers=workers,
        max_pending=max_pending,
        qos={"nice_min": 10, "math_threads_per_process": 1},
        warmup={"scope": "synthetic_precheck", "qualified": False},
        fallback={"action": "bare", "cutoff": "strict_less_than", "two_tick_margin": True},
        recorder={"schema": "bl-paced-identity-journal/v1", "enabled": True},
        load_envelope={"phase0": "running", "latency_qualification": False},
        checkpoint_sha=FIXTURE_CHECKPOINT,
    )


def fixture_request(rid: str = "r1", **changes: Any) -> Request:
    request = Request(
        session="fixture-session",
        life=0,
        source_frame=0,
        target_frame=5,
        checkpoint_sha=FIXTURE_CHECKPOINT,
        shadow_epoch=0,
        root_digest="d" * 64,
        request_id=rid,
        payload_sha=sha256({"root": "fixture"}),
        cutoff_ns=100,
    )
    return replace(request, **changes)


def request_context(request: Request) -> dict[str, Any]:
    return {
        "session": request.session,
        "life": request.life,
        "frame": request.target_frame,
        "checkpoint_sha": request.checkpoint_sha,
        "shadow_epoch": request.shadow_epoch,
        "root_digest": request.root_digest,
    }


def fixture_result(request: Request, action: int = 2) -> dict[str, Any]:
    return {"request": asdict(request), "action": action, "bare": 0, "mask": [True] * 6}


def run_adversarial_fixtures(directory: Path) -> dict[str, Any]:
    """Enumerate legal two-request orders plus adversarial cutoff/fence fixtures."""
    rows: list[dict[str, Any]] = []

    def run(
        name: str,
        exercise: Callable[[PacedController, FixtureClock], None],
        *,
        workers: int = 1,
        capacity: int = 1,
        expected: str = "PASS",
    ) -> None:
        identity = fixture_identity(workers, capacity)
        journal = Journal(directory / name, asdict(identity), {"synthetic": True}, capacity=1024)
        clock = FixtureClock()
        controller = PacedController(identity, journal, clock=clock, synthetic=True)
        exercise(controller, clock)
        seal = journal.seal()
        header, events, _ = read_journal(journal.directory)
        report = replay_events(header, events)
        decisions = [event.data for event in events if event.kind == "decision"]
        accepted = [event["request_id"] for event in decisions if event["accepted"]]
        if len(set(accepted)) != len(accepted):
            raise AssertionError("request accepted more than once")
        if any(event["action"] != 0 for event in decisions if not event["accepted"]):
            raise AssertionError("rejected result did not use bare fallback")
        if report["scheduler_verdict"] != expected:
            raise AssertionError(f"{name}: {report}")
        rows.append(
            {
                "fixture_id": name,
                "expected": expected,
                "observed": report["scheduler_verdict"],
                "event_count": len(events),
                "seal": seal,
                "action_hash": report["action_hash"],
                "decisions": decisions,
            }
        )

    def boundary(when: int, validation: int | None = None, delivery: int | None = None) -> Any:
        def exercise(controller: PacedController, clock: FixtureClock) -> None:
            request = fixture_request()
            controller.submit(request, {"root": "fixture"})
            clock.now = when if delivery is None else delivery
            controller.worker_complete(request.request_id, min(99, when))
            controller.publish(request.request_id, fixture_result(request), min(99, when))
            clock.now = max(clock.now, when)
            action = controller.begin_poll(request_context(request), 0, [True] * 6)
            if action is None:
                clock.now = max(clock.now, when if validation is None else validation)
                controller.validation_complete(request.request_id, True)
                controller.consume(request.request_id)

        return exercise

    for when in (99, 100, 101):
        run(f"cutoff-{when}", boundary(when))
    run("completed-before-visible-after", boundary(99, delivery=101))
    run("validation-crossing-cutoff", boundary(99, validation=100))

    for field, value in (
        ("session", "stale"),
        ("life", 1),
        ("checkpoint_sha", "e" * 64),
        ("frame", 6),
        ("shadow_epoch", 1),
        ("root_digest", "f" * 64),
    ):

        def stale(
            controller: PacedController, clock: FixtureClock, f: str = field, v: Any = value
        ) -> None:
            request = fixture_request()
            controller.submit(request, {"root": "fixture"})
            controller.publish(request.request_id, fixture_result(request), 1)
            context = request_context(request)
            context[f] = v
            controller.poll(context, 0, [True] * 6, lambda _: True)

        run(f"stale-{field}", stale)

    def duplicate(controller: PacedController, clock: FixtureClock, conflict: bool = False) -> None:
        request = fixture_request()
        controller.submit(request, {"root": "fixture"})
        controller.submit(
            replace(request, target_frame=6) if conflict else request, {"root": "fixture"}
        )

    run("duplicate-identical", duplicate)
    run("duplicate-conflicting", lambda c, t: duplicate(c, t, True), expected="INVALID_IDENTITY")

    def saturation(controller: PacedController, clock: FixtureClock) -> None:
        for index in range(controller.identity.max_pending + 1):
            controller.submit(fixture_request(str(index)), {"root": "fixture"})
        if len(controller.pending) != controller.identity.max_pending:
            raise AssertionError("K accounting failed")
        controller.cancel("0")
        if controller.submit(fixture_request("still-full"), {"root": "fixture"}):
            raise AssertionError("cancelled running work must still occupy K")
        controller.publish("0", fixture_result(fixture_request("0")), 1)
        if not controller.submit(fixture_request("after-publication"), {"root": "fixture"}):
            raise AssertionError("collected cancellation must release K")

    run("saturation-w1-k1", saturation)
    run("saturation-w4-k8", saturation, workers=4, capacity=8)

    def crash(controller: PacedController, clock: FixtureClock) -> None:
        request = fixture_request()
        controller.submit(request, {"root": "fixture"})
        controller.worker_error("r1", "FixtureWorkerCrash")
        controller.poll(request_context(request), 0, [True] * 6, lambda _: True)

    run("worker-crash", crash)

    def intervention(controller: PacedController, clock: FixtureClock) -> None:
        first = fixture_request()
        controller.submit(first, {"root": "fixture"})
        controller.publish("r1", fixture_result(first), 1)
        controller.poll(request_context(first), 0, [True] * 6, lambda _: True)
        second = fixture_request("r2", source_frame=5, target_frame=10, shadow_epoch=1)
        controller.submit(second, {"root": "fixture"})
        controller.publish("r2", fixture_result(second), 1)
        controller.poll(request_context(second), 0, [True] * 6, lambda _: True)

    run("intervention-root-five-frames-later", intervention)
    for reason in ("respawn", "host_pause", "host_resume"):

        def invalidate(controller: PacedController, clock: FixtureClock, why: str = reason) -> None:
            request = fixture_request()
            controller.submit(request, {"root": "fixture"})
            controller.invalidate(why, 1)
            controller.publish("r1", fixture_result(request), 1)
            controller.poll(request_context(request), 0, [True] * 6, lambda _: True)

        run(reason, invalidate)

    # All 3! legal relative orders at equal monotonic time; coordinator seq is decisive.
    for order in itertools.permutations(("publish-a", "publish-b", "invalidate")):

        def interleave(
            controller: PacedController, clock: FixtureClock, sequence: tuple[str, ...] = order
        ) -> None:
            for rid in ("a", "b"):
                controller.submit(fixture_request(rid), {"root": "fixture"})
            for item in sequence:
                if item == "invalidate":
                    controller.invalidate("fixture-order", 1)
                else:
                    rid = item[-1]
                    controller.publish(rid, fixture_result(fixture_request(rid)), 1)
            controller.poll(request_context(fixture_request()), 0, [True] * 6, lambda _: True)

        run("order-" + "-".join(order), interleave, capacity=2)

    result = {"scope": "SYNTHETIC_PRECHECK_ONLY", "fixture_count": len(rows), "fixtures": rows}
    export_json(directory / "fixtures.json", copy.deepcopy(result))
    return result
