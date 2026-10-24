"""Synthetic coordinator preempt/resume checks; no gameplay or numerical imports."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels import factory_persistence as fp
from research.bl_labels import mac_driver as md
from research.bl_labels import mac_gates, slowdown_state, w12_runtime
from research.bl_labels.exports import aggregate_attempts, empty_ledger, publish_json, publish_life


@pytest.mark.parametrize(
    "fault", ["persistent", "timing", "health", "slowdown", "simultaneous", "monitor"]
)
def test_real_coordinator_stops_and_resumes_opaque_prefix(tmp_path, monkeypatch, fault):
    """Exercise admission, runtime event latch, immutable exports and exact charges."""
    clock = [1000.0]
    clock_api = SimpleNamespace(
        monotonic=lambda: clock[0],
        time=time.time,
        sleep=lambda n: clock.__setitem__(0, clock[0] + n),
    )
    monkeypatch.setattr(md, "time", clock_api)
    monkeypatch.setattr(fp, "time", clock_api)
    first_attempt = [True]
    events = SimpleNamespace(
        stopped=False, is_set=lambda: events.stopped, set=lambda: setattr(events, "stopped", True)
    )
    created = []

    def health(_):
        if first_attempt[0] and fault == "health" and clock[0] >= 1040:
            raise md.MacGuardError("synthetic health fault")
        return {"ac_power": True, "lid_open": True}

    monkeypatch.setattr(md, "_guard_snapshot", health)
    monkeypatch.setattr(
        md, "_timing_armed", lambda _: first_attempt[0] and fault == "timing" and clock[0] >= 1040
    )
    monkeypatch.setattr(md, "_rss_many", lambda pids: {pid: 0 for pid in pids})
    monkeypatch.setattr(mac_gates, "require_admission", lambda _: None)
    monkeypatch.setattr(w12_runtime, "resolve_resume_workers", lambda width, _: (width, []))
    monkeypatch.setenv("F0_COMMIT", "f" * 40)

    def table(_):
        if first_attempt[0] and created:
            if fault == "monitor":
                raise OSError("synthetic sensor failure")
            return "123 1 14 codex exec opaque\n"
        return "1 0 0 /sbin/launchd\n"

    monkeypatch.setattr(md, "_command_output", table)
    monkeypatch.setattr(
        fp,
        "_native_identity",
        lambda pid: {
            "birth": "native-fixture",
            "executable": "/opt/bin/codex",
            "birth_precision": "native tick",
        },
    )

    class Guard:
        def switch(self, *args):
            pass

        def admit(self, *args, **kwargs):
            pass

        def complete(self, *args):
            pass

        def boundary(self, *args, **kwargs):
            return None

        def snapshot(self, now):
            return {"width_wall_seconds": {"6": now - 1000}}

        def tick(self, now):
            if (
                first_attempt[0]
                and fault in {"slowdown", "simultaneous"}
                and now >= (1065 if fault == "simultaneous" else 1040)
            ):
                return {"event": "slowdown_guard", "width": 6, "stop_reason": "synthetic slowdown"}
            return None

    monkeypatch.setattr(slowdown_state, "build_guards", lambda *args: {"watch6": Guard()})
    monkeypatch.setattr(slowdown_state, "persist_baselines", lambda *args: None)

    class FixtureQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            if timeout is not None and self.empty():
                clock[0] += 5
                raise queue.Empty
            return super().get(block=False)

    class Process:
        def __init__(self, target, args):
            self.args = args
            self.pid = 900000 + len(created)
            self.exitcode = 0
            created.append(self)

        def start(self):
            job, namespace, allocations, _, _, output, _, _, _, _, messages = self.args
            life = 0 if first_attempt[0] else 1
            assert job.skip_lives == (set() if first_attempt[0] else {0})
            publish_life(
                output,
                namespace,
                allocations,
                {
                    "schema": "bl-labels-f0-life/v1",
                    "status": "COMPLETE",
                    "namespace": namespace,
                    "allocation_id": job.allocation["allocation_id"],
                    "allocation": job.allocation,
                    "life": life,
                    "rows": [],
                    "opaque_fixture": "same-fixed-prefix",
                },
            )
            if not first_attempt[0]:
                messages.put({"kind": "complete", "allocation_id": job.allocation["allocation_id"]})

        def is_alive(self):
            return first_attempt[0] and not events.stopped

        def join(self, timeout=None):
            pass

        def terminate(self):
            pytest.fail("cooperative stop must not require forced termination")

    monkeypatch.setattr(
        md.mp,
        "get_context",
        lambda _: SimpleNamespace(Queue=FixtureQueue, Event=lambda: events, Process=Process),
    )
    state = tmp_path / "state"
    allocation = [{"allocation_id": "opaque-000", "stream": "watch6", "horizon": 1, "seed": 7}]
    control = {
        "namespace": "synthetic-persistence",
        "source_commit": "f" * 40,
        "allocation": allocation,
        "mac_gate": {
            "stage": "d1024",
            "package_root": str(tmp_path),
            "deadline_seconds": 1200,
            "state_dir": str(state),
        },
    }
    initial = tmp_path / "initial.json"
    publish_json(initial, empty_ledger(control["namespace"], allocation))

    def run(ledger, name):
        return md.run(
            control,
            ledger,
            Path("b8"),
            Path("frp3"),
            tmp_path / name,
            workers=6,
            state_dir=state,
            deadline_seconds=1200,
            timing_armed=tmp_path / "unarmed",
        )

    first = run(initial, "first")
    assert first["status"] == "PREEMPTED"
    assert events.stopped
    summary = first["factory_process_monitor"]
    if fault in {"persistent", "simultaneous"}:
        assert summary["sightings"][0]["maximum_streak"] == 7
        assert summary["sightings"][0]["confirmed_trips"] == 1
    elif fault == "monitor":
        assert summary["observation_errors"]
    else:
        assert not any(row["confirmed_trips"] for row in summary["sightings"])
        assert first["elapsed_seconds"] < 60  # These trips never wait for persistence.
    assert first["slowdown_trips"] == (1 if fault in {"slowdown", "simultaneous"} else 0)
    if fault == "simultaneous":
        assert first["stop_reason"] == "synthetic slowdown"
        assert "persistent" in first["post_slowdown_stop_reason"]
    prefix = tmp_path / "first/lives/opaque-000--life-000000.json"
    before = prefix.read_bytes()
    ledger = tmp_path / "verified/ledger.json"
    aggregate_attempts([tmp_path / "first"], ledger, control["namespace"], allocation)
    first_attempt[0] = False
    events.stopped = False
    resumed = run(ledger, "second")
    assert resumed["status"] == "COMPLETE"
    assert prefix.read_bytes() == before
    assert (ledger.parent / "lives" / prefix.name).read_bytes() == before
    for value in (first, resumed):
        assert value["worker_seconds_denominator"] == 6 * value["elapsed_seconds"]
        assert (
            value["cumulative_worker_seconds_denominator"]
            == value["prior_worker_seconds_denominator"] + 6 * value["elapsed_seconds"]
        )
    assert resumed["prior_elapsed_seconds"] == first["elapsed_seconds"]
    assert resumed["prior_worker_seconds_denominator"] == first["worker_seconds_denominator"]
    assert (
        json.loads((tmp_path / "second/lives/opaque-000--life-000001.json").read_text())["life"]
        == 1
    )
