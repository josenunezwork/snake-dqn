"""Focused unit coverage for the first served-tick diagnostic package."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_served_search import served_tick_probe as probe


def _manifest() -> dict:
    return json.loads((Path(probe.__file__).with_name("served_tick_manifest.json")).read_text())


def test_preflight_validates_domain_separated_seed_panel() -> None:
    value = probe.preflight(_manifest())
    assert value["status"] == "PREFLIGHT_OK_NOT_TIMING"
    assert len(value["effective_numpy_seeds"]) == 4


def test_preflight_rejects_changed_exposure() -> None:
    value = _manifest()
    value["measured_ticks"] = 120
    with pytest.raises(ValueError, match="frozen"):
        probe.preflight(value)


def test_pair_requires_actions_state_and_detached_dto() -> None:
    left = {"actions": [[1]], "state_digest": ["a"], "dto_digest": ["b"]}
    assert probe._verify_pair(left, copy.deepcopy(left))["pass"]
    right = copy.deepcopy(left)
    right["dto_digest"] = ["changed"]
    result = probe._verify_pair(left, right)
    assert not result["pass"] and result["mismatches"] == ["dto_digest"]


def test_canonical_excludes_only_declared_runtime_duration_fields() -> None:
    baseline = {"timeout_seconds": 10, "apply_seconds_total": 1.0}
    assert probe._digest(baseline) == probe._digest(
        {"timeout_seconds": 10, "apply_seconds_total": 2.0}
    )
    assert probe._digest(baseline) != probe._digest(
        {"timeout_seconds": 11, "apply_seconds_total": 1.0}
    )


def test_reducer_reports_separate_engineering_bar() -> None:
    raw = {
        "manifest": {"measured_ticks": 2, "fps": 24},
        "cells": [
            {
                "cell": {"arm": "inline"},
                "result": {
                    "trace": [
                        {
                            "tick": 0,
                            "target_ns": 0,
                            "wake_ns": 0,
                            "start_ns": 0,
                            "end_ns": 1,
                            "spans": {"game_update": {"wall_ns": 1_000_000, "calls": 1}},
                        },
                        {
                            "tick": 1,
                            "target_ns": probe.FRAME_NS,
                            "wake_ns": probe.FRAME_NS,
                            "start_ns": probe.FRAME_NS,
                            "end_ns": probe.FRAME_NS * 2,
                            "spans": {"game_update": {"wall_ns": 12_000_000, "calls": 1}},
                        },
                    ]
                },
            }
        ],
    }
    report = probe.reduce_trace(raw)
    assert report["verdict"] == "INVALID"


def test_run_panel_uses_create_only_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    manifest["measured_ticks"] = 1
    manifest["order"] = [{"arm": "timer", "seed": manifest["seeds"][0]}]
    monkeypatch.setattr(
        probe,
        "run_inline_arm",
        lambda spec: {
            "trace": [{"target_ns": 0, "end_ns": 1, "spans": {}}],
            "actions": [],
            "state_digest": [],
            "dto_digest": [],
        },
    )
    raw = probe.run_panel(manifest, tmp_path)
    assert raw["cells"][0]["result"]["trace"]
    assert not (tmp_path / "raw.json").exists()
    assert (tmp_path / "cell-00.json").exists()
    with pytest.raises(FileExistsError):
        probe.run_panel(manifest, tmp_path)


def _production_fixture() -> dict:
    from research.bl_served_search.tick_trace import TraceRecorder

    trace = TraceRecorder(4)
    rows = []
    frames = [1, 2, 1, 2, 1, 2]
    sends, receipts = [], [{"source": "connect_snapshot", "frame": 0, "receive_ns": 1}]
    for index, frame in enumerate(frames):
        start = (index + 1) * 1_000_000
        sends.append(
            {"frame": frame, "start_ns": start + 500, "end_ns": start + 600, "status": "ok"}
        )
        receipts.append({"source": "hub_broadcast", "frame": frame, "receive_ns": start + 700})
        if index < 2:
            continue
        trace.begin_tick(index - 2, start, start)
        with trace.span("game_update"):
            pass
        trace.end_tick()
        row = trace.rows()[-1]
        # Separate worker clock domains; timings here are intentionally synthetic.
        row.update(
            game_age=frame,
            start_ns=start,
            end_ns=start + 100,
            start_cpu_ns=10,
            end_cpu_ns=20,
            snapshot_start_cpu_ns=100,
            snapshot_end_cpu_ns=130,
            gc_start=[0, 1, 0],
            gc_stop=[0, 1, 0],
            gc_overlap_ns=[0, 7, 0],
        )
        update = row["spans"]["game_update"]
        update.update(
            wall_ns=30,
            wall_exclusive_ns=30,
            cpu_ns=10,
            cpu_exclusive_ns=10,
            first_wall_start_ns=start + 10,
            last_wall_end_ns=start + 40,
        )
        row["spans"]["snapshot"].update(
            wall_ns=100, wall_exclusive_ns=20, cpu_ns=30, cpu_exclusive_ns=5, calls=1
        )
        row["spans"]["dto"].update(
            wall_ns=80, wall_exclusive_ns=80, cpu_ns=25, cpu_exclusive_ns=25, calls=1
        )
        row["spans"]["ipc_lock"].update(
            wall_ns=15, wall_exclusive_ns=15, cpu_ns=2, cpu_exclusive_ns=2, calls=1
        )
        rows.append(row)
    return {"trace": rows, "frame_send": sends, "client_receive": receipts, "warmup_ticks": 2}


def test_production_join_preserves_repeated_frames_across_resets_and_warmup() -> None:
    report = probe._production_summary(_production_fixture(), 4)
    assert report["verdict"] == "PASS"
    assert report["client_frame_gap_ns"]["p50"] == 1_000_000
    assert report["step_to_client_ns"]["max"] == 700
    assert report["span_summary"]["snapshot"]["wall_ns"]["p95"] == 100
    assert report["span_summary"]["dto"]["thread_cpu_ns"]["p95"] == 25
    assert report["span_summary"]["ipc_lock"]["wall_ns"]["p95"] == 15
    assert report["gc"]["start_counts"] == [0, 4, 0]
    assert report["gc"]["overlap_ns"]["p95"] == 7


@pytest.mark.parametrize("damage", ["missing_send", "missing_receive", "mismatch", "reorder"])
def test_production_join_refuses_missing_or_misaligned_delivery(damage: str) -> None:
    result = _production_fixture()
    if damage == "missing_send":
        result["frame_send"].pop()
    elif damage == "missing_receive":
        result["client_receive"].pop()
    elif damage == "mismatch":
        result["frame_send"][3]["frame"] = 99
    else:
        result["client_receive"][-1], result["client_receive"][-3] = (
            result["client_receive"][-3],
            result["client_receive"][-1],
        )
    assert probe._production_summary(result, 4)["verdict"] == "INVALID"


def test_host_polling_is_off_tick_and_stops_on_exit() -> None:
    import threading

    owner = threading.get_ident()
    observed = threading.Event()
    calls = []

    class Monitor:
        def check(self) -> None:
            calls.append(threading.get_ident())
            if calls[-1] != owner:
                observed.set()

    with probe._poll_host(Monitor(), interval_seconds=0.001):
        assert observed.wait(1)
    assert calls[0] == calls[-1] == owner
    assert any(identifier != owner for identifier in calls)
    assert not any(t.name == "served-tick-host-poll" for t in threading.enumerate())


def test_host_polling_failure_is_propagated_without_per_tick_checks() -> None:
    import threading

    owner = threading.get_ident()
    observed = threading.Event()

    class Monitor:
        def check(self) -> None:
            if threading.get_ident() != owner:
                observed.set()
                raise RuntimeError("guard failure")

    with pytest.raises(RuntimeError, match="host polling failed"):
        with probe._poll_host(Monitor(), interval_seconds=0.001):
            assert observed.wait(1)


def test_canonical_inherited_slots_exclude_runtime_timings() -> None:
    class Base:
        __slots__ = ("decision",)

    class Child(Base):
        __slots__ = ("apply_seconds_total",)

    item = Child()
    item.decision, item.apply_seconds_total = 1, 1.0
    initial = probe._digest(item)
    item.apply_seconds_total = 2.0
    assert probe._digest(item) == initial
    item.decision = 2
    assert probe._digest(item) != initial


def test_panel_polls_all_arm_types_and_observers_uniformly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from research.bl_served_search import isolated_game, production_probe

    owner = threading.get_ident()
    polled = threading.Event()
    workloads = []

    class Monitor:
        receipt = {}

        def check(self) -> None:
            if threading.get_ident() != owner:
                polled.set()

    def arm(spec: dict, *args: object, **kwargs: object) -> dict:
        assert "host_monitor" not in spec
        polled.clear()
        assert polled.wait(1), spec["arm"]
        workloads.append(spec["arm"])
        return {
            "seed": spec["seed"],
            "actions": [[0]],
            "state_digest": {"x": 1},
            "dto_digest": {"x": 2},
        }

    original_poll = probe._poll_host
    monkeypatch.setattr(probe, "_poll_host", lambda monitor: original_poll(monitor, 0.001))
    monkeypatch.setattr(probe, "run_inline_arm", arm)
    monkeypatch.setattr(isolated_game, "run_isolated_arm", arm)
    monkeypatch.setattr(production_probe, "run_production_arm", arm)
    probe.run_panel(_manifest(), tmp_path, Monitor())
    assert set(workloads) == {"timer", "inline", "isolated", "production"}
    assert len(workloads) == 14


def test_inline_loop_does_not_poll_host_each_tick() -> None:
    class Monitor:
        def check(self) -> None:
            pytest.fail("per-tick monitor cost")

    result = probe.run_inline_arm(
        {
            "arm": "timer",
            "seed": 1,
            "warmup_ticks": 0,
            "measured_ticks": 2,
            "host_monitor": Monitor(),
        }
    )
    assert len(result["trace"]) == 2


def test_final_invalid_host_sample_still_exports_raw_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_paced_identity.record import export_json, read_export
    from research.bl_served_search import probe_host

    output = tmp_path / "attempt"
    binding = {"commit": "a" * 40, "tracked_dirty": "", "source_sha256": {"probe": "b"}}
    monkeypatch.setattr(probe, "source_binding", lambda **kwargs: binding)

    def guarded(destination: Path, callback: object) -> dict:
        export_json(destination / "host.json", {"status": "INVALID", "reason": "final sample"})
        return {"manifest": _manifest(), "cells": []}

    monkeypatch.setattr(probe_host, "guarded_run", guarded)
    assert probe.main(["--output", str(output)]) == 2
    assert read_export(output / "raw.json")["host"]["status"] == "INVALID"
    assert read_export(output / "report.json")["verdict"] == "INVALID"


def test_reducer_rejects_failed_monitor_receipt_even_after_complete_export() -> None:
    manifest = _manifest()
    binding = {"commit": "a" * 40, "tracked_dirty": "", "source_sha256": {"probe": "b"}}
    raw = {
        "manifest": manifest,
        "cells": [{"cell": cell} for cell in manifest["order"]],
        "source_before": binding,
        "source_after": binding,
        "host": {"status": "COMPLETE"},
        "lease": {"monitor_status": "INVALID"},
    }
    assert probe.reduce_trace(raw) == {
        "verdict": "INVALID",
        "reason": "missing or contaminated host evidence",
    }


def test_observer_processing_brackets_all_recorder_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from research.bl_served_search import tick_trace
    from research.t1_commitment_20261015 import replay

    clock = [0]

    def advance(amount: int) -> None:
        clock[0] += amount

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            pass

        def attach(self, session: object) -> probe._nullcontext:
            return probe._nullcontext()

        def begin_tick(self, *args: object) -> None:
            advance(10)

        def capture_session_context(self, session: object) -> None:
            advance(40)

        def end_tick(self, *args: object) -> None:
            advance(50)

        def rows(self) -> list:
            return []

    def snapshot() -> dict:
        advance(30)
        return {}

    session = SimpleNamespace(
        game=SimpleNamespace(snakes=[]),
        step=lambda: advance(20),
        snapshot=snapshot,
        close=lambda **kwargs: None,
    )
    monkeypatch.setattr(tick_trace, "TraceRecorder", Recorder)
    monkeypatch.setattr(replay, "configure_cpu", lambda: None)
    monkeypatch.setattr(probe, "new_probe_watch", lambda seed: (session, {}))
    monkeypatch.setattr(probe, "session_state_digest", lambda session: "state")
    monkeypatch.setattr(probe, "_identity", lambda *args: {})
    monkeypatch.setattr(probe, "_sleep_until", lambda target: clock[0])
    monkeypatch.setattr(probe.time, "perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(probe.time, "thread_time_ns", lambda: clock[0])
    spec = {"arm": "inline", "seed": 1, "warmup_ticks": 0, "measured_ticks": 1}
    traced = probe.run_inline_arm(spec)
    plain = probe.run_inline_arm({**spec, "trace_enabled": False})
    assert traced["processing_wall_ns"] == traced["processing_thread_cpu_ns"] == [150]
    assert plain["processing_wall_ns"] == plain["processing_thread_cpu_ns"] == [50]
