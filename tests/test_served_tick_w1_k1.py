"""Focused unit and integration coverage for the W1/K1 contention-only arm."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import struct
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from research.bl_served_search import contention_worker as cw
from research.bl_served_search import isolated_game
from research.bl_served_search import served_tick_probe as probe


def _w1_manifest() -> dict:
    return json.loads(
        (Path(probe.__file__).with_name("served_tick_w1_k1_manifest.json")).read_text()
    )


def _synthetic_fixtures() -> tuple[list[dict], dict[str, int]]:
    fixtures = [
        {
            "fixture_id": f"syn-root-{i}",
            "mix": "scripted" if i == 0 else "mixed",
            "target_length": 600,
            "legal_actions": [0, 1, 2],
            "expected_action": 1,
        }
        for i in range(3)
    ]
    actions = {f["fixture_id"]: 1 for f in fixtures}
    return fixtures, actions


def _dummy_session() -> SimpleNamespace:
    snake0 = SimpleNamespace(
        id=0,
        length=12,
        is_alive=True,
        is_boosting=False,
        direction=(1, 0),
        last_action=0,
        segments=[(10.0, 20.0), (9.0, 20.0), (8.0, 20.0)],
    )
    snake1 = SimpleNamespace(
        id=1,
        length=8,
        is_alive=True,
        is_boosting=True,
        direction=(0, 1),
        last_action=2,
        segments=[(40.0, 50.0), (40.0, 49.0)],
    )
    food = [SimpleNamespace(x=15.0, y=25.0), SimpleNamespace(x=30.0, y=35.0)]
    game = SimpleNamespace(frame=42, alive_snakes=2, snakes=[snake0, snake1], food=food)
    return SimpleNamespace(hero_id=0, game=game)


def test_w1_k1_manifest_preflight_validates_frozen_panel_and_ruler_bindings() -> None:
    manifest = _w1_manifest()
    result = probe.preflight(manifest)
    assert result["status"] == "PREFLIGHT_OK_NOT_TIMING"
    assert result["manifest_id"] == probe.W1_K1_MANIFEST_ID
    assert result["workers"] == 1
    assert result["max_pending"] == 1
    assert result["worker_mode"] == "active"
    assert result["operator_mode"] == "exact_stop"
    assert result["cells"] == 6
    assert result["fixtures_sha256"] == cw.Y1_FIXTURES_SHA256
    assert result["reference_sha256"] == cw.Y1_REFERENCE_SHA256


def test_w1_k1_manifest_preflight_rejects_tampered_workers_modes_or_hashes() -> None:
    manifest = _w1_manifest()
    manifest["max_pending"] = 2
    with pytest.raises(ValueError, match="invalid W1/K1"):
        probe.preflight(manifest)

    manifest2 = _w1_manifest()
    manifest2["fixtures_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the committed fixed panel"):
        probe.preflight(manifest2)

    # Finding 3 regression: custom manifest cannot silently set worker_mode="idle"
    # or operator_mode="synthetic_fast" or inject extra keys.
    manifest_idle = _w1_manifest()
    manifest_idle["worker_mode"] = "idle"
    with pytest.raises(ValueError, match="invalid W1/K1|differs from the committed fixed panel"):
        probe.preflight(manifest_idle)

    manifest_syn = _w1_manifest()
    manifest_syn["operator_mode"] = "synthetic_fast"
    with pytest.raises(ValueError, match="invalid W1/K1|differs from the committed fixed panel"):
        probe.preflight(manifest_syn)

    manifest_extra = _w1_manifest()
    manifest_extra["prewarm_count"] = 1
    with pytest.raises(ValueError, match="differs from the committed fixed panel"):
        probe.preflight(manifest_extra)

    w0_manifest = json.loads(
        (Path(probe.__file__).with_name("served_tick_manifest.json")).read_text()
    )
    w0_manifest["worker_mode"] = "idle"
    with pytest.raises(ValueError, match="differs from the committed fixed panel"):
        probe.preflight(w0_manifest)


def test_fixture_rotation_matches_m1_interleaved_schedule() -> None:
    from research.bl_k1.m1 import schedule as m1_schedule

    loaded = cw.load_ruler_stress_inputs()
    fixtures = loaded["fixtures"]
    expected = m1_schedule(fixtures)
    actual = [cw.fixture_at(fixtures, idx) for idx in range(len(expected))]
    assert actual == expected


def test_worker_lifecycle_prewarm_and_clean_shutdown() -> None:
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        operator_mode="synthetic_fast",
    )
    info = worker.start_and_prewarm(prewarm_count=3)
    try:
        assert worker.prewarmed is True
        assert info["worker_pid"] is not None and info["worker_pid"] > 0
        assert info["prewarm_completed_count"] == 3
        assert len(worker.cold_records) == 3
        assert worker.current_pending_count == 0
        assert worker.worker_threads["device"] == "cpu"
        assert worker.worker_threads["torch_num_threads"] == 1
        assert worker.worker_threads["torch_num_interop_threads"] == 1
        assert worker.worker_threads["numba_num_threads"] == 1
        assert all(
            r["phase"] == "cold" and r["action_applied"] is False for r in worker.cold_records
        )
        assert info["worker_rss_bytes_initial"] > 0
        assert info["worker_rss_bytes_after_prewarm"] > 0
    finally:
        shutdown = worker.close()
    assert shutdown["clean_shutdown"] is True
    assert shutdown["worker_exitcode"] == 0
    assert shutdown["worker_rss_bytes_final"] > 0


def test_worker_lifecycle_crash_is_nonblocking_and_counted() -> None:
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=1)
    try:
        session = _dummy_session()
        packet, copy_bytes = worker.copy_state_and_prepare_request(session, 0)
        # Replace command with simulate_crash so worker exits mid-request.
        crash_packet = {**packet, "cmd": "simulate_crash", "exitcode": 137}
        assert worker.try_publish(crash_packet, copy_bytes, 0) is True
        assert worker.current_pending_count == 1

        # Wait briefly for child process to exit, then poll on tick 1.
        deadline = time.monotonic() + 5.0
        while (
            worker._process is not None
            and worker._process.is_alive()
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert worker.poll(1) is None
        assert worker.worker_crash_count == 1
        assert worker.fallback_count >= 1
        assert worker.current_pending_count == 0

        # Subsequent tick attempts fail closed without raising or blocking.
        packet2, copy_bytes2 = worker.copy_state_and_prepare_request(session, 2)
        assert worker.try_publish(packet2, copy_bytes2, 2) is False
        assert worker.actions_applied_count == 0
    finally:
        shutdown = worker.close()
    assert shutdown["clean_shutdown"] is False
    payload = worker.export_payload(game_rss_before=1000, game_rss_after=1000, game_maxrss=1000)
    assert payload["summary"]["verdict"] == "INVALID"


def test_worker_lifecycle_timeout_is_nonblocking_and_counted() -> None:
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        request_timeout_ns=30_000_000,  # 30 ms timeout
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=1)
    try:
        session = _dummy_session()
        packet, copy_bytes = worker.copy_state_and_prepare_request(session, 0)
        hang_packet = {**packet, "cmd": "simulate_hang", "seconds": 10.0}
        assert worker.try_publish(hang_packet, copy_bytes, 0) is True
        assert worker.current_pending_count == 1

        time.sleep(0.05)  # Exceed the 30 ms request timeout.
        assert worker.poll(1) is None
        assert worker.worker_timeout_count == 1
        assert worker.fallback_count >= 1
        assert worker.current_pending_count == 0
        assert worker._stop_event.is_set() is True

        # Subsequent offers are refused immediately without blocking.
        packet2, copy_bytes2 = worker.copy_state_and_prepare_request(session, 2)
        assert worker.try_publish(packet2, copy_bytes2, 2) is False
    finally:
        shutdown = worker.close()
    assert shutdown["clean_shutdown"] is False
    payload = worker.export_payload(game_rss_before=1000, game_rss_after=1000, game_maxrss=1000)
    assert payload["summary"]["verdict"] == "INVALID"


def test_bounded_queue_enforces_k_le_1_and_counts_overflows() -> None:
    fixtures, actions = _synthetic_fixtures()
    with pytest.raises(ValueError, match="max_pending == 1"):
        cw.PersistentExactStopWorker(
            fixtures=fixtures,
            reference_actions=actions,
            measured_ticks=4,
            max_pending=2,
            operator_mode="synthetic_fast",
        )

    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=6,
        operator_mode="synthetic_fast",
        operator_kwargs={"delay_seconds": 0.08},
    )
    worker.start_and_prewarm(prewarm_count=1)
    try:
        session = _dummy_session()
        # Tick 0: idle -> admits request 0 (fixture syn-root-0).
        worker.step_tick(session, 0)
        assert worker.submitted_count == 1
        assert worker.current_pending_count == 1
        assert worker.max_observed_pending == 1
        assert worker.queue_overflow_count == 0

        # Tick 1 & Tick 2: worker still busy -> K<=1 refuses immediately without advancing cursor.
        worker.step_tick(session, 1)
        worker.step_tick(session, 2)
        assert worker.submitted_count == 1
        assert worker.current_pending_count == 1
        assert worker.max_observed_pending == 1
        assert worker.queue_overflow_count == 2
        assert worker.fallback_count == 2

        # Wait for request 0 to complete, then run Tick 3: poll consumes 0 and admits 1.
        time.sleep(0.10)
        worker.step_tick(session, 3)
        assert worker.completed_count == 1
        assert worker.submitted_count == 2
        assert worker.current_pending_count == 1
        assert worker.max_observed_pending == 1
        assert worker.warm_records[0]["fixture_id"] == "syn-root-0"
        assert worker.pending_request is not None
        assert worker.pending_request["fixture_id"] == "syn-root-1"
    finally:
        worker.close()


def test_no_action_application_leaves_live_session_and_snakes_untouched() -> None:
    fixtures, actions = _synthetic_fixtures()
    # Configure worker to choose action 2 while live hero has last_action=0.
    for f in fixtures:
        f["expected_action"] = 2
    actions = {f["fixture_id"]: 2 for f in fixtures}

    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        operator_mode="synthetic_fast",
        operator_kwargs={"chosen_override": 2},
    )
    worker.start_and_prewarm(prewarm_count=1)
    try:
        session = _dummy_session()
        before_snapshot = deepcopy(session)
        for tick in range(4):
            worker.step_tick(session, tick)
            time.sleep(0.02)
        assert worker.completed_count >= 1
        assert worker.warm_records[0]["chosen_action"] == 2
        assert worker.warm_records[0]["action_applied"] is False
        assert worker.actions_applied_count == 0
        # Live session state is 100% unchanged by worker replies.
        assert session.game.frame == before_snapshot.game.frame
        assert [s.last_action for s in session.game.snakes] == [
            s.last_action for s in before_snapshot.game.snakes
        ]
        assert [s.segments for s in session.game.snakes] == [
            s.segments for s in before_snapshot.game.snakes
        ]
    finally:
        worker.close()
    payload = worker.export_payload(game_rss_before=1000, game_rss_after=1000, game_maxrss=1000)
    assert payload["summary"]["verdict"] == "PASS"

    # Tampering with actions_applied_count must fail closed as INVALID.
    tampered = deepcopy(payload)
    tampered["counts"]["actions_applied_count"] = 1
    assert cw.reduce_worker_contention(tampered, expected_ticks=4)["verdict"] == "INVALID"


@pytest.mark.skipif(
    not all(
        Path("saved_snakes", name).is_file()
        for name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth")
    ),
    reason="requires local released checkpoint artifacts",
)
def test_game_loop_parity_with_worker_idle_and_active_real_exact_stop() -> None:
    """Verify exact state/DTO/action parity between W0 and W1/K1 (both idle and active)."""
    manifest = _w1_manifest()
    seed = manifest["seeds"][0]
    base_spec = {
        "arm": "inline",
        "seed": seed,
        "warmup_ticks": 1,
        "measured_ticks": 3,
    }
    w0_result = probe.run_inline_arm(base_spec)

    # 1. Worker idle (functional parity check only; contention reducer must reject idle mode).
    idle_spec = {
        "arm": "w1_k1",
        "seed": seed,
        "warmup_ticks": 1,
        "measured_ticks": 3,
        "worker_mode": "idle",
        "prewarm_count": 1,
    }
    idle_result = probe.run_inline_arm(idle_spec)
    idle_parity = probe._verify_pair(w0_result, idle_result)
    assert idle_parity["pass"], idle_parity
    idle_worker_raw = idle_result["worker_contention"]
    assert idle_worker_raw["counts"]["prewarm_completed_count"] == 1
    assert idle_worker_raw["counts"]["submitted_count"] == 0
    assert idle_worker_raw["counts"]["actions_applied_count"] == 0
    assert idle_worker_raw["summary"]["verdict"] == "INVALID"

    # 2. Worker active (prewarmed real ExactT3Operator + on-tick state-copy/publish/poll).
    active_spec = {
        "arm": "w1_k1",
        "seed": seed,
        "warmup_ticks": 1,
        "measured_ticks": 3,
        "worker_mode": "active",
        "prewarm_count": 1,
    }
    active_result = isolated_game.run_isolated_arm(active_spec, timeout_seconds=120)
    active_parity = probe._verify_pair(w0_result, active_result)
    assert active_parity["pass"], active_parity
    active_summary = active_result["worker_contention"]["summary"]
    assert active_summary["verdict"] == "PASS"
    assert active_summary["counts"]["prewarm_completed_count"] == 1
    assert active_summary["counts"]["offered_count"] == 3
    assert active_summary["counts"]["submitted_count"] >= 1
    assert active_summary["counts"]["max_observed_pending"] == 1
    assert active_summary["counts"]["actions_applied_count"] == 0
    assert active_summary["tick_ipc_costs"]["state_copy_bytes"]["max"] > 0
    assert active_summary["rss"]["game_rss_bytes_after"] > 0
    assert active_summary["rss"]["worker_rss_bytes_after_prewarm"] > 0


def test_reducer_reports_all_w1_k1_fields_and_follows_ruling_decision_rules() -> None:
    manifest = _w1_manifest()
    binding = {"commit": "a" * 40, "tracked_dirty": "", "source_sha256": {"probe": "b" * 64}}

    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=2400,
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=3)
    try:
        session = _dummy_session()
        for tick in range(2400):
            worker.step_tick(session, tick)
            if tick < 4:
                time.sleep(0.002)
    finally:
        worker.close()
    worker_payload = worker.export_payload(
        game_rss_before=100_000_000,
        game_rss_after=102_000_000,
        game_maxrss=105_000_000,
    )
    assert worker_payload["summary"]["verdict"] == "PASS"
    assert "warm_s_p95_limit_ns" not in worker_payload["summary"]
    assert "warm_s_p95_pass" not in worker_payload["summary"]
    assert worker_payload["summary"]["service_latency"]["cold_s_ns"]["count"] == 3
    assert worker_payload["summary"]["service_latency"]["warm_s_ns"]["count"] >= 1

    # Scored panel reducer requires operator_mode == "exact_stop".
    scored_worker_payload = deepcopy(worker_payload)
    scored_worker_payload["operator_mode"] = "exact_stop"

    from research.bl_served_search.tick_trace import TraceRecorder

    wall_clock = [0]
    cpu_clock = [0]
    recorder = TraceRecorder(
        2400,
        wall_clock_ns=lambda: wall_clock[0],
        cpu_clock_ns=lambda: cpu_clock[0],
    )
    for index in range(2400):
        target = (index * 1_000_000_000) // probe.FPS
        wall_clock[0] = target + 10_000
        cpu_clock[0] = index * 1_500_000
        recorder.begin_tick(index, target, target + 5_000)
        with recorder.span("session_step"):
            with recorder.span("game_update"):
                wall_clock[0] += 1_000_000
                cpu_clock[0] += 800_000
        with recorder.span("snapshot"):
            with recorder.span("dto"):
                wall_clock[0] += 400_000
                cpu_clock[0] += 300_000
        wall_clock[0] += 100_000
        cpu_clock[0] += 50_000
        recorder.end_tick()
    rows = recorder.rows()

    base_result = {
        "actions": [[1]] * 2400,
        "state_digest": {"initial": "a", "final": "b"},
        "dto_digest": {"final": "c"},
        "processing_wall_ns": [2_000_000] * 2400,
        "processing_thread_cpu_ns": [1_500_000] * 2400,
        "trace": rows,
        "fps": 24,
        "warmup_ticks": 200,
        "measured_ticks": 2400,
    }
    observer_result = {
        "actions": [[1]] * 120,
        "state_digest": {"initial": "a", "final": "b"},
        "dto_digest": {"final": "c"},
        "processing_wall_ns": [2_000_000] * 120,
        "processing_thread_cpu_ns": [1_500_000] * 120,
    }
    raw = {
        "manifest": manifest,
        "source_before": binding,
        "source_after": deepcopy(binding),
        "host": {"status": "COMPLETE"},
        "cells": [
            {
                "cell": cell,
                "result": {
                    **deepcopy(base_result),
                    "seed": manifest["seeds"][cell["seed_index"]],
                    **(
                        {"worker_contention": deepcopy(scored_worker_payload)}
                        if cell["arm"] == "w1_k1"
                        else {}
                    ),
                },
            }
            for cell in manifest["order"]
        ],
        "observer_control": {
            "ticks": 120,
            "arms": [
                {"name": name, "result": deepcopy(observer_result)}
                for name in manifest["instrumented_observer_order"]
            ],
        },
    }
    report = probe.reduce_trace(raw)
    assert report["verdict"] == "PASS"
    assert all(p["pass"] for p in report["parity"])
    w1_arms = [a for a in report["arms"] if a["cell"]["arm"] == "w1_k1"]
    assert len(w1_arms) == 3
    for arm_report in w1_arms:
        assert "update_ns" in arm_report and "update_thread_cpu_ns" in arm_report
        assert "step_ns" in arm_report and "step_thread_cpu_ns" in arm_report
        assert "snapshot_ns" in arm_report and "snapshot_thread_cpu_ns" in arm_report
        assert "tick_ns" in arm_report and "tick_thread_cpu_ns" in arm_report
        assert "wake_lateness_ns" in arm_report
        assert "service_latency" in arm_report
        assert "tick_ipc_costs" in arm_report
        assert "rss" in arm_report
        assert "overflow_fallback_counts" in arm_report
        assert "native_v8_totals" in arm_report

    # Finding 2 regression: warm S_ns of 260 ms (or 650 ms) is reported descriptively
    # and MUST NOT flip a cell/panel with 0 overruns and update p95 <= 11.688 ms to FAIL.
    slow_s_raw = deepcopy(raw)
    for rec in slow_s_raw["cells"][1]["result"]["worker_contention"]["warm_records"]:
        rec["S_ns"] = 260_000_000
    slow_s_worker = cw.reduce_worker_contention(
        slow_s_raw["cells"][1]["result"]["worker_contention"],
        expected_ticks=2400,
        require_exact_stop=True,
    )
    assert slow_s_worker["verdict"] == "PASS"
    assert slow_s_worker["service_latency"]["warm_s_ns"]["p95"] == 260_000_000.0
    slow_s_report = probe.reduce_trace(slow_s_raw)
    assert slow_s_report["verdict"] == "PASS"

    # Ruling FAIL rule: update p95 > 11.688 ms or any absolute overrun produces FAIL.
    update_fail_raw = deepcopy(raw)
    for row in update_fail_raw["cells"][1]["result"]["trace"]:
        row["spans"]["game_update"]["wall_ns"] = 12_000_000
        row["spans"]["game_update"]["last_wall_end_ns"] = (
            row["spans"]["game_update"]["first_wall_start_ns"] + 12_000_000
        )
        row["end_ns"] = row["spans"]["game_update"]["last_wall_end_ns"] + 100_000
    assert probe.reduce_trace(update_fail_raw)["verdict"] == "FAIL"

    # Finding 3 regression: synthetic_fast or idle worker_mode in scored panel is INVALID.
    syn_raw = deepcopy(raw)
    syn_raw["cells"][1]["result"]["worker_contention"]["operator_mode"] = "synthetic_fast"
    assert probe.reduce_trace(syn_raw)["verdict"] == "INVALID"

    idle_raw = deepcopy(raw)
    idle_raw["cells"][1]["result"]["worker_contention"]["worker_mode"] = "idle"
    assert probe.reduce_trace(idle_raw)["verdict"] == "INVALID"


def test_try_publish_never_blocks_and_arms_timeout_before_send() -> None:
    """Finding 1 regression: non-blocking send with timeout armed first; overflow = fallback."""
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=1)
    try:
        session = _dummy_session()
        packet, copy_bytes = worker.copy_state_and_prepare_request(session, 0)

        # 1. Verify pending_request and admitted_ns are armed BEFORE _try_send_nonblocking runs.
        observed_during_send: list[tuple[dict | None, int]] = []
        real_send = cw._try_send_nonblocking

        def spy_send(conn: object, payload: dict, max_bytes: int) -> str:
            observed_during_send.append(
                (deepcopy(worker.pending_request), int(worker.current_pending_count))
            )
            return real_send(conn, payload, max_bytes)

        cw._try_send_nonblocking = spy_send  # type: ignore[assignment]
        try:
            assert worker.try_publish(packet, copy_bytes, 0) is True
        finally:
            cw._try_send_nonblocking = real_send  # type: ignore[assignment]

        assert len(observed_during_send) == 1
        armed_req, armed_count = observed_during_send[0]
        assert armed_req is not None
        assert armed_req["tick_index"] == 0
        assert armed_req["admitted_ns"] > 0
        assert armed_count == 1

        # Drain reply from tick 0 so the worker slot is idle again.
        deadline = time.monotonic() + 2.0
        while worker.poll(0) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.current_pending_count == 0

        # 2. Reproduce Astra's check: real multiprocessing Pipe with undrained receiver
        # and 499,200 bytes (6 * 10,400 * 2 float32 coordinates). Must return immediately
        # without blocking, roll back pending_request, and count overflow + fallback.
        parent_conn, child_conn = mp.Pipe(duplex=True)
        saved_conn = worker._parent_conn
        worker._parent_conn = parent_conn
        try:
            big_coords = b"\x00" * 499_200
            big_packet = {
                **packet,
                "state_snapshot": {
                    "frame": 1,
                    "hero_id": 0,
                    "alive_snakes": 6,
                    "snakes": [{"id": i, "segments_f32": big_coords[:83_200]} for i in range(6)],
                    "food_f32": b"",
                },
            }
            done = threading.Event()
            outcome: list[bool] = []

            def run_publish() -> None:
                outcome.append(worker.try_publish(big_packet, 499_200, 1))
                done.set()

            pub_thread = threading.Thread(target=run_publish)
            pub_thread.start()
            assert done.wait(0.2), "try_publish blocked on undrained Pipe"
            pub_thread.join(timeout=1.0)
            assert outcome == [False]
            assert worker.pending_request is None
            assert worker.current_pending_count == 0
            assert worker.queue_overflow_count == 1
            assert worker.fallback_count == 1
            assert worker.tick_overflow[1] == 1

            # Even if max_packet_bytes is raised above 499,200, an unresized default Pipe
            # (SO_SNDBUF=8192) still refuses non-blockingly as overflow without blocking.
            worker.max_packet_bytes = 2_000_000
            started = time.perf_counter()
            assert worker.try_publish(big_packet, 499_200, 2) is False
            assert (time.perf_counter() - started) < 0.2
            assert worker.pending_request is None
            assert worker.current_pending_count == 0
            assert worker.queue_overflow_count == 2
            assert worker.fallback_count == 2
            assert worker.tick_overflow[2] == 1
        finally:
            worker._parent_conn = saved_conn
            parent_conn.close()
            child_conn.close()
    finally:
        worker.close()


def test_poll_never_blocks_on_partial_frame_and_enforces_request_timeout() -> None:
    """Finding 1 regression: partial message on pipe must not block poll() or evade timeout."""
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        request_timeout_ns=20_000_000,  # 20 ms
        operator_mode="synthetic_fast",
    )
    parent_conn, child_conn = mp.Pipe(duplex=True)
    worker._parent_conn = parent_conn
    worker._process = SimpleNamespace(is_alive=lambda: True)
    try:
        worker.pending_request = {
            "request_id": 1,
            "tick_index": 0,
            "admitted_ns": time.perf_counter_ns(),
            "fixture_id": "syn-root-0",
        }
        worker.current_pending_count = 1
        # Write a 4-byte header claiming 4096 bytes, followed by only 16 bytes.
        os.write(child_conn.fileno(), struct.pack("!i", 4096) + b"partial_payload_")

        started = time.perf_counter()
        assert worker.poll(0) is None
        assert (time.perf_counter() - started) < 0.1
        assert worker.pending_request is not None

        # After 25 ms, poll(1) must trip the 20 ms request timeout without blocking.
        time.sleep(0.025)
        assert worker.poll(1) is None
        assert worker.worker_timeout_count == 1
        assert worker.fallback_count == 1
        assert worker.pending_request is None
        assert worker.current_pending_count == 0
    finally:
        worker._parent_conn = None
        parent_conn.close()
        child_conn.close()
        worker._stop_event.close()


def test_worker_reducer_reconciles_exposure_counts_flags_and_record_identities() -> None:
    """Finding 4 regression: contradictory or zeroed worker exposure must return INVALID."""
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=8,
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=3)
    try:
        session = _dummy_session()
        for tick in range(8):
            worker.step_tick(session, tick)
            time.sleep(0.005)
    finally:
        worker.close()
    valid_payload = worker.export_payload(
        game_rss_before=1000, game_rss_after=1000, game_maxrss=1000
    )
    assert cw.reduce_worker_contention(valid_payload, expected_ticks=8)["verdict"] == "PASS"

    # 1. Astra's reproduction: zeroing offered/submitted counts and tick-cost/flag arrays.
    zeroed = deepcopy(valid_payload)
    zeroed["counts"]["offered_count"] = 0
    zeroed["counts"]["submitted_count"] = 0
    for key in zeroed["tick_costs"]:
        zeroed["tick_costs"][key] = [0] * 8
    assert cw.reduce_worker_contention(zeroed, expected_ticks=8)["verdict"] == "INVALID"

    # 2. offered_count != expected_ticks.
    bad_offered = deepcopy(valid_payload)
    bad_offered["counts"]["offered_count"] = 7
    assert cw.reduce_worker_contention(bad_offered, expected_ticks=8)["verdict"] == "INVALID"

    # 3. submitted_count + queue_overflow_count != offered_count.
    bad_sum = deepcopy(valid_payload)
    bad_sum["counts"]["queue_overflow_count"] += 1
    bad_sum["counts"]["fallback_count"] += 1
    assert cw.reduce_worker_contention(bad_sum, expected_ticks=8)["verdict"] == "INVALID"

    # 4. completed_count + in_flight_at_end_count != submitted_count.
    bad_inflight = deepcopy(valid_payload)
    bad_inflight["counts"]["in_flight_at_end_count"] = (
        1 - bad_inflight["counts"]["in_flight_at_end_count"]
    )
    assert cw.reduce_worker_contention(bad_inflight, expected_ticks=8)["verdict"] == "INVALID"

    # 5. Non-binary flag in tick_admitted.
    bad_flag = deepcopy(valid_payload)
    bad_flag["tick_costs"]["tick_admitted"][0] = 2
    assert cw.reduce_worker_contention(bad_flag, expected_ticks=8)["verdict"] == "INVALID"

    # 6. Tick with neither admission nor overflow (admitted + overflow == 0).
    hole_flag = deepcopy(valid_payload)
    hole_flag["tick_costs"]["tick_admitted"][0] = 0
    hole_flag["counts"]["submitted_count"] -= 1
    assert cw.reduce_worker_contention(hole_flag, expected_ticks=8)["verdict"] == "INVALID"

    # 7. Zeroed state_copy_wall_ns or state_copy_bytes on a measured tick.
    zero_cost = deepcopy(valid_payload)
    zero_cost["tick_costs"]["state_copy_wall_ns"][3] = 0
    assert cw.reduce_worker_contention(zero_cost, expected_ticks=8)["verdict"] == "INVALID"

    zero_bytes = deepcopy(valid_payload)
    zero_bytes["tick_costs"]["state_copy_bytes"][3] = 0
    assert cw.reduce_worker_contention(zero_bytes, expected_ticks=8)["verdict"] == "INVALID"

    # 8. Corrupted warm_records request_id or tick_index identity.
    bad_rec = deepcopy(valid_payload)
    bad_rec["warm_records"][0]["request_id"] = 999
    assert cw.reduce_worker_contention(bad_rec, expected_ticks=8)["verdict"] == "INVALID"

    bad_tick_rec = deepcopy(valid_payload)
    bad_tick_rec["warm_records"][0]["tick_index"] = 7
    assert cw.reduce_worker_contention(bad_tick_rec, expected_ticks=8)["verdict"] == "INVALID"


def _game_owner_with_hanging_worker(conn: Any) -> None:
    """Helper child process that spawns PersistentExactStopWorker, hangs it, and waits."""
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=1)
    session = _dummy_session()
    packet, copy_bytes = worker.copy_state_and_prepare_request(session, 0)
    hang_packet = {**packet, "cmd": "simulate_hang", "seconds": 60.0}
    worker.try_publish(hang_packet, copy_bytes, 0)
    conn.send({"game_pid": os.getpid(), "worker_pid": worker.worker_pid})
    conn.close()
    while True:
        time.sleep(1.0)


def test_killing_owning_game_process_reaps_worker_without_orphan() -> None:
    """Finding 5 regression: killing the game process must reap the persistent worker."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    game_proc = ctx.Process(target=_game_owner_with_hanging_worker, args=(child_conn,))
    game_proc.start()
    child_conn.close()
    try:
        assert parent_conn.poll(30.0), "game child did not start worker in time"
        pids = parent_conn.recv()
        worker_pid = int(pids["worker_pid"])
        assert psutil.pid_exists(worker_pid)

        # Abruptly kill the owning game process while the worker is mid-hang.
        game_proc.terminate()
        game_proc.join(timeout=5.0)
        if game_proc.is_alive():
            game_proc.kill()
            game_proc.join(timeout=5.0)

        # The worker's parent-death watchdog must terminate the worker promptly.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                proc = psutil.Process(worker_pid)
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.05)
        else:
            try:
                psutil.Process(worker_pid).kill()
            except psutil.Error:
                pass
            pytest.fail(f"worker process {worker_pid} survived termination of its owning game")
    finally:
        parent_conn.close()
        if game_proc.is_alive():
            game_proc.kill()
            game_proc.join(timeout=5.0)


def test_isolated_game_timeout_reaps_game_and_worker_descendants() -> None:
    """Finding 5 regression: isolated_game timeout must reap both game and worker processes."""
    fixtures, actions = _synthetic_fixtures()
    spec = {
        "arm": "w1_k1",
        "seed": _w1_manifest()["seeds"][0],
        "warmup_ticks": 0,
        "measured_ticks": 100,
        "fixtures": fixtures,
        "reference_actions": actions,
        "operator_mode": "synthetic_fast",
        "operator_kwargs": {"delay_seconds": 5.0},
        "prewarm_count": 1,
    }
    owner = psutil.Process(os.getpid())
    before_children = {p.pid for p in owner.children(recursive=True)}
    with pytest.raises(TimeoutError, match="isolated game arm exceeded"):
        isolated_game.run_isolated_arm(spec, timeout_seconds=0.3)

    leaked = [
        p
        for p in owner.children(recursive=True)
        if p.pid not in before_children and p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    ]
    assert not leaked, f"isolated_game leaked descendant processes: {leaked}"


def test_invalid_reply_and_reference_mismatch_fail_closed() -> None:
    fixtures, actions = _synthetic_fixtures()
    worker = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=actions,
        measured_ticks=4,
        operator_mode="synthetic_fast",
    )
    worker.start_and_prewarm(prewarm_count=1)
    try:
        session = _dummy_session()
        packet, copy_bytes = worker.copy_state_and_prepare_request(session, 0)
        bad_packet = {**packet, "cmd": "simulate_invalid_reply"}
        assert worker.try_publish(bad_packet, copy_bytes, 0) is True
        time.sleep(0.03)
        assert worker.poll(1) is None
        assert worker.invalid_reply_count == 1
        assert worker.fallback_count >= 1
    finally:
        worker.close()
    payload = worker.export_payload(game_rss_before=1000, game_rss_after=1000, game_maxrss=1000)
    assert payload["summary"]["verdict"] == "INVALID"

    # Reference action mismatch also fails closed.
    wrong_actions = {f["fixture_id"]: 0 for f in fixtures}  # worker returns 1, reference expects 0
    worker2 = cw.PersistentExactStopWorker(
        fixtures=fixtures,
        reference_actions=wrong_actions,
        measured_ticks=2,
        operator_mode="synthetic_fast",
    )
    with pytest.raises(RuntimeError, match="invalid reply"):
        worker2.start_and_prewarm(prewarm_count=1)
    assert worker2.reference_mismatch_count == 1


def test_w1_k1_panel_dispatches_interleaved_isolated_and_w1_k1_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _w1_manifest()
    dispatched: list[tuple[str, int]] = []

    def fake_isolated(spec: dict, timeout_seconds: float) -> dict:
        dispatched.append((spec["arm"], spec["seed"]))
        return {
            "seed": spec["seed"],
            "actions": [[0]],
            "state_digest": {"initial": "a", "final": "b"},
            "dto_digest": {"final": "c"},
            "trace": [{"target_ns": 0, "end_ns": 1, "spans": {}}],
        }

    monkeypatch.setattr(isolated_game, "run_isolated_arm", fake_isolated)
    monkeypatch.setattr(
        probe,
        "run_inline_arm",
        lambda spec: {
            "seed": spec["seed"],
            "actions": [[0]],
            "state_digest": {"initial": "a", "final": "b"},
            "dto_digest": {"final": "c"},
            "processing_wall_ns": [10] * spec["measured_ticks"],
            "processing_thread_cpu_ns": [5] * spec["measured_ticks"],
            "trace": [],
        },
    )
    raw = probe.run_panel(manifest, tmp_path)
    assert [arm for arm, _ in dispatched] == [
        "isolated",
        "w1_k1",
        "w1_k1",
        "isolated",
        "isolated",
        "w1_k1",
    ]
    assert len(raw["parity"]) == 3
    assert all(p["parity"]["pass"] for p in raw["parity"])
    assert (tmp_path / "cell-05.json").exists()
    assert (tmp_path / "observer-control.json").exists()


def test_all_three_committed_manifests_accepted_routed_and_reject_modified_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify bare W0, DTO-split W0/K0, and W1/K1 manifests are accepted, routed, and frozen."""
    from research.bl_served_search import dto_split, isolated_game, production_probe

    base_dir = Path(probe.__file__).parent
    bare = json.loads((base_dir / "served_tick_manifest.json").read_text())
    split = json.loads((base_dir / "served_dto_split_manifest.json").read_text())
    w1 = json.loads((base_dir / "served_tick_w1_k1_manifest.json").read_text())

    # 1. Each committed manifest is accepted by preflight.
    bare_pf = probe.preflight(bare)
    split_pf = probe.preflight(split)
    w1_pf = probe.preflight(w1)
    assert bare_pf["status"] == "PREFLIGHT_OK_NOT_TIMING" and bare_pf["cells"] == 12
    assert split_pf["status"] == "PREFLIGHT_OK_NOT_TIMING" and split_pf["cells"] == 12
    assert (
        w1_pf["status"] == "PREFLIGHT_OK_NOT_TIMING"
        and w1_pf["manifest_id"] == probe.W1_K1_MANIFEST_ID
        and w1_pf["cells"] == 6
    )

    # 2. Each rejects a modified copy (including study/id tampering).
    for original, field, bad_value in (
        (bare, "timeout_seconds", 3601),
        (bare, "decision", "tampered"),
        (bare, "extra_field", True),
        (split, "timeout_seconds", 3601),
        (split, "transport", {}),
        (split, "study", "served-tick-20261026"),
        (w1, "timeout_seconds", 3601),
        (w1, "fixtures_sha256", "0" * 64),
        (w1, "manifest_id", "tampered-id"),
    ):
        modified = deepcopy(original)
        modified[field] = bad_value
        with pytest.raises(ValueError, match="committed fixed panel"):
            probe.preflight(modified)

    # 3. Each routes to its own arm in panel execution and inline workload dispatch.
    calls: dict[str, list[str]] = {"inline": [], "isolated": [], "production": [], "dto_game": []}

    def fake_result(spec: dict) -> dict:
        return {
            "arm": spec["arm"],
            "seed": spec["seed"],
            "fps": 24,
            "warmup_ticks": spec["warmup_ticks"],
            "measured_ticks": spec["measured_ticks"],
            "actions": [[0]],
            "state_digest": {"initial": "a", "final": "b"},
            "dto_digest": {"final": "c"},
            "processing_wall_ns": [10] * int(spec["measured_ticks"]),
            "processing_thread_cpu_ns": [5] * int(spec["measured_ticks"]),
            "trace": [{"target_ns": 0, "end_ns": 1, "spans": {}}],
        }

    real_run_inline = probe.run_inline_arm

    def spy_dto_game(spec: dict) -> dict:
        calls["dto_game"].append(spec["arm"])
        return fake_result(spec)

    monkeypatch.setattr(dto_split, "run_game", spy_dto_game)
    dto_inline_out = real_run_inline(
        {"arm": "dto_split_w0_k0", "seed": 1, "warmup_ticks": 1, "measured_ticks": 2}
    )
    assert calls["dto_game"] == ["dto_split_w0_k0"]
    assert dto_inline_out["arm"] == "dto_split_w0_k0"

    def fake_inline(spec: dict, fresh_process: bool = False) -> dict:
        calls["inline"].append(spec["arm"])
        return fake_result(spec)

    def fake_isolated(spec: dict, timeout_seconds: float) -> dict:
        calls["isolated"].append(spec["arm"])
        if spec["arm"] == "w1_k1":
            assert spec.get("worker_mode") == "active"
            assert spec.get("operator_mode") == "exact_stop"
        return fake_result(spec)

    def fake_production(spec: dict, timeout_seconds: float) -> dict:
        calls["production"].append(spec["arm"])
        return fake_result(spec)

    monkeypatch.setattr(probe, "run_inline_arm", fake_inline)
    monkeypatch.setattr(isolated_game, "run_isolated_arm", fake_isolated)
    monkeypatch.setattr(production_probe, "run_production_arm", fake_production)

    calls["inline"].clear()
    calls["isolated"].clear()
    calls["production"].clear()
    bare_raw = probe.run_panel(bare, tmp_path / "bare")
    assert set(calls["isolated"]) == {"isolated"}
    assert set(calls["production"]) == {"production"}
    assert "W0 bare Watch6" in bare_raw["scope"]

    calls["inline"].clear()
    calls["isolated"].clear()
    calls["production"].clear()
    split_raw = probe.run_panel(split, tmp_path / "split")
    assert set(calls["isolated"]) == {"isolated", "dto_split_w0_k0"}
    assert calls["production"] == []
    assert "W0/K0 authoritative game/DTO split" in split_raw["scope"]

    calls["inline"].clear()
    calls["isolated"].clear()
    calls["production"].clear()
    w1_raw = probe.run_panel(w1, tmp_path / "w1")
    assert set(calls["isolated"]) == {"isolated", "w1_k1"}
    assert calls["production"] == []
    assert "W1/K1 contention-only Watch6 diagnostic" in w1_raw["scope"]
