"""Focused unit and integration coverage for the W1/K1 contention-only arm."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

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
    assert result["cells"] == 6
    assert result["fixtures_sha256"] == cw.Y1_FIXTURES_SHA256
    assert result["reference_sha256"] == cw.Y1_REFERENCE_SHA256


def test_w1_k1_manifest_preflight_rejects_tampered_workers_or_hashes() -> None:
    manifest = _w1_manifest()
    manifest["max_pending"] = 2
    with pytest.raises(ValueError, match="invalid W1/K1"):
        probe.preflight(manifest)

    manifest2 = _w1_manifest()
    manifest2["fixtures_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the committed fixed panel"):
        probe.preflight(manifest2)


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
        worker.step_tick(session, 0)
        time.sleep(0.03)
        worker.step_tick(session, 1)
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

    # 1. Worker idle (prewarmed real ExactT3Operator, no requests submitted during measured ticks).
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
    idle_summary = idle_result["worker_contention"]["summary"]
    assert idle_summary["verdict"] == "PASS"
    assert idle_summary["counts"]["prewarm_completed_count"] == 1
    assert idle_summary["counts"]["submitted_count"] == 0
    assert idle_summary["counts"]["actions_applied_count"] == 0
    assert idle_summary["service_latency"]["cold_worker_service_ns"]["count"] == 1

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
    assert active_summary["counts"]["submitted_count"] >= 1
    assert active_summary["counts"]["max_observed_pending"] == 1
    assert active_summary["counts"]["actions_applied_count"] == 0
    assert active_summary["tick_ipc_costs"]["state_copy_bytes"]["max"] > 0
    assert active_summary["rss"]["game_rss_bytes_after"] > 0
    assert active_summary["rss"]["worker_rss_bytes_after_prewarm"] > 0


def test_reducer_reports_all_w1_k1_fields_and_enforces_warm_s_bar() -> None:
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
        for tick in range(10):
            worker.step_tick(session, tick)
            time.sleep(0.002)
        for tick in range(10, 2400):
            worker.poll(tick)
    finally:
        worker.close()
    worker_payload = worker.export_payload(
        game_rss_before=100_000_000,
        game_rss_after=102_000_000,
        game_maxrss=105_000_000,
    )
    assert worker_payload["summary"]["verdict"] == "PASS"
    assert worker_payload["summary"]["service_latency"]["cold_s_ns"]["count"] == 3
    assert worker_payload["summary"]["service_latency"]["warm_s_ns"]["count"] >= 1

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
                        {"worker_contention": deepcopy(worker_payload)}
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

    # Exceeding the 250 ms warm S p95 bar must flip verdict to FAIL.
    slow_raw = deepcopy(raw)
    for rec in slow_raw["cells"][1]["result"]["worker_contention"]["warm_records"]:
        rec["S_ns"] = 260_000_000
    slow_report = probe.reduce_trace(slow_raw)
    assert slow_report["verdict"] == "FAIL"


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
