"""Untimed M1 wiring and evidence tests; no real workers/game or rollout cells."""

from __future__ import annotations

import copy
import queue
from collections import Counter
from types import SimpleNamespace

import pytest

from research.bl_k1 import m1
from research.bl_mac_latency import l1
from research.bl_served_search.exports import export_json
from research.bl_served_search.native_ruler import THREAD_ENV
from research.bl_served_search.ruler_timing import STAGES


def fixtures() -> list[dict]:
    return [
        {
            "fixture_id": f"{mix}-{length}",
            "mix": mix,
            "target_length": length,
            "legal_actions": [0, 1],
        }
        for mix in ("frozen", "mixed", "scripted")
        for length in (800, 1200, 1500)
    ]


def payload(fixture: dict) -> dict:
    native = 0 if fixture["mix"] == "scripted" else 1500
    return {
        "semantic": {
            "search_mode": "exact_stop",
            "fired": True,
            "chosen": 1,
            "bare": 1,
            "outcomes": {
                "0": {"status": "NOT_EVALUATED"},
                "1": {"frames_executed": 300, "survives": True},
            },
            "evaluation_order": [1],
            "branches_evaluated": 1,
            "branches_skipped": 1,
            "rollout_branch_frames": 300,
        },
        "native_v8": {
            "mode": "kernel",
            "kernel_rows": native,
            "stock_rows": 0,
            "noncandidate_rows": 0,
            "fallback_rows": 0,
            "calls": native // 5,
        },
        "stage_seconds": dict.fromkeys((*STAGES, "submit_visible", "submit_consumed"), 0.0),
        "worker_wall_seconds": 0.1,
        "worker_threads": {"environment": dict.fromkeys(THREAD_ENV, "1"), "torch": 1, "numba": 1},
        "execution_environment": {"torch_num_interop_threads": 1},
        "worker_started_ns": 1,
        "worker_completed_ns": 100000000,
        "worker_pid": 7,
    }


def sample(fixture: dict, phase: str, repeat: int) -> dict:
    return {
        "phase": phase,
        "repeat": repeat,
        "fixture_id": fixture["fixture_id"],
        "payload": payload(fixture),
        "admitted_ns": 0,
        "visible_ns": 200000000,
        "S_seconds": 0.2,
    }


def test_frozen_cells_and_interleaved_allocation() -> None:
    assert [(c["workers"], c["max_pending"]) for c in m1.cells()] == [
        (1, 1),
        (2, 2),
        (4, 4),
        (4, 8),
    ]
    jobs = m1.schedule(fixtures())
    assert len(jobs) == 270
    assert set(Counter(f["fixture_id"] for _, f in jobs).values()) == {30}
    for start in range(0, 270, 9):
        assert len({f["fixture_id"] for _, f in jobs[start : start + 9]}) == 9


def test_l1_constructs_actual_stop_factory(monkeypatch) -> None:
    from research.bl_served_search import ruler_operator

    marker = object()
    monkeypatch.setattr(ruler_operator, "create_stop_operator", lambda: marker)
    assert l1.DeviceOperator("cpu")._exact is marker


@pytest.mark.parametrize(
    "change", ["kernel_rows", "stock_rows", "fallback_rows", "chosen", "threads"]
)
def test_execution_counter_and_action_negative_controls(change: str) -> None:
    row = fixtures()[0]
    value = payload(row)
    if change == "chosen":
        value["semantic"]["chosen"] = 0
    elif change == "threads":
        value["worker_threads"]["torch"] = 2
    else:
        value["native_v8"][change] = 0 if change == "kernel_rows" else 1
    with pytest.raises(ValueError):
        m1.validate_sample(value, row, 1)


def test_scripted_counter_exemption_is_explicit() -> None:
    row = fixtures()[-1]
    m1.validate_sample(payload(row), row, 1)
    value = payload(row)
    value["native_v8"]["kernel_rows"] = 1
    with pytest.raises(ValueError):
        m1.validate_sample(value, row, 1)


def complete_unit() -> tuple[dict, dict]:
    fs = fixtures()
    cell = m1.cells()[0]
    samples = [sample(fs[0], "cold", 0)] + [sample(f, "warm", r) for r, f in m1.schedule(fs)]
    for index, item in enumerate(samples):
        admission = index * 200000000
        item.update(admitted_ns=admission, visible_ns=admission + 200000000)
        item["payload"].update(
            worker_started_ns=admission + 1, worker_completed_ns=admission + 199999999
        )
    end = samples[-1]["visible_ns"]
    ticks = [
        {
            "target_ns": i * 1000000000 // 24,
            "tick_started_ns": i * 1000000000 // 24,
            "finished_ns": i * 1000000000 // 24 + 1000000,
            "update_seconds": 0.001,
            "overrun": False,
        }
        for i in range((end * 24) // 1000000000 + 2)
    ]
    row = {
        **cell,
        "schema": m1.SCHEMA + "/cell",
        "namespace": "n",
        "samples": samples,
        "ticks": ticks,
        "epoch_ns": 0,
        "stop_reason": None,
        "mac_lease": {"id": "lease"},
        "host": [{"thermal": "nominal", "process_screen": {}, "coordinator_pid": 1}],
        "actual_max_pending": 1,
        "contention": [],
    }
    row["summary"] = m1.summarize(cell, samples, ticks, True)
    return row, {"namespace": "n", "fixtures": fs, "actions": {f["fixture_id"]: 1 for f in fs}}


def test_summary_keeps_cold_separate_dense_caps_descriptive_and_raw_audit() -> None:
    row, data = complete_unit()
    assert row["summary"]["verdict"] == "PASS"
    assert row["summary"]["S_seconds"]["count"] == 270
    assert len(row["summary"]["cold_process_wave"]) == 1
    assert row["summary"]["native_v8_totals_warm"]["stock_rows"] == 0
    m1.audit_unit(row, data)
    row["samples"][2]["payload"]["native_v8"].pop("kernel_rows")
    with pytest.raises(ValueError):
        m1.audit_unit(row, data)


@pytest.mark.parametrize(
    "change", ["overrun", "slow_update", "slow_doom", "missing", "counter_error"]
)
def test_frozen_bars_fail_closed(change: str) -> None:
    row, _ = complete_unit()
    cell = m1.cells()[2]
    samples = row["samples"][:1] * 4 + row["samples"][1:]
    ticks = row["ticks"]
    if change == "overrun":
        ticks[0]["overrun"] = True
    elif change == "slow_update":
        for tick in ticks:
            tick["update_seconds"] = 0.012
    elif change == "slow_doom":
        for s in samples:
            if s["fixture_id"] == "frozen-800" and s["phase"] == "warm":
                s["S_seconds"] = 5.01
    elif change == "missing":
        samples.pop()
    else:
        samples[5]["error"] = "missing kernel counters"
    expected = {"missing": "INCOMPLETE", "counter_error": "INVALID"}.get(change, "BLOCKED")
    assert m1.summarize(cell, samples, ticks, True)["verdict"] == expected


@pytest.mark.parametrize(
    "change", ["duplicate", "foreign", "tamper_time", "tamper_summary", "missing_host"]
)
def test_resumed_units_recompute_evidence(change: str) -> None:
    row, data = complete_unit()
    if change == "duplicate":
        row["samples"].append(copy.deepcopy(row["samples"][1]))
    elif change == "foreign":
        row["namespace"] = "other"
    elif change == "tamper_time":
        row["samples"][1]["visible_ns"] += 1
    elif change == "tamper_summary":
        row["summary"]["S_seconds"]["p99_seconds"] = 0
    else:
        row["host"] = []
    with pytest.raises(ValueError):
        m1.audit_unit(row, data)


def test_ledgers_verify_exact_bytes_refuse_foreign_duplicate_and_partial(tmp_path) -> None:
    row, _ = complete_unit()
    unit = tmp_path / "cpu-w1-k1.json"
    export_json(unit, row)
    ledger = tmp_path / "ledger.json"
    m1.make_ledger(ledger, "n", [unit])
    assert set(m1.verify_ledger(ledger, "n")) == {"cpu-w1-k1"}
    with pytest.raises(ValueError, match="namespace"):
        m1.verify_ledger(ledger, "other")
    with pytest.raises(ValueError, match="duplicate"):
        m1.make_ledger(tmp_path / "duplicate.json", "n", [unit, unit])
    unit.write_text("{}")
    with pytest.raises(ValueError):
        m1.verify_ledger(ledger, "n")


def test_empty_ledger_and_incomplete_reduce_never_pass(tmp_path) -> None:
    ledger = tmp_path / "empty.json"
    m1.make_ledger(ledger, "n", [])
    assert m1.verify_ledger(ledger, "n") == {}
    assert m1.reduce_units([], "n")["verdict"] == "INCOMPLETE"


def test_legacy_reference_uses_action_not_full_stop_semantic(monkeypatch, tmp_path) -> None:
    reference = {"records": [{"fixture_id": "frozen-800", "batches": [payload(fixtures()[0])]}]}
    monkeypatch.setattr(l1, "_read_completed_json", lambda path: (reference, "sha"))
    assert l1._semantic_reference({"path": str(tmp_path / "ref")}) == {"frozen-800": 1}


@pytest.mark.parametrize("fault", ["none", "counter", "cap", "load_failure"])
@pytest.mark.parametrize("cell", m1.cells(), ids=lambda c: c["cell_id"])
def test_full_orchestration_synthetic_only(monkeypatch, cell, fault) -> None:
    """Exercise actual queue/clock orchestration with no native processes/timers."""
    fs = fixtures()
    data = {"namespace": "n", "fixtures": fs, "actions": {f["fixture_id"]: 1 for f in fs}}
    clock = SimpleNamespace(ns=1000000000)
    monkeypatch.setattr(m1.time, "monotonic_ns", lambda: clock.ns)
    monkeypatch.setattr(m1.time, "monotonic", lambda: clock.ns / 1e9)
    monkeypatch.setattr(
        m1.time, "sleep", lambda seconds: setattr(clock, "ns", clock.ns + int(seconds * 1e9))
    )

    class Event:
        def set(self):
            pass

    load_times = {}

    class Queue:
        def close(self):
            pass

        def get(self, **kwargs):
            if fault == "load_failure":
                raise queue.Empty
            return {
                "started_ns": load_times["start"],
                "finished_ns": load_times["start"] + 30000000000,
                "cpu_seconds": 29.5,
                "pid": 33,
            }

    class Process:
        def start(self):
            load_times["start"] = clock.ns

        def join(self):
            pass

        def is_alive(self):
            return clock.ns < load_times["start"] + 30000000000

    monkeypatch.setattr(
        m1.multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(
            Event=Event, Queue=Queue, Process=lambda **kwargs: Process()
        ),
    )

    class Future:
        def __init__(self, fixture, index):
            self.fixture, self.index = fixture, index

        def done(self):
            return fault not in ("counter", "cap") or self.index == 0

        def result(self):
            if fault == "cap":
                raise RuntimeError("measurement stopped before next rollout step")
            result = payload(self.fixture)
            if fault == "counter":
                result["native_v8"].pop("kernel_rows")
            result.update(
                worker_started_ns=clock.ns - 1,
                worker_completed_ns=clock.ns,
                worker_pid=10 + self.index % cell["workers"],
            )
            return result

    class Pool:
        index = 0
        joined = False

        def submit(self, function, fixture):
            clock.ns += 1
            result = Future(fixture, self.index)
            self.index += 1
            return result

        def shutdown(self, **kwargs):
            self.joined = True

    pool = Pool()
    monkeypatch.setattr(m1.concurrent.futures, "ProcessPoolExecutor", lambda **kwargs: pool)
    monkeypatch.setattr(m1, "new_watch", lambda seed: (object(), {"watch_roster": "mixed"}))

    def tick(session):
        clock.ns += 1000000
        return 0.001

    monkeypatch.setattr(m1, "timed_tick", tick)

    monitors = []

    class Monitor:
        error = None
        rows = [{"process_screen": {}, "coordinator_pid": 1}]
        thread = SimpleNamespace(start=lambda: None)

        def __init__(self, observe):
            monitors.append(self)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(m1, "HostMonitor", Monitor)
    row = m1.run_cell(cell, data, float("inf"), lambda: {})
    assert monitors[0].closed
    if fault == "load_failure" and cell["cell_id"] == "cpu-w4-k4":
        assert row["summary"]["verdict"] == "INCOMPLETE"
        assert "missing contention" in row["stop_reason"] and pool.joined
        return
    if fault in ("counter", "cap"):
        assert row["summary"]["verdict"] == ("INVALID" if fault == "counter" else "INCOMPLETE")
        assert pool.joined
        return
    assert row["summary"]["verdict"] == "PASS"
    assert row["actual_max_pending"] == cell["max_pending"]
    assert len(row["samples"]) == 270 + cell["max_pending"]
    assert row["pending_at_stop"] == 0 and pool.joined
    row["mac_lease"] = {"id": "test-only"}
    m1.audit_unit(row, data)


def test_watch_disables_inherited_pathb_and_requires_no_transform(monkeypatch) -> None:
    import os

    from research.t15_mixed_roster_20261017 import episodes

    snake = SimpleNamespace(auto_respawn=True)
    session = SimpleNamespace(
        speed=24, game=SimpleNamespace(snakes=[snake] * 6), set_speed=lambda speed: None
    )
    observed = []

    def new_session(seed, count):
        observed.append(os.environ["SNAKE_SERVE_PATHB"])
        return session

    monkeypatch.setenv("SNAKE_SERVE_PATHB", "true")
    monkeypatch.setattr(episodes, "new_session", new_session)
    monkeypatch.setattr(episodes, "roster_identity", lambda session: {"watch_roster": "mixed"})
    assert m1.new_watch(123)[0] is session
    assert observed == ["0"] and os.environ["SNAKE_SERVE_PATHB"] == "true"
    snake.action_transform = object()
    with pytest.raises(ValueError, match="assisted"):
        m1.new_watch(123)


def test_reducer_refuses_field_only_pass() -> None:
    row, _ = complete_unit()
    with pytest.raises(ValueError, match="raw evidence"):
        m1.reduce_units([row], "n")


def test_recovery_check_deducts_cap_and_rejects_second_completion(
    monkeypatch, tmp_path, capsys
) -> None:
    data = {"namespace": "n", "sources": {}}
    monkeypatch.setattr(m1, "inputs", lambda *args: data)
    monkeypatch.setattr(m1, "verify_ledger", lambda *args: {})
    previous = {
        "schema": m1.SCHEMA,
        "namespace": "n",
        "attempt_number": 1,
        "verdict": "INCOMPLETE",
        "verified_stop": True,
        "mechanical_failure": True,
        "completed_unit_exports": {},
        "charged_seconds": 123.0,
    }
    path = tmp_path / "previous.json"
    export_json(path, previous)
    argv = [
        "m1",
        "--check",
        "--fixtures",
        "f",
        "--reference",
        "r",
        "--ledger",
        "l",
        "--previous-attempt",
        str(path),
    ]
    monkeypatch.setattr("sys.argv", argv)
    m1.main()
    checked = capsys.readouterr().out
    assert '"timed_cells_run": false' in checked
    assert '"remaining_seconds": 14277.0' in checked
    previous["attempt_number"] = 2
    second = tmp_path / "second.json"
    export_json(second, previous)
    monkeypatch.setattr("sys.argv", argv[:-1] + [str(second)])
    with pytest.raises(ValueError, match="one verified mechanical"):
        m1.main()


def test_cap_worker_failure_is_incomplete_not_invalid() -> None:
    row, _ = complete_unit()
    row["samples"][-1] = {
        "phase": "warm",
        "fixture_id": "scripted-1500",
        "repeat": 29,
        "mechanical_error": "measurement stopped before next rollout step",
    }
    assert (
        m1.summarize(m1.cells()[0], row["samples"], row["ticks"], False)["verdict"] == "INCOMPLETE"
    )


def test_check_entrypoint_cannot_dispatch_without_explicit_mode(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["m1", "--fixtures", "f", "--reference", "r"])
    with pytest.raises(SystemExit):
        m1.main()


def test_audit_rejects_truncated_tick_residence() -> None:
    row, data = complete_unit()
    row["ticks"] = row["ticks"][:1]
    row["summary"] = m1.summarize(m1.cells()[0], row["samples"], row["ticks"], True)
    with pytest.raises(ValueError, match="cover all result residence"):
        m1.audit_unit(row, data)


def test_host_ancestry_excludes_owned_workers_but_refuses_foreign_load() -> None:
    screen = {
        "processes": (
            "PID PPID %CPU COMM\n10 1 1 python\n11 10 99 worker\n"
            "12 11 99 child\n13 1 60 foreign\n"
        ),
        "other_heavy_processes": [{"pid": 11}, {"pid": 12}, {"pid": 13}],
    }
    assert m1.foreign_load(screen, 10) == [{"pid": 13}]
