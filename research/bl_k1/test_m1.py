"""Untimed M1 wiring and evidence tests; no real workers/game or rollout cells."""

from __future__ import annotations

import copy
import queue
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_k1 import m1
from research.bl_mac_latency import l1
from research.bl_served_search.exports import export_json
from research.bl_served_search.native_ruler import THREAD_ENV
from research.bl_served_search.ruler_timing import STAGES


@pytest.fixture(autouse=True)
def isolate_judged2_artifact_root(monkeypatch, tmp_path) -> None:
    """Never let pinned-proof tests read a live artifact receipt directory."""
    monkeypatch.setattr(m1, "JUDGED2_JUDGMENT_PATH", tmp_path / "JUDGMENT_M1_ATTEMPT5.md")


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
        "host": [
            {
                "monotonic_ns": end,
                "foreign_heavy_processes": [],
                "thermal": "nominal",
                "process_screen": {},
                "coordinator_pid": 1,
            }
        ],
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
        rows = [
            {
                "monotonic_ns": 1,
                "foreign_heavy_processes": [],
                "process_screen": {},
                "coordinator_pid": 1,
            }
        ]
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

    # Match ordinary AISnake: the optional respawn opt-out is absent.
    snake = SimpleNamespace()
    session = SimpleNamespace(
        mode="watch",
        playing=True,
        speed=24,
        game=SimpleNamespace(snakes=[snake] * 6),
        set_speed=lambda speed: None,
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
    snake.auto_respawn = False
    with pytest.raises(ValueError, match="continuous-respawn"):
        m1.new_watch(123)
    del snake.auto_respawn
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


@pytest.mark.parametrize("cleanup_error", [False, True])
def test_partial_setup_cleanup_truthfully_records_stop(monkeypatch, cleanup_error) -> None:
    """A queue setup exception must still drain an already constructed pool."""
    monkeypatch.setattr(m1, "new_watch", lambda seed: (object(), {}))
    pool = SimpleNamespace(joined=False)

    def shutdown(**kwargs):
        if cleanup_error:
            raise RuntimeError("unverified pool cleanup")
        pool.joined = True

    pool.shutdown = shutdown
    monkeypatch.setattr(m1.concurrent.futures, "ProcessPoolExecutor", lambda **kwargs: pool)
    monkeypatch.setattr(
        m1.multiprocessing,
        "get_context",
        lambda method: SimpleNamespace(
            Event=lambda: SimpleNamespace(set=lambda: None),
            Queue=lambda: (_ for _ in ()).throw(RuntimeError("queue setup")),
        ),
    )
    state = {"verified_stop": True}
    data = {"namespace": "n", "fixtures": fixtures()}
    if cleanup_error:
        with pytest.raises(RuntimeError, match="unverified pool"):
            m1.run_cell(m1.cells()[0], data, float("inf"), lambda: {}, state)
        assert state["verified_stop"] is False
    else:
        row = m1.run_cell(m1.cells()[0], data, float("inf"), lambda: {}, state)
        assert row["summary"]["verdict"] == "INCOMPLETE"
        assert state["verified_stop"] is True and pool.joined


def test_prepool_exception_receipt_keeps_verified_stop(monkeypatch, tmp_path) -> None:
    from contextlib import contextmanager

    from research import mac_lease
    from research.bl_mac_latency import host
    from research.bl_served_search.exports import verify_export

    @contextmanager
    def held(*args, **kwargs):
        yield SimpleNamespace(receipt={"id": "test-only"})

    monkeypatch.setattr(mac_lease, "held", held)
    monkeypatch.setattr(host, "snapshot", lambda: {"quiet": True})
    monkeypatch.setattr(m1, "wait_quiet", lambda *args, **kwargs: {"quiet": True, "samples": []})
    monkeypatch.setattr(m1, "inputs", lambda *args: {"namespace": "n", "sources": {}})
    monkeypatch.setattr(m1, "sources", lambda: {})
    monkeypatch.setattr(m1, "verify_ledger", lambda *args: {})

    def fail(seed):
        raise AttributeError("pre-pool Watch construction failure")

    monkeypatch.setattr(m1, "new_watch", fail)
    ledger = tmp_path / "empty.json"
    ledger.write_text("{}")
    out = tmp_path / "attempt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--run",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            str(ledger),
            "--out",
            str(out),
        ],
    )
    m1.main()
    verify_export(out / "receipt.json")
    import json

    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["verified_stop"] is True
    assert receipt["mechanical_failure"] is True
    assert receipt["verdict"] == "INCOMPLETE"
    assert receipt["cells"] == [] and receipt["completed_unit_exports"] == {}


def test_reconciliation_refuses_other_failure_or_target(monkeypatch, tmp_path) -> None:
    """A generic asserted PASS/stop certificate cannot authorize a retry."""
    proof = {
        "schema": m1.SCHEMA + "/precell-reconciliation",
        "previous_receipt_path": str(tmp_path / "receipt.json"),
        "reviewer": "Codex-V",
        "status": "VERIFIED_PRECELL_MECHANICAL_STOP",
        "target_namespace": "n",
        "previous_receipt_sha256": "other",
    }
    monkeypatch.setattr(l1, "_read_completed_json", lambda path: (proof, "other"))
    with pytest.raises(ValueError, match="exact audited"):
        m1.reconciled_precell_stop(tmp_path / "proof.json", {"namespace": "n"})


@pytest.mark.parametrize("change", [None, "namespace", "cells", "retry", "debit", "source"])
def test_reconciliation_is_empty_only_and_keeps_cumulative_debit(
    monkeypatch, tmp_path, change
) -> None:
    import json

    previous = {
        "schema": m1.SCHEMA,
        "namespace": "old",
        "attempt_number": 1,
        "verdict": "INCOMPLETE",
        "verified_stop": False,
        "mechanical_failure": True,
        "reason": "AttributeError: 'AISnake' object has no attribute 'auto_respawn'",
        "cells": [],
        "completed_unit_exports": {},
        "charged_seconds": 0.82,
    }
    if change == "cells":
        previous["cells"] = [{"cell_id": "cpu-w1-k1"}]
    if change == "retry":
        previous["attempt_number"] = 2
    prior = tmp_path / "attempt"
    prior.mkdir()
    receipt_sha = export_json(prior / "receipt.json", previous)["sha256"]
    attempt = {"namespace": "old", "sources": {"research/bl_k1/m1.py": "old-source"}}
    attempt_sha = export_json(prior / "attempt.json", attempt)["sha256"]
    log = tmp_path / "attempt.log"
    log.write_text("mechanical failure")
    monkeypatch.setattr(m1, "PRECELL_RECEIPT_SHA256", receipt_sha)
    monkeypatch.setattr(m1, "PRECELL_ATTEMPT_SHA256", attempt_sha)
    monkeypatch.setattr(m1, "PRECELL_LOG_SHA256", m1.digest(log))
    monkeypatch.setattr(
        m1, "PRECELL_SOURCE_SHA256", "wrong" if change == "source" else "old-source"
    )
    proof = {
        "schema": m1.SCHEMA + "/precell-reconciliation",
        "reviewer": "Codex-V",
        "status": "VERIFIED_PRECELL_MECHANICAL_STOP",
        "target_namespace": "wrong" if change == "namespace" else "new",
        "previous_receipt_path": str(prior / "receipt.json"),
        "previous_receipt_sha256": receipt_sha,
        "previous_attempt_sha256": attempt_sha,
        "previous_log_sha256": m1.digest(log),
        "charged_seconds": 0 if change == "debit" else 0.82,
    }
    path = tmp_path / "proof.json"
    export_json(path, proof)
    if change:
        with pytest.raises(ValueError, match="exact audited"):
            m1.reconciled_precell_stop(path, {"namespace": "new"})
        return
    data = {"namespace": "new", "sources": {}}
    assert m1.reconciled_precell_stop(path, data) == previous
    monkeypatch.setattr(m1, "inputs", lambda *args: data)
    ledger = tmp_path / "ledger.json"
    m1.make_ledger(ledger, "new", [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--check",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            str(ledger),
            "--reconciled-precell-stop",
            str(path),
        ],
    )
    m1.main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--run",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--out",
            str(tmp_path / "new-attempt"),
            "--ledger",
            str(ledger),
            "--reconciled-precell-stop",
            str(path),
        ],
    )

    # Stop before the lease; prove recovery cannot reset the debit/envelope.
    def reject_lease(*args, **kwargs):
        assert kwargs["seconds"] == m1.CAP_SECONDS - 0.82
        raise RuntimeError("test refuses lease")

    from research import mac_lease

    monkeypatch.setattr(mac_lease, "held", reject_lease)
    with pytest.raises(RuntimeError, match="test refuses lease"):
        m1.main()
    assert json.loads(path.read_text())["charged_seconds"] == 0.82


@pytest.fixture
def judged_case(monkeypatch, tmp_path):
    """Completed mock exports exercise the real sidecar readers without workers."""
    previous = {
        "schema": m1.SCHEMA,
        "namespace": "old",
        "attempt_number": 2,
        "verdict": "INCOMPLETE",
        "verified_stop": True,
        "mechanical_failure": True,
        "completed_unit_exports": {},
        "reconciled_precell_stop_sha256": m1.JUDGED_PRECELL_SHA256,
        "charged_seconds": m1.JUDGED_CHARGED_SECONDS,
        "cells": [{"cell_id": "cpu-w1-k1", "complete": False}],
    }
    prior = tmp_path / "attempt-03"
    prior.mkdir()
    attempt = {
        "namespace": "old",
        "sources": {"research/bl_k1/m1.py": m1.JUDGED_SOURCE_SHA256},
    }
    cell = {"stop_reason": "foreign heavy load during M1", "summary": {"complete": False}}
    judgment = tmp_path / "judgment.md"
    judgment.write_text("Fable judgment")
    log = prior.with_suffix(".log")
    log.write_text("mechanical stop")
    return SimpleNamespace(
        previous=previous,
        attempt=attempt,
        cell=cell,
        prior=prior,
        judgment=judgment,
        log=log,
        proof_path=tmp_path / "proof.json",
        data={"namespace": "new", "sources": {}},
    )


def write_judged_case(monkeypatch, case) -> dict:
    """Pin mock artifacts as the historical constants, preserving real IO checks."""
    receipt_sha = export_json(case.prior / "receipt.json", case.previous)["sha256"]
    attempt_sha = export_json(case.prior / "attempt.json", case.attempt)["sha256"]
    cell_sha = export_json(case.prior / "cpu-w1-k1.json", case.cell)["sha256"]
    monkeypatch.setattr(m1, "JUDGED_RECEIPT_SHA256", receipt_sha)
    monkeypatch.setattr(m1, "JUDGED_ATTEMPT_SHA256", attempt_sha)
    monkeypatch.setattr(m1, "JUDGED_LOG_SHA256", m1.digest(case.log))
    monkeypatch.setattr(m1, "JUDGED_CELL_SHA256", {"cpu-w1-k1": cell_sha})
    return {
        "schema": m1.SCHEMA + "/judged-recovery",
        "judge": "Fable-M1-attempt4",
        "status": "JUDGED_FOREIGN_LOAD_STOP",
        "target_namespace": "new",
        "judgment_path": str(case.judgment),
        "judgment_sha256": m1.digest(case.judgment),
        "previous_receipt_path": str(case.prior / "receipt.json"),
        "previous_receipt_sha256": receipt_sha,
        "previous_attempt_sha256": attempt_sha,
        "previous_log_sha256": m1.digest(case.log),
        "previous_cell_sha256": {"cpu-w1-k1": cell_sha},
        "charged_seconds": m1.JUDGED_CHARGED_SECONDS,
        "next_attempt_number": 3,
        "final_attempt": True,
    }


@pytest.fixture
def judged2_case(monkeypatch, tmp_path):
    """A complete mock of attempt-04 exercises the second pinned proof path."""
    prior = tmp_path / "bl-m1-attempt-04"
    prior.mkdir()
    judgment = m1.JUDGED2_JUDGMENT_PATH
    judgment.write_text("Fable attempt-05 judgment")
    log = tmp_path / "bl-m1-attempt-04.prep2.log"
    log.write_text("mechanical foreign-load stop")
    previous = {
        "schema": m1.SCHEMA,
        "namespace": "attempt-04-namespace",
        "attempt_number": 3,
        "final_attempt": True,
        "verdict": "INCOMPLETE",
        "verified_stop": True,
        "mechanical_failure": True,
        "completed_unit_exports": {},
        "judged_recovery_sha256": "prior-judged-proof",
        "charged_seconds": 36.47473570803413,
        "cells": [{"cell_id": "cpu-w1-k1", "complete": False}],
    }
    attempt = {"namespace": previous["namespace"], "sources": {"research/bl_k1/m1.py": "old"}}
    cell = {"stop_reason": "foreign heavy load during M1", "summary": {"complete": False}}
    receipt_sha = export_json(prior / "receipt.json", previous)["sha256"]
    attempt_sha = export_json(prior / "attempt.json", attempt)["sha256"]
    cell_sha = export_json(prior / "cpu-w1-k1.json", cell)["sha256"]
    monkeypatch.setattr(m1, "JUDGED2_RECEIPT_SHA256", receipt_sha)
    monkeypatch.setattr(m1, "JUDGED2_ATTEMPT_SHA256", attempt_sha)
    monkeypatch.setattr(m1, "JUDGED2_LOG_SHA256", m1.digest(log))
    monkeypatch.setattr(m1, "JUDGED2_CELL_SHA256", {"cpu-w1-k1": cell_sha})
    monkeypatch.setattr(m1, "JUDGED2_SOURCE_SHA256", "old")
    monkeypatch.setattr(m1, "JUDGED2_PREVIOUS_JUDGED_SHA256", "prior-judged-proof")
    data = {
        "namespace": "new",
        "sources": {"research/bl_k1/m1.py": m1.digest(Path(m1.__file__))},
    }
    proof = {
        "schema": m1.SCHEMA + "/judged-recovery",
        "judge": "Fable-M1-attempt5",
        "status": "JUDGED_FOREIGN_LOAD_STOP_2",
        "judgment_path": str(judgment),
        "judgment_sha256": m1.digest(judgment),
        "previous_receipt_path": str(prior / "receipt.json"),
        "previous_receipt_sha256": receipt_sha,
        "previous_attempt_sha256": attempt_sha,
        "previous_log_path": str(log),
        "previous_log_sha256": m1.digest(log),
        "previous_cell_sha256": {"cpu-w1-k1": cell_sha},
        "previous_namespace": previous["namespace"],
        "previous_judged_recovery_sha256": "prior-judged-proof",
        "charged_seconds": previous["charged_seconds"],
        "remaining_seconds": m1.CAP_SECONDS - previous["charged_seconds"],
        "next_attempt_number": 4,
        "final_attempt": True,
        "restart_policy_required": True,
        "target_namespace": "new",
        "transferred_cell_or_timing_evidence": False,
        "operator_dispatch_once": True,
    }
    return SimpleNamespace(
        previous=previous,
        attempt=attempt,
        proof=proof,
        path=tmp_path / "judged2-proof.json",
        data=data,
    )


def test_judged2_recovery_happy_path(judged2_case) -> None:
    export_json(judged2_case.path, judged2_case.proof)
    assert m1.judged_recovery(judged2_case.path, judged2_case.data) == judged2_case.previous


@pytest.mark.parametrize(
    "change",
    ["hash", "attempt", "final", "judgment", "old_source", "missing_policy"],
)
def test_judged2_recovery_refuses_required_boundaries(monkeypatch, judged2_case, change) -> None:
    case = judged2_case
    if change == "hash":
        case.proof["previous_receipt_sha256"] = "wrong"
    elif change == "attempt":
        case.previous["attempt_number"] = 2
    elif change == "final":
        case.previous.pop("final_attempt")
    elif change == "judgment":
        case.proof["judgment_sha256"] = "wrong"
    elif change == "old_source":
        case.data["sources"]["research/bl_k1/m1.py"] = "old"
    elif change == "missing_policy":
        monkeypatch.setattr(m1, "RESTART_POLICY", {})
    if change in {"attempt", "final"}:
        receipt_path = Path(case.proof["previous_receipt_path"])
        for candidate in (
            receipt_path,
            receipt_path.with_suffix(receipt_path.suffix + ".sha256"),
            receipt_path.with_suffix(receipt_path.suffix + ".complete"),
        ):
            candidate.unlink()
        receipt_sha = export_json(receipt_path, case.previous)["sha256"]
        case.proof["previous_receipt_sha256"] = receipt_sha
        monkeypatch.setattr(m1, "JUDGED2_RECEIPT_SHA256", receipt_sha)
    export_json(case.path, case.proof)
    with pytest.raises(ValueError, match="exact audited attempt-04 foreign-load stop"):
        m1.judged_recovery(case.path, case.data)


@pytest.mark.parametrize(
    "change",
    [
        "receipt_hash",
        "attempt_hash",
        "log_hash",
        "cell_hash",
        "missing_cell",
        "missing_complete",
        "attempt_number",
        "verified_stop",
        "completed_exports",
        "target_namespace",
        "judgment_hash",
        "schema",
        "judge",
        "status",
        "next_attempt",
        "final_attempt",
        "source",
        "previous_schema",
        "namespace",
        "verdict",
        "mechanical_failure",
        "precell",
        "debit",
        "proof_debit",
        "cells_count",
        "cell_id",
        "cell_complete",
        "stop_reason",
        "summary_complete",
    ],
)
def test_judged_recovery_refusals(monkeypatch, judged_case, change) -> None:
    case = judged_case
    previous_changes = {
        "attempt_number": ("attempt_number", 1),
        "verified_stop": ("verified_stop", False),
        "completed_exports": ("completed_unit_exports", {"cpu-w1-k1": {}}),
        "previous_schema": ("schema", "foreign"),
        "namespace": ("namespace", "foreign"),
        "verdict": ("verdict", "PASS"),
        "mechanical_failure": ("mechanical_failure", False),
        "precell": ("reconciled_precell_stop_sha256", "wrong"),
        "debit": ("charged_seconds", 0.82),
        "cells_count": ("cells", []),
        "cell_id": ("cells", [{"cell_id": "cpu-w2-k2", "complete": False}]),
        "cell_complete": ("cells", [{"cell_id": "cpu-w1-k1", "complete": True}]),
    }
    if change in previous_changes:
        key, value = previous_changes[change]
        case.previous[key] = value
    if change == "source":
        case.attempt["sources"]["research/bl_k1/m1.py"] = "wrong"
    if change == "stop_reason":
        case.cell["stop_reason"] = "other stop"
    if change == "summary_complete":
        case.cell["summary"]["complete"] = True
    proof = write_judged_case(monkeypatch, case)
    proof_changes = {
        "receipt_hash": ("previous_receipt_sha256", "wrong"),
        "attempt_hash": ("previous_attempt_sha256", "wrong"),
        "log_hash": ("previous_log_sha256", "wrong"),
        "cell_hash": ("previous_cell_sha256", {}),
        "target_namespace": ("target_namespace", "foreign"),
        "judgment_hash": ("judgment_sha256", "wrong"),
        "schema": ("schema", "foreign"),
        "judge": ("judge", "other"),
        "status": ("status", "other"),
        "next_attempt": ("next_attempt_number", 4),
        "final_attempt": ("final_attempt", False),
        "proof_debit": ("charged_seconds", 0.82),
    }
    if change in proof_changes:
        key, value = proof_changes[change]
        proof[key] = value
    if change == "missing_cell":
        (case.prior / "cpu-w1-k1.json").unlink()
    if change == "missing_complete":
        (case.prior / "cpu-w1-k1.json.complete").unlink()
    export_json(case.proof_path, proof)
    with pytest.raises(ValueError, match="exact audited attempt-03 foreign-load stop"):
        m1.judged_recovery(case.proof_path, case.data)


def test_judged_recovery_check_and_final_receipt(monkeypatch, judged_case, capsys) -> None:
    import json
    from contextlib import contextmanager

    from research import mac_lease
    from research.bl_mac_latency import host

    case = judged_case
    proof = write_judged_case(monkeypatch, case)
    export_json(case.proof_path, proof)
    assert m1.judged_recovery(case.proof_path, case.data) == case.previous
    monkeypatch.setattr(m1, "inputs", lambda *args: case.data)
    monkeypatch.setattr(m1, "sources", lambda: {})
    ledger = case.prior.parent / "empty-ledger.json"
    m1.make_ledger(ledger, "new", [])
    argv = [
        "m1",
        "--check",
        "--fixtures",
        "f",
        "--reference",
        "r",
        "--ledger",
        str(ledger),
        "--judged-recovery",
        str(case.proof_path),
    ]
    monkeypatch.setattr("sys.argv", argv)
    m1.main()
    check = json.loads(capsys.readouterr().out)
    assert check["skipped"] == [] and check["timed_cells_run"] is False
    assert check["remaining_seconds"] == pytest.approx(14394.000786, abs=1e-6)

    @contextmanager
    def held(*args, **kwargs):
        assert kwargs["seconds"] == m1.CAP_SECONDS - m1.JUDGED_CHARGED_SECONDS
        yield SimpleNamespace(receipt={"id": "test-only"})

    monkeypatch.setattr(mac_lease, "held", held)
    monkeypatch.setattr(host, "snapshot", lambda: {"quiet": True})
    monkeypatch.setattr(m1, "wait_quiet", lambda *args, **kwargs: {"quiet": True, "samples": []})
    clock = iter([100.0, 102.0])
    monkeypatch.setattr(m1.time, "monotonic", lambda: next(clock, 102.0))

    def stop(*args):
        raise RuntimeError("mock mechanical stop; no pool or timing run")

    monkeypatch.setattr(m1, "run_cell", stop)
    out = case.prior.parent / "attempt-04"
    monkeypatch.setattr("sys.argv", [argv[0], "--run", *argv[2:], "--out", str(out)])
    m1.main()
    attempt, _ = l1._read_completed_json(out / "attempt.json")
    receipt, _ = l1._read_completed_json(out / "receipt.json")
    assert attempt["judged_recovery_sha256"] == m1.digest(case.proof_path)
    assert receipt["judged_recovery_sha256"] == m1.digest(case.proof_path)
    assert receipt["attempt_number"] == 3 and receipt["final_attempt"] is True
    assert receipt["charged_seconds"] == pytest.approx(m1.JUDGED_CHARGED_SECONDS + 2)
    assert receipt["completed_unit_exports"] == {} and receipt["cells"] == []
    assert receipt["mechanical_failure"] is True


def test_judged_recovery_refuses_nonempty_ledger(monkeypatch, judged_case) -> None:
    case = judged_case
    proof = write_judged_case(monkeypatch, case)
    export_json(case.proof_path, proof)
    monkeypatch.setattr(m1, "inputs", lambda *args: case.data)
    monkeypatch.setattr(m1, "verify_ledger", lambda *args: {"cpu-w1-k1": {}})
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--check",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            "l",
            "--judged-recovery",
            str(case.proof_path),
        ],
    )
    with pytest.raises(ValueError, match="judged recovery requires an empty new ledger"):
        m1.main()


@pytest.mark.parametrize("other", ["--previous-attempt", "--reconciled-precell-stop"])
def test_judged_recovery_is_mutually_exclusive(monkeypatch, other) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--check",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--judged-recovery",
            "proof",
            other,
            "other",
        ],
    )
    with pytest.raises(SystemExit):
        m1.main()


def test_judged_recovery_refused_for_smoke(monkeypatch) -> None:
    monkeypatch.setattr(m1, "inputs", lambda *args: {})
    monkeypatch.setattr(
        "sys.argv",
        ["m1", "--smoke", "--fixtures", "f", "--reference", "r", "--judged-recovery", "p"],
    )
    with pytest.raises(SystemExit):
        m1.main()


def test_wait_quiet_requires_three_foreign_free_samples(monkeypatch) -> None:
    """The quiet gate keeps every observed high-load process, including foreign ones."""
    clock = SimpleNamespace(now_ns=0)
    monkeypatch.setattr(m1.time, "monotonic_ns", lambda: clock.now_ns)
    monkeypatch.setattr(m1.time, "monotonic", lambda: clock.now_ns / 1e9)
    monkeypatch.setattr(
        m1.time,
        "sleep",
        lambda seconds: setattr(clock, "now_ns", clock.now_ns + int(seconds * 1e9)),
    )
    screens = iter(
        [
            "PID PPID %CPU COMM\n10 1 1 python\n20 1 25 daemon\n",
            "PID PPID %CPU COMM\n10 1 1 python\n",
            "PID PPID %CPU COMM\n10 1 1 python\n",
            "PID PPID %CPU COMM\n10 1 1 python\n",
        ]
    )
    monkeypatch.setattr(m1.os, "getpid", lambda: 10)
    receipt = m1.wait_quiet(
        lambda: {"process_screen": {"processes": next(screens)}},
        threshold_percent=20.0,
        consecutive=3,
        interval_seconds=10.0,
        max_wait_seconds=60.0,
    )
    assert receipt["quiet"] is True and len(receipt["samples"]) == 4
    assert receipt["samples"][0]["foreign_heavy_processes"] == [
        {"pid": 20, "ppid": 1, "cpu_percent": 25.0, "command": "daemon"}
    ]


def test_audit_discarded_refuses_nonforeign_reason_and_shared_epoch(tmp_path) -> None:
    """A discard is only a foreign-load diagnostic and can never share a retained epoch."""
    retained, _ = complete_unit()
    retained["epoch_ns"] = 200
    discarded = copy.deepcopy(retained)
    discarded.update(
        epoch_ns=100,
        stop_reason="foreign heavy load during M1",
        host=[
            {"monotonic_ns": 99, "foreign_heavy_processes": []},
            {"monotonic_ns": 101, "foreign_heavy_processes": [{"pid": 99}]},
        ],
    )
    discarded["summary"] = m1.summarize(m1.cells()[0], [], [], False)
    discarded["summary"]["complete"] = False
    discarded["summary"]["verdict"] = "INCOMPLETE"
    export_json(tmp_path / "cpu-w1-k1.discarded-1.json", discarded)
    receipt = {
        "namespace": "n",
        "discarded_runs": [
            {
                "cell_id": "cpu-w1-k1",
                "n": 1,
                "stop_reason": "foreign heavy load during M1",
                "foreign_heavy_processes": [{"pid": 99}],
                "started_ns": 1,
                "finished_ns": 300,
                "ticks": len(discarded["ticks"]),
                "samples_count": len(discarded["samples"]),
            }
        ],
    }
    m1.audit_discarded(tmp_path, receipt, [retained])
    receipt["discarded_runs"][0]["stop_reason"] = "worker failure"
    with pytest.raises(ValueError, match="non-foreign discard"):
        m1.audit_discarded(tmp_path, receipt, [retained])
    receipt["discarded_runs"][0]["stop_reason"] = "foreign heavy load during M1"
    retained["epoch_ns"] = 100
    with pytest.raises(ValueError, match="shared epoch"):
        m1.audit_discarded(tmp_path, receipt, [retained])


@pytest.mark.parametrize(
    ("constant", "value"), [("MAX_CELL_RESTARTS", 0), ("MAX_DISCARDED_RUNS", 0)]
)
def test_audit_discarded_enforces_cell_and_attempt_budgets(
    monkeypatch, tmp_path, constant, value
) -> None:
    """Both restart ceilings are checked again at reduction, not trusted from the loop."""
    retained, _ = complete_unit()
    retained["epoch_ns"] = 200
    discarded = copy.deepcopy(retained)
    discarded.update(
        epoch_ns=100,
        stop_reason="foreign heavy load during M1",
        host=[
            {"monotonic_ns": 99, "foreign_heavy_processes": []},
            {"monotonic_ns": 101, "foreign_heavy_processes": [{"pid": 99}]},
        ],
    )
    discarded["summary"] = m1.summarize(m1.cells()[0], [], [], False)
    export_json(tmp_path / "cpu-w1-k1.discarded-1.json", discarded)
    receipt = {
        "namespace": "n",
        "discarded_runs": [
            {
                "cell_id": "cpu-w1-k1",
                "n": 1,
                "stop_reason": "foreign heavy load during M1",
                "foreign_heavy_processes": [{"pid": 99}],
                "started_ns": 1,
                "finished_ns": 300,
                "ticks": len(discarded["ticks"]),
                "samples_count": len(discarded["samples"]),
            }
        ],
    }
    monkeypatch.setattr(m1, constant, value)
    with pytest.raises(ValueError, match="invalid M1 discard"):
        m1.audit_discarded(tmp_path, receipt, [retained])


def test_wait_quiet_timeout_records_foreign_samples(monkeypatch) -> None:
    clock = SimpleNamespace(now_ns=0)
    monkeypatch.setattr(m1.time, "monotonic_ns", lambda: clock.now_ns)
    monkeypatch.setattr(m1.time, "monotonic", lambda: clock.now_ns / 1e9)
    monkeypatch.setattr(
        m1.time,
        "sleep",
        lambda seconds: setattr(clock, "now_ns", clock.now_ns + int(seconds * 1e9)),
    )
    monkeypatch.setattr(m1.os, "getpid", lambda: 10)
    receipt = m1.wait_quiet(
        lambda: {
            "process_screen": {"processes": "PID PPID %CPU COMM\n10 1 1 python\n20 1 25 daemon\n"}
        },
        consecutive=3,
        interval_seconds=10.0,
        max_wait_seconds=20.0,
    )
    assert receipt["quiet"] is False
    assert len(receipt["samples"]) == 2
    assert all(sample["foreign_heavy_processes"] for sample in receipt["samples"])


@pytest.mark.parametrize("quiet_after_restart", [True, False])
def test_main_restarts_only_foreign_stop_and_exports_discard(
    monkeypatch, tmp_path, quiet_after_restart
) -> None:
    """A stopped foreign-load epoch is immutable diagnostic output before a whole-cell rerun."""
    from contextlib import contextmanager

    from research import mac_lease
    from research.bl_mac_latency import host

    cell = m1.cells()[0]
    retained, _ = complete_unit()
    stopped = copy.deepcopy(retained)
    stopped.update(
        epoch_ns=100,
        stop_reason="foreign heavy load during M1",
        host=[
            {"monotonic_ns": 99, "foreign_heavy_processes": []},
            {"monotonic_ns": 101, "foreign_heavy_processes": [{"pid": 99}]},
        ],
    )
    stopped["summary"] = m1.summarize(cell, [], [], False)
    stopped["summary"]["complete"] = False
    stopped["summary"]["verdict"] = "INCOMPLETE"
    data = {
        "namespace": "n",
        "sources": {},
        "fixtures": fixtures(),
        "actions": {fixture["fixture_id"]: 1 for fixture in fixtures()},
    }
    ledger = tmp_path / "ledger.json"
    m1.make_ledger(ledger, "n", [])
    monkeypatch.setattr(m1, "inputs", lambda *args: data)
    monkeypatch.setattr(m1, "sources", lambda: {})
    monkeypatch.setattr(m1, "cells", lambda: [cell])
    monkeypatch.setattr(host, "snapshot", lambda: {"quiet": True})
    quiet = []

    def wait(*args, **kwargs):
        quiet.append(kwargs)
        return {"quiet": len(quiet) == 1 or quiet_after_restart, "samples": []}

    monkeypatch.setattr(m1, "wait_quiet", wait)
    rows = iter([stopped, retained])
    monkeypatch.setattr(m1, "run_cell", lambda *args: next(rows))
    clock = SimpleNamespace(value=99)
    monkeypatch.setattr(m1.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        m1.time,
        "monotonic_ns",
        lambda: setattr(clock, "value", clock.value + 1) or clock.value,
    )

    @contextmanager
    def held(*args, **kwargs):
        yield SimpleNamespace(receipt={"id": "test-only"})

    monkeypatch.setattr(mac_lease, "held", held)
    out = tmp_path / "attempt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--run",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            str(ledger),
            "--out",
            str(out),
        ],
    )
    m1.main()
    receipt, _ = l1._read_completed_json(out / "receipt.json")
    assert (out / "cpu-w1-k1.discarded-1.json.complete").exists()
    if quiet_after_restart:
        assert receipt["verdict"] == "INCOMPLETE", receipt
    else:
        assert receipt["verdict"] == "INCOMPLETE"
        assert receipt["reason"] == "host not quiet for M1 restart"
    assert receipt["discarded_runs"][0]["cell_id"] == "cpu-w1-k1"
    assert len(receipt["quiet_waits"]) == 2 and len(quiet) == 2
    assert receipt["restart_policy"] == m1.RESTART_POLICY


def test_main_does_not_restart_nonforeign_stop(monkeypatch, tmp_path) -> None:
    """A stop with another reason remains an incomplete attempt, never a discard."""
    from contextlib import contextmanager

    from research import mac_lease
    from research.bl_mac_latency import host

    cell = m1.cells()[0]
    row, _ = complete_unit()
    row.update(stop_reason="worker failure", host=[], epoch_ns=100)
    row["summary"] = m1.summarize(cell, [], [], False)
    row["summary"]["complete"] = False
    row["summary"]["verdict"] = "INCOMPLETE"
    ledger = tmp_path / "ledger.json"
    m1.make_ledger(ledger, "n", [])
    monkeypatch.setattr(
        m1,
        "inputs",
        lambda *args: {
            "namespace": "n",
            "sources": {},
            "fixtures": fixtures(),
            "actions": {fixture["fixture_id"]: 1 for fixture in fixtures()},
        },
    )
    monkeypatch.setattr(m1, "sources", lambda: {})
    monkeypatch.setattr(m1, "cells", lambda: [cell])
    monkeypatch.setattr(host, "snapshot", lambda: {"quiet": True})
    monkeypatch.setattr(m1, "wait_quiet", lambda *args, **kwargs: {"quiet": True, "samples": []})
    monkeypatch.setattr(m1, "run_cell", lambda *args: row)
    monkeypatch.setattr(m1.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(m1.time, "monotonic_ns", lambda: 100)

    @contextmanager
    def held(*args, **kwargs):
        yield SimpleNamespace(receipt={"id": "test-only"})

    monkeypatch.setattr(mac_lease, "held", held)
    out = tmp_path / "attempt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--run",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            str(ledger),
            "--out",
            str(out),
        ],
    )
    m1.main()
    receipt, _ = l1._read_completed_json(out / "receipt.json")
    assert receipt["verdict"] == "INCOMPLETE" and receipt["discarded_runs"] == []
    assert not list(out.glob("*.discarded-*.json"))


def test_wrapper_passes_judged_recovery_without_dispatch(tmp_path) -> None:
    import json
    import subprocess
    from pathlib import Path

    root = tmp_path / "repo with spaces"
    script = root / "research/bl_k1/run_m1.sh"
    script.parent.mkdir(parents=True)
    python = tmp_path / "fake-python"
    python.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    python.chmod(0o755)
    original = Path(m1.__file__).with_name("run_m1.sh").read_text()
    script.write_text(
        original.replace(
            "PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python", f'PY="{python}"'
        )
    )
    proof = str(tmp_path / "proof with spaces.json")
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--check",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--judged-recovery",
            proof,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [
        "-m",
        "research.bl_k1.m1",
        "--check",
        "--fixtures",
        "f",
        "--reference",
        "r",
        "--judged-recovery",
        proof,
    ]


def run_mock_attempt(
    monkeypatch,
    tmp_path: Path,
    stops: dict[str, int],
    *,
    charged: float = 0.0,
    initial_quiet: bool = True,
    carried: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Exercise main's real loop/reducer/exports with no Watch, workers or host calls."""
    from contextlib import contextmanager

    from research import mac_lease
    from research.bl_mac_latency import host

    clock = SimpleNamespace(now=100.0)
    calls = Counter()
    waits = []
    carried = {} if carried is None else carried
    data = {"namespace": "n", "sources": {}}
    monkeypatch.setattr(m1, "inputs", lambda *args: data)
    monkeypatch.setattr(m1, "sources", lambda: {})
    monkeypatch.setattr(host, "snapshot", lambda: {"quiet": True})
    monkeypatch.setattr(m1.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(m1.time, "monotonic_ns", lambda: int(clock.now * 1e9))
    monkeypatch.setattr(m1, "audit_unit", lambda *args: None)

    def quiet(*args, **kwargs):
        waits.append(kwargs)
        start = int(clock.now * 1e9)
        clock.now += min(20.0, kwargs["max_wait_seconds"])
        return {
            "quiet": initial_quiet,
            "started_ns": start,
            "finished_ns": int(clock.now * 1e9),
            "samples": [],
        }

    def run(cell, *args):
        unit_id = cell["cell_id"]
        calls[unit_id] += 1
        stopped = calls[unit_id] <= stops.get(unit_id, 0)
        clock.now += 0.5
        epoch = int(clock.now * 1e9)
        clock.now += 0.5
        row = {
            **cell,
            "schema": m1.SCHEMA + "/cell",
            "namespace": "n",
            "epoch_ns": epoch,
            "summary": {"complete": not stopped, "verdict": "INCOMPLETE" if stopped else "PASS"},
            "stop_reason": m1.RESTART_POLICY["trigger"] if stopped else None,
            "host": [
                {"monotonic_ns": epoch, "foreign_heavy_processes": []},
                {
                    "monotonic_ns": epoch + 1,
                    "foreign_heavy_processes": [{"pid": 99}] if stopped else [],
                },
            ],
            "samples": [],
            "ticks": [],
        }
        kind = carried.get(unit_id) if stopped else None
        if kind == "overrun":
            row["ticks"] = [{"finished_ns": epoch, "overrun": True, "update_seconds": 0.001}]
        elif kind == "tail_overrun":
            row["ticks"] = [{"finished_ns": epoch + 1, "overrun": True, "update_seconds": 0.001}]
        elif kind == "doom":
            row["samples"] = [
                {
                    "phase": "warm",
                    "fixture_id": "frozen-800",
                    "S_seconds": 5.1,
                    "visible_ns": epoch,
                }
            ]
        elif kind == "p95":
            row["ticks"] = [{"finished_ns": epoch, "overrun": False, "update_seconds": 0.012}]
        return row

    @contextmanager
    def held(*args, **kwargs):
        assert kwargs["seconds"] == m1.CAP_SECONDS - charged
        yield SimpleNamespace(receipt={"id": "mock-only"})

    monkeypatch.setattr(m1, "wait_quiet", quiet)
    monkeypatch.setattr(m1, "run_cell", run)
    monkeypatch.setattr(mac_lease, "held", held)
    ledger, out = tmp_path / "ledger.json", tmp_path / "attempt"
    m1.make_ledger(ledger, "n", [])
    argv = [
        "m1",
        "--run",
        "--fixtures",
        "f",
        "--reference",
        "r",
        "--ledger",
        str(ledger),
        "--out",
        str(out),
    ]
    if charged:
        prior = tmp_path / "previous.json"
        export_json(
            prior,
            {
                "schema": m1.SCHEMA,
                "namespace": "n",
                "attempt_number": 1,
                "verdict": "INCOMPLETE",
                "verified_stop": True,
                "mechanical_failure": True,
                "completed_unit_exports": {},
                "charged_seconds": charged,
            },
        )
        argv += ["--previous-attempt", str(prior)]
    monkeypatch.setattr("sys.argv", argv)
    if not initial_quiet:
        with pytest.raises(RuntimeError, match="host not quiet for M1 restart"):
            m1.main()
        assert not out.exists()
        return SimpleNamespace(waits=waits, calls=calls)
    m1.main()
    receipt, _ = l1._read_completed_json(out / "receipt.json")
    assert receipt["charged_seconds"] == charged + clock.now - 100.0
    return SimpleNamespace(receipt=receipt, waits=waits, calls=calls, out=out)


@pytest.mark.parametrize("scope", ["cell", "attempt", "reserve"])
def test_main_enforces_actual_restart_boundaries(monkeypatch, tmp_path, scope) -> None:
    if scope == "cell":
        case = run_mock_attempt(monkeypatch, tmp_path, {"cpu-w1-k1": 99})
        assert case.calls == {"cpu-w1-k1": 4}
        assert len(case.receipt["discarded_runs"]) == 3
    elif scope == "attempt":
        case = run_mock_attempt(monkeypatch, tmp_path, {c["cell_id"]: 3 for c in m1.cells()})
        assert list(case.calls.values()) == [4, 4, 3]
        assert len(case.receipt["discarded_runs"]) == 8
        assert max(Counter(r["cell_id"] for r in case.receipt["discarded_runs"]).values()) == 3
    else:
        case = run_mock_attempt(
            monkeypatch, tmp_path, {"cpu-w1-k1": 99}, charged=m1.CAP_SECONDS - 610
        )
        assert case.calls == {"cpu-w1-k1": 1}
        assert case.receipt["discarded_runs"] == []
    assert case.receipt["verdict"] == "INCOMPLETE"
    assert case.receipt["mechanical_failure"] is True


def test_main_charges_preflight_discard_restart_and_quiet_waits(monkeypatch, tmp_path) -> None:
    case = run_mock_attempt(monkeypatch, tmp_path, {"cpu-w1-k1": 1}, charged=36.0)
    assert case.receipt["verdict"] == "PASS"
    assert len(case.receipt["quiet_waits"]) == 2
    assert case.receipt["charged_seconds"] == 36.0 + 40.0 + 5.0


def test_main_preflight_timeout_creates_no_directory_and_clips_cap(monkeypatch, tmp_path) -> None:
    case = run_mock_attempt(
        monkeypatch, tmp_path, {}, charged=m1.CAP_SECONDS - 10, initial_quiet=False
    )
    assert case.waits[0]["max_wait_seconds"] == 10
    assert case.calls == {}


def test_judged2_check_refuses_nonempty_ledger(monkeypatch, judged2_case) -> None:
    case = judged2_case
    export_json(case.path, case.proof)
    monkeypatch.setattr(m1, "inputs", lambda *args: case.data)
    monkeypatch.setattr(m1, "verify_ledger", lambda *args: {"cpu-w1-k1": {}})
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--check",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            "l",
            "--judged-recovery",
            str(case.path),
        ],
    )
    with pytest.raises(ValueError, match="judged recovery requires an empty new ledger"):
        m1.main()


@pytest.mark.parametrize("kind", ["judged", "precell"])
def test_consumed_history_is_anchored_despite_relocated_artifacts(
    monkeypatch, tmp_path, kind
) -> None:
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    monkeypatch.setattr(m1, "JUDGED2_JUDGMENT_PATH", authority_root / "judgment.md")
    field = "judged_recovery_sha256" if kind == "judged" else "reconciled_precell_stop_sha256"
    export_json(
        authority_root / "bl-m1-attempt-04/receipt.json",
        {"schema": m1.SCHEMA, "attempt_number": 3, field: "consumed-proof-sha"},
    )
    with pytest.raises(ValueError, match="already consumed"):
        m1.refuse_consumed(tmp_path / "copied-history/attempt/receipt.json", field, 2)


def test_reserialized_judged_proof_cannot_reuse_consumed_debit(monkeypatch, judged_case) -> None:
    case = judged_case
    proof = write_judged_case(monkeypatch, case)
    proof["ignored_metadata"] = "different proof bytes do not reset consumption"
    export_json(case.proof_path, proof)
    export_json(
        case.prior.parent / "bl-m1-attempt-04/receipt.json",
        {"schema": m1.SCHEMA, "attempt_number": 3, "judged_recovery_sha256": "original-proof-sha"},
    )
    with pytest.raises(ValueError, match="exact audited attempt-03"):
        m1.judged_recovery(case.proof_path, case.data)


@pytest.fixture
def judged3_case(monkeypatch, tmp_path):
    """A complete mock attempt-05 exercises every third-proof pin."""
    prior = tmp_path / "bl-m1-attempt-05"
    prior.mkdir()
    judgment = tmp_path / "JUDGMENT_M1_ATTEMPT6.md"
    judgment.write_text("Fable attempt-06 judgment")
    monkeypatch.setattr(m1, "JUDGED3_JUDGMENT_PATH", judgment)
    log = tmp_path / "bl-m1-attempt-05.try1.log"
    log.write_text("mechanical foreign-load stop")
    previous = {
        "schema": m1.SCHEMA,
        "namespace": "attempt-05-namespace",
        "attempt_number": 4,
        "final_attempt": True,
        "verdict": "INCOMPLETE",
        "verified_stop": True,
        "mechanical_failure": True,
        "completed_unit_exports": {},
        "judged_recovery_sha256": "attempt-05-prior-judged-proof",
        "charged_seconds": 771.5500928750262,
        "cells": [{"cell_id": "cpu-w1-k1", "complete": False}],
        "failure": None,
        "discarded_runs": [{"stop_reason": m1.RESTART_POLICY["trigger"]} for _ in range(3)],
        "quiet_waits": [{"quiet": True} for _ in range(4)],
    }
    attempt = {"namespace": previous["namespace"], "sources": {"research/bl_k1/m1.py": "old"}}
    cell = {"stop_reason": m1.RESTART_POLICY["trigger"], "summary": {"complete": False}}
    idle = {"quiet": True}
    receipt_sha = export_json(prior / "receipt.json", previous)["sha256"]
    attempt_sha = export_json(prior / "attempt.json", attempt)["sha256"]
    cell_sha = export_json(prior / "cpu-w1-k1.json", cell)["sha256"]
    idle_sha = export_json(prior / "idle.json", idle)["sha256"]
    discarded = {}
    for n in range(1, 4):
        name = f"cpu-w1-k1.discarded-{n}.json"
        discarded[name] = export_json(prior / name, {"n": n})["sha256"]
    monkeypatch.setattr(m1, "JUDGED3_RECEIPT_SHA256", receipt_sha)
    monkeypatch.setattr(m1, "JUDGED3_ATTEMPT_SHA256", attempt_sha)
    monkeypatch.setattr(m1, "JUDGED3_LOG_SHA256", m1.digest(log))
    monkeypatch.setattr(m1, "JUDGED3_IDLE_SHA256", idle_sha)
    monkeypatch.setattr(m1, "JUDGED3_CELL_SHA256", {"cpu-w1-k1": cell_sha})
    monkeypatch.setattr(m1, "JUDGED3_DISCARDED_SHA256", discarded)
    monkeypatch.setattr(m1, "JUDGED3_SOURCE_SHA256", "old")
    monkeypatch.setattr(m1, "JUDGED3_PREVIOUS_JUDGED_SHA256", "attempt-05-prior-judged-proof")
    monkeypatch.setattr(m1, "JUDGED3_PREVIOUS_NAMESPACE", previous["namespace"])
    data = {
        "namespace": "new",
        "sources": {"research/bl_k1/m1.py": m1.digest(Path(m1.__file__))},
    }
    proof = {
        "schema": m1.SCHEMA + "/judged-recovery",
        "judge": "Fable-M1-attempt6",
        "status": "JUDGED_FOREIGN_LOAD_STOP_3",
        "judgment_path": str(judgment),
        "judgment_sha256": m1.digest(judgment),
        "previous_receipt_path": str(prior / "receipt.json"),
        "previous_receipt_sha256": receipt_sha,
        "previous_attempt_sha256": attempt_sha,
        "previous_log_path": str(log),
        "previous_log_sha256": m1.digest(log),
        "previous_idle_sha256": idle_sha,
        "previous_cell_sha256": {"cpu-w1-k1": cell_sha},
        "previous_discarded_sha256": discarded,
        "previous_namespace": previous["namespace"],
        "previous_judged_recovery_sha256": "attempt-05-prior-judged-proof",
        "charged_seconds": previous["charged_seconds"],
        "remaining_seconds": m1.CAP_SECONDS - previous["charged_seconds"],
        "next_attempt_number": 5,
        "final_attempt": True,
        "restart_policy_required": True,
        "carry_forward_required": True,
        "target_namespace": "new",
        "transferred_cell_or_timing_evidence": False,
        "operator_dispatch_once": True,
    }
    return SimpleNamespace(
        previous=previous,
        proof=proof,
        path=tmp_path / "judged3-proof.json",
        data=data,
        prior=prior,
    )


def test_judged3_recovery_happy_path(judged3_case) -> None:
    export_json(judged3_case.path, judged3_case.proof)
    assert m1.judged_recovery(judged3_case.path, judged3_case.data) == judged3_case.previous


@pytest.mark.parametrize(
    "change", ["hash", "attempt4", "discard_count", "old_source", "no_carry_flag"]
)
def test_judged3_recovery_refuses_required_boundaries(monkeypatch, judged3_case, change) -> None:
    case = judged3_case
    if change == "hash":
        case.proof["previous_idle_sha256"] = "wrong"
    elif change == "attempt4":
        case.previous["attempt_number"] = 3
    elif change == "discard_count":
        case.previous["discarded_runs"].pop()
    elif change == "old_source":
        case.data["sources"]["research/bl_k1/m1.py"] = "old"
    elif change == "no_carry_flag":
        case.proof["carry_forward_required"] = False
    if change in {"attempt4", "discard_count"}:
        receipt = Path(case.proof["previous_receipt_path"])
        for artifact in (
            receipt,
            receipt.with_suffix(".json.sha256"),
            receipt.with_suffix(".json.complete"),
        ):
            artifact.unlink()
        receipt_sha = export_json(receipt, case.previous)["sha256"]
        case.proof["previous_receipt_sha256"] = receipt_sha
        monkeypatch.setattr(m1, "JUDGED3_RECEIPT_SHA256", receipt_sha)
    export_json(case.path, case.proof)
    with pytest.raises(ValueError, match="exact audited attempt-05 foreign-load stop"):
        m1.judged_recovery(case.path, case.data)


def test_judged3_check_refuses_nonempty_ledger(monkeypatch, judged3_case) -> None:
    case = judged3_case
    export_json(case.path, case.proof)
    monkeypatch.setattr(m1, "inputs", lambda *args: case.data)
    monkeypatch.setattr(m1, "verify_ledger", lambda *args: {"cpu-w1-k1": {}})
    monkeypatch.setattr(
        "sys.argv",
        [
            "m1",
            "--check",
            "--fixtures",
            "f",
            "--reference",
            "r",
            "--ledger",
            "l",
            "--judged-recovery",
            str(case.path),
        ],
    )
    with pytest.raises(ValueError, match="judged recovery requires an empty new ledger"):
        m1.main()


def test_judged2_is_consumed_when_attempt05_receipt_exists(monkeypatch, judged2_case) -> None:
    case = judged2_case
    export_json(case.path, case.proof)
    export_json(
        case.path.parent / "bl-m1-attempt-05/receipt.json",
        {"schema": m1.SCHEMA, "attempt_number": 4, "judged_recovery_sha256": "attempt-05-proof"},
    )
    with pytest.raises(ValueError, match="exact audited attempt-04 foreign-load stop"):
        m1.judged_recovery(case.path, case.data)


def test_main_clean_prefix_overrun_is_decided_blocked_and_continues(monkeypatch, tmp_path) -> None:
    case = run_mock_attempt(
        monkeypatch, tmp_path, {"cpu-w1-k1": 1}, carried={"cpu-w1-k1": "overrun"}
    )
    assert case.receipt["verdict"] == "BLOCKED"
    assert case.calls == {cell["cell_id"]: 1 for cell in m1.cells()}
    assert case.receipt["discarded_runs"] == []
    saved, _ = l1._read_completed_json(case.out / "cpu-w1-k1.json")
    assert saved["summary"]["decided"] is True
    assert saved["summary"]["carried_violation"] == {
        "T_clean_ns": 120500000000,
        "carried_overruns": 1,
        "carried_doom": 0,
        "run_n": 1,
        "rule": "R1",
    }


def test_main_four_resolved_carries_yield_blocked(monkeypatch, tmp_path) -> None:
    carried = {cell["cell_id"]: "overrun" for cell in m1.cells()}
    case = run_mock_attempt(monkeypatch, tmp_path, {key: 1 for key in carried}, carried=carried)
    assert case.receipt["verdict"] == "BLOCKED"
    assert [cell["verdict"] for cell in case.receipt["cells"]] == ["BLOCKED"] * 4
    assert all(cell["decided"] for cell in case.receipt["cells"])


@pytest.mark.parametrize("kind", ["tail_overrun", "p95"])
def test_main_restarts_when_prefix_event_is_not_carried(monkeypatch, tmp_path, kind) -> None:
    case = run_mock_attempt(monkeypatch, tmp_path, {"cpu-w1-k1": 1}, carried={"cpu-w1-k1": kind})
    assert case.receipt["verdict"] == "PASS"
    assert case.calls["cpu-w1-k1"] == 2
    assert len(case.receipt["discarded_runs"]) == 1


def test_main_w4_frozen800_51_prefix_carries(monkeypatch, tmp_path) -> None:
    case = run_mock_attempt(monkeypatch, tmp_path, {"cpu-w4-k4": 1}, carried={"cpu-w4-k4": "doom"})
    assert case.receipt["verdict"] == "BLOCKED"
    w4 = next(cell for cell in case.receipt["cells"] if cell["cell_id"] == "cpu-w4-k4")
    assert w4["carried_violation"]["carried_doom"] == 1
