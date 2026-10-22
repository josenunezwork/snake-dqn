"""Independent synthetic review probes. No workers, Watch, lease, or production I/O."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from research.bl_served_search import c1, c1_bank, c1_reduce
from research.bl_served_search.exports import export_json


def setup_fake_run(tmp_path, monkeypatch, fault):
    import research.mac_lease as mac_lease
    import research.published_ledger as published

    root = tmp_path / "study"
    root.mkdir()
    monkeypatch.setattr(c1_bank, "STUDY_ROOT", root)
    monkeypatch.setattr(c1_bank, "freshness", lambda: {"status": "CLEAR"})
    allocation = c1_bank.allocation()
    data = {"identity": "test-only", "allocation": allocation, "sources": {}}
    export_json(root / "study.json", data)
    ledger = {}
    for row in allocation[:-1]:
        path = root / (row["unit_id"] + ".json")
        export_json(path, row)
        ledger[row["unit_id"]] = c1.inventory(path, row["unit_id"])
    ledger_path = root / "ledger.json"
    export_json(ledger_path, {"units": list(ledger.values())})

    clock = {"now": 0.0}
    monkeypatch.setattr(c1, "time", SimpleNamespace(monotonic=lambda: clock["now"], time=lambda: 0))
    monkeypatch.setattr(c1, "platform", SimpleNamespace(system=lambda: "Darwin"))
    monkeypatch.setattr(c1.os, "getpriority", lambda *_: 10)
    monkeypatch.setattr(c1, "sources", lambda: {})
    monkeypatch.setattr(c1.host, "snapshot", lambda: {"quiet": True})
    monkeypatch.setattr(
        published,
        "require_published_ledger",
        lambda *_: {"authoring_head": "published-test", "origin_main": "published-test"},
    )

    @contextmanager
    def held(*_args, **_kwargs):
        yield SimpleNamespace(receipt={"test": True}, until_epoch=1e12)

    monkeypatch.setattr(mac_lease, "held", held)
    monkeypatch.setattr(
        c1.subprocess,
        "Popen",
        lambda *_: SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **_: 0),
    )
    monkeypatch.setattr(
        c1.multiprocessing,
        "get_context",
        lambda *_: SimpleNamespace(Event=lambda: SimpleNamespace(set=lambda: None)),
    )
    monitor = SimpleNamespace(
        error=None, rows=[], thread=SimpleNamespace(start=lambda: None), close=lambda: None
    )
    monkeypatch.setattr(c1.m1, "HostMonitor", lambda *_: monitor)

    class Pool:
        def __init__(self, **_kwargs):
            pass

        def submit(self, _function, row, _data, out):
            # Synthetic immutable pair, never call run_unit or a session constructor.
            path = out / (row["unit_id"] + ".json")
            export_json(path, row)
            descriptor = c1.inventory(path, row["unit_id"])

            def result():
                if fault == "late-host":
                    monitor.error = "host guard: lid_not_known_open"
                    monitor.rows.append({"reasons": ["lid_not_known_open"]})
                return descriptor

            return SimpleNamespaceFuture(result)

        def shutdown(self, **_kwargs):
            pass

    monkeypatch.setattr(c1.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(c1.concurrent.futures, "wait", lambda pending, **_: (set(pending), set()))
    monkeypatch.setattr(c1, "verify_unit", lambda *_: None)
    monkeypatch.setattr(c1, "verify_completion", lambda *_: None)

    def reduce(_units):
        if fault == "cap-in-look":
            clock["now"] = c1.CAP_SECONDS + 1
        return {"verdict": "GO", "synthetic_review_only": True}

    monkeypatch.setattr(c1_reduce, "reduce_sessions", reduce)
    return data, ledger, ledger_path, root


class SimpleNamespaceFuture:
    def __init__(self, callback):
        self.callback = callback

    def result(self):
        return self.callback()


def test_late_host_fault_must_block_registered_look(tmp_path, monkeypatch):
    data, ledger, path, root = setup_fake_run(tmp_path, monkeypatch, "late-host")
    c1.run(data, ledger, 2, 0, path)
    receipt = c1.read(root / "attempt-02/receipt.json")
    assert receipt["status"] != "COMPLETE", receipt
    assert not (root / "decision.json").exists()


def test_cap_expiry_during_look_must_not_publish_efficacy(tmp_path, monkeypatch):
    data, ledger, path, root = setup_fake_run(tmp_path, monkeypatch, "cap-in-look")
    c1.run(data, ledger, 2, 0, path)
    receipt = c1.read(root / "attempt-02/receipt.json")
    assert receipt["status"] == "INCOMPLETE"
    assert not (root / "decision.json").exists(), json.loads((root / "decision.json").read_text())


def test_resume_ledger_must_refuse_foreign_payload_before_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(c1_bank, "STUDY_ROOT", tmp_path)
    data = {"identity": "test-own", "allocation": c1_bank.allocation()}
    unit_path = tmp_path / "session-00.json"
    export_json(
        unit_path,
        {
            "unit_id": "session-00",
            "namespace": "foreign",
            "identity": "foreign",
            "seed": 123,
            "horizon": 5,
        },
    )
    descriptor = c1.inventory(unit_path, "session-00")
    ledger_path = tmp_path / "ledger.json"
    export_json(
        ledger_path,
        {"schema": c1.SCHEMA + "/ledger", "identity": data["identity"], "units": [descriptor]},
    )
    with pytest.raises(ValueError):
        c1.verify_ledger(ledger_path, data)


def test_frozen_source_identity_includes_admission_and_thermal_policy():
    sources = c1.sources()
    assert {"research/published_ledger.py", "research/compute/thermal_guard.py"} <= set(sources)


def test_mechanical_caffeinate_start_failure_must_publish_stop_receipt(tmp_path, monkeypatch):
    data, ledger, path, root = setup_fake_run(tmp_path, monkeypatch, "none")

    def unavailable(*_args, **_kwargs):
        raise OSError("synthetic process creation failure")

    monkeypatch.setattr(c1.subprocess, "Popen", unavailable)
    c1.run(data, ledger, 2, 0, path)
    receipt = c1.read(root / "attempt-02/receipt.json")
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["verified_stop"] is True


def test_run_setup_freshness_time_must_be_charged_to_cumulative_cap(tmp_path, monkeypatch):
    data, ledger, path, root = setup_fake_run(tmp_path, monkeypatch, "none")
    clock = {"now": 0.0}
    monkeypatch.setattr(c1.time, "monotonic", lambda: clock["now"])

    def freshness():
        clock["now"] += 600.0
        return {"status": "CLEAR"}

    monkeypatch.setattr(c1_bank, "freshness", freshness)
    c1.run(data, ledger, 2, 0, path)
    receipt = c1.read(root / "attempt-02/receipt.json")
    assert receipt["charged_seconds"] >= 600.0
